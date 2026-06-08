"""
FAST TEST version of REINFORCE-based active learning for binary RB/FS classification.

Structure mirrors your original multi-label pipeline so results are
directly comparable. Key simplifications:
  - State uses a single entropy scalar (binary entropy on RB probability)
    plus the backbone feature vector
  - No CIW / priority machinery (only one class)
  - Reward = delta in best-F2 (primary metric for binary task)
  - Added fast diagnostic modes: main-model query, random/entropy/policy query, no-query control,
    query-only statistics, query-pool limit, train/val batch limits, and selection statistics.
  - Cold start strategies come from rb_cold_start.py

Run example:
    python rb_active_learning.py \
        --experiment_name rb_al_run \
        --cold_start_strategy clipiqa \
        --secondary_strategy balanced \
        --initial_percentage 0.005 \
        --al_budget 1000 --al_cycles 10
"""

import os
import sys
import json
import time
import datetime
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataloaders"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataloaders.sewerml_dataset import MultiLabelDataset

from rb_binary_dataset import RBBinaryDataset
from rb_metrics import binary_metrics, format_metrics
from rb_cold_start import apply_cold_start


# ===========================
# Policy net (unchanged from original)
# ===========================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.image_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sample_states, global_state):
        return self.image_head(sample_states)


# ===========================
# Models
# ===========================
class BinaryResNet(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True):
        super().__init__()
        backbone = getattr(models, arch)(pretrained=pretrained)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim   = backbone.fc.in_features
        self.classifier = nn.Linear(self.feat_dim, 1)

    def get_bottleneck_features(self, x):
        f = self.features(x)
        return torch.flatten(f, 1)

    def forward(self, x):
        f = self.get_bottleneck_features(x)
        return self.classifier(f).squeeze(-1)


# ===========================
# Logging
# ===========================
def setup_logging(name, log_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# ===========================
# Data
# ===========================
def get_data(args):
    train_tf = transforms.Compose([
        transforms.Resize((args.scale_size, args.scale_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.523, 0.453, 0.345], [0.210, 0.199, 0.154]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((args.scale_size, args.scale_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.523, 0.453, 0.345], [0.210, 0.199, 0.154]),
    ])

    sewer_root = os.path.join(args.dataroot, "Sewer_ML")
    import pandas as pd
    train_csv = os.path.join(sewer_root, "SewerML_train.csv")
    val_csv   = os.path.join(sewer_root, "SewerML_valid.csv")
    train_list = pd.read_csv(train_csv)["Filename"].tolist() if os.path.exists(train_csv) else []
    val_list   = pd.read_csv(val_csv)["Filename"].tolist()   if os.path.exists(val_csv)   else []

    base_train = MultiLabelDataset(
        img_dir=sewer_root, image_transform=train_tf,
        labels_path=sewer_root, val_list=val_list, train_list=train_list,
        known_labels=1.0, testing=False, split="train",
    )
    base_val = MultiLabelDataset(
        img_dir=sewer_root, image_transform=test_tf,
        labels_path=sewer_root, val_list=val_list, train_list=train_list,
        known_labels=1.0, testing=True, split="valid",
    )
    train_ds = RBBinaryDataset(base_train, target_class=args.target_class)
    val_ds   = RBBinaryDataset(base_val,   target_class=args.target_class)
    return train_ds, val_ds


def get_feature_cache_path(args):
    cache_dir = os.path.join(args.dataroot, "feature_cache_rb")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "sewerml_rb_{}_{}".format(args.target_class, args.scale_size))


# ===========================
# AL Pipeline
# ===========================
class RBActiveLearningPipeline:

    def __init__(self, train_dataset, args, device="cuda"):
        self.train_dataset = train_dataset
        self.args          = args
        self.device        = torch.device(device if torch.cuda.is_available() else "cpu")

        log_file = os.path.join("logs", "RB_AL_{}.log".format(
            datetime.datetime.now().strftime("%m%d_%H%M")
        ))
        self.logger = setup_logging("RB_AL", log_path=log_file)

        # Pools
        self.labeled_indices   = []
        self.unlabeled_indices = list(range(len(train_dataset)))

        # Models
        self.oracle_model = BinaryResNet(arch="resnet34",  pretrained=True).to(self.device)
        self.main_model   = BinaryResNet(arch=args.main_arch, pretrained=True).to(self.device)

        # PolicyNet/state model:
        # state = [bottleneck_feats | entropy | 1-confidence | 1-margin].
        # In the original script this always used a frozen oracle model.
        # For fast tests, --query_model main uses the current main model instead.
        self.query_model_name = getattr(args, "query_model", "oracle")
        if self.query_model_name == "main":
            self.state_dim = self.main_model.feat_dim + 3
        else:
            self.state_dim = self.oracle_model.feat_dim + 3

        self.policy    = PolicyNet(
            state_dim=self.state_dim,
            hidden_dim=getattr(args, "policy_hidden", 256),
        ).to(self.device)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=getattr(args, "policy_lr", 1e-4),
        )

        # REINFORCE
        self.entropy_beta      = getattr(args, "entropy_beta", 1e-3)
        self.policy_temp       = getattr(args, "policy_temp_start", 1.0)
        self.policy_temp_end   = getattr(args, "policy_temp_end", 0.5)
        self.reward_baseline   = 0.0
        self.baseline_momentum = 0.9
        self.prev_score        = None

        self.history = {}
        self.cycle   = 0
        self._init_results_path()

        self.logger.info(
            "Device: {} | state_dim: {} | train: {} | target: {}".format(
                self.device, self.state_dim, len(train_dataset), args.target_class
            )
        )

    # ---- Results path ----
    def _init_results_path(self):
        date_folder = datetime.datetime.now().strftime("%m_%d")
        time_stamp  = datetime.datetime.now().strftime("%H%M")
        results_dir = os.path.join(self.args.results_dir, date_folder)
        os.makedirs(results_dir, exist_ok=True)
        self.results_path = os.path.join(
            results_dir,
            "{}_{}_{}_{}.json".format(
                self.args.experiment_name,
                self.args.target_class,
                self.args.cold_start_strategy,
                time_stamp,
            ),
        )

    def save_results(self):
        with open(self.results_path, "w") as f:
            json.dump({"config": vars(self.args), "history": self.history}, f, indent=2)

    # ---- Weights ----
    def _compute_pos_weight(self, indices, clip_max=50.0, eps=1e-6):
        labels = self.train_dataset.binary_labels
        if labels is None:
            # Fallback via loader
            sub = Subset(self.train_dataset, indices)
            loader = DataLoader(sub, batch_size=128, shuffle=False, num_workers=2)
            lbs = []
            for b in loader:
                lbs.append(b["labels"].cpu().numpy())
            labels_sub = np.concatenate(lbs)
        else:
            labels_sub = labels[indices]
        n_pos = max(1, int(labels_sub.sum()))
        n_neg = max(1, len(labels_sub) - n_pos)
        pw = min(float(n_neg / (n_pos + eps)), clip_max)
        self.logger.info("pos_weight: n_pos={} n_neg={} -> {:.2f}".format(n_pos, n_neg, pw))
        return torch.tensor([pw], dtype=torch.float32, device=self.device)

    # ---- Cold start ----
    def cold_start(self):
        strategy   = self.args.cold_start_strategy
        n_init_arg = self.args.initial_percentage
        n_initial  = int(n_init_arg * len(self.train_dataset)) if n_init_arg <= 1.0 else int(n_init_arg)
        n_initial  = max(1, n_initial)

        self.logger.info(
            "Cold start [{}]: {} samples ({:.2f}%)".format(
                strategy, n_initial, 100.0 * n_initial / len(self.train_dataset)
            )
        )

        cache_path = get_feature_cache_path(self.args)

        date_folder = datetime.datetime.now().strftime("%m_%d")
        report_dir  = os.path.join(self.args.results_dir, date_folder)
        os.makedirs(report_dir, exist_ok=True)
        time_stamp  = datetime.datetime.now().strftime("%H%M")
        report_path = os.path.join(
            report_dir,
            "selection_report_{}_{}_{}.json".format(
                strategy, self.args.experiment_name, time_stamp
            )
        )

        selected = apply_cold_start(
            strategy_name=strategy,
            dataset=self.train_dataset,
            n_initial=n_initial,
            device=self.device,
            cache_path=cache_path,
            seed=self.args.seed,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            clipiqa_json_path=getattr(self.args, "clipiqa_json_path", None),
            clipiqa_threshold=getattr(self.args, "clipiqa_threshold", 0.3),
            secondary_strategy=getattr(self.args, "secondary_strategy", "balanced"),
            pos_ratio=getattr(self.args, "pos_ratio", 0.5),
            save_selection_path=report_path,
        )

        mask = np.zeros(len(self.train_dataset), dtype=bool)
        mask[selected] = True
        self.labeled_indices   = list(selected)
        self.unlabeled_indices = np.where(~mask)[0].tolist()

        self.logger.info("Cold start done: {} labeled | {} unlabeled".format(
            len(self.labeled_indices), len(self.unlabeled_indices)
        ))
        self.history["cold_start_strategy"]    = strategy
        self.history["cold_start_labeled"]     = len(self.labeled_indices)
        self.history["cold_start_report_path"] = report_path
        self.save_results()
        return self.labeled_indices

    # ---- Oracle training ----
    def train_oracle_model(self, epochs):
        self.logger.info("Training oracle for {} epochs on {} samples".format(
            epochs, len(self.labeled_indices)
        ))
        sub    = Subset(self.train_dataset, self.labeled_indices)
        loader = DataLoader(sub, batch_size=self.args.batch_size,
                            shuffle=True, num_workers=self.args.workers)
        pos_w  = self._compute_pos_weight(self.labeled_indices)
        crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt    = optim.Adam(self.oracle_model.parameters(), lr=1e-4)

        self.oracle_model.train()
        for ep in range(epochs):
            running = 0.0
            for batch in tqdm(loader, desc="Oracle ep{}/{}".format(ep + 1, epochs), leave=False):
                imgs   = batch["image"].to(self.device)
                labels = batch["labels"].to(self.device)
                opt.zero_grad()
                loss = crit(self.oracle_model(imgs), labels)
                loss.backward()
                opt.step()
                running += loss.item()
            self.logger.info("Oracle ep {}/{} | loss {:.4f}".format(
                ep + 1, epochs, running / max(1, len(loader))
            ))

        self.oracle_model.eval()
        for p in self.oracle_model.parameters():
            p.requires_grad = False
        self.logger.info("Oracle frozen.")

    # ---- State computation ----
    def _compute_state(self, images):
        """State = [feats | entropy | 1-confidence | 1-margin].

        --query_model oracle: use the frozen oracle model, as in the original code.
        --query_model main:   use the current main model, updated after warm-up/cycles.
        """
        model = self.main_model if getattr(self.args, "query_model", "oracle") == "main" else self.oracle_model
        model.eval()
        with torch.no_grad():
            feats  = model.get_bottleneck_features(images)  # [B, D]
            logits = model(images)                          # [B]
            p      = torch.sigmoid(logits)                  # [B]
            eps    = 1e-8

            # Binary entropy on scalar probability
            entropy    = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            confidence = torch.max(p, 1 - p)
            margin     = (p - 0.5).abs()

            uncertainty = torch.stack(
                [entropy, 1.0 - confidence, 1.0 - margin], dim=1
            )                                                            # [B, 3]
            return torch.cat([feats, uncertainty], dim=1)

    # ---- Query helpers ----
    def _limit_unlabeled_pool_for_query(self):
        """Optionally scan only a random subset of the unlabeled pool to make tests faster."""
        max_pool = getattr(self.args, "max_query_pool", 0)
        if max_pool is None or max_pool <= 0 or max_pool >= len(self.unlabeled_indices):
            return list(self.unlabeled_indices)

        rng = np.random.default_rng(self.args.seed + 1000 * self.cycle)
        chosen = rng.choice(self.unlabeled_indices, size=max_pool, replace=False)
        return chosen.tolist()

    def _label_stats(self, indices):
        labels = getattr(self.train_dataset, "binary_labels", None)
        n = len(indices)
        if labels is None or n == 0:
            return {
                "n": n,
                "pos": None,
                "neg": None,
                "pos_rate": None,
            }
        y = labels[indices]
        n_pos = int(np.sum(y))
        n_neg = int(n - n_pos)
        return {
            "n": int(n),
            "pos": n_pos,
            "neg": n_neg,
            "pos_rate": float(n_pos / max(1, n)),
        }

    def _save_selection_stats(self, selected, query_info):
        selected_stats = self._label_stats(selected)
        candidate_stats = self._label_stats(query_info.get("candidate_indices", []))
        scanned_stats = self._label_stats(query_info.get("scanned_indices", []))

        stats = {
            "cycle": int(self.cycle),
            "query_strategy": getattr(self.args, "query_strategy", "policy"),
            "query_model": getattr(self.args, "query_model", "oracle"),
            "selected_n": selected_stats["n"],
            "selected_pos": selected_stats["pos"],
            "selected_neg": selected_stats["neg"],
            "selected_pos_rate": selected_stats["pos_rate"],
            "candidate_n": candidate_stats["n"],
            "candidate_pos": candidate_stats["pos"],
            "candidate_neg": candidate_stats["neg"],
            "candidate_pos_rate": candidate_stats["pos_rate"],
            "scanned_n": scanned_stats["n"],
            "scanned_pos": scanned_stats["pos"],
            "scanned_neg": scanned_stats["neg"],
            "scanned_pos_rate": scanned_stats["pos_rate"],
            "candidate_entropy_mean": query_info.get("candidate_entropy_mean"),
            "candidate_entropy_std": query_info.get("candidate_entropy_std"),
            "selected_entropy_mean": query_info.get("selected_entropy_mean"),
            "selected_entropy_std": query_info.get("selected_entropy_std"),
        }

        self.history.setdefault("selection_stats", []).append(stats)

        # Also save as column-style arrays to make later table creation easier.
        for k, v in stats.items():
            self.history.setdefault("selection_{}".format(k), []).append(v)

        self.logger.info(
            "Selected stats | strategy={} model={} | "
            "selected n={} pos={} neg={} pos_rate={} | "
            "candidates n={} pos_rate={} | scanned n={} pos_rate={}".format(
                stats["query_strategy"],
                stats["query_model"],
                stats["selected_n"],
                stats["selected_pos"],
                stats["selected_neg"],
                "{:.4f}".format(stats["selected_pos_rate"]) if stats["selected_pos_rate"] is not None else "NA",
                stats["candidate_n"],
                "{:.4f}".format(stats["candidate_pos_rate"]) if stats["candidate_pos_rate"] is not None else "NA",
                stats["scanned_n"],
                "{:.4f}".format(stats["scanned_pos_rate"]) if stats["scanned_pos_rate"] is not None else "NA",
            )
        )
        return stats

    # ---- Query ----
    def query(self, budget):
        """Return selected indices plus REINFORCE terms.

        --query_strategy random:
            no model; randomly select from scanned unlabeled pool.

        --query_strategy entropy:
            use entropy only; select highest-entropy samples.

        --query_strategy policy:
            original behaviour: entropy pre-filter then policy sampling.
        """
        if not self.unlabeled_indices:
            return [], None, None, {}

        query_strategy = getattr(self.args, "query_strategy", "policy")
        scanned_pool = self._limit_unlabeled_pool_for_query()
        budget = max(1, min(budget, len(scanned_pool)))

        # Fast random baseline. This is important: if random does the same or better,
        # the learned policy is not adding value.
        if query_strategy == "random":
            rng = np.random.default_rng(self.args.seed + self.cycle)
            selected = rng.choice(scanned_pool, size=budget, replace=False).tolist()
            query_info = {
                "scanned_indices": scanned_pool,
                "candidate_indices": scanned_pool,
                "selected_indices": selected,
            }
            self.logger.info("Query[random]: {} from scanned pool {}".format(
                len(selected), len(scanned_pool)
            ))
            return selected, None, None, query_info

        loader = DataLoader(
            Subset(self.train_dataset, scanned_pool),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
        )

        states_chunks = []
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                imgs = batch["image"].to(self.device)
                states_chunks.append(self._compute_state(imgs))

                max_query_batches = getattr(self.args, "max_query_batches", 0)
                if max_query_batches and (batch_i + 1) >= max_query_batches:
                    break

        states = torch.cat(states_chunks, dim=0).detach()   # [N, state_dim]

        # If max_query_batches stopped early, trim scanned_pool to match states.
        scanned_pool = scanned_pool[:states.shape[0]]

        entropy_scores  = states[:, -3]
        candidate_ratio = self.args.candidate_ratio
        top_k = max(1, min(int(candidate_ratio * len(entropy_scores)), len(entropy_scores)))
        _, cand_idx     = torch.topk(entropy_scores, top_k)

        cand_idx_cpu    = cand_idx.cpu().tolist()
        cand_states     = states[cand_idx]
        cand_pool       = [scanned_pool[i] for i in cand_idx_cpu]
        cand_entropy    = entropy_scores[cand_idx]

        # Entropy-only test: no policy, no REINFORCE. This checks whether uncertainty
        # itself is useful or damaging.
        if query_strategy == "entropy":
            k = max(1, min(budget, len(cand_pool)))
            selected = cand_pool[:k]
            selected_entropy = cand_entropy[:k]
            query_info = {
                "scanned_indices": scanned_pool,
                "candidate_indices": cand_pool,
                "selected_indices": selected,
                "candidate_entropy_mean": float(cand_entropy.mean().item()),
                "candidate_entropy_std": float(cand_entropy.std().item()) if len(cand_entropy) > 1 else 0.0,
                "selected_entropy_mean": float(selected_entropy.mean().item()),
                "selected_entropy_std": float(selected_entropy.std().item()) if len(selected_entropy) > 1 else 0.0,
            }
            self.logger.info("Query[entropy]: {} from {} cands (scanned {}, full pool {})".format(
                len(selected), len(cand_pool), len(scanned_pool), len(self.unlabeled_indices)
            ))
            return selected, None, None, query_info

        # Original policy sampling from top-entropy candidates.
        global_state = cand_states.mean(dim=0)
        image_logits = self.policy(cand_states, global_state)   # [K, 1]

        budget       = max(1, min(budget, len(cand_pool)))
        image_probs  = F.softmax(
            image_logits.squeeze() / self.policy_temp, dim=0
        ).clamp_min(1e-12)

        sel_pos      = torch.multinomial(image_probs, num_samples=budget, replacement=False)
        log_prob_sum = torch.log(image_probs[sel_pos]).sum()
        entropy      = -(image_probs * torch.log(image_probs)).sum()

        selected = [cand_pool[i] for i in sel_pos.tolist()]
        selected_entropy = cand_entropy[sel_pos]

        query_info = {
            "scanned_indices": scanned_pool,
            "candidate_indices": cand_pool,
            "selected_indices": selected,
            "candidate_entropy_mean": float(cand_entropy.mean().item()),
            "candidate_entropy_std": float(cand_entropy.std().item()) if len(cand_entropy) > 1 else 0.0,
            "selected_entropy_mean": float(selected_entropy.mean().item()),
            "selected_entropy_std": float(selected_entropy.std().item()) if len(selected_entropy) > 1 else 0.0,
        }

        self.logger.info("Query[policy]: {} from {} cands (scanned {}, full pool {})".format(
            len(selected), len(cand_pool), len(scanned_pool), len(self.unlabeled_indices)
        ))
        return selected, log_prob_sum, entropy, query_info

    # ---- Main model training ----
    def train_main_model(self, val_loader, epochs):
        self.logger.info("Main training: {} epochs on {} samples".format(
            epochs, len(self.labeled_indices)
        ))
        sub    = Subset(self.train_dataset, self.labeled_indices)
        loader = DataLoader(sub, batch_size=self.args.batch_size,
                            shuffle=True, num_workers=self.args.workers)
        pos_w  = self._compute_pos_weight(self.labeled_indices)
        crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt    = optim.Adam(self.main_model.parameters(), lr=self.args.lr,
                            weight_decay=getattr(self.args, "weight_decay", 1e-4))

        last_metrics = None
        for ep in range(epochs):
            self.main_model.train()
            t0, running = time.time(), 0.0
            batches_seen = 0
            for batch_i, batch in enumerate(tqdm(loader, desc="Main ep{}/{}".format(ep + 1, epochs), leave=False)):
                imgs   = batch["image"].to(self.device)
                labels = batch["labels"].to(self.device)
                opt.zero_grad()
                loss = crit(self.main_model(imgs), labels)
                loss.backward()
                opt.step()
                running += loss.item()
                batches_seen += 1

                max_train_batches = getattr(self.args, "max_train_batches", 0)
                if max_train_batches and batches_seen >= max_train_batches:
                    break
            train_loss = running / max(1, batches_seen)
            metrics    = self.evaluate_main_model(val_loader)
            epoch_time = time.time() - t0

            self.logger.info(
                "Cycle {} ep {}/{} | loss {:.4f} | {} | {:.1f}s".format(
                    self.cycle, ep + 1, epochs, train_loss,
                    format_metrics(metrics), epoch_time,
                )
            )
            self._log_epoch(ep, train_loss, epoch_time, metrics)
            self.save_results()
            last_metrics = metrics
        return last_metrics

    def _log_epoch(self, epoch, train_loss, epoch_time, metrics):
        self.history.setdefault("epoch", []).append(epoch)
        self.history.setdefault("cycle", []).append(self.cycle)
        self.history.setdefault("labeled_count", []).append(len(self.labeled_indices))
        self.history.setdefault("train_loss", []).append(train_loss)
        self.history.setdefault("train_time", []).append(epoch_time)
        for k, v in metrics.items():
            self.history.setdefault("val_{}".format(k), []).append(v)

    # ---- Evaluation ----
    def evaluate_main_model(self, val_loader):
        self.main_model.eval()
        scores_all, labels_all = [], []
        with torch.no_grad():
            for batch_i, batch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
                imgs   = batch["image"].to(self.device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                logits = self.main_model(imgs)
                scores_all.append(torch.sigmoid(logits).cpu().numpy())
                labels_all.append(labels)

                max_val_batches = getattr(self.args, "max_val_batches", 0)
                if max_val_batches and (batch_i + 1) >= max_val_batches:
                    break
        y_score = np.concatenate(scores_all)
        y_true  = np.concatenate(labels_all).astype(np.int32)
        return binary_metrics(y_true, y_score, threshold=0.5, beta=2.0)

    # ---- Policy update ----
    def _update_policy(self, log_prob_sum, entropy, reward):
        advantage = torch.tensor(
            reward - self.reward_baseline, dtype=torch.float32, device=self.device
        )
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * reward
        )
        loss = -(advantage * log_prob_sum) - self.entropy_beta * entropy

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        return advantage.item(), loss.item()

    # ---- One cycle ----
    def run_cycle(self, val_loader, budget, cycle_epochs):
        self.policy_temp = max(
            self.policy_temp_end,
            self.args.policy_temp_start * (0.95 ** self.cycle),
        )

        test_mode = getattr(self.args, "test_mode", "al")

        # Control test: continue training without adding new samples.
        # If performance goes down here too, the problem is training schedule,
        # not selected samples.
        if test_mode == "no_query":
            self.logger.info("Cycle {}: NO-QUERY control; training without adding samples".format(self.cycle))
            metrics = self.train_main_model(val_loader, epochs=cycle_epochs)
            score = metrics["best_f2"]
            reward = 0.0 if self.prev_score is None else float(np.clip(score - self.prev_score, -0.1, 0.1))
            self.prev_score = score

            self.history.setdefault("cycle_end", []).append(self.cycle)
            self.history.setdefault("reward",      []).append(reward)
            self.history.setdefault("advantage",   []).append(None)
            self.history.setdefault("policy_loss", []).append(None)
            self.logger.info(
                "Cycle {} no-query done | best-F2 {:.4f} | delta {:+.4f} | labeled {}".format(
                    self.cycle, score, reward, len(self.labeled_indices)
                )
            )
            self.cycle += 1
            return metrics

        self.logger.info("Cycle {}: querying {} samples".format(self.cycle, budget))
        new_idx, log_prob_sum, entropy, query_info = self.query(budget)
        if not new_idx:
            self.logger.info("No samples selected; skipping.")
            self.cycle += 1
            return None

        stats = self._save_selection_stats(new_idx, query_info)

        # Query-only mode is a fast diagnostic. It checks which samples would be selected,
        # logs their label distribution, then stops before expensive retraining.
        if test_mode == "query_only":
            self.logger.info("Cycle {} query-only done; not adding samples and not training.".format(self.cycle))
            self.history.setdefault("cycle_end", []).append(self.cycle)
            self.history.setdefault("reward",      []).append(None)
            self.history.setdefault("advantage",   []).append(None)
            self.history.setdefault("policy_loss", []).append(None)
            self.save_results()
            self.cycle += 1
            return None

        new_set = set(new_idx)
        self.labeled_indices.extend(new_idx)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in new_set]

        metrics = self.train_main_model(val_loader, epochs=cycle_epochs)
        score   = metrics["best_f2"]   # primary reward signal

        if self.prev_score is None:
            reward = 0.0
        else:
            reward = float(np.clip(score - self.prev_score, -0.1, 0.1))
        self.prev_score = score

        # Only the policy strategy has log_prob/entropy. Random and entropy are controls.
        if log_prob_sum is not None and entropy is not None and getattr(self.args, "query_strategy", "policy") == "policy":
            advantage, policy_loss = self._update_policy(log_prob_sum, entropy, reward)
        else:
            advantage, policy_loss = None, None

        self.history.setdefault("cycle_end", []).append(self.cycle)
        self.history.setdefault("reward",      []).append(reward)
        self.history.setdefault("advantage",   []).append(advantage)
        self.history.setdefault("policy_loss", []).append(policy_loss)

        self.logger.info(
            "Cycle {} done | best-F2 {:.4f} | reward {:+.4f} | "
            "baseline {:.4f} | labeled {}".format(
                self.cycle, score, reward, self.reward_baseline,
                len(self.labeled_indices),
            )
        )
        self.cycle += 1
        return metrics

    # ---- Full run ----
    def run(self, val_loader):
        t_start = datetime.datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("STARTING RL AL for binary {} classification".format(self.args.target_class))
        self.logger.info("=" * 60)

        self.cold_start()

        if getattr(self.args, "query_model", "oracle") == "oracle":
            self.train_oracle_model(epochs=self.args.oracle_epochs)
        else:
            self.logger.info("Skipping oracle training because --query_model main")

        self.logger.info("Warm-up main model")
        init_metrics   = self.train_main_model(val_loader, epochs=self.args.initial_epochs)
        self.prev_score = init_metrics["best_f2"]
        self.history["init_metrics"] = init_metrics

        for c in range(self.args.al_cycles):
            self.logger.info("=" * 60)
            self.logger.info("AL Cycle {}/{} | labeled {} | unlabeled {}".format(
                c + 1, self.args.al_cycles,
                len(self.labeled_indices), len(self.unlabeled_indices),
            ))
            self.run_cycle(val_loader, budget=self.args.al_budget,
                           cycle_epochs=self.args.cycle_epochs)

        self.history["run_time"] = str(datetime.datetime.now() - t_start)
        self.save_results()
        self.logger.info("Completed in {}".format(self.history["run_time"]))
        return self.history


# ===========================
# Entry
# ===========================
def build_argparser():
    p = argparse.ArgumentParser(description="RL AL for binary RB")
    # Data
    p.add_argument("--dataroot",          type=str,   default="../../../Datasets/")
    p.add_argument("--scale_size",        type=int,   default=224)
    p.add_argument("--batch_size",        type=int,   default=64)
    p.add_argument("--workers",           type=int,   default=4)
    p.add_argument("--target_class",      type=str,   default="RB")

    # AL
    p.add_argument("--main_arch",         type=str,   default="resnet50")
    p.add_argument("--al_cycles",         type=int,   default=10)
    p.add_argument("--al_budget",         type=int,   default=1000)
    p.add_argument("--candidate_ratio",   type=float, default=0.2)

    # Cold start
    p.add_argument("--cold_start_strategy", type=str, default="clipiqa",
                   choices=["random", "balanced", "diversity", "coreset", "clipiqa"])
    p.add_argument("--initial_percentage",  type=float, default=0.005)
    p.add_argument("--secondary_strategy",  type=str,   default="balanced",
                   choices=["random", "balanced", "diversity", "coreset"])
    p.add_argument("--pos_ratio",           type=float, default=0.5,
                   help="Target positive fraction in cold start (balanced strategies)")
    p.add_argument("--clipiqa_json_path",   type=str,
                   default="../IQA/clip_iqa_train_all.json")
    p.add_argument("--clipiqa_threshold",   type=float, default=0.3)

    # Training
    p.add_argument("--oracle_epochs",       type=int,   default=15)
    p.add_argument("--initial_epochs",      type=int,   default=10)
    p.add_argument("--cycle_epochs",        type=int,   default=10)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--weight_decay",        type=float, default=1e-4)

    # Policy
    p.add_argument("--policy_lr",           type=float, default=1e-4)
    p.add_argument("--policy_hidden",       type=int,   default=256)
    p.add_argument("--policy_temp_start",   type=float, default=1.0)
    p.add_argument("--policy_temp_end",     type=float, default=0.5)
    p.add_argument("--entropy_beta",        type=float, default=1e-3)

    # Fast diagnostic tests
    p.add_argument("--query_model",         type=str,   default="oracle",
                   choices=["oracle", "main"],
                   help="Model used to compute query features/entropy. Use main to avoid stale frozen oracle.")
    p.add_argument("--query_strategy",      type=str,   default="policy",
                   choices=["policy", "entropy", "random"],
                   help="policy=original RL policy; entropy=top entropy; random=random baseline.")
    p.add_argument("--test_mode",           type=str,   default="al",
                   choices=["al", "no_query", "query_only"],
                   help="al=normal AL; no_query=continue training without adding samples; query_only=log selected sample stats only.")
    p.add_argument("--max_query_pool",      type=int,   default=0,
                   help="If >0, scan only this many unlabeled samples per query for quick tests.")
    p.add_argument("--max_query_batches",   type=int,   default=0,
                   help="If >0, stop query-state computation after this many batches.")
    p.add_argument("--max_train_batches",   type=int,   default=0,
                   help="If >0, train on only this many batches per epoch for quick smoke tests.")
    p.add_argument("--max_val_batches",     type=int,   default=0,
                   help="If >0, evaluate on only this many validation batches. Fast but approximate metrics.")

    # Misc
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--gpu",                 type=int,   default=0)
    p.add_argument("--results_dir",         type=str,   default="./results_rb")
    p.add_argument("--experiment_name",     type=str,   default="rb_al_run")
    p.add_argument("--dataset_type",        type=str,   default="sewerml")
    return p


def main():
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("GPU {}: {}".format(args.gpu, torch.cuda.get_device_name(args.gpu)))

    train_ds, val_ds = get_data(args)
    print("Train: {} ({}+: {})  |  Val: {} ({}+: {})".format(
        len(train_ds),
        args.target_class,
        int(train_ds.binary_labels.sum()) if train_ds.binary_labels is not None else -1,
        len(val_ds),
        args.target_class,
        int(val_ds.binary_labels.sum()) if val_ds.binary_labels is not None else -1,
    ))

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    pipeline = RBActiveLearningPipeline(
        train_dataset=train_ds, args=args,
        device="cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu",
    )
    history = pipeline.run(val_loader)

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    if "init_metrics" in history:
        print("Init best-F2:  {:.4f}".format(history["init_metrics"]["best_f2"]))
    if history.get("val_best_f2"):
        print("Final best-F2: {:.4f}".format(history["val_best_f2"][-1]))
        if "init_metrics" in history:
            print("Improvement:   {:+.4f}".format(
                history["val_best_f2"][-1] - history["init_metrics"]["best_f2"]
            ))
    print("Labeled: {} / {}".format(
        len(pipeline.labeled_indices), len(train_ds)
    ))


if __name__ == "__main__":
    main()
