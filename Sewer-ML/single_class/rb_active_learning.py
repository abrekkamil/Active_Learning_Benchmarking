"""
REINFORCE-based active learning for binary RB classification.

Structure mirrors your original multi-label pipeline so results are
directly comparable. Key simplifications:
  - State uses a single entropy scalar (binary entropy on RB probability)
    plus the backbone feature vector
  - No CIW / priority machinery (only one class)
  - Reward = delta in best-F2 (primary metric for binary task)
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

        # PolicyNet: state = [bottleneck_feats | entropy | 1-confidence | 1-margin]
        # feat_dim for resnet34 = 512
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
        """State = [feats | entropy | 1-confidence | 1-margin] all from oracle."""
        with torch.no_grad():
            feats  = self.oracle_model.get_bottleneck_features(images)  # [B, D]
            logits = self.oracle_model(images)                           # [B]
            p      = torch.sigmoid(logits)                               # [B]
            eps    = 1e-8

            # Binary entropy on scalar probability
            entropy    = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            confidence = torch.max(p, 1 - p)
            margin     = (p - 0.5).abs()

            uncertainty = torch.stack(
                [entropy, 1.0 - confidence, 1.0 - margin], dim=1
            )                                                            # [B, 3]
            return torch.cat([feats, uncertainty], dim=1)

    # ---- Query ----
    def query(self, budget):
        if not self.unlabeled_indices:
            return [], None, None

        loader = DataLoader(
            Subset(self.train_dataset, self.unlabeled_indices),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
        )
        states_chunks = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(self.device)
                states_chunks.append(self._compute_state(imgs))
        states = torch.cat(states_chunks, dim=0).detach()   # [N, state_dim]

        # Entropy-based pre-filter (top candidate_ratio)
        entropy_scores  = states[:, -3]
        candidate_ratio = self.args.candidate_ratio
        top_k = max(1, min(int(candidate_ratio * len(entropy_scores)), len(entropy_scores)))
        _, cand_idx     = torch.topk(entropy_scores, top_k)
        cand_idx_cpu    = cand_idx.cpu().tolist()
        cand_states     = states[cand_idx]
        cand_pool       = [self.unlabeled_indices[i] for i in cand_idx_cpu]

        # Policy forward
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
        self.logger.info("Query: {} from {} cands (pool {})".format(
            len(selected), len(cand_pool), len(self.unlabeled_indices)
        ))
        return selected, log_prob_sum, entropy

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
            for batch in tqdm(loader, desc="Main ep{}/{}".format(ep + 1, epochs), leave=False):
                imgs   = batch["image"].to(self.device)
                labels = batch["labels"].to(self.device)
                opt.zero_grad()
                loss = crit(self.main_model(imgs), labels)
                loss.backward()
                opt.step()
                running += loss.item()
            train_loss = running / max(1, len(loader))
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
            for batch in tqdm(val_loader, desc="Val", leave=False):
                imgs   = batch["image"].to(self.device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                logits = self.main_model(imgs)
                scores_all.append(torch.sigmoid(logits).cpu().numpy())
                labels_all.append(labels)
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

        self.logger.info("Cycle {}: querying {} samples".format(self.cycle, budget))
        new_idx, log_prob_sum, entropy = self.query(budget)
        if not new_idx:
            self.logger.info("No samples selected; skipping.")
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

        advantage, policy_loss = self._update_policy(log_prob_sum, entropy, reward)

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
        self.train_oracle_model(epochs=self.args.oracle_epochs)

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
        print("Improvement:   {:+.4f}".format(
            history["val_best_f2"][-1] - history["init_metrics"]["best_f2"]
        ))
    print("Labeled: {} / {}".format(
        len(pipeline.labeled_indices), len(train_ds)
    ))


if __name__ == "__main__":
    main()
