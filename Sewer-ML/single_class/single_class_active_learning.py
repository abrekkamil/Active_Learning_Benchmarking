"""
Classical active learning for binary Sewer-ML defect classification.

This script is intentionally similar to rb_active_learning.py, but removes the
REINFORCE / PolicyNet part. It supports simple cold-start strategies and simple
query strategies so you can test random vs CLIP-IQA based selection.

Main use cases:

1) Random cold start + random query
    python rb_classical_active_learning.py \
        --target_class FS \
        --experiment_name fs_classical_random_random \
        --cold_start_strategy random \
        --query_strategy random \
        --initial_percentage 0.01 \
        --al_budget 2000 \
        --al_cycles 10

2) Highest-IQA cold start + random query
    python rb_classical_active_learning.py \
        --target_class FS \
        --experiment_name fs_classical_iqa_high_random \
        --cold_start_strategy iqa_high \
        --query_strategy random \
        --clipiqa_json_path ../IQA/clip_iqa_train_all.json \
        --initial_percentage 0.01 \
        --al_budget 2000 \
        --al_cycles 10

3) Random cold start + highest-IQA query
    python rb_classical_active_learning.py \
        --target_class FS \
        --experiment_name fs_classical_random_iqa_high \
        --cold_start_strategy random \
        --query_strategy iqa_high \
        --clipiqa_json_path ../IQA/clip_iqa_train_all.json \
        --initial_percentage 0.01 \
        --al_budget 2000 \
        --al_cycles 10

4) Lowest-IQA experiment
    python rb_classical_active_learning.py \
        --target_class FS \
        --experiment_name fs_classical_iqa_low_random \
        --cold_start_strategy iqa_low \
        --query_strategy random \
        --clipiqa_json_path ../IQA/clip_iqa_train_all.json \
        --initial_percentage 0.01 \
        --al_budget 2000 \
        --al_cycles 10

Notes:
- The script expects rb_binary_dataset.py to have the correct label order for
  your current Sewer-ML setup. If VA is excluded, the order should start with RB.
- The metric used to choose the best epoch is best-F2, same as your previous code.
- Query strategies do not simulate human labelling cost differently; they only
  decide which indices are moved from the unlabeled pool to the labeled pool.
"""

import os
import sys
import json
import time
import datetime
import logging
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from tqdm import tqdm

# Make multiprocessing more stable on HPC filesystems.
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataloaders"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataloaders.sewerml_dataset import MultiLabelDataset

from rb_binary_dataset import RBBinaryDataset
from rb_metrics import binary_metrics, format_metrics


# ===========================
# Model
# ===========================
class BinaryResNet(nn.Module):
    """ResNet with one binary output logit."""

    def __init__(self, arch: str = "resnet50", pretrained: bool = True):
        super().__init__()
        backbone_fn = getattr(models, arch)
        backbone = backbone_fn(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = backbone.fc.in_features
        self.classifier = nn.Linear(self.feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = torch.flatten(f, 1)
        return self.classifier(f).squeeze(-1)


# ===========================
# Logging
# ===========================
def setup_logging(name: str, log_path: Optional[str] = None) -> logging.Logger:
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
    val_csv = os.path.join(sewer_root, "SewerML_valid.csv")

    train_list = pd.read_csv(train_csv)["Filename"].tolist() if os.path.exists(train_csv) else []
    val_list = pd.read_csv(val_csv)["Filename"].tolist() if os.path.exists(val_csv) else []

    base_train = MultiLabelDataset(
        img_dir=sewer_root,
        image_transform=train_tf,
        labels_path=sewer_root,
        val_list=val_list,
        train_list=train_list,
        known_labels=1.0,
        testing=False,
        split="train",
    )

    base_val = MultiLabelDataset(
        img_dir=sewer_root,
        image_transform=test_tf,
        labels_path=sewer_root,
        val_list=val_list,
        train_list=train_list,
        known_labels=1.0,
        testing=True,
        split="valid",
    )

    train_ds = RBBinaryDataset(base_train, target_class=args.target_class)
    val_ds = RBBinaryDataset(base_val, target_class=args.target_class)

    train_pos = int(train_ds.binary_labels.sum()) if train_ds.binary_labels is not None else -1
    val_pos = int(val_ds.binary_labels.sum()) if val_ds.binary_labels is not None else -1
    train_rate = float(train_ds.binary_labels.mean()) if train_ds.binary_labels is not None else -1
    val_rate = float(val_ds.binary_labels.mean()) if val_ds.binary_labels is not None else -1

    print(
        "Train: {} ({}+: {}, {:.4f}) | Val: {} ({}+: {}, {:.4f})".format(
            len(train_ds), args.target_class, train_pos, train_rate,
            len(val_ds), args.target_class, val_pos, val_rate,
        )
    )

    return train_ds, val_ds


# ===========================
# CLIP-IQA utilities
# ===========================
def _get_dataset_filenames(dataset) -> List[str]:
    """Return filenames aligned with dataset indices."""
    filenames = None

    for obj in (dataset, getattr(dataset, "base", None), getattr(dataset, "dataset", None)):
        if obj is not None and hasattr(obj, "imgPaths"):
            filenames = obj.imgPaths
            break

    if filenames is None:
        raise RuntimeError("Could not find imgPaths on dataset/base/dataset wrapper.")

    if len(filenames) != len(dataset):
        if hasattr(dataset, "indices"):
            filenames = [filenames[i] for i in dataset.indices]
        else:
            raise RuntimeError(
                "Filename count {} does not match dataset length {}.".format(
                    len(filenames), len(dataset)
                )
            )

    return filenames


def load_iqa_scores(dataset, json_path: str) -> np.ndarray:
    """Load CLIP-IQA scores and align them to dataset indices."""
    if json_path is None or not os.path.exists(json_path):
        raise FileNotFoundError(
            "CLIP-IQA JSON not found. Provide --clipiqa_json_path for IQA strategies."
        )

    with open(json_path, "r") as f:
        score_dict = json.load(f)

    filenames = _get_dataset_filenames(dataset)
    scores = np.zeros(len(dataset), dtype=np.float32)
    matched = 0

    for i, fn in enumerate(filenames):
        bare = os.path.basename(fn)
        if bare in score_dict:
            scores[i] = float(score_dict[bare])
            matched += 1

    print("CLIP-IQA: matched {}/{} filenames".format(matched, len(dataset)))
    return scores


# ===========================
# Selection helpers
# ===========================
def _select_random(pool: List[int], budget: int, rng: np.random.Generator) -> List[int]:
    budget = min(budget, len(pool))
    if budget <= 0:
        return []
    return rng.choice(pool, size=budget, replace=False).astype(int).tolist()

def _select_natural_fraction(pool, labels, fraction, rng):
    """
    Select the same fraction from positives and negatives.
    Example:
        fraction = 0.1
        -> select 10% of positive samples and 10% of negative samples.
    """
    pool_arr = np.asarray(pool, dtype=int)
    pool_labels = labels[pool_arr]

    pos_pool = pool_arr[pool_labels == 1]
    neg_pool = pool_arr[pool_labels == 0]

    n_pos = int(round(len(pos_pool) * fraction))
    n_neg = int(round(len(neg_pool) * fraction))

    sel_pos = rng.choice(pos_pool, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(neg_pool, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)

    selected = np.concatenate([sel_pos, sel_neg]).astype(int).tolist()
    rng.shuffle(selected)

    return selected

def _select_balanced(
    pool: List[int],
    labels: np.ndarray,
    budget: int,
    pos_ratio: float,
    rng: np.random.Generator,
) -> List[int]:
    """Select a requested positive/negative ratio from a candidate pool."""
    pool_arr = np.asarray(pool, dtype=int)
    pool_labels = labels[pool_arr]

    pos_pool = pool_arr[pool_labels == 1]
    neg_pool = pool_arr[pool_labels == 0]

    n_pos_target = int(round(budget * pos_ratio))
    n_neg_target = budget - n_pos_target

    n_pos = min(n_pos_target, len(pos_pool))
    n_neg = min(n_neg_target, len(neg_pool))

    # Fill missing positive budget using negatives, or vice versa.
    if n_pos < n_pos_target:
        n_neg = min(budget - n_pos, len(neg_pool))
    if n_neg < n_neg_target:
        n_pos = min(budget - n_neg, len(pos_pool))

    sel_pos = rng.choice(pos_pool, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(neg_pool, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)

    selected = np.concatenate([sel_pos, sel_neg]).astype(int).tolist()
    rng.shuffle(selected)
    return selected


def _select_by_iqa(
    pool: List[int],
    iqa_scores: np.ndarray,
    budget: int,
    mode: str,
    rng: np.random.Generator,
    threshold: Optional[float] = None,
) -> List[int]:
    """
    Select from a pool using IQA values.

    Supported modes:
        iqa_high       -> highest scores
        iqa_low        -> lowest scores
        iqa_random     -> random among samples passing threshold, if threshold is set
        iqa_threshold_high -> highest among scores >= threshold
        iqa_threshold_low  -> lowest among scores <= threshold
    """
    if not pool:
        return []

    pool_arr = np.asarray(pool, dtype=int)
    scores = iqa_scores[pool_arr]

    if mode == "iqa_high":
        order = np.argsort(-scores)
        selected = pool_arr[order[:budget]]

    elif mode == "iqa_low":
        order = np.argsort(scores)
        selected = pool_arr[order[:budget]]

    elif mode == "iqa_random":
        if threshold is not None:
            candidates = pool_arr[scores >= threshold]
            if len(candidates) == 0:
                candidates = pool_arr
        else:
            candidates = pool_arr
        selected = rng.choice(candidates, size=min(budget, len(candidates)), replace=False)

    elif mode == "iqa_threshold_high":
        if threshold is None:
            raise ValueError("iqa_threshold_high requires --clipiqa_threshold")
        candidates = pool_arr[scores >= threshold]
        if len(candidates) == 0:
            raise RuntimeError("No samples pass IQA threshold >= {}".format(threshold))
        cand_scores = iqa_scores[candidates]
        order = np.argsort(-cand_scores)
        selected = candidates[order[:budget]]

    elif mode == "iqa_threshold_low":
        if threshold is None:
            raise ValueError("iqa_threshold_low requires --clipiqa_threshold")
        candidates = pool_arr[scores <= threshold]
        if len(candidates) == 0:
            raise RuntimeError("No samples pass IQA threshold <= {}".format(threshold))
        cand_scores = iqa_scores[candidates]
        order = np.argsort(cand_scores)
        selected = candidates[order[:budget]]

    else:
        raise ValueError("Unknown IQA mode: {}".format(mode))

    return selected.astype(int).tolist()


def summarize_selection(
    name: str,
    selected: List[int],
    labels: Optional[np.ndarray] = None,
    iqa_scores: Optional[np.ndarray] = None,
) -> Dict:
    report = {
        "name": name,
        "selected": int(len(selected)),
    }

    if labels is not None and len(selected) > 0:
        sel_labels = labels[np.asarray(selected, dtype=int)]
        report["positives"] = int(sel_labels.sum())
        report["negatives"] = int(len(sel_labels) - sel_labels.sum())
        report["pos_rate"] = float(sel_labels.mean())

    if iqa_scores is not None and len(selected) > 0:
        sel_scores = iqa_scores[np.asarray(selected, dtype=int)]
        report["iqa_mean"] = float(sel_scores.mean())
        report["iqa_min"] = float(sel_scores.min())
        report["iqa_max"] = float(sel_scores.max())

    return report


# ===========================
# Classical active learning pipeline
# ===========================
class ClassicalActiveLearningPipeline:
    def __init__(self, train_dataset, args, device: str = "cuda"):
        self.train_dataset = train_dataset
        self.args = args
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(args.seed)

        log_file = os.path.join(
            "logs",
            "CLASSICAL_AL_{}.log".format(datetime.datetime.now().strftime("%m%d_%H%M")),
        )
        self.logger = setup_logging("CLASSICAL_AL", log_path=log_file)

        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = list(range(len(train_dataset)))

        self.model = BinaryResNet(arch=args.arch, pretrained=True).to(self.device)
        self.history: Dict = {}
        self.cycle = 0

        self.iqa_scores: Optional[np.ndarray] = None
        if self._needs_iqa():
            self.iqa_scores = load_iqa_scores(train_dataset, args.clipiqa_json_path)

        self._init_results_path()

        self.logger.info(
            "Device: {} | train: {} | target: {} | cold_start={} | query={}".format(
                self.device,
                len(train_dataset),
                args.target_class,
                args.cold_start_strategy,
                args.query_strategy,
            )
        )

    def _needs_iqa(self) -> bool:
        iqa_strategies = {
            "iqa_high",
            "iqa_low",
            "iqa_random",
            "iqa_threshold_high",
            "iqa_threshold_low",
        }
        return (
            self.args.cold_start_strategy in iqa_strategies
            or self.args.query_strategy in iqa_strategies
        )

    def _init_results_path(self):
        date_folder = datetime.datetime.now().strftime("%m_%d")
        time_stamp = datetime.datetime.now().strftime("%H%M")
        results_dir = os.path.join(self.args.results_dir, date_folder)
        os.makedirs(results_dir, exist_ok=True)

        self.results_path = os.path.join(
            results_dir,
            "{}_{}_cold-{}_query-{}_{}.json".format(
                self.args.experiment_name,
                self.args.target_class,
                self.args.cold_start_strategy,
                self.args.query_strategy,
                time_stamp,
            ),
        )

    def save_results(self):
        with open(self.results_path, "w") as f:
            json.dump({"config": vars(self.args), "history": self.history}, f, indent=2)

    def _compute_pos_weight(self, indices: List[int], clip_max: float = 50.0) -> torch.Tensor:
        labels = self.train_dataset.binary_labels
        if labels is None:
            sub = Subset(self.train_dataset, indices)
            loader = DataLoader(sub, batch_size=128, shuffle=False, num_workers=2)
            lbs = []
            for b in loader:
                lbs.append(b["labels"].cpu().numpy())
            labels_sub = np.concatenate(lbs)
        else:
            labels_sub = labels[np.asarray(indices, dtype=int)]

        n_pos = max(1, int(labels_sub.sum()))
        n_neg = max(1, len(labels_sub) - n_pos)
        pw = float(min(n_neg / n_pos, clip_max))

        self.logger.info("pos_weight: n_pos={} n_neg={} -> {:.2f}".format(n_pos, n_neg, pw))
        return torch.tensor([pw], dtype=torch.float32, device=self.device)

    def _select_from_pool(self, strategy: str, pool: List[int], budget: int, name: str) -> List[int]:
        labels = self.train_dataset.binary_labels

        if strategy == "random":
            selected = _select_random(pool, budget, self.rng)

        elif strategy == "balanced":
            if labels is None:
                raise RuntimeError("balanced selection requires train_dataset.binary_labels")
            selected = _select_balanced(pool, labels, budget, self.args.pos_ratio, self.rng)

        elif strategy in {
            "iqa_high",
            "iqa_low",
            "iqa_random",
            "iqa_threshold_high",
            "iqa_threshold_low",
        }:
            if self.iqa_scores is None:
                raise RuntimeError("IQA strategy requested but IQA scores were not loaded.")
            selected = _select_by_iqa(
                pool=pool,
                iqa_scores=self.iqa_scores,
                budget=budget,
                mode=strategy,
                rng=self.rng,
                threshold=self.args.clipiqa_threshold,
            )

        else:
            raise ValueError("Unknown selection strategy: {}".format(strategy))

        report = summarize_selection(name, selected, labels=labels, iqa_scores=self.iqa_scores)
        self.logger.info("{} selection report: {}".format(name, report))
        self.history.setdefault("selection_reports", []).append(report)
        return selected

    def cold_start(self):
        n_init_arg = self.args.initial_percentage
        n_initial = int(n_init_arg * len(self.train_dataset)) if n_init_arg <= 1.0 else int(n_init_arg)
        n_initial = max(1, n_initial)

        self.logger.info(
            "Cold start [{}]: {} samples ({:.2f}%)".format(
                self.args.cold_start_strategy,
                n_initial,
                100.0 * n_initial / len(self.train_dataset),
            )
        )

        if self.args.cold_start_strategy == "natural":
            labels = self.train_dataset.binary_labels
            if labels is None:
                raise RuntimeError("natural cold start requires binary_labels")

            if n_init_arg > 1.0:
                raise ValueError(
                    "natural cold start expects --initial_percentage <= 1.0, "
                    "because it samples that fraction from positives and negatives."
                )

            selected = _select_natural_fraction(
                pool=self.unlabeled_indices,
                labels=labels,
                fraction=n_init_arg,
                rng=self.rng,
            )

            self.logger.info(
                "Natural cold start selected {} samples using fraction {:.4f}".format(
                    len(selected), n_init_arg
                )
            )

        else:
            selected = self._select_from_pool(
                strategy=self.args.cold_start_strategy,
                pool=self.unlabeled_indices,
                budget=n_initial,
                name="cold_start",
            )

        selected_set = set(selected)
        self.labeled_indices = list(selected)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in selected_set]

        self.history["cold_start_strategy"] = self.args.cold_start_strategy
        self.history["cold_start_labeled"] = len(self.labeled_indices)
        self.history["unlabeled_after_cold_start"] = len(self.unlabeled_indices)
        self._log_pool_distribution(stage="after_cold_start")
        self.save_results()
    def query(self, budget: int) -> List[int]:
        if not self.unlabeled_indices:
            return []

        budget = min(budget, len(self.unlabeled_indices))
        selected = self._select_from_pool(
            strategy=self.args.query_strategy,
            pool=self.unlabeled_indices,
            budget=budget,
            name="query_cycle_{}".format(self.cycle),
        )
        return selected

    def _add_to_labeled_pool(self, new_indices: List[int]):
        new_set = set(new_indices)
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in new_set]

    def train_model(self, val_loader, epochs: int):
        self.logger.info(
            "Training model: {} epochs on {} labeled samples".format(
                epochs, len(self.labeled_indices)
            )
        )

        sub = Subset(self.train_dataset, self.labeled_indices)
        loader = DataLoader(
            sub,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=False,
        )

        pos_w = self._compute_pos_weight(self.labeled_indices)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        last_metrics = None
        for ep in range(epochs):
            self.model.train()
            t0 = time.time()
            running_loss = 0.0
            n_batches = 0

            for batch in tqdm(loader, desc="Cycle {} ep {}/{}".format(self.cycle, ep + 1, epochs), leave=False):
                imgs = batch["image"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(1, n_batches)
            epoch_time = time.time() - t0
            metrics = self.evaluate(val_loader)

            self.logger.info(
                "Cycle {} ep {}/{} | loss {:.4f} | {} | {:.1f}s".format(
                    self.cycle,
                    ep + 1,
                    epochs,
                    train_loss,
                    format_metrics(metrics),
                    epoch_time,
                )
            )

            self._log_epoch(ep, train_loss, epoch_time, metrics)
            self.save_results()
            last_metrics = metrics

        return last_metrics

    def evaluate(self, val_loader):
        self.model.eval()
        scores_all = []
        labels_all = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                imgs = batch["image"].to(self.device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                logits = self.model(imgs)
                scores = torch.sigmoid(logits).cpu().numpy()

                scores_all.append(scores)
                labels_all.append(labels)

        y_score = np.concatenate(scores_all)
        y_true = np.concatenate(labels_all).astype(np.int32)
        return binary_metrics(y_true, y_score, threshold=0.5, beta=2.0)

    def _log_pool_distribution(self, stage: str):
        labels = self.train_dataset.binary_labels
        if labels is None:
            return

        labeled_arr = np.asarray(self.labeled_indices, dtype=int)
        unlabeled_arr = np.asarray(self.unlabeled_indices, dtype=int)

        labeled_labels = labels[labeled_arr] if len(labeled_arr) else np.array([])
        unlabeled_labels = labels[unlabeled_arr] if len(unlabeled_arr) else np.array([])

        record = {
            "stage": stage,
            "cycle": int(self.cycle),
            "labeled_count": int(len(self.labeled_indices)),
            "unlabeled_count": int(len(self.unlabeled_indices)),
            "labeled_pos": int(labeled_labels.sum()) if len(labeled_labels) else 0,
            "labeled_pos_rate": float(labeled_labels.mean()) if len(labeled_labels) else 0.0,
            "unlabeled_pos": int(unlabeled_labels.sum()) if len(unlabeled_labels) else 0,
            "unlabeled_pos_rate": float(unlabeled_labels.mean()) if len(unlabeled_labels) else 0.0,
        }

        self.history.setdefault("pool_distribution", []).append(record)
        self.logger.info("Pool distribution: {}".format(record))

    def _log_epoch(self, epoch: int, train_loss: float, epoch_time: float, metrics: Dict):
        self.history.setdefault("epoch", []).append(int(epoch))
        self.history.setdefault("cycle", []).append(int(self.cycle))
        self.history.setdefault("labeled_count", []).append(int(len(self.labeled_indices)))
        self.history.setdefault("train_loss", []).append(float(train_loss))
        self.history.setdefault("train_time", []).append(float(epoch_time))

        labels = self.train_dataset.binary_labels
        if labels is not None and len(self.labeled_indices) > 0:
            sel_labels = labels[np.asarray(self.labeled_indices, dtype=int)]
            self.history.setdefault("labeled_pos", []).append(int(sel_labels.sum()))
            self.history.setdefault("labeled_pos_rate", []).append(float(sel_labels.mean()))

        for k, v in metrics.items():
            self.history.setdefault("val_{}".format(k), []).append(v)

    def run_cycle(self, val_loader, budget: int, cycle_epochs: int):
        self.logger.info(
            "AL cycle {} | labeled {} | unlabeled {} | query {}".format(
                self.cycle,
                len(self.labeled_indices),
                len(self.unlabeled_indices),
                budget,
            )
        )

        new_indices = self.query(budget)
        if not new_indices:
            self.logger.info("No new samples selected; stopping this cycle.")
            return None

        self._add_to_labeled_pool(new_indices)
        self._log_pool_distribution(stage="after_query_cycle_{}".format(self.cycle))

        metrics = self.train_model(val_loader, epochs=cycle_epochs)

        self.history.setdefault("cycle_end", []).append(int(self.cycle))
        self.history.setdefault("cycle_best_f2", []).append(float(metrics["best_f2"]))
        self.history.setdefault("cycle_AP", []).append(float(metrics["AP"]))
        self.history.setdefault("cycle_AUC", []).append(float(metrics["AUC"]))

        self.logger.info(
            "Cycle {} done | best-F2 {:.4f} | labeled {}".format(
                self.cycle,
                metrics["best_f2"],
                len(self.labeled_indices),
            )
        )

        self.cycle += 1
        self.save_results()
        return metrics

    def run(self, val_loader):
        t_start = datetime.datetime.now()

        self.logger.info("=" * 60)
        self.logger.info("STARTING CLASSICAL AL for binary {} classification".format(self.args.target_class))
        self.logger.info("=" * 60)

        self.cold_start()

        self.logger.info("Initial training after cold start")
        init_metrics = self.train_model(val_loader, epochs=self.args.initial_epochs)
        self.history["init_metrics"] = init_metrics
        self.save_results()

        for c in range(self.args.al_cycles):
            self.run_cycle(
                val_loader,
                budget=self.args.al_budget,
                cycle_epochs=self.args.cycle_epochs,
            )

        self.history["run_time"] = str(datetime.datetime.now() - t_start)
        self.save_results()

        self.logger.info("Completed in {}".format(self.history["run_time"]))
        self.logger.info("Results saved to {}".format(self.results_path))
        return self.history


# ===========================
# Arguments
# ===========================
def build_argparser():
    p = argparse.ArgumentParser(description="Classical active learning for binary Sewer-ML classification")

    # Data
    p.add_argument("--dataroot", type=str, default="../../../Datasets/")
    p.add_argument("--scale_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--target_class", type=str, default="RB")

    # Model
    p.add_argument("--arch", type=str, default="resnet50")

    # Active learning
    p.add_argument("--al_cycles", type=int, default=10)
    p.add_argument("--al_budget", type=int, default=1000)
    p.add_argument("--initial_percentage", type=float, default=0.01,
                   help="If <=1, fraction of train set. If >1, absolute number of samples.")

    # Classical selection strategies
    p.add_argument(
        "--cold_start_strategy",
        type=str,
        default="random",
        choices=[
        "random",
        "natural",
        "balanced",
        "iqa_high",
        "iqa_low",
        "iqa_random",
        "iqa_threshold_high",
        "iqa_threshold_low",
    	],
    )
    p.add_argument(
        "--query_strategy",
        type=str,
        default="random",
        choices=[
            "random",
            "balanced",
            "iqa_high",
            "iqa_low",
            "iqa_random",
            "iqa_threshold_high",
            "iqa_threshold_low",
        ],
    )
    p.add_argument("--pos_ratio", type=float, default=0.3,
                   help="Target positive fraction for balanced selection.")
    p.add_argument("--clipiqa_json_path", type=str, default="../IQA/clip_iqa_train_all.json")
    p.add_argument("--clipiqa_threshold", type=float, default=0.5)

    # Training
    p.add_argument("--initial_epochs", type=int, default=10)
    p.add_argument("--cycle_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--results_dir", type=str, default="./results_rb")
    p.add_argument("--experiment_name", type=str, default="classical_al_run")
    p.add_argument("--dataset_type", type=str, default="sewerml")

    return p


# ===========================
# Main
# ===========================
def main():
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("GPU {}: {}".format(args.gpu, torch.cuda.get_device_name(args.gpu)))

    train_ds, val_ds = get_data(args)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    pipeline = ClassicalActiveLearningPipeline(
        train_dataset=train_ds,
        args=args,
        device="cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu",
    )

    history = pipeline.run(val_loader)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    if "init_metrics" in history:
        print("Init best-F2:  {:.4f}".format(history["init_metrics"]["best_f2"]))
    if history.get("val_best_f2"):
        print("Final best-F2: {:.4f}".format(history["val_best_f2"][-1]))
    print("Labeled: {} / {}".format(len(pipeline.labeled_indices), len(train_ds)))
    print("Results: {}".format(pipeline.results_path))


if __name__ == "__main__":
    main()
