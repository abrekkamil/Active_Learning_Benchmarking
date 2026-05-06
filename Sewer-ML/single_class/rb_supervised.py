"""
Supervised baseline for binary RB classification.

Two modes (controlled by --train_fraction):
  1. train_fraction = 1.0  -> full training set  (ceiling / reference)
  2. train_fraction < 1.0  -> random subset      (no-AL comparison)

Usage:
    # Full-data ceiling
    python rb_supervised.py --experiment_name rb_full --train_fraction 1.0

    # Matched budget for AL comparison (e.g. 10k samples out of ~1M)
    python rb_supervised.py --experiment_name rb_random_10k --train_fraction 0.01

The matched-budget run should use the SAME total number of samples as
your final AL run (initial + all queried samples) so the comparison is
fair.
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
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataloaders"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataloaders.sewerml_dataset import MultiLabelDataset

from rb_binary_dataset import RBBinaryDataset
from rb_metrics import binary_metrics, format_metrics


# ===========================
# Model
# ===========================
class BinaryResNet(nn.Module):
    """ResNet with a single-logit classifier for binary RB."""

    def __init__(self, arch="resnet50", pretrained=True):
        super().__init__()
        backbone_fn = getattr(models, arch)
        backbone    = backbone_fn(pretrained=pretrained)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim   = backbone.fc.in_features
        self.classifier = nn.Linear(self.feat_dim, 1)   # single logit

    def forward(self, x):
        f = self.features(x)
        f = torch.flatten(f, 1)
        return self.classifier(f).squeeze(-1)   # [B]


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

    print("Train: {} (RB+: {})  |  Val: {} (RB+: {})".format(
        len(train_ds),
        int(train_ds.binary_labels.sum()) if train_ds.binary_labels is not None else -1,
        len(val_ds),
        int(val_ds.binary_labels.sum()) if val_ds.binary_labels is not None else -1,
    ))

    return train_ds, val_ds


# ===========================
# Subset selection
# ===========================
def select_training_indices(train_ds, args):
    """Choose which training indices to actually use."""
    n_total = len(train_ds)
    rng     = np.random.default_rng(args.seed)

    if args.train_fraction >= 1.0:
        indices = list(range(n_total))
        print("Using full training set: {} samples".format(n_total))
        return indices

    # Random subset, optionally stratified by RB label
    n_target = int(args.train_fraction * n_total) if args.train_fraction <= 1.0 else int(args.train_fraction)

    if args.stratified and train_ds.binary_labels is not None:
        labels  = train_ds.binary_labels
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_pos_target = int(round(n_target * args.pos_ratio))
        n_neg_target = n_target - n_pos_target
        n_pos = min(n_pos_target, len(pos_idx))
        n_neg = min(n_neg_target, len(neg_idx))
        if n_pos < n_pos_target:
            n_neg = min(n_target - n_pos, len(neg_idx))
        sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        indices = np.concatenate([sel_pos, sel_neg]).tolist()
        print("Stratified subset: {} pos + {} neg = {} samples".format(n_pos, n_neg, len(indices)))
    else:
        indices = rng.choice(n_total, size=n_target, replace=False).tolist()
        print("Random subset: {} samples".format(len(indices)))

    return indices


# ===========================
# Training
# ===========================
def compute_pos_weight(train_ds, indices, device, clip_max=50.0):
    """pos_weight = n_neg / n_pos, clipped to avoid extreme values."""
    labels = train_ds.binary_labels
    if labels is None:
        # Fallback: iterate
        sub = Subset(train_ds, indices)
        loader = DataLoader(sub, batch_size=128, shuffle=False, num_workers=2)
        lbs = []
        for b in loader:
            lbs.append(b["labels"].cpu().numpy())
        labels_sub = np.concatenate(lbs)
    else:
        labels_sub = labels[indices]

    n_pos = max(1, int(labels_sub.sum()))
    n_neg = max(1, len(labels_sub) - n_pos)
    pw    = n_neg / n_pos
    pw    = float(min(pw, clip_max))
    print("pos_weight: n_pos={} n_neg={} -> {:.2f}".format(n_pos, n_neg, pw))
    return torch.tensor([pw], dtype=torch.float32, device=device)


def evaluate(model, val_loader, device, desc="Val"):
    model.eval()
    scores_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc, leave=False):
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()
            logits = model(imgs)
            scores = torch.sigmoid(logits).cpu().numpy()
            scores_all.append(scores)
            labels_all.append(labels)
    y_score = np.concatenate(scores_all)
    y_true  = np.concatenate(labels_all).astype(np.int32)
    return binary_metrics(y_true, y_score, threshold=0.5, beta=2.0)


def train(model, train_ds, indices, val_loader, device, args, logger, results):
    sub       = Subset(train_ds, indices)
    loader    = DataLoader(
        sub, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    pos_w     = compute_pos_weight(train_ds, indices, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f2, best_epoch = -1.0, -1
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches    = 0
        for batch in tqdm(loader, desc="Epoch {}/{}".format(ep + 1, args.epochs), leave=False):
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = running_loss / max(1, n_batches)
        epoch_time = time.time() - t0

        metrics = evaluate(model, val_loader, device, desc="Val ep{}".format(ep + 1))
        logger.info(
            "Ep {:>3d}/{} | loss {:.4f} | {} | lr {:.2e} | {:.1f}s".format(
                ep + 1, args.epochs, train_loss,
                format_metrics(metrics), optimizer.param_groups[0]["lr"], epoch_time,
            )
        )

        results["history"].append({
            "epoch":      ep + 1,
            "train_loss": train_loss,
            "epoch_time": epoch_time,
            "lr":         optimizer.param_groups[0]["lr"],
            **{k: v for k, v in metrics.items()},
        })

        if metrics["best_f2"] > best_f2:
            best_f2    = metrics["best_f2"]
            best_epoch = ep + 1
            results["best"] = {
                "epoch":   ep + 1,
                "metrics": metrics,
            }
            if args.save_model:
                ckpt_path = os.path.join(args.model_save_path,
                                         "{}_best.pth".format(args.experiment_name))
                os.makedirs(args.model_save_path, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch":            ep + 1,
                    "metrics":          metrics,
                    "args":             vars(args),
                }, ckpt_path)
                logger.info("Saved best model -> {}".format(ckpt_path))

        # Write results after every epoch
        with open(results["results_path"], "w") as f:
            json.dump(results, f, indent=2)

    logger.info("Best best-F2: {:.4f} at epoch {}".format(best_f2, best_epoch))
    return results


# ===========================
# Main
# ===========================
def build_argparser():
    p = argparse.ArgumentParser(description="Supervised binary RB classification on Sewer-ML")
    # Data
    p.add_argument("--dataroot",         type=str,   default="../../../Datasets/")
    p.add_argument("--scale_size",       type=int,   default=224)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--workers",          type=int,   default=4)
    p.add_argument("--target_class",     type=str,   default="RB",
                   help="Which class to treat as positive (default RB)")

    # Training subset
    p.add_argument("--train_fraction",   type=float, default=1.0,
                   help="Fraction of train set to use. 1.0 = full data.")
    p.add_argument("--stratified",       action="store_true",
                   help="If subsetting, enforce pos_ratio rather than pure random")
    p.add_argument("--pos_ratio",        type=float, default=0.5,
                   help="Target positive fraction when stratified=True")

    # Model & optimisation
    p.add_argument("--arch",             type=str,   default="resnet50")
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)

    # Misc
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--gpu",              type=int,   default=0)
    p.add_argument("--experiment_name",  type=str,   default="rb_supervised")
    p.add_argument("--results_dir",      type=str,   default="./results_rb")
    p.add_argument("--save_model",       action="store_true")
    p.add_argument("--model_save_path",  type=str,   default="./checkpoints_rb")
    return p


def main():
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Paths
    date_folder = datetime.datetime.now().strftime("%m_%d")
    time_stamp  = datetime.datetime.now().strftime("%H%M")
    results_dir = os.path.join(args.results_dir, date_folder)
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join("logs", "rb_supervised_{}.log".format(
        datetime.datetime.now().strftime("%m%d_%H%M")
    ))
    logger = setup_logging("rb_supervised", log_path=log_file)
    logger.info("Args: {}".format(vars(args)))

    results_path = os.path.join(
        results_dir,
        "{}_{}_tf{:g}_{}.json".format(
            args.experiment_name, args.target_class,
            args.train_fraction, time_stamp,
        ),
    )
    results = {
        "config":        vars(args),
        "results_path":  results_path,
        "history":       [],
        "best":          None,
    }

    # Data
    train_ds, val_ds = get_data(args)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    indices = select_training_indices(train_ds, args)
    results["n_train_used"] = len(indices)
    results["n_train_total"] = len(train_ds)
    if train_ds.binary_labels is not None:
        sel_labels = train_ds.binary_labels[indices]
        results["n_pos_in_train"] = int(sel_labels.sum())
        results["pos_rate_train"] = float(sel_labels.mean())

    # Model
    model = BinaryResNet(arch=args.arch, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model {} | params: {:,}".format(args.arch, n_params))

    # Train
    results = train(model, train_ds, indices, val_loader, device, args, logger, results)

    logger.info("Results saved to {}".format(results_path))
    if results["best"] is not None:
        logger.info("Best: {}".format(format_metrics(results["best"]["metrics"])))


if __name__ == "__main__":
    main()
