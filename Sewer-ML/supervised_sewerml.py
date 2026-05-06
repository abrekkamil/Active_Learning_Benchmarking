import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from torch.utils.data import DataLoader
import os
import sys
import json
import datetime
import time
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dataloaders'))

from torchvision import transforms, models
from dataloaders.sewerml_dataset import MultiLabelDataset


# ==============================================================
# Logging
# ==============================================================
def setup_logging(name, log_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        if log_path is not None:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    return logger


# ==============================================================
# Model  (identical to AL pipeline: ResNet-101)
# ==============================================================
class SewerMLModel(nn.Module):
    """
    ResNet-101 backbone for Sewer-ML multi-label classification.
    Same architecture used in the RL-AL pipeline so results
    are directly comparable.
    """

    def __init__(self, num_classes, pretrained=True):
        super(SewerMLModel, self).__init__()

        backbone        = models.resnet101(pretrained=pretrained)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim   = backbone.fc.in_features
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        feats  = self.features(x)
        feats  = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        return logits


# ==============================================================
# Data Loading
# ==============================================================
def get_data(args):
    trainTransform = transforms.Compose([
        transforms.Resize((args.scale_size, args.scale_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.523, 0.453, 0.345],
            std=[0.210, 0.199, 0.154]
        )
    ])

    testTransform = transforms.Compose([
        transforms.Resize((args.scale_size, args.scale_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.523, 0.453, 0.345],
            std=[0.210, 0.199, 0.154]
        )
    ])

    sewer_root = os.path.join(args.dataroot, 'Sewer_ML')
    anno_dir   = sewer_root

    import pandas as pd
    train_csv = os.path.join(anno_dir, 'SewerML_train.csv')
    val_csv   = os.path.join(anno_dir, 'SewerML_valid.csv')

    train_list = (
        pd.read_csv(train_csv)['Filename'].tolist()
        if os.path.exists(train_csv) else []
    )
    val_list = (
        pd.read_csv(val_csv)['Filename'].tolist()
        if os.path.exists(val_csv) else []
    )

    train_dataset = MultiLabelDataset(
        img_dir=sewer_root,
        image_transform=trainTransform,
        labels_path=anno_dir,
        val_list=val_list,
        train_list=train_list,
        known_labels=1.0,
        testing=False,
        split='train'
    )
    valid_dataset = MultiLabelDataset(
        img_dir=sewer_root,
        image_transform=testTransform,
        labels_path=anno_dir,
        val_list=val_list,
        train_list=train_list,
        known_labels=1.0,
        testing=True,
        split='valid'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print("Train: {} | Val: {}".format(
        len(train_dataset), len(valid_dataset)
    ))
    return train_loader, valid_loader, train_dataset.class_weights


# ==============================================================
# Evaluation
# ==============================================================
CIW = {
    "VA": 0.0310, "RB": 1.0000, "OB": 0.5518, "PF": 0.2896,
    "DE": 0.1622, "FS": 0.6419, "IS": 0.1847, "RO": 0.3559,
    "IN": 0.3131, "AF": 0.0811, "BE": 0.2275, "FO": 0.2477,
    "GR": 0.0901, "PH": 0.4167, "PB": 0.4167, "OS": 0.9009,
    "OP": 0.3829, "OK": 0.4396
}
CIW_WEIGHT_SUM = sum(CIW.values())   # 6.2938


def evaluate(model, val_loader, device, logger):
    class_names = val_loader.dataset.LabelNames
    model.eval()

    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            imgs   = batch['image'].to(device)
            labels = batch['labels'].to(device)
            scores = torch.sigmoid(model(imgs))
            preds  = (scores > 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    y_pred  = np.concatenate(all_preds)
    y_true  = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

    macro_f1 = f1_score(y_true, y_pred, average='macro',  zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro',  zero_division=0)
    ov_p     = precision_score(y_true, y_pred, average='micro', zero_division=0)
    ov_r     = recall_score(y_true,    y_pred, average='micro', zero_division=0)
    ov_f1    = (2 * ov_p * ov_r) / (ov_p + ov_r + 1e-8)
    pc_p     = precision_score(y_true, y_pred, average='macro', zero_division=0)
    pc_r     = recall_score(y_true,    y_pred, average='macro', zero_division=0)
    pc_f1    = (2 * pc_p * pc_r) / (pc_p + pc_r + 1e-8)
    zero_one = float(np.mean(np.all(y_true == y_pred, axis=1)))
    mAP      = average_precision_score(y_true, y_score, average='macro')

    # CIW-F2 (normalised)
    beta        = 2
    ciw_f2_raw  = 0.0
    per_class   = {}
    ciw_w_sum  = sum(CIW.get(c, 0.0) for c in class_names)

    for i, cls in enumerate(class_names):
        w    = CIW.get(cls, 0.0)
        p_c  = float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        r_c  = float(recall_score(y_true[:, i],    y_pred[:, i], zero_division=0))
        f1_c = float(f1_score(y_true[:, i],        y_pred[:, i], zero_division=0))
        ap_c = float(average_precision_score(y_true[:, i], y_score[:, i]))
        f2_c = (1 + beta ** 2) * (p_c * r_c) / ((beta ** 2 * p_c) + r_c + 1e-8)
        n_pos_true = int(y_true[:, i].sum())
        n_pos_pred = int(y_pred[:, i].sum())
        ciw_f2_raw += w * f2_c

        per_class[cls] = dict(
            ciw=w,
            precision=round(p_c,  4),
            recall=round(r_c,     4),
            f1=round(f1_c,        4),
            ap=round(ap_c,        4),
            f2=round(f2_c,        4),
            ciw_f2=round(w * f2_c, 6),
            n_pos_true=n_pos_true,
            n_pos_pred=n_pos_pred,
        )

    ciw_f2 = ciw_f2_raw / (ciw_w_sum + 1e-8)
    # --------------------------------------------------
    # Log per-class table to logger
    # --------------------------------------------------
    header = (
        "{:<6} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>9} {:>9}".format(
            "Class", "CIW", "P", "R", "F1", "AP", "F2",
            "CIW-F2", "TruePos", "PredPos"
        )
    )
    logger.info("Per-class metrics:")
    logger.info(header)
    logger.info("-" * len(header))
    for cls in class_names:
        m = per_class[cls]
        logger.info(
            "{:<6} {:>5.3f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} "
            "{:>7.4f} {:>8.6f} {:>9d} {:>9d}".format(
                cls,
                m["ciw"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["ap"],
                m["f2"],
                m["ciw_f2"],
                m["n_pos_true"],
                m["n_pos_pred"],
            )
        )
    logger.info("-" * len(header))

    # Flag classes with low F1 relative to their CIW importance
    critical_threshold = 0.3
    weak_classes = [
        cls for cls in class_names
        if per_class[cls]["ciw"] >= 0.3
        and per_class[cls]["f1"] < critical_threshold
    ]
    if weak_classes:
        logger.info(
            "WARNING: high-CIW classes below F1={}: {}".format(
                critical_threshold,
                ", ".join(
                    "{} (CIW={:.2f} F1={:.4f})".format(
                        c, per_class[c]["ciw"], per_class[c]["f1"]
                    )
                    for c in weak_classes
                )
            )
        )

    metrics = dict(
        macro_f1=macro_f1, micro_f1=micro_f1,
        ov_p=ov_p, ov_r=ov_r, ov_f1=ov_f1,
        pc_p=pc_p, pc_r=pc_r, pc_f1=pc_f1,
        zero_one=zero_one, mAP=mAP, ciw_f2=ciw_f2,
        per_class=per_class,
    )
    logger.info(
        "m-F1: {:.4f} | M-F1: {:.4f} | mAP: {:.4f} | CIW-F2: {:.4f}".format(
            micro_f1, macro_f1, mAP, ciw_f2
        )
    )

    return metrics


# ==============================================================
# Results saving  (day folder + time tag, same as AL pipeline)
# ==============================================================
def init_results_path(args):
    date_folder = datetime.datetime.now().strftime("%m_%d")
    results_dir = os.path.join(args.results_dir, date_folder)
    os.makedirs(results_dir, exist_ok=True)

    time_stamp   = datetime.datetime.now().strftime("%H%M")
    results_path = os.path.join(
        results_dir,
        "{}_supervised_{}.json".format(args.experiment_name, time_stamp)
    )
    return results_path


def save_results(results_path, config, history):
    results = {"config": config, "history": history}
    print(f"Saving results to {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


# ==============================================================
# Training loop
# ==============================================================
def train(args):
    system_start = datetime.datetime.now()

    # Directories
    os.makedirs(args.results_dir,     exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs("logs",               exist_ok=True)

    # Results path
    results_path = init_results_path(args)

    # Logger (stream + file)
    log_file = os.path.join(
        "logs",
        "supervised_{}.log".format(
            datetime.datetime.now().strftime("%m%d_%H%M")
        )
    )
    logger = setup_logging("SewerML_Supervised", log_path=log_file)
    logger.info("Results will be saved to: {}".format(results_path))

    # Device
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info("GPU {}: {}".format(
            args.gpu, torch.cuda.get_device_name(args.gpu)
        ))
    else:
        logger.info("CUDA not available, using CPU")

    # Data
    logger.info("Loading data from {}".format(args.dataroot))
    train_loader, valid_loader, class_weights = get_data(args)
    logger.info("Train batches: {} | Val batches: {}".format(
        len(train_loader), len(valid_loader)
    ))

    # Model
    model = SewerMLModel(num_classes=17, pretrained=True).to(device)
    logger.info("Model: ResNet-101 | Params: {:,}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))

    # Loss
    if args.use_weighted_loss:
        pos_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Using weighted BCEWithLogitsLoss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Using standard BCEWithLogitsLoss")

    # Optimiser + scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # History
    history = {
        "train_loss":    [],
        "epoch_time":    [],
        "val_macro_f1":  [],
        "val_micro_f1":  [],
        "val_ov_f1":     [],
        "val_ov_p":      [],
        "val_ov_r":      [],
        "val_pc_f1":     [],
        "val_pc_p":      [],
        "val_pc_r":      [],
        "val_zero_one":  [],
        "val_mAP":       [],
        "val_ciw_f2":    [],
        "best_epoch":    0,
        "best_macro_f1": 0.0,
        "best_map":      0.0,
        "best_ciw_f2":   0.0,
    }

    best_macro_f1  = 0.0
    best_ckpt_path = os.path.join(args.model_save_path, "best_model.pth")
    last_ckpt_path = os.path.join(args.model_save_path, "last_model.pth")

    logger.info("Starting training for {} epochs".format(args.epochs))

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        total_loss   = 0.0
        total_batches = len(train_loader)

        for batch in tqdm(
            train_loader,
            desc="Epoch {}/{}".format(epoch + 1, args.epochs),
            leave=False
        ):
            imgs   = batch['image'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss   = total_loss / total_batches
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_loss)
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch {}/{} | loss {:.4f} | lr {:.2e} | time {:.1f}s".format(
                epoch + 1, args.epochs, avg_loss, current_lr, epoch_time
            )
        )

        # Validate every eval_freq epochs and on the last epoch
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, valid_loader, device, logger)

            history["val_macro_f1"].append(metrics["macro_f1"])
            history["val_micro_f1"].append(metrics["micro_f1"])
            history["val_ov_f1"].append(metrics["ov_f1"])
            history["val_ov_p"].append(metrics["ov_p"])
            history["val_ov_r"].append(metrics["ov_r"])
            history["val_pc_f1"].append(metrics["pc_f1"])
            history["val_pc_p"].append(metrics["pc_p"])
            history["val_pc_r"].append(metrics["pc_r"])
            history["val_zero_one"].append(metrics["zero_one"])
            history["val_mAP"].append(metrics["mAP"])
            history["val_ciw_f2"].append(metrics["ciw_f2"])
            per_class = metrics["per_class"]
            if per_class:
                history.setdefault("per_class_history", []).append({
                    "epoch":        epoch,
                    "classes":      per_class,
                })

            logger.info(
                "Epoch {}/{} | Val M-F1: {:.4f} | mAP: {:.4f} | CIW-F2: {:.4f}\n ".format(
                    epoch + 1, args.epochs, metrics["macro_f1"],
                    metrics["mAP"], metrics["ciw_f2"]
                )
            )
            # Save best checkpoint by macro-F1
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                history["best_epoch"]    = epoch + 1
                history["best_macro_f1"] = best_macro_f1
                history["best_map"]      = metrics["mAP"]
                history["best_ciw_f2"]   = metrics["ciw_f2"]

                torch.save({
                    "epoch":            epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "metrics":          metrics,
                }, best_ckpt_path)
                logger.info(
                    "Best model saved (epoch {} | M-F1 {:.4f})".format(
                        epoch + 1, best_macro_f1
                    )
                )

            save_results(results_path, vars(args), history)

    # Save last checkpoint
    torch.save({
        "epoch":            args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
    }, last_ckpt_path)

    run_time    = str(datetime.datetime.now() - system_start)
    history["run_time"] = run_time
    save_results(results_path, vars(args), history)

    logger.info("Training complete in {}".format(run_time))
    logger.info("Best epoch: {} | M-F1: {:.4f} | mAP: {:.4f} | CIW-F2: {:.4f}".format(
        history["best_epoch"],
        history["best_macro_f1"],
        history["best_map"],
        history["best_ciw_f2"]
    ))
    logger.info("Results saved to: {}".format(results_path))

    return history


# ==============================================================
# Entry point
# ==============================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Supervised baseline - Sewer-ML'
    )

    # Data
    parser.add_argument('--dataroot',type=str, default='../../Datasets/')
    parser.add_argument('--scale_size',      type=int,   default=224)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--test_batch_size', type=int,   default=32)
    parser.add_argument('--workers',         type=int,   default=8)

    # Training
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--eval_freq',       type=int,   default=1,
        help='Validate every N epochs')
    parser.add_argument('--use_weighted_loss', action='store_true', default=True)

    # Results
    parser.add_argument('--results_dir',     type=str,   default='./results')
    parser.add_argument('--experiment_name', type=str,   default='sewerml_supervised')
    parser.add_argument('--model_save_path', type=str,   default='./checkpoints_supervised')
    parser.add_argument('--gpu',             type=int,   default=0)

    args = parser.parse_args()

    train(args)
