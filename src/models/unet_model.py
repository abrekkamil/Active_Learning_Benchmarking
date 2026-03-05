import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict
from .utils import _ensure_rgb, _batched

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
)

class UNetModel:
    """
    Wrapper for U-Net semantic segmentation.
    Assumes dataset returns:
      - image: Tensor [3,H,W]
      - mask:  Tensor [H,W] (class indices) OR [C,H,W] one-hot
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        from src.networks.unet import UNetExact

        self.device = device
        self.config = config
        self.num_classes = num_classes

        self.model = UNetExact(
            in_channels=3,
            out_channels=num_classes,
            norm=getattr(config, "unet_norm", "bn")
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-3),
            weight_decay=getattr(config, "weight_decay", 0.0)
        )

        self.criterion = nn.CrossEntropyLoss()

    # ---- passthrough (needed by QueryStrategies) ----
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # ---- core API ----
    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 8),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            pin_memory=True,
        )

        total_loss = 0.0
        start = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False,)
        for images, masks in pbar:
            # images: BCHW
            images = images.to(self.device)

            # masks can be [B,H,W] or [B,C,H,W]
            masks = masks.to(self.device)
            if masks.dim() == 4:
                masks = masks.argmax(dim=1)  # [B,H,W]

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)              # [B,C,H,W]
            loss = self.criterion(logits, masks)     # CE expects class index mask
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return {
            "train_loss": float(total_loss / max(len(loader), 1)),
            "training_time": float(time.time() - start),
        }
    
    def _compute_iou_and_dice(self, preds, targets):
        ious, dices = [], []
        tp = fp = fn = 0

        for c in range(1, self.num_classes):  # skip background
            p = preds == c
            t = targets == c

            inter = (p & t).sum().item()
            union = (p | t).sum().item()
            denom = p.sum().item() + t.sum().item()

            tp += inter
            fp += (p & ~t).sum().item()
            fn += (~p & t).sum().item()

            if union > 0:
                ious.append(inter / union)
            if denom > 0:
                dices.append(2 * inter / denom)

        return (
            float(np.mean(ious)) if ious else 0.0,
            float(np.mean(dices)) if dices else 0.0,
            tp, fp, fn,
        )

    def evaluate(self, dataset) -> Dict[str, float]:
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 2),
            pin_memory=True,
        )

        ious, dices, pixel_accs = [], [], []
        all_preds, all_targets = [], []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Validation", leave=False)
            tp = fp = fn = 0
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                if masks.dim() == 4:
                    masks = masks.argmax(dim=1)

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)

                # ---------- segmentation metrics ----------
                pixel_accs.append((preds == masks).float().mean().item())

                miou, mdice, tp_i, fp_i, fn_i = self._compute_iou_and_dice(preds, masks)
                ious.append(miou)
                dices.append(mdice)
                tp += tp_i
                fp += fp_i
                fn += fn_i
                f1_running = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                # ---------- pixel-wise metrics ----------
                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_targets.append(masks.cpu().numpy().reshape(-1))

                # ---------- live display ----------
                pbar.set_postfix(
                    dice=f"{np.mean(dices):.4f}",
                    iou=f"{np.mean(ious):.4f}",
                    f1=f"{f1_running:.4f}",
                )

        # ---------- aggregate pixel-wise ----------
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
        recall    = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        f1        = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
        iou_px    = jaccard_score(all_targets, all_preds, pos_label=1, zero_division=0)
        acc       = accuracy_score(all_targets, all_preds)

        return {
            # region-based (segmentation)
            "mean_iou": float(np.mean(ious)),
            "dice": float(np.mean(dices)),

            # pixel-wise
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou_pixel": float(iou_px),
            "accuracy": float(acc),

            # auxiliary
            "pixel_acc": float(np.mean(pixel_accs)),
        }

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Return predicted class mask(s): [B,H,W]."""
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            logits = self.model(batch)
            return torch.argmax(logits, dim=1).cpu()

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Pixel entropy averaged over image, averaged over pixels.
        Returns shape [N].
        """
        self.model.eval()
        scores = []

        with torch.no_grad():
            for img in images:
                img = _ensure_rgb(img).unsqueeze(0).to(self.device)  # [1,3,H,W]
                logits = self.model(img)                               # [1,C,H,W]
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                scores.append(float(entropy.item()))

        return np.array(scores, dtype=np.float32)

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_classes": self.num_classes,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

