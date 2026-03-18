"""
Multi-label image classification model for Sewer-ML (and any multi-label task).

Backbone: ResNet-50 (pretrained on ImageNet).
Head    : single Linear(2048, num_classes) – no sigmoid (raw logits).
Loss    : BCEWithLogitsLoss with per-class pos_weight for class imbalance.

Implements the unified BaseModel API:
    train_epoch / evaluate / predict / get_uncertainty / save / load
    + get_bottleneck_features   (required by RL pipeline)

Evaluation metrics returned by evaluate():
    macro_f1, micro_f1, per_class_f1 (list), map (mean average precision),
    hamming_loss, subset_accuracy
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    hamming_loss,
    accuracy_score,
)
from typing import List, Dict, Optional
from tqdm import tqdm

from .base_model import BaseModel
from .utils import _ensure_rgb, _batched


# ---------------------------------------------------------------------------
# Backbone wrapper that exposes get_bottleneck_features()
# ---------------------------------------------------------------------------

class ResNetMultiLabel(nn.Module):
    """
    ResNet-50 backbone + multi-label head.
    Mirrors the get_bottleneck_features() interface used by rl_active_learning.py.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original classifier
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # [B,2048,1,1]
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                    # [B,2048,1,1]
        feat = feat.view(feat.size(0), -1)         # [B,2048]
        return self.classifier(feat)               # [B,num_classes]  logits

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features [B, 2048] – no grad, for RL state."""
        with torch.no_grad():
            feat = self.features(x)                # [B,2048,1,1]
            return feat.view(feat.size(0), -1)     # [B,2048]


# ---------------------------------------------------------------------------
# BaseModel wrapper
# ---------------------------------------------------------------------------

class MultiLabelClassificationModel(BaseModel):
    """
    Multi-label classification wrapper following the repo's unified API.

    task_type = 'multilabel_classification'

    Primary metric (used by get_primary_metric in active_learning.py):
        returned as 'macro_f1'

    Dataset convention (same as DeepCrack / UNet): __getitem__ returns
        (image: Tensor[3,H,W],  label: Tensor[num_classes])

    Parameters
    ----------
    num_classes : int
        Number of label dimensions (17 for Sewer-ML).
    device : torch.device
    config : ActiveLearningConfig
        Uses: lr, weight_decay, batch_size, num_workers, pretrained.
        Optional: class_weights (Tensor[num_classes]) for pos_weight.
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        super().__init__(num_classes, device, config)

        self.task_type = "multilabel_classification"

        pretrained = getattr(config, "pretrained", True)
        self.model = ResNetMultiLabel(num_classes, pretrained=pretrained).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-4),
            weight_decay=getattr(config, "weight_decay", 0.0),
        )

        # Class-imbalance weighting – can be passed in config or left as None
        pos_weight: Optional[torch.Tensor] = getattr(config, "class_weights", None)
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Prediction threshold (default 0.5, can be tuned per-class)
        self.threshold = getattr(config, "cls_threshold", 0.5)

    # -----------------------------------------------------------------------
    # Passthrough
    # -----------------------------------------------------------------------

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 32),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

        total_loss = 0.0
        start = time.time()
        pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)

        for images, labels in pbar:
            images = images.to(self.device)           # [B,3,H,W]
            labels = labels.to(self.device).float()   # [B,C]

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)               # [B,C]
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n = max(len(loader), 1)
        return {
            "train_loss":   float(total_loss / n),
            "training_time": float(time.time() - start),
        }

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate(self, dataset) -> Dict[str, float]:
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 32),
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 4),
            pin_memory=True,
        )

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Eval", leave=False):
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()   # [N, C]
        all_labels = torch.cat(all_labels, dim=0).numpy()   # [N, C]
        all_preds  = (all_logits > 0.0).astype(int)         # threshold at logit=0

        # --- F1 ---
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro",  zero_division=0))
        micro_f1 = float(f1_score(all_labels, all_preds, average="micro",  zero_division=0))
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()

        # --- mAP (mean average precision) ---
        try:
            # average_precision_score needs at least one positive per class
            probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
            map_score = float(average_precision_score(all_labels, probs, average="macro"))
        except Exception:
            map_score = 0.0

        # --- Hamming loss ---
        h_loss = float(hamming_loss(all_labels, all_preds))

        # --- Subset (exact match) accuracy ---
        subset_acc = float(accuracy_score(all_labels, all_preds))

        return {
            "macro_f1":      macro_f1,
            "micro_f1":      micro_f1,
            "map":           map_score,
            "hamming_loss":  h_loss,
            "subset_acc":    subset_acc,
            "per_class_f1":  per_class_f1,
            # expose as 'f1' so existing get_primary_metric() still works
            "f1":            macro_f1,
        }

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns binary prediction tensor [N, C].
        """
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            logits = self.model(batch)
            return (torch.sigmoid(logits) > self.threshold).float().cpu()

    # -----------------------------------------------------------------------
    # Uncertainty (for query strategies)
    # -----------------------------------------------------------------------

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Predictive uncertainty = mean binary entropy across all labels.

        H(y) = - sum_c [ p_c * log(p_c) + (1-p_c) * log(1-p_c) ] / C

        Higher → the model is most unsure which labels apply.
        """
        self.model.eval()
        scores = []

        with torch.no_grad():
            for img in images:
                img = _ensure_rgb(img).unsqueeze(0).to(self.device)
                logits = self.model(img)                          # [1, C]
                probs  = torch.sigmoid(logits)                   # [1, C]

                # Binary entropy per class, then mean
                eps = 1e-8
                entropy = -(
                    probs * torch.log(probs + eps)
                    + (1 - probs) * torch.log(1 - probs + eps)
                ).mean(dim=1)                                      # [1]

                scores.append(float(entropy.item()))

        return np.array(scores, dtype=np.float32)

    # -----------------------------------------------------------------------
    # RL feature extraction (required by rl_active_learning.py)
    # -----------------------------------------------------------------------

    def get_bottleneck_features(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract backbone features for RL state construction.
        Returns [N, 2048].
        """
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            return self.model.get_bottleneck_features(batch).cpu()

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "state_dict":  self.model.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "num_classes": self.num_classes,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
