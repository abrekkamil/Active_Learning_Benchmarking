import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

from .base_model import BaseModel
from .utils import _batched, _ensure_rgb


class ConvNeXtSegmentationNet(nn.Module):
    """
    ConvNeXt encoder + lightweight segmentation decoder.

    ConvNeXt is an image-classification backbone, so for semantic segmentation
    we keep only `model.features`, attach a small convolutional decode head, and
    upsample logits back to the input image size.
    """

    def __init__(
        self,
        variant: str,
        num_classes: int,
        pretrained: bool = True,
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        backbone, feature_dim = self._build_backbone(variant, pretrained)

        self.variant = variant
        self.feature_dim = feature_dim
        self.backbone = backbone.features

        self.decode_head = nn.Sequential(
            nn.Conv2d(feature_dim, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    @staticmethod
    def _build_backbone(variant: str, pretrained: bool) -> Tuple[nn.Module, int]:
        try:
            from torchvision.models import (
                ConvNeXt_Base_Weights,
                ConvNeXt_Large_Weights,
                ConvNeXt_Small_Weights,
                ConvNeXt_Tiny_Weights,
                convnext_base,
                convnext_large,
                convnext_small,
                convnext_tiny,
            )
        except ImportError as exc:
            raise ImportError(
                "ConvNeXt requires a torchvision version that includes ConvNeXt. "
                "Upgrade torchvision, e.g. `pip install -U torchvision`, or use "
                "one of the existing models."
            ) from exc

        variant = variant.lower()
        builders = {
            "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
            "small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
            "base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT, 1024),
            "large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT, 1536),
        }

        if variant not in builders:
            raise ValueError(
                f"Unknown ConvNeXt variant '{variant}'. "
                "Use one of: tiny, small, base, large."
            )

        builder, default_weights, feature_dim = builders[variant]
        weights = default_weights if pretrained else None
        return builder(weights=weights), feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)                         # [B, C, h, w]
        logits = self.decode_head(features)                 # [B, num_classes, h, w]
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                         # [B, C, h, w]
        return F.adaptive_avg_pool2d(features, 1).flatten(1) # [B, C]


class ConvNeXtModel(BaseModel):
    """
    Active-learning wrapper for ConvNeXt semantic segmentation.

    It follows the same API as UNetModel / DeepLabV3Model / SegFormerModel:
      - train_epoch(dataset, epoch, total_epochs)
      - evaluate(dataset)
      - predict(images)
      - get_uncertainty(images)
      - save(path) / load(path)
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        super().__init__(num_classes, device, config)
        self.task_type = "semantic_segmentation"

        self.variant = getattr(config, "convnext_variant", "tiny")
        self.pretrained = getattr(config, "pretrained", True)
        self.decoder_channels = getattr(config, "convnext_decoder_channels", 256)
        self.dropout = getattr(config, "convnext_dropout", 0.1)

        self.model = ConvNeXtSegmentationNet(
            variant=self.variant,
            num_classes=num_classes,
            pretrained=self.pretrained,
            decoder_channels=self.decoder_channels,
            dropout=self.dropout,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-4),
            weight_decay=getattr(config, "weight_decay", 1e-4),
        )

        ignore_index = getattr(config, "ignore_index", -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 4),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            pin_memory=True,
            drop_last=getattr(self.config, "drop_last", False),
        )

        total_loss = 0.0
        start = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            if masks.dim() == 4:
                masks = masks.argmax(dim=1)
            masks = masks.long()

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

        return {
            "train_loss": float(total_loss / max(len(loader), 1)),
            "training_time": float(time.time() - start),
        }

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

            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                if masks.dim() == 4:
                    masks = masks.argmax(dim=1)
                masks = masks.long()

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)

                pixel_accs.append((preds == masks).float().mean().item())

                # Region metrics: skip background class 0.
                for c in range(1, self.num_classes):
                    p = preds == c
                    t = masks == c

                    inter = (p & t).sum().item()
                    union = (p | t).sum().item()
                    denom = p.sum().item() + t.sum().item()

                    if union > 0:
                        ious.append(inter / union)
                    if denom > 0:
                        dices.append(2 * inter / denom)

                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_targets.append(masks.cpu().numpy().reshape(-1))

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        if self.num_classes == 2:
            precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
            recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
            f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
            iou_px = jaccard_score(all_targets, all_preds, pos_label=1, zero_division=0)
        else:
            labels = list(range(1, self.num_classes))
            precision = precision_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
            recall = recall_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
            f1 = f1_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
            iou_px = jaccard_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)

        acc = accuracy_score(all_targets, all_preds)

        return {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou_pixel": float(iou_px),
            "accuracy": float(acc),
            "pixel_acc": float(np.mean(pixel_accs)) if pixel_accs else 0.0,
        }

    def forward_model(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            logits = self.model(batch)
            return torch.argmax(logits, dim=1).cpu()

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """Mean pixel entropy. Returns one uncertainty score per image."""
        self.model.eval()
        scores = []

        with torch.no_grad():
            for img in images:
                x = _ensure_rgb(img).unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                scores.append(float(entropy.item()))

        return np.array(scores, dtype=np.float32)

    def get_bottleneck_features(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            features = self.model.get_bottleneck_features(batch)
        return features.detach().cpu()

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_classes": self.num_classes,
                "variant": self.variant,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
