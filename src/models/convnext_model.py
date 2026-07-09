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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvNeXtFPNSegNet(nn.Module):
    """ConvNeXt encoder + lightweight FPN decoder for semantic segmentation.

    Fixes vs the single-scale version:
      * taps ALL FOUR backbone stages (strides 4/8/16/32) instead of only the
        stride-32 output, so thin cracks remain representable;
      * predicts at stride 4 and upsamples x4 (instead of x32);
      * applies ImageNet normalisation internally (the AL pipeline feeds 0..1).
    """

    # torchvision convnext `.features` children:
    # [0]=stem(s4) [1]=stage1 [2]=down [3]=stage2 [4]=down [5]=stage3 [6]=down [7]=stage4
    STAGE_ENDS = (1, 3, 5, 7)

    def __init__(self, variant: str, num_classes: int, pretrained: bool = True,
                 fpn_channels: int = 128, dropout: float = 0.1):
        super().__init__()
        backbone, dims = self._build_backbone(variant, pretrained)
        self.variant = variant
        self.feature_dim = dims[-1]
        self.stages = nn.ModuleList()
        prev = 0
        for end in self.STAGE_ENDS:
            self.stages.append(nn.Sequential(*list(backbone.features.children())[prev:end + 1]))
            prev = end + 1

        self.lateral = nn.ModuleList(nn.Conv2d(d, fpn_channels, 1) for d in dims)
        self.smooth = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @staticmethod
    def _build_backbone(variant: str, pretrained: bool) -> Tuple[nn.Module, Tuple[int, ...]]:
        from torchvision.models import (
            ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
            ConvNeXt_Small_Weights, ConvNeXt_Tiny_Weights,
            convnext_base, convnext_large, convnext_small, convnext_tiny,
        )
        builders = {
            "tiny":  (convnext_tiny,  ConvNeXt_Tiny_Weights.DEFAULT,  (96, 192, 384, 768)),
            "small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, (96, 192, 384, 768)),
            "base":  (convnext_base,  ConvNeXt_Base_Weights.DEFAULT,  (128, 256, 512, 1024)),
            "large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT, (192, 384, 768, 1536)),
        }
        variant = variant.lower()
        if variant not in builders:
            raise ValueError(f"Unknown ConvNeXt variant '{variant}'. Use tiny/small/base/large.")
        builder, weights, dims = builders[variant]
        return builder(weights=weights if pretrained else None), dims

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        # AL pipeline feeds float images in [0,1]; skip if already normalised.
        if x.min() >= -0.1:
            x = (x - self._mean) / self._std
        return x

    def _encode(self, x: torch.Tensor):
        feats = []
        out = self._normalise(x)
        for stage in self.stages:
            out = stage(out)
            feats.append(out)
        return feats  # strides 4, 8, 16, 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        c2, c3, c4, c5 = self._encode(x)
        p5 = self.lateral[3](c5)
        p4 = self.lateral[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        logits = self.head(self.smooth(p2))                       # stride 4
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        c5 = self._encode(x)[-1]
        return F.adaptive_avg_pool2d(c5, 1).flatten(1)            # [B, feature_dim]


class ConvNeXtModel(BaseModel):
    """Active-learning wrapper: same API as UNetModel / SegFormerModel."""

    def __init__(self, num_classes: int, device: torch.device, config):
        super().__init__(num_classes, device, config)
        self.task_type = "semantic_segmentation"

        self.variant = getattr(config, "convnext_variant", "tiny")
        self.model = ConvNeXtFPNSegNet(
            variant=self.variant,
            num_classes=num_classes,
            pretrained=getattr(config, "pretrained", True),
            fpn_channels=getattr(config, "convnext_fpn_channels", 128),
            dropout=getattr(config, "convnext_dropout", 0.1),
        ).to(device)

        # IMPORTANT: do NOT reuse config.lr (SGD-era 1.25e-3) for AdamW fine-tuning.
        lr_backbone = getattr(config, "convnext_lr", 6e-5)
        lr_head = lr_backbone * getattr(config, "convnext_head_lr_mult", 10.0)
        decoder_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            (backbone_params if name.startswith("stages") else decoder_params).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": lr_backbone},
             {"params": decoder_params, "lr": lr_head}],
            weight_decay=getattr(config, "weight_decay", 1e-4),
        )

        weight = None
        w_crack = getattr(config, "crack_class_weight", None)   # e.g. 5.0; optional
        if w_crack is not None and num_classes == 2:
            weight = torch.tensor([1.0, float(w_crack)], device=device)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=getattr(config, "ignore_index", -100))

    # ---- passthrough ----
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # ---- RL feature hook ----
    def get_bottleneck_features(self, images):
        self.model.eval()
        with torch.no_grad():
            if isinstance(images, list):
                batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            else:
                batch = images.to(self.device)
            return self.model.get_bottleneck_features(batch).detach().cpu()

    # ---- core API ----
    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        loader = DataLoader(dataset,
                            batch_size=getattr(self.config, "batch_size", 4),
                            shuffle=True,
                            num_workers=getattr(self.config, "num_workers", 2),
                            pin_memory=True,
                            drop_last=getattr(self.config, "drop_last", False))
        total_loss, start = 0.0, time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            if masks.dim() == 4:
                masks = masks.argmax(dim=1)
            masks = masks.long()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(images), masks)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})
        return {"train_loss": float(total_loss / max(len(loader), 1)),
                "training_time": float(time.time() - start)}

    def evaluate(self, dataset) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=getattr(self.config, "num_workers", 2),
                            pin_memory=True)
        ious, dices, pixel_accs = [], [], []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Validation", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)
                if masks.dim() == 4:
                    masks = masks.argmax(dim=1)
                masks = masks.long()
                preds = torch.argmax(self.model(images), dim=1)
                pixel_accs.append((preds == masks).float().mean().item())
                for c in range(1, self.num_classes):
                    p, t = preds == c, masks == c
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
        return {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "precision": float(precision_score(all_targets, all_preds, pos_label=1, zero_division=0)),
            "recall": float(recall_score(all_targets, all_preds, pos_label=1, zero_division=0)),
            "f1": float(f1_score(all_targets, all_preds, pos_label=1, zero_division=0)),
            "iou_pixel": float(jaccard_score(all_targets, all_preds, pos_label=1, zero_division=0)),
            "accuracy": float(accuracy_score(all_targets, all_preds)),
            "pixel_acc": float(np.mean(pixel_accs)) if pixel_accs else 0.0,
        }

    def forward_model(self, images):
        return self.model(images)

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            return torch.argmax(self.model(batch), dim=1).cpu()

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        self.model.eval()
        scores = []
        with torch.no_grad():
            for img in images:
                x = _ensure_rgb(img).unsqueeze(0).to(self.device)
                probs = F.softmax(self.model(x), dim=1)
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                scores.append(float(ent.item()))
        return np.array(scores, dtype=np.float32)

    def save(self, path: str):
        torch.save({"state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "variant": self.variant,
                    "num_classes": self.num_classes}, path if path.endswith(".pt") else path + ".pt")

    def load(self, path: str):
        ckpt = torch.load(path if path.endswith(".pt") else path + ".pt",
                          map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])