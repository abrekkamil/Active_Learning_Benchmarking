
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
)
from typing import List

from .utils import _ensure_rgb, _batched

class DeepLabWithBottleneck(nn.Module):
    """
    Wrap torchvision DeepLabV3 to expose get_bottleneck_features(x)
    like your UNetExact does.
    """
    def __init__(self, deeplab_model: nn.Module):
        super().__init__()
        self.deeplab = deeplab_model

    def forward(self, x):
        return self.deeplab(x)

    def get_bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return global pooled backbone features: [B, C]
        Uses backbone 'out' feature map.
        """
        feats = self.deeplab.backbone(x)["out"]          # [B, C, H, W]
        return feats.mean(dim=[2, 3])                    # [B, C]
    


class DeepLabV3Model:
    """
    Torchvision DeepLabV3 wrapper for semantic segmentation.

    Assumes dataset returns:
      - image: Tensor [3,H,W]
      - mask:  Tensor [H,W] (class indices) OR [C,H,W] one-hot
    """

    def __init__(self, num_classes: int, device: torch.device, config):


        self.device = device
        self.config = config
        self.num_classes = num_classes

        pretrained = getattr(config, "pretrained", False)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None

        base_model = deeplabv3_resnet50(weights=weights)

        # Replace classifier head
        base_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.model = DeepLabWithBottleneck(base_model).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-4),
            weight_decay=getattr(config, "weight_decay", 0.0)
        )

        self.criterion = nn.CrossEntropyLoss()

    # ---- passthrough ----
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # ---- training ----
    def train_epoch(self, dataset, epoch: int, total_epochs: int):
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 4),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            pin_memory=True,
            drop_last=True,
        )

        total_loss = 0.0
        start = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            if masks.dim() == 4:
                masks = masks.argmax(dim=1)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)["out"]  # [B,C,H,W]
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {
            "train_loss": float(total_loss / max(len(loader), 1)),
            "training_time": float(time.time() - start),
        }

    # ---- evaluation ----
    def evaluate(self, dataset):
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

                logits = self.model(images)["out"]
                preds = torch.argmax(logits, dim=1)

                pixel_accs.append((preds == masks).float().mean().item())

                # IoU & Dice (binary assumption skip background)
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

        precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
        recall    = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        f1        = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
        iou_px    = jaccard_score(all_targets, all_preds, pos_label=1, zero_division=0)
        acc       = accuracy_score(all_targets, all_preds)

        return {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou_pixel": float(iou_px),
            "accuracy": float(acc),
            "pixel_acc": float(np.mean(pixel_accs)),
        }
    def get_bottleneck_features(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract backbone features for RL state.
        Returns pooled feature vector per image: [N, C]
        """
        self.model.eval()

        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)

            # Get backbone features
            features = self.model.backbone(batch)["out"]   # [B, C, H, W]

            # Global average pooling
            pooled = F.adaptive_avg_pool2d(features, 1)    # [B, C, 1, 1]
            pooled = pooled.view(pooled.size(0), -1)       # [B, C]

        return pooled.cpu()
    # ---- inference ----
    def predict(self, images: List[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            logits = self.model(batch)["out"]
            return torch.argmax(logits, dim=1).cpu()

    # ---- uncertainty (entropy) ----
    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        self.model.eval()
        scores = []

        with torch.no_grad():
            for img in images:
                img = _ensure_rgb(img).unsqueeze(0).to(self.device)
                logits = self.model(img)["out"]
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

