"""
Model definitions for Active Learning Benchmarking.

Wrappers provide a unified API across tasks:
- train_epoch(dataset, epoch) -> dict
- evaluate(dataset) -> dict
- predict(images) -> predictions
- get_uncertainty(images) -> np.ndarray
- train() / eval() passthrough
"""

import time
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


# ============================================================
# Helpers
# ============================================================

def _ensure_chw(img: torch.Tensor) -> torch.Tensor:
    """Ensure image is CHW tensor."""
    if not isinstance(img, torch.Tensor):
        raise TypeError("Expected torch.Tensor image.")
    if img.dim() == 2:
        img = img.unsqueeze(0)  # 1HW
    if img.dim() != 3:
        raise ValueError(f"Expected image dim=3 (C,H,W) or dim=2 (H,W), got {img.shape}")
    return img


def _ensure_rgb(img: torch.Tensor) -> torch.Tensor:
    """If 1-channel, repeat to 3-channel."""
    img = _ensure_chw(img)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img


def _batched(imgs: List[torch.Tensor]) -> torch.Tensor:
    """Stack list of CHW into BCHW."""
    imgs = [_ensure_chw(i) for i in imgs]
    return torch.stack(imgs, dim=0)


# ============================================================
# UNET MODEL (SEGMENTATION)
# ============================================================

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
    def train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
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

        for images, masks in loader:
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

        ious = []

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                if masks.dim() == 4:
                    masks = masks.argmax(dim=1)  # [1,H,W]

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)  # [1,H,W]

                # binary IoU (class 1 = foreground)
                inter = ((preds == 1) & (masks == 1)).sum().item()
                union = ((preds == 1) | (masks == 1)).sum().item()
                if union > 0:
                    ious.append(inter / union)

        return {"mean_iou": float(np.mean(ious)) if len(ious) else 0.0}

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


# ============================================================
# MASK R-CNN MODEL (DETECTION / INSTANCE SEGMENTATION)
# ============================================================

class MaskRCNNModel:
    """
    Wrapper around the local `src/pytorch_mask_rcnn` implementation.

    Expects dataset __getitem__ -> (image: Tensor[3,H,W], target: dict)
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        import src.pytorch_mask_rcnn as pmr

        self.device = device
        self.config = config
        self.num_classes = num_classes

        self.model = pmr.maskrcnn_resnet50(
            pretrained=True,
            num_classes=num_classes
        ).to(device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-3),
            momentum=getattr(config, "momentum", 0.9),
            weight_decay=getattr(config, "weight_decay", 0.0),
        )

    # ---- passthrough (needed by QueryStrategies) ----
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # ---- core API ----
    def train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
        import src.pytorch_mask_rcnn as pmr

        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 2),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True,
        )

        args = SimpleNamespace(
            lr_epoch=getattr(self.config, "lr", 1e-3),
            iters=getattr(self.config, "iters", -1),
            print_freq=getattr(self.config, "print_freq", 20),
            distributed=False,
            output_dir=getattr(self.config, "output_dir", "."),
        )

        start = time.time()
        pmr.train_one_epoch(
            self.model,
            self.optimizer,
            loader,
            self.device,
            epoch,
            args
        )

        return {"training_time": float(time.time() - start)}

    def evaluate(self, dataset) -> Dict[str, float]:
        import src.pytorch_mask_rcnn as pmr

        # their evaluate might accept dataset; if not, pass DataLoader
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True,
        )

        args = SimpleNamespace(
            print_freq=getattr(self.config, "print_freq", 100),
            distributed=False,
            output_dir=getattr(self.config, "output_dir", "."),
        )

        _, _, metrics = pmr.evaluate(
            self.model,
            loader,
            self.device,
            0,
            args
        )

        if metrics and "bbox" in metrics:
            ap = metrics["bbox"].get("AP@[IoU=0.50:0.95]", 0.0)
            return {"bbox_AP": float(ap)}

        return {"bbox_AP": 0.0}

    def predict(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Returns list of predictions (torchvision-style):
          [{"boxes":..., "labels":..., "scores":..., "masks":...}, ...]
        """
        self.model.eval()

        imgs = [_ensure_rgb(img).to(self.device) for img in images]

        with torch.no_grad():
            # IMPORTANT: torchvision-style expects list[Tensor]
            outputs = self.model(imgs)

        preds: List[Dict[str, torch.Tensor]] = []
        for out in outputs:
            preds.append({k: v.detach().cpu() for k, v in out.items()})
        return preds

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Entropy over detection scores (per image). If no detections -> 1.0
        """
        self.model.eval()
        scores = []

        preds = self.predict(images)
        for p in preds:
            if "scores" not in p or len(p["scores"]) == 0:
                scores.append(1.0)
            else:
                s = p["scores"]
                prob = F.softmax(s, dim=0)
                ent = -(prob * torch.log(prob + 1e-8)).sum().item()
                scores.append(float(ent))

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


# ============================================================
# WEAK MODEL (COLD START)
# ============================================================

class WeakModel:
    """
    Small classification model used for uncertainty scoring in cold start.

    Input: list of images [C,H,W]
    Output: entropy over logits
    """

    def __init__(self, num_classes: int, device: torch.device, pretrained: bool = True):
        self.device = device
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.model = tv_models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device).eval()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        img = _ensure_rgb(img)
        img = img.unsqueeze(0)  # [1,3,H,W]
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        return img.to(self.device)

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        xs = torch.cat([self._prep(im) for im in images], dim=0)
        with torch.no_grad():
            return self.model(xs).cpu()

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        logits = self.predict(images)  # [N,C]
        p = F.softmax(logits, dim=1)
        ent = -(p * torch.log(p + 1e-8)).sum(dim=1)
        return ent.numpy().astype(np.float32)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class FeatureExtractor:
    """
    Extract deep features using pretrained CNNs.
    Returns tensor of shape [N, out_dim]
    """

    def __init__(self, model_name: str = "resnet18", pretrained: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet18(weights=weights)
            self.out_dim = 512
        elif model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet50(weights=weights)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def extract(self, images: List[torch.Tensor]) -> torch.Tensor:
        feats = []

        with torch.no_grad():
            for image in images:
                image = _ensure_rgb(image).unsqueeze(0)  # [1,3,H,W]
                image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
                image = image.to(self.device)

                feat = self.model(image)          # [1,C,1,1]
                feat = feat.view(1, -1).cpu()     # [1,C]
                feats.append(feat)

        return torch.cat(feats, dim=0)  # [N,C]


# ===========================
# RL POLICY (TRUE RL: REINFORCE)
# ===========================

class PolicyNet(nn.Module):
    """
    Given state vectors [B, D], outputs logits [B] representing desirability.
    We'll softmax across candidate pool to get a distribution over actions.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)  # [B]
