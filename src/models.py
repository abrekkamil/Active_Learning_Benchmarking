import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import List, Dict


# ============================================================
# UNET MODEL (SEGMENTATION)
# ============================================================

class UNetModel:
    def __init__(self, num_classes, device, config):
        from networks.unet import UNetExact

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
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

    # ---------------- TRAIN ----------------
    def train_epoch(self, dataset, epoch):
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )

        total_loss = 0.0
        start = time.time()

        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {
            "train_loss": total_loss / len(loader),
            "training_time": time.time() - start
        }

    # ---------------- EVAL ----------------
    def evaluate(self, dataset):
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        ious = []

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)

                inter = ((preds == 1) & (masks == 1)).sum().item()
                union = ((preds == 1) | (masks == 1)).sum().item()
                if union > 0:
                    ious.append(inter / union)

        return {"mean_iou": float(np.mean(ious)) if ious else 0.0}

    # ---------------- PREDICT ----------------
    def predict(self, images: List[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            batch = torch.stack(images).to(self.device)
            logits = self.model(batch)
            return torch.argmax(logits, dim=1)

    # ---------------- UNCERTAINTY ----------------
    def get_uncertainty(self, images: List[torch.Tensor]):
        self.model.eval()
        scores = []

        with torch.no_grad():
            for img in images:
                logits = self.model(img.unsqueeze(0).to(self.device))
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                scores.append(entropy.item())

        return np.array(scores)


# ============================================================
# MASK R-CNN MODEL (DETECTION)
# ============================================================

class MaskRCNNModel:
    def __init__(self, num_classes, device, config):
        import pytorch_mask_rcnn as pmr
        from types import SimpleNamespace

        self.device = device
        self.config = config

        self.model = pmr.maskrcnn_resnet50(
            pretrained=True,
            num_classes=num_classes
        ).to(device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

    # ---------------- TRAIN ----------------
    def train_epoch(self, dataset, epoch):
        import pytorch_mask_rcnn as pmr
        from types import SimpleNamespace

        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )

        args = SimpleNamespace(
            lr_epoch=self.config.lr,
            iters=-1,
            print_freq=20,
            distributed=False,
            output_dir="."
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

        return {"training_time": time.time() - start}

    # ---------------- EVAL ----------------
    def evaluate(self, dataset):
        import pytorch_mask_rcnn as pmr
        from types import SimpleNamespace

        args = SimpleNamespace(
            print_freq=100,
            distributed=False,
            output_dir="."
        )

        _, _, metrics = pmr.evaluate(
            self.model,
            dataset,
            self.device,
            0,
            args
        )

        if metrics and "bbox" in metrics:
            return {"bbox_AP": metrics["bbox"]["AP@[IoU=0.50:0.95]"]}

        return {"bbox_AP": 0.0}

    # ---------------- PREDICT ----------------
    def predict(self, images: List[torch.Tensor]):
        self.model.eval()
        preds = []

        with torch.no_grad():
            for img in images:
                assert img.dim() == 3
                out = self.model(img.to(self.device))
                preds.append({k: v.cpu() for k, v in out.items()})

        return preds

    # ---------------- UNCERTAINTY ----------------
    def get_uncertainty(self, images: List[torch.Tensor]):
        scores = []

        with torch.no_grad():
            for img in images:
                out = self.model(img.to(self.device))
                if len(out["scores"]) == 0:
                    scores.append(1.0)
                else:
                    p = F.softmax(out["scores"], dim=0)
                    scores.append(-(p * torch.log(p + 1e-8)).sum().item())

        return np.array(scores)


# ============================================================
# WEAK MODEL (COLD START)
# ============================================================

class WeakModel:
    def __init__(self, num_classes, device):
        self.device = device
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device).eval()

    def _prep(self, img):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = F.interpolate(img.unsqueeze(0), (224, 224), mode="bilinear")
        return img.to(self.device)

    def get_uncertainty(self, images: List[torch.Tensor]):
        scores = []

        with torch.no_grad():
            for img in images:
                x = self._prep(img)
                logits = self.model(x)
                p = F.softmax(logits, dim=1)
                scores.append(-(p * torch.log(p + 1e-8)).sum().item())

        return np.array(scores)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class FeatureExtractor:
    """
    Extract deep features using pretrained CNNs.
    """

    def __init__(self, model_name="resnet18"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.out_dim = 512
        elif model_name == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # remove classifier
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

    def extract(self, images):
        features = []

        with torch.no_grad():
            for image in images:
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                if image.shape[1] == 1:
                    image = image.repeat(1, 3, 1, 1)

                image = F.interpolate(
                    image, size=(224, 224),
                    mode="bilinear", align_corners=False
                )

                image = image.to(self.device)
                feat = self.model(image)      # (1, C, 1, 1)
                feat = feat.view(-1).cpu()    # (C,)
                features.append(feat)

        return torch.stack(features)


# ===========================
# RL POLICY (TRUE RL: REINFORCE)
# ===========================
class PolicyNet(nn.Module):
    """
    Given state vectors [B, D], outputs logits [B] representing desirability.
    We'll softmax across candidate pool to get a distribution over actions.
    """
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        return self.net(states).squeeze(-1)  # [B]