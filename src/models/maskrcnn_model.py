import time
import numpy as np
import torch

from typing import List, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base_model import BaseModel
from .utils import _ensure_rgb


class MaskRCNNModel(BaseModel):
    """
    Torchvision Mask R-CNN wrapper for instance segmentation.

    Dataset must return:
        image : Tensor [3,H,W]
        target: dict with keys
            boxes  : Tensor [N,4]
            labels : Tensor [N]
            masks  : Tensor [N,H,W]
    """

    def __init__(self, num_classes: int, device: torch.device, config):

        super().__init__(num_classes, device, config)

        self.task_type = "instance_segmentation"

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model = maskrcnn_resnet50_fpn(weights=None)

        # Replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes,
        )

        # Replace mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes,
        )

        self.model = model.to(device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-3),
            momentum=getattr(config, "momentum", 0.9),
            weight_decay=getattr(config, "weight_decay", 0.0),
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 2),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True,
        )

        self.model.train()

        start = time.time()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

        for images, targets in pbar:

            images = [_ensure_rgb(img).to(self.device) for img in images]

            targets = [
                {k: v.to(self.device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()

            pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        return {
            "train_loss": float(total_loss / max(len(loader), 1)),
            "training_time": float(time.time() - start),
        }

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def evaluate(self, dataset) -> Dict[str, float]:

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True,
        )

        self.model.eval()

        scores = []

        with torch.no_grad():

            pbar = tqdm(loader, desc="Validation", leave=False)

            for images, targets in pbar:

                images = [_ensure_rgb(img).to(self.device) for img in images]

                outputs = self.model(images)

                if len(outputs[0]["scores"]) > 0:
                    scores.append(outputs[0]["scores"].mean().item())

        if len(scores) == 0:
            return {"bbox_score": 0.0}

        return {
            "bbox_score": float(np.mean(scores))
        }

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:

        self.model.eval()

        imgs = [_ensure_rgb(img).to(self.device) for img in images]

        with torch.no_grad():

            outputs = self.model(imgs)

        preds = []

        for out in outputs:

            preds.append({
                k: v.detach().cpu()
                for k, v in out.items()
            })

        return preds

    # --------------------------------------------------
    # Active Learning uncertainty
    # --------------------------------------------------

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:

        self.model.eval()

        preds = self.predict(images)

        scores = []

        for p in preds:

            if "scores" not in p or len(p["scores"]) == 0:

                scores.append(1.0)

            else:

                s = p["scores"].float()

                k = min(len(s), 5)

                topk_mean = torch.topk(s, k).values.mean().item()

                scores.append(float(np.clip(1.0 - topk_mean, 0.0, 1.0)))

        return np.array(scores, dtype=np.float32)

    # --------------------------------------------------
    # Checkpointing
    # --------------------------------------------------

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