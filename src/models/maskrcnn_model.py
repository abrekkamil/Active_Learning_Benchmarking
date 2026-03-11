import time
import numpy as np
import torch
import torch.nn as nn

from typing import List, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNeXt101_64X4D_Weights

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base_model import BaseModel
from .utils import _ensure_rgb

from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval


class MaskRCNNWithBottleneck(nn.Module):
    """
    Wrapper for torchvision Mask R-CNN to expose get_bottleneck_features().
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, targets=None):
        return self.model(x, targets)

    def get_bottleneck_features(self, x: torch.Tensor):
        """
        Extract backbone FPN features and global average pool them.

        Returns:
            tensor [B, C]
        """

        feats = self.model.backbone(x)

        # choose first FPN level
        f = list(feats.values())[0]   # [B,256,H,W]

        return f.mean(dim=[2, 3])     # [B,256]


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
        backbone = resnet_fpn_backbone(
            backbone_name="resnext101_64x4d",
            weights=ResNeXt101_64X4D_Weights.DEFAULT,
            trainable_layers=3
        )
        model = MaskRCNN(
            backbone,
            num_classes=num_classes
        )
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

        self.model = MaskRCNNWithBottleneck(model).to(device)

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

        for images, targets_raw in pbar:

            images = [_ensure_rgb(img).to(self.device) for img in images]

            targets = []
            for t in targets_raw:
                targets.append({
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in t.items()
                })
            if all(len(t["boxes"]) == 0 for t in targets):
                continue
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
    def evaluate(self, dataset):

        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
        )

        coco_gt = dataset.coco
        # Fix missing COCO fields
        if "info" not in coco_gt.dataset:
            coco_gt.dataset["info"] = {}

        if "licenses" not in coco_gt.dataset:
            coco_gt.dataset["licenses"] = []
        results = []

        with torch.no_grad():

            for i, (images, targets) in enumerate(tqdm(loader, desc="Validation")):

                images = [_ensure_rgb(img).to(self.device) for img in images]

                outputs = self.model(images)

                image_id = int(dataset.ids[i])

                output = outputs[0]

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks = output["masks"].cpu().numpy()

                for box, score, label, mask in zip(boxes, scores, labels, masks):

                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1

                    # Convert mask to RLE
                    binary_mask = (mask[0] > 0.5).astype(np.uint8)
                    rle = maskUtils.encode(np.asfortranarray(binary_mask))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                        "segmentation": rle,
                    })

        if len(results) == 0:
            return {
                "bbox_AP": 0.0,
                "mask_AP": 0.0,
            }

        coco_dt = coco_gt.loadRes(results)

        # -------- bbox AP --------
        coco_eval_bbox = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()

        bbox_ap = coco_eval_bbox.stats[0]

        # -------- mask AP --------
        coco_eval_mask = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval_mask.evaluate()
        coco_eval_mask.accumulate()
        coco_eval_mask.summarize()

        mask_ap = coco_eval_mask.stats[0]

        return {
            "bbox_AP": float(bbox_ap),
            "mask_AP": float(mask_ap),
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

    def get_uncertainty(self, images):

        self.model.eval()

        imgs = [_ensure_rgb(img).to(self.device) for img in images]

        uncertainties = []

        with torch.no_grad():

            outputs = self.model(imgs)

            for out in outputs:

                scores = out["scores"]

                if len(scores) == 0:
                    uncertainty = 1.0
                else:
                    p = scores.clamp(min=1e-6, max=1-1e-6)
                    entropy = -(p * torch.log(p)).mean().item()
                    uncertainty = entropy

                uncertainties.append(uncertainty)

        return torch.tensor(uncertainties)

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