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
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
)
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
# YOLOv8 MODEL (DETECTION / INSTANCE SEGMENTATION)
# ============================================================
class YOLOv8Model:
    """
    Wrapper for Ultralytics YOLOv8 (detection or instance segmentation).

    Notes:
    - Ultralytics training expects a YOLO-format dataset on disk + a data.yaml.
      This wrapper therefore assumes your config provides `yolo_data_yaml`.
    - predict(images) accepts List[Tensor CHW] and returns Ultralytics Results objects.
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        config,
        weights: Optional[str] = None,   # e.g. "yolov8n.pt" or "yolov8n-seg.pt"
        task: str = "detect",            # "detect" or "segment"
    ):
        from ultralytics import YOLO

        self.device = device
        self.config = config
        self.num_classes = num_classes
        self.task = task

        self.weights = weights or ("yolov8n-seg.pt" if task == "segment" else "yolov8n.pt")
        self.model = YOLO(self.weights)

        # Move model to device (Ultralytics handles internally too, but we keep a flag)
        self._device_str = "cuda" if (device.type == "cuda") else "cpu"

    # ---- passthrough ----
    def fit(
            self,
            data_yaml: str,
            imgsz: int = 640,
            epochs: int = 1,
            batch: int = 8,
            workers: int = 2,
            device: Optional[Union[int, str]] = None,
            project: Optional[str] = None,
            name: Optional[str] = None,
            resume: bool = False,
            **kwargs,
        ) -> Dict[str, float]:
            """
            Train YOLOv8 using Ultralytics trainer.
            Returns a small dict for logging.
            """
            dev = device
            if dev is None:
                dev = 0 if self._device_str == "cuda" else "cpu"

            results = self.model.train(
                data=data_yaml,
                imgsz=imgsz,
                epochs=epochs,
                batch=batch,
                workers=workers,
                device=dev,
                project=project,
                name=name,
                resume=resume,
                **kwargs,
            )

            # best-effort metrics extraction (varies by ultralytics version)
            out = {}
            try:
                out["fitness"] = float(getattr(results, "fitness", 0.0))
            except Exception:
                pass
            return out

    def eval(self):
        # Inference is via model.predict(...)
        return

    # ---- core API ----
    def train_epoch(self, dataset, epoch: int, total_epochs: int = 1) -> Dict[str, float]:
        """
        'dataset' is unused for YOLO; training uses config.yolo_data_yaml.
        This keeps compatibility with your AL pipeline signature.
        """
        data_yaml = getattr(self.config, "yolo_data_yaml", None)
        if data_yaml is None:
            raise ValueError("config must include `yolo_data_yaml` (created by prepare_yolo_dataset).")

        return self.fit(
            data_yaml=str(data_yaml),
            imgsz=getattr(self.config, "img_size", 640),
            epochs=1,
            batch=getattr(self.config, "batch_size", 8),
            workers=getattr(self.config, "num_workers", 2),
            device=0 if self._device_str == "cuda" else "cpu",
            project=getattr(self.config, "yolo_project", None),
            name=getattr(self.config, "yolo_name", None),
            resume=True if epoch > 1 else False,
            verbose=False,
        )

    def evaluate(self, dataset=None) -> Dict[str, float]:
        """
        Runs Ultralytics validation (requires same data.yaml).
        Returns mAP metrics if available.
        """
        data_yaml = getattr(self.config, "yolo_data_yaml", None)
        if data_yaml is None:
            raise ValueError("config must include `yolo_data_yaml` (path to YOLO data.yaml).")

        imgsz = getattr(self.config, "imgsz", 640)
        batch = getattr(self.config, "batch_size", 8)

        metrics = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch,
            device=self._device_str,
            verbose=False,
        )

        out: Dict[str, float] = {}

        # Ultralytics exposes different attributes across versions; try common ones
        # Detection:
        for k in ["map50", "map", "map75"]:
            v = getattr(metrics, k, None)
            if v is not None:
                out[f"bbox_{k}"] = float(v)

        # Segmentation (if available):
        # Some versions provide metrics.seg.map, etc.
        try:
            seg = getattr(metrics, "seg", None)
            if seg is not None:
                for k in ["map50", "map", "map75"]:
                    v = getattr(seg, k, None)
                    if v is not None:
                        out[f"seg_{k}"] = float(v)
        except Exception:
            pass

        return out if out else {"metric": 0.0}

    def predict(self, images: List[torch.Tensor]):
        """
        Returns a list of Ultralytics Results.
        We pass tensors directly (BCHW). Ultralytics supports numpy/torch inputs.
        """
        self.model.predictor = None  # avoid stale predictor settings across calls

        batch = torch.stack([_ensure_rgb(im) for im in images], dim=0)

        # Ultralytics expects float in 0..255 or 0..1; we keep your tensor as-is.
        # If your tensors are normalized, you may want to de-normalize before passing.
        results = self.model.predict(
            source=batch,
            device=self._device_str,
            verbose=False,
        )
        return results

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Simple uncertainty from confidences:
          - no detections/masks => 1.0
          - else => 1 - mean(top-k conf)
        """
        results = self.predict(images)
        scores: List[float] = []

        for r in results:
            conf = None
            # r.boxes.conf for detection; for seg it's still boxes + masks
            try:
                if hasattr(r, "boxes") and r.boxes is not None and hasattr(r.boxes, "conf"):
                    conf = r.boxes.conf
            except Exception:
                conf = None

            if conf is None or len(conf) == 0:
                scores.append(1.0)
                continue

            conf = conf.detach().float().cpu()
            k = min(len(conf), 5)
            topk_mean = torch.topk(conf, k).values.mean().item()
            scores.append(float(np.clip(1.0 - topk_mean, 0.0, 1.0)))

        return np.array(scores, dtype=np.float32)

    def save(self, path: str):
        """
        Ultralytics manages checkpoints in runs/ directory.
        This method exports weights to a given path (best-effort).
        """
        # export current model weights
        self.model.save(path)

    def load(self, path: str):
        from ultralytics import YOLO
        self.model = YOLO(path)


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
            pretrained=False,
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
    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
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
            warmup_iters=max(1000, len(dataset))
        )

        start = time.time()
        print(f"Epoch {epoch}/{total_epochs}")
        pmr.train_one_epoch(
            self.model,
            self.optimizer,
            dataset,
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
        Supports:
        - Custom src.pytorch_mask_rcnn (expects Tensor image)
        - Torchvision Mask R-CNN (expects List[Tensor])
        """
        self.model.eval()

        imgs = [_ensure_rgb(img).to(self.device) for img in images]

        with torch.no_grad():
            try:
                # Try torchvision-style first: model(List[Tensor]) -> List[Dict]
                outputs = self.model(imgs)

                # If it returned a dict (single-image custom behavior), normalize to list
                if isinstance(outputs, dict):
                    outputs = [outputs]

            except AttributeError as e:
                # Common failure mode for custom model: it expects Tensor, not list
                # Fall back to per-image inference: model(Tensor) -> Dict
                outputs = []
                for im in imgs:
                    out = self.model(im)
                    if isinstance(out, dict):
                        outputs.append(out)
                    else:
                        # If some implementation returns list even for single image
                        outputs.extend(out)

        preds: List[Dict[str, torch.Tensor]] = []
        for out in outputs:
            preds.append({k: v.detach().cpu() for k, v in out.items()})
        return preds

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Detection uncertainty heuristic.
        High uncertainty if:
        - no detections
        - low confidence detections
        """
        self.model.eval()
        preds = self.predict(images)
        scores = []

        for p in preds:
            if "scores" not in p or len(p["scores"]) == 0:
                # No detections = very uncertain
                scores.append(1.0)
            else:
                s = p["scores"].float()
                k = min(len(s), 5)
                topk_mean = torch.topk(s, k).values.mean().item()
                scores.append(float(np.clip(1.0 - topk_mean, 0.0, 1.0)))

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
# DEEPLABV3 MODEL (Semantic Segmentation)
# ============================================================

class DeepLabV3Model:
    """
    Torchvision DeepLabV3 wrapper for semantic segmentation.

    Assumes dataset returns:
      - image: Tensor [3,H,W]
      - mask:  Tensor [H,W] (class indices) OR [C,H,W] one-hot
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

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


# ============================================================
# SEGFORMER MODEL (Semantic Segmentation, Transformers)
# ============================================================

class SegFormerModel:
    """
    Wrapper for HuggingFace SegFormer semantic segmentation.

    Dataset expected:
      - image: Tensor [3,H,W] float (preferably 0..1)
      - mask:  Tensor [H,W] (class indices) OR [C,H,W] one-hot

    Notes:
      - SegFormer uses LayerNorm (no BatchNorm issues with small batches).
      - HF model forward returns an output object with `.logits` and optionally `.loss`.
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        from transformers import SegformerForSemanticSegmentation

        self.device = device
        self.config = config
        self.num_classes = num_classes

        ckpt = getattr(
            config,
            "segformer_ckpt",
            "nvidia/segformer-b0-finetuned-ade-512-512",
        )

        base_model = SegformerForSemanticSegmentation.from_pretrained(
            ckpt,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # wrap model so RL can access bottleneck features
        self.model = SegFormerWithBottleneck(base_model).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(config, "lr", 6e-5),
            weight_decay=getattr(config, "weight_decay", 0.01),
        )

    # ---- passthrough ----
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # ---- RL feature hook ----
    def get_bottleneck_features(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns pooled transformer features [B, hidden_dim] for RL state.
        Uses SegFormer encoder last_hidden_state: [B, seq, hidden]
        """
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            enc = self.model.segformer(pixel_values=batch, return_dict=True)  # BaseModelOutput
            # last_hidden_state: [B, seq, hidden]
            feats = enc.last_hidden_state.mean(dim=1)  # [B, hidden]
        return feats.detach().cpu()

    # ---- core API ----
    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 4),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            pin_memory=True,
            drop_last=getattr(self.config, "drop_last", False),  # optional
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

            out = self.model(pixel_values=images, labels=masks, return_dict=True)
            loss = out.loss

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

        return {
            "train_loss": float(total_loss / max(len(loader), 1)),
            "training_time": float(time.time() - start),
        }

    def evaluate(self, dataset) -> Dict[str, float]:
        # reuse your UNet metrics style (pixel + region)
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

                out = self.model(pixel_values=images, return_dict=True)
                logits = out.logits  # [B,C,h,w] (often lower-res)

                # Upsample to mask resolution
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                preds = torch.argmax(logits, dim=1)

                pixel_accs.append((preds == masks).float().mean().item())

                # binary-style region metrics: skip background
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
            "pixel_acc": float(np.mean(pixel_accs)) if pixel_accs else 0.0,
        }

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            out = self.model(pixel_values=batch, return_dict=True)
            logits = out.logits
            # upsample to input resolution
            logits = F.interpolate(logits, size=batch.shape[-2:], mode="bilinear", align_corners=False)
            return torch.argmax(logits, dim=1).cpu()

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Mean pixel entropy as uncertainty. Returns [N].
        """
        self.model.eval()
        scores = []
        with torch.no_grad():
            for img in images:
                x = _ensure_rgb(img).unsqueeze(0).to(self.device)
                out = self.model(pixel_values=x, return_dict=True)
                logits = out.logits
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
                probs = F.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                scores.append(float(ent.item()))
        return np.array(scores, dtype=np.float32)

    def save(self, path: str):
        # Save HF weights + optimizer
        torch.save(
            {"optimizer": self.optimizer.state_dict(), "num_classes": self.num_classes},
            path + ".pt",
        )
        self.model.save_pretrained(path)

    def load(self, path: str):
        from transformers import SegformerForSemanticSegmentation
        self.model = SegformerForSemanticSegmentation.from_pretrained(path).to(self.device)
        ckpt = torch.load(path + ".pt", map_location=self.device)
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
    def __init__(self, state_dim, hidden_dim, num_budget_options):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.image_head = nn.Linear(hidden_dim, 1)
        self.budget_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, global_state=None):
        h = self.encoder(x)
        image_logits = self.image_head(h).squeeze(-1)  # [N]

        budget_logit = None
        if global_state is not None:
            g = self.encoder(global_state.unsqueeze(0))          # [1, H]
            budget_logit = self.budget_head(g).squeeze()         # scalar

        return image_logits, budget_logit

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
    

class SegFormerWithBottleneck(nn.Module):
    """
    Wrap HuggingFace SegFormer to expose get_bottleneck_features()
    so RL code can call:

        self.oracle_model.model.get_bottleneck_features(images)
    """

    def __init__(self, segformer_model):
        super().__init__()
        self.segformer = segformer_model

    def forward(self, pixel_values, labels=None):
        return self.segformer(pixel_values=pixel_values, labels=labels)

    def get_bottleneck_features(self, x):
        """
        Return global pooled encoder features: [B, hidden_dim]
        """
        outputs = self.segformer.segformer(pixel_values=x, return_dict=True)

        # last_hidden_state → [B, seq_len, hidden_dim]
        feats = outputs.last_hidden_state.mean(dim=1)

        return feats