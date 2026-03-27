import os

import numpy as np
from typing import List, Dict, Optional, Union

import torch
from ultralytics import YOLO


from .utils import _ensure_rgb, _batched

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
            data_folder = getattr(self.config, "data_dir", None)
            if data_folder is None:
                raise ValueError("config must include `yolo_data_yaml` (path to YOLO data.yaml) or `yolo_data_folder` (containing data.yaml).")
            else:
                data_yaml = os.path.join(data_folder, "data.yaml")
                
        return self.fit(
            data_yaml=str(data_yaml),
            imgsz=getattr(self.config, "img_size", 640),
            epochs=1,
            batch=getattr(self.config, "batch_size", 8),
            workers=getattr(self.config, "num_workers", 2),
            device=0 if self._device_str == "cuda" else "cpu",
            project=getattr(self.config, "yolo_project", None),
            name=getattr(self.config, "yolo_name", None),
            resume=False if epoch > 1 else False,
            verbose=True,
        )

    def evaluate(self, dataset=None) -> Dict[str, float]:
        """
        Runs Ultralytics validation (requires same data.yaml).
        Returns mAP metrics if available.
        """
        data_yaml = getattr(self.config, "yolo_data_yaml", None)
        if data_yaml is None:
            data_folder = getattr(self.config, "data_dir", None)
            if data_folder is None:
                raise ValueError("config must include `yolo_data_yaml` (path to YOLO data.yaml) or `yolo_data_folder` (containing data.yaml).")
            else:
                data_yaml = os.path.join(data_folder, "data.yaml")

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
        self.model = YOLO(path)
