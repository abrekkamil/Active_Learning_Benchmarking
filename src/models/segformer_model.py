import time

import numpy as np
from tqdm import tqdm
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
)


from .utils import _ensure_rgb, _batched



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
        outputs = self.segformer.segformer(pixel_values=x)

        # last_hidden_state → [B, seq_len, hidden_dim]
        feats = outputs.last_hidden_state.mean(dim=[2, 3])  # global average pool over spatial dimensions

        return feats
    

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
    def get_bottleneck_features(self, images):

        self.model.eval()
        with torch.no_grad():

            if isinstance(images, list):
                batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            else:
                batch = images.to(self.device)

            enc = self.model.segformer(pixel_values=batch)

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

            out = self.model(pixel_values=images, labels=masks)
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

                out = self.model(pixel_values=images)
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
    def forward_model(self, images):
        try:
            # HuggingFace-style
            return self.model(pixel_values=images)
        except TypeError:
            # Torchvision-style
            return self.model(images)
        
    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch = _batched([_ensure_rgb(i) for i in images]).to(self.device)
            out = self.model(pixel_values=batch)
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
                out = self.model(pixel_values=x)
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
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(path).to(self.device)
        ckpt = torch.load(path + ".pt", map_location=self.device)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

