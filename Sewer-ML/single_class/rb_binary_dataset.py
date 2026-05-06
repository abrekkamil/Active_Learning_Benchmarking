"""
Binary dataset wrapper for RB (Root Block) classification.

Wraps the existing MultiLabelDataset and exposes:
    batch['image']  -> image tensor (unchanged)
    batch['labels'] -> scalar tensor, 1.0 if RB positive else 0.0
    batch['labels_full'] -> original 17-dim vector (kept for diagnostics)

Sewer-ML label order (from MultiLabelDataset.LabelNames):
    ['VA','RB','OB','PF','DE','FS','IS','RO','IN','AF',
     'BE','FO','GR','PH','PB','OS','OP','OK']
RB is at index 1.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# Sewer-ML class index map (matches MultiLabelDataset.LabelNames)
SEWER_LABEL_ORDER = [
    "VA", "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN",
    "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(SEWER_LABEL_ORDER)}


class RBBinaryDataset(Dataset):
    """
    Wraps a multi-label dataset and exposes a binary RB label.

    Args:
        base_dataset  : an instance of MultiLabelDataset
        target_class  : class code to treat as positive (default 'RB')
    """

    def __init__(self, base_dataset, target_class="RB"):
        self.base         = base_dataset
        self.target_class = target_class
        if target_class not in CLASS_TO_IDX:
            raise ValueError(
                "Unknown target class '{}'. Must be one of {}".format(
                    target_class, list(CLASS_TO_IDX.keys())
                )
            )
        self.target_idx = CLASS_TO_IDX[target_class]

        # Expose filenames for CLIP-IQA and logging
        if hasattr(base_dataset, "imgPaths"):
            self.imgPaths = base_dataset.imgPaths

        # Pre-compute binary labels once, for fast positive/negative splits
        # This avoids iterating the whole loader every cold start.
        self._binary_labels = None
        self._try_build_label_cache()

    def _try_build_label_cache(self):
        """Try to read labels directly from the underlying dataset without
        running the transform pipeline. Falls back silently if unavailable."""
        for attr in ("labels", "label_matrix", "targets"):
            if hasattr(self.base, attr):
                raw = getattr(self.base, attr)
                if isinstance(raw, torch.Tensor):
                    raw = raw.cpu().numpy()
                raw = np.asarray(raw)
                if raw.ndim == 2 and raw.shape[1] > self.target_idx:
                    self._binary_labels = (raw[:, self.target_idx] > 0).astype(np.int64)
                    return

    @property
    def binary_labels(self):
        """Returns np.ndarray of 0/1 labels, one per sample. None if not cached."""
        return self._binary_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        full_labels = item["labels"]
        if isinstance(full_labels, torch.Tensor):
            rb = float(full_labels[self.target_idx].item() > 0)
        else:
            rb = float(full_labels[self.target_idx] > 0)

        out = {
            "image":       item["image"],
            "labels":      torch.tensor(rb, dtype=torch.float32),
            "labels_full": item["labels"],   # keep for diagnostics
        }
        # Pass through any optional keys the base dataset exposes
        for k in ("filename", "image_path", "idx"):
            if k in item:
                out[k] = item[k]
        return out

    # Compatibility shim: some code paths read LabelNames
    @property
    def LabelNames(self):
        return [self.target_class]
