"""
Sewer-ML multi-label classification dataset.

Directory structure expected:
    root/
        SewerML_train.csv
        SewerML_valid.csv
        (images referenced by Filename column in CSV)

CSV format:
    Filename, RB, OB, PF, DE, FS, IS, RO, IN, AF, BE, FO, GR, PH, PB, OS, OP, OK
    (17 binary label columns: 1 = defect present)

Returns: (image, label_vector)
    image        : Tensor [3, H, W]  – normalized RGB
    label_vector : Tensor [17]       – float32 multi-hot
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Official Sewer-ML defect class names in column order
SEWERML_CLASSES = [
    "RB", "OB", "PF", "DE", "FS", "IS",
    "RO", "IN", "AF", "BE", "FO", "GR",
    "PH", "PB", "OS", "OP", "OK",
]
NUM_CLASSES = len(SEWERML_CLASSES)  # 17

# Dataset-specific normalisation (provided by Sewer-ML authors)
SEWERML_MEAN = [0.523, 0.453, 0.345]
SEWERML_STD  = [0.210, 0.199, 0.154]


class SewerMLDataset(Dataset):
    """
    Multi-label image classification dataset for Sewer-ML.

    Parameters
    ----------
    root_dir : str
        Directory that contains SewerML_train.csv / SewerML_valid.csv
        and the image files.
    split : str
        One of 'train' or 'val'.
    img_size : int
        Square resize target (default 224).
    augment : bool
        Apply random horizontal flip + colour jitter during training.
    """

    _CSV_MAP = {
        "train": "SewerML_train.csv",
        "val":   "SewerML_valid.csv",
        # allow 'test' as alias for val if no separate test CSV exists
        "test":  "SewerML_valid.csv",
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = True,
    ):
        assert split in self._CSV_MAP, (
            f"split must be one of {list(self._CSV_MAP)}, got '{split}'"
        )

        self.root_dir = root_dir
        self.split    = split
        self.img_size = img_size

        # ------------------------------------------------------------------
        # Load CSV
        # ------------------------------------------------------------------
        csv_path = os.path.join(root_dir, self._CSV_MAP[split])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"SewerML CSV not found: {csv_path}\n"
                f"Expected one of: {[os.path.join(root_dir, v) for v in self._CSV_MAP.values()]}"
            )

        df = pd.read_csv(csv_path)

        # Validate label columns exist
        missing = [c for c in SEWERML_CLASSES if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing label columns: {missing}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        self.filenames = df["Filename"].tolist()
        self.labels    = torch.tensor(
            df[SEWERML_CLASSES].values, dtype=torch.float32
        )  # [N, 17]

        # ------------------------------------------------------------------
        # Class weights for BCEWithLogitsLoss pos_weight
        # Shape [17], weight_c = (N - pos_c) / pos_c
        # ------------------------------------------------------------------
        pos = self.labels.sum(dim=0).clamp(min=1)
        neg = len(self.labels) - pos
        self.class_weights = (neg / pos).clamp(max=50.0)

        # ------------------------------------------------------------------
        # Transforms
        # ------------------------------------------------------------------
        train_tf = [
            transforms.Resize((img_size, img_size)),
        ]
        if augment and split == "train":
            train_tf += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1,
                    saturation=0.1, hue=0.05
                ),
            ]
        train_tf += [
            transforms.ToTensor(),
            transforms.Normalize(mean=SEWERML_MEAN, std=SEWERML_STD),
        ]
        self.transform = transforms.Compose(train_tf)

        print(
            f"[SewerML] {len(self.filenames)} samples loaded "
            f"for split='{split}' | classes={NUM_CLASSES}"
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.root_dir, fname)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)          # [3, H, W]
        label = self.labels[idx]               # [17]  float32

        return image, label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def num_classes(self):
        return NUM_CLASSES

    @property
    def class_names(self):
        return SEWERML_CLASSES