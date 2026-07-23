from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class DomainAdaptationConfig:
    experiment_name: str = "da_segformer_supervisely_views_5pct_seed42"

    # Data paths
    supervisely_manifest: str = (
        "/nobackup/projects/bddur59/Datasets/"
        "supervisely_synthetic_cracks/converted/"
        "supervisely_geometry_manifest.csv"
    )
    roboflow_source_root: str = (
        "/nobackup/projects/bddur59/Datasets/"
        "crack_synthetic/Crack_Synthetic_Semantic"
    )
    target_root: str = (
        "/nobackup/projects/bddur59/Datasets/crackseg9k_disk"
    )

    # none | roboflow | supervisely_raw | supervisely_views
    source_condition: str = "supervisely_views"

    # target_only | joint
    adaptation_mode: str = "target_only"
    source_loss_weight: float = 1.0
    target_loss_weight: float = 1.0
    joint_adaptation_steps: int = 5000
    joint_eval_every_steps: int = 500

    # Model
    model_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    img_size: int = 256
    num_classes: int = 2

    # Optimisation
    source_steps: int = 5000
    source_eval_every_steps: int = 500
    adaptation_epochs: int = 15
    batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 6e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    amp: bool = True

    # Splits and target budget
    source_val_fraction: float = 0.10
    target_val_fraction: float = 0.10
    initial_target_fraction: float = 0.05

    # Reproducibility/runtime
    seed: int = 42
    num_workers: int = 4
    use_cuda: bool = True

    # Output
    output_dir: str = "results/domain_adaptation"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "DomainAdaptationConfig":
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}

        if not isinstance(values, dict):
            raise TypeError("The YAML root must be a mapping/dictionary.")

        valid_fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - valid_fields)
        if unknown:
            raise ValueError(
                "Unknown domain-adaptation configuration keys: "
                f"{unknown}. Fix the YAML instead of silently ignoring them."
            )

        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        valid_sources = {
            "none",
            "roboflow",
            "supervisely_raw",
            "supervisely_views",
        }
        if self.source_condition not in valid_sources:
            raise ValueError(
                f"source_condition must be one of {sorted(valid_sources)}, "
                f"got {self.source_condition!r}."
            )

        valid_adaptation_modes = {
            "target_only",
            "joint",
        }

        if self.adaptation_mode not in valid_adaptation_modes:
            raise ValueError(
                "adaptation_mode must be one of "
                f"{sorted(valid_adaptation_modes)}, "
                f"got {self.adaptation_mode!r}."
            )

        if self.adaptation_mode == "joint" and self.source_condition == "none":
            raise ValueError(
                "Joint adaptation requires a synthetic source dataset. "
                "source_condition cannot be 'none'."
            )

        if self.source_loss_weight < 0:
            raise ValueError("source_loss_weight cannot be negative.")

        if self.target_loss_weight <= 0:
            raise ValueError("target_loss_weight must be positive.")

        if self.num_classes != 2:
            raise ValueError("This first implementation expects binary masks and num_classes=2.")

        for name in ("source_val_fraction", "target_val_fraction"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")

        if not 0.0 < self.initial_target_fraction <= 1.0:
            raise ValueError(
                "initial_target_fraction must be in (0, 1], got "
                f"{self.initial_target_fraction}."
            )

        if self.source_steps < 0:
            raise ValueError("source_steps cannot be negative.")
        if self.adaptation_epochs <= 0:
            raise ValueError("adaptation_epochs must be positive.")
        if self.joint_adaptation_steps <= 0:
            raise ValueError("joint_adaptation_steps must be positive.")
        if self.joint_eval_every_steps <= 0:
            raise ValueError("joint_eval_every_steps must be positive.")
        if self.joint_eval_every_steps > self.joint_adaptation_steps:
            raise ValueError(
                "joint_eval_every_steps cannot exceed "
                "joint_adaptation_steps."
            )
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")
        if self.img_size <= 0:
            raise ValueError("img_size must be positive.")


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("domain_adaptation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred for the controlled pilot.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fraction_tag(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def resolve_device(config: DomainAdaptationConfig) -> torch.device:
    if config.use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Pair collection and transforms
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageMaskPair:
    sample_id: str
    image_path: str
    mask_path: str


def collect_image_mask_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
) -> List[ImageMaskPair]:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_root}")

    masks_by_stem: Dict[str, Path] = {}
    for path in mask_root.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in masks_by_stem:
            raise RuntimeError(
                f"Duplicate mask stem {path.stem!r} in {mask_root}."
            )
        masks_by_stem[path.stem] = path

    pairs: List[ImageMaskPair] = []
    missing_masks: List[str] = []

    for image_path in sorted(image_root.iterdir(), key=lambda item: item.name):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        pairs.append(
            ImageMaskPair(
                sample_id=image_path.stem,
                image_path=str(image_path.resolve()),
                mask_path=str(mask_path.resolve()),
            )
        )

    if missing_masks:
        preview = missing_masks[:10]
        raise RuntimeError(
            f"Found {len(missing_masks)} images without matching masks in "
            f"{image_root}. Examples: {preview}"
        )

    if not pairs:
        raise RuntimeError(
            f"No image-mask pairs were found in {image_root} and {mask_root}."
        )

    sample_ids = [pair.sample_id for pair in pairs]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"Duplicate sample stems found in {image_root}.")

    return pairs


def apply_joint_transform(
    image: Image.Image,
    mask: Image.Image,
    img_size: int,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = image.resize((img_size, img_size), resample=Image.BILINEAR)
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)

    if training:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        if random.random() < 0.5:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)

        number_of_rotations = random.randint(0, 3)
        if number_of_rotations:
            angle = 90 * number_of_rotations
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
            mask = mask.rotate(angle, resample=Image.NEAREST, expand=False)

        # Appearance transforms are applied to the image only.
        if random.random() < 0.5:
            brightness = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(brightness)

        if random.random() < 0.5:
            contrast = random.uniform(0.85, 1.15)
            image = ImageEnhance.Contrast(image).enhance(contrast)

    image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise RuntimeError(f"Expected RGB image, got shape {image_array.shape}.")

    mask_array = np.array(mask, dtype=np.uint8, copy=True)
    binary_mask = (mask_array > 0).astype(np.int64)

    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
    mask_tensor = torch.from_numpy(binary_mask).long()

    return image_tensor, mask_tensor


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


class PairedMaskDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[ImageMaskPair],
        img_size: int,
        training: bool,
    ) -> None:
        if not pairs:
            raise ValueError("PairedMaskDataset received no samples.")
        self.pairs = list(pairs)
        self.img_size = int(img_size)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        with Image.open(pair.image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(pair.mask_path) as mask_file:
            mask = mask_file.convert("L")

        return apply_joint_transform(
            image=image,
            mask=mask,
            img_size=self.img_size,
            training=self.training,
        )


@dataclass(frozen=True)
class GeometryView:
    geometry_id: str
    appearance: str
    image_path: str
    mask_path: str


def load_supervisely_manifest(path: str | Path) -> List[GeometryView]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Supervisely manifest not found: {manifest_path}")

    rows: List[GeometryView] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"geometry_id", "appearance", "image_path", "mask_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Manifest image not found: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Manifest mask not found: {mask_path}")

            rows.append(
                GeometryView(
                    geometry_id=row["geometry_id"],
                    appearance=row["appearance"].strip().lower(),
                    image_path=str(image_path.resolve()),
                    mask_path=str(mask_path.resolve()),
                )
            )

    if not rows:
        raise RuntimeError(f"Manifest contains no rows: {manifest_path}")
    return rows


class SuperviselyGeometryDataset(Dataset):
    """One item per distinct mask geometry.

    For training with ``random_views=True``, an available raw or styled image is
    selected deterministically from ``seed + epoch + index``. Validation uses a
    stable raw view when available.
    """

    def __init__(
        self,
        rows: Sequence[GeometryView],
        geometry_ids: Sequence[str],
        img_size: int,
        training: bool,
        random_views: bool,
        seed: int,
    ) -> None:
        rows_by_geometry: Dict[str, List[GeometryView]] = defaultdict(list)
        for row in rows:
            rows_by_geometry[row.geometry_id].append(row)

        self.geometry_ids = sorted(set(geometry_ids))
        missing = [gid for gid in self.geometry_ids if gid not in rows_by_geometry]
        if missing:
            raise RuntimeError(
                f"{len(missing)} requested geometry IDs are missing from the manifest."
            )

        self.rows_by_geometry = rows_by_geometry
        self.img_size = int(img_size)
        self.training = bool(training)
        self.random_views = bool(random_views)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.geometry_ids)

    def _select_view(self, index: int) -> GeometryView:
        geometry_id = self.geometry_ids[index]
        choices = self.rows_by_geometry[geometry_id]

        if self.training and self.random_views and len(choices) > 1:
            local_seed = self.seed + self.epoch * 1_000_003 + index * 9_973
            local_rng = random.Random(local_seed)
            return choices[local_rng.randrange(len(choices))]

        raw_choices = [row for row in choices if row.appearance == "raw"]
        if raw_choices:
            return sorted(raw_choices, key=lambda row: row.image_path)[0]
        return sorted(choices, key=lambda row: row.image_path)[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._select_view(index)
        with Image.open(row.image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(row.mask_path) as mask_file:
            mask = mask_file.convert("L")

        return apply_joint_transform(
            image=image,
            mask=mask,
            img_size=self.img_size,
            training=self.training,
        )


# -----------------------------------------------------------------------------
# Reproducible source and target splits
# -----------------------------------------------------------------------------


def deterministic_split_ids(
    identifiers: Sequence[str],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    unique_ids = sorted(set(identifiers))
    if len(unique_ids) < 2:
        raise ValueError("At least two unique identifiers are required for a split.")

    rng = random.Random(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)

    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)

    validation_ids = sorted(shuffled[:validation_count])
    train_ids = sorted(shuffled[validation_count:])
    return train_ids, validation_ids


def load_or_create_target_split(
    target_train_pairs: Sequence[ImageMaskPair],
    validation_fraction: float,
    seed: int,
    split_dir: Path,
) -> Tuple[List[ImageMaskPair], List[ImageMaskPair]]:
    split_path = split_dir / (
        f"target_pool_val_seed{seed}_{fraction_tag(validation_fraction)}.json"
    )
    pairs_by_id = {pair.sample_id: pair for pair in target_train_pairs}

    if split_path.exists():
        split = read_json(split_path)
        train_ids = split["target_pool_ids"]
        validation_ids = split["target_validation_ids"]
    else:
        train_ids, validation_ids = deterministic_split_ids(
            identifiers=list(pairs_by_id),
            validation_fraction=validation_fraction,
            seed=seed,
        )
        write_json(
            split_path,
            {
                "seed": seed,
                "validation_fraction": validation_fraction,
                "target_pool_ids": train_ids,
                "target_validation_ids": validation_ids,
            },
        )

    unknown = (set(train_ids) | set(validation_ids)) - set(pairs_by_id)
    if unknown:
        raise RuntimeError(
            "Saved target split refers to samples that no longer exist. "
            f"Examples: {sorted(unknown)[:10]}. Remove {split_path} only if you "
            "intentionally changed the dataset."
        )

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Target pool and target validation split overlap.")

    if set(train_ids) | set(validation_ids) != set(pairs_by_id):
        raise RuntimeError("Saved target split does not cover the complete target train set.")

    return (
        [pairs_by_id[sample_id] for sample_id in train_ids],
        [pairs_by_id[sample_id] for sample_id in validation_ids],
    )


def load_or_create_initial_target_selection(
    target_pool_pairs: Sequence[ImageMaskPair],
    target_fraction: float,
    seed: int,
    split_dir: Path,
) -> List[int]:
    selection_path = split_dir / (
        f"target_initial_seed{seed}_{fraction_tag(target_fraction)}.json"
    )
    pairs_by_id = {pair.sample_id: pair for pair in target_pool_pairs}

    if selection_path.exists():
        payload = read_json(selection_path)
        selected_ids = payload["selected_target_ids"]
    else:
        identifiers = sorted(pairs_by_id)
        selection_count = max(1, round(len(identifiers) * target_fraction))
        selection_count = min(selection_count, len(identifiers))

        rng = random.Random(seed)
        selected_ids = sorted(rng.sample(identifiers, selection_count))
        write_json(
            selection_path,
            {
                "seed": seed,
                "target_fraction": target_fraction,
                "original_target_pool_size": len(identifiers),
                "selected_count": selection_count,
                "selected_target_ids": selected_ids,
            },
        )

    missing = set(selected_ids) - set(pairs_by_id)
    if missing:
        raise RuntimeError(
            "Saved target selection contains missing samples. Examples: "
            f"{sorted(missing)[:10]}"
        )

    index_by_id = {
        pair.sample_id: index for index, pair in enumerate(target_pool_pairs)
    }
    return [index_by_id[sample_id] for sample_id in selected_ids]


def load_or_create_supervisely_geometry_split(
    rows: Sequence[GeometryView],
    validation_fraction: float,
    seed: int,
    split_dir: Path,
) -> Tuple[List[str], List[str]]:
    split_path = split_dir / (
        f"supervisely_geometry_train_val_seed{seed}_"
        f"{fraction_tag(validation_fraction)}.json"
    )
    all_geometry_ids = sorted({row.geometry_id for row in rows})

    if split_path.exists():
        payload = read_json(split_path)
        train_ids = payload["train_geometry_ids"]
        validation_ids = payload["validation_geometry_ids"]
    else:
        train_ids, validation_ids = deterministic_split_ids(
            identifiers=all_geometry_ids,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        write_json(
            split_path,
            {
                "seed": seed,
                "validation_fraction": validation_fraction,
                "train_geometry_ids": train_ids,
                "validation_geometry_ids": validation_ids,
            },
        )

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Supervisely source train and validation geometries overlap.")
    if set(train_ids) | set(validation_ids) != set(all_geometry_ids):
        raise RuntimeError("Saved Supervisely geometry split is inconsistent with the manifest.")

    return train_ids, validation_ids


# -----------------------------------------------------------------------------
# Dataset builders
# -----------------------------------------------------------------------------


def build_source_datasets(
    config: DomainAdaptationConfig,
    split_dir: Path,
) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    if config.source_condition == "none":
        return None, None

    if config.source_condition == "roboflow":
        root = Path(config.roboflow_source_root)
        train_pairs = collect_image_mask_pairs(
            root / "train" / "images",
            root / "train" / "masks",
        )

        validation_folder = root / "val"
        if not validation_folder.exists():
            validation_folder = root / "valid"

        validation_pairs = collect_image_mask_pairs(
            validation_folder / "images",
            validation_folder / "masks",
        )

        return (
            PairedMaskDataset(train_pairs, config.img_size, training=True),
            PairedMaskDataset(validation_pairs, config.img_size, training=False),
        )

    rows = load_supervisely_manifest(config.supervisely_manifest)
    train_geometry_ids, validation_geometry_ids = (
        load_or_create_supervisely_geometry_split(
            rows=rows,
            validation_fraction=config.source_val_fraction,
            seed=config.seed,
            split_dir=split_dir,
        )
    )

    if config.source_condition == "supervisely_raw":
        rows = [row for row in rows if row.appearance == "raw"]
        random_views = False
    elif config.source_condition == "supervisely_views":
        random_views = True
    else:
        raise AssertionError(f"Unhandled source condition: {config.source_condition}")

    source_train = SuperviselyGeometryDataset(
        rows=rows,
        geometry_ids=train_geometry_ids,
        img_size=config.img_size,
        training=True,
        random_views=random_views,
        seed=config.seed,
    )
    source_validation = SuperviselyGeometryDataset(
        rows=rows,
        geometry_ids=validation_geometry_ids,
        img_size=config.img_size,
        training=False,
        random_views=False,
        seed=config.seed,
    )
    return source_train, source_validation


def build_target_datasets(
    config: DomainAdaptationConfig,
    split_dir: Path,
) -> Tuple[
    PairedMaskDataset,
    PairedMaskDataset,
    PairedMaskDataset,
    List[ImageMaskPair],
]:
    target_root = Path(config.target_root)

    target_train_pairs = collect_image_mask_pairs(
        target_root / "train" / "images",
        target_root / "train" / "masks",
    )
    target_test_pairs = collect_image_mask_pairs(
        target_root / "test" / "images",
        target_root / "test" / "masks",
    )

    target_pool_pairs, target_validation_pairs = load_or_create_target_split(
        target_train_pairs=target_train_pairs,
        validation_fraction=config.target_val_fraction,
        seed=config.seed,
        split_dir=split_dir,
    )

    target_pool_dataset = PairedMaskDataset(
        target_pool_pairs,
        config.img_size,
        training=True,
    )
    target_validation_dataset = PairedMaskDataset(
        target_validation_pairs,
        config.img_size,
        training=False,
    )
    target_test_dataset = PairedMaskDataset(
        target_test_pairs,
        config.img_size,
        training=False,
    )

    return (
        target_pool_dataset,
        target_validation_dataset,
        target_test_dataset,
        target_pool_pairs,
    )


# -----------------------------------------------------------------------------
# Model, loaders, checkpoints
# -----------------------------------------------------------------------------


def build_model(
    config: DomainAdaptationConfig,
    device: torch.device,
) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.model_checkpoint,
        num_labels=config.num_classes,
        id2label={0: "background", 1: "crack"},
        label2id={"background": 0, "crack": 1},
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    config: DomainAdaptationConfig,
    seed_offset: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.use_cuda and torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    stage: str,
    step_or_epoch: int,
    metrics: Mapping[str, Any],
    config: DomainAdaptationConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "step_or_epoch": step_or_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metrics": dict(metrics),
        "config": asdict(config),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


# -----------------------------------------------------------------------------
# Metrics and evaluation
# -----------------------------------------------------------------------------


@dataclass
class ConfusionAccumulator:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    image_count: int = 0
    predicted_positive_pixels: int = 0
    target_positive_pixels: int = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = predictions.bool()
        targets = targets.bool()

        self.true_positive += int((predictions & targets).sum().item())
        self.false_positive += int((predictions & ~targets).sum().item())
        self.false_negative += int((~predictions & targets).sum().item())
        self.true_negative += int((~predictions & ~targets).sum().item())
        self.image_count += int(predictions.shape[0])
        self.predicted_positive_pixels += int(predictions.sum().item())
        self.target_positive_pixels += int(targets.sum().item())

    def compute(self) -> Dict[str, float | int]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        false_positive_pixel_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "images": self.image_count,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "dice": float(f1),
            "iou": float(iou),
            "accuracy": float(accuracy),
            "false_positive_pixel_rate": float(false_positive_pixel_rate),
            "predicted_positive_pixels": self.predicted_positive_pixels,
            "target_positive_pixels": self.target_positive_pixels,
        }


def evaluate_model(
    model: torch.nn.Module,
    dataset: Dataset,
    config: DomainAdaptationConfig,
    device: torch.device,
    description: str,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    loader = make_loader(
        dataset=dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        config=config,
        seed_offset=10_000,
        drop_last=False,
    )

    accumulators = {
        "all": ConfusionAccumulator(),
        "positive_images": ConfusionAccumulator(),
        "empty_images": ConfusionAccumulator(),
    }

    per_image_ious: List[float] = []
    per_image_dices: List[float] = []

    model.eval()
    with torch.no_grad():
        progress = tqdm(loader, desc=description, leave=False)
        for batch_index, (images, masks) in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(pixel_values=images)
            logits = F.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            predictions = torch.argmax(logits, dim=1)

            accumulators["all"].update(predictions, masks)

            for image_index in range(masks.shape[0]):
                prediction = predictions[image_index : image_index + 1]
                target = masks[image_index : image_index + 1]
                target_is_positive = bool(target.any().item())

                key = "positive_images" if target_is_positive else "empty_images"
                accumulators[key].update(prediction, target)

                intersection = int(((prediction == 1) & (target == 1)).sum().item())
                union = int(((prediction == 1) | (target == 1)).sum().item())
                denominator = int((prediction == 1).sum().item() + (target == 1).sum().item())

                if union > 0:
                    per_image_ious.append(intersection / union)
                if denominator > 0:
                    per_image_dices.append(2 * intersection / denominator)

    metrics: Dict[str, Any] = {
        name: accumulator.compute() for name, accumulator in accumulators.items()
    }
    metrics["mean_image_iou_nonempty_union"] = (
        float(np.mean(per_image_ious)) if per_image_ious else 0.0
    )
    metrics["mean_image_dice_nonempty_denominator"] = (
        float(np.mean(per_image_dices)) if per_image_dices else 0.0
    )
    return metrics


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def compute_segmentation_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> torch.Tensor:
    images = images.to(
        device,
        non_blocking=True,
    )

    masks = masks.to(
        device,
        non_blocking=True,
    ).long()

    if images.ndim != 4 or images.shape[1] != 3:
        raise RuntimeError(
            "Expected images [B,3,H,W], got "
            f"{tuple(images.shape)}"
        )

    if masks.ndim != 3:
        raise RuntimeError(
            "Expected masks [B,H,W], got "
            f"{tuple(masks.shape)}"
        )

    unique_values = torch.unique(masks)

    if not set(
        unique_values.detach().cpu().tolist()
    ).issubset({0, 1}):
        raise RuntimeError(
            "Masks contain unexpected labels: "
            f"{unique_values.tolist()}"
        )

    with torch.cuda.amp.autocast(
        enabled=amp_enabled
    ):
        outputs = model(
            pixel_values=images,
            labels=masks,
        )

        loss = outputs.loss

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss encountered: {loss.item()}"
        )

    return loss

def train_one_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    max_grad_norm: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)

    loss = compute_segmentation_loss(
        model=model,
        images=images,
        masks=masks,
        device=device,
        amp_enabled=amp_enabled,
    )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    return float(loss.detach().item())


def pretrain_on_source(
    model: torch.nn.Module,
    source_train: Dataset,
    source_validation: Dataset,
    config: DomainAdaptationConfig,
    device: torch.device,
    experiment_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    if config.source_steps <= 0:
        logger.info("source_steps=0; skipping source pretraining.")
        return {"skipped": True}

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    loader = make_loader(
        dataset=source_train,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,
        seed_offset=100,
        drop_last=len(source_train) >= config.batch_size,
    )

    best_f1 = -math.inf
    best_metrics: Dict[str, Any] = {}
    running_loss = 0.0
    iterator: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None
    source_epoch = 0
    started = time.time()

    progress = tqdm(range(1, config.source_steps + 1), desc="Source pretraining")
    for step in progress:
        if iterator is None:
            if hasattr(source_train, "set_epoch"):
                source_train.set_epoch(source_epoch)  # type: ignore[attr-defined]
            iterator = iter(loader)

        try:
            images, masks = next(iterator)  # type: ignore[arg-type]
        except StopIteration:
            source_epoch += 1
            if hasattr(source_train, "set_epoch"):
                source_train.set_epoch(source_epoch)  # type: ignore[attr-defined]
            iterator = iter(loader)
            images, masks = next(iterator)

        model.train()
        loss = train_one_batch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            images=images,
            masks=masks,
            device=device,
            amp_enabled=amp_enabled,
            max_grad_norm=config.max_grad_norm,
        )
        running_loss += loss
        progress.set_postfix(loss=f"{loss:.4f}")

        should_evaluate = (
            step % config.source_eval_every_steps == 0
            or step == config.source_steps
        )
        if not should_evaluate:
            continue

        mean_loss = running_loss / max(config.source_eval_every_steps, 1)
        running_loss = 0.0

        validation_metrics = evaluate_model(
            model=model,
            dataset=source_validation,
            config=config,
            device=device,
            description=f"Source validation @ step {step}",
        )
        validation_f1 = float(validation_metrics["all"]["f1"])

        logger.info(
            "Source step %d/%d | loss %.6f | val F1 %.6f | val IoU %.6f",
            step,
            config.source_steps,
            mean_loss,
            validation_f1,
            float(validation_metrics["all"]["iou"]),
        )

        save_checkpoint(
            experiment_dir / "source_last.pt",
            model=model,
            optimizer=optimizer,
            stage="source",
            step_or_epoch=step,
            metrics=validation_metrics,
            config=config,
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_metrics = validation_metrics
            save_checkpoint(
                experiment_dir / "source_best.pt",
                model=model,
                optimizer=optimizer,
                stage="source",
                step_or_epoch=step,
                metrics=validation_metrics,
                config=config,
            )

    load_model_checkpoint(experiment_dir / "source_best.pt", model, device)

    return {
        "best_source_validation": best_metrics,
        "best_source_validation_f1": best_f1,
        "elapsed_seconds": time.time() - started,
    }


def adapt_on_target(
    model: torch.nn.Module,
    target_labelled_dataset: Dataset,
    target_validation_dataset: Dataset,
    config: DomainAdaptationConfig,
    device: torch.device,
    experiment_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    # A new optimiser is deliberate: source optimiser state is not retained.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    loader = make_loader(
        dataset=target_labelled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,
        seed_offset=200,
        drop_last=len(target_labelled_dataset) >= config.batch_size,
    )

    best_f1 = -math.inf
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    started = time.time()

    for epoch in range(1, config.adaptation_epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Target adaptation {epoch}/{config.adaptation_epochs}", leave=False)

        for images, masks in progress:
            loss = train_one_batch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                images=images,
                masks=masks,
                device=device,
                amp_enabled=amp_enabled,
                max_grad_norm=config.max_grad_norm,
            )
            total_loss += loss
            progress.set_postfix(loss=f"{loss:.4f}")

        mean_loss = total_loss / max(len(loader), 1)
        validation_metrics = evaluate_model(
            model=model,
            dataset=target_validation_dataset,
            config=config,
            device=device,
            description=f"Target validation @ epoch {epoch}",
        )
        validation_f1 = float(validation_metrics["all"]["f1"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "validation": validation_metrics,
        }
        history.append(epoch_record)

        logger.info(
            "Target epoch %d/%d | loss %.6f | val F1 %.6f | val IoU %.6f | "
            "precision %.6f | recall %.6f",
            epoch,
            config.adaptation_epochs,
            mean_loss,
            validation_f1,
            float(validation_metrics["all"]["iou"]),
            float(validation_metrics["all"]["precision"]),
            float(validation_metrics["all"]["recall"]),
        )

        save_checkpoint(
            experiment_dir / "target_last.pt",
            model=model,
            optimizer=optimizer,
            stage="target",
            step_or_epoch=epoch,
            metrics=validation_metrics,
            config=config,
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_metrics = validation_metrics
            save_checkpoint(
                experiment_dir / "target_best.pt",
                model=model,
                optimizer=optimizer,
                stage="target",
                step_or_epoch=epoch,
                metrics=validation_metrics,
                config=config,
            )

    load_model_checkpoint(experiment_dir / "target_best.pt", model, device)

    return {
        "best_target_validation": best_metrics,
        "best_target_validation_f1": best_f1,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }

def adapt_joint_source_target(
    model: torch.nn.Module,
    source_train_dataset: Dataset,
    target_labelled_dataset: Dataset,
    target_validation_dataset: Dataset,
    config: DomainAdaptationConfig,
    device: torch.device,
    experiment_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Jointly adapt with fixed source-target optimisation steps.

    Each optimiser step receives one labelled source batch and one labelled
    target batch. Both loaders are cycled independently, so the optimisation
    budget is identical for 5% and 100% target-label conditions.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    source_loader = make_loader(
        dataset=source_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,
        seed_offset=400,
        drop_last=len(source_train_dataset) >= config.batch_size,
    )
    target_loader = make_loader(
        dataset=target_labelled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,
        seed_offset=300,
        drop_last=len(target_labelled_dataset) >= config.batch_size,
    )

    best_f1 = -math.inf
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    started = time.time()

    source_epoch = 0
    target_cycle = 0

    if hasattr(source_train_dataset, "set_epoch"):
        source_train_dataset.set_epoch(source_epoch)  # type: ignore[attr-defined]

    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)

    interval_combined_loss = 0.0
    interval_source_loss = 0.0
    interval_target_loss = 0.0
    interval_steps = 0

    progress = tqdm(
        range(1, config.joint_adaptation_steps + 1),
        desc="Joint adaptation",
    )

    for step in progress:
        try:
            source_images, source_masks = next(source_iterator)
        except StopIteration:
            source_epoch += 1
            if hasattr(source_train_dataset, "set_epoch"):
                source_train_dataset.set_epoch(source_epoch)  # type: ignore[attr-defined]
            source_iterator = iter(source_loader)
            source_images, source_masks = next(source_iterator)

        try:
            target_images, target_masks = next(target_iterator)
        except StopIteration:
            target_cycle += 1
            target_iterator = iter(target_loader)
            target_images, target_masks = next(target_iterator)

        optimizer.zero_grad(set_to_none=True)

        # Backpropagate sequentially to reduce peak GPU memory while preserving
        # the same gradient as the weighted sum of source and target losses.
        source_loss = compute_segmentation_loss(
            model=model,
            images=source_images,
            masks=source_masks,
            device=device,
            amp_enabled=amp_enabled,
        )
        weighted_source_loss = config.source_loss_weight * source_loss
        scaler.scale(weighted_source_loss).backward()

        target_loss = compute_segmentation_loss(
            model=model,
            images=target_images,
            masks=target_masks,
            device=device,
            amp_enabled=amp_enabled,
        )
        weighted_target_loss = config.target_loss_weight * target_loss
        scaler.scale(weighted_target_loss).backward()

        combined_loss = weighted_source_loss.detach() + weighted_target_loss.detach()
        if not torch.isfinite(combined_loss):
            raise FloatingPointError(
                f"Non-finite joint loss encountered: {combined_loss.item()}"
            )

        scaler.unscale_(optimizer)
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        source_value = float(source_loss.detach().item())
        target_value = float(target_loss.detach().item())
        combined_value = float(combined_loss.item())

        interval_source_loss += source_value
        interval_target_loss += target_value
        interval_combined_loss += combined_value
        interval_steps += 1

        progress.set_postfix(
            total=f"{combined_value:.4f}",
            source=f"{source_value:.4f}",
            target=f"{target_value:.4f}",
        )

        should_evaluate = (
            step % config.joint_eval_every_steps == 0
            or step == config.joint_adaptation_steps
        )
        if not should_evaluate:
            continue

        denominator = max(interval_steps, 1)
        mean_combined_loss = interval_combined_loss / denominator
        mean_source_loss = interval_source_loss / denominator
        mean_target_loss = interval_target_loss / denominator

        validation_metrics = evaluate_model(
            model=model,
            dataset=target_validation_dataset,
            config=config,
            device=device,
            description=f"Joint target validation @ step {step}",
        )
        validation_f1 = float(validation_metrics["all"]["f1"])

        record = {
            "step": step,
            "source_epoch": source_epoch,
            "target_cycle": target_cycle,
            "combined_train_loss": mean_combined_loss,
            "source_train_loss": mean_source_loss,
            "target_train_loss": mean_target_loss,
            "validation": validation_metrics,
        }
        history.append(record)

        logger.info(
            "Joint step %d/%d | combined loss %.6f | source loss %.6f | "
            "target loss %.6f | val F1 %.6f | val IoU %.6f | "
            "precision %.6f | recall %.6f",
            step,
            config.joint_adaptation_steps,
            mean_combined_loss,
            mean_source_loss,
            mean_target_loss,
            validation_f1,
            float(validation_metrics["all"]["iou"]),
            float(validation_metrics["all"]["precision"]),
            float(validation_metrics["all"]["recall"]),
        )

        save_checkpoint(
            experiment_dir / "joint_last.pt",
            model=model,
            optimizer=optimizer,
            stage="joint",
            step_or_epoch=step,
            metrics=validation_metrics,
            config=config,
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_metrics = validation_metrics
            save_checkpoint(
                experiment_dir / "joint_best.pt",
                model=model,
                optimizer=optimizer,
                stage="joint",
                step_or_epoch=step,
                metrics=validation_metrics,
                config=config,
            )

        interval_combined_loss = 0.0
        interval_source_loss = 0.0
        interval_target_loss = 0.0
        interval_steps = 0

    if not (experiment_dir / "joint_best.pt").exists():
        raise RuntimeError("Joint adaptation completed without creating a best checkpoint.")

    best_checkpoint = load_model_checkpoint(
        experiment_dir / "joint_best.pt",
        model,
        device,
    )

    return {
        "adaptation_mode": "joint",
        "source_loss_weight": config.source_loss_weight,
        "target_loss_weight": config.target_loss_weight,
        "joint_adaptation_steps": config.joint_adaptation_steps,
        "joint_eval_every_steps": config.joint_eval_every_steps,
        "best_step": int(best_checkpoint["step_or_epoch"]),
        "best_target_validation": best_metrics,
        "best_target_validation_f1": best_f1,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }


# -----------------------------------------------------------------------------
# Smoke test and experiment controller
# -----------------------------------------------------------------------------


def run_smoke_test(
    config: DomainAdaptationConfig,
    source_train: Optional[Dataset],
    target_labelled_dataset: Dataset,
    target_validation_dataset: Dataset,
    device: torch.device,
    experiment_dir: Path,
    logger: logging.Logger,
) -> None:
    logger.info("Starting smoke test on device %s", device)

    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    source_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if source_train is not None:
        source_subset = Subset(source_train, range(min(4, len(source_train))))
        source_loader = make_loader(
            source_subset,
            batch_size=min(2, len(source_subset)),
            shuffle=False,
            config=config,
            seed_offset=901,
        )
        source_batch = next(iter(source_loader))
        source_images, source_masks = source_batch
        logger.info(
            "Smoke source shapes: images=%s masks=%s labels=%s",
            tuple(source_images.shape),
            tuple(source_masks.shape),
            torch.unique(source_masks).tolist(),
        )

        # Tests the source-pretraining optimisation path.
        source_loss_value = train_one_batch(
            model,
            optimizer,
            scaler,
            source_images,
            source_masks,
            device,
            amp_enabled,
            config.max_grad_norm,
        )
        logger.info("Smoke source optimisation loss: %.6f", source_loss_value)

    target_subset = Subset(
        target_labelled_dataset,
        range(min(4, len(target_labelled_dataset))),
    )
    target_loader = make_loader(
        target_subset,
        batch_size=min(2, len(target_subset)),
        shuffle=False,
        config=config,
        seed_offset=902,
    )
    target_images, target_masks = next(iter(target_loader))
    logger.info(
        "Smoke target shapes: images=%s masks=%s labels=%s",
        tuple(target_images.shape),
        tuple(target_masks.shape),
        torch.unique(target_masks).tolist(),
    )

    if config.adaptation_mode == "joint":
        if source_batch is None:
            raise RuntimeError("Joint smoke test requires a source dataset.")

        source_images, source_masks = source_batch
        optimizer.zero_grad(set_to_none=True)

        source_loss = compute_segmentation_loss(
            model=model,
            images=source_images,
            masks=source_masks,
            device=device,
            amp_enabled=amp_enabled,
        )
        scaler.scale(config.source_loss_weight * source_loss).backward()

        target_loss = compute_segmentation_loss(
            model=model,
            images=target_images,
            masks=target_masks,
            device=device,
            amp_enabled=amp_enabled,
        )
        scaler.scale(config.target_loss_weight * target_loss).backward()

        scaler.unscale_(optimizer)
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        combined_value = (
            config.source_loss_weight * float(source_loss.detach().item())
            + config.target_loss_weight * float(target_loss.detach().item())
        )
        logger.info(
            "Smoke joint optimisation | combined %.6f | source %.6f | target %.6f",
            combined_value,
            float(source_loss.detach().item()),
            float(target_loss.detach().item()),
        )
    else:
        target_loss_value = train_one_batch(
            model,
            optimizer,
            scaler,
            target_images,
            target_masks,
            device,
            amp_enabled,
            config.max_grad_norm,
        )
        logger.info("Smoke target optimisation loss: %.6f", target_loss_value)

    smoke_metrics = evaluate_model(
        model=model,
        dataset=target_validation_dataset,
        config=config,
        device=device,
        description="Smoke validation",
        max_batches=2,
    )
    logger.info("Smoke validation F1: %.6f", smoke_metrics["all"]["f1"])

    checkpoint_path = experiment_dir / "smoke_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        stage="smoke",
        step_or_epoch=1,
        metrics=smoke_metrics,
        config=config,
    )
    load_model_checkpoint(checkpoint_path, model, device)
    logger.info("Smoke checkpoint save/reload: OK")
    logger.info("Smoke test completed successfully")

def run_experiment(config: DomainAdaptationConfig, smoke_test: bool = False) -> None:
    config.validate()
    set_global_seed(config.seed)

    output_root = Path(config.output_dir)
    split_dir = output_root / "splits"
    experiment_dir = output_root / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(experiment_dir / "experiment.log")
    device = resolve_device(config)

    write_json(experiment_dir / "resolved_config.json", asdict(config))

    logger.info("Experiment: %s", config.experiment_name)
    logger.info("Source condition: %s", config.source_condition)
    logger.info("Adaptation mode: %s", config.adaptation_mode)
    logger.info("Device: %s", device)
    logger.info("PyTorch: %s", torch.__version__)

    source_train, source_validation = build_source_datasets(config, split_dir)
    (
        target_pool_dataset,
        target_validation_dataset,
        target_test_dataset,
        target_pool_pairs,
    ) = build_target_datasets(config, split_dir)

    selected_target_indices = load_or_create_initial_target_selection(
        target_pool_pairs=target_pool_pairs,
        target_fraction=config.initial_target_fraction,
        seed=config.seed,
        split_dir=split_dir,
    )
    target_labelled_dataset = Subset(target_pool_dataset, selected_target_indices)

    logger.info(
        "Dataset sizes | source train=%s | source val=%s | target pool=%d | "
        "target labelled=%d | target val=%d | target test=%d",
        len(source_train) if source_train is not None else 0,
        len(source_validation) if source_validation is not None else 0,
        len(target_pool_dataset),
        len(target_labelled_dataset),
        len(target_validation_dataset),
        len(target_test_dataset),
    )

    if smoke_test:
        run_smoke_test(
            config=config,
            source_train=source_train,
            target_labelled_dataset=target_labelled_dataset,
            target_validation_dataset=target_validation_dataset,
            device=device,
            experiment_dir=experiment_dir,
            logger=logger,
        )
        return

    model = build_model(config, device)

    source_training_results: Dict[str, Any]
    if source_train is not None and source_validation is not None:
        source_training_results = pretrain_on_source(
            model=model,
            source_train=source_train,
            source_validation=source_validation,
            config=config,
            device=device,
            experiment_dir=experiment_dir,
            logger=logger,
        )
    else:
        source_training_results = {"skipped": True}
        logger.info("No synthetic source pretraining for source_condition='none'.")

    # Evaluate source/pretrained initialization on target validation before using
    # any selected real target labels. This is not used for final test selection.
    pre_adaptation_target_validation = evaluate_model(
        model=model,
        dataset=target_validation_dataset,
        config=config,
        device=device,
        description="Target validation before adaptation",
    )
    logger.info(
        "Before target adaptation | target val F1 %.6f | IoU %.6f",
        float(pre_adaptation_target_validation["all"]["f1"]),
        float(pre_adaptation_target_validation["all"]["iou"]),
    )

    if config.adaptation_mode == "target_only":
        target_training_results = adapt_on_target(
            model=model,
            target_labelled_dataset=(
                target_labelled_dataset
            ),
            target_validation_dataset=(
                target_validation_dataset
            ),
            config=config,
            device=device,
            experiment_dir=experiment_dir,
            logger=logger,
        )

    elif config.adaptation_mode == "joint":
        if source_train is None:
            raise RuntimeError(
                "Joint adaptation requires "
                "source_train to be available."
            )

        target_training_results = (
            adapt_joint_source_target(
                model=model,
                source_train_dataset=source_train,
                target_labelled_dataset=(
                    target_labelled_dataset
                ),
                target_validation_dataset=(
                    target_validation_dataset
                ),
                config=config,
                device=device,
                experiment_dir=experiment_dir,
                logger=logger,
            )
        )

    else:
        raise RuntimeError(
            "Unexpected adaptation mode: "
            f"{config.adaptation_mode}"
        )

    target_test_metrics = evaluate_model(
        model=model,
        dataset=target_test_dataset,
        config=config,
        device=device,
        description="Final target test",
    )

    logger.info(
        "Final target test | F1 %.6f | IoU %.6f | precision %.6f | recall %.6f",
        float(target_test_metrics["all"]["f1"]),
        float(target_test_metrics["all"]["iou"]),
        float(target_test_metrics["all"]["precision"]),
        float(target_test_metrics["all"]["recall"]),
    )

    result = {
        "config": asdict(config),
        "dataset_sizes": {
            "source_train": len(source_train) if source_train is not None else 0,
            "source_validation": (
                len(source_validation) if source_validation is not None else 0
            ),
            "target_pool": len(target_pool_dataset),
            "target_labelled": len(target_labelled_dataset),
            "target_validation": len(target_validation_dataset),
            "target_test": len(target_test_dataset),
        },
        "source_training": source_training_results,
        "pre_adaptation_target_validation": pre_adaptation_target_validation,
        "target_training": target_training_results,
        "target_test": target_test_metrics,
    }
    write_json(experiment_dir / "result.json", result)
    logger.info("Saved result: %s", experiment_dir / "result.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone SegFormer synthetic-to-CrackSeg9k adaptation."
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run data loading, one source step, one target step, evaluation and checkpoint test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DomainAdaptationConfig.from_yaml(args.config)
    run_experiment(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()