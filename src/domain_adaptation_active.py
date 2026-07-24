from __future__ import annotations

"""Active domain adaptation for synthetic-to-real crack segmentation.

This module extends ``src.domain_adaptation_joint`` without modifying the
existing target-only, joint-DA, classical AL, or RAL implementations.

Implemented acquisition strategies
----------------------------------
- ``random``: incremental random acquisition control.
- ``uncertainty``: top-k pixel entropy for sparse crack segmentation.
- ``damage_adaptive``: crack-aware uncertainty plus feature-diverse MMR.

The active-learning simulation never uses target masks during acquisition.
Masks are read only after an image has been selected and added to the labelled
training subset.
"""

import argparse
import dataclasses
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.domain_adaptation_joint import (
    ImageMaskPair,
    PairedMaskDataset,
    build_model,
    build_source_datasets,
    build_target_datasets,
    compute_segmentation_loss,
    configure_logging,
    evaluate_model,
    fraction_tag,
    load_model_checkpoint,
    make_loader,
    pretrain_on_source,
    resolve_device,
    save_checkpoint,
    set_global_seed,
    write_json,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ActiveDomainAdaptationConfig:
    experiment_name: str = "da_active_joint_uncertainty_1to5_seed42"

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

    # Source and adaptation protocol
    source_condition: str = "supervisely_views"
    adaptation_mode: str = "joint"
    active_learning: bool = True
    acquisition_strategy: str = "uncertainty"

    source_loss_weight: float = 1.0
    target_loss_weight: float = 1.0

    # Model
    model_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    img_size: int = 256
    num_classes: int = 2

    # Source pretraining
    source_steps: int = 5000
    source_eval_every_steps: int = 500
    source_checkpoint_path: str = ""

    # Active joint-adaptation schedule
    initial_target_fraction: float = 0.01
    query_fraction: float = 0.005
    final_target_fraction: float = 0.05
    al_cycles: int = 8

    initial_joint_steps: int = 1000
    steps_per_cycle: int = 500
    joint_adaptation_steps: int = 5000
    joint_eval_every_steps: int = 500

    # Optimisation
    adaptation_epochs: int = 15  # retained for configuration compatibility
    batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 6e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    amp: bool = True

    # Splits
    source_val_fraction: float = 0.10
    target_val_fraction: float = 0.10

    # Acquisition
    uncertainty_top_fraction: float = 0.10
    crack_top_fraction: float = 0.05
    candidate_pool_size: int = 512
    mmr_diversity_weight: float = 0.25

    damage_entropy_weight: float = 0.45
    damage_crack_weight: float = 0.20
    damage_edge_weight: float = 0.20
    damage_boundary_weight: float = 0.15

    # Set to a positive integer only for a fast smoke test. Scientific runs
    # should leave this at 0 so every unlabelled image is scored.
    scoring_max_samples: int = 0

    # Runtime/reproducibility
    seed: int = 42
    num_workers: int = 4
    use_cuda: bool = True

    # Output
    output_dir: str = "results/domain_adaptation"

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
    ) -> "ActiveDomainAdaptationConfig":
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
                "Unknown active-domain-adaptation configuration keys: "
                f"{unknown}."
            )

        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        valid_sources = {
            "roboflow",
            "supervisely_raw",
            "supervisely_views",
        }
        if self.source_condition not in valid_sources:
            raise ValueError(
                "Active joint DA requires a synthetic source. "
                f"source_condition must be one of {sorted(valid_sources)}, "
                f"got {self.source_condition!r}."
            )

        if self.adaptation_mode != "joint":
            raise ValueError(
                "Active domain adaptation requires adaptation_mode='joint'."
            )

        if not self.active_learning:
            raise ValueError("active_learning must be true for this module.")

        valid_strategies = {
            "random",
            "uncertainty",
            "damage_adaptive",
        }
        if self.acquisition_strategy not in valid_strategies:
            raise ValueError(
                "acquisition_strategy must be one of "
                f"{sorted(valid_strategies)}, got "
                f"{self.acquisition_strategy!r}."
            )

        if self.num_classes != 2:
            raise ValueError("This implementation requires num_classes=2.")

        if self.source_loss_weight < 0:
            raise ValueError("source_loss_weight cannot be negative.")
        if self.target_loss_weight <= 0:
            raise ValueError("target_loss_weight must be positive.")

        for name in ("source_val_fraction", "target_val_fraction"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}.")

        if not 0.0 < self.initial_target_fraction < 1.0:
            raise ValueError("initial_target_fraction must be in (0, 1).")
        if not 0.0 < self.query_fraction < 1.0:
            raise ValueError("query_fraction must be in (0, 1).")
        if not 0.0 < self.final_target_fraction <= 1.0:
            raise ValueError("final_target_fraction must be in (0, 1].")
        if self.initial_target_fraction >= self.final_target_fraction:
            raise ValueError(
                "initial_target_fraction must be smaller than "
                "final_target_fraction."
            )
        if self.al_cycles <= 0:
            raise ValueError("al_cycles must be positive.")

        expected_fraction = (
            self.initial_target_fraction
            + self.al_cycles * self.query_fraction
        )
        if not math.isclose(
            expected_fraction,
            self.final_target_fraction,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "initial_target_fraction + al_cycles * query_fraction "
                "must equal final_target_fraction. Got "
                f"{expected_fraction} versus {self.final_target_fraction}."
            )

        if self.source_steps < 0:
            raise ValueError("source_steps cannot be negative.")
        if self.source_eval_every_steps <= 0:
            raise ValueError("source_eval_every_steps must be positive.")
        if self.source_steps > 0 and (
            self.source_eval_every_steps > self.source_steps
        ):
            raise ValueError(
                "source_eval_every_steps cannot exceed source_steps."
            )

        if self.initial_joint_steps <= 0:
            raise ValueError("initial_joint_steps must be positive.")
        if self.steps_per_cycle <= 0:
            raise ValueError("steps_per_cycle must be positive.")
        if self.joint_eval_every_steps <= 0:
            raise ValueError("joint_eval_every_steps must be positive.")

        scheduled_steps = (
            self.initial_joint_steps
            + self.al_cycles * self.steps_per_cycle
        )
        if scheduled_steps != self.joint_adaptation_steps:
            raise ValueError(
                "initial_joint_steps + al_cycles * steps_per_cycle "
                "must equal joint_adaptation_steps. Got "
                f"{scheduled_steps} versus {self.joint_adaptation_steps}."
            )

        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")
        if self.img_size <= 0:
            raise ValueError("img_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm cannot be negative.")

        for name in ("uncertainty_top_fraction", "crack_top_fraction"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")

        if self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive.")
        if not 0.0 <= self.mmr_diversity_weight <= 1.0:
            raise ValueError("mmr_diversity_weight must be in [0, 1].")
        if self.scoring_max_samples < 0:
            raise ValueError("scoring_max_samples cannot be negative.")

        damage_weights = [
            self.damage_entropy_weight,
            self.damage_crack_weight,
            self.damage_edge_weight,
            self.damage_boundary_weight,
        ]
        if any(weight < 0 for weight in damage_weights):
            raise ValueError("Damage acquisition weights cannot be negative.")
        if sum(damage_weights) <= 0:
            raise ValueError("At least one damage acquisition weight is required.")


# -----------------------------------------------------------------------------
# Image-only deterministic scoring dataset
# -----------------------------------------------------------------------------


class TargetScoringDataset(Dataset):
    """Load deterministic target images without reading target masks."""

    def __init__(
        self,
        pairs: Sequence[ImageMaskPair],
        global_indices: Sequence[int],
        img_size: int,
    ) -> None:
        self.pairs = list(pairs)
        self.global_indices = list(global_indices)
        self.img_size = int(img_size)

        for index in self.global_indices:
            if index < 0 or index >= len(self.pairs):
                raise IndexError(f"Scoring index out of range: {index}")

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, local_index: int) -> Tuple[torch.Tensor, int]:
        global_index = self.global_indices[local_index]
        pair = self.pairs[global_index]

        with Image.open(pair.image_path) as image_file:
            image = image_file.convert("RGB")

        image = image.resize(
            (self.img_size, self.img_size),
            resample=Image.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim != 3 or array.shape[2] != 3:
            raise RuntimeError(
                f"Expected RGB image, got shape {array.shape} for "
                f"{pair.image_path}."
            )

        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
        return tensor, int(global_index)


# -----------------------------------------------------------------------------
# Reproducible cold start and utility helpers
# -----------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_or_create_active_initial_selection(
    target_pool_pairs: Sequence[ImageMaskPair],
    initial_fraction: float,
    seed: int,
    split_dir: Path,
) -> List[int]:
    """Create one shared random cold start for all acquisition strategies."""

    selection_path = split_dir / (
        f"target_active_initial_seed{seed}_"
        f"{fraction_tag(initial_fraction)}.json"
    )

    pairs_by_id = {pair.sample_id: pair for pair in target_pool_pairs}
    identifiers = sorted(pairs_by_id)
    selection_count = max(1, round(len(identifiers) * initial_fraction))
    selection_count = min(selection_count, len(identifiers))

    if selection_path.exists():
        payload = _read_json(selection_path)
        selected_ids = list(payload["selected_target_ids"])

        if int(payload.get("seed", seed)) != seed:
            raise RuntimeError(
                f"Cold-start file has a different seed: {selection_path}"
            )
        if len(selected_ids) != selection_count:
            raise RuntimeError(
                f"Cold-start file {selection_path} contains "
                f"{len(selected_ids)} IDs; expected {selection_count}."
            )
    else:
        rng = random.Random(seed)
        selected_ids = sorted(rng.sample(identifiers, selection_count))
        write_json(
            selection_path,
            {
                "seed": seed,
                "initial_target_fraction": initial_fraction,
                "original_target_pool_size": len(identifiers),
                "selected_count": len(selected_ids),
                "selected_target_ids": selected_ids,
            },
        )

    missing = set(selected_ids) - set(identifiers)
    if missing:
        raise RuntimeError(
            "Saved active cold start contains missing target samples. "
            f"Examples: {sorted(missing)[:10]}"
        )

    index_by_id = {
        pair.sample_id: index
        for index, pair in enumerate(target_pool_pairs)
    }
    return [index_by_id[sample_id] for sample_id in selected_ids]


def target_count_for_fraction(pool_size: int, fraction: float) -> int:
    return min(pool_size, max(1, round(pool_size * fraction)))


def desired_count_after_cycle(
    pool_size: int,
    config: ActiveDomainAdaptationConfig,
    cycle: int,
) -> int:
    if cycle < 0 or cycle > config.al_cycles:
        raise ValueError(f"Invalid cycle: {cycle}")

    fraction = min(
        config.final_target_fraction,
        config.initial_target_fraction + cycle * config.query_fraction,
    )
    return target_count_for_fraction(pool_size, fraction)


def normalise_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise FloatingPointError("Acquisition scores contain non-finite values.")
    if maximum - minimum < 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def make_scoring_loader(
    dataset: Dataset,
    config: ActiveDomainAdaptationConfig,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)

    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.use_cuda and torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


# -----------------------------------------------------------------------------
# Acquisition scoring
# -----------------------------------------------------------------------------


@dataclass
class AcquisitionRecord:
    global_index: int
    score: Optional[float]
    components: Dict[str, float]


def entropy_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return -(
        probabilities
        * torch.log(probabilities.clamp_min(1e-8))
    ).sum(dim=1)


def top_fraction_mean(
    values: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    flat = values.flatten(1)
    count = max(1, round(flat.shape[1] * fraction))
    count = min(count, flat.shape[1])
    return torch.topk(flat, k=count, dim=1).values.mean(dim=1)


def sobel_edge_strength(images: torch.Tensor) -> torch.Tensor:
    grayscale = (
        0.2989 * images[:, 0:1]
        + 0.5870 * images[:, 1:2]
        + 0.1140 * images[:, 2:3]
    )

    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3)

    gradient_x = F.conv2d(grayscale, kernel_x, padding=1)
    gradient_y = F.conv2d(grayscale, kernel_y, padding=1)
    magnitude = torch.sqrt(
        gradient_x.square() + gradient_y.square() + 1e-12
    ).squeeze(1)

    maximum = magnitude.flatten(1).amax(dim=1).view(-1, 1, 1)
    return magnitude / maximum.clamp_min(1e-8)


def predicted_boundary_mask(crack_probability: torch.Tensor) -> torch.Tensor:
    prediction = (crack_probability >= 0.5).float().unsqueeze(1)
    dilated = F.max_pool2d(prediction, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-prediction, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0).squeeze(1)


def extract_image_features(
    outputs: Any,
    logits: torch.Tensor,
) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        hidden = hidden_states[-1]
        if hidden.ndim == 4:
            features = F.adaptive_avg_pool2d(hidden, output_size=1).flatten(1)
        elif hidden.ndim == 3:
            features = hidden.mean(dim=1)
        else:
            features = logits.mean(dim=(-2, -1))
    else:
        features = logits.mean(dim=(-2, -1))

    return F.normalize(features.float(), p=2, dim=1, eps=1e-8)


def choose_scoring_indices(
    unlabelled_indices: Sequence[int],
    config: ActiveDomainAdaptationConfig,
    cycle: int,
) -> List[int]:
    indices = list(unlabelled_indices)
    if config.scoring_max_samples <= 0:
        return indices
    if len(indices) <= config.scoring_max_samples:
        return indices

    # This is only intended for smoke testing. Scientific configurations must
    # keep scoring_max_samples=0.
    rng = random.Random(config.seed + cycle * 1_000_003 + 77)
    return sorted(rng.sample(indices, config.scoring_max_samples))


def score_unlabelled_pool(
    model: torch.nn.Module,
    target_pool_pairs: Sequence[ImageMaskPair],
    unlabelled_indices: Sequence[int],
    config: ActiveDomainAdaptationConfig,
    device: torch.device,
    cycle: int,
    need_damage_components: bool,
) -> Tuple[List[int], Dict[str, np.ndarray], np.ndarray]:
    """Score target images without opening or using their masks."""

    scoring_indices = choose_scoring_indices(
        unlabelled_indices=unlabelled_indices,
        config=config,
        cycle=cycle,
    )
    if not scoring_indices:
        raise RuntimeError("No unlabelled samples are available for scoring.")

    dataset = TargetScoringDataset(
        pairs=target_pool_pairs,
        global_indices=scoring_indices,
        img_size=config.img_size,
    )
    loader = make_scoring_loader(
        dataset=dataset,
        config=config,
        seed_offset=30_000 + cycle,
    )

    global_order: List[int] = []
    top_entropy_values: List[float] = []
    crack_values: List[float] = []
    edge_values: List[float] = []
    boundary_values: List[float] = []
    feature_batches: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        progress = tqdm(
            loader,
            desc=f"Acquisition scoring cycle {cycle}",
            leave=False,
        )

        for images, global_indices in progress:
            images = images.to(device, non_blocking=True)

            outputs = model(
                pixel_values=images,
                output_hidden_states=need_damage_components,
                return_dict=True,
            )
            logits = F.interpolate(
                outputs.logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)
            entropy = entropy_from_probabilities(probabilities)

            top_entropy = top_fraction_mean(
                entropy,
                config.uncertainty_top_fraction,
            )

            global_order.extend(int(value) for value in global_indices.tolist())
            top_entropy_values.extend(top_entropy.cpu().tolist())

            if not need_damage_components:
                continue

            crack_probability = probabilities[:, 1]
            crack_presence = top_fraction_mean(
                crack_probability,
                config.crack_top_fraction,
            )

            edge_strength = sobel_edge_strength(images)
            edge_entropy = (
                (entropy * edge_strength).flatten(1).sum(dim=1)
                / edge_strength.flatten(1).sum(dim=1).clamp_min(1e-8)
            )

            boundary = predicted_boundary_mask(crack_probability)
            boundary_denominator = boundary.flatten(1).sum(dim=1)
            boundary_entropy = (
                (entropy * boundary).flatten(1).sum(dim=1)
                / boundary_denominator.clamp_min(1e-8)
            )
            boundary_entropy = torch.where(
                boundary_denominator > 0,
                boundary_entropy,
                top_entropy,
            )

            features = extract_image_features(outputs, logits)

            crack_values.extend(crack_presence.cpu().tolist())
            edge_values.extend(edge_entropy.cpu().tolist())
            boundary_values.extend(boundary_entropy.cpu().tolist())
            feature_batches.append(features.cpu().numpy())

    components: Dict[str, np.ndarray] = {
        "top_entropy": np.asarray(top_entropy_values, dtype=np.float64),
    }

    if need_damage_components:
        components.update(
            {
                "crack_probability": np.asarray(crack_values, dtype=np.float64),
                "edge_weighted_entropy": np.asarray(edge_values, dtype=np.float64),
                "boundary_entropy": np.asarray(boundary_values, dtype=np.float64),
            }
        )
        features_array = np.concatenate(feature_batches, axis=0)
    else:
        features_array = np.empty((len(global_order), 0), dtype=np.float32)

    expected = len(global_order)
    for name, values in components.items():
        if len(values) != expected:
            raise RuntimeError(
                f"Scoring component {name} has {len(values)} values; "
                f"expected {expected}."
            )

    return global_order, components, features_array


def acquire_random(
    unlabelled_indices: Sequence[int],
    query_count: int,
    seed: int,
    cycle: int,
) -> List[AcquisitionRecord]:
    if query_count <= 0:
        return []

    rng = random.Random(seed + cycle * 100_003)
    selected = rng.sample(
        list(unlabelled_indices),
        min(query_count, len(unlabelled_indices)),
    )
    return [
        AcquisitionRecord(
            global_index=int(index),
            score=None,
            components={},
        )
        for index in sorted(selected)
    ]


def acquire_uncertainty(
    model: torch.nn.Module,
    target_pool_pairs: Sequence[ImageMaskPair],
    unlabelled_indices: Sequence[int],
    query_count: int,
    config: ActiveDomainAdaptationConfig,
    device: torch.device,
    cycle: int,
) -> List[AcquisitionRecord]:
    global_indices, components, _ = score_unlabelled_pool(
        model=model,
        target_pool_pairs=target_pool_pairs,
        unlabelled_indices=unlabelled_indices,
        config=config,
        device=device,
        cycle=cycle,
        need_damage_components=False,
    )

    scores = components["top_entropy"]
    order = np.argsort(-scores, kind="mergesort")
    selected_positions = order[: min(query_count, len(order))]

    return [
        AcquisitionRecord(
            global_index=int(global_indices[position]),
            score=float(scores[position]),
            components={"top_entropy": float(scores[position])},
        )
        for position in selected_positions
    ]


def greedy_mmr_positions(
    relevance: np.ndarray,
    features: np.ndarray,
    query_count: int,
    diversity_weight: float,
) -> List[int]:
    if query_count <= 0 or relevance.size == 0:
        return []

    query_count = min(query_count, relevance.size)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != relevance.size:
        raise ValueError("MMR feature matrix shape is inconsistent.")

    if features.shape[1] == 0 or diversity_weight <= 0:
        return np.argsort(-relevance, kind="mergesort")[:query_count].tolist()

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)

    remaining = set(range(relevance.size))
    selected: List[int] = []

    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < query_count:
        remaining_list = np.asarray(sorted(remaining), dtype=np.int64)
        selected_features = features[np.asarray(selected, dtype=np.int64)]
        candidate_features = features[remaining_list]

        similarities = candidate_features @ selected_features.T
        maximum_similarity = np.max(similarities, axis=1)

        mmr_scores = (
            (1.0 - diversity_weight) * relevance[remaining_list]
            - diversity_weight * maximum_similarity
        )
        best_local = int(np.argmax(mmr_scores))
        best_position = int(remaining_list[best_local])

        selected.append(best_position)
        remaining.remove(best_position)

    return selected


def acquire_damage_adaptive(
    model: torch.nn.Module,
    target_pool_pairs: Sequence[ImageMaskPair],
    unlabelled_indices: Sequence[int],
    query_count: int,
    config: ActiveDomainAdaptationConfig,
    device: torch.device,
    cycle: int,
) -> List[AcquisitionRecord]:
    global_indices, raw_components, features = score_unlabelled_pool(
        model=model,
        target_pool_pairs=target_pool_pairs,
        unlabelled_indices=unlabelled_indices,
        config=config,
        device=device,
        cycle=cycle,
        need_damage_components=True,
    )

    normalised = {
        name: normalise_scores(values)
        for name, values in raw_components.items()
    }

    weight_sum = (
        config.damage_entropy_weight
        + config.damage_crack_weight
        + config.damage_edge_weight
        + config.damage_boundary_weight
    )

    relevance = (
        config.damage_entropy_weight * normalised["top_entropy"]
        + config.damage_crack_weight * normalised["crack_probability"]
        + config.damage_edge_weight * normalised["edge_weighted_entropy"]
        + config.damage_boundary_weight * normalised["boundary_entropy"]
    ) / weight_sum

    candidate_count = min(
        len(global_indices),
        max(query_count, config.candidate_pool_size),
    )
    candidate_positions = np.argsort(
        -relevance,
        kind="mergesort",
    )[:candidate_count]

    selected_candidate_positions = greedy_mmr_positions(
        relevance=relevance[candidate_positions],
        features=features[candidate_positions],
        query_count=query_count,
        diversity_weight=config.mmr_diversity_weight,
    )
    selected_positions = [
        int(candidate_positions[position])
        for position in selected_candidate_positions
    ]

    records: List[AcquisitionRecord] = []
    for position in selected_positions:
        records.append(
            AcquisitionRecord(
                global_index=int(global_indices[position]),
                score=float(relevance[position]),
                components={
                    name: float(values[position])
                    for name, values in raw_components.items()
                },
            )
        )
    return records


def acquire_samples(
    model: torch.nn.Module,
    target_pool_pairs: Sequence[ImageMaskPair],
    unlabelled_indices: Sequence[int],
    query_count: int,
    config: ActiveDomainAdaptationConfig,
    device: torch.device,
    cycle: int,
) -> List[AcquisitionRecord]:
    if query_count <= 0:
        return []

    if config.acquisition_strategy == "random":
        return acquire_random(
            unlabelled_indices=unlabelled_indices,
            query_count=query_count,
            seed=config.seed,
            cycle=cycle,
        )

    if config.acquisition_strategy == "uncertainty":
        return acquire_uncertainty(
            model=model,
            target_pool_pairs=target_pool_pairs,
            unlabelled_indices=unlabelled_indices,
            query_count=query_count,
            config=config,
            device=device,
            cycle=cycle,
        )

    if config.acquisition_strategy == "damage_adaptive":
        return acquire_damage_adaptive(
            model=model,
            target_pool_pairs=target_pool_pairs,
            unlabelled_indices=unlabelled_indices,
            query_count=query_count,
            config=config,
            device=device,
            cycle=cycle,
        )

    raise AssertionError(
        f"Unhandled acquisition strategy: {config.acquisition_strategy}"
    )


# -----------------------------------------------------------------------------
# Fixed-step joint training used throughout all active cycles
# -----------------------------------------------------------------------------


def save_active_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: ActiveDomainAdaptationConfig,
    global_step: int,
    cycle: int,
    labelled_count: int,
    validation_metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "active_joint",
        "global_step": int(global_step),
        "cycle": int(cycle),
        "labelled_count": int(labelled_count),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "metrics": dict(validation_metrics),
        "config": asdict(config),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_active_joint_stage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    source_train_dataset: Dataset,
    target_labelled_dataset: Dataset,
    target_validation_dataset: Dataset,
    config: ActiveDomainAdaptationConfig,
    device: torch.device,
    experiment_dir: Path,
    logger: logging.Logger,
    cycle: int,
    number_of_steps: int,
    global_step_start: int,
    best_tracker: Dict[str, Any],
) -> Dict[str, Any]:
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive.")
    if len(target_labelled_dataset) == 0:
        raise RuntimeError("Target labelled dataset is empty.")

    amp_enabled = config.amp and device.type == "cuda"

    source_loader = make_loader(
        dataset=source_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,  # duck-typed compatible configuration
        seed_offset=40_000 + cycle * 100,
        drop_last=len(source_train_dataset) >= config.batch_size,
    )
    target_loader = make_loader(
        dataset=target_labelled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,
        seed_offset=50_000 + cycle * 100,
        drop_last=len(target_labelled_dataset) >= config.batch_size,
    )

    source_epoch = cycle * 10_000
    target_epoch = 0
    if hasattr(source_train_dataset, "set_epoch"):
        source_train_dataset.set_epoch(source_epoch)  # type: ignore[attr-defined]

    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)

    interval_combined_loss = 0.0
    interval_source_loss = 0.0
    interval_target_loss = 0.0
    interval_steps = 0

    evaluations: List[Dict[str, Any]] = []
    stage_started = time.time()

    progress = tqdm(
        range(1, number_of_steps + 1),
        desc=f"Active joint cycle {cycle}",
    )

    for local_step in progress:
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
            target_epoch += 1
            target_iterator = iter(target_loader)
            target_images, target_masks = next(target_iterator)

        optimizer.zero_grad(set_to_none=True)

        # Sequential backpropagation lowers peak memory while preserving the
        # gradient of source_weight * source_loss + target_weight * target_loss.
        source_loss = compute_segmentation_loss(
            model=model,
            images=source_images,
            masks=source_masks,
            device=device,
            amp_enabled=amp_enabled,
        )
        weighted_source = config.source_loss_weight * source_loss
        scaler.scale(weighted_source).backward()

        target_loss = compute_segmentation_loss(
            model=model,
            images=target_images,
            masks=target_masks,
            device=device,
            amp_enabled=amp_enabled,
        )
        weighted_target = config.target_loss_weight * target_loss
        scaler.scale(weighted_target).backward()

        combined_loss = weighted_source.detach() + weighted_target.detach()
        if not torch.isfinite(combined_loss):
            raise FloatingPointError(
                f"Non-finite active joint loss: {combined_loss.item()}"
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

        global_step = global_step_start + local_step
        progress.set_postfix(
            total=f"{combined_value:.4f}",
            source=f"{source_value:.4f}",
            target=f"{target_value:.4f}",
        )

        should_evaluate = (
            global_step % config.joint_eval_every_steps == 0
            or local_step == number_of_steps
        )
        if not should_evaluate:
            continue

        denominator = max(interval_steps, 1)
        mean_combined = interval_combined_loss / denominator
        mean_source = interval_source_loss / denominator
        mean_target = interval_target_loss / denominator

        validation_metrics = evaluate_model(
            model=model,
            dataset=target_validation_dataset,
            config=config,
            device=device,
            description=(
                f"Active target validation cycle {cycle} "
                f"@ step {global_step}"
            ),
        )
        validation_f1 = float(validation_metrics["all"]["f1"])

        evaluation_record = {
            "cycle": cycle,
            "global_step": global_step,
            "labelled_count": len(target_labelled_dataset),
            "combined_train_loss": mean_combined,
            "source_train_loss": mean_source,
            "target_train_loss": mean_target,
            "validation": validation_metrics,
        }
        evaluations.append(evaluation_record)

        logger.info(
            "Active cycle %d | step %d/%d | labelled %d | "
            "combined loss %.6f | source loss %.6f | "
            "target loss %.6f | val F1 %.6f | val IoU %.6f | "
            "precision %.6f | recall %.6f",
            cycle,
            global_step,
            config.joint_adaptation_steps,
            len(target_labelled_dataset),
            mean_combined,
            mean_source,
            mean_target,
            validation_f1,
            float(validation_metrics["all"]["iou"]),
            float(validation_metrics["all"]["precision"]),
            float(validation_metrics["all"]["recall"]),
        )

        save_active_checkpoint(
            experiment_dir / "active_last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            global_step=global_step,
            cycle=cycle,
            labelled_count=len(target_labelled_dataset),
            validation_metrics=validation_metrics,
        )

        if validation_f1 > float(best_tracker["f1"]):
            best_tracker.update(
                {
                    "f1": validation_f1,
                    "cycle": cycle,
                    "global_step": global_step,
                    "labelled_count": len(target_labelled_dataset),
                    "metrics": validation_metrics,
                }
            )
            save_active_checkpoint(
                experiment_dir / "active_best_validation.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                global_step=global_step,
                cycle=cycle,
                labelled_count=len(target_labelled_dataset),
                validation_metrics=validation_metrics,
            )

        interval_combined_loss = 0.0
        interval_source_loss = 0.0
        interval_target_loss = 0.0
        interval_steps = 0

    if not evaluations:
        raise RuntimeError("Active stage completed without validation.")

    final_global_step = global_step_start + number_of_steps
    last_validation = evaluations[-1]["validation"]

    cycle_checkpoint = experiment_dir / f"cycle_{cycle:02d}.pt"
    save_active_checkpoint(
        cycle_checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        global_step=final_global_step,
        cycle=cycle,
        labelled_count=len(target_labelled_dataset),
        validation_metrics=last_validation,
    )

    return {
        "cycle": cycle,
        "global_step_start": global_step_start,
        "global_step_end": final_global_step,
        "number_of_steps": number_of_steps,
        "labelled_count": len(target_labelled_dataset),
        "evaluations": evaluations,
        "last_validation": last_validation,
        "elapsed_seconds": time.time() - stage_started,
        "checkpoint": str(cycle_checkpoint),
    }


# -----------------------------------------------------------------------------
# Experiment controller
# -----------------------------------------------------------------------------


def serialise_acquisition_records(
    records: Sequence[AcquisitionRecord],
    target_pool_pairs: Sequence[ImageMaskPair],
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for record in records:
        pair = target_pool_pairs[record.global_index]
        payload.append(
            {
                "global_index": int(record.global_index),
                "sample_id": pair.sample_id,
                "score": record.score,
                "components": record.components,
            }
        )
    return payload


def calculate_budget_auc(history: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if len(history) < 2:
        return {
            "raw_auc_f1_vs_fraction": 0.0,
            "normalised_auc_f1_vs_fraction": 0.0,
        }

    fractions = np.asarray(
        [float(item["labelled_fraction"]) for item in history],
        dtype=np.float64,
    )
    f1_values = np.asarray(
        [float(item["validation"]["all"]["f1"]) for item in history],
        dtype=np.float64,
    )

    order = np.argsort(fractions)
    fractions = fractions[order]
    f1_values = f1_values[order]

    raw_auc = float(np.trapz(f1_values, fractions))
    width = float(fractions[-1] - fractions[0])
    normalised_auc = raw_auc / width if width > 0 else 0.0

    return {
        "raw_auc_f1_vs_fraction": raw_auc,
        "normalised_auc_f1_vs_fraction": normalised_auc,
    }


def run_experiment(config: ActiveDomainAdaptationConfig) -> None:
    config.validate()
    set_global_seed(config.seed)

    output_root = Path(config.output_dir)
    split_dir = output_root / "splits"
    experiment_dir = output_root / config.experiment_name
    selection_dir = experiment_dir / "selections"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(experiment_dir / "experiment.log")
    device = resolve_device(config)

    write_json(experiment_dir / "resolved_config.json", asdict(config))

    logger.info("Experiment: %s", config.experiment_name)
    logger.info("Acquisition strategy: %s", config.acquisition_strategy)
    logger.info("Source condition: %s", config.source_condition)
    logger.info("Adaptation mode: %s", config.adaptation_mode)
    logger.info("Device: %s", device)
    logger.info("PyTorch: %s", torch.__version__)

    source_train, source_validation = build_source_datasets(
        config,
        split_dir,
    )
    if source_train is None or source_validation is None:
        raise RuntimeError("Active joint DA requires source datasets.")

    (
        _target_pool_dataset_from_builder,
        target_validation_dataset,
        target_test_dataset,
        target_pool_pairs,
    ) = build_target_datasets(config, split_dir)

    target_pool_training_dataset = PairedMaskDataset(
        target_pool_pairs,
        config.img_size,
        training=True,
    )

    initial_indices = load_or_create_active_initial_selection(
        target_pool_pairs=target_pool_pairs,
        initial_fraction=config.initial_target_fraction,
        seed=config.seed,
        split_dir=split_dir,
    )

    labelled_indices = sorted(set(initial_indices))
    all_indices = list(range(len(target_pool_pairs)))
    labelled_set = set(labelled_indices)
    unlabelled_indices = [
        index for index in all_indices if index not in labelled_set
    ]

    expected_initial_count = desired_count_after_cycle(
        len(target_pool_pairs),
        config,
        cycle=0,
    )
    if len(labelled_indices) != expected_initial_count:
        raise RuntimeError(
            f"Initial labelled count is {len(labelled_indices)}; "
            f"expected {expected_initial_count}."
        )

    logger.info(
        "Dataset sizes | source train=%d | source val=%d | "
        "target pool=%d | initial labelled=%d | target val=%d | "
        "target test=%d",
        len(source_train),
        len(source_validation),
        len(target_pool_pairs),
        len(labelled_indices),
        len(target_validation_dataset),
        len(target_test_dataset),
    )

    model = build_model(config, device)

    source_checkpoint_path = (
        Path(config.source_checkpoint_path)
        if config.source_checkpoint_path.strip()
        else None
    )

    if source_checkpoint_path is not None:
        if not source_checkpoint_path.exists():
            raise FileNotFoundError(
                f"source_checkpoint_path not found: {source_checkpoint_path}"
            )
        source_checkpoint = load_model_checkpoint(
            source_checkpoint_path,
            model,
            device,
        )
        source_training_results: Dict[str, Any] = {
            "loaded_checkpoint": str(source_checkpoint_path),
            "checkpoint_stage": source_checkpoint.get("stage"),
            "checkpoint_step_or_epoch": source_checkpoint.get("step_or_epoch"),
            "checkpoint_metrics": source_checkpoint.get("metrics", {}),
        }
        logger.info(
            "Loaded source checkpoint instead of source pretraining: %s",
            source_checkpoint_path,
        )
    else:
        source_training_results = pretrain_on_source(
            model=model,
            source_train=source_train,
            source_validation=source_validation,
            config=config,
            device=device,
            experiment_dir=experiment_dir,
            logger=logger,
        )

    pre_active_validation = evaluate_model(
        model=model,
        dataset=target_validation_dataset,
        config=config,
        device=device,
        description="Target validation before active adaptation",
    )
    logger.info(
        "Before active adaptation | target val F1 %.6f | IoU %.6f",
        float(pre_active_validation["all"]["f1"]),
        float(pre_active_validation["all"]["iou"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_tracker: Dict[str, Any] = {
        "f1": -math.inf,
        "cycle": None,
        "global_step": None,
        "labelled_count": None,
        "metrics": None,
    }

    active_history: List[Dict[str, Any]] = []
    total_started = time.time()
    global_step = 0

    # Cycle 0: train the common random cold start before model-guided queries.
    target_labelled_dataset = Subset(
        target_pool_training_dataset,
        labelled_indices,
    )
    initial_stage = train_active_joint_stage(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        source_train_dataset=source_train,
        target_labelled_dataset=target_labelled_dataset,
        target_validation_dataset=target_validation_dataset,
        config=config,
        device=device,
        experiment_dir=experiment_dir,
        logger=logger,
        cycle=0,
        number_of_steps=config.initial_joint_steps,
        global_step_start=global_step,
        best_tracker=best_tracker,
    )
    global_step = int(initial_stage["global_step_end"])

    initial_selection_payload = [
        {
            "global_index": int(index),
            "sample_id": target_pool_pairs[index].sample_id,
        }
        for index in labelled_indices
    ]
    write_json(
        selection_dir / "cycle_00_initial.json",
        {
            "cycle": 0,
            "strategy": "random_cold_start",
            "selected_count": len(initial_selection_payload),
            "selected": initial_selection_payload,
        },
    )

    active_history.append(
        {
            "cycle": 0,
            "strategy": "random_cold_start",
            "labelled_count": len(labelled_indices),
            "labelled_fraction": len(labelled_indices) / len(target_pool_pairs),
            "selected_count": len(labelled_indices),
            "selected_sample_ids": [
                target_pool_pairs[index].sample_id
                for index in labelled_indices
            ],
            "training": initial_stage,
            "validation": initial_stage["last_validation"],
        }
    )

    for cycle in range(1, config.al_cycles + 1):
        desired_count = desired_count_after_cycle(
            len(target_pool_pairs),
            config,
            cycle,
        )
        query_count = desired_count - len(labelled_indices)
        if query_count <= 0:
            raise RuntimeError(
                f"Cycle {cycle} produced a non-positive query count: "
                f"{query_count}."
            )

        logger.info(
            "Acquisition cycle %d/%d | strategy=%s | "
            "labelled=%d | unlabelled=%d | query=%d",
            cycle,
            config.al_cycles,
            config.acquisition_strategy,
            len(labelled_indices),
            len(unlabelled_indices),
            query_count,
        )

        records = acquire_samples(
            model=model,
            target_pool_pairs=target_pool_pairs,
            unlabelled_indices=unlabelled_indices,
            query_count=query_count,
            config=config,
            device=device,
            cycle=cycle,
        )
        selected_indices = [record.global_index for record in records]

        if len(selected_indices) != query_count:
            raise RuntimeError(
                f"Cycle {cycle} selected {len(selected_indices)} images; "
                f"expected {query_count}. For scientific runs, ensure "
                "scoring_max_samples is 0 or at least the query count."
            )
        if len(selected_indices) != len(set(selected_indices)):
            raise RuntimeError(f"Cycle {cycle} selected duplicate images.")
        if not set(selected_indices).issubset(set(unlabelled_indices)):
            raise RuntimeError(
                f"Cycle {cycle} selected an image outside the unlabelled pool."
            )

        labelled_indices = sorted(labelled_indices + selected_indices)
        selected_set = set(selected_indices)
        unlabelled_indices = [
            index
            for index in unlabelled_indices
            if index not in selected_set
        ]

        if len(labelled_indices) != desired_count:
            raise RuntimeError(
                f"Cycle {cycle} labelled count is {len(labelled_indices)}; "
                f"expected {desired_count}."
            )

        serialised_records = serialise_acquisition_records(
            records,
            target_pool_pairs,
        )
        write_json(
            selection_dir / f"cycle_{cycle:02d}.json",
            {
                "cycle": cycle,
                "strategy": config.acquisition_strategy,
                "query_count": query_count,
                "labelled_count_after_query": len(labelled_indices),
                "labelled_fraction_after_query": (
                    len(labelled_indices) / len(target_pool_pairs)
                ),
                "selected": serialised_records,
            },
        )

        target_labelled_dataset = Subset(
            target_pool_training_dataset,
            labelled_indices,
        )
        stage = train_active_joint_stage(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            source_train_dataset=source_train,
            target_labelled_dataset=target_labelled_dataset,
            target_validation_dataset=target_validation_dataset,
            config=config,
            device=device,
            experiment_dir=experiment_dir,
            logger=logger,
            cycle=cycle,
            number_of_steps=config.steps_per_cycle,
            global_step_start=global_step,
            best_tracker=best_tracker,
        )
        global_step = int(stage["global_step_end"])

        selected_scores = [
            record.score
            for record in records
            if record.score is not None
        ]

        active_history.append(
            {
                "cycle": cycle,
                "strategy": config.acquisition_strategy,
                "labelled_count": len(labelled_indices),
                "labelled_fraction": (
                    len(labelled_indices) / len(target_pool_pairs)
                ),
                "selected_count": len(records),
                "selected_sample_ids": [
                    target_pool_pairs[index].sample_id
                    for index in selected_indices
                ],
                "selected_score_mean": (
                    float(np.mean(selected_scores))
                    if selected_scores
                    else None
                ),
                "selected_score_min": (
                    float(np.min(selected_scores))
                    if selected_scores
                    else None
                ),
                "selected_score_max": (
                    float(np.max(selected_scores))
                    if selected_scores
                    else None
                ),
                "training": stage,
                "validation": stage["last_validation"],
            }
        )

    expected_final_count = target_count_for_fraction(
        len(target_pool_pairs),
        config.final_target_fraction,
    )
    if len(labelled_indices) != expected_final_count:
        raise RuntimeError(
            f"Final labelled count is {len(labelled_indices)}; "
            f"expected {expected_final_count}."
        )
    if global_step != config.joint_adaptation_steps:
        raise RuntimeError(
            f"Completed {global_step} joint steps; expected "
            f"{config.joint_adaptation_steps}."
        )

    # Save the exact final-budget model. This, rather than the best checkpoint
    # from a smaller earlier budget, is the primary model evaluated on test.
    save_active_checkpoint(
        experiment_dir / "active_final.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        global_step=global_step,
        cycle=config.al_cycles,
        labelled_count=len(labelled_indices),
        validation_metrics=active_history[-1]["validation"],
    )

    final_test_metrics = evaluate_model(
        model=model,
        dataset=target_test_dataset,
        config=config,
        device=device,
        description="Final active target test",
    )
    logger.info(
        "Final active target test | strategy %s | labelled %d | "
        "F1 %.6f | IoU %.6f | precision %.6f | recall %.6f",
        config.acquisition_strategy,
        len(labelled_indices),
        float(final_test_metrics["all"]["f1"]),
        float(final_test_metrics["all"]["iou"]),
        float(final_test_metrics["all"]["precision"]),
        float(final_test_metrics["all"]["recall"]),
    )

    auc_metrics = calculate_budget_auc(active_history)

    result = {
        "config": asdict(config),
        "dataset_sizes": {
            "source_train": len(source_train),
            "source_validation": len(source_validation),
            "target_pool": len(target_pool_pairs),
            "initial_target_labelled": len(initial_indices),
            "final_target_labelled": len(labelled_indices),
            "target_validation": len(target_validation_dataset),
            "target_test": len(target_test_dataset),
        },
        "source_training": source_training_results,
        "pre_active_target_validation": pre_active_validation,
        "active_history": active_history,
        "budget_auc": auc_metrics,
        "best_validation_checkpoint": best_tracker,
        "final_target_test": final_test_metrics,
        "final_labelled_sample_ids": [
            target_pool_pairs[index].sample_id
            for index in labelled_indices
        ],
        "elapsed_seconds": time.time() - total_started,
    }
    write_json(experiment_dir / "result.json", result)
    logger.info("Saved result: %s", experiment_dir / "result.json")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Active joint source-target adaptation for SegFormer crack "
            "segmentation."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the active-domain-adaptation YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ActiveDomainAdaptationConfig.from_yaml(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()