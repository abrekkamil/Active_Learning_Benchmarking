import yaml
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning experiments."""
    experiment_name: str = "experiment_1"
    # =====================
    # Dataset
    # =====================
    dataset: str = "coco"
    data_dir: str = "/path/to/dataset"
    
    dataset_type: str = "deepcrack"
    num_classes: int = 2
    img_size: int = 256

    # =====================
    # Active Learning
    # =====================
    initial_labeled: float = 0.1
    query_size: int = 5
    al_cycles: int = 5
    epochs_per_cycle: int = 15
    initial_training_epoch: int = 3

    skip_cold_start: bool = False   # 👈 NEW
    cold_start_strategy: str = "random"
    query_strategy: str = "uncertainty"
    # =====================
    # Task / Model
    # =====================
    pool: bool = False
    # =====================
    # Task / Model
    # =====================
    model_name: str = "unet"  # resnet | unet | resnet50_multilabel | deeplabv3 | segformer | maskrcnn | yolo
    task: str = "segmentation"  # segmentation | detection | instance_segmentation | multilabel_classification
    use_cuda: bool = True

    batch_size: int = 8
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_steps: List[int] = field(default_factory=lambda: [30, 50])

    unet_norm: str = "bn"        # 👈 NEW (used in UNetModel)


    convnext_variant: str = "tiny"  # tiny | small | base | large
    convnext_decoder_channels: int = 256
    convnext_dropout: float = 0.1
    convnext_variant: str = "tiny"  # tiny | small | base | large
    convnext_decoder_channels: int = 256   # (kept for back-compat; unused by FPN model)
    convnext_dropout: float = 0.1
    convnext_fpn_channels: int = 128
    convnext_lr: float = 6e-5              # AdamW backbone lr; do NOT reuse config.lr
    convnext_head_lr_mult: float = 10.0    # decoder lr = convnext_lr * this
    crack_class_weight: float = 0.0        # 0 = off; e.g. 5.0 to upweight crack class
    ignore_index: int = -100
    # =====================
    # Multi-label Classification (OPTIONAL)
    # =====================
    cls_threshold: float = 0.5   # sigmoid threshold for binary prediction
    pretrained: bool = True       # use ImageNet weights for classification backbone
    # =====================
    # RL Active Learning (OPTIONAL)
    # =====================
    use_rl: bool = False         # 👈 switch
    policy_lr: float = 1e-4
    policy_hidden: int = 256
    policy_temp: float = 1.0
    entropy_beta: float = 0.01

    al_budget: int = 5
    candidate_pool: int = 256

    oracle_epochs: int = 5
    cycle_epochs: int = 5

    feature_batch: int = 64
    state_batch: int = 32

    policy_temp_start: float = 1.5
    policy_temp_end: float = 0.7
    candidate_ratio: float = 0.4

    budget_options: List[int] = field(default_factory=lambda: [8, 16, 24])
    cost_lambda: float = 0.001

    dynamic_query_size: bool = False
    # =====================
    # Logging / Repro
    # =====================
    seed: int = 42
    num_workers: int = 4

    use_wandb: bool = False
    wandb_project: str = "AL_benchmark"
    print_freq: int = 100

    checkpoint_dir: str = "results/checkpoints"
    results_dir: str = "results"

    # =====================
    # Utils
    # =====================
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(config_dict) - valid
        if unknown:
            print(f"[config] WARNING ignoring unknown keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in config_dict.items() if k in valid})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
