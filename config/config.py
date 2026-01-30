import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ActiveLearningConfig:
    """Configuration for active learning experiments."""
    
    # Dataset settings
    dataset: str = "coco"
    data_dir: str = "/path/to/dataset"
    dataset_name: str = "experiment_1"
    dataset_type: str = "deepcrack"
    num_classes: int = 2  # Including background
    img_size: int = 256

    # Active learning parameters
    initial_labeled: float = 0.1  # or absolute number if > 1
    query_size: int = 5
    al_cycles: int = 5
    epochs_per_cycle: int = 15
    initial_training_epoch: int = 3
    
    # Model parameters
    use_cuda: bool = True
    lr: float = 0.02 * 1 / 16
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_steps: List[int] = field(default_factory=lambda: [30, 50])
    
    # Strategy selection
    cold_start_strategy: str = "random"
    query_strategy: str = "uncertainty"
    task: str = "detection"  # or "segmentation"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "AL_benchmark"
    print_freq: int = 100
    
    # Technical settings
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "results/checkpoints"
    results_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for WandB logging."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }