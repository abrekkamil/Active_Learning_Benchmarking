"""
Active Learning Benchmark for Mask R-CNN

A comprehensive framework for benchmarking active learning strategies
with Mask R-CNN for object detection and segmentation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main classes
from .active_learning import ActiveLearningSystem
from .cold_start_strategies import ColdStartStrategies
from .query_strategies import QueryStrategies
from .models import MaskRCNNModel, WeakModel, FeatureExtractor
from .utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    calculate_metrics,
    compute_query_efficiency,
    create_experiment_summary,
    save_results_json,
    load_results_json
)
from .visualization import ActiveLearningVisualizer

# Configuration
from config.config import ActiveLearningConfig

# Constants
SUPPORTED_DATASETS = ["coco", "voc", "custom"]
SUPPORTED_COLD_START_STRATEGIES = [
    "random",
    "simple_diversity",
    "diversity",
    "entropy_based_uncertainty",
    "uncertainty_weak",
    "weak_supervision",
    "self_supervised"
]
SUPPORTED_QUERY_STRATEGIES = [
    "uncertainty",
    "diversity",
    "hybrid",
    "k_center",
    "feature"
]

__all__ = [
    # Main classes
    "ActiveLearningSystem",
    "ColdStartStrategies",
    "QueryStrategies",
    "MaskRCNNModel",
    "WeakModel",
    "FeatureExtractor",
    
    # Utilities
    "setup_logging",
    "save_checkpoint",
    "load_checkpoint",
    "calculate_metrics",
    "compute_query_efficiency",
    "create_experiment_summary",
    "save_results_json",
    "load_results_json",
    
    # Visualization
    "ActiveLearningVisualizer",
    
    # Configuration
    "ActiveLearningConfig",
    "SUPPORTED_DATASETS",
    "SUPPORTED_COLD_START_STRATEGIES",
    "SUPPORTED_QUERY_STRATEGIES",
]