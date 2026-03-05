"""
Base model interface for Active Learning Benchmark.

All models must implement the same API so the Active Learning
pipeline can interact with them without knowing the architecture.

Required methods:
    train_epoch(dataset, epoch, total_epochs)
    evaluate(dataset)
    predict(images)
    get_uncertainty(images)
    save(path)
    load(path)
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import torch
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models used in the benchmark.
    """

    def __init__(self, num_classes: int, device: torch.device, config):
        self.num_classes = num_classes
        self.device = device
        self.config = config

        # Helps pipeline choose correct dataset / metrics
        self.task_type = "unknown"

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    @abstractmethod
    def train_epoch(self, dataset, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Train model for one epoch.

        Returns
        -------
        Dict[str, float]
            Dictionary of training metrics
        """
        pass

    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------

    @abstractmethod
    def evaluate(self, dataset) -> Dict[str, float]:
        """
        Evaluate model on validation dataset.
        """
        pass

    # -------------------------------------------------------
    # Prediction
    # -------------------------------------------------------

    @abstractmethod
    def predict(self, images: List[torch.Tensor]):
        """
        Run inference on list of images.

        Returns
        -------
        predictions
            Format depends on task type.
        """
        pass

    # -------------------------------------------------------
    # Active learning uncertainty
    # -------------------------------------------------------

    @abstractmethod
    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Compute uncertainty score for each image.

        Returns
        -------
        np.ndarray shape [N]
        """
        pass

    # -------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------

    @abstractmethod
    def save(self, path: str):
        """
        Save model checkpoint.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load model checkpoint.
        """
        pass

    # -------------------------------------------------------
    # Optional passthrough helpers
    # -------------------------------------------------------

    def train(self):
        """Switch model to training mode."""
        if hasattr(self, "model"):
            self.model.train()

    def eval(self):
        """Switch model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()