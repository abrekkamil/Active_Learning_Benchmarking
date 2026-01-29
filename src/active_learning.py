import torch
import numpy as np
import time
import os
import json
import glob
import re
from torch.utils.data import Subset
from typing import List, Tuple, Dict, Any, Optional

from .cold_start_strategies import ColdStartStrategies
from .query_strategies import QueryStrategies
from .models import MaskRCNNModel, UNetModel
from .utils import setup_logging, save_checkpoint, load_checkpoint
from .data_modules.factory import load_dataset


class ActiveLearningSystem:
    """
    Main active learning system combining cold start and query strategies.
    
    This class orchestrates the entire active learning process:
    1. Initialize with a cold start strategy
    2. Train model on initial labeled set
    3. Iteratively query new samples and retrain
    """
    
    def __init__(self, config):
        """Initialize active learning system."""
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        
        # Setup logging
        self.logger = setup_logging(config.dataset_name)
        
        # Load datasets
        self.num_classes = config.num_classes
        self.dataset_train = load_dataset(config, split="train")
        self.dataset_val   = load_dataset(config, split="val")
        
        # Initialize strategies
        self.cold_start = ColdStartStrategies(self.dataset_train, config)
        self.query_strategy = QueryStrategies(config)
        
        # Initialize model
        if config.task == "detection":
            self.model = MaskRCNNModel(config.num_classes, device, config)
        elif config.task == "segmentation":
            self.model = UNetModel(config.num_classes, self.device,config)
        else:
            raise ValueError("Unknown task")
        
        # Initialize pools
        self._init_pools()
        
        # Tracking
        self.cycle = 0
        self.best_ap = 0.0
        self.history = {
            'val_ap': [],
            'labeled_count': [],
            'cycle_metrics': []
        }
        
        print(f"Active Learning System initialized with:")
        print(f"  Device: {self.device}")
        print(f"  Cold Start Strategy: {config.cold_start_strategy}")
        print(f"  Query Strategy: {config.query_strategy}")
        print(f"  Initial labeled: {len(self.labeled_indices)} samples")
    
    def _load_datasets(self):
        """Load training and validation datasets."""
        # Using pytorch_mask_rcnn or custom dataset loader
        import pytorch_mask_rcnn as pmr
        
        self.dataset_train = pmr.datasets(
            self.config.dataset, 
            self.config.data_dir, 
            "train", 
            train=True
        )
        self.dataset_val = pmr.datasets(
            self.config.dataset,
            self.config.data_dir,
            "valid",
            train=True
        )
        self.num_classes = max(self.dataset_train.classes) + 1
    
    def _init_pools(self):
        """Initialize labeled and unlabeled pools using cold start strategy."""
        all_indices = list(range(len(self.dataset_train)))
        
        # Determine number of initial samples
        if 0 <= self.config.initial_labeled <= 1:
            n_labeled = int(self.config.initial_labeled * len(all_indices))
        else:
            n_labeled = min(int(self.config.initial_labeled), len(all_indices))
        
        # Apply cold start strategy
        self.labeled_indices = self.cold_start.apply(
            strategy_name=self.config.cold_start_strategy,
            n_samples=n_labeled,
            all_indices=all_indices
        )
        
        self.unlabeled_indices = [
            i for i in all_indices if i not in self.labeled_indices
        ]
        
        self.logger.info(
            f"Initialized with {len(self.labeled_indices)} labeled "
            f"and {len(self.unlabeled_indices)} unlabeled samples"
        )
    
    def train(self, epochs: Optional[int] = None):
        """Train model on current labeled set."""
        if epochs is None:
            epochs = self.config.epochs_per_cycle
        
        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)
        
        print(f"\nTraining cycle {self.cycle} with {len(self.labeled_indices)} samples")
        
        cycle_metrics = []
        for epoch in range(epochs):
            # Train one epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.model.train_epoch(
                labeled_dataset, 
                epoch + self.cycle * epochs
            )
            
            # Evaluate
            eval_metrics = self.model.evaluate(self.dataset_val)
            
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            self._log_metrics(
                epoch=epoch,
                train_time=epoch_time,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics
            )
            
            cycle_metrics.append(eval_metrics)
            
            # Save checkpoints
            current_ap = eval_metrics.get("bbox_AP@[IoU=0.50:0.95]", 0)
            if current_ap > self.best_ap:
                self.best_ap = current_ap
                save_checkpoint(
                    model=self.model,
                    cycle=self.cycle,
                    epoch=epoch,
                    ap=current_ap,
                    is_best=True,
                    config=self.config
                )
        
        self.cycle += 1
        return cycle_metrics
    
    def query(self, query_size: Optional[int] = None):
        """Query new samples using active learning strategy."""
        if query_size is None:
            query_size = self.config.query_size
        
        if len(self.unlabeled_indices) == 0:
            self.logger.warning("No unlabeled samples left!")
            return []
        
        # Get uncertainty scores
        uncertainties = self.query_strategy.calculate_uncertainty(
            model=self.model,
            dataset=self.dataset_train,
            indices=self.unlabeled_indices,
            device=self.device
        )
        
        # Select samples
        selected_indices = self.query_strategy.select_samples(
            strategy_name=self.config.query_strategy,
            uncertainties=uncertainties,
            dataset=self.dataset_train,
            indices=self.unlabeled_indices,
            query_size=query_size
        )
        
        # Update pools
        selected_global_indices = [
            self.unlabeled_indices[i] for i in selected_indices
        ]
        
        self.labeled_indices.extend(selected_global_indices)
        self.unlabeled_indices = [
            idx for i, idx in enumerate(self.unlabeled_indices)
            if i not in selected_indices
        ]
        
        self.logger.info(
            f"Selected {len(selected_global_indices)} new samples. "
            f"Now {len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )
        
        return selected_global_indices
    
    def run(self):
        """Run complete active learning process."""
        self.logger.info("Starting active learning process...")
        
        all_metrics = []
        
        # Initial training
        initial_metrics = self.train(epochs=self.config.initial_training_epoch)
        all_metrics.append(initial_metrics)
        
        # Active learning cycles
        for cycle in range(self.config.al_cycles):
            self.logger.info(f"\n=== AL Cycle {cycle + 1}/{self.config.al_cycles} ===")
            
            # Query new samples
            self.query()
            
            # Train on expanded dataset
            cycle_metrics = self.train()
            all_metrics.append(cycle_metrics)
        
        self.logger.info("Active learning completed!")
        return all_metrics
    
    def _log_metrics(self, epoch, train_time, train_metrics, eval_metrics):
        """Log training and evaluation metrics."""
        # Update history
        current_ap = eval_metrics.get("bbox_AP@[IoU=0.50:0.95]", 0)
        self.history['val_ap'].append(current_ap)
        self.history['labeled_count'].append(len(self.labeled_indices))
        
        # Log to console
        print(f"Epoch {epoch + 1}:")
        print(f"  Training time: {train_time:.1f}s")
        print(f"  AP@[IoU=0.50:0.95]: {current_ap:.4f}")
        
        # Log to WandB if enabled
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "cycle": self.cycle,
                "epoch": epoch + 1,
                "train_time": train_time,
                "labeled_count": len(self.labeled_indices),
                **train_metrics,
                **eval_metrics
            })