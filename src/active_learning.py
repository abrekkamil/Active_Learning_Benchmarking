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
    
    def __init__(self, config, skip_cold_start=False):
        """Initialize active learning system."""
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        
        # Setup logging
        self.logger = setup_logging(config.experiment_name)
        
        # Load datasets
        self.num_classes = config.num_classes
        self.dataset_train = load_dataset(config, split="train")
        self.dataset_val   = load_dataset(config, split="val")
        
        # Initialize strategies
        self.cold_start = ColdStartStrategies(self.dataset_train, config)
        self.query_strategy = QueryStrategies(config)
        
        # Initialize model
        if config.task == "detection":
            self.model = MaskRCNNModel(config.num_classes, self.device, config)
        elif config.task == "segmentation":
            self.model = UNetModel(config.num_classes, self.device,config)
        else:
            raise ValueError("Unknown task")
        
        # Initialize pools
        if not skip_cold_start:
            self._init_pools()
        
        # Tracking
        self.cycle = 0
        self.best_score = 0.0
        self.history = {}
        
        print(f"Active Learning System initialized with:")
        print(f"  Device: {self.device}")
        print(f"  Cold Start Strategy: {config.cold_start_strategy}")
        print(f"  Query Strategy: {config.query_strategy}")
        if not skip_cold_start:
            print(f"  Initial labeled: {len(self.labeled_indices)} samples")
        else:
            print(f"  Skipped cold start. Full dataset will be used for training.")
            self.labeled_indices  = list(range(len(self.dataset_train)))
            self.unlabeled_indices = []
    
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
    def set_labeled_indices(self, labeled_indices: List[int]):
        """
        Manually set the initial labeled pool (override cold start).
        Useful for cold start experiments and ablations.
        """
        all_indices = list(range(len(self.dataset_train)))

        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = [
            i for i in all_indices if i not in self.labeled_indices
        ]

        self.logger.info(
            f"Manually set labeled pool: "
            f"{len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )
    def train(self, epochs: Optional[int] = None):
        if epochs is None:
            epochs = self.config.epochs_per_cycle

        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)

        print(f"\nTraining cycle {self.cycle} with {len(self.labeled_indices)} samples")

        cycle_metrics = []

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.model.train_epoch(
                labeled_dataset,
                epoch + self.cycle * epochs,
                epochs
            )

            # Evaluate
            eval_metrics = self.model.evaluate(self.dataset_val)

            epoch_time = time.time() - epoch_start
            global_epoch = epoch + self.cycle * epochs

            self.logger.info(
                f"Dice={eval_metrics['dice']:.4f} | "
                f"F1={eval_metrics.get('f1', 0):.4f} | "
                f"IoU={eval_metrics.get('mean_iou', 0):.4f} | "
                f"PixelAcc={eval_metrics.get('pixel_acc', 0):.4f} | "
                f"Labeled={len(self.labeled_indices)}"
            )

            # ✅ LOG AFTER EACH EPOCH
            self._log_metrics(
                epoch=epoch,
                train_time=epoch_time,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )
            self.save_results()
            cycle_metrics.append(eval_metrics)

            # Save checkpoint
            current_score = eval_metrics["f1"] if "f1" in eval_metrics else eval_metrics["dice"]
            if current_score > self.best_score:
                self.best_score = current_score
                save_checkpoint(
                    model=self.model,
                    cycle=self.cycle,
                    epoch=epoch,
                    score=current_score,
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
        global_epoch = epoch + self.cycle * self.config.epochs_per_cycle

        self.history.setdefault("epoch", []).append(epoch)
        self.history.setdefault("global_epoch", []).append(global_epoch)
        self.history.setdefault("cycle", []).append(self.cycle)

        self.history.setdefault("train_loss", []).append(train_metrics["train_loss"])
        self.history.setdefault("val_dice", []).append(eval_metrics["dice"])
        self.history.setdefault("val_F1", []).append(eval_metrics["f1"])
        self.history.setdefault("val_mean_iou", []).append(eval_metrics["mean_iou"])
        self.history.setdefault("labeled_count", []).append(len(self.labeled_indices))
        self.history.setdefault("train_time", []).append(train_time)

        self.logger.info(
            f"Epoch {epoch+1} | "
            f"Loss: {train_metrics['train_loss']:.4f} | "
            f"Dice: {eval_metrics['dice']:.4f} | "
            f"Mean IoU: {eval_metrics['mean_iou']:.4f} | "
            f"Labeled: {len(self.labeled_indices)}"
        )

        if self.config.use_wandb:
            log_to_wandb(
                {
                    "epoch": epoch + 1,
                    "global_epoch": global_epoch,
                    "cycle": self.cycle,
                    "train_loss": train_metrics["train_loss"],
                    "val_dice": eval_metrics["dice"],
                    "val_iou": eval_metrics["mean_iou"],
                    "labeled_count": len(self.labeled_indices),
                },
                step=global_epoch,
            )

    def save_results(self):
        results_path = os.path.join(
            self.config.results_dir,
            f"{self.config.experiment_name}_"
            f"{self.config.dataset_type}_"
            f"{self.config.cold_start_strategy}_"
            f"{self.config.query_strategy}.json"
        )

        results = {
            "config": self._config_to_dict(),
            "history": self.history,
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    def _config_to_dict(self):
        # works for argparse.Namespace or simple config objects
        return vars(self.config)
    