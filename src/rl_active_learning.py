import torch
import torch.nn.functional as F
import numpy as np
import time
import datetime
from typing import List, Dict, Tuple, Optional
import os
import json

from torch.utils.data import DataLoader, Subset

from .models import UNetModel, PolicyNet
from .utils import setup_logging, set_seed
from .data_modules.factory import load_dataset
from .cold_start_strategies import ColdStartStrategies
import wandb


class ActiveLearningSystemRL:
    """
    Reinforcement Learning–based Active Learning system
    Compatible with ActiveLearningConfig and existing datasets.
    """

    def __init__(self, config,  skip_cold_start: bool = False):
        self.config = config
        set_seed(config.seed)
        self.cycle = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        self.prev_f1 = None

        self.logger = setup_logging(f"{config.experiment_name}_RL")
        self._init_results_path()
        # --------------------
        # Datasets
        # --------------------
        self.dataset_train = load_dataset(config, split="train")
        self.dataset_val   = load_dataset(config, split="val")

        # --------------------
        # Models
        # --------------------
        self.oracle_model = UNetModel(
            num_classes=config.num_classes,
            device=self.device,
            config=config,
        )

        self.main_model = UNetModel(
            num_classes=config.num_classes,
            device=self.device,
            config=config,
        )

        # --------------------
        # RL policy
        # --------------------
        self.state_dim = 1024 + 3   # bottleneck + uncertainty
        self.policy = PolicyNet(self.state_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=getattr(config, "policy_lr", 1e-4),
        )

        self.entropy_beta = getattr(config, "entropy_beta", 1e-3)
        self.policy_temp = getattr(config, "policy_temp", 1.0)

        # --------------------
        # Pools
        # --------------------
        all_indices = list(range(len(self.dataset_train)))

        if skip_cold_start:
            # FULL DATASET (upper bound)
            self.labeled_indices = all_indices
            self.unlabeled_indices = []

            self.logger.info(
                "RL AL initialized with FULL dataset (skip cold start)"
            )

        else:
            n_init = (
                int(config.initial_labeled * len(all_indices))
                if config.initial_labeled <= 1
                else int(config.initial_labeled)
            )

            cold_start = ColdStartStrategies(self.dataset_train, config)

            self.labeled_indices = cold_start.apply(
                strategy_name=config.cold_start_strategy,
                n_samples=n_init,
                all_indices=all_indices,
            )

            self.unlabeled_indices = [
                i for i in all_indices if i not in self.labeled_indices
            ]

            self.logger.info(
                f"RL AL initialized with {len(self.labeled_indices)} labeled "
                f"and {len(self.unlabeled_indices)} unlabeled samples"
            )


        # --------------------
        # Tracking
        # --------------------
        self.prev_dice = None
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9
        self.history = {}

        self.logger.info(
            f"RL AL initialized with {len(self.labeled_indices)} labeled samples"
        )
        # --------------------
        # Train oracle model (ONCE)
        # --------------------
        self.logger.info("Training oracle model on initial labeled set")

        oracle_dataset = Subset(self.dataset_train, self.labeled_indices)
        for ep in range(self.config.oracle_epochs):
            self.oracle_model.train_epoch(
                oracle_dataset, ep, self.config.oracle_epochs
            )

        self.oracle_model.eval()
        for p in self.oracle_model.model.parameters():
            p.requires_grad = False
    # ==========================================================
    # Feature + uncertainty → state
    # ==========================================================
    def _compute_state(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,H,W]
        returns: [B, 1027]
        """
        with torch.no_grad():
            feats = self.oracle_model.model.get_bottleneck_features(images)
            logits = self.oracle_model.model(images)

            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=[1, 2])
            confidence = probs.max(dim=1).values.mean(dim=[1, 2])
            margin = torch.topk(probs, 2, dim=1).values
            margin = (margin[:, 0] - margin[:, 1]).mean(dim=[1, 2])

            uncertainty = torch.stack(
                [entropy, 1.0 - confidence, 1.0 - margin], dim=1
            )

            return torch.cat([feats, uncertainty], dim=1)

    # ==========================================================
    # RL query step
    # ==========================================================
    def query(self, budget: int) -> Tuple[List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(self.unlabeled_indices) == 0:
            return [], None, None

        if self.config.query_size <= 1:
            budget = int(self.config.query_size * len(self.dataset_train))
        pool = np.random.choice(
            self.unlabeled_indices,
            size=len(self.unlabeled_indices),
            replace=False,
        ).tolist()

        subset = Subset(self.dataset_train, pool)
        loader = DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        states = []
        for images, _ in loader:
            images = images.to(self.device)
            states.append(self._compute_state(images))

        states = torch.cat(states, dim=0).to(self.device)

        states = states.detach()        
        logits = self.policy(states)

        probs = F.softmax(logits / self.policy_temp, dim=0)

        selected_pos = torch.multinomial(
            probs, num_samples=min(budget, len(pool)), replacement=False
        )

        chosen = probs[selected_pos].clamp_min(1e-12)
        log_prob_sum = torch.log(chosen).sum()
        entropy = -(probs * torch.log(probs + 1e-8)).sum()

        selected_indices = [pool[i] for i in selected_pos.tolist()]
        return selected_indices, log_prob_sum, entropy

    # ==========================================================
    # One AL cycle
    # ==========================================================
    def run_cycle(self):
        # Query
        new_indices, log_prob_sum, entropy = self.query(self.config.query_size)

        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [
            i for i in self.unlabeled_indices if i not in new_indices
        ]

        # Train
        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)
        for ep in range(self.config.epochs_per_cycle):
            train_metrics = self.main_model.train_epoch(labeled_dataset, ep, self.config.epochs_per_cycle)
            eval_metrics = self.main_model.evaluate(self.dataset_val)

            self._log_metrics(
                epoch=ep,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

            self.save_results()

        f1 = eval_metrics["f1"]

        reward = 0.0 if self.prev_f1 is None else f1 - self.prev_f1
        self.prev_f1 = f1

        # Policy update ONLY if a query actually happened
        if log_prob_sum is not None:
            advantage = reward - self.reward_baseline
            # Policy update
            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline
                + (1 - self.baseline_momentum) * reward
            )
            loss = -(advantage * log_prob_sum) - self.entropy_beta * entropy

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.policy_optimizer.step()

        else:
            self.logger.info("No policy update (no query this cycle)")

        self.logger.info("Reward: {:.4f} | Baseline: {:.4f} | Advantage: {:.4f}".format(
            reward, self.reward_baseline, advantage))
        
        self.cycle += 1

    # ==========================================================
    # Full run
    # ==========================================================
    def run(self):
        self.logger.info("Starting RL Active Learning")

        # Warm-up
        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)
        for ep in range(self.config.initial_training_epoch):
            train_metrics = self.main_model.train_epoch(labeled_dataset, ep, self.config.initial_training_epoch)

            eval_metrics = self.main_model.evaluate(self.dataset_val)

            self._log_metrics(
                epoch=ep,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

            self.save_results()
        self.cycle += 1

        self.prev_f1 = self.main_model.evaluate(self.dataset_val)["f1"]

        for cycle in range(self.config.al_cycles):
            self.logger.info(f"\n=== Reinforcement AL Cycle {cycle + 1}/{self.config.al_cycles} ===")
            self.run_cycle()

        return self.history


    def _log_reward(self, reward=None):
        self.history.setdefault("Reward", []).append(reward)
        self.logger.info(f"=== Reward {reward} ===")

    def _log_metrics(self, epoch, train_metrics, eval_metrics):
        global_epoch = epoch + self.cycle * self.config.epochs_per_cycle

        self.history.setdefault("epoch", []).append(epoch)
        self.history.setdefault("global_epoch", []).append(global_epoch)
        self.history.setdefault("cycle", []).append(self.cycle)

        self.history.setdefault("train_loss", []).append(train_metrics["train_loss"])
        self.history.setdefault("val_dice", []).append(eval_metrics["dice"])
        self.history.setdefault("val_F1", []).append(eval_metrics["f1"])
        self.history.setdefault("val_mean_iou", []).append(eval_metrics["mean_iou"])
        self.history.setdefault("labeled_count", []).append(len(self.labeled_indices))

        self.logger.info(
            f"Epoch {epoch+1} | "
            f"Loss: {train_metrics['train_loss']:.4f} | "
            f"F1: {eval_metrics.get('f1', 0):.4f} | "
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
        results = {
            "config": self._config_to_dict(),
            "history": self.history,
        }

        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)

    def _init_results_path(self):
        date_folder = datetime.datetime.now().strftime("%m_%d")
        results_dir = os.path.join(self.config.results_dir, date_folder)
        os.makedirs(results_dir, exist_ok=True)

        time_stamp = datetime.datetime.now().strftime("%H%M")

        self.results_path = os.path.join(
            results_dir,
            f"{self.config.experiment_name}_"
            f"{self.config.dataset_type}_"
            f"{self.config.cold_start_strategy}_"
            f"{self.config.query_strategy}_"
            f"{time_stamp}.json"
        )
    def _config_to_dict(self):
        # works for argparse.Namespace or simple config objects
        return vars(self.config)
    
