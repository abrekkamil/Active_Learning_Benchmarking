import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Any

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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        self.prev_f1 = None

        self.logger = setup_logging(f"{config.dataset_name}_RL")

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
        self.history: List[Dict[str, float]] = []

        self.logger.info(
            f"RL AL initialized with {len(self.labeled_indices)} labeled samples"
        )

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
    def query(self, budget: int) -> List[int]:
        if len(self.unlabeled_indices) == 0:
            return []

        pool = np.random.choice(
            self.unlabeled_indices,
            size=min(len(self.unlabeled_indices), getattr(self.config, "candidate_pool", 128)),
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

        states = torch.cat(states, dim=0)

        states = states.detach()        
        logits = self.policy(states)

        probs = F.softmax(logits / self.policy_temp, dim=0)

        selected_pos = torch.multinomial(
            probs, num_samples=min(budget, len(pool)), replacement=False
        )

        self.log_prob_sum = torch.log(probs[selected_pos]).sum()
        self.entropy = -(probs * torch.log(probs + 1e-8)).sum()

        selected_indices = [pool[i] for i in selected_pos.tolist()]
        return selected_indices

    # ==========================================================
    # One AL cycle
    # ==========================================================
    def run_cycle(self):
        # Query
        new_indices = self.query(self.config.query_size)

        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [
            i for i in self.unlabeled_indices if i not in new_indices
        ]

        # Train
        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)
        for ep in range(self.config.epochs_per_cycle):
            self.main_model.train_epoch(labeled_dataset, ep, self.config.epochs_per_cycle)

        # Evaluate
        metrics = self.main_model.evaluate(self.dataset_val)
        self.logger.info(
            f"[VAL] Dice={metrics['dice']:.4f} | "
            f"IoU={metrics.get('f1', 0):.4f} | "
            f"IoU={metrics.get('mean_iou', 0):.4f} | "
            f"PixelAcc={metrics.get('pixel_acc', 0):.4f} | "
            f"Labeled={len(self.labeled_indices)}"
        )
        f1 = metrics["f1"]

        reward = 0.0 if self.prev_f1 is None else f1 - self.prev_f1
        self.prev_f1 = f1

        # Policy update
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * reward
        )

        # Policy update ONLY if a query actually happened
        if hasattr(self, "log_prob_sum") and self.log_prob_sum is not None:

            advantage = reward - self.reward_baseline
            loss = -(advantage * self.log_prob_sum) - self.entropy_beta * self.entropy

            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

        else:
            self.logger.info("No policy update (no query this cycle)")


        metrics["reward"] = reward
        metrics["labeled"] = len(self.labeled_indices)
        logged_metrics = {
            "cycle": len(self.history),
            **metrics,
            "reward": reward,
            "labeled": len(self.labeled_indices),
        }

        self.history.append(logged_metrics)
        if self.config.use_wandb:
            wandb.log(
                {
                    "val/dice": metrics["dice"],
                    "val/iou": metrics.get("mean_iou", 0),
                    "val/pixel_acc": metrics.get("pixel_acc", 0),
                    "rl/reward": reward,
                    "pool/labeled": len(self.labeled_indices),
                },
                step=len(self.history),
            )
        return metrics

    # ==========================================================
    # Full run
    # ==========================================================
    def run(self):
        self.logger.info("Starting RL Active Learning")

        # Warm-up
        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)
        for ep in range(self.config.initial_training_epoch):
            self.main_model.train_epoch(labeled_dataset, ep)

        self.prev_dice = self.main_model.evaluate(self.dataset_val)["dice"]

        for cycle in range(self.config.al_cycles):
            self.logger.info(f"RL Cycle {cycle + 1}")
            self.run_cycle()

        return self.history
