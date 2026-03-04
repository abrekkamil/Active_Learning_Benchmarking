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
        self.system_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config = config
        set_seed(config.seed)
        self.cycle = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        self.prev_score = None
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
        if config.model_name == "maskrcnn":
            from .models import MaskRCNNModel
            self.oracle_model = MaskRCNNModel(config.num_classes, self.device, config)
            self.main_model = MaskRCNNModel(config.num_classes, self.device, config)
        elif config.model_name == "unet":
            from .models import UNetModel
            self.oracle_model = UNetModel(config.num_classes, self.device,config)
            self.main_model = UNetModel(config.num_classes, self.device,config)
        elif config.model_name == "Deeplabv3":
            from .models import DeepLabV3Model
            self.oracle_model = DeepLabV3Model(config.num_classes, self.device, config)
            self.main_model = DeepLabV3Model(config.num_classes, self.device, config)
        else:
            raise ValueError("Unknown task")


        # --------------------
        # RL policy
        # --------------------
        # Infer bottleneck dimension dynamically
        sample_img, _ = self.dataset_train[0]
        sample_img = sample_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.oracle_model.model.get_bottleneck_features(sample_img)

        bottleneck_dim = feat.shape[1]

        self.state_dim = bottleneck_dim + 3  # +3 for uncertainty features
        self.policy = PolicyNet(
            self.state_dim,
            hidden_dim=self.config.policy_hidden,
            num_budget_options=len(self.config.budget_options),
        ).to(self.device)
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

        
    def _compute_state(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,H,W]
        returns: [B, 1027]
        """
        with torch.no_grad():
            feats = self.oracle_model.model.get_bottleneck_features(images)
            outputs = self.oracle_model.model(images)
            # Handle different model outputs
            if isinstance(outputs, dict):                 # DeepLabV3
                logits = outputs["out"]
            elif hasattr(outputs, "logits"):              # SegFormer
                logits = outputs.logits
            else:                                         # UNet
                logits = outputs


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
    def query(self, _):

        if len(self.unlabeled_indices) == 0:
            return [], None, None, None

        # ==========================================================
        # Build pool
        # ==========================================================
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

        # ==========================================================
        # Candidate Filtering
        # ==========================================================
        entropy_scores = states[:, -3]

        candidate_ratio = getattr(self.config, "candidate_ratio", 0.4)
        top_k = max(1, int(candidate_ratio * len(entropy_scores)))

        _, candidate_idx = torch.topk(entropy_scores, top_k)

        candidate_states = states[candidate_idx]
        candidate_pool = [pool[i] for i in candidate_idx.tolist()]

        # ==========================================================
        # Policy Forward
        # ==========================================================
        global_state = candidate_states.mean(dim=0)
        image_logits, budget_logits = self.policy(candidate_states, global_state)

        # ==========================================================
        # --------- DYNAMIC QUERY SIZE ------------------------------
        # ==========================================================
        if getattr(self.config, "dynamic_query_size", False):

            budget_ratio = torch.sigmoid(budget_logits)
            budget_ratio = torch.clamp(budget_ratio, 0.05, 0.5)

            budget = int(budget_ratio.item() * len(candidate_pool))
            budget = max(1, budget)

            log_prob_budget = torch.log(budget_ratio + 1e-12)

            p = budget_ratio
            entropy_budget = -(p * torch.log(p + 1e-12) +
                            (1 - p) * torch.log(1 - p + 1e-12))

        # ==========================================================
        # --------- FIXED QUERY SIZE -------------------------------
        # ==========================================================
        else:

            budget = self.config.query_size
            budget = min(budget, len(candidate_pool))

            # no gradient from budget in fixed mode
            log_prob_budget = torch.tensor(0.0, device=self.device)
            entropy_budget = torch.tensor(0.0, device=self.device)

        # ==========================================================
        # Image Sampling
        # ==========================================================
        image_probs = F.softmax(
            image_logits.squeeze() / self.policy_temp,
            dim=0
        )
        image_probs = image_probs.clamp_min(1e-12)

        selected_pos = torch.multinomial(
            image_probs,
            num_samples=budget,
            replacement=False,
        )

        log_prob_images = torch.log(image_probs[selected_pos]).sum()

        log_prob_sum = log_prob_images + log_prob_budget

        entropy_images = -(image_probs * torch.log(image_probs)).sum()

        entropy = entropy_images + entropy_budget

        selected_indices = [
            candidate_pool[i] for i in selected_pos.tolist()
        ]

        return selected_indices, log_prob_sum, entropy, budget
    
    # ==========================================================
    # One AL cycle
    # ==========================================================
    def run_cycle(self):
        # Query
        self.policy_temp = max(
        self.config.policy_temp_end,
        self.config.policy_temp_start * (0.95 ** self.cycle)
        )
            
        new_indices, log_prob_sum, entropy, budget = self.query(None)
        if len(new_indices) == 0:
            self.logger.info("No samples selected this cycle.")
            self.cycle += 1
            return
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [
            i for i in self.unlabeled_indices if i not in new_indices
        ]

        # Train
        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)
        for ep in range(self.config.epochs_per_cycle):
            epoch_start = time.time()
            train_metrics = self.main_model.train_epoch(labeled_dataset, ep, self.config.epochs_per_cycle)
            eval_metrics = self.main_model.evaluate(self.dataset_val)
            epoch_time = time.time() - epoch_start

            self._log_metrics(
                epoch=ep,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
                epoch_time=epoch_time,
            )

            self.save_results()

        mean_iou = eval_metrics["mean_iou"]
        dice = eval_metrics["dice"]
        f1 = eval_metrics["f1"]
        score = dice + f1 + mean_iou
        
        if self.prev_score is None:
            reward = 0.0
        else:
            reward = (score - self.prev_score) / (abs(self.prev_score) + 1e-8)
        self.prev_score = score 
        # Cost penalty
        if getattr(self.config, "dynamic_query_size", False):
            reward = reward - self.config.cost_lambda * budget
        # Policy update ONLY if a query actually happened
        if log_prob_sum is not None:
            advantage = reward - self.reward_baseline
            advantage = torch.tensor(advantage, device=self.device)
            # Policy update
            self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * reward)

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
        run_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info("Starting RL Active Learning")

        # Warm-up
        labeled_dataset = Subset(self.dataset_train, self.labeled_indices)
        for ep in range(self.config.initial_training_epoch):
            epoch_start = time.time()
            train_metrics = self.main_model.train_epoch(labeled_dataset, ep, self.config.initial_training_epoch)

            eval_metrics = self.main_model.evaluate(self.dataset_val)
            epoch_time = time.time() - epoch_start
            self._log_metrics(
                epoch=ep,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
                epoch_time=epoch_time,
            )

            self.save_results()
        self.cycle += 1

        self.prev_f1 = self.main_model.evaluate(self.dataset_val)["f1"]

        for cycle in range(self.config.al_cycles):
            self.logger.info(f"\n=== Reinforcement AL Cycle {cycle + 1}/{self.config.al_cycles} ===")
            self.run_cycle()

        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") - run_start_time
        self.logger.info(f"RL Active Learning completed in {run_time}")
        self.history["run_time"] = run_time
        system_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") - self.system_start_time
        self.logger.info(f"Total system time: {system_time}")
        self.history["system_time"] = system_time
        self.save_results()
        return self.history


    def _log_reward(self, reward=None):
        self.history.setdefault("Reward", []).append(reward)
        self.logger.info(f"=== Reward {reward} ===")

    def _log_metrics(self, epoch, train_metrics, eval_metrics, epoch_time):
        global_epoch = epoch + self.cycle * self.config.epochs_per_cycle

        self.history.setdefault("epoch", []).append(epoch)
        self.history.setdefault("global_epoch", []).append(global_epoch)
        self.history.setdefault("cycle", []).append(self.cycle)
        self.history.setdefault("epoch_time", []).append(epoch_time)
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
    
