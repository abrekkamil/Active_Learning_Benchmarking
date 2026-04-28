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
from .utils import setup_logging, set_seed, universal_collate
from .data_modules.factory import load_dataset
from .cold_start_strategies import ColdStartStrategies
from .models import build_model
from .models.utils import _ensure_rgb

import wandb

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_train, dataset_pool, samples):
        self.dataset_train = dataset_train
        self.dataset_pool = dataset_pool
        self.samples = samples  # list of ("train"/"pool", idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        source, idx = self.samples[i]

        if source == "train":
            return self.dataset_train[idx]

        elif source == "pool":
            return self.dataset_pool[idx]

        else:
            raise ValueError(f"Unknown source: {source}")
        

class ActiveLearningSystemRL:
    """
    Reinforcement Learning–based Active Learning system
    Compatible with ActiveLearningConfig and existing datasets.
    """

    def __init__(self, config,  skip_cold_start: bool = False):
        self.system_start_time = datetime.datetime.now()
        self.config = config
        set_seed(config.seed)
        self.cycle = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_cuda else "cpu"
        )
        self.prev_score = None

        self.logger = setup_logging(f"{config.experiment_name}_RL")
        self._init_results_path()
        # --------------------
        # Datasets
        # --------------------
        self.dataset_train = load_dataset(config, split="train")
        self.dataset_val   = load_dataset(config, split="val")
        self.dataset_pool = None
        if config.pool:
            self.dataset_pool = load_dataset(config, split="pool")
        # --------------------
        # Models
        # --------------------
        self.oracle_model = build_model(
            config.model_name,
            num_classes=config.num_classes,
            device=self.device,
            config=config,
        )
        self.main_model = build_model(
            config.model_name,
            num_classes=config.num_classes,
            device=self.device,
            config=config,
        )


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
        train_indices = list(range(len(self.dataset_train)))
        pool_indices = list(range(len(self.dataset_pool))) if config.pool else []
        all_samples = [("train", i) for i in train_indices] + \
              [("pool", i) for i in pool_indices]
        
        if skip_cold_start:
            # FULL DATASET (upper bound)
            self.labeled_indices = all_samples
            self.unlabeled_indices = []

            self.logger.info(
                "RL AL initialized with FULL dataset (skip cold start)"
            )

        else:
            n_init = (
                int(config.initial_labeled * len(train_indices))
                if config.initial_labeled <= 1
                else int(config.initial_labeled)
            )

            cold_start = ColdStartStrategies(self.dataset_train, config)

            labeled_train = cold_start.apply(
                strategy_name=config.cold_start_strategy,
                n_samples=n_init,
                all_indices=train_indices
            )
            self.labeled_indices = [("train", i) for i in labeled_train]

            self.unlabeled_indices = [
                s for s in all_samples if s not in self.labeled_indices
            ]

            self.logger.info(
                f"RL AL initialized with {len(self.labeled_indices)} labeled "
                f"and {len(self.unlabeled_indices)} unlabeled samples"
            )


        # --------------------
        # Tracking
        # --------------------
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

        oracle_dataset = MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.labeled_indices
        )
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
    def set_labeled_indices(self, labeled_indices: List[Tuple[str, int]]):
        """
        Manually set the initial labeled pool (override cold start).
        Useful for cold start experiments and ablations.
        """
        all_samples = [("train", i) for i in range(len(self.dataset_train))]
        if self.config.pool:
            all_samples += [("pool", i) for i in range(len(self.dataset_pool))]

        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = [
            i for i in all_samples if i not in self.labeled_indices
        ]

        self.logger.info(
            f"Manually set labeled pool: "
            f"{len(self.labeled_indices)} labeled, "
            f"{len(self.unlabeled_indices)} unlabeled"
        )
    def get_primary_metric(self, task, metrics):

        if task == "segmentation":
            return metrics.get("f1", 0)

        if task == "instance_segmentation":
            return metrics.get("mask_AP", 0)

        if task == "detection":
            return metrics.get("bbox_AP", 0)
        if task == "multilabel_classification":
            return metrics.get("macro_f1", metrics.get("map", 0))
        

        return 0

    def _get_sample_name(self, source, idx):

        dataset = self.dataset_train if source == "train" else self.dataset_pool

        if hasattr(dataset, "coco"):
            img_id = dataset.ids[idx]
            return dataset.coco.imgs[img_id]["file_name"]

        return f"{source}_{idx}"   

    def forward_model(self, images):
        try:
            # HuggingFace-style (SegFormer)
            return self.oracle_model.model(pixel_values=images)
        except TypeError:
            # Torchvision / U-Net style
            return self.oracle_model.model(images)

    def _compute_state(self, images: torch.Tensor):

        with torch.no_grad():

            feats = self.oracle_model.model.get_bottleneck_features(images).detach()

            outputs = self.forward_model(images)

            # =========================
            # INSTANCE SEGMENTATION
            # =========================
            if self.config.task == "instance_segmentation":

                entropy_list = []
                confidence_list = []
                margin_list = []

                for out in outputs:

                    scores = out["scores"]

                    if len(scores) == 0:
                        entropy_list.append(torch.tensor(0.0, device=self.device))
                        confidence_list.append(torch.tensor(0.0, device=self.device))
                        margin_list.append(torch.tensor(0.0, device=self.device))
                        continue
                    
                    denom = scores.sum() + 1e-8
                    probs = scores / denom

                    entropy = -(probs * torch.log(probs + 1e-8)).sum()

                    confidence = scores.mean()

                    if len(scores) > 1:
                        top2 = torch.topk(scores, 2, dim=1).values
                        margin = (top2[:, 0] - top2[:, 1]).mean(dim=[1,2])
                    else:
                        margin = scores[0]

                    entropy_list.append(entropy)
                    confidence_list.append(confidence)
                    margin_list.append(margin)

                entropy = torch.stack(entropy_list)
                confidence = torch.stack(confidence_list)
                margin = torch.stack(margin_list)


            elif self.config.task == "multilabel_classification":
 
                # outputs is [B, num_classes] raw logits
                logits = outputs
                probs  = torch.sigmoid(logits)             # [B, C]  independent
 
                # Mean binary entropy across all labels → [B]
                eps = 1e-8
                entropy    = -(probs * torch.log(probs + eps)
                               + (1 - probs) * torch.log(1 - probs + eps)).mean(dim=1)
 
                # Mean confidence = mean of max(p, 1-p) across classes → [B]
                confidence = torch.max(probs, 1 - probs).values.mean(dim=1)
 
                # Margin = distance from 0.5, averaged across classes → [B]
                margin = (probs - 0.5).abs().mean(dim=1)

                
            # =========================
            # SEMANTIC SEGMENTATION
            # =========================
            else:
                if isinstance(outputs, dict):
                    logits = outputs["out"]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    raise ValueError(f"Unknown model output format: {type(outputs)}")
                
                probs = F.softmax(logits, dim=1)

                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=[1,2])

                confidence = probs.max(dim=1).values.mean(dim=[1,2])

                margin = torch.topk(probs, 2, dim=1).values
                margin = (margin[:,0] - margin[:,1]).mean(dim=[1,2])

            uncertainty = torch.stack(
                [entropy, 1.0 - confidence, 1.0 - margin],
                dim=1
            )

            return torch.cat([feats, uncertainty], dim=1)
    # ==========================================================
    # RL query step
    # ==========================================================
    def query(self):

        if len(self.unlabeled_indices) == 0:
            return [], None, None, None

        # ==========================================================
        # Build pool
        # ==========================================================
        unlabeled_dataset = MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.unlabeled_indices
        )

        loader = DataLoader(
            unlabeled_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=universal_collate
        )

        states = []

        with torch.no_grad():

            for images, _ in loader:

                images = [_ensure_rgb(img) for img in images]
                images = torch.stack(images).to(self.device)

                states.append(self._compute_state(images))

        states = torch.cat(states, dim=0).detach()

        # ==========================================================
        # Candidate Filtering
        # ==========================================================
        entropy_scores = states[:, -3]

        candidate_ratio = getattr(self.config, "candidate_ratio", 0.2)
        top_k = int(candidate_ratio * len(entropy_scores))
        top_k = max(1, min(top_k, len(entropy_scores)))

        _, candidate_idx = torch.topk(entropy_scores, top_k)

        candidate_states = states[candidate_idx]

        candidate_pool = [
            self.unlabeled_indices[i] for i in candidate_idx.tolist()
        ]

        # ==========================================================
        # Policy Forward
        # ==========================================================
        global_state = candidate_states.mean(dim=0)
        image_logits, budget_logits = self.policy(candidate_states, global_state)

        # ==========================================================
        # Query size
        # ==========================================================
        if getattr(self.config, "dynamic_query_size", False):

            budget_ratio = torch.sigmoid(budget_logits)
            budget_ratio = torch.clamp(budget_ratio, 0.01, 0.1)

            budget = int(budget_ratio.item() * len(candidate_pool))
            budget = max(1, budget)

            log_prob_budget = torch.log(budget_ratio + 1e-12)

            p = budget_ratio
            entropy_budget = -(p * torch.log(p + 1e-12) +
                            (1 - p) * torch.log(1 - p + 1e-12))

        else:

            if self.config.query_size <= 1:
                budget = int(self.config.query_size * len(candidate_pool))
            else:
                budget = int(self.config.query_size)

            budget = max(1, min(budget, len(candidate_pool)))

            log_prob_budget = torch.tensor(0.0, device=self.device)
            entropy_budget = torch.tensor(0.0, device=self.device)

        # ==========================================================
        # Image Sampling
        # ==========================================================
        budget = int(budget)
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
        selected_samples_info = []

        for source, idx in selected_indices:

            name = self._get_sample_name(source, idx)

            selected_samples_info.append({
                "source": source,
                "index": idx,
                "name": name
    })
        self.logger.info(f"Selected {len(selected_indices)} samples with budget {budget}")
        self.logger.info(f"Selected samples: {selected_samples_info}")
        self.history.setdefault("selected_samples", []).append(selected_samples_info)
        n_train = sum(1 for s,_ in selected_indices if s == "train")
        n_pool = sum(1 for s,_ in selected_indices if s == "pool")
        self.logger.info(
            f"Selected {len(selected_indices)} samples "
            f"({n_train} train, {n_pool} pool)"
        )

        self.history.setdefault("selected_train_count", []).append(n_train)
        self.history.setdefault("selected_pool_count", []).append(n_pool)
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
            
        new_indices, log_prob_sum, entropy, budget = self.query()
        if len(new_indices) == 0:
            self.logger.info("No samples selected this cycle.")
            self.cycle += 1
            return
        self.labeled_indices.extend(new_indices)
        new_set = set(new_indices)
        self.unlabeled_indices = [
            s for s in self.unlabeled_indices if s not in new_set
        ]

        # TrainSubset
        labeled_dataset = MixedDataset(
                self.dataset_train,
                self.dataset_pool,
                self.labeled_indices
            )
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

        score = self.get_primary_metric(self.config.task, eval_metrics)
        
        if self.prev_score is None:
            reward = 0.0
        else:
            reward = score - self.prev_score
            reward = float(np.clip(reward, -0.1, 0.1))
        self.prev_score = score 
        # Cost penalty
        if getattr(self.config, "dynamic_query_size", False):
            reward = reward - self.config.cost_lambda * (budget / len(self.unlabeled_indices))
        # Policy update ONLY if a query actually happened
        advantage = torch.tensor(0.0, device=self.device)
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
        run_start_time = datetime.datetime.now()
        self.logger.info("Starting RL Active Learning")

        # Warm-up
        labeled_dataset = MixedDataset(
            self.dataset_train,
            self.dataset_pool,
            self.labeled_indices
        )
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

        self.prev_score = self.get_primary_metric(self.config.task, eval_metrics)
        for cycle in range(self.config.al_cycles):
            self.logger.info(f"\n=== Reinforcement AL Cycle {cycle + 1}/{self.config.al_cycles} ===")
            self.run_cycle()

        run_time = str(datetime.datetime.now() - run_start_time)
        self.logger.info(f"RL Active Learning completed in {run_time}")
        self.history["run_time"] = run_time
        system_time = str(datetime.datetime.now() - self.system_start_time)
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
        self.history.setdefault("labeled_count", []).append(len(self.labeled_indices))


        # ==========================================
        # Semantic segmentation
        # ==========================================
        if self.config.task == "segmentation":

            f1 = eval_metrics.get("f1", 0)
            dice = eval_metrics.get("dice", 0)
            miou = eval_metrics.get("mean_iou", 0)

            self.history.setdefault("val_F1", []).append(f1)
            self.history.setdefault("val_dice", []).append(dice)
            self.history.setdefault("val_mean_iou", []).append(miou)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics['train_loss']:.4f} | "
                f"F1: {f1:.4f} | "
                f"Dice: {dice:.4f} | "
                f"Mean IoU: {miou:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

            if self.config.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics["train_loss"],
                        "val_F1": f1,
                        "val_dice": dice,
                        "val_mean_iou": miou,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )


        # ==========================================
        # Instance segmentation
        # ==========================================
        elif self.config.task in ["instance_segmentation", "detection"]:

            mask_ap = eval_metrics.get("mask_AP", 0)
            bbox_ap = eval_metrics.get("bbox_AP", 0)

            self.history.setdefault("val_mask_AP", []).append(mask_ap)
            self.history.setdefault("val_bbox_AP", []).append(bbox_ap)

            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics['train_loss']:.4f} | "
                f"Mask AP: {mask_ap:.4f} | "
                f"BBox AP: {bbox_ap:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )

            if self.config.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics["train_loss"],
                        "val_mask_AP": mask_ap,
                        "val_bbox_AP": bbox_ap,
                        "labeled_count": len(self.labeled_indices),
                    },
                    step=global_epoch,
                )
        # ==========================================
        # Multi-label classification
        # ==========================================
        elif self.config.task == "multilabel_classification":
 
            macro_f1 = eval_metrics.get("macro_f1", 0)
            micro_f1 = eval_metrics.get("micro_f1", 0)
            map_score = eval_metrics.get("map", 0)
            hamming  = eval_metrics.get("hamming_loss", 0)
 
            self.history.setdefault("val_macro_f1", []).append(macro_f1)
            self.history.setdefault("val_micro_f1", []).append(micro_f1)
            self.history.setdefault("val_map", []).append(map_score)
            self.history.setdefault("val_hamming_loss", []).append(hamming)
 
            self.logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {train_metrics['train_loss']:.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Micro-F1: {micro_f1:.4f} | "
                f"mAP: {map_score:.4f} | "
                f"Hamming: {hamming:.4f} | "
                f"Labeled: {len(self.labeled_indices)}"
            )
 
            if self.config.use_wandb:
                log_to_wandb(
                    {
                        "epoch": epoch + 1,
                        "global_epoch": global_epoch,
                        "cycle": self.cycle,
                        "train_loss": train_metrics["train_loss"],
                        "val_macro_f1": macro_f1,
                        "val_micro_f1": micro_f1,
                        "val_map": map_score,
                        "val_hamming_loss": hamming,
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
        self.logger.info(f"the results will be saved in: {self.results_path}")
    def _config_to_dict(self):
        # works for argparse.Namespace or simple config objects
        return vars(self.config)
    
