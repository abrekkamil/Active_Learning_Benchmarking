import os
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
)

from src.networks.unet import UNetExact
from src.utils import set_seed


# ============================================================
# Policy Network (REINFORCE)
# ============================================================
class PolicyNet(nn.Module):
    """
    Given state vectors [N, D], outputs logits [N].
    Softmax over logits defines a distribution over candidate samples.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)  # [N]


# ============================================================
# Feature extractor for diversity init (ResNet18)
# ============================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models

        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # [B,512,1,1]
        return x.view(x.size(0), -1)  # [B,512]


# ============================================================
# RL-based Active Learning for Segmentation
# ============================================================
class ActiveLearningSegmentationRL:
    """
    Reinforcement Learning Active Learning using REINFORCE.

    - Oracle model: frozen, used for state construction
    - Main model: trained on labeled set
    - Policy: learns to select samples maximizing ΔF1
    """

    def __init__(self, train_dataset, args, device: torch.device):
        self.train_dataset = train_dataset
        self.args = args
        self.device = device

        set_seed(args.seed)

        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = list(range(len(train_dataset)))

        # Models
        self.oracle_model = UNetExact(
            in_channels=3,
            out_channels=2,
            norm=args.norm
        ).to(device)

        self.main_model = UNetExact(
            in_channels=3,
            out_channels=2,
            norm=args.norm
        ).to(device)

        # State = bottleneck(1024) + uncertainty(3)
        self.state_dim = 1027
        self.policy = PolicyNet(
            self.state_dim,
            hidden_dim=args.policy_hidden
        ).to(device)

        self.policy_opt = optim.Adam(
            self.policy.parameters(),
            lr=args.policy_lr
        )

        # REINFORCE baseline
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9

        self.prev_f1: Optional[float] = None

    # ========================================================
    # Cold start: diversity initialization
    # ========================================================
    def diversity_based_initialization(self, initial_percentage: float = 0.1) -> List[int]:
        print(f"Selecting {initial_percentage*100:.1f}% using diversity sampling")

        extractor = FeatureExtractor().to(self.device)
        extractor.eval()

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.feature_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        features = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc="Extracting features"):
                images = images.to(self.device, non_blocking=True)
                f = extractor(images).cpu().numpy()
                features.append(f)

        features = np.vstack(features)

        n_init = max(1, int(len(features) * initial_percentage))
        kmeans = MiniBatchKMeans(
            n_clusters=n_init,
            random_state=self.args.seed,
            batch_size=4096,
            n_init=3,
        )
        kmeans.fit(features)

        dists = euclidean_distances(features, kmeans.cluster_centers_)
        selected = []

        for c in range(n_init):
            order = np.argsort(dists[:, c])
            for idx in order:
                if int(idx) not in selected:
                    selected.append(int(idx))
                    break

        self.labeled_indices = selected
        self.unlabeled_indices = [
            i for i in range(len(self.train_dataset))
            if i not in set(selected)
        ]

        print(f"Initialized with {len(self.labeled_indices)} labeled samples")
        return self.labeled_indices

    # ========================================================
    # Training utilities
    # ========================================================
    def _train_model(self, model, labeled_indices: List[int], epochs: int):
        subset = Subset(self.train_dataset, labeled_indices)
        loader = DataLoader(
            subset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        model.train()
        for _ in range(epochs):
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

    def train_oracle_model(self):
        print("Training oracle model")
        self._train_model(
            self.oracle_model,
            self.labeled_indices,
            self.args.oracle_epochs
        )

    def train_main_model(self):
        print(f"Training main model on {len(self.labeled_indices)} samples")
        self._train_model(
            self.main_model,
            self.labeled_indices,
            self.args.cycle_epochs
        )

    # ========================================================
    # State construction
    # ========================================================
    @staticmethod
    def _uncertainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=[1, 2])
        confidence = probs.max(dim=1).values.mean(dim=[1, 2])
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).mean(dim=[1, 2])
        return torch.stack([entropy, 1 - confidence, 1 - margin], dim=1)

    def compute_states(self, indices: List[int]) -> torch.Tensor:
        self.oracle_model.eval()

        subset = Subset(self.train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.args.state_batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )

        states = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                feats = self.oracle_model.get_bottleneck_features(images)
                logits = self.oracle_model(images)
                unc = self._uncertainty_from_logits(logits)
                states.append(torch.cat([feats, unc], dim=1).cpu())

        return torch.cat(states, dim=0)

    # ========================================================
    # Evaluation
    # ========================================================
    def evaluate_main_model(self, val_loader) -> Dict[str, float]:
        self.main_model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                logits = self.main_model(images)
                p = torch.argmax(logits, dim=1)

                preds.append(p.cpu().numpy().ravel())
                targets.append(masks.cpu().numpy().ravel())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        return {
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
            "f1": f1_score(targets, preds, zero_division=0),
            "iou": jaccard_score(targets, preds, zero_division=0),
            "accuracy": accuracy_score(targets, preds),
        }

    # ========================================================
    # RL selection + update
    # ========================================================
    def select_with_policy(self, budget: int):
        if len(self.unlabeled_indices) == 0:
            return [], None, None

        pool_size = min(self.args.candidate_pool, len(self.unlabeled_indices))
        candidates = np.random.choice(
            self.unlabeled_indices,
            size=pool_size,
            replace=False
        ).tolist()

        states = self.compute_states(candidates).to(self.device)
        logits = self.policy(states)
        probs = F.softmax(logits / self.args.policy_temp, dim=0)

        k = min(budget, pool_size)
        chosen_pos = torch.multinomial(probs, k, replacement=False)
        chosen_probs = probs[chosen_pos].clamp_min(1e-12)

        log_prob_sum = torch.log(chosen_probs).sum()
        entropy = -(probs * torch.log(probs + 1e-12)).sum()

        chosen_indices = [candidates[i] for i in chosen_pos.cpu().tolist()]
        return chosen_indices, log_prob_sum, entropy

    def policy_update(self, reward: float, logp: torch.Tensor, entropy: torch.Tensor):
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline
            + (1 - self.baseline_momentum) * reward
        )

        advantage = reward - self.reward_baseline
        loss = -(advantage * logp) - self.args.entropy_beta * entropy

        self.policy_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()

    # ========================================================
    # One AL cycle
    # ========================================================
    def run_cycle(self, val_loader):
        new_indices, logp, entropy = self.select_with_policy(self.args.al_budget)
        if not new_indices:
            return None

        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [
            i for i in self.unlabeled_indices if i not in set(new_indices)
        ]

        self.train_main_model()
        metrics = self.evaluate_main_model(val_loader)

        reward = 0.0 if self.prev_f1 is None else metrics["f1"] - self.prev_f1
        self.prev_f1 = metrics["f1"]

        self.policy_update(reward, logp, entropy)
        return metrics
