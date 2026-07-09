import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from typing import List, Optional
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
class QueryStrategies:
    """Implement various active learning query strategies."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_uncertainty(self, model, dataset, indices, device):

        model.eval()

        uncertainties = []

        for i in range(0, len(indices), self.config.batch_size):

            batch_indices = indices[i:i+self.config.batch_size]

            images = []
            for idx in batch_indices:
                sample = dataset[idx]
                image = sample[0] if isinstance(sample, (tuple, list)) else sample
                images.append(image.to(device))

            u = model.get_uncertainty(images)
            uncertainties.extend(u)

        return uncertainties

    
    def select_samples(self, strategy_name, uncertainties, dataset, indices, query_size, model=None, device=None, cycle=0, history=None):
        """Select samples using specified strategy."""
        strategy_map = {
            'uncertainty': self.select_by_uncertainty,
            'diversity': self.select_by_diversity,
            'hybrid': self.select_hybrid,
            "damage_adaptive_batch": self.select_damage_adaptive_batch,
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown query strategy: {strategy_name}")
        
        if strategy_name == "damage_adaptive_batch":
            return self.select_damage_adaptive_batch(
                uncertainties=uncertainties,
                dataset=dataset,
                indices=indices,
                query_size=query_size,
                model=model,
                device=device,
                cycle=cycle,
                history=history or {},
            )
        return strategy_map[strategy_name](
            uncertainties, dataset, indices, query_size
        )
    
    def select_by_uncertainty(self, uncertainties, dataset, indices, query_size):
        """Select samples with highest uncertainty."""
        query_size = min(query_size, len(uncertainties))
        return np.argsort(uncertainties)[-query_size:].tolist()
    
    def select_by_diversity(self, uncertainties, dataset, indices, query_size):
        """Select diverse samples using CoreSet approach."""
        # Extract features
        features = self._extract_features(dataset, indices)
        
        if len(features) == 0:
            return self.select_by_uncertainty(uncertainties, dataset, indices, query_size)
        
        # Calculate distance matrix
        distances = pairwise_distances(features, features, metric='euclidean')
        
        # Greedy selection
        selected_indices = [np.random.randint(0, len(features))]
        
        for _ in range(1, query_size):
            min_distances = np.min(distances[selected_indices, :], axis=0)
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return selected_indices
    
    def select_hybrid(self, uncertainties, dataset, indices, query_size, alpha=0.5):
        """Hybrid selection combining uncertainty and diversity."""
        from sklearn.preprocessing import minmax_scale
        
        # Extract features
        features = self._extract_features(dataset, indices)
        
        if len(features) == 0:
            return self.select_by_uncertainty(uncertainties, dataset, indices, query_size)
        
        # Normalize uncertainties
        norm_uncertainties = minmax_scale(uncertainties)
        
        # Calculate diversity scores
        distances = pairwise_distances(features, metric='euclidean')
        diversity_scores = np.mean(distances, axis=1)
        norm_diversity = minmax_scale(diversity_scores)
        
        # Combine scores
        combined_scores = alpha * norm_uncertainties + (1 - alpha) * norm_diversity
        
        # Select samples with highest combined scores
        selected_indices = np.argsort(combined_scores)[-query_size:].tolist()
        
        return selected_indices
    
    # ------------------------------------------------------------------
    # Damage-Aware Adaptive Batch selection for quick DeepCrack testing
    # ------------------------------------------------------------------
    def select_damage_adaptive_batch(
        self,
        uncertainties,
        dataset,
        indices,
        query_size,
        model=None,
        device=None,
        cycle=0,
        history=None,
    ):
        """
        DeepCrack-focused adaptive batch acquisition.

        It combines:
        1) top-k pixel entropy:
        sparse-crack uncertainty, not background-dominated mean entropy;

        2) image-edge-weighted entropy:
        uncertain crack-like structures;

        3) predicted-crack probability:
        avoids selecting only empty/background images;

        4) predicted-boundary uncertainty:
        asks for hard crack boundaries;

        5) feature novelty + MMR:
        avoids selecting near-duplicate images.

        Ground-truth masks are NOT used here.
        All damage-aware scores come from the current model prediction and image gradients.
        """

        query_size = min(int(query_size), len(indices))

        if query_size <= 0:
            return []

        # Fallback safety.
        # If this is not a segmentation wrapper with wrapper.model, use hybrid.
        if model is None or not hasattr(model, "model"):
            return self.select_hybrid(uncertainties, dataset, indices, query_size)

        uncertainties = np.asarray(uncertainties, dtype=np.float32)
        n_pool = len(indices)

        candidate_pool = getattr(
            self.config,
            "da_candidate_pool",
            getattr(self.config, "candidate_pool", 256),
        )

        candidate_pool = min(max(query_size, int(candidate_pool)), n_pool)

        # First prefilter to likely useful candidates.
        # This keeps the method fast for quick testing.
        if getattr(self.config, "da_prefilter_uncertain", True):
            cand_local = np.argsort(uncertainties)[-candidate_pool:]
        else:
            cand_local = np.arange(n_pool)

        scores = self._deepcrack_candidate_scores(
            model=model,
            dataset=dataset,
            local_indices=cand_local,
            device=device,
        )

        weights, mmr_lambda = self._adaptive_deepcrack_weights(
            history or {},
            cycle,
        )

        utility = (
            weights["uncertainty"] * self._robust_norm(scores["top_entropy"])
            + weights["edge"] * self._robust_norm(scores["edge_entropy"])
            + weights["foreground"] * self._robust_norm(scores["top_crack_prob"])
            + weights["boundary"] * self._robust_norm(scores["boundary_entropy"])
            + weights["novelty"] * self._robust_norm(scores["novelty"])
        )

        selected_rel = self._mmr_select(
            utility=utility,
            features=scores["features"],
            query_size=query_size,
            mmr_lambda=mmr_lambda,
        )

        selected_local = cand_local[selected_rel].astype(int).tolist()

        # Useful for logging/debugging.
        self.last_damage_debug = {
            "weights": {k: float(v) for k, v in weights.items()},
            "mmr_lambda": float(mmr_lambda),
            "candidate_pool": int(candidate_pool),
            "selected_utility": [float(utility[i]) for i in selected_rel],
            "selected_local_indices": selected_local,
        }

        return selected_local


    def _deepcrack_candidate_scores(
        self,
        model,
        dataset,
        local_indices,
        device=None,
    ):
        """
        Compute damage-aware scores for candidate images.

        This function does NOT use the ground-truth masks.
        """

        wrapper = model
        net = wrapper.model
        net.eval()

        if device is None:
            device = getattr(
                wrapper,
                "device",
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        batch_size = int(
            getattr(
                self.config,
                "da_score_batch",
                getattr(self.config, "batch_size", 8),
            )
        )

        top_frac = float(getattr(self.config, "da_topk_fraction", 0.10))
        top_frac = min(max(top_frac, 0.01), 0.50)

        all_top_entropy = []
        all_edge_entropy = []
        all_top_crack_prob = []
        all_boundary_entropy = []
        all_features = []

        with torch.no_grad():

            for start in range(0, len(local_indices), batch_size):

                batch_ids = local_indices[start:start + batch_size]

                images = []

                for li in batch_ids:

                    sample = dataset[int(li)]

                    if isinstance(sample, (tuple, list)):
                        img = sample[0]
                    else:
                        img = sample

                    if img.dim() == 2:
                        img = img.unsqueeze(0)

                    if img.shape[0] == 1:
                        img = img.repeat(3, 1, 1)
                    elif img.shape[0] > 3:
                        img = img[:3]

                    images.append(img)

                x = torch.stack(images, dim=0).to(device)

                logits = net(x)

                # Safety for models that return dicts/tuples.
                if isinstance(logits, dict):
                    logits = logits.get("out", next(iter(logits.values())))

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                probs = F.softmax(logits, dim=1)

                # For binary DeepCrack segmentation:
                # class 0 = background
                # class 1 = crack
                p_crack = probs[:, 1]

                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

                flat_entropy = entropy.flatten(1)
                flat_p = p_crack.flatten(1)

                k = max(1, int(flat_entropy.shape[1] * top_frac))

                # Top-k entropy:
                # mean only over the most uncertain pixels.
                # This prevents background pixels from dominating the score.
                top_entropy = torch.topk(
                    flat_entropy,
                    k=k,
                    dim=1,
                ).values.mean(dim=1)

                # Top-k predicted crack probability:
                # asks whether the image contains crack-like regions.
                top_crack_prob = torch.topk(
                    flat_p,
                    k=k,
                    dim=1,
                ).values.mean(dim=1)

                # Edge map from the input image.
                # High edge entropy = uncertain around crack-like structures.
                image_gray = x.mean(dim=1)
                image_edge = self._sobel_magnitude(image_gray)

                # Boundary map from predicted crack probability.
                # High boundary entropy = ambiguous predicted crack boundary.
                pred_boundary = self._sobel_magnitude(p_crack)

                edge_entropy = (entropy * image_edge).flatten(1).sum(dim=1) / (
                    image_edge.flatten(1).sum(dim=1) + 1e-8
                )

                boundary_entropy = (entropy * pred_boundary).flatten(1).sum(dim=1) / (
                    pred_boundary.flatten(1).sum(dim=1) + 1e-8
                )

                # Use U-Net bottleneck features if available.
                # Your UNetExact has get_bottleneck_features().
                if hasattr(net, "get_bottleneck_features"):
                    feats = net.get_bottleneck_features(x)
                else:
                    # Fallback: very weak feature, but keeps code safe.
                    feats = F.adaptive_avg_pool2d(logits, 1).flatten(1)

                all_top_entropy.append(top_entropy.cpu().numpy())
                all_edge_entropy.append(edge_entropy.cpu().numpy())
                all_top_crack_prob.append(top_crack_prob.cpu().numpy())
                all_boundary_entropy.append(boundary_entropy.cpu().numpy())
                all_features.append(feats.detach().cpu().numpy())

        features = np.concatenate(all_features, axis=0).astype(np.float32)
        features = self._standardize_features(features)

        # Novelty is distance from the candidate feature centre.
        # Later we can improve this by measuring distance from the labelled set.
        center = features.mean(axis=0, keepdims=True)
        novelty = np.linalg.norm(features - center, axis=1)

        return {
            "top_entropy": np.concatenate(all_top_entropy).astype(np.float32),
            "edge_entropy": np.concatenate(all_edge_entropy).astype(np.float32),
            "top_crack_prob": np.concatenate(all_top_crack_prob).astype(np.float32),
            "boundary_entropy": np.concatenate(all_boundary_entropy).astype(np.float32),
            "novelty": novelty.astype(np.float32),
            "features": features,
        }


    def _adaptive_deepcrack_weights(
        self,
        history,
        cycle,
    ):
        """
        Simple adaptive controller for the quick test.

        Later, replace this function with a learned RL policy that outputs:
        - acquisition weights
        - MMR lambda
        - maybe dynamic query budget
        """

        precision = self._last_history_value(
            history,
            "val_precision",
            default=0.5,
        )

        recall = self._last_history_value(
            history,
            "val_recall",
            default=0.5,
        )

        # If recall is worse than precision, the model is missing cracks.
        # So we should select more crack-like / edge-like candidates.
        recall_gap = max(0.0, precision - recall)

        # If precision is worse than recall, the model is over-predicting cracks.
        # So we should select more boundary-hard cases.
        precision_gap = max(0.0, recall - precision)

        cycle_frac = min(1.0, max(0.0, float(cycle) / 5.0))

        weights = {
            "uncertainty": float(getattr(self.config, "da_w_uncertainty", 1.0)),

            # Increase edge score when recall is poor.
            "edge": float(getattr(
                self.config,
                "da_w_edge",
                0.8 + 1.2 * recall_gap,
            )),

            # Increase crack-likelihood when recall is poor.
            "foreground": float(getattr(
                self.config,
                "da_w_foreground",
                0.4 + 0.8 * recall_gap,
            )),

            # Increase boundary cases when precision is poor.
            "boundary": float(getattr(
                self.config,
                "da_w_boundary",
                0.5 + 1.0 * precision_gap,
            )),

            # Increase novelty slowly across cycles.
            "novelty": float(getattr(
                self.config,
                "da_w_novelty",
                0.2 + 0.3 * cycle_frac,
            )),
        }

        mmr_lambda = float(getattr(self.config, "da_mmr_lambda", 0.45))

        return weights, mmr_lambda


    def _mmr_select(
        self,
        utility,
        features,
        query_size,
        mmr_lambda=0.45,
    ):
        """
        Greedy utility selection with cosine-similarity redundancy penalty.

        It selects high-scoring samples but penalises samples similar to
        already-selected samples.
        """

        utility = np.asarray(utility, dtype=np.float32)

        features = self._standardize_features(
            np.asarray(features, dtype=np.float32)
        )

        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        feats = features / norms

        selected = []
        remaining = set(range(len(utility)))

        while remaining and len(selected) < query_size:

            if not selected:

                best = int(max(remaining, key=lambda i: utility[i]))

            else:

                rem = np.array(list(remaining), dtype=int)

                sim_to_selected = feats[rem] @ feats[np.array(selected)].T
                redundancy = sim_to_selected.max(axis=1)

                mmr_score = utility[rem] - mmr_lambda * redundancy

                best = int(rem[int(np.argmax(mmr_score))])

            selected.append(best)
            remaining.remove(best)

        return selected


    def _sobel_magnitude(
        self,
        img_bhw,
    ):
        """
        Sobel magnitude for a [B,H,W] tensor, normalized per image.
        """

        if img_bhw.dim() != 3:
            raise ValueError(f"Expected [B,H,W], got {img_bhw.shape}")

        x = img_bhw.unsqueeze(1)

        kx = torch.tensor(
            [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)

        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8).squeeze(1)

        denom = mag.flatten(1).amax(dim=1).view(-1, 1, 1) + 1e-8

        return mag / denom


    def _robust_norm(
        self,
        x,
    ):
        """
        Percentile normalization.

        This is safer than min-max because one extreme candidate
        will not dominate the whole score.
        """

        x = np.asarray(x, dtype=np.float32)

        if x.size == 0:
            return x

        lo, hi = np.percentile(x, [5, 95])

        if hi <= lo + 1e-8:
            return np.zeros_like(x, dtype=np.float32)

        return np.clip(
            (x - lo) / (hi - lo + 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)


    def _standardize_features(
        self,
        x,
    ):
        """
        Standardize feature matrix.
        """

        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2 or x.shape[0] == 0:
            return x

        mu = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8

        return (x - mu) / std


    def _last_history_value(
        self,
        history,
        key,
        default=0.0,
    ):
        """
        Safely read the latest metric value from history.
        """

        vals = history.get(key, []) if isinstance(history, dict) else []

        if not vals:
            return float(default)

        try:
            return float(vals[-1])
        except Exception:
            return float(default)


    def _extract_features(self, dataset, indices):
        """Extract features from images for clustering"""
        features = []
        
        # Use a pretrained model for feature extraction

        
        # Load pretrained model
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification layer
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Define transforms for tensor input (since your dataset already returns tensors)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with torch.no_grad():
            for idx in indices:
                image, _ = dataset[idx]  # Get image, ignore mask
                
                # Your dataset already returns tensors, so handle them appropriately
                if isinstance(image, torch.Tensor):
                    # Ensure image has 3 channels (RGB)
                    if image.shape[0] == 1:  # Grayscale
                        image = image.repeat(3, 1, 1)
                    elif image.shape[0] > 3:  # If there are extra channels, take first 3
                        image = image[:3, :, :]
                    
                    # Apply transforms
                    image_tensor = transform(image).unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        image_tensor = image_tensor.cuda()
                    
                    # Extract features
                    feature = model(image_tensor)
                    feature = feature.view(feature.size(0), -1).cpu().numpy()
                    features.append(feature[0])
                else:
                    # Fallback in case some images aren't tensors
                    print(f"Warning: Unexpected image type {type(image)} for index {idx}")
                    continue
        
        return np.array(features)
