import torch
import numpy as np
import hashlib, os
from typing import List, Optional
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Subset
import logging
import time
import cv2
import torch.nn.functional as F

## TODO: Add clipIQA model for image quality assessment
class ColdStartStrategies:
    """Implement various cold start initialization strategies."""
    
    def __init__(self, dataset, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_train = dataset
        self.config = config
        self.logger = logging.getLogger(__name__)

    
    def apply(self, strategy_name: str, n_samples: int, all_indices: List[int]) -> List[int]:
        """Apply specified cold start strategy."""
        strategy_map = {
            'random': self.random_sampling,
            'simple_diversity': self.simple_diversity_sampling,
            'diversity': self.diversity_based_sampling,
            'entropy_based_uncertainty': self.entropy_based_uncertainty,
            'uncertainty_weak': self.uncertainty_sampling_weak,
            'weak_supervision': self.weak_supervision_sampling,
            'self_supervised': self.self_supervised_sampling,
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown cold start strategy: {strategy_name}")
        
        self.logger.info(f"Applying cold start strategy: {strategy_name}")
        return strategy_map[strategy_name](all_indices, n_samples)
    
    def random_sampling(self, all_indices, n_samples):
        """Random sampling (baseline)."""
        return torch.randperm(len(all_indices))[:n_samples].tolist()
    
    def _feature_cache_path(self, tag, indices):
        key = f"{self.config.dataset}_{self.config.img_size}_{tag}_{len(indices)}"
        h = hashlib.md5(str(indices).encode()).hexdigest()[:8]
        cache_dir = os.path.join(self.config.data_dir, "feature_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{key}_{h}.npy")
    
    def _extract_features_batched(self, indices, model, tag, batch_size=64, num_workers=0):
        path = self._feature_cache_path(tag, indices)
        if os.path.exists(path):
            self.logger.info(f"Loading cached features from {path}")
            return np.load(path)

        subset = Subset(self.dataset_train, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        feats = []
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                elif images.shape[1] > 3:
                    images = images[:, :3]
                images = F.interpolate(images, size=(224, 224),
                                    mode='bilinear', align_corners=False)
                images = normalize(images).to(self.device, non_blocking=True)
                feats.append(model(images).flatten(1).cpu().numpy())
                if i % 20 == 0:
                    self.logger.info(f"Feature extraction: batch {i}/{len(loader)}")

        out = np.concatenate(feats, 0)
        np.save(path, out)
        self.logger.info(f"Cached features to {path}  shape={out.shape}")
        return out


    def _extract_features(self, indices):
        """Extract deep features using pretrained model."""
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval().to(self.device)
        return self._extract_features_batched(indices, model, tag="resnet18_imagenet")


    def _extract_self_supervised_features(self, indices):
        """Extract features using a self-supervised (SWSL) backbone."""
        try:
            model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models',
                'resnet18_swsl')
        except Exception as e:
            self.logger.warning(f"SWSL hub load failed ({e}), falling back to ImageNet weights")
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval().to(self.device)
        return self._extract_features_batched(indices, model, tag="resnet18_swsl")


    def simple_diversity_sampling(self, all_indices, n_samples):
        """Simple diversity using image statistics."""
        features = []
        for idx in all_indices:
            image, _ = self.dataset_train[idx]
            img_np = image.numpy() if isinstance(image, torch.Tensor) else np.array(image)
            if img_np.ndim == 3:
                feature = np.concatenate([img_np.mean(axis=(1, 2)),
                                        img_np.std(axis=(1, 2))])
            else:
                feature = np.array([img_np.mean(), img_np.std()])
            features.append(feature)

        features = np.asarray(features, dtype=np.float32)
        return self._cluster_and_select(features, all_indices, n_samples)


    def diversity_based_sampling(self, all_indices, n_samples):
        """Diversity sampling using deep features."""
        features = self._extract_features(all_indices)
        return self._cluster_and_select(features, all_indices, n_samples)


    def _cluster_and_select(self, features, all_indices, k):
        """PCA -> MiniBatchKMeans -> pick the member closest to each centroid."""
        from sklearn.decomposition import PCA

        X = np.asarray(features, dtype=np.float32)
        self.logger.info(f"Clustering {X.shape[0]} samples with {X.shape[1]} features into {k} clusters")
        if X.shape[1] > 64:
            t = time.time()
            Xt = torch.from_numpy(X).to(self.device)
            Xt = Xt - Xt.mean(dim=0, keepdim=True)
            # economy SVD on GPU
            U, S, V = torch.pca_lowrank(Xt, q=64, center=False, niter=4)
            X = (Xt @ V[:, :64]).cpu().numpy().astype(np.float32)
            del Xt, U, S, V
            torch.cuda.empty_cache()
            self.logger.info(f"PCA (GPU) -> {X.shape} in {time.time()-t:.1f}s")

        t = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=1024,
            n_init=3,
            max_iter=100,
            max_no_improvement=10,
            reassignment_ratio=0.01,
        )
        labels = kmeans.fit_predict(X)
        self.logger.info(f"MiniBatchKMeans k={k} in {time.time()-t:.1f}s")

        selected = []
        for cid in range(k):
            member_pos = np.flatnonzero(labels == cid)
            if member_pos.size == 0:
                continue
            d = np.linalg.norm(X[member_pos] - kmeans.cluster_centers_[cid], axis=1)
            selected.append(all_indices[member_pos[int(np.argmin(d))]])

        if len(selected) < k:
            chosen = set(selected)
            pool = [i for i in all_indices if i not in chosen]
            extra = torch.randperm(len(pool))[:k - len(selected)].tolist()
            selected.extend([pool[i] for i in extra])

        self.logger.info(f"Selected {len(selected)} samples from {k} clusters")
        return selected[:k]
    
    def entropy_based_uncertainty(self, all_indices, n_samples):
        """Uncertainty sampling using image entropy."""
        self.logger.info("Using entropy-based uncertainty sampling...")
        subset = Subset(self.dataset_train, all_indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=False)

        uncertainties = []
        for i, (images, _) in enumerate(loader):
            # images: (B, C, H, W) -> greyscale (B, H*W)
            if images.shape[1] == 3:
                gray = images.mean(dim=1)
            else:
                gray = images[:, 0]
            gray = gray.flatten(1).numpy()

            for row in gray:
                hist, _ = np.histogram(row, bins=32, density=True)
                hist = hist[hist > 0]
                uncertainties.append(float(-np.sum(hist * np.log2(hist))))

            if i % 20 == 0:
                self.logger.info(f"Entropy: batch {i}/{len(loader)}")

        uncertainties = np.asarray(uncertainties)
        top = np.argsort(uncertainties)[-n_samples:]
        return [all_indices[i] for i in top]
    
    
    def uncertainty_sampling_weak(self, all_indices, n_labeled):
        """Uncertainty sampling using a weak model for cold start."""
        self.logger.info("Using weak model uncertainty sampling...")
        weak_model = self._create_weak_model()
        weak_model.eval()

        subset = Subset(self.dataset_train, all_indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)

        uncertainties = []
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                elif images.shape[1] > 3:
                    images = images[:, :3]
                images = images.to(self.device, non_blocking=True)

                logits = weak_model(images)                       # (B, num_classes)
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)   # (B,)
                uncertainties.append(ent.cpu().numpy())

                if i % 20 == 0:
                    self.logger.info(f"Weak uncertainty: batch {i}/{len(loader)}")

        uncertainties = np.concatenate(uncertainties, 0)
        top = np.argsort(uncertainties)[-n_labeled:]
        return [all_indices[i] for i in top]
    
    def _create_weak_model(self):
        """Create a weak model for initial uncertainty estimation"""
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights
        
        # Use a smaller pretrained model as weak model
        weak_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify for detection-like uncertainty (optional)
        # You can modify this based on your specific needs
        num_features = weak_model.fc.in_features
        weak_model.fc = torch.nn.Linear(num_features, self.config.num_classes)
        
        if torch.cuda.is_available():
            weak_model = weak_model.cuda()
        
        return weak_model
    
    def weak_supervision_sampling(self, all_indices, n_labeled):
        """Weak supervision using heuristic rules."""
        self.logger.info("Using weak supervision sampling...")
        subset = Subset(self.dataset_train, all_indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False,
                            num_workers=8, pin_memory=False)

        scores = []
        for i, (images, _) in enumerate(loader):
            batch_np = images.numpy()            # (B, C, H, W)
            for img in batch_np:
                scores.append(self._calculate_weak_supervision_score(img))
            if i % 20 == 0:
                self.logger.info(f"Weak supervision: batch {i}/{len(loader)}")

        scores = np.asarray(scores)
        top = np.argsort(scores)[-n_labeled:]
        return [all_indices[i] for i in top]
    
    def _calculate_weak_supervision_score(self, image):
        """Calculate weak supervision score using heuristics"""
        
        if len(image.shape) == 3:
            # RGB image
            gray = np.mean(image, axis=0)
        else:
            gray = image
        
        # Multiple heuristics for weak supervision
        heuristics = []
        
        # 1. Edge density (more edges might indicate more complex objects)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        heuristics.append(edge_density)
        
        # 2. Texture complexity (using variance)
        texture_complexity = np.var(gray)
        heuristics.append(texture_complexity)
        
        # 3. Color diversity (for RGB images)
        if len(image.shape) == 3:
            color_diversity = np.mean([np.std(image[i]) for i in range(3)])
            heuristics.append(color_diversity)
        
        # 4. Contrast
        contrast = np.max(gray) - np.min(gray)
        heuristics.append(contrast)
        
        # Combine heuristics (you can weight them differently)
        combined_score = np.mean(heuristics)
        
        return combined_score
    
    def self_supervised_sampling(self, all_indices, n_labeled):
        self.logger.info("Using self-supervised sampling...")
        features = self._extract_self_supervised_features(all_indices)
        return self._cluster_and_select(features, all_indices, n_labeled)