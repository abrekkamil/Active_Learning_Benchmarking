import torch
import numpy as np
from typing import List, Optional
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.cluster import KMeans
import cv2
import torch.nn.functional as F

## TODO: Add clipIQA model for image quality assessment
class ColdStartStrategies:
    """Implement various cold start initialization strategies."""
    
    def __init__(self, dataset, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_train = dataset
        self.config = config
    
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
        
        print(f"Applying cold start strategy: {strategy_name}")
        return strategy_map[strategy_name](all_indices, n_samples)
    
    def random_sampling(self, all_indices, n_samples):
        """Random sampling (baseline)."""
        return torch.randperm(len(all_indices))[:n_samples].tolist()
    
    def simple_diversity_sampling(self, all_indices, n_samples):
        """Simple diversity using image statistics."""
        features = []
        for idx in all_indices:
            image, _ = self.dataset_train[idx]
            
            if isinstance(image, torch.Tensor):
                img_np = image.numpy()
            else:
                img_np = np.array(image)
            
            # Calculate simple statistics
            if len(img_np.shape) == 3:
                feature = np.concatenate([
                    img_np.mean(axis=(0, 1)),
                    img_np.std(axis=(0, 1))
                ])
            else:
                feature = np.array([img_np.mean(), img_np.std()])
            
            features.append(feature)
        
        features = np.array(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select one sample per cluster
        selected_indices = []
        for cluster_id in range(n_samples):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_samples = [all_indices[i] for i in range(len(all_indices)) 
                             if cluster_mask[i]]
            
            if cluster_samples:
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_features = features[cluster_mask]
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                selected_indices.append(cluster_samples[closest_idx])
        
        return selected_indices
    
    def diversity_based_sampling(self, all_indices, n_samples):
        """Diversity sampling using deep features."""
        features = self._extract_features(all_indices)
        
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        selected_indices = []
        for cluster_id in range(n_samples):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_samples = [all_indices[i] for i in range(len(all_indices)) 
                             if cluster_mask[i]]
            
            if cluster_samples:
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_features = features[cluster_mask]
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                selected_indices.append(cluster_samples[closest_idx])
        
        return selected_indices
    
    def entropy_based_uncertainty(self, all_indices, n_samples):
        """Uncertainty sampling using image entropy."""
        uncertainties = []
        for idx in all_indices:
            image, _ = self.dataset_train[idx]
            
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
            else:
                img_np = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_np.shape) == 3:
                img_gray = np.mean(img_np, axis=0)
            else:
                img_gray = img_np
            
            # Calculate entropy
            hist, _ = np.histogram(img_gray, bins=32, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            uncertainties.append(entropy)
        
        # Select most uncertain samples
        uncertainties = np.array(uncertainties)
        selected_indices = np.argsort(uncertainties)[-n_samples:].tolist()
        return [all_indices[i] for i in selected_indices]
    
    def _extract_features(self, indices):
        """Extract deep features using pretrained model."""
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        features = []
        with torch.no_grad():
            for idx in indices:
                image, _ = self.dataset_train[idx]
                
                if isinstance(image, torch.Tensor):
                    if image.shape[0] == 1:  # Grayscale
                        image = image.repeat(3, 1, 1)
                    image_tensor = transform(image).unsqueeze(0)
                    
                    if torch.cuda.is_available():
                        image_tensor = image_tensor.cuda()
                    
                    feature = model(image_tensor)
                    feature = feature.view(feature.size(0), -1).cpu().numpy()
                    features.append(feature[0])
        
        return np.array(features)
    
    def uncertainty_sampling_weak(self, all_indices, n_labeled):
        """Uncertainty sampling using a weak model for cold start"""
        print("Using weak model uncertainty sampling...")
        
        # Create a simple weak model for initial uncertainty estimation
        weak_model = self._create_weak_model()
        weak_model.eval()
        
        uncertainties = []
        
        with torch.no_grad():
            for idx in all_indices:
                image, _ = self.dataset_train[idx]
                
                # Prepare image for weak model
                if isinstance(image, torch.Tensor):
                    image_tensor = image.unsqueeze(0).to(self.device)
                else:
                    # Convert PIL to tensor if needed
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Get predictions from weak model
                weak_output = weak_model(image_tensor)
                
                # Calculate uncertainty using entropy
                if hasattr(weak_output, 'scores'):  # For detection models
                    scores = weak_output['scores']
                    if len(scores) > 0:
                        probs = F.softmax(scores, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                        uncertainties.append(entropy.item())
                    else:
                        uncertainties.append(1.0)  # High uncertainty if no detections
                else:  # For classification models
                    probs = F.softmax(weak_output, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    uncertainties.append(entropy.item())
        
        # Select most uncertain samples
        uncertainties = np.array(uncertainties)
        selected_indices = np.argsort(uncertainties)[-n_labeled:].tolist()
        
        return [all_indices[i] for i in selected_indices]
    def _create_weak_model(self):
        """Create a weak model for initial uncertainty estimation"""
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights
        
        # Use a smaller pretrained model as weak model
        weak_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify for detection-like uncertainty (optional)
        # You can modify this based on your specific needs
        num_features = weak_model.fc.in_features
        weak_model.fc = torch.nn.Linear(num_features, self.num_classes)
        
        if torch.cuda.is_available():
            weak_model = weak_model.cuda()
        
        return weak_model
    
    def weak_supervision_sampling(self, all_indices, n_labeled):
        """Weak supervision using heuristic rules or pre-trained features"""
        print("Using weak supervision sampling...")
        
        scores = []
        
        for idx in all_indices:
            image, _ = self.dataset_train[idx]
            
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
            else:
                img_np = np.array(image)
            
            # Calculate weak supervision score based on heuristics
            score = self._calculate_weak_supervision_score(img_np)
            scores.append(score)
        
        # Select samples with highest weak supervision scores
        scores = np.array(scores)
        selected_indices = np.argsort(scores)[-n_labeled:].tolist()
        
        return [all_indices[i] for i in selected_indices]
    
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
        """Self-supervised sampling using contrastive learning features"""
        print("Using self-supervised sampling...")
        
        # Extract self-supervised features
        features = self._extract_self_supervised_features(all_indices)
        
        # Use clustering to select diverse samples
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_labeled, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select one sample from each cluster
        selected_indices = []
        for cluster_id in range(n_labeled):
            cluster_samples = [all_indices[i] for i in range(len(all_indices)) 
                             if cluster_labels[i] == cluster_id]
            
            if cluster_samples:
                # Select sample closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_features = features[cluster_labels == cluster_id]
                
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                selected_indices.append(cluster_samples[closest_idx])
        
        # If we need more samples than clusters, add random ones
        if len(selected_indices) < n_labeled:
            remaining = n_labeled - len(selected_indices)
            remaining_indices = [i for i in all_indices if i not in selected_indices]
            if remaining_indices:
                additional = torch.randperm(len(remaining_indices))[:remaining].tolist()
                selected_indices.extend([remaining_indices[i] for i in additional])
        
        return selected_indices[:n_labeled]
    
    def _extract_self_supervised_features(self, indices):
        """Extract features using self-supervised learning"""
        import torchvision.models as models
        import torchvision.transforms as transforms
        
        # Load a self-supervised model (SimCLR, MoCo, etc.)
        # For this example, we'll use a pretrained model that has seen similar data
        try:
            # Try to load a self-supervised model
            model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
        except:
            # Fallback to regular pretrained model
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification layer
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        features = []
        with torch.no_grad():
            for idx in indices:
                image, _ = self.dataset_train[idx]
                
                if isinstance(image, torch.Tensor):
                    if image.shape[0] == 1:  # Grayscale
                        image = image.repeat(3, 1, 1)
                    elif image.shape[0] > 3:
                        image = image[:3, :, :]
                    
                    image_tensor = transform(image).unsqueeze(0)
                else:
                    # Handle PIL Image
                    transform_pil = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    image_tensor = transform_pil(image).unsqueeze(0)
                
                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()
                
                feature = model(image_tensor)
                feature = feature.view(feature.size(0), -1).cpu().numpy()
                features.append(feature[0])
        
        return np.array(features)