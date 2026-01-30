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
        """
        Calculate uncertainty scores for a list of dataset indices.

        This function is MODEL-AGNOSTIC.
        It delegates uncertainty computation to the model wrapper.
        """

        model.eval()

        images = []
        for idx in indices:
            sample = dataset[idx]
            image = sample[0] if isinstance(sample, (tuple, list)) else sample
            images.append(image.to(device))

        # 🔑 delegate to model
        uncertainties = model.get_uncertainty(images)

        return uncertainties

    
    def select_samples(self, strategy_name, uncertainties, dataset, indices, query_size):
        """Select samples using specified strategy."""
        strategy_map = {
            'uncertainty': self.select_by_uncertainty,
            'diversity': self.select_by_diversity,
            'hybrid': self.select_hybrid,
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown query strategy: {strategy_name}")
        
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
