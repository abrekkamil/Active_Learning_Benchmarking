"""
Model definitions for active learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights


class UNetModel:
    """
    Wrapper for U-Net semantic segmentation.
    """

    def __init__(self, num_classes, device, config):
        self.num_classes = num_classes
        self.device = device
        self.config = config

        from networks.unet import UNetExact

        self.model = UNetExact(
            in_channels=3,
            out_channels=num_classes,
            norm=getattr(config, "unet_norm", "bn"),
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataset, epoch):
        self.model.train()

        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        total_loss = 0.0
        start = time.time()

        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {
            "train_loss": total_loss / len(loader),
            "training_time": time.time() - start,
        }

    def evaluate(self, dataset):
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        total_iou = 0.0
        count = 0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)

                intersection = ((preds == 1) & (masks == 1)).sum().item()
                union = ((preds == 1) | (masks == 1)).sum().item()

                if union > 0:
                    total_iou += intersection / union
                    count += 1

        return {"mean_iou": total_iou / max(count, 1)}

    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            images = torch.stack(images).to(self.device)
            logits = self.model(images)
            return torch.argmax(logits, dim=1)

    def get_uncertainty(self, images):
        """
        Pixel entropy averaged over image.
        """
        self.model.eval()
        uncertainties = []

        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0).to(self.device)
                logits = self.model(img)
                probs = F.softmax(logits, dim=1)

                entropy = -torch.sum(
                    probs * torch.log(probs + 1e-8),
                    dim=1
                ).mean()

                uncertainties.append(entropy.item())

        return np.array(uncertainties)

class MaskRCNNModel:
    """
    Wrapper for Mask R-CNN model with training and evaluation capabilities.
    """
    
    def __init__(self, num_classes: int, device: torch.device, config):
        """
        Initialize Mask R-CNN model.
        
        Args:
            num_classes: Number of classes (including background)
            device: PyTorch device (cuda or cpu)
            config: Configuration object
        """
        self.num_classes = num_classes
        self.device = device
        self.config = config
        
        # Initialize model
        self.model = self._create_model()
        
        # Optimizer
        self.optimizer = torch.optim.SGD(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_steps,
            gamma=0.1
        )
        
        # Training statistics
        self.train_losses = []
        self.val_metrics = []
        
    def _create_model(self):
        """Create and initialize Mask R-CNN model."""
        import pytorch_mask_rcnn as pmr
        
        # Load pretrained Mask R-CNN
        model = pmr.maskrcnn_resnet50(
            pretrained=True,
            num_classes=self.num_classes
        ).to(self.device)
        
        return model
    
    def train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            dataset: Training dataset
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        import pytorch_mask_rcnn as pmr
        
        self.model.train()
        
        # Set learning rate for this epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr_epoch if hasattr(self.config, 'lr_epoch') else self.config.lr
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        epoch_losses = []
        start_time = time.time()
        
        # Train one epoch using pytorch_mask_rcnn's training function
        args = self.config
        args.lr_epoch = self.config.lr_epoch if hasattr(self.config, 'lr_epoch') else self.config.lr
        
        # We'll use the pmr.train_one_epoch function
        # This assumes pmr.train_one_epoch returns losses
        train_stats = pmr.train_one_epoch(
            self.model,
            self.optimizer,
            dataset,
            self.device,
            epoch,
            args
        )
        
        # Update learning rate scheduler
        self.lr_scheduler.step()
        
        training_time = time.time() - start_time
        
        # Parse and return training metrics
        metrics = {
            'training_time': training_time,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Add specific losses if available
        if isinstance(train_stats, dict):
            for key, value in train_stats.items():
                if hasattr(value, 'item'):
                    metrics[f'train_{key}'] = value.item()
        
        return metrics
    
    def evaluate(self, dataset) -> Dict[str, float]:
        """
        Evaluate model on validation dataset.
        
        Args:
            dataset: Validation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        import pytorch_mask_rcnn as pmr
        
        self.model.eval()
        
        args = self.config
        
        # Use pmr.evaluate function
        eval_output, _, metrics = pmr.evaluate(
            self.model,
            dataset,
            self.device,
            0,  # epoch (not used for evaluation)
            args
        )
        
        # Flatten metrics dictionary for easier logging
        flattened_metrics = {}
        if metrics:
            for category, category_metrics in metrics.items():
                for metric_name, metric_value in category_metrics.items():
                    flattened_metrics[f"{category}_{metric_name}"] = metric_value
        
        # Add overall AP if available
        if metrics and "bbox" in metrics:
            flattened_metrics["AP@[IoU=0.50:0.95]"] = metrics["bbox"]["AP@[IoU=0.50:0.95]"]
        
        return flattened_metrics
    
    def predict(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions on input images.
        
        Args:
            images: List of input images as tensors
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for image in images:
                
                image = image.to(self.device)
                output = self.model(image)
                
                # Convert to CPU and numpy for easier handling
                prediction = {
                    'boxes': output['boxes'].cpu(),
                    'labels': output['labels'].cpu(),
                    'scores': output['scores'].cpu(),
                    'masks': output['masks'].cpu() if 'masks' in output else None
                }
                predictions.append(prediction)
        
        return predictions
    
    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Calculate uncertainty scores for input images.
        
        Args:
            images: List of input images
            
        Returns:
            Array of uncertainty scores
        """
        self.model.eval()
        
        uncertainties = []
        with torch.no_grad():
            for image in images:
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                
                image = image.to(self.device)
                output = self.model(image)
                
                # Calculate uncertainty based on detection scores
                if len(output["scores"]) > 0:
                    scores = output['scores']
                    probs = F.softmax(scores, dim=0)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    uncertainties.append(entropy.item())
                else:
                    # No detections - high uncertainty
                    uncertainties.append(1.0)
        
        return np.array(uncertainties)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'num_classes': self.num_classes
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    
    def get_features(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Extract features from images using the model's backbone.
        
        Args:
            images: List of input images
            
        Returns:
            Feature vectors
        """
        self.model.eval()
        
        # Hook to extract features
        features = []
        
        def hook_fn(module, input, output):
            features.append(output.detach().cpu())
        
        # Register hook to the backbone's last layer
        if hasattr(self.model, 'backbone'):
            handle = self.model.backbone.register_forward_hook(hook_fn)
        else:
            # Try to find the backbone
            for name, module in self.model.named_modules():
                if 'backbone' in name or 'body' in name:
                    handle = module.register_forward_hook(hook_fn)
                    break
        
        # Forward pass
        with torch.no_grad():
            for image in images:
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                
                image = image.to(self.device)
                _ = self.model(image)
        
        # Remove hook
        handle.remove()
        
        # Process features
        if features:
            # Global average pooling
            pooled_features = F.adaptive_avg_pool2d(features[0], (1, 1))
            flattened = pooled_features.view(pooled_features.size(0), -1)
            return flattened.numpy()
        
        return np.array([])


class WeakModel:
    """
    Weak model for cold start uncertainty estimation.
    Uses a smaller pretrained model for initial uncertainty scoring.
    """
    
    def __init__(self, num_classes: int, device: torch.device):
        """
        Initialize weak model.
        
        Args:
            num_classes: Number of classes
            device: PyTorch device
        """
        self.num_classes = num_classes
        self.device = device
        
        # Use ResNet18 as weak model
        self.model = models.resnet18(weights=None)
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Make predictions on input images.
        
        Args:
            images: List of input images
            
        Returns:
            Model predictions
        """
        with torch.no_grad():
            # Preprocess images
            processed_images = []
            for image in images:
                # Handle different input formats
                if isinstance(image, torch.Tensor):
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    
                    # Convert to RGB if grayscale
                    if image.shape[1] == 1:
                        image = image.repeat(1, 3, 1, 1)
                    
                    # Resize and normalize
                    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
                    
                    # Normalize
                    image = self._normalize_tensor(image)
                    processed_images.append(image)
                else:
                    # Assume PIL Image or numpy array
                    import torchvision.transforms.functional as F_tv
                    if not isinstance(image, torch.Tensor):
                        image = F_tv.to_tensor(image)
                    
                    if image.shape[0] == 1:
                        image = image.repeat(3, 1, 1)
                    
                    image = F_tv.resize(image, (224, 224))
                    image = self.transform(image).unsqueeze(0)
                    processed_images.append(image)
            
            # Batch predictions
            if processed_images:
                batch = torch.cat(processed_images, dim=0).to(self.device)
                outputs = self.model(batch)
                return outputs
            else:
                return torch.tensor([])
    
    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Calculate uncertainty scores using the weak model.
        
        Args:
            images: List of input images
            
        Returns:
            Uncertainty scores
        """
        outputs = self.predict(images)
        
        if len(outputs) == 0:
            return np.array([1.0] * len(images))
        
        # Calculate entropy as uncertainty measure
        probs = F.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        return entropy.cpu().numpy()
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to ImageNet statistics."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
        
        return (tensor - mean) / std


class FeatureExtractor:
    """
    Utility class for extracting features from images.
    Supports multiple feature types and models.
    """
    
    def __init__(self, feature_type: str = "deep", model_name: str = "resnet18"):
        """
        Initialize feature extractor.
        
        Args:
            feature_type: Type of features ("deep", "statistical", "self_supervised")
            model_name: Model architecture for deep features
        """
        self.feature_type = feature_type
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if feature_type == "deep":
            self.model = self._load_pretrained_model(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
        elif feature_type == "self_supervised":
            self.model = self._load_self_supervised_model()
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def _load_pretrained_model(self, model_name: str) -> nn.Module:
        """Load pretrained model for feature extraction."""
        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if self.use_pretrained else None

            model = models.resnet18(weights=weights)
            # Remove classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if self.use_pretrained else None
            model = models.resnet50(weights=weights)
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model
    
    def _load_self_supervised_model(self) -> nn.Module:
        """Load self-supervised model."""
        try:
            # Try to load a self-supervised model
            model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models',
                'resnet18_swsl'
            )
        except:
            # Fallback to regular pretrained
            weights = ResNet18_Weights.DEFAULT if self.use_pretrained else None

            model = models.resnet18(weights=weights)
        
        # Remove classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        return model
    
    def extract(self, images: List[torch.Tensor]) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: List of input images
            
        Returns:
            Feature vectors
        """
        if self.feature_type == "statistical":
            return self._extract_statistical_features(images)
        else:
            return self._extract_deep_features(images)
    
    def _extract_statistical_features(self, images: List[torch.Tensor]) -> np.ndarray:
        """Extract statistical features from images."""
        features = []
        
        for image in images:
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
            else:
                img_np = np.array(image)
            
            # Calculate statistical features
            if len(img_np.shape) == 3:  # RGB
                channel_means = np.mean(img_np, axis=(1, 2))
                channel_stds = np.std(img_np, axis=(1, 2))
                feature = np.concatenate([channel_means, channel_stds])
            else:  # Grayscale
                feature = np.array([np.mean(img_np), np.std(img_np)])
            
            features.append(feature)
        
        return np.array(features)
    
    def _extract_deep_features(self, images: List[torch.Tensor]) -> np.ndarray:
        """Extract deep features using neural network."""
        features = []
        
        with torch.no_grad():
            for image in images:
                # Prepare image
                if isinstance(image, torch.Tensor):
                    if image.dim() == 3:
                        image = image.unsqueeze(0)
                    
                    # Convert to RGB if needed
                    if image.shape[1] == 1:
                        image = image.repeat(1, 3, 1, 1)
                    
                    # Resize
                    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
                else:
                    # Convert PIL/numpy to tensor
                    import torchvision.transforms.functional as F_tv
                    image = F_tv.to_tensor(image)
                    if image.shape[0] == 1:
                        image = image.repeat(3, 1, 1)
                    image = F_tv.resize(image, (224, 224)).unsqueeze(0)
                
                # Normalize
                image = self._normalize_image(image)
                image = image.to(self.device)
                
                # Extract features
                feature = self.model(image)
                feature = feature.view(feature.size(0), -1).cpu().numpy()
                features.append(feature[0])
        
        return np.array(features)
    
    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to ImageNet statistics."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if image.device != mean.device:
            mean = mean.to(image.device)
            std = std.to(image.device)
        
        return (image - mean) / std