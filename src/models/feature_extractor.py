
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


from .utils import _ensure_rgb

class FeatureExtractor:
    """
    Extract deep features using pretrained CNNs.
    Returns tensor of shape [N, out_dim]
    """

    def __init__(self, model_name: str = "resnet18", pretrained: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet18(weights=weights)
            self.out_dim = 512
        elif model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet50(weights=weights)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def extract(self, images: List[torch.Tensor]) -> torch.Tensor:
        feats = []

        with torch.no_grad():
            for image in images:
                image = _ensure_rgb(image).unsqueeze(0)  # [1,3,H,W]
                image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
                image = image.to(self.device)

                feat = self.model(image)          # [1,C,1,1]
                feat = feat.view(1, -1).cpu()     # [1,C]
                feats.append(feat)

        return torch.cat(feats, dim=0)  # [N,C]
