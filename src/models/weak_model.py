import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights

from .utils import _ensure_rgb

class WeakModel:
    """
    Small classification model used for uncertainty scoring in cold start.

    Input: list of images [C,H,W]
    Output: entropy over logits
    """

    def __init__(self, num_classes: int, device: torch.device, pretrained: bool = True):
        self.device = device
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.model = tv_models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device).eval()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        img = _ensure_rgb(img)
        img = img.unsqueeze(0)  # [1,3,H,W]
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        return img.to(self.device)

    def predict(self, images: List[torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        xs = torch.cat([self._prep(im) for im in images], dim=0)
        with torch.no_grad():
            return self.model(xs).cpu()

    def get_uncertainty(self, images: List[torch.Tensor]) -> np.ndarray:
        logits = self.predict(images)  # [N,C]
        p = F.softmax(logits, dim=1)
        ent = -(p * torch.log(p + 1e-8)).sum(dim=1)
        return ent.numpy().astype(np.float32)
