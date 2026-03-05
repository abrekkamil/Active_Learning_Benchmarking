import time
import torch
from typing import List

from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .utils import _ensure_rgb

class MaskRCNNModel:

    def __init__(self, num_classes, device, config):

        self.device = device
        self.config = config
        self.num_classes = num_classes

        model = maskrcnn_resnet50_fpn(weights=None)

        # replace classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # replace mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )

        self.model = model.to(device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=getattr(config, "lr", 1e-3),
            momentum=0.9,
            weight_decay=0.0,
        )
    def train_epoch(self, dataset, epoch, total_epochs):

        loader = DataLoader(
            dataset,
            batch_size=getattr(self.config, "batch_size", 2),
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            collate_fn=lambda x: tuple(zip(*x)),
        )

        self.model.train()
        start = time.time()

        for images, targets in loader:

            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        return {"training_time": time.time() - start}
    
    def predict(self, images: List[torch.Tensor]):

        self.model.eval()

        imgs = [_ensure_rgb(img).to(self.device) for img in images]

        with torch.no_grad():
            outputs = self.model(imgs)

        preds = []
        for out in outputs:
            preds.append({k: v.detach().cpu() for k, v in out.items()})

        return preds