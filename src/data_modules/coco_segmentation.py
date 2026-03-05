import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from .coco_instance import CocoDetectionDataset
import torch.nn.functional as F

class CocoSemanticSegmentationDataset(CocoDetectionDataset):
    def __init__(self, root_dir, split, img_size):
        super().__init__(root_dir, split, is_train=True)
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Resize(
            (img_size, img_size), interpolation=Image.NEAREST
        )

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        image = self.img_transform(self._load_image(img_id))
        target = self._load_target(img_id)

        if len(target["masks"]) == 0:
            mask = torch.zeros(
                (self.img_size, self.img_size), dtype=torch.long
            )
        else:
            masks = target["masks"].float()  # [N, H, W]
            merged = torch.max(masks, dim=0, keepdim=True)[0]  # [1, H, W]

            merged = F.interpolate(
                merged.unsqueeze(0),  # [1, 1, H, W]
                size=(self.img_size, self.img_size),
                mode="nearest",
            ).squeeze(0).squeeze(0)

            mask = (merged > 0.5).long()

        return image, mask
