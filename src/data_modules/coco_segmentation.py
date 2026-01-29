import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from coco_detection import CocoDetectionDataset


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
            mask = torch.zeros((self.img_size, self.img_size), dtype=torch.long)
        else:
            merged = torch.max(target["masks"].float(), dim=0)[0]
            merged = self.mask_transform(merged.unsqueeze(0))
            mask = (merged > 0.5).long().squeeze(0)

        return image, mask
