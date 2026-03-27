import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import cv2


class YoloInstanceSegmentationDataset(Dataset):
    """
    YOLO-format dataset for instance segmentation (polygons → masks).
    """

    def __init__(self, root_dir, split="train", img_size=640, is_train=True):
        self.root = root_dir
        self.split = split
        self.is_train = is_train
        self.img_size = img_size
        if split == 'val':
            split = 'valid' # Sewerml only has Train and Test splits, so we use Test for validation
        self.img_dir = os.path.join(root_dir, split,"images" )
        self.label_dir = os.path.join(root_dir, split, "labels")

        self.images = sorted(os.listdir(self.img_dir))

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        image = self.transform(image)

        if self.is_train:
            target = self._load_target(img_name, H, W)
        else:
            target = {}

        return image, target

    def _load_target(self, img_name, H, W):
        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                polygon = parts[1:]

                # Convert polygon → mask
                mask = self._polygon_to_mask(polygon, H, W)

                # Bounding box from mask
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(cls)
                masks.append(torch.tensor(mask, dtype=torch.uint8))
                areas.append((x_max - x_min) * (y_max - y_min))
                iscrowd.append(0)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks)

            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": areas,
            "iscrowd": iscrowd,
        }

    def _polygon_to_mask(self, polygon, H, W):
        """
        Convert YOLO normalized polygon → binary mask
        """
        points = np.array(polygon).reshape(-1, 2)

        # denormalize
        points[:, 0] *= W
        points[:, 1] *= H

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)

        return mask