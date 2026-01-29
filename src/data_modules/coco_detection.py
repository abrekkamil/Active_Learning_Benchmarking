import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDetectionDataset(Dataset):
    """
    COCO-style dataset for detection / instance segmentation (Mask R-CNN).
    """

    def __init__(self, root_dir, split="train", is_train=True):
        self.root = root_dir
        self.split = split
        self.is_train = is_train

        ann_file = os.path.join(
            root_dir, split, "_annotations.coco.json"
        )

        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transforms.ToTensor()

        self.classes = {
            k: v["name"] for k, v in self.coco.cats.items()
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        image = self._load_image(img_id)
        image = self.transform(image)

        target = self._load_target(img_id) if self.is_train else {}
        return image, target

    def _load_image(self, img_id):
        info = self.coco.imgs[img_id]
        path = os.path.join(self.root, self.split, info["file_name"])
        return Image.open(path).convert("RGB")

    def _load_target(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []

        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
            masks.append(
                torch.tensor(self.coco.annToMask(ann), dtype=torch.uint8)
            )

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]  # xywh → xyxy
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
        else:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.long)
            masks = torch.zeros((0, 1, 1), dtype=torch.uint8)

        return {
            "image_id": torch.tensor([img_id]),
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }
