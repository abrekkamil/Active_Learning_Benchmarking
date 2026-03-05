import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoInstanceSegmentationDataset(Dataset):
    """
    COCO-style dataset for instance segmentation (Mask R-CNN).
    """

    def __init__(self, root_dir, split="train", is_train=True):
        self.root = root_dir
        self.split = split
        self.is_train = is_train

        ann_file = os.path.join(root_dir, split, "_annotations.coco.json")

        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

        self.transform = transforms.ToTensor()

        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}

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

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
            masks.append(torch.tensor(self.coco.annToMask(ann), dtype=torch.uint8))
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy

            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.stack(masks)

            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        else:
            info = self.coco.imgs[img_id]
            H = info["height"]
            W = info["width"]

            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        return {
            "image_id": torch.tensor([img_id]),
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": areas,
            "iscrowd": iscrowd,
        }