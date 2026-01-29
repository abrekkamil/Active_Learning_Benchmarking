import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DeepCrackSegmentationDataset(Dataset):
    """
    DeepCrack dataset with pixel-wise annotations stored as images.

    Directory structure:
        root/
          train/
          train_lab/
          test/
          test_lab/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 256,
    ):
        assert split in ["train", "test", "val"]

        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        if split == "train":
            self.image_dir = os.path.join(root_dir, "train")
            self.mask_dir = os.path.join(root_dir, "train_lab")
        else:
            self.image_dir = os.path.join(root_dir, "test")
            self.mask_dir = os.path.join(root_dir, "test_lab")

        self.samples = self._collect_pairs()

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        print(f"[DeepCrack] {len(self.samples)} samples loaded for split='{split}'")

    def _collect_pairs(self):
        image_files = []
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
            image_files.extend(
                f for f in os.listdir(self.image_dir) if f.endswith(ext)
            )

        pairs = []
        for img_file in image_files:
            base = os.path.splitext(img_file)[0]
            for m_ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                mask_file = base + m_ext
                if os.path.exists(os.path.join(self.mask_dir, mask_file)):
                    pairs.append((img_file, mask_file))
                    break
            else:
                print(f"[Warning] No mask found for {img_file}")

        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, mask_file = self.samples[idx]

        image = Image.open(
            os.path.join(self.image_dir, img_file)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.mask_dir, mask_file)
        ).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).long()

        # One-hot: background / crack
        mask_onehot = torch.zeros((2, *mask.shape[1:]), dtype=torch.float32)
        mask_onehot[0] = (mask == 0)
        mask_onehot[1] = (mask == 1)

        return image, mask_onehot
