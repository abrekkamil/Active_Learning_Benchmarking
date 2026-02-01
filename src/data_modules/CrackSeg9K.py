import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CrackSeg9KDataset(Dataset):
    """
    CrackSeg9K paired image–mask dataset without predefined splits.

    Structure:
        root/
          Images/
          Images-2/        (optional)
          Final_Masks/Masks/
    """

    def __init__(self, root_dir, indices=None, img_size=256):
        self.root_dir = root_dir
        self.img_size = img_size

        self.image_dirs = [
            os.path.join(root_dir, "Images"),
            os.path.join(root_dir, "Images-2"),
        ]
        self.mask_dir = os.path.join(root_dir, "Final_Masks", "Masks")

        self.samples = self._collect_pairs()

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        print(f"[CrackSeg9K] {len(self.samples)} samples loaded")

    def _collect_pairs(self):
        images = []
        for d in self.image_dirs:
            if os.path.exists(d):
                images.extend(
                    os.path.join(d, f)
                    for f in os.listdir(d)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                )

        pairs = []
        for img_path in images:
            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, base + ".png")
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
            else:
                print(f"[Warning] No mask for {base}")

        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).long()

        mask_onehot = torch.zeros((2, *mask.shape[1:]), dtype=torch.float32)
        mask_onehot[0] = (mask == 0)
        mask_onehot[1] = (mask == 1)

        return image, mask_onehot
