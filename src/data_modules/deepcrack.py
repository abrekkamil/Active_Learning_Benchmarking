import os
import random
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

    Leakage-safe three-way split, with a switch for where val comes from:

      val_source = "train":
          train : train/ MINUS a seeded val slice   (selection pool shrinks)
          val   : a seeded slice of train/           (drives reward + selection)
          test  : all of test/                       (full-size, reported once)

      val_source = "test":
          train : all of train/                      (pool matches the paper)
          val   : a seeded half of test/             (drives reward + selection)
          test  : the OTHER seeded half of test/     (reported once)

    In both cases val and test are disjoint, and val is what the RL reward /
    checkpoint selection use while test is evaluated only at the end.

    Set val_fraction = 0.0 to reproduce the old behaviour (val == whole test).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 256,
        val_source: str = "train",   # "train" | "test"
        val_fraction: float = 0.2,   # train-carve fraction, OR test-half fraction
        val_seed: int = 12345,
    ):
        assert split in ["train", "test", "val"]
        assert val_source in ["train", "test"]

        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.val_source = val_source
        self.val_fraction = val_fraction
        self.val_seed = val_seed

        if val_source == "train":
            if split == "test":
                self.image_dir = os.path.join(root_dir, "test")
                self.mask_dir = os.path.join(root_dir, "test_lab")
            else:
                self.image_dir = os.path.join(root_dir, "train")
                self.mask_dir = os.path.join(root_dir, "train_lab")
        else:  # val_source == "test"
            if split == "train":
                self.image_dir = os.path.join(root_dir, "train")
                self.mask_dir = os.path.join(root_dir, "train_lab")
            else:
                self.image_dir = os.path.join(root_dir, "test")
                self.mask_dir = os.path.join(root_dir, "test_lab")

        self.samples = self._collect_pairs()
        self.samples = self._apply_split()

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        print(f"[DeepCrack] {len(self.samples)} samples for split='{split}' "
              f"(val_source={val_source}, val_fraction={val_fraction}, "
              f"val_seed={val_seed})")

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

    def _seeded_partition(self, pairs, fraction):
        """Deterministic split: returns (first, second) where first is
        `fraction` of the total, shuffled by val_seed."""
        ordered = sorted(pairs, key=lambda p: p[0])
        rng = random.Random(self.val_seed)
        shuffled = ordered[:]
        rng.shuffle(shuffled)
        n_first = max(1, int(round(len(shuffled) * fraction)))
        first = sorted(shuffled[:n_first], key=lambda p: p[0])
        second = sorted(shuffled[n_first:], key=lambda p: p[0])
        return first, second

    def _apply_split(self):
        if self.val_fraction <= 0.0:
            return self.samples

        if self.val_source == "train":
            if self.split == "test":
                return self.samples
            val_pairs, train_pairs = self._seeded_partition(
                self.samples, self.val_fraction
            )
            return val_pairs if self.split == "val" else train_pairs

        # val_source == "test"
        if self.split == "train":
            return self.samples
        val_pairs, test_pairs = self._seeded_partition(
            self.samples, self.val_fraction
        )
        return val_pairs if self.split == "val" else test_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, mask_file = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, img_file)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_file)).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).long()

        mask_onehot = torch.zeros((2, *mask.shape[1:]), dtype=torch.float32)
        mask_onehot[0] = (mask == 0)
        mask_onehot[1] = (mask == 1)
        return image, mask_onehot