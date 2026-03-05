import torch
from typing import List

def _ensure_chw(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 2:
        img = img.unsqueeze(0)

    if img.dim() != 3:
        raise ValueError(f"Expected CHW image, got {img.shape}")

    return img


def _ensure_rgb(img: torch.Tensor) -> torch.Tensor:
    img = _ensure_chw(img)

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    return img


def _batched(imgs: List[torch.Tensor]) -> torch.Tensor:
    imgs = [_ensure_chw(i) for i in imgs]
    return torch.stack(imgs, dim=0)