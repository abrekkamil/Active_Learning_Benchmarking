# src/datasets/yolo_utils.py

from __future__ import annotations

import os
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError as e:
    raise ImportError("Please `pip install pyyaml` to use yolo_utils.") from e


# -----------------------------
# Small helpers
# -----------------------------

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        # fallback to copy if symlink not allowed
        shutil.copy2(src, dst)

def _list_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return [p for p in image_dir.iterdir() if p.suffix in exts]

def _find_by_stem(directory: Path, stem: str) -> Optional[Path]:
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    for e in exts:
        p = directory / f"{stem}{e}"
        if p.exists():
            return p
    return None

def write_yolo_data_yaml(
    out_root: Union[str, Path],
    names: Union[List[str], Dict[int, str]],
    train_subdir: str = "images/train",
    val_subdir: str = "images/val",
    filename: str = "data.yaml",
) -> Path:
    """
    Writes Ultralytics data.yaml.
    `names` can be a list ["crack", ...] or dict {0:"crack"}.
    """
    out_root = Path(out_root)
    if isinstance(names, list):
        names_dict = {i: n for i, n in enumerate(names)}
    else:
        names_dict = dict(names)

    data = {
        "path": str(out_root),
        "train": train_subdir,
        "val": val_subdir,
        "names": names_dict,
    }
    yaml_path = out_root / filename
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return yaml_path


# -----------------------------
# Mask-pairs -> YOLO-seg
# -----------------------------

def _mask_to_polygons_cv2(mask_np: np.ndarray, min_area: float = 50.0) -> List[np.ndarray]:
    """
    mask_np: HxW bool
    returns list of polygons (Nx2 array of xy pixel coords)
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV is required for mask->polygon. Install: pip install opencv-python") from e

    mask_u8 = (mask_np.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cnt = cnt.squeeze(1)  # Nx2
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue
        polys.append(cnt)
    return polys

def prepare_yolo_from_mask_pairs(
    root_dir: Union[str, Path],
    out_root: Union[str, Path],
    train_images_dir: Union[str, Path],
    train_masks_dir: Union[str, Path],
    val_images_dir: Union[str, Path],
    val_masks_dir: Union[str, Path],
    class_name: str = "crack",
    min_area: float = 100.0,
    use_symlinks: bool = True,
) -> Path:
    """
    Convert datasets stored as image/mask pairs into YOLOv8 segmentation format.

    Output:
      out_root/images/train, out_root/labels/train
      out_root/images/val,   out_root/labels/val
      out_root/data.yaml

    This assumes binary masks (foreground crack vs background).
    YOLO class index = 0.
    """
    root_dir = Path(root_dir)
    out_root = Path(out_root)

    for split, img_dir, msk_dir in [
        ("train", Path(train_images_dir), Path(train_masks_dir)),
        ("val",   Path(val_images_dir),   Path(val_masks_dir)),
    ]:
        out_img = _ensure_dir(out_root / "images" / split)
        out_lbl = _ensure_dir(out_root / "labels" / split)

        img_files = _list_images(img_dir)
        missing_masks = 0
        empty_masks = 0

        for img_path in img_files:
            stem = img_path.stem
            mask_path = _find_by_stem(msk_dir, stem)
            if mask_path is None:
                missing_masks += 1
                continue

            # place image
            dst_img = out_img / img_path.name
            if use_symlinks:
                _safe_symlink(img_path, dst_img)
            else:
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

            # read mask (no resize here; YOLO handles imgsz)
            mask = Image.open(mask_path).convert("L")
            mask_np = (np.array(mask) > 127)

            lbl_path = out_lbl / f"{stem}.txt"
            if mask_np.sum() == 0:
                lbl_path.write_text("")
                empty_masks += 1
                continue

            H, W = mask_np.shape
            polys = _mask_to_polygons_cv2(mask_np, min_area=min_area)

            lines = []
            for poly in polys:
                coords = []
                for x, y in poly:
                    coords.append(f"{(x / W):.6f}")
                    coords.append(f"{(y / H):.6f}")
                lines.append("0 " + " ".join(coords))

            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        print(f"[mask_pairs:{split}] images={len(img_files)} missing_masks={missing_masks} empty_masks={empty_masks}")

    yaml_path = write_yolo_data_yaml(out_root, names=[class_name])
    print(f"[mask_pairs] Wrote {yaml_path}")
    return yaml_path


# -----------------------------
# COCO -> YOLO (detect or seg)
# -----------------------------

def _load_coco(coco_json: Union[str, Path]) -> dict:
    coco_json = Path(coco_json)
    return json.loads(coco_json.read_text())

def _category_mapping(coco: dict) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    COCO category ids can be non-contiguous; YOLO wants 0..nc-1.
    Returns:
      cat_id_to_yolo_idx, yolo_idx_to_name
    """
    cats = coco.get("categories", [])
    cat_ids = sorted([c["id"] for c in cats])
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_name = {cat_id_to_idx[c["id"]]: c.get("name", f"class_{cat_id_to_idx[c['id']]}") for c in cats}
    return cat_id_to_idx, idx_to_name

def _bbox_xywh_to_yolo(xywh, img_w, img_h) -> Tuple[float, float, float, float]:
    x, y, w, h = xywh
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    ww = w / img_w
    hh = h / img_h
    return cx, cy, ww, hh

def prepare_yolo_from_coco(
    coco_train_json: Union[str, Path],
    coco_val_json: Union[str, Path],
    images_root: Union[str, Path],
    out_root: Union[str, Path],
    task: str = "detect",  # "detect" or "segment"
    use_symlinks: bool = True,
    min_area: float = 50.0,
) -> Path:
    """
    Convert COCO dataset to YOLO format.
    - detect: labels are bbox lines: cls cx cy w h
    - segment: labels are polygon lines: cls x1 y1 x2 y2 ...

    images_root: base folder for file_name entries in COCO.
    """
    assert task in ("detect", "segment")
    images_root = Path(images_root)
    out_root = Path(out_root)

    def convert_split(coco_json, split_name: str, cat_id_to_idx: Dict[int, int]):
        coco = _load_coco(coco_json)
        imgs = {im["id"]: im for im in coco["images"]}
        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        out_img = _ensure_dir(out_root / "images" / split_name)
        out_lbl = _ensure_dir(out_root / "labels" / split_name)

        missing_imgs = 0

        for img_id, im in imgs.items():
            fn = im["file_name"]
            w, h = im["width"], im["height"]

            src_img = images_root / fn
            if not src_img.exists():
                missing_imgs += 1
                continue

            dst_img = out_img / Path(fn).name
            if use_symlinks:
                _safe_symlink(src_img, dst_img)
            else:
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

            stem = Path(fn).stem
            label_path = out_lbl / f"{stem}.txt"
            lines = []

            for ann in anns_by_img.get(img_id, []):
                cls = cat_id_to_idx[ann["category_id"]]

                if task == "detect":
                    cx, cy, ww, hh = _bbox_xywh_to_yolo(ann["bbox"], w, h)
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                else:
                    seg = ann.get("segmentation", None)
                    if seg is None:
                        continue

                    # COCO polygon format: list of lists
                    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                        # pick largest polygon by number of points
                        poly = max(seg, key=lambda p: len(p))
                        if len(poly) < 6:
                            continue
                        coords = []
                        for i in range(0, len(poly), 2):
                            x = poly[i] / w
                            y = poly[i + 1] / h
                            coords.append(f"{x:.6f}")
                            coords.append(f"{y:.6f}")
                        lines.append(f"{cls} " + " ".join(coords))
                    else:
                        # Likely RLE dict -> needs pycocotools to decode + contour extraction
                        raise ValueError(
                            "COCO segmentation appears to be RLE (not polygon). "
                            "Install pycocotools and convert RLE->polygons (I can provide that)."
                        )

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        print(f"[coco:{split_name}] imgs={len(imgs)} missing_imgs={missing_imgs}")

    coco_train = _load_coco(coco_train_json)
    cat_id_to_idx, idx_to_name = _category_mapping(coco_train)

    convert_split(coco_train_json, "train", cat_id_to_idx)
    convert_split(coco_val_json, "val", cat_id_to_idx)

    yaml_path = write_yolo_data_yaml(out_root, names=idx_to_name)
    print(f"[coco] Wrote {yaml_path}")
    return yaml_path