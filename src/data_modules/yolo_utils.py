# src/datasets/yolo_utils.py
from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError as e:
    raise ImportError("Please `pip install pyyaml` to use yolo_utils.") from e


# -----------------------------
# Common helpers
# -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_link_or_copy(src: Path, dst: Path, use_symlinks: bool = True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlinks:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)

def _list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return [p for p in d.iterdir() if p.suffix in IMG_EXTS]

def _find_mask_for_stem(mask_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def write_yolo_data_yaml(
    out_root: Path,
    names: Union[List[str], Dict[int, str]],
    train_rel: str = "images/train",
    val_rel: str = "images/val",
) -> Path:
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    data = {"path": str(out_root), "train": train_rel, "val": val_rel, "names": dict(names)}
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return yaml_path


# -----------------------------
# Mask pairs -> YOLOv8-seg
# -----------------------------

def _mask_to_polygons(mask_np: np.ndarray, min_area: float = 100.0) -> List[np.ndarray]:
    """
    Uses OpenCV to extract external contours as polygons.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("Install OpenCV for mask->polygon: pip install opencv-python") from e

    mask_u8 = (mask_np.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cnt = cnt.squeeze(1)  # Nx2
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue
        polys.append(cnt)
    return polys

def _infer_maskpair_split_dirs(root: Path) -> Tuple[Tuple[Path, Path], Tuple[Path, Path]]:
    """
    Expected layouts:
      root/train/images, root/train/masks
      root/val/images,   root/val/masks
    or (if no val):
      root/test/images,  root/test/masks
    """
    tr_img = root / "train" / "images"
    tr_msk = root / "train" / "masks"
    if not tr_img.exists() or not tr_msk.exists():
        raise FileNotFoundError(f"Could not find train/images and train/masks under {root}")

    # prefer val if exists; otherwise map test->val
    va_img = root / "val" / "images"
    va_msk = root / "val" / "masks"
    if va_img.exists() and va_msk.exists():
        return (tr_img, tr_msk), (va_img, va_msk)

    te_img = root / "test" / "images"
    te_msk = root / "test" / "masks"
    if te_img.exists() and te_msk.exists():
        return (tr_img, tr_msk), (te_img, te_msk)

    raise FileNotFoundError(f"Could not find val/* or test/* split under {root}")

def prepare_yolo_from_mask_pairs_auto(
    root_dir: Union[str, Path],
    out_root: Optional[Union[str, Path]] = None,
    class_name: str = "crack",
    min_area: float = 100.0,
    use_symlinks: bool = True,
) -> Path:
    root = Path(root_dir)
    (tr_img, tr_msk), (va_img, va_msk) = _infer_maskpair_split_dirs(root)

    out_root = Path(out_root) if out_root is not None else (root / "yolo_seg")
    out_root = Path(out_root)

    for split, img_dir, msk_dir in [("train", tr_img, tr_msk), ("val", va_img, va_msk)]:
        out_img = _ensure_dir(out_root / "images" / split)
        out_lbl = _ensure_dir(out_root / "labels" / split)

        imgs = _list_images(img_dir)
        missing = 0
        empty = 0

        for img_path in imgs:
            stem = img_path.stem
            mask_path = _find_mask_for_stem(msk_dir, stem)
            if mask_path is None:
                missing += 1
                continue

            _safe_link_or_copy(img_path, out_img / img_path.name, use_symlinks=use_symlinks)

            mask = Image.open(mask_path).convert("L")
            mask_np = (np.array(mask) > 127)

            lbl_path = out_lbl / f"{stem}.txt"
            if mask_np.sum() == 0:
                lbl_path.write_text("")
                empty += 1
                continue

            H, W = mask_np.shape
            polys = _mask_to_polygons(mask_np, min_area=min_area)

            lines = []
            for poly in polys:
                coords = []
                for x, y in poly:
                    coords.append(f"{(x / W):.6f}")
                    coords.append(f"{(y / H):.6f}")
                lines.append("0 " + " ".join(coords))

            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        print(f"[mask_pairs:{split}] images={len(imgs)} missing_masks={missing} empty_masks={empty}")

    yaml_path = write_yolo_data_yaml(out_root, names=[class_name])
    print(f"[mask_pairs] wrote {yaml_path}")
    return yaml_path


# -----------------------------
# COCO -> YOLO (auto)
# -----------------------------

def _find_coco_annotation_files(root: Path) -> Tuple[Path, Path]:
    """
    More robust COCO split discovery.
    Supports:
      - annotations/instances_train.json + annotations/instances_val.json
      - train/_annotations.coco.json + valid/_annotations.coco.json
      - train/_annotations.coco.json + test/_annotations.coco.json
      - *_train*.json and *_val*/valid*/test*.json anywhere under root
    """
    # search recursively but keep it bounded to json files only
    all_json = list(root.rglob("*.json"))

    if not all_json:
        raise FileNotFoundError(f"No .json files found under {root}")

    def score_train(p: Path) -> int:
        n = p.name.lower()
        s = 0
        if "train" in str(p.parent).lower(): s += 3
        if "train" in n: s += 3
        if "_annotations.coco.json" == n: s += 5
        if "instances_train" in n: s += 5
        if "annotations" in str(p.parent).lower(): s += 1
        return s

    def score_val(p: Path) -> int:
        n = p.name.lower()
        s = 0
        parent = str(p.parent).lower()
        if "val" in parent or "valid" in parent: s += 3
        if "test" in parent: s += 2  # fallback: treat test as val
        if "val" in n or "valid" in n: s += 3
        if "test" in n: s += 2
        if "_annotations.coco.json" == n: s += 5
        if "instances_val" in n or "instances_valid" in n: s += 5
        if "annotations" in parent: s += 1
        return s

    # common explicit pairs
    # 1) train/_annotations.coco.json + (valid|val|test)/_annotations.coco.json
    train_candidates = [p for p in all_json if p.name.lower() == "_annotations.coco.json" and p.parent.name.lower() == "train"]
    if train_candidates:
        train_json = train_candidates[0]
        for split in ["val", "valid", "test"]:
            cand = root / split / "_annotations.coco.json"
            if cand.exists():
                return train_json, cand

    # 2) classic COCO names anywhere
    for base in [root / "annotations", root]:
        tr = base / "instances_train.json"
        va = base / "instances_val.json"
        if tr.exists() and va.exists():
            return tr, va

    # scored fallback
    train_json = max(all_json, key=score_train)
    val_json = max(all_json, key=score_val)

    if score_train(train_json) == 0 or score_val(val_json) == 0:
        raise FileNotFoundError(
            f"Could not infer COCO train/val json under {root}. "
            f"Found json files: {[str(p.relative_to(root)) for p in all_json[:20]]}"
        )

    # If the "best val" equals train, try pick second best for val
    if val_json == train_json:
        sorted_val = sorted(all_json, key=score_val, reverse=True)
        for p in sorted_val:
            if p != train_json and score_val(p) > 0:
                val_json = p
                break

    return train_json, val_json


def _infer_coco_images_root(root: Path, coco_json: Path) -> Path:
    """
    Finds folder that makes (images_root / file_name) exist for at least one sample.
    Handles Roboflow layouts:
      root/train/*.jpg
      root/train/images/*.jpg
      root/test/*.jpg
      root/test/images/*.jpg
    """
    coco = json.loads(coco_json.read_text())
    if not coco.get("images"):
        raise ValueError(f"No 'images' field in {coco_json}")

    sample_fn = coco["images"][0]["file_name"]
    sample_basename = Path(sample_fn).name

    candidates = [
        root / "images",
        root / "train" / "images",
        root / "val" / "images",
        root / "valid" / "images",
        root / "test" / "images",
        root / "train",
        root / "val",
        root / "valid",
        root / "test",
        root,
    ]

    for c in candidates:
        if (c / sample_fn).exists() or (c / sample_basename).exists():
            return c

    # fallback: search for that basename somewhere under root (one file only)
    found = list(root.rglob(sample_basename))
    if found:
        return found[0].parent

    raise FileNotFoundError(f"Could not locate images root for COCO file_name '{sample_fn}' under {root}")


def _category_mapping(coco: dict) -> Tuple[Dict[int, int], Dict[int, str]]:
    cats = coco.get("categories", [])
    cat_ids = sorted([c["id"] for c in cats])
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_name = {cat_id_to_idx[c["id"]]: c.get("name", f"class_{cat_id_to_idx[c['id']]}") for c in cats}
    return cat_id_to_idx, idx_to_name

def _bbox_xywh_to_yolo(xywh, img_w, img_h):
    x, y, w, h = xywh
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return cx, cy, (w / img_w), (h / img_h)

def prepare_yolo_from_coco_auto(
    root_dir: Union[str, Path],
    out_root: Optional[Union[str, Path]] = None,
    task: str = "detect",  # "detect" or "segment"
    use_symlinks: bool = True,
) -> Path:
    assert task in ("detect", "segment")
    root = Path(root_dir)
    out_root = Path(out_root) if out_root is not None else (root / f"yolo_{task}")

    train_json, val_json = _find_coco_annotation_files(root)
    images_root = _infer_coco_images_root(root, train_json)

    coco_train = json.loads(Path(train_json).read_text())
    cat_id_to_idx, idx_to_name = _category_mapping(coco_train)

    def convert_split(coco_json: Path, split: str):
        coco = json.loads(Path(coco_json).read_text())
        imgs = {im["id"]: im for im in coco["images"]}

        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        out_img = _ensure_dir(out_root / "images" / split)
        out_lbl = _ensure_dir(out_root / "labels" / split)

        missing_imgs = 0
        for img_id, im in imgs.items():
            fn = im["file_name"]
            w, h = im["width"], im["height"]
            src_img = images_root / fn
            if not src_img.exists():
                # try basename fallback
                src_img = images_root / Path(fn).name
            if not src_img.exists():
                missing_imgs += 1
                continue

            _safe_link_or_copy(src_img, out_img / src_img.name, use_symlinks=use_symlinks)

            stem = Path(src_img.name).stem
            label_path = out_lbl / f"{stem}.txt"
            lines = []

            for ann in anns_by_img.get(img_id, []):
                cls = cat_id_to_idx[ann["category_id"]]
                if task == "detect":
                    cx, cy, ww, hh = _bbox_xywh_to_yolo(ann["bbox"], w, h)
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                else:
                    seg = ann.get("segmentation")
                    if seg is None:
                        continue
                    # polygon format
                    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                        poly = max(seg, key=lambda p: len(p))
                        if len(poly) < 6:
                            continue
                        coords = []
                        for i in range(0, len(poly), 2):
                            coords.append(f"{(poly[i] / w):.6f}")
                            coords.append(f"{(poly[i+1] / h):.6f}")
                        lines.append(f"{cls} " + " ".join(coords))
                    else:
                        raise ValueError(
                            "COCO segmentation appears to be RLE. "
                            "If you need this, I’ll add an RLE->polygon path using pycocotools."
                        )

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        print(f"[coco:{split}] imgs={len(imgs)} missing_imgs={missing_imgs}")

    convert_split(train_json, "train")
    convert_split(val_json, "val")

    yaml_path = write_yolo_data_yaml(out_root, names=idx_to_name)
    print(f"[coco] train_json={train_json.name} val_json={val_json.name}")
    print(f"[coco] images_root={images_root}")
    print(f"[coco] wrote {yaml_path}")
    return yaml_path

# -----------------------------
# YOLO -> COCO (instance segmentation)
# -----------------------------

def convert_yolo_to_coco(
    yolo_root: Union[str, Path],
    output_json: Union[str, Path],
    images_dir: Optional[Union[str, Path]] = None,
):
    """
    Converts YOLO detection or YOLO segmentation dataset to COCO format.
    """

    yolo_root = Path(yolo_root)
    images_dir = Path(images_dir) if images_dir else yolo_root / "images"
    labels_dir = yolo_root / "labels"

    images = []
    annotations = []
    categories = {}

    ann_id = 1
    img_id = 1

    for img_path in sorted(_list_images(images_dir)):

        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        label_path = labels_dir / f"{img_path.stem}.txt"

        if label_path.exists():

            for line in label_path.read_text().splitlines():

                parts = line.strip().split()
                cls = int(parts[0])

                categories.setdefault(cls, f"class_{cls}")

                coords = list(map(float, parts[1:]))

                # ---------- detection ----------
                if len(coords) == 4:

                    cx, cy, bw, bh = coords

                    bw *= w
                    bh *= h
                    x = (cx * w) - bw / 2
                    y = (cy * h) - bh / 2

                    segmentation = []

                # ---------- segmentation ----------
                else:

                    poly = []
                    for i in range(0, len(coords), 2):
                        px = coords[i] * w
                        py = coords[i+1] * h
                        poly.extend([px, py])

                    xs = poly[0::2]
                    ys = poly[1::2]

                    x = min(xs)
                    y = min(ys)
                    bw = max(xs) - x
                    bh = max(ys) - y

                    segmentation = [poly]

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                })

                ann_id += 1

        img_id += 1

    coco = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name}
            for cid, name in categories.items()
        ],
    }

    output_json = Path(output_json)
    output_json.write_text(json.dumps(coco))

    print(f"COCO file written to {output_json}")

# -----------------------------
# YOLO -> COCO Wrapper (instance segmentation / detection)
# -----------------------------

def prepare_coco_from_yolo(config, out_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Converts a YOLO dataset into COCO format so it can be used with
    MaskRCNN or other COCO-based loaders.

    Expected YOLO structure:
        dataset/
            images/
                train/
                val/
            labels/
                train/
                val/

    Output:
        dataset/coco/
            train/_annotations.coco.json
            val/_annotations.coco.json
    """

    root = Path(config.data_dir)

    images_root = root / "images"
    labels_root = root / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"YOLO images folder not found: {images_root}")

    if not labels_root.exists():
        raise FileNotFoundError(f"YOLO labels folder not found: {labels_root}")

    out_root = Path(out_root) if out_root else (root / "coco")

    splits = ["train", "val"]

    for split in splits:

        img_dir = images_root / split
        lbl_dir = labels_root / split

        out_dir = _ensure_dir(out_root / split)
        out_json = out_dir / "_annotations.coco.json"

        images = []
        annotations = []
        categories = {}

        ann_id = 1
        img_id = 1

        for img_path in sorted(_list_images(img_dir)):

            img = Image.open(img_path)
            w, h = img.size

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            })

            label_file = lbl_dir / f"{img_path.stem}.txt"

            if label_file.exists():

                for line in label_file.read_text().splitlines():

                    parts = line.split()
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    categories.setdefault(cls, f"class_{cls}")

                    # -----------------
                    # YOLO detection
                    # -----------------
                    if len(coords) == 4:

                        cx, cy, bw, bh = coords

                        bw *= w
                        bh *= h

                        x = (cx * w) - bw / 2
                        y = (cy * h) - bh / 2

                        segmentation = []

                    # -----------------
                    # YOLO segmentation
                    # -----------------
                    else:

                        poly = []
                        for i in range(0, len(coords), 2):
                            px = coords[i] * w
                            py = coords[i + 1] * h
                            poly.extend([px, py])

                        xs = poly[0::2]
                        ys = poly[1::2]

                        x = min(xs)
                        y = min(ys)
                        bw = max(xs) - x
                        bh = max(ys) - y

                        segmentation = [poly]

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        "segmentation": segmentation,
                    })

                    ann_id += 1

            img_id += 1

        coco = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": cid, "name": name}
                for cid, name in categories.items()
            ],
        }

        out_json.write_text(json.dumps(coco))

        print(f"[yolo->{split}] wrote {out_json}")

    print(f"[yolo->coco] dataset ready at {out_root}")

    return out_root
# -----------------------------
# Public entrypoint: use config only
# -----------------------------

def prepare_yolo_dataset(config, out_root: Optional[Union[str, Path]] = None, task: Optional[str] = None) -> Path:
    """
    Single entrypoint:
      - mask-pairs datasets -> YOLO segmentation
      - coco_detection -> YOLO detection
      - coco_segmentation -> YOLO segmentation (polygon-only)
    """
    root = Path(config.data_dir)
    dtype = getattr(config, "dataset_type", None)

    if dtype in ("crackseg9k", "deepcrack"):
        return prepare_yolo_from_mask_pairs_auto(
            root_dir=root,
            out_root=out_root,
            class_name=getattr(config, "yolo_class_name", "crack"),
            min_area=float(getattr(config, "yolo_min_area", 100.0)),
            use_symlinks=bool(getattr(config, "yolo_use_symlinks", True)),
        )

    if dtype == "coco_detection":
        t = task or "detect"
        return prepare_yolo_from_coco_auto(
            root_dir=root,
            out_root=out_root,
            task=t,
            use_symlinks=bool(getattr(config, "yolo_use_symlinks", True)),
        )

    if dtype == "coco_segmentation":
        t = task or "segment"
        return prepare_yolo_from_coco_auto(
            root_dir=root,
            out_root=out_root,
            task=t,
            use_symlinks=bool(getattr(config, "yolo_use_symlinks", True)),
        )

    raise ValueError(f"prepare_yolo_dataset: unsupported dataset_type={dtype}")