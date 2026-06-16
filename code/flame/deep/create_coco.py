"""Build a COCO-format detection dataset (fire bounding boxes) for training the
MMDet CenterNet box detector used by the Deep Snake test pipeline.

GT boxes are derived from the GT mask (FLAME-3: thermal threshold; FLAME-1:
hand-labeled PNG) via connected components. RGB images are symlinked into
<dataset>_coco/images/ so the COCO img_prefix resolves without duplicating the
(large) source frames.

Usage:
    python -m flame.deep.create_coco              # FLAME-3 (default)
    python -m flame.deep.create_coco --dataset flame1
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path

from flame.data import (DEFAULT_DATASET, PROJECT_ROOT, RGB_DIR,
                        FLAME1_RGB_DIR, FLAME2_RGB_DIR, load_frame)
from flame.splits import make_splits

MIN_CC_PX = 30


def _dataset_paths(dataset: str):
    """Return (coco_root, src_rgb_dir, img_ext) for the given dataset.

    Boxes are derived at the source frame's NATIVE resolution so they match the
    symlinked native image (CenterNet's own pipeline handles resizing to 512).
    """
    coco_root = PROJECT_ROOT / f"{dataset}_coco"
    if dataset == "flame3":
        return coco_root, RGB_DIR, ".JPG"
    if dataset == "flame1":
        return coco_root, FLAME1_RGB_DIR, ".jpg"
    if dataset == "flame2":
        return coco_root, FLAME2_RGB_DIR, ".jpg"
    raise ValueError(f"unknown dataset {dataset!r}")


def _link_image(fid: str, src_dir: Path, images_dir: Path, ext: str) -> None:
    """Symlink the source RGB frame into the COCO images/ folder (idempotent)."""
    dst = images_dir / f"{fid}{ext}"
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to((src_dir / f"{fid}{ext}").resolve())


def create_coco_json(frame_ids, output_path, dataset, src_dir, images_dir, ext):
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fire"}],  # single class
    }

    ann_id = 1
    for img_id, fid in enumerate(frame_ids):
        # Native-resolution GT mask (no max_side) so boxes match the linked image.
        frame = load_frame(fid, dataset=dataset)
        h, w = frame.gt_mask.shape

        _link_image(fid, src_dir, images_dir, ext)
        coco_dict["images"].append({
            "id": img_id,
            "file_name": f"{fid}{ext}",  # matches the symlinked filename
            "width": w,
            "height": h,
        })

        # Distinct fire instances via connected components.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (frame.gt_mask > 0).astype(np.uint8))

        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] < MIN_CC_PX:
                continue

            x, y, bw, bh, area = stats[lbl]
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,  # "fire"
                "bbox": [int(x), int(y), int(bw), int(bh)],  # COCO [x, y, w, h]
                "area": int(area),
                "iscrowd": 0,
            })
            ann_id += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_dict, f)
    print(f"wrote {output_path}: {len(coco_dict['images'])} images, "
          f"{len(coco_dict['annotations'])} boxes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["flame3", "flame1", "flame2"], default=DEFAULT_DATASET)
    args = ap.parse_args()

    coco_root, src_dir, ext = _dataset_paths(args.dataset)
    images_dir = coco_root / "images"
    ann_dir = coco_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits(dataset=args.dataset)
    for split in ("train", "val"):
        create_coco_json(splits[split], ann_dir / f"{split}.json",
                         args.dataset, src_dir, images_dir, ext)
    print(f"COCO dataset for {args.dataset} -> {coco_root}")


if __name__ == "__main__":
    main()
