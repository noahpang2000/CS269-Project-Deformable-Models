"""Torch datasets over FLAME-3: RGB input + thermal-thresholded GT mask.

Network resolution is a fixed square (NET_SIZE); predictions are scored back at
native resolution by the eval loop. RGB-only input at inference (no oracle init).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from code.flame.data import DEFAULT_DATASET, DEFAULT_THRESHOLD_C, load_frame
from code.flame.contour_utils import (
    largest_component_mask,
    outer_contour,
    resample_polygon,
)

NET_SIZE = 512


def _has_fire(fid: str, threshold_c: float, dataset: str = DEFAULT_DATASET) -> bool:
    return load_frame(fid, threshold_c=threshold_c, dataset=dataset).gt_mask.max() > 0


class FlameDataset(Dataset):
    """Per-pixel datasets (U-Net, DALS): returns image [3,H,W] and mask [1,H,W]."""

    def __init__(self, frame_ids: list[str], threshold_c: float = DEFAULT_THRESHOLD_C,
                 size: int = NET_SIZE, augment: bool = False,
                 dataset: str = DEFAULT_DATASET):
        self.frame_ids = frame_ids
        self.threshold_c = threshold_c
        self.size = size
        self.augment = augment
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_resized(self, fid: str):
        # Everything is resized to a fixed NET_SIZE square here, so FLAME-1's
        # native 3840x2160 needs no separate max_side handling on the deep path.
        f = load_frame(fid, threshold_c=self.threshold_c, dataset=self.dataset)
        rgb = cv2.resize(f.rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(f.gt_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return rgb, mask

    def __getitem__(self, i: int) -> dict:
        rgb, mask = self._load_resized(self.frame_ids[i])
        if self.augment and np.random.rand() < 0.5:
            rgb, mask = rgb[:, ::-1].copy(), mask[:, ::-1].copy()
        image = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        target = torch.from_numpy((mask > 0).astype(np.float32))[None]
        return {"frame_id": self.frame_ids[i], "image": image, "mask": target}


class SnakeDataset(FlameDataset):
    """Deep Snake training data: image + initial contour + GT contour ([N,2] xy).

    Single largest component per frame (the detector-free simplification), so
    frames with no fire are dropped. Init is a circle at the component centroid.
    """

    def __init__(self, frame_ids: list[str], threshold_c: float = DEFAULT_THRESHOLD_C,
                 size: int = NET_SIZE, augment: bool = False, n_points: int = 128,
                 dataset: str = DEFAULT_DATASET):
        usable = [f for f in frame_ids if _has_fire(f, threshold_c, dataset)]
        super().__init__(usable, threshold_c, size, augment, dataset)
        self.n_points = n_points

    def __getitem__(self, i: int) -> dict:
        rgb, mask = self._load_resized(self.frame_ids[i])
        if self.augment and np.random.rand() < 0.5:
            rgb, mask = rgb[:, ::-1].copy(), mask[:, ::-1].copy()

        contour = outer_contour(largest_component_mask(mask))
        ang = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        unit = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        if len(contour) < 3:
            # Fire too small to survive the resize: benign centered placeholder.
            center = np.array([self.size / 2, self.size / 2], dtype=np.float32)
            gt_poly = center + 2.0 * unit
            init_poly = center + 4.0 * unit
        else:
            gt_poly = resample_polygon(contour, self.n_points)
            center = gt_poly.mean(axis=0)
            radius = float(np.linalg.norm(gt_poly - center, axis=1).mean()) + 1e-3
            init_poly = center + radius * unit

        image = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        target = torch.from_numpy((mask > 0).astype(np.float32))[None]
        return {
            "frame_id": self.frame_ids[i],
            "image": image,
            "mask": target,
            "init_contour": torch.from_numpy(init_poly.astype(np.float32)),
            "gt_contour": torch.from_numpy(gt_poly.astype(np.float32)),
        }


MIN_CC_PX = 30


class PaperSnakeDataset(FlameDataset):
    """Per-INSTANCE training data for deep_snake_paper.

    One sample per fire connected component (not per frame): the component's
    tight bounding box + that component's resampled contour. This matches how the
    paper pipeline operates at test time (detector emits one box per blob, the
    snake refines each), and fixes the single-octagon-over-many-blobs mismatch
    that made the whole-frame box collapse.

    Returns a per-sample box [x0,y0,x1,y1] (network coords) and gt_contour [N,2].
    """

    def __init__(self, frame_ids, threshold_c=DEFAULT_THRESHOLD_C, size=NET_SIZE,
                 augment=False, n_points=128, dataset=DEFAULT_DATASET):
        usable = [f for f in frame_ids if _has_fire(f, threshold_c, dataset)]
        super().__init__(usable, threshold_c, size, augment, dataset)
        self.n_points = n_points
        # Index every (frame, component) pair up front so __len__ = #instances.
        self.instances: list[tuple[int, int]] = []
        for fi, fid in enumerate(self.frame_ids):
            _, mask = self._load_resized(fid)
            num, _, stats, _ = cv2.connectedComponentsWithStats(
                (mask > 0).astype(np.uint8), connectivity=8)
            for lbl in range(1, num):
                if stats[lbl, cv2.CC_STAT_AREA] >= MIN_CC_PX:
                    self.instances.append((fi, lbl))

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        fi, lbl = self.instances[idx]
        rgb, mask = self._load_resized(self.frame_ids[fi])
        if self.augment and np.random.rand() < 0.5:
            rgb, mask = rgb[:, ::-1].copy(), mask[:, ::-1].copy()

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8)
        # Augmentation can relabel components; fall back to the largest if the
        # original label index is gone (flip changes nothing topologically, but
        # be safe).
        if lbl >= num:
            lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if num > 1 else 0
        comp = ((labels == lbl).astype(np.uint8)) * 255

        x, y, bw, bh, _ = stats[lbl]
        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)

        contour = outer_contour(comp)
        ang = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        unit = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        if len(contour) < 3:
            c = np.array([(x + bw / 2), (y + bh / 2)], dtype=np.float32)
            gt_poly = c + 2.0 * unit
        else:
            gt_poly = resample_polygon(contour, self.n_points)

        image = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        return {
            "frame_id": self.frame_ids[fi],
            "image": image,
            "box": torch.from_numpy(box),                       # [4] x0,y0,x1,y1
            "gt_contour": torch.from_numpy(gt_poly.astype(np.float32)),  # [N,2]
        }
