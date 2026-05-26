"""Torch datasets over FLAME-3: RGB input + thermal-thresholded GT mask.

Network resolution is a fixed square (NET_SIZE); predictions are scored back at
native resolution by the eval loop. RGB-only input at inference (no oracle init).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from flame.data import DEFAULT_THRESHOLD_C, load_frame
from flame.contour_utils import (
    largest_component_mask,
    outer_contour,
    resample_polygon,
)

NET_SIZE = 512


def _has_fire(fid: str, threshold_c: float) -> bool:
    return load_frame(fid, threshold_c=threshold_c).gt_mask.max() > 0


class FlameDataset(Dataset):
    """Per-pixel datasets (U-Net, DALS): returns image [3,H,W] and mask [1,H,W]."""

    def __init__(self, frame_ids: list[str], threshold_c: float = DEFAULT_THRESHOLD_C,
                 size: int = NET_SIZE, augment: bool = False):
        self.frame_ids = frame_ids
        self.threshold_c = threshold_c
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_resized(self, fid: str):
        f = load_frame(fid, threshold_c=self.threshold_c)
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
                 size: int = NET_SIZE, augment: bool = False, n_points: int = 128):
        usable = [f for f in frame_ids if _has_fire(f, threshold_c)]
        super().__init__(usable, threshold_c, size, augment)
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
