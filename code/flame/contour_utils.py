"""Fire-energy features and polygon/level-set geometry helpers."""
from __future__ import annotations

import cv2
import numpy as np


def fire_energy(rgb: np.ndarray, mode: str = "rg",
                thermal_c: np.ndarray | None = None) -> np.ndarray:
    if mode == "rg":
        rgb_f = rgb.astype(np.float32)
        return np.clip(rgb_f[..., 0] - rgb_f[..., 1], 0, 255) / 255.0
    if mode == "thermal":
        if thermal_c is None:
            raise ValueError("energy_mode='thermal' requires thermal_c")
        return np.clip(thermal_c.astype(np.float32), 0.0, 300.0) / 300.0
    raise ValueError(f"unknown fire_energy mode: {mode!r} (expected 'rg' or 'thermal')")


def largest_component_mask(binary_mask: np.ndarray) -> np.ndarray:
    bm = (binary_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
    if num <= 1:
        return bm * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


def outer_contour(binary_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        (binary_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)


def scale_polygon(poly_xy: np.ndarray, factor: float) -> np.ndarray:
    if len(poly_xy) == 0:
        return poly_xy
    centroid = poly_xy.mean(axis=0, keepdims=True)
    return centroid + (poly_xy - centroid) * factor


def resample_polygon(poly_xy: np.ndarray, n_points: int) -> np.ndarray:
    if len(poly_xy) == 0:
        return poly_xy
    closed = np.vstack([poly_xy, poly_xy[:1]])
    seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total == 0:
        return np.repeat(poly_xy[:1], n_points, axis=0)
    targets = np.linspace(0, total, n_points, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def polygon_to_mask(poly_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if len(poly_xy) < 3:
        return mask
    pts = np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def init_snake_from_mask(gt_mask: np.ndarray, dilate_factor: float,
                         n_points: int) -> np.ndarray:
    contour_xy = outer_contour(largest_component_mask(gt_mask))
    return resample_polygon(scale_polygon(contour_xy, dilate_factor), n_points)


def dilated_level_set(gt_mask: np.ndarray, dilate_factor: float) -> np.ndarray:
    bm = (gt_mask > 0).astype(np.uint8)
    target_area = bm.sum() * (dilate_factor ** 2)
    if target_area <= 0:
        return bm
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = bm.copy()
    for _ in range(50):
        if dilated.sum() >= target_area:
            break
        dilated = cv2.dilate(dilated, kernel, iterations=1)
    return dilated
