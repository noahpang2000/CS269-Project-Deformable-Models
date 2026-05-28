"""Data augmentation for the per-pixel detector.

The learning curve showed the U-Net is data-starved (val IoU still rising at
100% of 435 frames). Heavier augmentation multiplies effective samples without
new labels. Geometric transforms (flip/rotate/scale/crop) MUST apply identically
to image and mask; photometric jitter applies to the image ONLY (the mask is a
label, not a brightness). Operates on uint8 RGB [H,W,3] + uint8 mask [H,W].
"""
from __future__ import annotations

import cv2
import numpy as np


def _photometric(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness/contrast + hue jitter on the image only."""
    out = rgb.astype(np.float32)
    # brightness (additive) and contrast (multiplicative)
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:                       # mild hue shift
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-8, 9)) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def medium_augment(rgb: np.ndarray, mask: np.ndarray,
                   rng: np.random.Generator | None = None):
    """Gentle augmentation: flips + small rotation/scale/shift + mild brightness/
    contrast only. No hue shift and no large warps -- preserves the faint fire
    cue that aggressive geometry/photometry can destroy.
    """
    rng = rng or np.random.default_rng()
    h, w = mask.shape
    if rng.random() < 0.5:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.95, 1.10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += rng.uniform(-0.04, 0.04) * w
    M[1, 2] += rng.uniform(-0.04, 0.04) * h
    rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # mild brightness/contrast only (image)
    out = np.clip(rgb.astype(np.float32) * rng.uniform(0.92, 1.08)
                  + rng.uniform(-8, 8), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out), np.ascontiguousarray(mask)


def strong_augment(rgb: np.ndarray, mask: np.ndarray,
                   rng: np.random.Generator | None = None):
    """Random flip + rotation + scale + crop (shared) and photometric (image-only).

    Returns (rgb, mask) at the same H,W as the input.
    """
    rng = rng or np.random.default_rng()
    h, w = mask.shape

    # horizontal flip
    if rng.random() < 0.5:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    # vertical flip (aerial view: no canonical up)
    if rng.random() < 0.5:
        rgb, mask = rgb[::-1, :], mask[::-1, :]

    # rotation + scale about center (shared affine)
    angle = rng.uniform(-25, 25)
    scale = rng.uniform(0.85, 1.20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # random crop back to full size (translation jitter) when scale > 1
    # (warpAffine already keeps size; this adds a small shift)
    max_shift = int(0.08 * min(h, w))
    if max_shift > 0:
        dx, dy = rng.integers(-max_shift, max_shift + 1, size=2)
        Mt = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb = cv2.warpAffine(rgb, Mt, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, Mt, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    rgb = _photometric(np.ascontiguousarray(rgb), rng)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(mask)
