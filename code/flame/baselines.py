"""RGB color-threshold floor baseline (no model, no init, RGB-only).

Thresholds the red-yellow opponency (R-G) feature with the same morphological
cleanup as the GT generator. Serves two roles: a trivial floor that says how
much the learned/contour methods buy over a plain color rule, and the
image-derived prior that initialises the classical methods without the oracle.
"""
from __future__ import annotations

import cv2
import numpy as np

from flame.contour_utils import fire_energy

# R-G barely separates the thermal-defined fire from background on this data
# (the hot core reads bright/white, not red), so any tau scores near zero; this
# default at least yields a non-empty mask rather than collapsing to nothing.
DEFAULT_TAU = 0.08


def color_threshold_mask(rgb: np.ndarray, tau: float = DEFAULT_TAU,
                         min_blob_px: int = 20) -> np.ndarray:
    binary = (fire_energy(rgb, "rg") >= tau).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    if min_blob_px > 0 and binary.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
                keep[labels == lbl] = 1
        binary = keep
    return binary * 255
