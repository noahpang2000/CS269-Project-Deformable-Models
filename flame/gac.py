"""Geodesic Active Contours via morphological_geodesic_active_contour."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
)

from flame.data import Frame
from flame.contour_utils import fire_energy, dilated_level_set


@dataclass
class GACConfig:
    num_iter: int = 25
    smoothing: int = 2
    balloon: int = 0
    threshold: str | float = "auto"
    sigma: float = 2.0
    alpha: float = 10.0
    dilate_factor: float = 1.15
    energy_mode: str = "rg"


def run_gac(frame: Frame, cfg: GACConfig = GACConfig(),
            init_mask: np.ndarray | None = None) -> np.ndarray:
    """init_mask seeds the level set; None = oracle init from the GT mask."""
    src_mask = frame.gt_mask if init_mask is None else init_mask
    energy = fire_energy(frame.rgb, cfg.energy_mode, thermal_c=frame.thermal_c)
    g = inverse_gaussian_gradient(energy, alpha=cfg.alpha, sigma=cfg.sigma)
    init = dilated_level_set(src_mask, cfg.dilate_factor)
    level_set = morphological_geodesic_active_contour(
        g, num_iter=cfg.num_iter, init_level_set=init,
        smoothing=cfg.smoothing, balloon=cfg.balloon, threshold=cfg.threshold,
    )
    return (level_set > 0).astype(np.uint8) * 255
