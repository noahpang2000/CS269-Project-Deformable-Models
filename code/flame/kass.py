"""Kass et al. 1988 Snakes via scikit-image's active_contour."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from flame.data import Frame
from flame.contour_utils import fire_energy, init_snake_from_mask, polygon_to_mask


@dataclass
class KassConfig:
    alpha: float = 0.01
    beta: float = 0.1
    w_line: float = 1.0
    w_edge: float = 1.0
    gamma: float = 0.01
    max_iter: int = 200
    sigma: float = 2.0
    n_points: int = 200
    dilate_factor: float = 1.15
    energy_mode: str = "rg"


def run_kass(frame: Frame, cfg: KassConfig = KassConfig(),
             init_mask: np.ndarray | None = None) -> np.ndarray:
    """init_mask seeds the contour; None = oracle init from the GT mask."""
    src_mask = frame.gt_mask if init_mask is None else init_mask
    energy = fire_energy(frame.rgb, cfg.energy_mode, thermal_c=frame.thermal_c)
    smoothed = gaussian(energy, sigma=cfg.sigma, preserve_range=True)
    init_xy = init_snake_from_mask(src_mask, cfg.dilate_factor, cfg.n_points)
    if len(init_xy) < 3:
        return np.zeros(energy.shape, dtype=np.uint8)
    snake_rc = active_contour(
        smoothed, init_xy[:, ::-1],
        alpha=cfg.alpha, beta=cfg.beta, w_line=cfg.w_line, w_edge=cfg.w_edge,
        gamma=cfg.gamma, max_num_iter=cfg.max_iter, boundary_condition="periodic",
    )
    return polygon_to_mask(snake_rc[:, ::-1], energy.shape)
