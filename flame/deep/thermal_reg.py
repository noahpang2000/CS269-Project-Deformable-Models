"""Cross-modal thermal-regression regularizer (RGB -> continuous temperature).

Inverse-problem view: recovering the >=150C support x from RGB y is ill-posed
because the smoke-occlusion forward operator A nearly collapses fire and
background in RGB (the color floor scores ~0). Rather than learn y->x directly
(a sparse binary target that discards almost all signal about A), we learn the
structured intermediate y -> T_hat (a continuous temperature field) and set
x = 1[T_hat >= 150C]. The dense graded target regularizes the hypothesis space
toward maps that explain the whole thermal field, i.e. that internalize A^{-1}.

Normalization: Celsius is mapped linearly to [0,1] over [TMIN, TMAX] and clipped.
The 150C decision threshold maps to a fixed constant THR_NORM, so thresholding
T_hat is a parameter-free physical operation, not a learned cutoff.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from flame.deep.unet import UNet

TMIN, TMAX = -50.0, 400.0          # clip range; covers the bulk, caps the rare 600C spikes
THR_C = 150.0
THR_NORM = (THR_C - TMIN) / (TMAX - TMIN)   # ~0.444


def celsius_to_norm(t_c: torch.Tensor | "np.ndarray"):
    return ((t_c - TMIN) / (TMAX - TMIN)).clip(0.0, 1.0)


class ThermalRegUNet(nn.Module):
    """U-Net trunk with a 1-channel regression head + sigmoid -> normalized temp."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.unet = UNet(in_ch=3, out_ch=1, base=base)

    def forward(self, x):
        # UNet head already gives [B,1,H,W] logits; sigmoid maps to normalized [0,1] temp.
        return torch.sigmoid(self.unet(x))
