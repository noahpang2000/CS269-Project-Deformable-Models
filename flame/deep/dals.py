"""DALS-style deep active level set (Hatamizadeh et al. 2019, simplified).

A U-Net trunk predicts an initial level set phi0; a differentiable Chan-Vese
evolution layer then unrolls K steps with learnable weights (mu, lambda1,
lambda2, dt). Region means c1/c2 are computed from the evolving soft Heaviside
of the input intensity. Output: Heaviside(phi_K). Native topology via the level
set, so multi-component fires need no special handling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flame.deep.unet import UNet


def _heaviside(phi: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    return 0.5 * (1 + (2 / torch.pi) * torch.atan(phi / eps))


def _dirac(phi: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    return (eps / torch.pi) / (eps ** 2 + phi ** 2)


def _derivatives(phi: torch.Tensor):
    p = F.pad(phi, (1, 1, 1, 1), mode="replicate")
    px = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) / 2
    py = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) / 2
    pxx = p[..., 1:-1, 2:] - 2 * phi + p[..., 1:-1, :-2]
    pyy = p[..., 2:, 1:-1] - 2 * phi + p[..., :-2, 1:-1]
    pxy = (p[..., 2:, 2:] - p[..., 2:, :-2] - p[..., :-2, 2:] + p[..., :-2, :-2]) / 4
    return px, py, pxx, pyy, pxy


def _curvature(phi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    px, py, pxx, pyy, pxy = _derivatives(phi)
    num = pxx * py ** 2 - 2 * px * py * pxy + pyy * px ** 2
    return num / (px ** 2 + py ** 2 + eps) ** 1.5


class DALS(nn.Module):
    def __init__(self, n_iter: int = 5, base: int = 32,
                 mu: float = 0.2, lam1: float = 1.0,
                 lam2: float = 1.0, dt: float = 0.1):
        super().__init__()
        self.trunk = UNet(in_ch=3, out_ch=1, base=base)
        self.n_iter = n_iter
        self.mu = nn.Parameter(torch.tensor(float(mu)))
        self.lam1 = nn.Parameter(torch.tensor(float(lam1)))
        self.lam2 = nn.Parameter(torch.tensor(float(lam2)))
        self.dt = nn.Parameter(torch.tensor(float(dt)))

    def forward(self, x):
        phi = self.trunk(x)                       # initial level set (logits)
        prob_logits = phi                         # intermediate per-pixel head
        intensity = x.mean(dim=1, keepdim=True)   # grayscale region feature
        for _ in range(self.n_iter):
            h = _heaviside(phi)
            c1 = (h * intensity).sum(dim=(2, 3), keepdim=True) / (h.sum(dim=(2, 3), keepdim=True) + 1e-6)
            c2 = ((1 - h) * intensity).sum(dim=(2, 3), keepdim=True) / ((1 - h).sum(dim=(2, 3), keepdim=True) + 1e-6)
            region = -self.lam1 * (intensity - c1) ** 2 + self.lam2 * (intensity - c2) ** 2
            dphi = _dirac(phi) * (self.mu * _curvature(phi) + region)
            phi = phi + self.dt * dphi
        return phi, prob_logits
