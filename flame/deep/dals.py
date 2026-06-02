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


# class DALS(nn.Module):
#     def __init__(self, n_iter: int = 5, base: int = 32):
#         super().__init__()
#         self.trunk = UNet(in_ch=3, out_ch=1, base=base)
#         self.n_iter = n_iter
#         self.mu = nn.Parameter(torch.tensor(0.2))
#         self.lam1 = nn.Parameter(torch.tensor(1.0))
#         self.lam2 = nn.Parameter(torch.tensor(1.0))
#         self.dt = nn.Parameter(torch.tensor(0.1))

#     def forward(self, x):
#         phi = self.trunk(x)                       # initial level set (logits)
#         prob_logits = phi                         # intermediate per-pixel head
#         intensity = x.mean(dim=1, keepdim=True)   # grayscale region feature
#         for _ in range(self.n_iter):
#             h = _heaviside(phi)
#             c1 = (h * intensity).sum(dim=(2, 3), keepdim=True) / (h.sum(dim=(2, 3), keepdim=True) + 1e-6)
#             c2 = ((1 - h) * intensity).sum(dim=(2, 3), keepdim=True) / ((1 - h).sum(dim=(2, 3), keepdim=True) + 1e-6)
#             region = -self.lam1 * (intensity - c1) ** 2 + self.lam2 * (intensity - c2) ** 2
#             dphi = _dirac(phi) * (self.mu * _curvature(phi) + region)
#             phi = phi + self.dt * dphi
#         return phi, prob_logits

class DALS(nn.Module):
    """
    Refactored DALS. 
    Evolves the level set over deep semantic features with spatially varying parameters.
    """
    def __init__(self, n_iter: int = 10, base: int = 32):
        super().__init__()
        self.n_iter = n_iter
        
        # 1. The Trunk now outputs deep features, not just a 1-channel mask guess.
        # We assume the UNet is modified to output its final feature map (e.g., 32 channels)
        self.trunk = UNet(in_ch=3, out_ch=base, base=base) 
        
        # 2. Prediction Heads
        self.phi_head = nn.Conv2d(base, 1, kernel_size=1)
        
        # Predicts 3 spatial parameters (mu, lam1, lam2) for every pixel
        self.params_head = nn.Conv2d(base, 3, kernel_size=1)
        
        # dt can remain a global learnable scalar for temporal stability
        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # Extract deep semantic features [B, C, H, W]
        features = self.trunk(x, return_features=True)                       
        
        # Predict initial level set [B, 1, H, W]
        phi = self.phi_head(features)
        prob_logits = phi.clone() # Keep the pre-evolution logits for the auxiliary loss

        # Predict spatially varying parameters [B, 3, H, W]
        # We use softplus to ensure all PDE parameters are strictly positive
        params = F.softplus(self.params_head(features)) 
        mu = params[:, 0:1, :, :]
        lam1 = params[:, 1:2, :, :]
        lam2 = params[:, 2:3, :, :]

        for _ in range(self.n_iter):
            h = _heaviside(phi)
            
            # --- THE DEEP FEATURE UPGRADE ---
            # Compute region means IN FEATURE SPACE. c1 and c2 are now vectors of size C,
            # representing the average semantic feature inside and outside the contour.
            h_sum = h.sum(dim=(2, 3), keepdim=True) + 1e-6
            inv_h_sum = (1 - h).sum(dim=(2, 3), keepdim=True) + 1e-6
            
            c1 = (h * features).sum(dim=(2, 3), keepdim=True) / h_sum
            c2 = ((1 - h) * features).sum(dim=(2, 3), keepdim=True) / inv_h_sum
            
            # Region energies are now the squared L2 distance in the high-dimensional feature space
            # dist1, dist2 shape: [B, 1, H, W]
            dist1 = ((features - c1) ** 2).sum(dim=1, keepdim=True)
            dist2 = ((features - c2) ** 2).sum(dim=1, keepdim=True)
            
            # Note: lam1 and lam2 are now spatial matrices, not global scalars
            region = -lam1 * dist1 + lam2 * dist2
            
            # --- EVOLUTION ---
            # mu is also a spatial matrix now, allowing variable smoothness
            dphi = _dirac(phi) * (mu * _curvature(phi) + region)
            phi = phi + self.dt * dphi
            
        return phi, prob_logits