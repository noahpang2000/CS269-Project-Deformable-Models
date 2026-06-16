"""Deep Snake-style contour deformation (Peng et al. 2020, simplified).

Faithful to the paper's core idea -- iterative contour deformation by circular
convolution over the vertex sequence -- but NOT a port of the official pipeline:
the CenterNet detector + deformable-conv ops are replaced by a coarse
segmentation head whose connected components seed the initial contours.

The backbone is the SAME U-Net used by the unet/dals baselines (shared so the
deep comparison isolates the head, not the feature extractor): its decoder
features are sampled at each contour vertex, its 1x1 head gives the coarse seg.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flame.deep.unet import UNet


class CircConv(nn.Module):
    """1D convolution with circular padding -- treats the contour as periodic."""

    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv1d(cin, cout, k)

    def forward(self, x):  # x: [B, C, N]
        x = torch.cat([x[..., -self.pad:], x, x[..., :self.pad]], dim=-1)
        return self.conv(x)


class SnakeHead(nn.Module):
    def __init__(self, feat_c: int, hidden: int = 128, n_layers: int = 4, k: int = 9):
        super().__init__()
        chans = [feat_c + 2] + [hidden] * n_layers
        self.blocks = nn.ModuleList(CircConv(chans[i], chans[i + 1], k) for i in range(n_layers))
        self.out = nn.Conv1d(hidden, 2, 1)

    def forward(self, feat_at_verts, coords_norm):  # [B,Cf,N], [B,2,N]
        x = torch.cat([feat_at_verts, coords_norm], dim=1)
        for b in self.blocks:
            x = F.relu(b(x))
        return self.out(x)  # offsets [B,2,N]


class DeepSnake(nn.Module):
    def __init__(self, n_iter: int = 3, base: int = 32):
        super().__init__()
        self.n_iter = n_iter
        self.unet = UNet(in_ch=3, out_ch=1, base=base)
        self.snakes = nn.ModuleList(SnakeHead(base) for _ in range(n_iter))

    def _sample(self, feat, contour, size):
        # contour [B,N,2] in pixel coords -> normalised grid for grid_sample
        grid = (contour / (size - 1)) * 2 - 1
        sampled = F.grid_sample(feat, grid.unsqueeze(1), align_corners=True)  # [B,C,1,N]
        return sampled.squeeze(2)

    def forward(self, image, init_contour=None):
        """Coarse seg logits [B,1,H,W] + list of deformed contours [B,N,2].

        With init_contour=None only the coarse head runs (used at inference to
        seed contours from connected components).
        """
        size = image.shape[-1]
        coarse, feat = self.unet(image, return_features=True)
        if init_contour is None:
            return coarse, []
        contour = init_contour
        outputs = []
        for head in self.snakes:
            verts = self._sample(feat, contour, size)               # [B,base,N]
            coords_n = ((contour / (size - 1)) * 2 - 1).transpose(1, 2)  # [B,2,N]
            offset = head(verts, coords_n).transpose(1, 2)          # [B,N,2]
            contour = contour + offset
            outputs.append(contour)
        return coarse, outputs
