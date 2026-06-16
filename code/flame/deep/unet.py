"""Compact U-Net segmentation baseline (control: pure CNN, no contour head)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.pool = nn.MaxPool2d(2)
        self.d1, self.d2 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1])
        self.d3, self.d4 = DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.u3, self.u2 = DoubleConv(c[3], c[2]), DoubleConv(c[2], c[1])
        self.u1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x, return_features: bool = False):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        y = self.u3(torch.cat([self.up3(x4), x3], dim=1))
        y = self.u2(torch.cat([self.up2(y), x2], dim=1))
        feat = self.u1(torch.cat([self.up1(y), x1], dim=1))  # [B, base, H, W]
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits
