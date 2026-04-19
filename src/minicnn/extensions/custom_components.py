from __future__ import annotations

import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int | None = None, out_channels: int = 32, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        if in_channels is None:
            raise ValueError('ConvBNReLU requires in_channels when used as a custom factory. Prefer built-in Conv2d + BatchNorm2d + ReLU for auto-inference.')
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
