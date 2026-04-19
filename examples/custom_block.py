from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


class Swish(nn.Module if torch is not None else object):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Module if torch is not None else object):
    def __init__(self, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        if torch is None:
            raise RuntimeError('PyTorch is required for ConvBNReLU')
        self.block = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class CustomHead(nn.Module if torch is not None else object):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if torch is None:
            raise RuntimeError('PyTorch is required for CustomHead')
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
