from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


_MODULE_BASE = nn.Module if nn is not None else object
_SEQUENTIAL_BASE = nn.Sequential if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            'minicnn.extensions.custom_components is torch-only.\n'
            'Install it with:\n'
            '  pip install -e .[torch]'
        )


class Swish(_MODULE_BASE):
    def __init__(self):
        _require_torch()
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(_SEQUENTIAL_BASE):
    def __init__(self, in_channels: int | None = None, out_channels: int = 32, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        _require_torch()
        if in_channels is None:
            raise ValueError('ConvBNReLU requires in_channels when used as a custom factory. Prefer built-in Conv2d + BatchNorm2d + ReLU for auto-inference.')
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
