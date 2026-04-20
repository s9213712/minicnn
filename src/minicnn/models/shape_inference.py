from __future__ import annotations

import math
from typing import Any


def infer_layer_shape(layer_type: str, cfg: dict[str, Any], shape: tuple[int, ...]) -> tuple[int, ...]:
    if layer_type == 'Conv2d':
        c, h, w = shape
        kh = int(cfg.get('kernel_size', 3))
        padding = int(cfg.get('padding', 0))
        stride = int(cfg.get('stride', 1))
        out_c = int(cfg['out_channels'])
        kw = int(cfg.get('kernel_size', 3))
        return (out_c, math.floor((h + 2 * padding - kh) / stride + 1), math.floor((w + 2 * padding - kw) / stride + 1))
    if layer_type == 'MaxPool2d':
        c, h, w = shape
        kernel = int(cfg.get('kernel_size', 2))
        stride = int(cfg.get('stride', kernel))
        return (c, math.floor((h - kernel) / stride + 1), math.floor((w - kernel) / stride + 1))
    if layer_type in {'BatchNorm2d', 'ReLU', 'ResidualBlock'}:
        return shape
    if layer_type == 'Flatten':
        total = 1
        for dim in shape:
            total *= dim
        return (total,)
    if layer_type == 'Linear':
        return (int(cfg['out_features']),)
    return shape
