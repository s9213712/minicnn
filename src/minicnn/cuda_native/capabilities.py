"""Capability surface for the cuda_native backend.

This module is the single source of truth for what cuda_native currently
supports.  All other modules (api, validators, diagnostics) should read
from here rather than duplicating the list.
"""
from __future__ import annotations

CUDA_NATIVE_CAPABILITIES: dict[str, object] = {
    'experimental': True,
    'sequential_only': True,
    'forward_only': True,
    'training': False,
    'backward': False,
    'dynamic_shapes': False,
    'branching_graph': False,
    'amp': False,
    'supported_ops': [
        'BatchNorm2d',
        'Conv2d',
        'ReLU',
        'LeakyReLU',
        'Flatten',
        'Linear',
        'MaxPool2d',
        'AvgPool2d',
    ],
    'planned_ops': [],
    'unsupported_ops': [
        'GroupNorm',
        'LayerNorm',
        'ResidualBlock',
    ],
}


def get_cuda_native_capabilities() -> dict[str, object]:
    """Return a copy of the current cuda_native capability descriptor."""
    caps = dict(CUDA_NATIVE_CAPABILITIES)
    caps['supported_ops'] = list(caps['supported_ops'])
    caps['planned_ops'] = list(caps['planned_ops'])
    caps['unsupported_ops'] = list(caps['unsupported_ops'])
    return caps
