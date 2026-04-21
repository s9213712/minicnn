"""Capability surface for the cuda_native backend.

This module is the single source of truth for what cuda_native currently
supports.  All other modules (api, validators, diagnostics) should read
from here rather than duplicating the list.
"""
from __future__ import annotations

CUDA_NATIVE_CAPABILITIES: dict[str, object] = {
    'experimental': True,
    'production_ready': False,
    'numpy_reference': True,
    'sequential_only': True,
    'forward_only': False,
    'training': True,
    'training_stable': False,
    'backward': True,
    'backward_stable': False,
    'dynamic_shapes': False,
    'branching_graph': False,
    'amp': False,
    'supported_datasets': ['random', 'cifar10', 'mnist'],
    'supported_losses': ['CrossEntropyLoss', 'MSELoss'],
    'supported_optimizers': ['SGD'],
    'supported_schedulers': [],
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
    'notes': [
        'Backward and training are research prototypes, not production-ready.',
        'BatchNorm2d forward exists, but backward is not implemented yet.',
        'validate-cuda-native-config enforces the current train-native contract.',
    ],
}


def get_cuda_native_capabilities() -> dict[str, object]:
    """Return a copy of the current cuda_native capability descriptor."""
    caps = dict(CUDA_NATIVE_CAPABILITIES)
    caps['supported_ops'] = list(caps['supported_ops'])
    caps['planned_ops'] = list(caps['planned_ops'])
    caps['unsupported_ops'] = list(caps['unsupported_ops'])
    caps['supported_datasets'] = list(caps['supported_datasets'])
    caps['supported_losses'] = list(caps['supported_losses'])
    caps['supported_optimizers'] = list(caps['supported_optimizers'])
    caps['supported_schedulers'] = list(caps['supported_schedulers'])
    caps['notes'] = list(caps['notes'])
    return caps
