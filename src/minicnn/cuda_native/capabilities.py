"""Capability surface for the cuda_native backend.

This module is the single source of truth for what cuda_native currently
supports.  All other modules (api, validators, diagnostics) should read
from here rather than duplicating the list.
"""
from __future__ import annotations

from typing import Any

CAPABILITY_SCHEMA_VERSION = 1

CUDA_NATIVE_CAPABILITIES: dict[str, object] = {
    'experimental': True,
    'production_ready': False,
    'numpy_reference': True,
    'sequential_only': False,
    'forward_only': False,
    'training': True,
    'training_stable': False,
    'backward': True,
    'backward_stable': False,
    'dynamic_shapes': False,
    'branching_graph': True,
    'amp': True,
    'supports_depthwise_conv': True,
    'supports_pointwise_conv': True,
    'supports_groupnorm': True,
    'supports_layernorm': True,
    'supports_layernorm2d': True,
    'supports_gelu': True,
    'supports_residual_add': True,
    'supports_convnext_block': True,
    'supported_datasets': ['random', 'cifar10', 'mnist'],
    'supported_losses': ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss'],
    'supported_optimizers': ['SGD', 'Adam', 'AdamW', 'RMSprop'],
    'supported_schedulers': [
        'StepLR',
        'CosineAnnealingLR',
        'ReduceLROnPlateau',
    ],
    'supported_ops': [
        'BatchNorm2d',
        'Concat',
        'Conv2d',
        'DepthwiseConv2d',
        'PointwiseConv2d',
        'GroupNorm',
        'LayerNorm',
        'LayerNorm2d',
        'DropPath',
        'Dropout',
        'ReLU',
        'LeakyReLU',
        'Sigmoid',
        'Tanh',
        'SiLU',
        'GELU',
        'Flatten',
        'Linear',
        'MaxPool2d',
        'AvgPool2d',
        'AdaptiveAvgPool2d',
        'Add',
        'GlobalAvgPool2d',
        'Identity',
        'ResidualBlock',
        'ConvNeXtBlock',
        'depthwise_conv2d',
        'pointwise_conv2d',
        'layernorm2d',
        'convnext_block',
    ],
    'planned_ops': [],
    'unsupported_ops': [
        'Embedding',
        'SelfAttention',
        'Upsample',
    ],
    'notes': [
        'Backward and training are research prototypes, not production-ready.',
        'BatchNorm2d forward/backward exist as prototypes; training remains experimental.',
        'DepthwiseConv2d, PointwiseConv2d, GroupNorm, LayerNorm, LayerNorm2d, GELU, and global pooling use numpy reference kernels.',
        'ResidualBlock, ConvNeXtBlock, Dropout, and DropPath run through experimental composite/reference numpy kernels.',
        'Explicit ordered DAG wiring is supported through named tensor outputs plus Add/Concat multi-input nodes.',
        'train-native supports SGD, Adam, AdamW, RMSprop, BCEWithLogitsLoss, label_smoothing for cross entropy, grad_accum_steps >= 1, and experimental AMP with loss scaling / overflow backoff.',
        'validate-cuda-native-config enforces the current train-native support boundary.',
    ],
}


def _sorted_unique_strings(items: object) -> list[str]:
    return sorted({str(item) for item in items if str(item)})


def _kernel_registry_surface() -> list[dict[str, str]]:
    from minicnn.cuda_native.kernels import DEFAULT_KERNEL_SPECS

    return [
        {
            'op_name': spec.op_name,
            'category': spec.category,
        }
        for spec in sorted(DEFAULT_KERNEL_SPECS, key=lambda spec: spec.op_name)
    ]


def get_cuda_native_capabilities() -> dict[str, Any]:
    """Return a versioned, machine-readable cuda_native capability descriptor."""
    caps = dict(CUDA_NATIVE_CAPABILITIES)
    caps['supported_ops'] = _sorted_unique_strings(caps['supported_ops'])
    caps['planned_ops'] = _sorted_unique_strings(caps['planned_ops'])
    caps['unsupported_ops'] = _sorted_unique_strings(caps['unsupported_ops'])
    caps['supported_datasets'] = _sorted_unique_strings(caps['supported_datasets'])
    caps['supported_losses'] = _sorted_unique_strings(caps['supported_losses'])
    caps['supported_optimizers'] = _sorted_unique_strings(caps['supported_optimizers'])
    caps['supported_schedulers'] = _sorted_unique_strings(caps['supported_schedulers'])
    caps['notes'] = [str(note) for note in caps['notes']]
    kernel_surface = _kernel_registry_surface()
    caps.update({
        'schema_version': CAPABILITY_SCHEMA_VERSION,
        'backend': 'cuda_native',
        'status': 'ok',
        'summary_status': 'experimental',
        'capability_kind': 'backend_capability_summary',
        'supported_op_categories': sorted({entry['category'] for entry in kernel_surface}),
        'kernel_registry_surface': kernel_surface,
    })
    return caps
