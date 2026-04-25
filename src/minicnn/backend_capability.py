from __future__ import annotations

from typing import Any

from minicnn.cuda_native.capabilities import CUDA_NATIVE_CAPABILITIES
from minicnn.model_spec import resolve_model_config

KNOWN_BACKENDS = frozenset({'torch', 'cuda_legacy', 'cuda_native'})

_TORCH_SUPPORTED_OPS = sorted({
    'AdaptiveAvgPool2d',
    'AvgPool2d',
    'BatchNorm2d',
    'Conv2d',
    'ConvNeXtBlock',
    'DepthwiseConv2d',
    'Dropout',
    'Flatten',
    'GELU',
    'GlobalAvgPool2d',
    'GroupNorm',
    'Identity',
    'LayerNorm',
    'LayerNorm2d',
    'LeakyReLU',
    'Linear',
    'MaxPool2d',
    'PointwiseConv2d',
    'ReLU',
    'ResidualBlock',
    'SiLU',
    'Sigmoid',
    'Tanh',
    'convnext_block',
    'depthwise_conv2d',
    'layernorm2d',
    'pointwise_conv2d',
})

_CUDA_LEGACY_SUPPORTED_OPS = sorted({
    'Conv2d',
    'Flatten',
    'LeakyReLU',
    'Linear',
    'MaxPool2d',
    'ReLU',
})

_CUDA_NATIVE_SUPPORTED_OPS = sorted(str(item) for item in CUDA_NATIVE_CAPABILITIES['supported_ops'])

_LAYER_CAPABILITY_FLAG = {
    'DepthwiseConv2d': 'supports_depthwise_conv',
    'depthwise_conv2d': 'supports_depthwise_conv',
    'GroupNorm': 'supports_groupnorm',
    'LayerNorm': 'supports_layernorm',
    'PointwiseConv2d': 'supports_pointwise_conv',
    'pointwise_conv2d': 'supports_pointwise_conv',
    'LayerNorm2d': 'supports_layernorm2d',
    'layernorm2d': 'supports_layernorm2d',
    'GELU': 'supports_gelu',
    'ConvNeXtBlock': 'supports_convnext_block',
    'convnext_block': 'supports_convnext_block',
}


_BACKEND_CAPABILITIES: dict[str, dict[str, Any]] = {
    'torch': {
        'backend': 'torch',
        'supports_depthwise_conv': True,
        'supports_pointwise_conv': True,
        'supports_groupnorm': True,
        'supports_layernorm': True,
        'supports_layernorm2d': True,
        'supports_gelu': True,
        'supports_residual_add': True,
        'supports_convnext_block': True,
        'supported_ops': _TORCH_SUPPORTED_OPS,
    },
    'cuda_legacy': {
        'backend': 'cuda_legacy',
        'supports_depthwise_conv': False,
        'supports_pointwise_conv': False,
        'supports_groupnorm': False,
        'supports_layernorm': False,
        'supports_layernorm2d': False,
        'supports_gelu': False,
        'supports_residual_add': False,
        'supports_convnext_block': False,
        'supported_ops': _CUDA_LEGACY_SUPPORTED_OPS,
    },
    'cuda_native': {
        'backend': 'cuda_native',
        'supports_depthwise_conv': True,
        'supports_pointwise_conv': True,
        'supports_groupnorm': True,
        'supports_layernorm': True,
        'supports_layernorm2d': True,
        'supports_gelu': True,
        'supports_residual_add': True,
        'supports_convnext_block': True,
        'supported_ops': _CUDA_NATIVE_SUPPORTED_OPS,
    },
}


def validate_backend_name(backend: str | None) -> list[str]:
    normalized = str(backend or 'torch')
    if normalized not in KNOWN_BACKENDS:
        supported = ', '.join(sorted(KNOWN_BACKENDS))
        return [f"Unknown backend {normalized!r}. Supported backends: {supported}"]
    return []


def get_backend_capabilities(backend: str | None) -> dict[str, Any]:
    normalized = str(backend or 'torch')
    if normalized not in _BACKEND_CAPABILITIES:
        supported = ', '.join(sorted(KNOWN_BACKENDS))
        raise ValueError(f"Unknown backend {normalized!r}. Supported backends: {supported}")
    return dict(_BACKEND_CAPABILITIES[normalized])


def validate_backend_model_capabilities(model_cfg: dict[str, Any], backend: str | None) -> list[str]:
    backend_errors = validate_backend_name(backend)
    if backend_errors:
        return backend_errors
    resolved = resolve_model_config(model_cfg)
    caps = get_backend_capabilities(backend)
    supported_ops = set(str(item) for item in caps['supported_ops'])
    errors: list[str] = []
    layers = resolved.get('layers', [])
    if not isinstance(layers, list):
        return ['model.layers must be a list']
    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            errors.append(f'model.layers[{idx}] must be a mapping')
            continue
        op = str(layer.get('type', ''))
        if not op:
            errors.append(f'model.layers[{idx}] is missing required key: type')
            continue
        if op not in supported_ops:
            supported_text = ', '.join(sorted(supported_ops))
            errors.append(
                f'backend {caps["backend"]} does not support model.layers[{idx}].type={op!r}. '
                f'Supported ops: {supported_text}'
            )
            continue
        capability_flag = _LAYER_CAPABILITY_FLAG.get(op)
        if capability_flag and not bool(caps.get(capability_flag, False)):
            errors.append(
                f'backend {caps["backend"]} does not support {op} '
                f'({capability_flag}=false).'
            )
    return errors
