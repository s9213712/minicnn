"""Input validation for cuda_native graphs and configs.

Validators return a list of error strings (empty = valid) so callers can
collect all problems before raising.
"""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.capabilities import CUDA_NATIVE_CAPABILITIES
from minicnn.model_spec import resolve_model_config

_SUPPORTED_OPS: frozenset[str] = frozenset(CUDA_NATIVE_CAPABILITIES['supported_ops'])


def validate_op_type(op: str, node_name: str = '') -> list[str]:
    """Return errors if *op* is not in the supported set."""
    if op not in _SUPPORTED_OPS:
        loc = f' (node={node_name})' if node_name else ''
        hint = 'cuda_native currently supports: ' + ', '.join(sorted(_SUPPORTED_OPS))
        return [f'Unsupported cuda_native op: {op}{loc}. {hint}']
    return []


def _validate_conv2d_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    errors: list[str] = []
    if 'out_channels' not in attrs:
        errors.append(
            f'Conv2d node={node_name}: missing required attr "out_channels"'
        )
    for key in ('kernel_size', 'stride', 'padding'):
        val = attrs.get(key)
        if val is not None:
            try:
                if isinstance(val, (list, tuple)):
                    [int(v) for v in val]
                else:
                    int(val)
            except (TypeError, ValueError):
                errors.append(
                    f'Conv2d node={node_name}: attr "{key}" must be an integer or pair, got {val!r}'
                )
    return errors


def _validate_linear_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    if 'out_features' not in attrs:
        return [f'Linear node={node_name}: missing required attr "out_features"']
    return []


def _validate_depthwise_conv2d_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    errors: list[str] = []
    for key in ('kernel_size', 'stride', 'padding', 'dilation'):
        val = attrs.get(key)
        if val is None:
            continue
        try:
            if isinstance(val, (list, tuple)):
                [int(v) for v in val]
            else:
                int(val)
        except (TypeError, ValueError):
            errors.append(
                f'DepthwiseConv2d node={node_name}: attr "{key}" must be an integer or pair, got {val!r}'
            )
    for key in ('out_channels', 'channel_multiplier'):
        if key not in attrs:
            continue
        try:
            if int(attrs[key]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f'DepthwiseConv2d node={node_name}: attr "{key}" must be a positive integer, got {attrs[key]!r}'
            )
    return errors


def _validate_pointwise_conv2d_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    if 'out_channels' not in attrs:
        return [f'PointwiseConv2d node={node_name}: missing required attr "out_channels"']
    try:
        if int(attrs['out_channels']) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return [
            f'PointwiseConv2d node={node_name}: attr "out_channels" must be a positive integer, got {attrs["out_channels"]!r}'
        ]
    return []


def _validate_layernorm2d_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    errors: list[str] = []
    for key in ('num_channels', 'channels'):
        if key not in attrs:
            continue
        try:
            if int(attrs[key]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f'LayerNorm2d node={node_name}: attr "{key}" must be a positive integer, got {attrs[key]!r}'
            )
    if 'eps' in attrs:
        try:
            float(attrs['eps'])
        except (TypeError, ValueError):
            errors.append(
                f'LayerNorm2d node={node_name}: attr "eps" must be numeric, got {attrs["eps"]!r}'
            )
    return errors


def _validate_adaptive_avgpool2d_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    output_size = attrs.get('output_size', 1)
    normalized = tuple(output_size) if isinstance(output_size, (list, tuple)) else output_size
    if normalized in {1, (1, 1)}:
        return []
    return [
        f'AdaptiveAvgPool2d node={node_name}: only output_size=1 or (1, 1) is supported by cuda_native, got {output_size!r}'
    ]


def _validate_dropout_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    if 'p' not in attrs:
        return []
    try:
        p = float(attrs['p'])
    except (TypeError, ValueError):
        return [f'Dropout node={node_name}: attr "p" must be numeric, got {attrs["p"]!r}']
    if not (0.0 <= p < 1.0):
        return [f'Dropout node={node_name}: attr "p" must be in [0, 1), got {p!r}']
    return []


def _validate_residual_block_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    errors: list[str] = []
    for key in ('channels', 'out_channels', 'in_channels', 'kernel_size', 'stride'):
        if key not in attrs:
            continue
        try:
            if int(attrs[key]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f'ResidualBlock node={node_name}: attr "{key}" must be a positive integer, got {attrs[key]!r}'
            )
    if 'padding' in attrs:
        try:
            int(attrs['padding'])
        except (TypeError, ValueError):
            errors.append(
                f'ResidualBlock node={node_name}: attr "padding" must be an integer, got {attrs["padding"]!r}'
            )
    return errors


def _validate_convnext_block_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    errors: list[str] = []
    for key in ('channels', 'in_channels', 'kernel_size', 'hidden_channels'):
        if key not in attrs:
            continue
        try:
            if int(attrs[key]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f'ConvNeXtBlock node={node_name}: attr "{key}" must be a positive integer, got {attrs[key]!r}'
            )
    if 'kernel_size' in attrs:
        try:
            kernel_size = int(attrs['kernel_size'])
            if kernel_size % 2 == 0:
                errors.append(
                    f'ConvNeXtBlock node={node_name}: attr "kernel_size" must be odd, got {kernel_size!r}'
                )
        except (TypeError, ValueError):
            pass
    for key in ('expansion_ratio', 'layer_scale_init_value', 'layer_norm_eps'):
        if key not in attrs:
            continue
        try:
            float(attrs[key])
        except (TypeError, ValueError):
            errors.append(
                f'ConvNeXtBlock node={node_name}: attr "{key}" must be numeric, got {attrs[key]!r}'
            )
    return errors


def validate_layer_attrs(op: str, attrs: dict[str, Any], node_name: str) -> list[str]:
    """Validate op-specific attributes."""
    if op == 'Conv2d':
        return _validate_conv2d_attrs(attrs, node_name)
    if op in {'DepthwiseConv2d', 'depthwise_conv2d'}:
        return _validate_depthwise_conv2d_attrs(attrs, node_name)
    if op in {'PointwiseConv2d', 'pointwise_conv2d'}:
        return _validate_pointwise_conv2d_attrs(attrs, node_name)
    if op == 'Linear':
        return _validate_linear_attrs(attrs, node_name)
    if op in {'LayerNorm2d', 'layernorm2d'}:
        return _validate_layernorm2d_attrs(attrs, node_name)
    if op == 'AdaptiveAvgPool2d':
        return _validate_adaptive_avgpool2d_attrs(attrs, node_name)
    if op == 'Dropout':
        return _validate_dropout_attrs(attrs, node_name)
    if op == 'ResidualBlock':
        return _validate_residual_block_attrs(attrs, node_name)
    if op in {'ConvNeXtBlock', 'convnext_block'}:
        return _validate_convnext_block_attrs(attrs, node_name)
    return []


def validate_layer_list(layers: list[dict[str, Any]]) -> list[str]:
    """Validate a list of layer dicts from a model config."""
    errors: list[str] = []
    if not isinstance(layers, list):
        return ['model.layers must be a list']
    for i, layer in enumerate(layers):
        op = str(layer.get('type', ''))
        if not op:
            errors.append(f'Layer {i}: missing "type" key')
            continue
        node_name = f'layer_{i}'
        op_errors = validate_op_type(op, node_name=node_name)
        errors.extend(op_errors)
        if not op_errors:
            attrs = {k: v for k, v in layer.items() if k != 'type'}
            errors.extend(validate_layer_attrs(op, attrs, node_name))
    return errors


def validate_cuda_native_model_config(model_cfg: dict[str, Any]) -> list[str]:
    """Validate a model config dict for cuda_native compatibility."""
    model_cfg = resolve_model_config(model_cfg)
    layers = model_cfg.get('layers', [])
    return validate_layer_list(layers)
