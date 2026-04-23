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


def validate_layer_attrs(op: str, attrs: dict[str, Any], node_name: str) -> list[str]:
    """Validate op-specific attributes."""
    if op == 'Conv2d':
        return _validate_conv2d_attrs(attrs, node_name)
    if op == 'Linear':
        return _validate_linear_attrs(attrs, node_name)
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
