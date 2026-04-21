"""Input validation for cuda_native graphs and configs.

Validators return a list of error strings (empty = valid) so callers can
collect all problems before raising.
"""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.capabilities import CUDA_NATIVE_CAPABILITIES

_SUPPORTED_OPS: frozenset[str] = frozenset(CUDA_NATIVE_CAPABILITIES['supported_ops'])


def validate_op_type(op: str, node_name: str = '') -> list[str]:
    """Return errors if *op* is not in the supported set."""
    if op not in _SUPPORTED_OPS:
        loc = f' (node={node_name})' if node_name else ''
        hint = 'cuda_native currently supports: ' + ', '.join(sorted(_SUPPORTED_OPS))
        return [f'Unsupported cuda_native op: {op}{loc}. {hint}']
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
        errors.extend(validate_op_type(op, node_name=f'layer_{i}'))
    return errors


def validate_cuda_native_model_config(model_cfg: dict[str, Any]) -> list[str]:
    """Validate a model config dict for cuda_native compatibility."""
    layers = model_cfg.get('layers', [])
    return validate_layer_list(layers)
