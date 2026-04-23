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


def _validate_groupnorm_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    errors: list[str] = []
    if 'num_groups' not in attrs:
        errors.append(
            f'GroupNorm node={node_name}: missing required attr "num_groups"'
        )
    else:
        try:
            if int(attrs['num_groups']) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f'GroupNorm node={node_name}: attr "num_groups" must be a positive integer, got {attrs["num_groups"]!r}'
            )
    for key in ('num_channels', 'channels'):
        if key not in attrs:
            continue
        try:
            if int(attrs[key]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f'GroupNorm node={node_name}: attr "{key}" must be a positive integer, got {attrs[key]!r}'
            )
    if 'eps' in attrs:
        try:
            float(attrs['eps'])
        except (TypeError, ValueError):
            errors.append(
                f'GroupNorm node={node_name}: attr "eps" must be numeric, got {attrs["eps"]!r}'
            )
    return errors


def _validate_layernorm_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    normalized_shape = attrs.get('normalized_shape')
    if normalized_shape is None:
        return [f'LayerNorm node={node_name}: missing required attr "normalized_shape"']
    values = normalized_shape if isinstance(normalized_shape, (list, tuple)) else [normalized_shape]
    errors: list[str] = []
    if len(values) == 0:
        errors.append(
            f'LayerNorm node={node_name}: attr "normalized_shape" must contain at least one dimension'
        )
    for idx, value in enumerate(values):
        try:
            if int(value) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f'LayerNorm node={node_name}: normalized_shape[{idx}] must be a positive integer, got {value!r}'
            )
    if 'eps' in attrs:
        try:
            float(attrs['eps'])
        except (TypeError, ValueError):
            errors.append(
                f'LayerNorm node={node_name}: attr "eps" must be numeric, got {attrs["eps"]!r}'
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


def _validate_droppath_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    if 'p' not in attrs:
        return []
    try:
        p = float(attrs['p'])
    except (TypeError, ValueError):
        return [f'DropPath node={node_name}: attr "p" must be numeric, got {attrs["p"]!r}']
    if not (0.0 <= p < 1.0):
        return [f'DropPath node={node_name}: attr "p" must be in [0, 1), got {p!r}']
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


def _validate_add_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    inputs = attrs.get('inputs')
    if not isinstance(inputs, list):
        return [f'Add node={node_name}: attr "inputs" must be a list of tensor names']
    if len(inputs) < 2:
        return [f'Add node={node_name}: attr "inputs" must contain at least two tensor names']
    errors: list[str] = []
    for idx, name in enumerate(inputs):
        if not str(name).strip():
            errors.append(
                f'Add node={node_name}: inputs[{idx}] must be a non-empty tensor name, got {name!r}'
            )
    return errors


def _validate_concat_attrs(attrs: dict[str, Any], node_name: str) -> list[str]:
    inputs = attrs.get('inputs')
    if not isinstance(inputs, list):
        return [f'Concat node={node_name}: attr "inputs" must be a list of tensor names']
    if len(inputs) < 2:
        return [f'Concat node={node_name}: attr "inputs" must contain at least two tensor names']
    errors: list[str] = []
    for idx, name in enumerate(inputs):
        if not str(name).strip():
            errors.append(
                f'Concat node={node_name}: inputs[{idx}] must be a non-empty tensor name, got {name!r}'
            )
    if 'axis' in attrs:
        try:
            int(attrs['axis'])
        except (TypeError, ValueError):
            errors.append(
                f'Concat node={node_name}: attr "axis" must be an integer, got {attrs["axis"]!r}'
            )
    return errors


def _validate_graph_binding_attrs(op: str, attrs: dict[str, Any], node_name: str) -> list[str]:
    errors: list[str] = []
    if 'name' in attrs and not str(attrs['name']).strip():
        errors.append(f'{op} node={node_name}: attr "name" must be a non-empty string')
    if 'output' in attrs and not str(attrs['output']).strip():
        errors.append(f'{op} node={node_name}: attr "output" must be a non-empty tensor name')
    if 'inputs' in attrs and op not in {'Add', 'Concat'}:
        inputs = attrs['inputs']
        if not isinstance(inputs, list):
            errors.append(f'{op} node={node_name}: attr "inputs" must be a list of tensor names')
        elif len(inputs) != 1:
            errors.append(
                f'{op} node={node_name}: explicit attr "inputs" currently requires exactly one tensor name'
            )
        elif not str(inputs[0]).strip():
            errors.append(
                f'{op} node={node_name}: inputs[0] must be a non-empty tensor name, got {inputs[0]!r}'
            )
    return errors


def validate_layer_attrs(op: str, attrs: dict[str, Any], node_name: str) -> list[str]:
    """Validate op-specific attributes."""
    errors = _validate_graph_binding_attrs(op, attrs, node_name)
    if op == 'Conv2d':
        return errors + _validate_conv2d_attrs(attrs, node_name)
    if op in {'DepthwiseConv2d', 'depthwise_conv2d'}:
        return errors + _validate_depthwise_conv2d_attrs(attrs, node_name)
    if op in {'PointwiseConv2d', 'pointwise_conv2d'}:
        return errors + _validate_pointwise_conv2d_attrs(attrs, node_name)
    if op == 'Linear':
        return errors + _validate_linear_attrs(attrs, node_name)
    if op == 'GroupNorm':
        return errors + _validate_groupnorm_attrs(attrs, node_name)
    if op == 'LayerNorm':
        return errors + _validate_layernorm_attrs(attrs, node_name)
    if op in {'LayerNorm2d', 'layernorm2d'}:
        return errors + _validate_layernorm2d_attrs(attrs, node_name)
    if op == 'AdaptiveAvgPool2d':
        return errors + _validate_adaptive_avgpool2d_attrs(attrs, node_name)
    if op == 'Add':
        return errors + _validate_add_attrs(attrs, node_name)
    if op == 'Concat':
        return errors + _validate_concat_attrs(attrs, node_name)
    if op == 'Dropout':
        return errors + _validate_dropout_attrs(attrs, node_name)
    if op == 'DropPath':
        return errors + _validate_droppath_attrs(attrs, node_name)
    if op == 'ResidualBlock':
        return errors + _validate_residual_block_attrs(attrs, node_name)
    if op in {'ConvNeXtBlock', 'convnext_block'}:
        return errors + _validate_convnext_block_attrs(attrs, node_name)
    return errors


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
