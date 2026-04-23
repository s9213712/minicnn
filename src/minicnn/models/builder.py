from __future__ import annotations

from copy import deepcopy
from typing import Any

from minicnn.model_spec import resolve_model_config
from minicnn.models.registry import get_model_component
from minicnn.models.shape_inference import infer_layer_shape
from minicnn.nn import Sequential


def _validate_shape(layer_type: str, shape: tuple[int, ...]) -> None:
    if any(int(dim) <= 0 for dim in shape):
        raise ValueError(f'{layer_type} inferred invalid non-positive shape: {shape}')


def build_model_from_config(
    model_cfg: dict[str, Any],
    input_shape: list[int] | tuple[int, ...] | None = None,
    rng=None,
):
    if not isinstance(model_cfg, dict):
        raise TypeError('model config must be a mapping')
    model_cfg = resolve_model_config(model_cfg)
    input_shape = tuple(input_shape or model_cfg.get('input_shape', [1, 4, 4]))
    _validate_shape('input', input_shape)
    shape = input_shape
    modules = []
    history = [shape]
    layers = model_cfg.get('layers', [])
    if not isinstance(layers, list):
        raise TypeError('model.layers must be a list of layer mappings')
    for idx, raw in enumerate(layers):
        if not isinstance(raw, dict):
            raise TypeError(f'model.layers[{idx}] must be a mapping')
        cfg = deepcopy(raw)
        layer_type = cfg.pop('type', None)
        if not layer_type:
            raise ValueError(f'model.layers[{idx}] is missing required key: type')
        if layer_type in {'Conv2d', 'BatchNorm2d', 'ResidualBlock'} and len(shape) != 3:
            raise ValueError(f'{layer_type} expects CHW input shape, got {shape}')
        if layer_type == 'Linear' and len(shape) != 1:
            raise ValueError(f'Linear expects flat input shape, got {shape}; add Flatten first')
        if rng is not None and layer_type in {'Conv2d', 'Linear'} and 'rng' not in cfg:
            cfg['rng'] = rng
        if layer_type == 'Conv2d' and 'in_channels' not in cfg:
            cfg['in_channels'] = shape[0]
        if layer_type == 'BatchNorm2d' and 'num_features' not in cfg:
            cfg['num_features'] = shape[0]
        if layer_type == 'ResidualBlock' and 'channels' not in cfg:
            cfg['channels'] = shape[0]
        if layer_type == 'Linear' and 'in_features' not in cfg:
            cfg['in_features'] = shape[0]
        module = get_model_component(layer_type)(**cfg)
        modules.append(module)
        shape = infer_layer_shape(layer_type, cfg, shape)
        _validate_shape(layer_type, shape)
        history.append(shape)
    model = Sequential(*modules)
    model.input_shape = input_shape
    model.inferred_shapes = history
    return model, shape
