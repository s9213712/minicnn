from __future__ import annotations

from copy import deepcopy
from typing import Any

from minicnn.models.registry import get_model_component
from minicnn.models.shape_inference import infer_layer_shape
from minicnn.nn import Sequential


def build_model_from_config(model_cfg: dict[str, Any], input_shape: list[int] | tuple[int, ...] | None = None):
    input_shape = tuple(input_shape or model_cfg.get('input_shape', [1, 4, 4]))
    shape = input_shape
    modules = []
    history = [shape]
    for raw in model_cfg.get('layers', []):
        cfg = deepcopy(raw)
        layer_type = cfg.pop('type')
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
        history.append(shape)
    model = Sequential(*modules)
    model.input_shape = input_shape
    model.inferred_shapes = history
    return model, shape
