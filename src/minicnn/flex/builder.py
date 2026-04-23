from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .importing import import_from_string
from .registry import REGISTRY
from . import components  # noqa: F401

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover
    nn = None


CUDA_LEGACY_OPTIMIZER_KEYS = {'lr_conv1', 'lr_conv', 'lr_fc', 'grad_clip_global'}

# Block presets expand a single layer config into multiple layers.
_BLOCK_PRESETS: dict[str, list[dict[str, Any]]] = {
    'conv_relu': [
        {'type': 'Conv2d'},
        {'type': 'ReLU'},
    ],
    'conv_bn_relu': [
        {'type': 'Conv2d'},
        {'type': 'BatchNorm2d'},
        {'type': 'ReLU'},
    ],
    'conv_bn_silu': [
        {'type': 'Conv2d'},
        {'type': 'BatchNorm2d'},
        {'type': 'SiLU'},
    ],
}


def _expand_presets(layers_cfg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace any preset layer entry with its constituent layer configs."""
    expanded: list[dict[str, Any]] = []
    for raw in layers_cfg:
        layer_type = raw.get('type', '')
        if layer_type in _BLOCK_PRESETS:
            conv_keys = {'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'bias', 'groups'}
            conv_cfg = {k: v for k, v in raw.items() if k in conv_keys or k == 'type'}
            for template in _BLOCK_PRESETS[layer_type]:
                entry = deepcopy(template)
                if entry['type'] == 'Conv2d':
                    entry.update({k: v for k, v in raw.items() if k in conv_keys})
                expanded.append(entry)
        else:
            expanded.append(raw)
    return expanded
_PASSTHROUGH_LAYERS = {
    'BatchNorm2d',
    'ConvNeXtBlock',
    'Dropout',
    'Dropout2d',
    'ELU',
    'GELU',
    'Hardtanh',
    'Hardswish',
    'Identity',
    'LeakyReLU',
    'Mish',
    'ReLU',
    'ReLU6',
    'Sigmoid',
    'SiLU',
    'Softplus',
    'Softsign',
    'Tanh',
}


_SEQUENTIAL_BASE = nn.Sequential if nn is not None else object


class ConfigurableSequential(_SEQUENTIAL_BASE):
    def __init__(self, *modules: nn.Module, input_shape: tuple[int, ...] | None = None, inferred_shapes: list[tuple[int, ...]] | None = None):
        if nn is None:  # pragma: no cover - guarded by build_model
            raise RuntimeError('PyTorch is required for configurable model building')
        super().__init__(*modules)
        self.input_shape = input_shape
        self.inferred_shapes = inferred_shapes or []


class ShapeTracer:
    def __init__(self, input_shape: tuple[int, ...]):
        self.shape = tuple(int(x) for x in input_shape)
        self.history: list[tuple[int, ...]] = [self.shape]

    def update(self, new_shape: tuple[int, ...]):
        self.shape = new_shape
        self.history.append(new_shape)

    @property
    def channels(self) -> int:
        return int(self.shape[0]) if len(self.shape) >= 1 else 0

    @property
    def flattened(self) -> int:
        return math.prod(self.shape)


def _ensure_tuple2(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


def _infer_output_shape(layer_type: str, kwargs: dict[str, Any], tracer: ShapeTracer) -> tuple[int, ...]:
    shape = tracer.shape
    if layer_type == 'Conv2d':
        c, h, w = shape
        ks = _ensure_tuple2(kwargs.get('kernel_size', 1))
        st = _ensure_tuple2(kwargs.get('stride', 1))
        pd = _ensure_tuple2(kwargs.get('padding', 0))
        dl = _ensure_tuple2(kwargs.get('dilation', 1))
        out_c = int(kwargs['out_channels'])
        out_h = math.floor((h + 2*pd[0] - dl[0]*(ks[0]-1) - 1) / st[0] + 1)
        out_w = math.floor((w + 2*pd[1] - dl[1]*(ks[1]-1) - 1) / st[1] + 1)
        return (out_c, out_h, out_w)
    if layer_type in {'MaxPool2d', 'AvgPool2d'}:
        c, h, w = shape
        ks = _ensure_tuple2(kwargs.get('kernel_size', 1))
        st = _ensure_tuple2(kwargs.get('stride', kwargs.get('kernel_size', 1)))
        pd = _ensure_tuple2(kwargs.get('padding', 0))
        out_h = math.floor((h + 2*pd[0] - ks[0]) / st[0] + 1)
        out_w = math.floor((w + 2*pd[1] - ks[1]) / st[1] + 1)
        return (c, out_h, out_w)
    if layer_type == 'AdaptiveAvgPool2d':
        c, _, _ = shape
        out = kwargs.get('output_size', 1)
        if isinstance(out, int):
            out = (out, out)
        return (c, int(out[0]), int(out[1]))
    if layer_type == 'GlobalAvgPool2d':
        return (shape[0], 1, 1)
    if layer_type == 'ResidualBlock':
        c, h, w = shape
        out_c = int(kwargs.get('out_channels', kwargs.get('channels', c)))
        st = int(kwargs.get('stride', 1))
        return (out_c, math.floor((h - 1) / st + 1), math.floor((w - 1) / st + 1))
    if layer_type in _PASSTHROUGH_LAYERS:
        return shape
    if layer_type == 'Flatten':
        return (tracer.flattened,)
    if layer_type == 'Linear':
        return (int(kwargs['out_features']),)
    return shape


def _resolve_factory(category: str, type_name: str):
    if REGISTRY.has(category, type_name):
        return REGISTRY.get(category, type_name)
    if category == 'layers' and REGISTRY.has('activations', type_name):
        return REGISTRY.get('activations', type_name)
    if '.' in type_name:
        return import_from_string(type_name)
    if category == 'layers' and nn is not None and hasattr(nn, type_name):
        return getattr(nn, type_name)
    raise KeyError(f'Unknown {category} component: {type_name}')


def _materialize_layer(layer_cfg: dict[str, Any], tracer: ShapeTracer):
    cfg = deepcopy(layer_cfg)
    layer_type = cfg.pop('type')
    if layer_type == 'Conv2d' and 'in_channels' not in cfg:
        cfg['in_channels'] = tracer.channels
    if layer_type == 'BatchNorm2d' and 'num_features' not in cfg:
        cfg['num_features'] = tracer.channels
    if layer_type == 'ResidualBlock' and 'in_channels' not in cfg:
        cfg['in_channels'] = tracer.channels
    if layer_type == 'ConvNeXtBlock' and 'channels' not in cfg and 'in_channels' not in cfg:
        cfg['channels'] = tracer.channels
    if layer_type == 'Linear' and 'in_features' not in cfg:
        cfg['in_features'] = tracer.flattened
    factory = _resolve_factory('layers', layer_type)
    module = factory(**cfg)
    new_shape = _infer_output_shape(layer_type, cfg, tracer)
    if any(d <= 0 for d in new_shape):
        raise ValueError(
            f"Layer '{layer_type}' produces invalid output shape {new_shape}: all dimensions must be > 0. "
            f"Check kernel_size, stride, and padding against input shape {tracer.shape}."
        )
    tracer.update(new_shape)
    return module


def build_model(model_cfg: dict[str, Any], input_shape: list[int] | tuple[int, ...]):
    if nn is None:
        raise RuntimeError('PyTorch is required for configurable model building')
    tracer = ShapeTracer(tuple(int(x) for x in input_shape))
    modules: list[nn.Module] = []
    if 'factory' in model_cfg:
        factory = import_from_string(model_cfg['factory'])
        return factory(model_cfg)
    for layer_cfg in _expand_presets(model_cfg.get('layers', [])):
        if not isinstance(layer_cfg, dict) or 'type' not in layer_cfg:
            raise TypeError(f'Each model layer entry must be a mapping with a type, got: {layer_cfg!r}')
        modules.append(_materialize_layer(layer_cfg, tracer))
    return ConfigurableSequential(*modules, input_shape=tuple(input_shape), inferred_shapes=tracer.history)


def build_loss(loss_cfg: dict[str, Any]):
    cfg = deepcopy(loss_cfg)
    type_name = cfg.pop('type')
    factory = _resolve_factory('losses', type_name)
    return factory(**cfg)


def build_optimizer(params, optim_cfg: dict[str, Any]):
    cfg = deepcopy(optim_cfg)
    type_name = cfg.pop('type')
    for key in CUDA_LEGACY_OPTIMIZER_KEYS:
        cfg.pop(key, None)
    factory = _resolve_factory('optimizers', type_name)
    return factory(params, **cfg)


def build_scheduler(optimizer, sched_cfg: dict[str, Any] | None):
    if not sched_cfg or not sched_cfg.get('enabled', False):
        return None
    cfg = deepcopy(sched_cfg)
    cfg.pop('enabled', None)
    type_name = cfg.pop('type')
    factory = _resolve_factory('schedulers', type_name)
    return factory(optimizer, **cfg)
