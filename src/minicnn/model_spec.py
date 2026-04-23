from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from minicnn.user_errors import format_user_error


def _convnext_tiny_spec(model_cfg: dict[str, Any]) -> dict[str, Any]:
    stem_channels = int(model_cfg.get('stem_channels', 64))
    stage2_channels = int(model_cfg.get('stage2_channels', 128))
    if stem_channels <= 0 or stage2_channels <= 0:
        raise ValueError('convnext_tiny requires positive channel sizes')
    return {
        'type': 'convnext_tiny',
        'layers': [
            {
                'type': 'Conv2d',
                'out_channels': stem_channels,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
            },
            {'type': 'ConvNeXtBlock'},
            {'type': 'ConvNeXtBlock'},
            {
                'type': 'Conv2d',
                'out_channels': stage2_channels,
                'kernel_size': 2,
                'stride': 2,
            },
            {'type': 'ConvNeXtBlock'},
            {'type': 'ConvNeXtBlock'},
            {'type': 'GlobalAvgPool2d'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ],
    }


_MODEL_SPECS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    'convnext_tiny': _convnext_tiny_spec,
}


def list_named_model_specs() -> list[str]:
    return sorted(_MODEL_SPECS)


def resolve_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model_cfg, dict):
        raise TypeError('model config must be a mapping')
    data = deepcopy(model_cfg)
    model_name = data.get('name')
    if model_name is None:
        return data

    normalized_name = str(model_name)
    factory = _MODEL_SPECS.get(normalized_name)
    if factory is None:
        raise ValueError(format_user_error(
            'Invalid model entry',
            cause=f'model.name={normalized_name!r} is not registered.',
            fix='Use one of the registered model names or switch back to model.layers[].',
            example='model:\n  name: convnext_tiny',
        ))

    resolved = factory(data)
    if not isinstance(resolved, dict):
        raise TypeError(f'model spec factory for {normalized_name!r} must return a mapping')

    merged = deepcopy(resolved)
    for key, value in data.items():
        if key == 'name':
            continue
        merged[key] = deepcopy(value)
    merged.setdefault('type', normalized_name)
    merged['name'] = normalized_name
    return merged
