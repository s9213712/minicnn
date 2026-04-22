from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_PARAMETERIZED_TYPES = {
    'Conv2d',
    'DepthwiseConv2d',
    'BatchNorm2d',
    'LayerNorm',
    'Linear',
    'Embedding',
}

_CHILD_KEYS = ('children', 'layers', 'blocks')


@dataclass
class ModelLayerView:
    index: int
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)
    composite: bool = False
    children: list['ModelLayerView'] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelView:
    model_type: str
    input_shape: list[int] | None
    backend_intent: str | None
    layers: list[ModelLayerView]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_view_from_config(cfg: dict[str, Any]) -> ModelView:
    model_cfg = cfg.get('model', {})
    model_type = str(model_cfg.get('type', 'Sequential'))
    input_shape = _extract_input_shape(cfg)
    backend_intent = _extract_backend_intent(cfg)
    layers = _extract_model_layers(model_cfg.get('layers', []))
    summary = _build_model_summary(layers)
    return ModelView(
        model_type=model_type,
        input_shape=input_shape,
        backend_intent=backend_intent,
        layers=layers,
        summary=summary,
    )


def _extract_input_shape(cfg: dict[str, Any]) -> list[int] | None:
    dataset_cfg = cfg.get('dataset', {})
    shape = dataset_cfg.get('input_shape') or dataset_cfg.get('input_size')
    if isinstance(shape, (list, tuple)):
        return [int(x) for x in shape]
    return None


def _extract_backend_intent(cfg: dict[str, Any]) -> str | None:
    backend = cfg.get('engine', {}).get('backend')
    return str(backend) if backend is not None else None


def _extract_model_layers(layers_cfg: Any) -> list[ModelLayerView]:
    if not isinstance(layers_cfg, list):
        return []
    return [_layer_view(i, layer) for i, layer in enumerate(layers_cfg) if isinstance(layer, dict)]


def _layer_view(index: int, layer_cfg: dict[str, Any]) -> ModelLayerView:
    layer_type = str(layer_cfg.get('type', 'Unknown'))
    child_layers: list[ModelLayerView] = []
    for key in _CHILD_KEYS:
        children_cfg = layer_cfg.get(key)
        if isinstance(children_cfg, list):
            child_layers = [_layer_view(i, child) for i, child in enumerate(children_cfg) if isinstance(child, dict)]
            break
    attrs = {key: value for key, value in layer_cfg.items() if key not in {'type', *_CHILD_KEYS}}
    composite = bool(child_layers or layer_type.endswith('Block') or layer_type.endswith('Stage'))
    return ModelLayerView(
        index=index,
        type=layer_type,
        attrs=attrs,
        composite=composite,
        children=child_layers,
    )


def _build_model_summary(layers: list[ModelLayerView]) -> dict[str, Any]:
    flat = list(_walk_layers(layers))
    return {
        'layer_count': len(layers),
        'expanded_layer_count': len(flat),
        'parameterized_layers': sum(1 for layer in flat if layer.type in _PARAMETERIZED_TYPES),
        'composite_blocks': sum(1 for layer in flat if layer.composite),
    }


def _walk_layers(layers: list[ModelLayerView]):
    for layer in layers:
        yield layer
        if layer.children:
            yield from _walk_layers(layer.children)


def render_model_view_text(view: ModelView) -> str:
    lines = ['Model Summary', '=============', f'Type: {view.model_type}']
    if view.backend_intent:
        lines.append(f'Backend intent: {view.backend_intent}')
    if view.input_shape is not None:
        lines.append(f'Input shape: {view.input_shape}')
    lines.append('')
    lines.append('Layers:')
    if not view.layers:
        lines.append('(none)')
    else:
        for layer in view.layers:
            lines.extend(_render_layer_text(layer))
    lines.append('')
    lines.append('Totals:')
    for key, value in view.summary.items():
        lines.append(f'- {key}: {value}')
    return '\n'.join(lines)


def _render_layer_text(layer: ModelLayerView, indent: int = 0) -> list[str]:
    prefix = '  ' * indent
    attrs = _render_attrs(layer.attrs)
    line = f'{prefix}[{layer.index}] {layer.type}'
    if attrs:
        line += f'({attrs})'
    lines = [line]
    for child in layer.children:
        lines.extend(_render_layer_text(child, indent + 1))
    return lines


def _render_attrs(attrs: dict[str, Any]) -> str:
    if not attrs:
        return ''
    return ', '.join(f'{key}={value}' for key, value in attrs.items())
