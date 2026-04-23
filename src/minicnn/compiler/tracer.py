from __future__ import annotations

from minicnn.compiler.ir import IRGraph, IRNode
from minicnn.model_spec import resolve_model_config


def trace_model_config(model_cfg: dict, input_name: str = 'input') -> IRGraph:
    model_cfg = resolve_model_config(model_cfg)
    nodes = []
    prev = input_name
    for idx, layer in enumerate(model_cfg.get('layers', [])):
        name = layer.get('name', f'{layer["type"].lower()}_{idx}')
        nodes.append(IRNode(
            name=name,
            op=layer['type'],
            inputs=[prev],
            attrs={k: v for k, v in layer.items() if k not in {'name', 'type'}},
            fusible=layer['type'] in {'Conv2d', 'BatchNorm2d', 'ReLU'},
        ))
        prev = name
    return IRGraph(nodes)
