from __future__ import annotations

from minicnn.runtime.graph import Graph, Node


def build_graph_from_config(model_cfg: dict) -> Graph:
    graph = Graph()
    for node_cfg in model_cfg.get('graph', {}).get('nodes', []):
        graph.add_node(Node(
            name=node_cfg['name'],
            op=node_cfg['type'],
            inputs=list(node_cfg.get('inputs', [])),
            attrs={k: v for k, v in node_cfg.items() if k not in {'name', 'type', 'inputs'}},
        ))
    return graph
