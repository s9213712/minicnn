from __future__ import annotations

from minicnn.compiler import optimize, trace_model_config


def build_graph_view_from_config(cfg: dict) -> dict:
    graph = optimize(trace_model_config(cfg.get('model', {})))
    nodes = []
    for index, node in enumerate(graph.topological_order()):
        nodes.append({
            'index': index,
            'id': node.name,
            'op_type': node.op,
            'inputs': list(node.inputs),
            'outputs': [node.name],
            'attrs': dict(node.attrs),
            'metadata': {
                'fusible': bool(node.fusible),
                'dtype': node.dtype,
                'shape': list(node.shape) if node.shape is not None else None,
            },
        })
    return {
        'graph': {
            'inputs': ['input'],
            'outputs': [nodes[-1]['id']] if nodes else ['input'],
            'node_count': len(nodes),
            'primitive_node_count': len(nodes),
            'nodes': nodes,
        },
    }


def render_graph_view_text(payload: dict) -> str:
    graph = payload['graph']
    lines = [
        'Canonical Graph',
        '===============',
        f"Input: {', '.join(graph['inputs'])}",
        f"Output: {', '.join(graph['outputs'])}",
        '',
        'Nodes:',
    ]
    if not graph['nodes']:
        lines.append('(none)')
    for node in graph['nodes']:
        attrs = _render_attrs(node['attrs'])
        inputs = ', '.join(node['inputs'])
        outputs = ', '.join(node['outputs'])
        line = f"[{node['index']}] {node['op_type']}(inputs=[{inputs}], outputs=[{outputs}]"
        if attrs:
            line += f', {attrs}'
        line += ')'
        lines.append(line)
    lines.extend([
        '',
        'Graph metadata:',
        f"- node_count: {graph['node_count']}",
        f"- primitive_node_count: {graph['primitive_node_count']}",
    ])
    return '\n'.join(lines)


def _render_attrs(attrs: dict) -> str:
    if not attrs:
        return ''
    return ', '.join(f'{key}={value}' for key, value in attrs.items())
