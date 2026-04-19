from __future__ import annotations

from minicnn.compiler.ir import IRGraph, IRNode


def remove_identity_nodes(graph: IRGraph) -> IRGraph:
    return IRGraph([node for node in graph.nodes if node.op != 'Identity'])


def detect_conv_bn_relu(graph: IRGraph) -> list[tuple[str, str, str]]:
    patterns = []
    nodes = graph.nodes
    for i in range(len(nodes) - 2):
        a, b, c = nodes[i:i + 3]
        if (a.op, b.op, c.op) == ('Conv2d', 'BatchNorm2d', 'ReLU') and b.inputs == [a.name] and c.inputs == [b.name]:
            patterns.append((a.name, b.name, c.name))
    return patterns


def annotate_fusion_patterns(graph: IRGraph) -> IRGraph:
    patterns = detect_conv_bn_relu(graph)
    pattern_names = {name for pattern in patterns for name in pattern}
    for node in graph.nodes:
        if node.name in pattern_names:
            node.attrs['fusion_pattern'] = 'Conv2d+BatchNorm2d+ReLU'
    return graph
