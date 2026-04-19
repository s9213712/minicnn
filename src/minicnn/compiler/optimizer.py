from __future__ import annotations

from minicnn.compiler.ir import IRGraph
from minicnn.compiler.passes import annotate_fusion_patterns, remove_identity_nodes


def optimize(graph: IRGraph) -> IRGraph:
    return annotate_fusion_patterns(remove_identity_nodes(graph))
