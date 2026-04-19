from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IRNode:
    name: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    shape: tuple[int, ...] | None = None
    dtype: str = 'float32'
    fusible: bool = False


class IRGraph:
    def __init__(self, nodes: list[IRNode] | None = None):
        self.nodes = nodes or []

    def topological_order(self) -> list[IRNode]:
        return list(self.nodes)

    def summary(self) -> dict[str, object]:
        return {'num_nodes': len(self.nodes), 'ops': [node.op for node in self.nodes]}
