from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    name: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Graph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        if node.name in self.nodes:
            raise ValueError(f'Duplicate graph node: {node.name}')
        self.nodes[node.name] = node

    def topological_order(self) -> list[Node]:
        order: list[Node] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited or name == 'input':
                return
            if name in visiting:
                raise ValueError(f'Cycle detected at graph node {name}')
            if name not in self.nodes:
                raise KeyError(f'Unknown graph input node {name}')
            visiting.add(name)
            for dep in self.nodes[name].inputs:
                visit(dep)
            visiting.remove(name)
            visited.add(name)
            order.append(self.nodes[name])

        for name in self.nodes:
            visit(name)
        return order
