from __future__ import annotations

from minicnn.models.registry import get_model_component
from minicnn.nn.tensor import Tensor


class GraphExecutor:
    def __init__(self, graph):
        self.graph = graph

    def run(self, inputs):
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)
        values = {'input': inputs}
        modules = {}
        for node in self.graph.topological_order():
            args = [values[name] for name in node.inputs]
            if len(args) != 1:
                raise NotImplementedError('GraphExecutor currently supports single-input module nodes')
            module = modules.get(node.name)
            if module is None:
                module = get_model_component(node.op)(**node.attrs)
                modules[node.name] = module
            values[node.name] = module(args[0])
        return values
