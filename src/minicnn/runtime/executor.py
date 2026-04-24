from __future__ import annotations

import numpy as np

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
                values[node.name] = self._run_multi_input_node(node, args)
                continue
            module = modules.get(node.name)
            if module is None:
                module = get_model_component(node.op)(**node.attrs)
                modules[node.name] = module
            values[node.name] = module(args[0])
        return values

    def _run_multi_input_node(self, node, args):
        if node.op == 'Add':
            out = args[0]
            for arg in args[1:]:
                out = out + arg
            return out
        if node.op == 'Concat':
            axis = int(node.attrs.get('axis', 1))
            requires_grad = any(getattr(arg, 'requires_grad', False) for arg in args)
            data = np.concatenate([arg.data for arg in args], axis=axis)
            return Tensor(data, requires_grad=requires_grad)
        raise NotImplementedError(
            f'GraphExecutor currently supports multi-input nodes only for Add/Concat, got {node.op!r}'
        )
