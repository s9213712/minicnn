"""Forward-only executor for cuda_native sequential graphs."""
from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.kernels import KernelRegistry, make_default_registry


class ForwardExecutor:
    """Runs a NativeGraph forward pass using a KernelRegistry.

    The executor is stateless with respect to graph weights — callers
    pass weights via the *params* dict (see run()).
    """

    def __init__(
        self,
        registry: KernelRegistry | None = None,
        debug: bool = False,
    ) -> None:
        self.registry = registry if registry is not None else make_default_registry()
        self.debug = debug

    def run(
        self,
        graph: NativeGraph,
        feeds: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute *graph* forward pass.

        Args:
            graph:  NativeGraph produced by build_graph().
            feeds:  {'input': np.ndarray of shape matching graph.input_spec}
            params: optional weight dict.  Keys follow the convention
                    '_w_{node.name}' for weight and '_b_{node.name}' for bias.
                    If None, the executor assumes no learnable parameters (useful
                    for testing activation / flatten / reshape nodes).

        Returns:
            context dict containing all intermediate and final tensors.
        """
        ctx: dict[str, Any] = dict(feeds)
        if params:
            ctx.update(params)

        for node in graph.topological_order():
            kernel = self.registry.get(node.op_type)
            if self.debug:
                in_shapes = {k: ctx[k].shape for k in node.inputs if k in ctx}
                print(f'[cuda_native] exec {node.name} ({node.op_type}) in={in_shapes}')
            kernel(node, ctx)
            if self.debug:
                out_shapes = {k: ctx[k].shape for k in node.outputs if k in ctx}
                print(f'[cuda_native]   -> out={out_shapes}')

        return ctx

    def run_inference(
        self,
        graph: NativeGraph,
        x: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Convenience wrapper: feed *x* as 'input', return final output array."""
        if graph.input_spec is None:
            raise ValueError('Graph has no input_spec; was it built with build_graph()?')
        ctx = self.run(graph, {graph.input_spec.name: x}, params=params)
        out_name = graph.output_spec.name if graph.output_spec else graph.nodes[-1].outputs[0]
        return ctx[out_name]
