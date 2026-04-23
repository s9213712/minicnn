"""Forward executor for cuda_native sequential graphs."""
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
        mode: str = 'eval',
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
        if mode not in {'eval', 'train'}:
            raise ValueError(f"Unsupported cuda_native execution mode {mode!r}; expected 'eval' or 'train'")
        ctx: dict[str, Any] = dict(feeds)
        if params:
            ctx.update(params)
        ctx['__mode__'] = mode

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
        mode: str = 'eval',
    ) -> np.ndarray:
        """Convenience wrapper: feed *x* as 'input', return final output array."""
        if graph.input_spec is None:
            raise ValueError('Graph has no input_spec; was it built with build_graph()?')
        ctx = self.run(graph, {graph.input_spec.name: x}, params=params, mode=mode)
        out_name = graph.output_spec.name if graph.output_spec else graph.nodes[-1].outputs[0]
        return ctx[out_name]

    def run_with_cache(
        self,
        graph: NativeGraph,
        feeds: dict[str, Any],
        params: dict[str, Any] | None = None,
        mode: str = 'eval',
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward pass that also builds a backward cache.

        The cache stores, for each node:
          - 'fwd_{node.name}_in':       the input tensor (needed by most bwd kernels)
          - 'fwd_{node.name}_in_shape': original shape before Flatten
          - '_w_{node.name}' / '_b_{node.name}': weight copies for grad computation

        Returns:
            (ctx, cache) where ctx is the full tensor context and cache has the
            backward-specific entries.
        """
        if mode not in {'eval', 'train'}:
            raise ValueError(f"Unsupported cuda_native execution mode {mode!r}; expected 'eval' or 'train'")
        ctx: dict[str, Any] = dict(feeds)
        if params:
            ctx.update(params)
        ctx['__mode__'] = mode

        cache: dict[str, Any] = {}

        for node in graph.topological_order():
            # Save input tensor before kernel mutates ctx
            if node.inputs:
                in_val = ctx.get(node.inputs[0])
                if in_val is not None:
                    cache[f'fwd_{node.name}_in'] = in_val
                    if node.op_type == 'Flatten':
                        cache[f'fwd_{node.name}_in_shape'] = in_val.shape
                    if node.op_type == 'BatchNorm2d':
                        cache[f'fwd_{node.name}_mode'] = mode
            # Save params needed for grad computation
            for key in (
                f'_w_{node.name}',
                f'_b_{node.name}',
                f'_running_mean_{node.name}',
                f'_running_var_{node.name}',
            ):
                if key in ctx:
                    cache[key] = ctx[key]
            node_suffix = f'_{node.name}'
            for key, value in ctx.items():
                if key.startswith('_') and key.endswith(node_suffix):
                    cache[key] = value

            kernel = self.registry.get(node.op_type)
            kernel(node, ctx)
            composite_cache_key = f'__cache_{node.name}'
            if composite_cache_key in ctx:
                cache[composite_cache_key] = ctx[composite_cache_key]

        return ctx, cache
