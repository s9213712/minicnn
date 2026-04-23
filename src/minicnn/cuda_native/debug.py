"""Observability and debug helpers for cuda_native.

Provides human-readable dumps of graphs, execution plans, and runtime
execution traces. All functions are pure output helpers — they do not
modify the graph or plan.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Graph dump
# ---------------------------------------------------------------------------


def _format_tensor_shapes(specs) -> object:
    if not specs:
        return '?'
    if len(specs) == 1:
        return specs[0].shape
    return [spec.shape for spec in specs]


def dump_graph(graph, indent: int = 2) -> str:
    """Return a human-readable string representation of a NativeGraph.

    Example:
        NativeGraph  input=(8, 1, 8, 8)  output=(8, 2)  nodes=3
          [0] conv2d_0     Conv2d    (8,1,8,8) -> (8,4,6,6)  kernel=3
          [1] flatten_1    Flatten   (8,4,6,6) -> (8,144)
          [2] linear_2     Linear    (8,144)   -> (8,2)      out_features=2
    """
    pad = ' ' * indent
    buf = io.StringIO()
    in_shape = graph.input_spec.shape if graph.input_spec else '?'
    out_shape = graph.output_spec.shape if graph.output_spec else '?'
    buf.write(f'NativeGraph  input={in_shape}  output={out_shape}  nodes={len(graph.nodes)}\n')
    for i, node in enumerate(graph.nodes):
        in_s = _format_tensor_shapes(node.input_specs)
        out_s = _format_tensor_shapes(node.output_specs)
        attrs_str = '  ' + '  '.join(f'{k}={v}' for k, v in node.attrs.items()) if node.attrs else ''
        buf.write(
            f'{pad}[{i}] {node.name:<14} {node.op_type:<12} '
            f'{str(in_s):<16} -> {str(out_s):<16}{attrs_str}\n'
        )
    return buf.getvalue()


def print_graph(graph) -> None:
    """Print dump_graph() to stdout."""
    print(dump_graph(graph), end='')


# ---------------------------------------------------------------------------
# Plan dump
# ---------------------------------------------------------------------------

def dump_plan(plan, indent: int = 2) -> str:
    """Return a human-readable string of an ExecutionPlan.

    Example:
        ExecutionPlan  buffers=4  total=14.0 KB
          step  0  conv2d_0    Conv2d   [buf_0] -> [buf_1]
          step  1  flatten_1   Flatten  [buf_1] -> [buf_2]
          step  2  linear_2    Linear   [buf_2] -> [buf_3]
          Buffers:
            buf_0    2048 B  (input)
            buf_1    1152 B
    """
    pad = ' ' * indent
    bp = plan.buffer_plan
    buf = io.StringIO()
    total_kb = round(bp.total_nbytes / 1024, 2)
    peak_live_kb = round(bp.peak_live_bytes / 1024, 2)
    strategy = getattr(plan, 'strategy', 'naive')
    buf.write(
        f'ExecutionPlan  strategy={strategy}  buffers={bp.num_buffers}  total={total_kb} KB  '
        f'peak_live={peak_live_kb} KB  reuse_events={bp.reuse_events}\n'
    )
    for i, step in enumerate(plan.steps):
        ins = ', '.join(step.input_buffers)
        outs = ', '.join(step.output_buffers)
        allocated = f' alloc={step.allocated_buffers}' if step.allocated_buffers else ''
        reused = f' reuse={step.reused_buffers}' if step.reused_buffers else ''
        released = f' release={step.released_buffers}' if step.released_buffers else ''
        live = (
            f' live={round(step.live_bytes_before / 1024, 2)}->{round(step.live_bytes_after / 1024, 2)}KB'
        )
        reserved = f' reserved={round(step.reserved_bytes_after / 1024, 2)}KB'
        pressure = f' pressure={step.pressure_after:.2f}'
        slack = f' slack={step.reuse_slack_bytes}B' if step.reuse_slack_bytes else ''
        buf.write(
            f'{pad}step {i:>2}  {step.node_name:<14} {step.op_type:<12} '
            f'[{ins}] -> [{outs}]{allocated}{reused}{released}{live}{reserved}{pressure}{slack}\n'
        )
    buf.write(f'{pad}Buffers:\n')
    buf_to_tensor = {v: k for k, v in bp.tensor_to_buffer.items()}
    for buf_name, nbytes in bp.buffer_nbytes.items():
        tensor = buf_to_tensor.get(buf_name, '')
        label = f'  ({tensor})' if tensor else ''
        buf.write(f'{pad}  {buf_name:<8} {nbytes:>8} B{label}\n')
    return buf.getvalue()


def print_plan(plan) -> None:
    """Print dump_plan() to stdout."""
    print(dump_plan(plan), end='')


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

@dataclass
class NodeTrace:
    """Record of one node's execution."""
    node_name: str
    op_type: str
    category: str
    input_shapes: list[tuple]
    output_shapes: list[tuple]
    elapsed_ms: float
    attrs: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        ins = ', '.join(str(s) for s in self.input_shapes)
        outs = ', '.join(str(s) for s in self.output_shapes)
        return (
            f'{self.node_name:<16} {self.op_type:<12} '
            f'in=[{ins}]  out=[{outs}]  {self.elapsed_ms:.2f}ms'
        )


@dataclass
class ExecutionTrace:
    """Collected trace of a full forward pass."""
    node_traces: list[NodeTrace] = field(default_factory=list)

    def total_ms(self) -> float:
        return sum(t.elapsed_ms for t in self.node_traces)

    def dump(self, indent: int = 2) -> str:
        pad = ' ' * indent
        buf = io.StringIO()
        buf.write(f'ExecutionTrace  steps={len(self.node_traces)}  total={self.total_ms():.2f}ms\n')
        for t in self.node_traces:
            buf.write(f'{pad}{t}\n')
        return buf.getvalue()

    def print(self) -> None:
        print(self.dump(), end='')

    def summary(self) -> dict[str, Any]:
        return {
            'total_ms': round(self.total_ms(), 3),
            'steps': [
                {
                    'node': t.node_name,
                    'op': t.op_type,
                    'category': t.category,
                    'elapsed_ms': round(t.elapsed_ms, 3),
                    'input_shapes': t.input_shapes,
                    'output_shapes': t.output_shapes,
                }
                for t in self.node_traces
            ],
        }


class TracingForwardExecutor:
    """ForwardExecutor wrapper that records a per-node execution trace.

    Usage::

        executor = TracingForwardExecutor()
        ctx, trace = executor.run(graph, feeds, params)
        trace.print()
    """

    def run(
        self,
        graph,
        feeds: dict[str, Any],
        params: dict[str, Any] | None = None,
        mode: str = 'eval',
    ) -> tuple[dict[str, Any], ExecutionTrace]:
        from minicnn.cuda_native.kernels import make_default_registry
        if mode not in {'eval', 'train'}:
            raise ValueError(f"Unsupported cuda_native execution mode {mode!r}; expected 'eval' or 'train'")
        registry = make_default_registry()
        ctx: dict[str, Any] = dict(feeds)
        if params:
            ctx.update(params)
        ctx['__mode__'] = mode
        trace = ExecutionTrace()

        for node in graph.topological_order():
            kernel = registry.get(node.op_type)
            spec = registry.spec(node.op_type)
            in_shapes = [
                tuple(ctx[inp].shape)
                for inp in node.inputs
                if inp in ctx and hasattr(ctx[inp], 'shape')
            ]
            t0 = time.perf_counter()
            kernel(node, ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            out_shapes = [
                tuple(ctx[o].shape)
                for o in node.outputs
                if o in ctx and hasattr(ctx[o], 'shape')
            ]
            trace.node_traces.append(NodeTrace(
                node_name=node.name,
                op_type=node.op_type,
                category=spec.category,
                input_shapes=in_shapes,
                output_shapes=out_shapes,
                elapsed_ms=elapsed_ms,
                attrs=dict(node.attrs),
            ))

        return ctx, trace


# ---------------------------------------------------------------------------
# Convenience: combined graph + plan inspection
# ---------------------------------------------------------------------------

def inspect(graph, plan=None) -> str:
    """Return a combined graph and optional plan dump."""
    out = dump_graph(graph)
    if plan is not None:
        out += '\n' + dump_plan(plan)
    return out


def print_inspect(graph, plan=None) -> None:
    """Print inspect() to stdout."""
    print(inspect(graph, plan), end='')
