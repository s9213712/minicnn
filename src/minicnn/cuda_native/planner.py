"""Execution planner for cuda_native.

The planner converts a NativeGraph into an ExecutionPlan that records:
- which physical buffer backs each logical tensor
- the byte size of every buffer
- the ordered execution steps

Phase 2 ships a conservative (one-tensor-one-buffer) planner.  The API
is designed to accept a reuse strategy in a future phase without changing
callers.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from minicnn.cuda_native.graph import NativeGraph
from minicnn.cuda_native.nodes import TensorSpec


class BufferType(enum.Enum):
    ACTIVATION = 'activation'
    PARAMETER = 'parameter'
    STATISTIC = 'statistic'
    GRADIENT = 'gradient'


@dataclass
class BufferPlan:
    """Maps logical tensor names to physical buffer IDs and sizes."""
    tensor_to_buffer: dict[str, str] = field(default_factory=dict)
    buffer_nbytes: dict[str, int] = field(default_factory=dict)
    tensor_last_use: dict[str, int] = field(default_factory=dict)
    peak_live_bytes: int = 0
    reuse_events: int = 0
    release_events: int = 0
    allocation_events: int = 0
    reuse_slack_bytes: int = 0
    oversized_reuse_avoided: int = 0

    @property
    def total_nbytes(self) -> int:
        return sum(self.buffer_nbytes.values())

    @property
    def num_buffers(self) -> int:
        return len(self.buffer_nbytes)

    def summary(self) -> dict[str, Any]:
        return {
            'num_buffers': self.num_buffers,
            'total_bytes': self.total_nbytes,
            'total_kb': round(self.total_nbytes / 1024, 2),
            'peak_live_bytes': self.peak_live_bytes,
            'peak_live_kb': round(self.peak_live_bytes / 1024, 2),
            'reuse_events': self.reuse_events,
            'release_events': self.release_events,
            'allocation_events': self.allocation_events,
            'reuse_slack_bytes': self.reuse_slack_bytes,
            'oversized_reuse_avoided': self.oversized_reuse_avoided,
            'buffers': dict(self.buffer_nbytes),
            'tensor_last_use': dict(self.tensor_last_use),
        }


@dataclass
class ExecutionStep:
    node_name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    input_buffers: list[str]
    output_buffers: list[str]
    allocated_buffers: list[str] = field(default_factory=list)
    reused_buffers: list[str] = field(default_factory=list)
    released_buffers: list[str] = field(default_factory=list)
    live_bytes_before: int = 0
    live_bytes_after: int = 0
    reserved_bytes_after: int = 0
    pressure_after: float = 0.0
    reuse_slack_bytes: int = 0


@dataclass
class ExecutionPlan:
    buffer_plan: BufferPlan
    strategy: str = 'naive'
    steps: list[ExecutionStep] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            'strategy': self.strategy,
            'steps': [
                {'node': s.node_name, 'op': s.op_type,
                 'in': s.input_buffers, 'out': s.output_buffers,
                 'allocated': s.allocated_buffers, 'reused': s.reused_buffers,
                 'released': s.released_buffers,
                 'live_bytes_before': s.live_bytes_before,
                 'live_bytes_after': s.live_bytes_after,
                 'reserved_bytes_after': s.reserved_bytes_after,
                 'pressure_after': round(s.pressure_after, 4),
                 'reuse_slack_bytes': s.reuse_slack_bytes}
                for s in self.steps
            ],
            'buffer_plan': self.buffer_plan.summary(),
        }


def _spec_nbytes(spec: TensorSpec) -> int:
    _bytes = {'float32': 4, 'float16': 2, 'int32': 4, 'int64': 8}
    numel = 1
    for d in spec.shape:
        numel *= d
    return numel * _bytes.get(spec.dtype, 4)


def analyze_live_ranges(graph: NativeGraph) -> dict[str, int]:
    """Return last-use step index for each logical tensor in *graph*.

    The returned index is the final step at which the tensor is consumed as an
    input. Outputs that are never consumed are pinned to their producer step so
    planners can still reason about them.
    """
    last_use: dict[str, int] = {}
    if graph.input_spec is not None:
        last_use[graph.input_spec.name] = -1
    for step_idx, node in enumerate(graph.topological_order()):
        for input_name in node.inputs:
            last_use[input_name] = step_idx
        for output_name in node.outputs:
            last_use.setdefault(output_name, step_idx)
    return last_use


def analyze_live_tensor_sets(graph: NativeGraph) -> list[set[str]]:
    """Return live logical tensors after each execution step."""
    live_sets: list[set[str]] = []
    last_use = analyze_live_ranges(graph)
    live_tensors: set[str] = set()
    if graph.input_spec is not None:
        live_tensors.add(graph.input_spec.name)
    for step_idx, node in enumerate(graph.topological_order()):
        for output_name in node.outputs:
            live_tensors.add(output_name)
        live_sets.append(set(live_tensors))
        to_release = [
            tensor_name
            for tensor_name in live_tensors
            if last_use.get(tensor_name, step_idx) == step_idx
        ]
        for tensor_name in to_release:
            live_tensors.discard(tensor_name)
    return live_sets


def estimate_peak_live_bytes(graph: NativeGraph) -> int:
    """Estimate logical peak live activation bytes from tensor liveness."""
    tensor_nbytes: dict[str, int] = {}
    if graph.input_spec is not None:
        tensor_nbytes[graph.input_spec.name] = _spec_nbytes(graph.input_spec)
    for node in graph.topological_order():
        for spec in node.output_specs:
            tensor_nbytes[spec.name] = _spec_nbytes(spec)
    live_sets = analyze_live_tensor_sets(graph)
    if not live_sets and graph.input_spec is not None:
        return tensor_nbytes.get(graph.input_spec.name, 0)
    return max(
        (sum(tensor_nbytes.get(name, 0) for name in live_set) for live_set in live_sets),
        default=0,
    )


def make_naive_plan(graph: NativeGraph) -> ExecutionPlan:
    """Conservative planner: one unique buffer per output tensor.

    No buffer reuse is attempted.  The API reserves space for a reuse
    strategy via the buffer_plan structure.
    """
    buf_plan = BufferPlan(
        tensor_last_use=analyze_live_ranges(graph),
        peak_live_bytes=estimate_peak_live_bytes(graph),
        reuse_events=0,
        release_events=0,
        allocation_events=0,
        reuse_slack_bytes=0,
        oversized_reuse_avoided=0,
    )
    steps: list[ExecutionStep] = []
    counter = 0

    # Allocate a buffer for the input tensor
    if graph.input_spec:
        buf_name = f'buf_{counter}'
        counter += 1
        buf_plan.tensor_to_buffer[graph.input_spec.name] = buf_name
        buf_plan.buffer_nbytes[buf_name] = _spec_nbytes(graph.input_spec)
        buf_plan.allocation_events += 1

    for node in graph.topological_order():
        in_bufs = [buf_plan.tensor_to_buffer.get(t, f'?{t}') for t in node.inputs]
        out_bufs = []
        for spec in node.output_specs:
            buf_name = f'buf_{counter}'
            counter += 1
            buf_plan.tensor_to_buffer[spec.name] = buf_name
            buf_plan.buffer_nbytes[buf_name] = _spec_nbytes(spec)
            buf_plan.allocation_events += 1
            out_bufs.append(buf_name)

        steps.append(ExecutionStep(
            node_name=node.name,
            op_type=node.op_type,
            inputs=list(node.inputs),
            outputs=list(node.outputs),
            input_buffers=in_bufs,
            output_buffers=out_bufs,
            allocated_buffers=list(out_bufs),
            live_bytes_before=0,
            live_bytes_after=0,
            reserved_bytes_after=buf_plan.total_nbytes,
            pressure_after=0.0,
            reuse_slack_bytes=0,
        ))

    return ExecutionPlan(buffer_plan=buf_plan, strategy='naive', steps=steps)


def make_reuse_plan(
    graph: NativeGraph,
    *,
    max_reuse_slack_ratio: float = 4.0,
    pressure_reuse_threshold: float = 0.85,
) -> ExecutionPlan:
    """Topology-aware planner that reuses buffers once tensors reach last use.

    Args:
        max_reuse_slack_ratio:
            If the best free buffer is much larger than the requested output
            size, allocate a fresh buffer instead of reusing it unless memory
            pressure is already high.
        pressure_reuse_threshold:
            When current logical live bytes / reserved bytes reaches this value,
            prefer reusing an oversized buffer over allocating a new one.
    """
    buf_plan = BufferPlan(
        tensor_last_use=analyze_live_ranges(graph),
        peak_live_bytes=estimate_peak_live_bytes(graph),
        reuse_events=0,
        release_events=0,
        allocation_events=0,
        reuse_slack_bytes=0,
        oversized_reuse_avoided=0,
    )
    steps: list[ExecutionStep] = []
    counter = 0
    free_buffers: list[str] = []
    active_tensors: set[str] = set()

    def _current_live_bytes() -> int:
        live_buffers = {
            buf_plan.tensor_to_buffer[tensor_name]
            for tensor_name in active_tensors
            if tensor_name in buf_plan.tensor_to_buffer
        }
        return sum(buf_plan.buffer_nbytes[buf_name] for buf_name in live_buffers)

    def _alloc_buffer(required_nbytes: int) -> tuple[str, bool, int]:
        nonlocal counter
        best_idx: int | None = None
        best_size: int | None = None
        for idx, buf_name in enumerate(free_buffers):
            size = buf_plan.buffer_nbytes[buf_name]
            if size < required_nbytes:
                continue
            if best_size is None or size < best_size:
                best_idx = idx
                best_size = size
        if best_idx is not None:
            buf_name = free_buffers.pop(best_idx)
            slack_bytes = buf_plan.buffer_nbytes[buf_name] - required_nbytes
            slack_ratio = float(slack_bytes) / float(required_nbytes) if required_nbytes > 0 else 0.0
            reserved_bytes = sum(buf_plan.buffer_nbytes.values())
            live_bytes = _current_live_bytes()
            pressure = float(live_bytes) / float(reserved_bytes) if reserved_bytes > 0 else 1.0
            if slack_ratio <= max_reuse_slack_ratio or pressure >= pressure_reuse_threshold:
                buf_plan.reuse_events += 1
                buf_plan.reuse_slack_bytes += slack_bytes
                return buf_name, True, slack_bytes
            free_buffers.append(buf_name)
            buf_plan.oversized_reuse_avoided += 1
        buf_name = f'buf_{counter}'
        counter += 1
        buf_plan.buffer_nbytes[buf_name] = required_nbytes
        buf_plan.allocation_events += 1
        return buf_name, False, 0

    if graph.input_spec:
        buf_name = f'buf_{counter}'
        counter += 1
        buf_plan.tensor_to_buffer[graph.input_spec.name] = buf_name
        buf_plan.buffer_nbytes[buf_name] = _spec_nbytes(graph.input_spec)
        buf_plan.allocation_events += 1
        active_tensors.add(graph.input_spec.name)

    for step_idx, node in enumerate(graph.topological_order()):
        live_bytes_before = _current_live_bytes()
        in_bufs = [buf_plan.tensor_to_buffer.get(t, f'?{t}') for t in node.inputs]
        out_bufs: list[str] = []
        allocated_bufs: list[str] = []
        reused_bufs: list[str] = []
        step_reuse_slack_bytes = 0
        for spec in node.output_specs:
            required_nbytes = _spec_nbytes(spec)
            buf_name, reused, slack_bytes = _alloc_buffer(required_nbytes)
            if buf_plan.buffer_nbytes.get(buf_name, 0) < required_nbytes:
                buf_plan.buffer_nbytes[buf_name] = required_nbytes
            buf_plan.tensor_to_buffer[spec.name] = buf_name
            out_bufs.append(buf_name)
            if reused:
                reused_bufs.append(buf_name)
                step_reuse_slack_bytes += slack_bytes
            else:
                allocated_bufs.append(buf_name)
            active_tensors.add(spec.name)

        released_bufs: list[str] = []
        for input_name in node.inputs:
            if buf_plan.tensor_last_use.get(input_name) == step_idx:
                input_buf = buf_plan.tensor_to_buffer.get(input_name)
                if input_buf is not None and input_buf not in free_buffers:
                    free_buffers.append(input_buf)
                    released_bufs.append(input_buf)
                    buf_plan.release_events += 1
                active_tensors.discard(input_name)

        live_bytes_after = _current_live_bytes()
        reserved_bytes_after = buf_plan.total_nbytes
        pressure_after = (
            float(live_bytes_after) / float(reserved_bytes_after)
            if reserved_bytes_after > 0 else 0.0
        )

        steps.append(ExecutionStep(
            node_name=node.name,
            op_type=node.op_type,
            inputs=list(node.inputs),
            outputs=list(node.outputs),
            input_buffers=in_bufs,
            output_buffers=out_bufs,
            allocated_buffers=allocated_bufs,
            reused_buffers=reused_bufs,
            released_buffers=released_bufs,
            live_bytes_before=live_bytes_before,
            live_bytes_after=live_bytes_after,
            reserved_bytes_after=reserved_bytes_after,
            pressure_after=pressure_after,
            reuse_slack_bytes=step_reuse_slack_bytes,
        ))

    return ExecutionPlan(buffer_plan=buf_plan, strategy='reuse', steps=steps)


def make_plan(graph: NativeGraph, strategy: str = 'naive', **kwargs: Any) -> ExecutionPlan:
    """Build an ExecutionPlan using the requested strategy."""
    normalized = str(strategy or 'naive').lower()
    if normalized == 'naive':
        return make_naive_plan(graph)
    if normalized == 'reuse':
        return make_reuse_plan(graph, **kwargs)
    raise ValueError(
        f'Unsupported cuda_native planning strategy {strategy!r}; expected "naive" or "reuse"'
    )
