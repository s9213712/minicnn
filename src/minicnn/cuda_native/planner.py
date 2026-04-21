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
            'buffers': dict(self.buffer_nbytes),
        }


@dataclass
class ExecutionStep:
    node_name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    input_buffers: list[str]
    output_buffers: list[str]


@dataclass
class ExecutionPlan:
    buffer_plan: BufferPlan
    steps: list[ExecutionStep] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            'steps': [
                {'node': s.node_name, 'op': s.op_type,
                 'in': s.input_buffers, 'out': s.output_buffers}
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


def make_naive_plan(graph: NativeGraph) -> ExecutionPlan:
    """Conservative planner: one unique buffer per output tensor.

    No buffer reuse is attempted.  The API reserves space for a reuse
    strategy via the buffer_plan structure.
    """
    buf_plan = BufferPlan()
    steps: list[ExecutionStep] = []
    counter = 0

    # Allocate a buffer for the input tensor
    if graph.input_spec:
        buf_name = f'buf_{counter}'
        counter += 1
        buf_plan.tensor_to_buffer[graph.input_spec.name] = buf_name
        buf_plan.buffer_nbytes[buf_name] = _spec_nbytes(graph.input_spec)

    for node in graph.topological_order():
        in_bufs = [buf_plan.tensor_to_buffer.get(t, f'?{t}') for t in node.inputs]
        out_bufs = []
        for spec in node.output_specs:
            buf_name = f'buf_{counter}'
            counter += 1
            buf_plan.tensor_to_buffer[spec.name] = buf_name
            buf_plan.buffer_nbytes[buf_name] = _spec_nbytes(spec)
            out_bufs.append(buf_name)

        steps.append(ExecutionStep(
            node_name=node.name,
            op_type=node.op_type,
            inputs=list(node.inputs),
            outputs=list(node.outputs),
            input_buffers=in_bufs,
            output_buffers=out_bufs,
        ))

    return ExecutionPlan(buffer_plan=buf_plan, steps=steps)
