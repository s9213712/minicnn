from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs
from minicnn.cuda_native.graph import NativeGraph


@dataclass(frozen=True)
class GpuDispatchStep:
    node_name: str
    op_name: str
    category: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    param_keys: tuple[str, ...]
    forward_status: str
    backward_status: str
    supported: bool = True


@dataclass(frozen=True)
class GpuDispatchPlan:
    execution_mode: str
    ready: bool
    steps: tuple[GpuDispatchStep, ...]
    unsupported_ops: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            'execution_mode': self.execution_mode,
            'ready': self.ready,
            'num_steps': len(self.steps),
            'unsupported_ops': list(self.unsupported_ops),
            'steps': [
                {
                    'node_name': step.node_name,
                    'op_name': step.op_name,
                    'category': step.category,
                    'input_names': list(step.input_names),
                    'output_names': list(step.output_names),
                    'param_keys': list(step.param_keys),
                    'forward_status': step.forward_status,
                    'backward_status': step.backward_status,
                    'supported': step.supported,
                }
                for step in self.steps
            ],
        }


def _node_param_keys(node) -> tuple[str, ...]:
    keys: list[str] = []
    if node.op_type in {'Conv2d', 'Linear'}:
        keys.append(f'_w_{node.name}')
        if bool(node.attrs.get('bias', True)):
            keys.append(f'_b_{node.name}')
    return tuple(keys)


def build_gpu_dispatch_plan(graph: NativeGraph) -> GpuDispatchPlan:
    registry = {
        spec.op_name: spec
        for spec in list_gpu_kernel_specs()
    }
    steps: list[GpuDispatchStep] = []
    unsupported_ops: list[str] = []
    for node in graph.topological_order():
        spec = registry.get(node.op_type)
        if spec is None:
            unsupported_ops.append(str(node.op_type))
            steps.append(
                GpuDispatchStep(
                    node_name=str(node.name),
                    op_name=str(node.op_type),
                    category='unsupported',
                    input_names=tuple(str(name) for name in node.inputs),
                    output_names=tuple(str(name) for name in node.outputs),
                    param_keys=tuple(),
                    forward_status='unsupported',
                    backward_status='unsupported',
                    supported=False,
                )
            )
            continue
        steps.append(
            GpuDispatchStep(
                node_name=str(node.name),
                op_name=str(node.op_type),
                category=str(spec.category),
                input_names=tuple(str(name) for name in node.inputs),
                output_names=tuple(str(name) for name in node.outputs),
                param_keys=_node_param_keys(node),
                forward_status=str(spec.forward_status),
                backward_status=str(spec.backward_status),
                supported=True,
            )
        )
    return GpuDispatchPlan(
        execution_mode='gpu_native',
        ready=len(unsupported_ops) == 0,
        steps=tuple(steps),
        unsupported_ops=tuple(sorted(set(unsupported_ops))),
    )
