from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.executor import ForwardExecutor
from minicnn.cuda_native.gpu_dispatch import GpuDispatchPlan, build_gpu_dispatch_plan
from minicnn.cuda_native.graph import NativeGraph


@dataclass(frozen=True)
class GpuStubExecutionResult:
    output_name: str
    output: np.ndarray
    dispatch_plan: GpuDispatchPlan

    def summary(self) -> dict[str, Any]:
        return {
            'output_name': self.output_name,
            'output_shape': list(self.output.shape),
            'dispatch_plan': self.dispatch_plan.summary(),
        }


class GpuStubExecutor:
    """Planned GPU execution seam backed by reference forward execution.

    This does not claim real GPU kernel execution. It exists to freeze the
    future lowering/dispatch boundary for bootstrap-subset graphs.
    """

    def __init__(
        self,
        *,
        forward_executor: ForwardExecutor | None = None,
        device_runtime: DeviceRuntime | None = None,
    ) -> None:
        self.forward_executor = forward_executor if forward_executor is not None else ForwardExecutor()
        self.device_runtime = device_runtime if device_runtime is not None else DeviceRuntime(
            execution_mode='gpu_native',
            tensor_execution_device='gpu',
        )

    def run(
        self,
        graph: NativeGraph,
        x: np.ndarray,
        *,
        params: dict[str, Any] | None = None,
        mode: str = 'eval',
    ) -> GpuStubExecutionResult:
        dispatch_plan = build_gpu_dispatch_plan(graph)
        if not dispatch_plan.ready:
            raise ValueError(
                'gpu_stub executor only supports bootstrap-subset graphs; '
                f'unsupported_ops={list(dispatch_plan.unsupported_ops)}'
            )
        if graph.input_spec is None or graph.output_spec is None:
            raise ValueError('gpu_stub executor requires graph input_spec and output_spec')
        staged_input = self.device_runtime.stage_to_device(
            x,
            name=graph.input_spec.name,
            prefer_reserved=True,
        )
        ctx = self.forward_executor.run(
            graph,
            {graph.input_spec.name: staged_input.data},
            params=params,
            mode=mode,
        )
        logits = np.asarray(ctx[graph.output_spec.name])
        staged_output = self.device_runtime.allocate_staging_buffer(
            logits.shape,
            dtype=logits.dtype,
            name=graph.output_spec.name,
        )
        np.copyto(staged_output.data, logits)
        host_output = self.device_runtime.stage_to_host(staged_output, copy=True)
        self.device_runtime.record_execution(
            'gpu_stub_forward',
            input_name=graph.input_spec.name,
            output_name=graph.output_spec.name,
            node_count=len(dispatch_plan.steps),
        )
        self.device_runtime.synchronize('gpu-stub-forward')
        self.device_runtime.release_buffer(staged_input)
        self.device_runtime.release_buffer(staged_output)
        return GpuStubExecutionResult(
            output_name=graph.output_spec.name,
            output=host_output,
            dispatch_plan=dispatch_plan,
        )
