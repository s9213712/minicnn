from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_bridge import (
    GpuFixedKernelCall,
    GpuFlatKernelRequest,
    GpuKernelBridgeRequest,
    build_fixed_kernel_trace,
    build_flat_gpu_bridge_trace,
    build_gpu_bridge_trace,
)
from minicnn.cuda_native.gpu_bridge_adapter import (
    GpuBackendStubAdapter,
    GpuFixedBridgeAdapter,
    GpuFlatBridgeAdapter,
    GpuKernelBridgeAdapter,
    GpuStubBridgeAdapter,
)
from minicnn.cuda_native.gpu_dispatch import (
    GpuDispatchPlan,
    GpuLaunchPacket,
    build_gpu_dispatch_plan,
    build_gpu_launch_packet,
)
from minicnn.cuda_native.gpu_lowering import (
    GpuLoweringContext,
    GpuLoweringRegistry,
    make_default_gpu_lowering_registry,
)
from minicnn.cuda_native.graph import NativeGraph


@dataclass(frozen=True)
class GpuStubExecutionResult:
    output_name: str
    output: np.ndarray
    dispatch_plan: GpuDispatchPlan
    launch_trace: tuple[GpuLaunchPacket, ...]
    bridge_trace: tuple[GpuKernelBridgeRequest, ...]
    flat_bridge_trace: tuple[GpuFlatKernelRequest, ...]
    fixed_bridge_trace: tuple[GpuFixedKernelCall, ...]
    bridge_results: tuple[dict[str, Any], ...]
    flat_bridge_results: tuple[dict[str, Any], ...]
    fixed_bridge_results: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        return {
            'output_name': self.output_name,
            'output_shape': list(self.output.shape),
            'dispatch_plan': self.dispatch_plan.summary(),
            'launch_trace': [packet.summary() for packet in self.launch_trace],
            'bridge_trace': [request.summary() for request in self.bridge_trace],
            'flat_bridge_trace': [request.summary() for request in self.flat_bridge_trace],
            'fixed_bridge_trace': [request.summary() for request in self.fixed_bridge_trace],
            'bridge_results': [dict(result) for result in self.bridge_results],
            'flat_bridge_results': [dict(result) for result in self.flat_bridge_results],
            'fixed_bridge_results': [dict(result) for result in self.fixed_bridge_results],
        }


class GpuStubExecutor:
    """Planned GPU execution seam backed by reference forward execution.

    This does not claim real GPU kernel execution. It exists to freeze the
    future lowering/dispatch boundary for bootstrap-subset graphs.
    """

    def __init__(
        self,
        *,
        lowering_registry: GpuLoweringRegistry | None = None,
        device_runtime: DeviceRuntime | None = None,
        bridge_adapter: GpuKernelBridgeAdapter | None = None,
        flat_bridge_adapter: GpuFlatBridgeAdapter | None = None,
        fixed_bridge_adapter: GpuFixedBridgeAdapter | None = None,
    ) -> None:
        self.lowering_registry = (
            lowering_registry if lowering_registry is not None else make_default_gpu_lowering_registry()
        )
        self.bridge_adapter = bridge_adapter if bridge_adapter is not None else GpuStubBridgeAdapter()
        self.flat_bridge_adapter = flat_bridge_adapter if flat_bridge_adapter is not None else GpuFlatBridgeAdapter()
        self.fixed_bridge_adapter = fixed_bridge_adapter if fixed_bridge_adapter is not None else GpuBackendStubAdapter()
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
        graph_nodes = {node.name: node for node in graph.topological_order()}
        remaining_uses = Counter(
            input_name
            for node in graph.topological_order()
            for input_name in node.inputs
        )
        launch_trace: list[GpuLaunchPacket] = []
        staged_input = self.device_runtime.stage_to_device(
            x,
            name=graph.input_spec.name,
            prefer_reserved=True,
        )
        tensors = {graph.input_spec.name: staged_input}
        lowering_ctx = GpuLoweringContext(
            tensors=tensors,
            params=dict(params or {}),
            runtime=self.device_runtime,
            mode=mode,
        )
        for step in dispatch_plan.steps:
            node = graph_nodes[step.node_name]
            lowering = self.lowering_registry.get(step.op_name)
            launch_packet = build_gpu_launch_packet(step)
            launch_trace.append(launch_packet)
            staged_output = lowering(node, lowering_ctx)
            output_name = staged_output.name or node.outputs[0]
            tensors[output_name] = staged_output
            self.device_runtime.record_execution(
                f'gpu_stub_kernel:{step.op_name}',
                input_name=step.input_names[0] if step.input_names else None,
                output_name=output_name,
                node_count=1,
            )
            self.device_runtime.record_execution(
                f'gpu_stub_launch:{step.launch_family}',
                input_name=step.input_names[0] if step.input_names else None,
                output_name=output_name,
                node_count=0,
            )
            self.device_runtime.record_execution(
                f'gpu_stub_packet:{step.launch_family}',
                input_name=step.input_names[0] if step.input_names else None,
                output_name=output_name,
                node_count=0,
            )
            for input_name in step.input_names:
                remaining_uses[input_name] -= 1
                if (
                    remaining_uses[input_name] == 0
                    and input_name in tensors
                    and input_name != graph.output_spec.name
                ):
                    self.device_runtime.release_buffer(tensors.pop(input_name))
        output_tensor = tensors[graph.output_spec.name]
        host_output = self.device_runtime.stage_to_host(output_tensor, copy=True)
        self.device_runtime.record_execution(
            'gpu_stub_forward',
            input_name=graph.input_spec.name,
            output_name=graph.output_spec.name,
            node_count=0,
        )
        self.device_runtime.synchronize('gpu-stub-forward')
        for tensor_name, tensor in list(tensors.items()):
            if tensor_name != graph.output_spec.name:
                self.device_runtime.release_buffer(tensor)
                tensors.pop(tensor_name)
        self.device_runtime.release_buffer(output_tensor)
        bridge_trace = build_gpu_bridge_trace(tuple(launch_trace))
        flat_bridge_trace = build_flat_gpu_bridge_trace(bridge_trace)
        fixed_bridge_trace = build_fixed_kernel_trace(flat_bridge_trace)
        bridge_results: list[dict[str, Any]] = []
        flat_bridge_results: list[dict[str, Any]] = []
        fixed_bridge_results: list[dict[str, Any]] = []
        for request in bridge_trace:
            bridge_results.append(dict(self.bridge_adapter.submit(request)))
            self.device_runtime.record_execution(
                f'gpu_stub_bridge:{request.launch_family}',
                input_name=request.tensor_args[0]['binding'] if request.tensor_args else None,
                output_name=request.node_name,
                node_count=0,
            )
        for request in flat_bridge_trace:
            flat_bridge_results.append(dict(self.flat_bridge_adapter.submit_flat(request)))
            self.device_runtime.record_execution(
                f'gpu_stub_flat_bridge:{request.launch_family}',
                input_name=request.tensor_bindings[0] if request.tensor_bindings else None,
                output_name=request.node_name,
                node_count=0,
            )
        for request in fixed_bridge_trace:
            fixed_bridge_results.append(dict(self.fixed_bridge_adapter.submit_fixed(request)))
            self.device_runtime.record_execution(
                f'gpu_stub_fixed_bridge:{request.launch_family}',
                input_name=request.input_binding,
                output_name=request.node_name,
                node_count=0,
            )
        return GpuStubExecutionResult(
            output_name=graph.output_spec.name,
            output=host_output,
            dispatch_plan=dispatch_plan,
            launch_trace=tuple(launch_trace),
            bridge_trace=bridge_trace,
            flat_bridge_trace=flat_bridge_trace,
            fixed_bridge_trace=fixed_bridge_trace,
            bridge_results=tuple(bridge_results),
            flat_bridge_results=tuple(flat_bridge_results),
            fixed_bridge_results=tuple(fixed_bridge_results),
        )
