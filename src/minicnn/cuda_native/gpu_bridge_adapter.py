from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any

from minicnn.cuda_native.gpu_bridge import GpuFixedKernelCall, GpuFlatKernelRequest, GpuKernelBridgeRequest


class GpuKernelBridgeAdapter(Protocol):
    def submit(self, request: GpuKernelBridgeRequest) -> dict[str, Any]:
        ...


@dataclass
class GpuStubBridgeAdapter:
    submitted_requests: list[GpuKernelBridgeRequest] = field(default_factory=list)

    def submit(self, request: GpuKernelBridgeRequest) -> dict[str, Any]:
        self.submitted_requests.append(request)
        return {
            'request_id': request.request_id,
            'dispatch_mode': request.dispatch_mode,
            'launch_family': request.launch_family,
            'accepted': True,
        }


@dataclass
class GpuFlatBridgeAdapter:
    submitted_requests: list[GpuFlatKernelRequest] = field(default_factory=list)

    def submit_flat(self, request: GpuFlatKernelRequest) -> dict[str, Any]:
        self.submitted_requests.append(request)
        return {
            'request_id': request.request_id,
            'dispatch_mode': request.dispatch_mode,
            'launch_family': request.launch_family,
            'flat_tensor_arg_count': len(request.tensor_bindings),
            'flat_scalar_arg_count': len(request.scalar_names),
            'accepted': True,
        }


@dataclass
class GpuFixedBridgeAdapter:
    submitted_requests: list[GpuFixedKernelCall] = field(default_factory=list)

    def submit_fixed(self, request: GpuFixedKernelCall) -> dict[str, Any]:
        self.submitted_requests.append(request)
        return {
            'request_id': request.request_id,
            'dispatch_mode': request.dispatch_mode,
            'launch_family': request.launch_family,
            'has_weight_binding': bool(request.weight_binding),
            'matmul_signature': [request.matmul_m, request.matmul_k, request.matmul_n],
            'accepted': True,
        }
