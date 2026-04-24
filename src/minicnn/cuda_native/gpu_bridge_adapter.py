from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any

from minicnn.cuda_native.gpu_bridge import GpuKernelBridgeRequest


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
