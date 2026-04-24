from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minicnn.cuda_native.gpu_dispatch import GpuLaunchPacket


@dataclass(frozen=True)
class GpuKernelBridgeRequest:
    request_id: str
    node_name: str
    op_name: str
    launch_family: str
    lowering_kind: str
    preferred_layout: str
    tensor_args: tuple[dict[str, Any], ...]
    scalar_args: tuple[dict[str, Any], ...]
    dispatch_mode: str = 'gpu_bridge_stub'

    def summary(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'node_name': self.node_name,
            'op_name': self.op_name,
            'launch_family': self.launch_family,
            'lowering_kind': self.lowering_kind,
            'preferred_layout': self.preferred_layout,
            'dispatch_mode': self.dispatch_mode,
            'tensor_args': [dict(arg) for arg in self.tensor_args],
            'scalar_args': [dict(arg) for arg in self.scalar_args],
        }


def build_gpu_bridge_request(packet: GpuLaunchPacket, *, index: int) -> GpuKernelBridgeRequest:
    return GpuKernelBridgeRequest(
        request_id=f'{packet.node_name}:{index}',
        node_name=packet.node_name,
        op_name=packet.op_name,
        launch_family=packet.launch_family,
        lowering_kind=packet.lowering_kind,
        preferred_layout=packet.preferred_layout,
        tensor_args=tuple(dict(arg) for arg in packet.tensor_args),
        scalar_args=tuple(dict(arg) for arg in packet.scalar_args),
    )


def build_gpu_bridge_trace(packets: tuple[GpuLaunchPacket, ...]) -> tuple[GpuKernelBridgeRequest, ...]:
    return tuple(
        build_gpu_bridge_request(packet, index=index)
        for index, packet in enumerate(packets)
    )
