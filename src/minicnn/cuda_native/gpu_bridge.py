from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minicnn.cuda_native.gpu_dispatch import GpuLaunchPacket


def _scalar_arg_map(packet: GpuLaunchPacket) -> dict[str, Any]:
    return {
        str(arg['name']): arg['value']
        for arg in packet.scalar_args
    }


def _input_arg(packet: GpuLaunchPacket, index: int = 0) -> dict[str, Any]:
    inputs = [arg for arg in packet.tensor_args if arg.get('kind') == 'input']
    return dict(inputs[index])


def _output_arg(packet: GpuLaunchPacket, index: int = 0) -> dict[str, Any]:
    outputs = [arg for arg in packet.tensor_args if arg.get('kind') == 'output']
    return dict(outputs[index])


def _param_args(packet: GpuLaunchPacket) -> list[dict[str, Any]]:
    return [dict(arg) for arg in packet.tensor_args if arg.get('kind') == 'param']


def _build_bridge_payload(packet: GpuLaunchPacket) -> dict[str, Any]:
    scalar_args = _scalar_arg_map(packet)
    input_arg = _input_arg(packet)
    output_arg = _output_arg(packet)
    payload: dict[str, Any] = {
        'op_name': packet.op_name,
        'launch_family': packet.launch_family,
        'preferred_layout': packet.preferred_layout,
        'input_shape': list(input_arg.get('shape', [])),
        'output_shape': list(output_arg.get('shape', [])),
        'tensor_dtype': input_arg.get('dtype', 'float32'),
    }
    if packet.op_name == 'Linear':
        weight_arg = _param_args(packet)[0]
        payload.update({
            'matmul_m': int(input_arg['shape'][0]),
            'matmul_k': int(input_arg['shape'][-1]),
            'matmul_n': int(output_arg['shape'][-1]),
            'weight_binding': weight_arg['binding'],
            'weight_layout': weight_arg['layout'],
            'has_bias': any(arg['binding'].startswith('_b_') for arg in _param_args(packet)),
        })
    elif packet.op_name == 'Conv2d':
        weight_arg = _param_args(packet)[0]
        n, c_in, h, w = [int(v) for v in input_arg['shape']]
        _, c_out, h_out, w_out = [int(v) for v in output_arg['shape']]
        payload.update({
            'batch_size': n,
            'in_channels': c_in,
            'out_channels': c_out,
            'input_hw': [h, w],
            'output_hw': [h_out, w_out],
            'stride': scalar_args.get('stride', 1),
            'padding': scalar_args.get('padding', 0),
            'groups': int(scalar_args.get('groups', 1)),
            'weight_binding': weight_arg['binding'],
            'weight_layout': weight_arg['layout'],
            'has_bias': any(arg['binding'].startswith('_b_') for arg in _param_args(packet)),
        })
    elif packet.op_name == 'Concat':
        payload['axis'] = int(scalar_args.get('axis', 1))
    elif packet.op_name == 'LeakyReLU':
        payload['negative_slope'] = float(scalar_args.get('negative_slope', 0.01))
    elif packet.op_name == 'MaxPool2d':
        payload.update({
            'kernel_size': scalar_args.get('kernel_size', 2),
            'stride': scalar_args.get('stride', 2),
            'padding': scalar_args.get('padding', 0),
        })
    return payload


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
    bridge_payload: dict[str, Any]
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
            'bridge_payload': dict(self.bridge_payload),
        }


@dataclass(frozen=True)
class GpuFlatKernelRequest:
    request_id: str
    node_name: str
    op_name: str
    launch_family: str
    dispatch_mode: str
    preferred_layout: str
    tensor_bindings: tuple[str, ...]
    tensor_roles: tuple[str, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    tensor_dtypes: tuple[str, ...]
    tensor_layouts: tuple[str, ...]
    scalar_names: tuple[str, ...]
    scalar_values: tuple[Any, ...]
    param_bindings: tuple[str, ...]
    bridge_payload: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'node_name': self.node_name,
            'op_name': self.op_name,
            'launch_family': self.launch_family,
            'dispatch_mode': self.dispatch_mode,
            'preferred_layout': self.preferred_layout,
            'tensor_bindings': list(self.tensor_bindings),
            'tensor_roles': list(self.tensor_roles),
            'tensor_shapes': [list(shape) for shape in self.tensor_shapes],
            'tensor_dtypes': list(self.tensor_dtypes),
            'tensor_layouts': list(self.tensor_layouts),
            'scalar_names': list(self.scalar_names),
            'scalar_values': list(self.scalar_values),
            'param_bindings': list(self.param_bindings),
            'bridge_payload': dict(self.bridge_payload),
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
        bridge_payload=_build_bridge_payload(packet),
    )


def build_gpu_bridge_trace(packets: tuple[GpuLaunchPacket, ...]) -> tuple[GpuKernelBridgeRequest, ...]:
    return tuple(
        build_gpu_bridge_request(packet, index=index)
        for index, packet in enumerate(packets)
    )


def flatten_gpu_bridge_request(request: GpuKernelBridgeRequest) -> GpuFlatKernelRequest:
    tensor_bindings: list[str] = []
    tensor_roles: list[str] = []
    tensor_shapes: list[tuple[int, ...]] = []
    tensor_dtypes: list[str] = []
    tensor_layouts: list[str] = []
    param_bindings: list[str] = []
    for arg in request.tensor_args:
        role = str(arg.get('kind', 'tensor'))
        binding = str(arg.get('binding'))
        tensor_bindings.append(binding)
        tensor_roles.append(role)
        tensor_shapes.append(tuple(int(v) for v in arg.get('shape', ())))
        tensor_dtypes.append(str(arg.get('dtype', request.bridge_payload.get('tensor_dtype', 'float32'))))
        tensor_layouts.append(str(arg.get('layout', request.preferred_layout)))
        if role == 'param':
            param_bindings.append(binding)
    scalar_names = tuple(str(arg.get('name')) for arg in request.scalar_args)
    scalar_values = tuple(arg.get('value') for arg in request.scalar_args)
    return GpuFlatKernelRequest(
        request_id=request.request_id,
        node_name=request.node_name,
        op_name=request.op_name,
        launch_family=request.launch_family,
        dispatch_mode=request.dispatch_mode,
        preferred_layout=request.preferred_layout,
        tensor_bindings=tuple(tensor_bindings),
        tensor_roles=tuple(tensor_roles),
        tensor_shapes=tuple(tensor_shapes),
        tensor_dtypes=tuple(tensor_dtypes),
        tensor_layouts=tuple(tensor_layouts),
        scalar_names=scalar_names,
        scalar_values=scalar_values,
        param_bindings=tuple(param_bindings),
        bridge_payload=dict(request.bridge_payload),
    )


def build_flat_gpu_bridge_trace(
    requests: tuple[GpuKernelBridgeRequest, ...],
) -> tuple[GpuFlatKernelRequest, ...]:
    return tuple(flatten_gpu_bridge_request(request) for request in requests)
