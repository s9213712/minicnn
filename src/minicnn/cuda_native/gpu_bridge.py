from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minicnn.cuda_native.gpu_dispatch import GpuLaunchPacket


GPU_OP_CODES = {
    'Flatten': 1,
    'Linear': 2,
    'ReLU': 3,
    'LeakyReLU': 4,
    'Add': 5,
    'Concat': 6,
    'Conv2d': 7,
    'MaxPool2d': 8,
    'BatchNorm2d': 9,
    'AdaptiveAvgPool2d': 10,
    'GlobalAvgPool2d': 11,
    'GELU': 12,
    'Sigmoid': 13,
    'SiLU': 14,
    'Tanh': 15,
}

GPU_LAUNCH_FAMILY_CODES = {
    'reshape_view': 1,
    'gemm_affine': 2,
    'elementwise_unary': 3,
    'elementwise_merge': 4,
    'concat_merge': 5,
    'conv2d_nchw': 6,
    'pool2d_nchw': 7,
    'batchnorm2d_nchw': 8,
    'global_avgpool2d_nchw': 9,
}

GPU_LAYOUT_CODES = {
    'unknown': 0,
    'row_major': 1,
    'NCHW': 2,
    'OI': 3,
    'O': 4,
    'OIHW': 5,
    'match_input': 6,
    'match_inputs': 7,
}

GPU_DTYPE_CODES = {
    'float32': 1,
    'float16': 2,
    'int32': 3,
    'int64': 4,
}


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


@dataclass(frozen=True)
class GpuFixedKernelCall:
    request_id: str
    node_name: str
    op_name: str
    launch_family: str
    dispatch_mode: str
    preferred_layout: str
    input_binding: str
    output_binding: str
    weight_binding: str
    bias_binding: str
    tensor_dtype: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    stride_h: int
    stride_w: int
    padding_h: int
    padding_w: int
    groups: int
    matmul_m: int
    matmul_k: int
    matmul_n: int

    def summary(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'node_name': self.node_name,
            'op_name': self.op_name,
            'launch_family': self.launch_family,
            'dispatch_mode': self.dispatch_mode,
            'preferred_layout': self.preferred_layout,
            'input_binding': self.input_binding,
            'output_binding': self.output_binding,
            'weight_binding': self.weight_binding,
            'bias_binding': self.bias_binding,
            'tensor_dtype': self.tensor_dtype,
            'input_shape': list(self.input_shape),
            'output_shape': list(self.output_shape),
            'stride_h': self.stride_h,
            'stride_w': self.stride_w,
            'padding_h': self.padding_h,
            'padding_w': self.padding_w,
            'groups': self.groups,
            'matmul_m': self.matmul_m,
            'matmul_k': self.matmul_k,
            'matmul_n': self.matmul_n,
        }


@dataclass(frozen=True)
class GpuCAbiKernelCall:
    request_id: str
    node_name: str
    op_name: str
    op_code: int
    launch_family: str
    launch_family_code: int
    dtype_code: int
    preferred_layout_code: int
    input_binding: str
    output_binding: str
    weight_binding: str
    bias_binding: str
    input_rank: int
    output_rank: int
    input_shape4: tuple[int, int, int, int]
    output_shape4: tuple[int, int, int, int]
    int_args8: tuple[int, int, int, int, int, int, int, int]
    flags: tuple[int, int, int, int]

    def summary(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'node_name': self.node_name,
            'op_name': self.op_name,
            'op_code': self.op_code,
            'launch_family': self.launch_family,
            'launch_family_code': self.launch_family_code,
            'dtype_code': self.dtype_code,
            'preferred_layout_code': self.preferred_layout_code,
            'input_binding': self.input_binding,
            'output_binding': self.output_binding,
            'weight_binding': self.weight_binding,
            'bias_binding': self.bias_binding,
            'input_rank': self.input_rank,
            'output_rank': self.output_rank,
            'input_shape4': list(self.input_shape4),
            'output_shape4': list(self.output_shape4),
            'int_args8': list(self.int_args8),
            'flags': list(self.flags),
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


def build_fixed_kernel_call(request: GpuFlatKernelRequest) -> GpuFixedKernelCall:
    tensor_shape_map = {
        binding: shape
        for binding, shape in zip(request.tensor_bindings, request.tensor_shapes)
    }
    payload = request.bridge_payload
    input_binding = next((binding for binding, role in zip(request.tensor_bindings, request.tensor_roles) if role == 'input'), '')
    output_binding = next((binding for binding, role in zip(request.tensor_bindings, request.tensor_roles) if role == 'output'), '')
    param_bindings = list(request.param_bindings)
    weight_binding = param_bindings[0] if param_bindings else ''
    bias_binding = param_bindings[1] if len(param_bindings) > 1 else ''
    stride = payload.get('stride', 0)
    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = int(stride[0]), int(stride[1])
    else:
        stride_h = stride_w = int(stride)
    padding = payload.get('padding', 0)
    if isinstance(padding, (list, tuple)):
        padding_h, padding_w = int(padding[0]), int(padding[1])
    else:
        padding_h = padding_w = int(padding)
    return GpuFixedKernelCall(
        request_id=request.request_id,
        node_name=request.node_name,
        op_name=request.op_name,
        launch_family=request.launch_family,
        dispatch_mode='gpu_fixed_bridge_stub',
        preferred_layout=request.preferred_layout,
        input_binding=input_binding,
        output_binding=output_binding,
        weight_binding=weight_binding,
        bias_binding=bias_binding,
        tensor_dtype=str(payload.get('tensor_dtype', 'float32')),
        input_shape=tuple(int(v) for v in tensor_shape_map.get(input_binding, ())),
        output_shape=tuple(int(v) for v in tensor_shape_map.get(output_binding, ())),
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        groups=int(payload.get('groups', 1)),
        matmul_m=int(payload.get('matmul_m', 0)),
        matmul_k=int(payload.get('matmul_k', 0)),
        matmul_n=int(payload.get('matmul_n', 0)),
    )


def build_fixed_kernel_trace(
    requests: tuple[GpuFlatKernelRequest, ...],
) -> tuple[GpuFixedKernelCall, ...]:
    return tuple(build_fixed_kernel_call(request) for request in requests)


def _shape4(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    padded = list(shape[:4])
    while len(padded) < 4:
        padded.append(1)
    return tuple(int(v) for v in padded[:4])


def build_c_abi_kernel_call(request: GpuFixedKernelCall) -> GpuCAbiKernelCall:
    has_weight = 1 if request.weight_binding else 0
    has_bias = 1 if request.bias_binding else 0
    return GpuCAbiKernelCall(
        request_id=request.request_id,
        node_name=request.node_name,
        op_name=request.op_name,
        op_code=int(GPU_OP_CODES.get(request.op_name, 0)),
        launch_family=request.launch_family,
        launch_family_code=int(GPU_LAUNCH_FAMILY_CODES.get(request.launch_family, 0)),
        dtype_code=int(GPU_DTYPE_CODES.get(request.tensor_dtype, 0)),
        preferred_layout_code=int(GPU_LAYOUT_CODES.get(request.preferred_layout, 0)),
        input_binding=request.input_binding,
        output_binding=request.output_binding,
        weight_binding=request.weight_binding,
        bias_binding=request.bias_binding,
        input_rank=len(request.input_shape),
        output_rank=len(request.output_shape),
        input_shape4=_shape4(request.input_shape),
        output_shape4=_shape4(request.output_shape),
        int_args8=(
            int(request.stride_h),
            int(request.stride_w),
            int(request.padding_h),
            int(request.padding_w),
            int(request.groups),
            int(request.matmul_m),
            int(request.matmul_k),
            int(request.matmul_n),
        ),
        flags=(has_weight, has_bias, 0, 0),
    )


def build_c_abi_kernel_trace(
    requests: tuple[GpuFixedKernelCall, ...],
) -> tuple[GpuCAbiKernelCall, ...]:
    return tuple(build_c_abi_kernel_call(request) for request in requests)
