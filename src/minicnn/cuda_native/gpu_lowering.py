from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime, DeviceTensor
from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs
from minicnn.cuda_native.kernels import _attr_pair, _conv2d_forward_array, _pool2d_windows
from minicnn.cuda_native.nodes import Node


@dataclass
class GpuLoweringContext:
    tensors: dict[str, DeviceTensor]
    params: dict[str, Any]
    runtime: DeviceRuntime
    mode: str = 'eval'


GpuLoweringFn = Callable[[Node, GpuLoweringContext], DeviceTensor]


@dataclass(frozen=True)
class GpuLoweringSpec:
    op_name: str
    lowering_kind: str
    kernel_category: str
    fn: GpuLoweringFn

    def __iter__(self) -> Iterator[Any]:
        yield self.op_name
        yield self.fn


class GpuLoweringRegistry:
    def __init__(self) -> None:
        self._dispatch: dict[str, GpuLoweringFn] = {}
        self._specs: dict[str, GpuLoweringSpec] = {}

    def register(
        self,
        op_name: str,
        *,
        lowering_kind: str,
        kernel_category: str,
        fn: GpuLoweringFn,
    ) -> 'GpuLoweringRegistry':
        spec = GpuLoweringSpec(
            op_name=op_name,
            lowering_kind=lowering_kind,
            kernel_category=kernel_category,
            fn=fn,
        )
        self._dispatch[op_name] = fn
        self._specs[op_name] = spec
        return self

    def get(self, op_name: str) -> GpuLoweringFn:
        if op_name not in self._dispatch:
            raise KeyError(f'No gpu lowering shim registered for op: {op_name}')
        return self._dispatch[op_name]

    def spec(self, op_name: str) -> GpuLoweringSpec:
        if op_name not in self._specs:
            raise KeyError(f'No gpu lowering shim registered for op: {op_name}')
        return self._specs[op_name]

    def has(self, op_name: str) -> bool:
        return op_name in self._dispatch

    def registered_ops(self) -> list[str]:
        return sorted(self._dispatch)

    def registered_specs(self) -> list[GpuLoweringSpec]:
        return [self._specs[op_name] for op_name in self.registered_ops()]


def _allocate_output(node: Node, ctx: GpuLoweringContext, output: np.ndarray) -> DeviceTensor:
    staged = ctx.runtime.allocate_staging_buffer(
        tuple(node.output_specs[0].shape),
        dtype=output.dtype,
        name=node.outputs[0],
    )
    np.copyto(staged.data, np.asarray(output, dtype=np.float32))
    ctx.runtime.sync_tensor_to_device(staged)
    return staged


def _input_tensor(node: Node, ctx: GpuLoweringContext, index: int = 0) -> DeviceTensor:
    return ctx.tensors[node.inputs[index]]


def _lower_flatten(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    x = np.asarray(_input_tensor(node, ctx).data, dtype=np.float32)
    output = x.reshape(x.shape[0], -1).astype(np.float32)
    return _allocate_output(node, ctx, output)


def _lower_linear(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    w = np.asarray(ctx.params[f'_w_{node.name}'], dtype=np.float32)
    b = ctx.params.get(f'_b_{node.name}')
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'dense_forward')
    ):
        output = ctx.runtime.allocate_staging_buffer(
            tuple(node.output_specs[0].shape),
            dtype='float32',
            name=node.outputs[0],
        )
        bias = np.asarray(
            np.zeros(w.shape[0], dtype=np.float32) if b is None else b,
            dtype=np.float32,
        )
        weight_tensor = ctx.runtime.stage_to_device(w, name=f'_w_{node.name}')
        bias_tensor = ctx.runtime.stage_to_device(bias, name=f'_b_{node.name}')
        ctx.runtime.bound_lib.dense_forward(
            input_tensor.device_ptr,
            weight_tensor.device_ptr,
            bias_tensor.device_ptr,
            output.device_ptr,
            int(x.shape[0]),
            int(w.shape[1]),
            int(w.shape[0]),
        )
        ctx.runtime.record_execution(
            'gpu_native_kernel:dense_forward',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output)
        ctx.runtime.release_buffer(weight_tensor)
        ctx.runtime.release_buffer(bias_tensor)
        return output
    output = x @ w.T
    if b is not None:
        output = output + np.asarray(b, dtype=np.float32)
    return _allocate_output(node, ctx, output.astype(np.float32))


def _lower_relu(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    x = np.asarray(_input_tensor(node, ctx).data, dtype=np.float32)
    return _allocate_output(node, ctx, np.maximum(x, 0.0).astype(np.float32))


def _lower_leaky_relu(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    x = np.asarray(_input_tensor(node, ctx).data, dtype=np.float32)
    alpha = float(node.attrs.get('negative_slope', 0.01))
    output = np.where(x >= 0.0, x, alpha * x).astype(np.float32)
    return _allocate_output(node, ctx, output)


def _lower_add(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    arrays = [np.asarray(ctx.tensors[name].data, dtype=np.float32) for name in node.inputs]
    ref_shape = arrays[0].shape
    output = arrays[0].copy()
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.shape != ref_shape:
            raise ValueError(
                f'Add node={node.name}: all inputs must share the same shape, '
                f'got input[0]={ref_shape} and input[{idx}]={arr.shape}'
            )
        output += arr
    return _allocate_output(node, ctx, output.astype(np.float32))


def _lower_concat(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    arrays = [np.asarray(ctx.tensors[name].data, dtype=np.float32) for name in node.inputs]
    axis = int(node.attrs.get('axis', 1))
    try:
        output = np.concatenate(arrays, axis=axis).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f'Concat node={node.name}: {exc}') from exc
    return _allocate_output(node, ctx, output)


def _lower_conv2d(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    x = np.asarray(_input_tensor(node, ctx).data, dtype=np.float32)
    w = np.asarray(ctx.params[f'_w_{node.name}'], dtype=np.float32)
    b = ctx.params.get(f'_b_{node.name}')
    groups = int(node.attrs.get('groups', x.shape[1] if node.op_type in {'DepthwiseConv2d', 'depthwise_conv2d'} else 1))
    stride = _attr_pair(node.attrs.get('stride', 1), label='stride', node=node)
    padding = _attr_pair(node.attrs.get('padding', 0), label='padding', node=node)
    output = _conv2d_forward_array(
        x,
        w,
        bias=None if b is None else np.asarray(b, dtype=np.float32),
        stride=stride,
        padding=padding,
        groups=groups,
        node_desc=f'Conv2d node={node.name}',
    )
    return _allocate_output(node, ctx, output.astype(np.float32))


def _lower_maxpool2d(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    x = np.asarray(_input_tensor(node, ctx).data, dtype=np.float32)
    windows, _kernel, _stride = _pool2d_windows(node, x)
    output = windows.max(axis=(-2, -1)).astype(np.float32)
    return _allocate_output(node, ctx, output)


def make_default_gpu_lowering_registry() -> GpuLoweringRegistry:
    registry = GpuLoweringRegistry()
    kernel_categories = {
        spec.op_name: spec.category
        for spec in list_gpu_kernel_specs()
    }
    registry.register(
        'Add',
        lowering_kind='merge_add_shim',
        kernel_category=kernel_categories['Add'],
        fn=_lower_add,
    )
    registry.register(
        'Concat',
        lowering_kind='merge_concat_shim',
        kernel_category=kernel_categories['Concat'],
        fn=_lower_concat,
    )
    registry.register(
        'Conv2d',
        lowering_kind='conv2d_reference_shim',
        kernel_category=kernel_categories['Conv2d'],
        fn=_lower_conv2d,
    )
    registry.register(
        'Flatten',
        lowering_kind='shape_flatten_shim',
        kernel_category=kernel_categories['Flatten'],
        fn=_lower_flatten,
    )
    registry.register(
        'LeakyReLU',
        lowering_kind='activation_leaky_relu_shim',
        kernel_category=kernel_categories['LeakyReLU'],
        fn=_lower_leaky_relu,
    )
    registry.register(
        'Linear',
        lowering_kind='linear_affine_shim',
        kernel_category=kernel_categories['Linear'],
        fn=_lower_linear,
    )
    registry.register(
        'MaxPool2d',
        lowering_kind='pool_maxpool2d_shim',
        kernel_category=kernel_categories['MaxPool2d'],
        fn=_lower_maxpool2d,
    )
    registry.register(
        'ReLU',
        lowering_kind='activation_relu_shim',
        kernel_category=kernel_categories['ReLU'],
        fn=_lower_relu,
    )
    return registry


def list_gpu_lowering_specs() -> list[GpuLoweringSpec]:
    return make_default_gpu_lowering_registry().registered_specs()
