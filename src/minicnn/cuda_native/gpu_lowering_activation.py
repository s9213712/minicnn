from __future__ import annotations

from typing import Callable

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceTensor
from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringContext
from minicnn.cuda_native.gpu_lowering_utils import (
    allocate_output as _allocate_output,
    input_tensor as _input_tensor,
)
from minicnn.cuda_native.nodes import Node


def _lower_relu(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'apply_relu')
    ):
        output = ctx.runtime.allocate_staging_buffer(
            tuple(int(v) for v in x.shape),
            dtype='float32',
            name=node.outputs[0],
        )
        np.copyto(output.data, x)
        ctx.runtime.sync_tensor_to_device(output)
        ctx.runtime.bound_lib.apply_relu(output.device_ptr, int(output.data.size))
        ctx.runtime.record_execution(
            'gpu_native_kernel:apply_relu',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output)
        return output
    return _allocate_output(node, ctx, np.maximum(x, 0.0).astype(np.float32))


def _lower_leaky_relu(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    alpha = float(node.attrs.get('negative_slope', 0.01))
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'leaky_relu_forward')
    ):
        output_shape = tuple(int(v) for v in x.shape)
        output = ctx.runtime.allocate_staging_buffer(output_shape, dtype='float32', name=node.outputs[0])
        np.copyto(output.data, x)
        ctx.runtime.sync_tensor_to_device(output)
        ctx.runtime.bound_lib.leaky_relu_forward(output.device_ptr, float(alpha), int(output.data.size))
        ctx.runtime.record_execution(
            'gpu_native_kernel:leaky_relu_forward',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output)
        return output
    output = np.where(x >= 0.0, x, alpha * x).astype(np.float32)
    return _allocate_output(node, ctx, output)


def _lower_elementwise_unary(
    node: Node,
    ctx: GpuLoweringContext,
    *,
    native_symbol: str,
    native_kind: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, native_symbol)
    ):
        output = ctx.runtime.allocate_staging_buffer(
            tuple(int(v) for v in x.shape),
            dtype='float32',
            name=node.outputs[0],
        )
        np.copyto(output.data, x)
        ctx.runtime.sync_tensor_to_device(output)
        getattr(ctx.runtime.bound_lib, native_symbol)(output.device_ptr, int(output.data.size))
        ctx.runtime.record_execution(
            native_kind,
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output)
        return output
    return _allocate_output(node, ctx, transform(x).astype(np.float32))


def _lower_sigmoid(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    return _lower_elementwise_unary(
        node,
        ctx,
        native_symbol='sigmoid_forward',
        native_kind='gpu_native_kernel:sigmoid_forward',
        transform=lambda x: 1.0 / (1.0 + np.exp(-x)),
    )


def _lower_tanh(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    return _lower_elementwise_unary(
        node,
        ctx,
        native_symbol='tanh_forward',
        native_kind='gpu_native_kernel:tanh_forward',
        transform=np.tanh,
    )


def _lower_silu(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    return _lower_elementwise_unary(
        node,
        ctx,
        native_symbol='silu_forward',
        native_kind='gpu_native_kernel:silu_forward',
        transform=lambda x: x / (1.0 + np.exp(-x)),
    )


def _lower_gelu(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    return _lower_elementwise_unary(
        node,
        ctx,
        native_symbol='gelu_forward',
        native_kind='gpu_native_kernel:gelu_forward',
        transform=lambda x: 0.5 * x * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        ),
    )
