from __future__ import annotations

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceTensor
from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringContext
from minicnn.cuda_native.gpu_lowering_utils import (
    allocate_output as _allocate_output,
    input_tensor as _input_tensor,
)
from minicnn.cuda_native.nodes import Node


def _lower_flatten(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    output = x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and x.ctypes.data == input_tensor.data.ctypes.data
    ):
        owns_device_ptr = input_tensor.owns_device_ptr
        input_tensor.owns_device_ptr = False
        ctx.runtime.record_execution(
            'gpu_native_alias:flatten',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        return DeviceTensor(
            data=output,
            device=input_tensor.device,
            execution_mode=input_tensor.execution_mode,
            name=node.outputs[0],
            reservation_id=input_tensor.reservation_id,
            device_ptr=input_tensor.device_ptr,
            owns_device_ptr=owns_device_ptr,
        )
    return _allocate_output(node, ctx, output)


def _lower_identity_alias(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    if node.op_type == 'Dropout':
        p = float(node.attrs.get('p', 0.5))
        if p != 0.0:
            raise ValueError(f'Dropout node={node.name}: gpu_native alias supports only p=0, got {p}')
    elif node.op_type == 'DropPath':
        p = float(node.attrs.get('p', 0.0))
        if p != 0.0:
            raise ValueError(f'DropPath node={node.name}: gpu_native alias supports only p=0, got {p}')
    input_tensor = _input_tensor(node, ctx)
    output = np.asarray(input_tensor.data, dtype=np.float32)
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and output.ctypes.data == input_tensor.data.ctypes.data
    ):
        owns_device_ptr = input_tensor.owns_device_ptr
        input_tensor.owns_device_ptr = False
        ctx.runtime.record_execution(
            f'gpu_native_alias:{node.op_type}',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        return DeviceTensor(
            data=output,
            device=input_tensor.device,
            execution_mode=input_tensor.execution_mode,
            name=node.outputs[0],
            reservation_id=input_tensor.reservation_id,
            device_ptr=input_tensor.device_ptr,
            owns_device_ptr=owns_device_ptr,
        )
    return _allocate_output(node, ctx, output)
