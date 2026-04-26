from __future__ import annotations

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceTensor
from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringContext
from minicnn.cuda_native.gpu_lowering_utils import (
    allocate_output as _allocate_output,
)
from minicnn.cuda_native.nodes import Node


def _lower_add(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensors = [ctx.tensors[name] for name in node.inputs]
    arrays = [np.asarray(tensor.data, dtype=np.float32) for tensor in input_tensors]
    ref_shape = arrays[0].shape
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.shape != ref_shape:
            raise ValueError(
                f'Add node={node.name}: all inputs must share the same shape, '
                f'got input[0]={ref_shape} and input[{idx}]={arr.shape}'
            )
    if (
        len(input_tensors) == 2
        and ctx.runtime.native_device_pointers_enabled
        and all(tensor.device_ptr is not None for tensor in input_tensors)
        and hasattr(ctx.runtime.bound_lib, 'add_forward')
    ):
        output_tensor = ctx.runtime.allocate_staging_buffer(
            tuple(int(v) for v in ref_shape),
            dtype='float32',
            name=node.outputs[0],
        )
        ctx.runtime.bound_lib.add_forward(
            input_tensors[0].device_ptr,
            input_tensors[1].device_ptr,
            output_tensor.device_ptr,
            int(output_tensor.data.size),
        )
        ctx.runtime.record_execution(
            'gpu_native_kernel:add_forward',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output_tensor)
        return output_tensor
    output = arrays[0].copy()
    for arr in arrays[1:]:
        output += arr
    return _allocate_output(node, ctx, output.astype(np.float32))


def _lower_concat(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensors = [ctx.tensors[name] for name in node.inputs]
    arrays = [np.asarray(tensor.data, dtype=np.float32) for tensor in input_tensors]
    axis = int(node.attrs.get('axis', 1))
    rank = arrays[0].ndim
    normalized_axis = axis + rank if axis < 0 else axis
    if normalized_axis < 0 or normalized_axis >= rank:
        raise ValueError(f'Concat node={node.name}: axis {axis} is out of bounds for rank {rank}')
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.ndim != rank:
            raise ValueError(
                f'Concat node={node.name}: all inputs must have rank {rank}, got input[{idx}] rank {arr.ndim}'
            )
        for dim_idx, (left, right) in enumerate(zip(arrays[0].shape, arr.shape)):
            if dim_idx != normalized_axis and left != right:
                raise ValueError(
                    f'Concat node={node.name}: input[{idx}] shape {arr.shape} mismatches '
                    f'input[0] shape {arrays[0].shape} outside axis {axis}'
                )
    if (
        len(input_tensors) == 2
        and ctx.runtime.native_device_pointers_enabled
        and all(tensor.device_ptr is not None for tensor in input_tensors)
        and hasattr(ctx.runtime.bound_lib, 'concat_forward')
    ):
        output_shape = list(arrays[0].shape)
        output_shape[normalized_axis] = sum(int(arr.shape[normalized_axis]) for arr in arrays)
        output_tensor = ctx.runtime.allocate_staging_buffer(
            tuple(output_shape),
            dtype='float32',
            name=node.outputs[0],
        )
        outer = int(np.prod(arrays[0].shape[:normalized_axis], dtype=np.int64))
        inner = int(np.prod(arrays[0].shape[normalized_axis + 1:], dtype=np.int64))
        ctx.runtime.bound_lib.concat_forward(
            input_tensors[0].device_ptr,
            input_tensors[1].device_ptr,
            output_tensor.device_ptr,
            outer,
            int(arrays[0].shape[normalized_axis]),
            int(arrays[1].shape[normalized_axis]),
            inner,
        )
        ctx.runtime.record_execution(
            'gpu_native_kernel:concat_forward',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output_tensor)
        return output_tensor
    try:
        output = np.concatenate(arrays, axis=normalized_axis).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f'Concat node={node.name}: {exc}') from exc
    return _allocate_output(node, ctx, output)
