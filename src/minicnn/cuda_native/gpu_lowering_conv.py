from __future__ import annotations

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceTensor
from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringContext
from minicnn.cuda_native.gpu_lowering_utils import (
    allocate_output as _allocate_output,
    input_tensor as _input_tensor,
)
from minicnn.cuda_native.kernels import (
    _attr_pair,
    _conv2d_forward_array,
    _pool2d_windows,
)
from minicnn.cuda_native.nodes import Node


def _lower_conv2d(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    w = np.asarray(ctx.params[f'_w_{node.name}'], dtype=np.float32)
    b = ctx.params.get(f'_b_{node.name}')
    groups = int(node.attrs.get('groups', x.shape[1] if node.op_type in {'DepthwiseConv2d', 'depthwise_conv2d'} else 1))
    stride = _attr_pair(node.attrs.get('stride', 1), label='stride', node=node)
    padding = _attr_pair(node.attrs.get('padding', 0), label='padding', node=node)
    if (
        node.op_type == 'DepthwiseConv2d'
        and ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and groups == x.shape[1]
        and hasattr(ctx.runtime.bound_lib, 'depthwise_conv2d_forward')
    ):
        n, c_in, h_in, w_in = [int(v) for v in x.shape]
        c_out, _w_c, kh, kw = [int(v) for v in w.shape]
        out_h = (h_in + 2 * int(padding[0]) - kh) // int(stride[0]) + 1
        out_w = (w_in + 2 * int(padding[1]) - kw) // int(stride[1]) + 1
        output = ctx.runtime.allocate_staging_buffer(
            (n, c_out, out_h, out_w),
            dtype='float32',
            name=node.outputs[0],
        )
        weight_tensor = ctx.runtime.stage_to_device(w, name=f'_w_{node.name}')
        bias = np.asarray(
            np.zeros(c_out, dtype=np.float32) if b is None else b,
            dtype=np.float32,
        )
        bias_tensor = ctx.runtime.stage_to_device(bias, name=f'_b_{node.name}')
        try:
            ctx.runtime.bound_lib.depthwise_conv2d_forward(
                input_tensor.device_ptr,
                weight_tensor.device_ptr,
                bias_tensor.device_ptr,
                output.device_ptr,
                n,
                c_in,
                h_in,
                w_in,
                c_out,
                kh,
                kw,
                out_h,
                out_w,
                int(stride[0]),
                int(stride[1]),
                int(padding[0]),
                int(padding[1]),
                1 if b is not None else 0,
            )
            ctx.runtime.record_execution(
                'gpu_native_kernel:depthwise_conv2d_forward',
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                node_count=1,
            )
            ctx.runtime.sync_tensor_to_host(output)
        finally:
            ctx.runtime.release_buffer(weight_tensor)
            ctx.runtime.release_buffer(bias_tensor)
        return output
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and b is None
        and groups == 1
        and stride == (1, 1)
        and padding == (0, 0)
        and hasattr(ctx.runtime.bound_lib, 'im2col_forward')
        and hasattr(ctx.runtime.bound_lib, 'gemm_forward')
        and hasattr(ctx.runtime.bound_lib, 'cnhw_to_nchw')
    ):
        n, c_in, h_in, w_in = [int(v) for v in x.shape]
        c_out, _w_c, kh, kw = [int(v) for v in w.shape]
        out_h = h_in - kh + 1
        out_w = w_in - kw + 1
        output_shape = (n, c_out, out_h, out_w)
        output = ctx.runtime.allocate_staging_buffer(output_shape, dtype='float32', name=node.outputs[0])
        weight_tensor = ctx.runtime.stage_to_device(w.reshape(c_out, -1), name=f'_w_{node.name}')
        col_elems = c_in * kh * kw * n * out_h * out_w
        raw_elems = c_out * n * out_h * out_w
        col_ptr = ctx.runtime.bound_lib.gpu_malloc(int(col_elems * 4))
        raw_ptr = ctx.runtime.bound_lib.gpu_malloc(int(raw_elems * 4))
        try:
            ctx.runtime.bound_lib.im2col_forward(
                input_tensor.device_ptr,
                col_ptr,
                n,
                c_in,
                h_in,
                w_in,
                kh,
                kw,
                out_h,
                out_w,
            )
            ctx.runtime.bound_lib.gemm_forward(
                weight_tensor.device_ptr,
                col_ptr,
                raw_ptr,
                c_out,
                n * out_h * out_w,
                c_in * kh * kw,
            )
            ctx.runtime.bound_lib.cnhw_to_nchw(raw_ptr, output.device_ptr, n, c_out, out_h, out_w)
            ctx.runtime.record_execution(
                'gpu_native_kernel:conv2d_im2col_gemm',
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                node_count=1,
            )
            ctx.runtime.sync_tensor_to_host(output)
        finally:
            ctx.runtime.bound_lib.gpu_free(col_ptr)
            ctx.runtime.bound_lib.gpu_free(raw_ptr)
            ctx.runtime.release_buffer(weight_tensor)
        return output
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
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    kh, kw = _attr_pair(node.attrs.get('kernel_size', 2), label='kernel_size', node=node)
    sh, sw = _attr_pair(node.attrs.get('stride', 2), label='stride', node=node)
    ph, pw = _attr_pair(node.attrs.get('padding', 0), label='padding', node=node)
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'apply_maxpool')
        and x.ndim == 4
        and (kh, kw) == (2, 2)
        and (sh, sw) == (2, 2)
        and (ph, pw) == (0, 0)
    ):
        out_h = (int(x.shape[2]) + 2 * int(ph) - int(kh)) // int(sh) + 1
        out_w = (int(x.shape[3]) + 2 * int(pw) - int(kw)) // int(sw) + 1
        output = ctx.runtime.allocate_staging_buffer(
            (int(x.shape[0]), int(x.shape[1]), out_h, out_w),
            dtype='float32',
            name=node.outputs[0],
        )
        ctx.runtime.bound_lib.apply_maxpool(
            input_tensor.device_ptr,
            output.device_ptr,
            int(x.shape[0]),
            int(x.shape[1]),
            int(x.shape[2]),
            int(x.shape[3]),
        )
        ctx.runtime.record_execution(
            'gpu_native_kernel:apply_maxpool',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output)
        return output
    windows, _kernel, _stride = _pool2d_windows(node, x)
    output = windows.max(axis=(-2, -1)).astype(np.float32)
    return _allocate_output(node, ctx, output)


def _lower_avgpool2d(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    kh, kw = _attr_pair(node.attrs.get('kernel_size', 2), label='kernel_size', node=node)
    sh, sw = _attr_pair(node.attrs.get('stride', (kh, kw)), label='stride', node=node)
    ph, pw = _attr_pair(node.attrs.get('padding', 0), label='padding', node=node)
    if kh <= 0 or kw <= 0:
        raise ValueError(f'AvgPool2d node={node.name}: kernel_size must be positive, got {(kh, kw)}')
    if sh <= 0 or sw <= 0:
        raise ValueError(f'AvgPool2d node={node.name}: stride must be positive, got {(sh, sw)}')
    if ph < 0 or pw < 0:
        raise ValueError(f'AvgPool2d node={node.name}: padding must be non-negative, got {(ph, pw)}')
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'avgpool2d_forward')
        and x.ndim == 4
    ):
        out_h = (int(x.shape[2]) + 2 * int(ph) - int(kh)) // int(sh) + 1
        out_w = (int(x.shape[3]) + 2 * int(pw) - int(kw)) // int(sw) + 1
        output_shape = (int(x.shape[0]), int(x.shape[1]), out_h, out_w)
        output = ctx.runtime.allocate_staging_buffer(
            output_shape,
            dtype='float32',
            name=node.outputs[0],
        )
        ctx.runtime.bound_lib.avgpool2d_forward(
            input_tensor.device_ptr,
            output.device_ptr,
            int(x.shape[0]),
            int(x.shape[1]),
            int(x.shape[2]),
            int(x.shape[3]),
            int(output_shape[2]),
            int(output_shape[3]),
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
        )
        ctx.runtime.record_execution(
            'gpu_native_kernel:avgpool2d_forward',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output)
        return output
    windows, _kernel, _stride = _pool2d_windows(node, x)
    output = windows.mean(axis=(-2, -1)).astype(np.float32)
    return _allocate_output(node, ctx, output)


def _lower_global_avgpool2d(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f'{node.op_type} node={node.name}: expected 4-D NCHW input, got {x.shape}')
    if node.op_type == 'AdaptiveAvgPool2d':
        output_size = node.attrs.get('output_size', 1)
        if output_size not in (1, (1, 1), [1, 1]):
            raise ValueError(
                f'AdaptiveAvgPool2d node={node.name}: gpu lowering supports only output_size=1 or (1, 1), got {output_size!r}'
            )
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'global_avgpool2d_forward')
    ):
        output = ctx.runtime.allocate_staging_buffer(
            (int(x.shape[0]), int(x.shape[1]), 1, 1),
            dtype='float32',
            name=node.outputs[0],
        )
        ctx.runtime.bound_lib.global_avgpool2d_forward(
            input_tensor.device_ptr,
            output.device_ptr,
            int(x.shape[0]),
            int(x.shape[1]),
            int(x.shape[2]),
            int(x.shape[3]),
        )
        ctx.runtime.record_execution(
            'gpu_native_kernel:global_avgpool2d_forward',
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            node_count=1,
        )
        ctx.runtime.sync_tensor_to_host(output)
        return output
    output = x.mean(axis=(2, 3), keepdims=True).astype(np.float32)
    return _allocate_output(node, ctx, output)
