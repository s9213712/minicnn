from __future__ import annotations

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceTensor
from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringContext
from minicnn.cuda_native.gpu_lowering_utils import (
    allocate_output as _allocate_output,
    input_tensor as _input_tensor,
)
from minicnn.cuda_native.kernels import (
    _batchnorm2d_forward_array,
    _groupnorm_forward_array,
    _layernorm2d_forward_array,
    _layernorm_forward_array,
)
from minicnn.cuda_native.nodes import Node


def lower_batchnorm2d(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    channels = int(x.shape[1])
    gamma = np.asarray(ctx.params.get(f'_w_{node.name}', np.ones(channels, dtype=np.float32)), dtype=np.float32)
    beta = np.asarray(ctx.params.get(f'_b_{node.name}', np.zeros(channels, dtype=np.float32)), dtype=np.float32)
    running_mean_key = f'_running_mean_{node.name}'
    running_var_key = f'_running_var_{node.name}'
    running_mean = np.asarray(ctx.params.get(running_mean_key, np.zeros(channels, dtype=np.float32)), dtype=np.float32)
    running_var = np.asarray(ctx.params.get(running_var_key, np.ones(channels, dtype=np.float32)), dtype=np.float32)
    for label, value in (
        ('weight', gamma),
        ('bias', beta),
        ('running_mean', running_mean),
        ('running_var', running_var),
    ):
        if value.shape != (channels,):
            raise ValueError(f'BatchNorm2d node={node.name}: {label} must have shape {(channels,)}, got {value.shape}')
    eps = float(node.attrs.get('eps', 1e-5))
    momentum = float(node.attrs.get('momentum', 0.1))
    mode = str(node.attrs.get('mode', 'eval'))
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'bn_eval_forward')
        and mode == 'eval'
    ):
        output = ctx.runtime.allocate_staging_buffer(
            tuple(int(v) for v in x.shape),
            dtype='float32',
            name=node.outputs[0],
        )
        running_mean_t = ctx.runtime.stage_to_device(running_mean, name=running_mean_key)
        running_var_t = ctx.runtime.stage_to_device(running_var, name=running_var_key)
        gamma_t = ctx.runtime.stage_to_device(gamma, name=f'_w_{node.name}')
        beta_t = ctx.runtime.stage_to_device(beta, name=f'_b_{node.name}')
        try:
            ctx.runtime.bound_lib.bn_eval_forward(
                output.device_ptr,
                input_tensor.device_ptr,
                running_mean_t.device_ptr,
                running_var_t.device_ptr,
                gamma_t.device_ptr,
                beta_t.device_ptr,
                int(x.shape[0]),
                int(x.shape[1]),
                int(x.shape[2]),
                int(x.shape[3]),
                eps,
            )
            ctx.runtime.record_execution(
                'gpu_native_kernel:bn_eval_forward',
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                node_count=1,
            )
            ctx.runtime.sync_tensor_to_host(output)
        finally:
            ctx.runtime.release_buffer(running_mean_t)
            ctx.runtime.release_buffer(running_var_t)
            ctx.runtime.release_buffer(gamma_t)
            ctx.runtime.release_buffer(beta_t)
        return output
    output, cache = _batchnorm2d_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        running_mean=running_mean,
        running_var=running_var,
        eps=eps,
        momentum=momentum,
        mode=mode,
    )
    ctx.params[running_mean_key] = cache['running_mean']
    ctx.params[running_var_key] = cache['running_var']
    return _allocate_output(node, ctx, output)


def lower_layernorm2d(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f'LayerNorm2d node={node.name}: expected 4-D NCHW input, got {x.shape}')
    channels = int(x.shape[1])
    gamma = np.asarray(ctx.params.get(f'_w_{node.name}', np.ones(channels, dtype=np.float32)), dtype=np.float32)
    beta = np.asarray(ctx.params.get(f'_b_{node.name}', np.zeros(channels, dtype=np.float32)), dtype=np.float32)
    for label, value in (('weight', gamma), ('bias', beta)):
        if value.shape != (channels,):
            raise ValueError(f'LayerNorm2d node={node.name}: {label} must have shape {(channels,)}, got {value.shape}')
    eps = float(node.attrs.get('eps', 1e-6))
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'layernorm2d_forward')
    ):
        output = ctx.runtime.allocate_staging_buffer(
            tuple(int(v) for v in x.shape),
            dtype='float32',
            name=node.outputs[0],
        )
        gamma_t = ctx.runtime.stage_to_device(gamma, name=f'_w_{node.name}')
        beta_t = ctx.runtime.stage_to_device(beta, name=f'_b_{node.name}')
        try:
            ctx.runtime.bound_lib.layernorm2d_forward(
                input_tensor.device_ptr,
                gamma_t.device_ptr,
                beta_t.device_ptr,
                output.device_ptr,
                int(x.shape[0]),
                int(x.shape[1]),
                int(x.shape[2]),
                int(x.shape[3]),
                eps,
            )
            ctx.runtime.record_execution(
                'gpu_native_kernel:layernorm2d_forward',
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                node_count=1,
            )
            ctx.runtime.sync_tensor_to_host(output)
        finally:
            ctx.runtime.release_buffer(gamma_t)
            ctx.runtime.release_buffer(beta_t)
        return output
    output, _cache = _layernorm2d_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        eps=eps,
    )
    return _allocate_output(node, ctx, output)


def lower_layernorm(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    normalized_shape_attr = node.attrs.get('normalized_shape')
    if normalized_shape_attr is None:
        raise ValueError(f'LayerNorm node={node.name}: missing required attr "normalized_shape"')
    if isinstance(normalized_shape_attr, int):
        normalized_shape = (int(normalized_shape_attr),)
    else:
        normalized_shape = tuple(int(v) for v in normalized_shape_attr)
    if not normalized_shape:
        raise ValueError(f'LayerNorm node={node.name}: normalized_shape must contain at least one dimension')
    gamma = np.asarray(
        ctx.params.get(f'_w_{node.name}', np.ones(normalized_shape, dtype=np.float32)),
        dtype=np.float32,
    )
    beta = np.asarray(
        ctx.params.get(f'_b_{node.name}', np.zeros(normalized_shape, dtype=np.float32)),
        dtype=np.float32,
    )
    for label, value in (('weight', gamma), ('bias', beta)):
        if value.shape != normalized_shape:
            raise ValueError(
                f'LayerNorm node={node.name}: {label} must have shape {normalized_shape}, got {value.shape}'
            )
    eps = float(node.attrs.get('eps', 1e-5))
    feature_count = int(np.prod(normalized_shape, dtype=np.int64))
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'layernorm_nd_forward')
    ):
        outer_shape = tuple(int(v) for v in x.shape[:-len(normalized_shape)]) if len(normalized_shape) < x.ndim else tuple()
        rows = int(np.prod(outer_shape, dtype=np.int64)) if outer_shape else 1
        output = ctx.runtime.allocate_staging_buffer(
            tuple(int(v) for v in x.shape),
            dtype='float32',
            name=node.outputs[0],
        )
        gamma_t = ctx.runtime.stage_to_device(gamma, name=f'_w_{node.name}')
        beta_t = ctx.runtime.stage_to_device(beta, name=f'_b_{node.name}')
        try:
            ctx.runtime.bound_lib.layernorm_nd_forward(
                input_tensor.device_ptr,
                gamma_t.device_ptr,
                beta_t.device_ptr,
                output.device_ptr,
                rows,
                feature_count,
                float(eps),
            )
            return output
        finally:
            ctx.runtime.release_buffer(gamma_t)
            ctx.runtime.release_buffer(beta_t)
    output, _cache = _layernorm_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        normalized_shape=normalized_shape,
        eps=eps,
    )
    return _allocate_output(node, ctx, output)


def lower_groupnorm(node: Node, ctx: GpuLoweringContext) -> DeviceTensor:
    input_tensor = _input_tensor(node, ctx)
    x = np.asarray(input_tensor.data, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f'GroupNorm node={node.name}: expected 4-D NCHW input, got {x.shape}')
    channels = int(x.shape[1])
    num_groups = int(node.attrs.get('num_groups', 0))
    if num_groups <= 0:
        raise ValueError(f'GroupNorm node={node.name}: attr "num_groups" must be > 0, got {num_groups}')
    if channels % num_groups != 0:
        raise ValueError(f'GroupNorm node={node.name}: num_groups={num_groups} must divide channels={channels}')
    gamma = np.asarray(ctx.params.get(f'_w_{node.name}', np.ones(channels, dtype=np.float32)), dtype=np.float32)
    beta = np.asarray(ctx.params.get(f'_b_{node.name}', np.zeros(channels, dtype=np.float32)), dtype=np.float32)
    for label, value in (('weight', gamma), ('bias', beta)):
        if value.shape != (channels,):
            raise ValueError(f'GroupNorm node={node.name}: {label} must have shape {(channels,)}, got {value.shape}')
    eps = float(node.attrs.get('eps', 1e-5))
    if (
        ctx.runtime.native_device_pointers_enabled
        and input_tensor.device_ptr is not None
        and hasattr(ctx.runtime.bound_lib, 'groupnorm_forward')
    ):
        output = ctx.runtime.allocate_staging_buffer(
            tuple(int(v) for v in x.shape),
            dtype='float32',
            name=node.outputs[0],
        )
        gamma_t = ctx.runtime.stage_to_device(gamma, name=f'_w_{node.name}')
        beta_t = ctx.runtime.stage_to_device(beta, name=f'_b_{node.name}')
        try:
            ctx.runtime.bound_lib.groupnorm_forward(
                input_tensor.device_ptr,
                gamma_t.device_ptr,
                beta_t.device_ptr,
                output.device_ptr,
                int(x.shape[0]),
                channels,
                int(x.shape[2]),
                int(x.shape[3]),
                num_groups,
                eps,
            )
            ctx.runtime.record_execution(
                'gpu_native_kernel:groupnorm_forward',
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                node_count=1,
            )
            ctx.runtime.sync_tensor_to_host(output)
        finally:
            ctx.runtime.release_buffer(gamma_t)
            ctx.runtime.release_buffer(beta_t)
        return output
    output, _cache = _groupnorm_forward_array(
        x,
        gamma=gamma,
        beta=beta,
        num_groups=num_groups,
        eps=eps,
    )
    return _allocate_output(node, ctx, output)
