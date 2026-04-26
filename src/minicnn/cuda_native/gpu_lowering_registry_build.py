from __future__ import annotations

from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs
from minicnn.cuda_native.gpu_lowering_activation import (
    _lower_gelu,
    _lower_leaky_relu,
    _lower_relu,
    _lower_sigmoid,
    _lower_silu,
    _lower_tanh,
)
from minicnn.cuda_native.gpu_lowering_conv import (
    _lower_avgpool2d,
    _lower_conv2d,
    _lower_global_avgpool2d,
    _lower_maxpool2d,
)
from minicnn.cuda_native.gpu_lowering_merge import (
    _lower_add,
    _lower_concat,
)
from minicnn.cuda_native.gpu_lowering_norm import (
    lower_batchnorm2d,
    lower_groupnorm,
    lower_layernorm,
    lower_layernorm2d,
)
from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringRegistry
from minicnn.cuda_native.gpu_lowering_shape import (
    _lower_flatten,
    _lower_identity_alias,
)
from minicnn.cuda_native.gpu_lowering_utils import (
    allocate_output as _allocate_output,
    input_tensor as _input_tensor,
)
from minicnn.cuda_native.nodes import Node
import numpy as np


def _lower_linear(node: Node, ctx) -> object:
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
            (int(x.shape[0]), int(w.shape[0])),
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


def make_default_gpu_lowering_registry() -> GpuLoweringRegistry:
    registry = GpuLoweringRegistry()
    kernel_categories = {
        spec.op_name: spec.category
        for spec in list_gpu_kernel_specs()
    }
    registry.register('Add', lowering_kind='merge_add_shim', kernel_category=kernel_categories['Add'], fn=_lower_add)
    registry.register('AvgPool2d', lowering_kind='pool_avgpool2d_shim', kernel_category=kernel_categories['AvgPool2d'], fn=_lower_avgpool2d)
    registry.register('Concat', lowering_kind='merge_concat_shim', kernel_category=kernel_categories['Concat'], fn=_lower_concat)
    registry.register('BatchNorm2d', lowering_kind='normalization_batchnorm2d_shim', kernel_category=kernel_categories['BatchNorm2d'], fn=lower_batchnorm2d)
    registry.register('Conv2d', lowering_kind='conv2d_reference_shim', kernel_category=kernel_categories['Conv2d'], fn=_lower_conv2d)
    registry.register('DepthwiseConv2d', lowering_kind='depthwise_conv2d_shim', kernel_category=kernel_categories['DepthwiseConv2d'], fn=_lower_conv2d)
    registry.register('Dropout', lowering_kind='regularization_dropout_p0_alias_shim', kernel_category=kernel_categories['Dropout'], fn=_lower_identity_alias)
    registry.register('DropPath', lowering_kind='regularization_droppath_p0_alias_shim', kernel_category=kernel_categories['DropPath'], fn=_lower_identity_alias)
    registry.register('PointwiseConv2d', lowering_kind='conv2d_reference_shim', kernel_category=kernel_categories['PointwiseConv2d'], fn=_lower_conv2d)
    registry.register('Flatten', lowering_kind='shape_flatten_shim', kernel_category=kernel_categories['Flatten'], fn=_lower_flatten)
    registry.register('GELU', lowering_kind='activation_gelu_shim', kernel_category=kernel_categories['GELU'], fn=_lower_gelu)
    registry.register('AdaptiveAvgPool2d', lowering_kind='pool_global_avgpool2d_shim', kernel_category=kernel_categories['AdaptiveAvgPool2d'], fn=_lower_global_avgpool2d)
    registry.register('GlobalAvgPool2d', lowering_kind='pool_global_avgpool2d_shim', kernel_category=kernel_categories['GlobalAvgPool2d'], fn=_lower_global_avgpool2d)
    registry.register('GroupNorm', lowering_kind='normalization_groupnorm_shim', kernel_category=kernel_categories['GroupNorm'], fn=lower_groupnorm)
    registry.register('Identity', lowering_kind='shape_identity_alias_shim', kernel_category=kernel_categories['Identity'], fn=_lower_identity_alias)
    registry.register('LayerNorm', lowering_kind='normalization_layernorm_shim', kernel_category=kernel_categories['LayerNorm'], fn=lower_layernorm)
    registry.register('LayerNorm2d', lowering_kind='normalization_layernorm2d_shim', kernel_category=kernel_categories['LayerNorm2d'], fn=lower_layernorm2d)
    registry.register('LeakyReLU', lowering_kind='activation_leaky_relu_shim', kernel_category=kernel_categories['LeakyReLU'], fn=_lower_leaky_relu)
    registry.register('Linear', lowering_kind='linear_affine_shim', kernel_category=kernel_categories['Linear'], fn=_lower_linear)
    registry.register('MaxPool2d', lowering_kind='pool_maxpool2d_shim', kernel_category=kernel_categories['MaxPool2d'], fn=_lower_maxpool2d)
    registry.register('ReLU', lowering_kind='activation_relu_shim', kernel_category=kernel_categories['ReLU'], fn=_lower_relu)
    registry.register('Sigmoid', lowering_kind='activation_sigmoid_shim', kernel_category=kernel_categories['Sigmoid'], fn=_lower_sigmoid)
    registry.register('SiLU', lowering_kind='activation_silu_shim', kernel_category=kernel_categories['SiLU'], fn=_lower_silu)
    registry.register('Tanh', lowering_kind='activation_tanh_shim', kernel_category=kernel_categories['Tanh'], fn=_lower_tanh)
    return registry
