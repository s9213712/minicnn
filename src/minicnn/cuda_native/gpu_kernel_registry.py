from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuKernelSpec:
    op_name: str
    category: str
    launch_family: str
    input_arity: int
    output_arity: int
    param_kinds: tuple[str, ...]
    preferred_layout: str
    forward_status: str
    backward_status: str


GPU_KERNEL_BOOTSTRAP_SPECS: tuple[GpuKernelSpec, ...] = (
    GpuKernelSpec('Add', 'merge', 'elementwise_merge', 2, 1, tuple(), 'match_inputs', 'native_forward', 'planned'),
    GpuKernelSpec('AvgPool2d', 'pool', 'avgpool2d_nchw', 1, 1, tuple(), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('BatchNorm2d', 'normalization', 'batchnorm2d_nchw', 1, 1, ('weight', 'bias', 'running_mean', 'running_var'), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('Concat', 'merge', 'concat_merge', 2, 1, tuple(), 'match_inputs', 'native_forward', 'planned'),
    GpuKernelSpec('Conv2d', 'conv', 'conv2d_nchw', 1, 1, ('weight', 'bias'), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('AdaptiveAvgPool2d', 'pool', 'global_avgpool2d_nchw', 1, 1, tuple(), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('DepthwiseConv2d', 'conv', 'depthwise_conv2d_nchw', 1, 1, ('weight', 'bias'), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('Dropout', 'regularization', 'identity_alias', 1, 1, tuple(), 'match_input', 'conditional_alias', 'planned'),
    GpuKernelSpec('DropPath', 'regularization', 'identity_alias', 1, 1, tuple(), 'match_input', 'conditional_alias', 'planned'),
    GpuKernelSpec('Flatten', 'shape', 'reshape_view', 1, 1, tuple(), 'row_major', 'native_alias', 'not_needed'),
    GpuKernelSpec('GELU', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'partial_native', 'partial_native'),
    GpuKernelSpec('GlobalAvgPool2d', 'pool', 'global_avgpool2d_nchw', 1, 1, tuple(), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('GroupNorm', 'normalization', 'groupnorm_nchw', 1, 1, ('weight', 'bias'), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('Identity', 'shape', 'identity_alias', 1, 1, tuple(), 'match_input', 'native_alias', 'not_needed'),
    GpuKernelSpec('LayerNorm', 'normalization', 'layernorm_nd', 1, 1, ('weight', 'bias'), 'match_input', 'partial_native', 'partial_native'),
    GpuKernelSpec('LayerNorm2d', 'normalization', 'layernorm2d_nchw', 1, 1, ('weight', 'bias'), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('LeakyReLU', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'native_forward', 'planned'),
    GpuKernelSpec('Linear', 'linear', 'gemm_affine', 1, 1, ('weight', 'bias'), 'row_major', 'native_forward', 'partial_native'),
    GpuKernelSpec('MaxPool2d', 'pool', 'pool2d_nchw', 1, 1, tuple(), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('PointwiseConv2d', 'conv', 'conv2d_nchw', 1, 1, ('weight', 'bias'), 'NCHW', 'partial_native', 'partial_native'),
    GpuKernelSpec('ReLU', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'native_forward', 'partial_native'),
    GpuKernelSpec('Sigmoid', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'partial_native', 'partial_native'),
    GpuKernelSpec('SiLU', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'partial_native', 'partial_native'),
    GpuKernelSpec('Tanh', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'partial_native', 'partial_native'),
)


def list_gpu_kernel_specs() -> list[GpuKernelSpec]:
    return sorted(GPU_KERNEL_BOOTSTRAP_SPECS, key=lambda spec: spec.op_name)
