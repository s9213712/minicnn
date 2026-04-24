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
    GpuKernelSpec('Add', 'merge', 'elementwise_merge', 2, 1, tuple(), 'match_inputs', 'planned', 'planned'),
    GpuKernelSpec('Concat', 'merge', 'concat_merge', 2, 1, tuple(), 'match_inputs', 'planned', 'planned'),
    GpuKernelSpec('Conv2d', 'conv', 'conv2d_nchw', 1, 1, ('weight', 'bias'), 'NCHW', 'planned', 'planned'),
    GpuKernelSpec('Flatten', 'shape', 'reshape_view', 1, 1, tuple(), 'row_major', 'planned', 'not_needed'),
    GpuKernelSpec('LeakyReLU', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'planned', 'planned'),
    GpuKernelSpec('Linear', 'linear', 'gemm_affine', 1, 1, ('weight', 'bias'), 'row_major', 'planned', 'planned'),
    GpuKernelSpec('MaxPool2d', 'pool', 'pool2d_nchw', 1, 1, tuple(), 'NCHW', 'planned', 'planned'),
    GpuKernelSpec('ReLU', 'activation', 'elementwise_unary', 1, 1, tuple(), 'match_input', 'planned', 'planned'),
)


def list_gpu_kernel_specs() -> list[GpuKernelSpec]:
    return sorted(GPU_KERNEL_BOOTSTRAP_SPECS, key=lambda spec: spec.op_name)
