from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuKernelSpec:
    op_name: str
    category: str
    forward_status: str
    backward_status: str


GPU_KERNEL_BOOTSTRAP_SPECS: tuple[GpuKernelSpec, ...] = (
    GpuKernelSpec('Add', 'merge', 'planned', 'planned'),
    GpuKernelSpec('Concat', 'merge', 'planned', 'planned'),
    GpuKernelSpec('Conv2d', 'conv', 'planned', 'planned'),
    GpuKernelSpec('Flatten', 'shape', 'planned', 'not_needed'),
    GpuKernelSpec('LeakyReLU', 'activation', 'planned', 'planned'),
    GpuKernelSpec('Linear', 'linear', 'planned', 'planned'),
    GpuKernelSpec('MaxPool2d', 'pool', 'planned', 'planned'),
    GpuKernelSpec('ReLU', 'activation', 'planned', 'planned'),
)


def list_gpu_kernel_specs() -> list[GpuKernelSpec]:
    return sorted(GPU_KERNEL_BOOTSTRAP_SPECS, key=lambda spec: spec.op_name)
