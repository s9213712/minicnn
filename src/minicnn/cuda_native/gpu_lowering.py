from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs
from minicnn.cuda_native.kernels import KernelRegistry, make_default_registry
from minicnn.cuda_native.nodes import Node


GpuLoweringFn = Callable[[Node, dict[str, Any]], None]


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

    def register_reference_kernel(
        self,
        op_name: str,
        *,
        kernel_registry: KernelRegistry,
    ) -> 'GpuLoweringRegistry':
        kernel_spec = kernel_registry.spec(op_name)
        spec = GpuLoweringSpec(
            op_name=op_name,
            lowering_kind='reference_kernel_shim',
            kernel_category=kernel_spec.category,
            fn=kernel_spec.fn,
        )
        self._dispatch[op_name] = kernel_spec.fn
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


def make_default_gpu_lowering_registry(
    *,
    kernel_registry: KernelRegistry | None = None,
) -> GpuLoweringRegistry:
    registry = GpuLoweringRegistry()
    kernels = kernel_registry if kernel_registry is not None else make_default_registry()
    for kernel_spec in list_gpu_kernel_specs():
        registry.register_reference_kernel(kernel_spec.op_name, kernel_registry=kernels)
    return registry


def list_gpu_lowering_specs() -> list[GpuLoweringSpec]:
    return make_default_gpu_lowering_registry().registered_specs()
