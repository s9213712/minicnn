from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from minicnn.cuda_native.device_runtime import DeviceRuntime, DeviceTensor
from minicnn.cuda_native.nodes import Node


@dataclass
class GpuLoweringContext:
    tensors: dict[str, DeviceTensor]
    params: dict[str, Any]
    runtime: DeviceRuntime
    mode: str = 'eval'


GpuLoweringFn = Callable[[Node, GpuLoweringContext], DeviceTensor]


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

    def register(
        self,
        op_name: str,
        *,
        lowering_kind: str,
        kernel_category: str,
        fn: GpuLoweringFn,
    ) -> 'GpuLoweringRegistry':
        spec = GpuLoweringSpec(
            op_name=op_name,
            lowering_kind=lowering_kind,
            kernel_category=kernel_category,
            fn=fn,
        )
        self._dispatch[op_name] = fn
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
