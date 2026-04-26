from __future__ import annotations

from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringSpec
from minicnn.cuda_native.gpu_lowering_registry_build import (
    make_default_gpu_lowering_registry,
)


def list_gpu_lowering_specs() -> list[GpuLoweringSpec]:
    return make_default_gpu_lowering_registry().registered_specs()
