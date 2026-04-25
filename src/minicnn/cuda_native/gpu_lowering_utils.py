from __future__ import annotations

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceTensor
from minicnn.cuda_native.gpu_lowering_registry import GpuLoweringContext
from minicnn.cuda_native.nodes import Node


def allocate_output(node: Node, ctx: GpuLoweringContext, output: np.ndarray) -> DeviceTensor:
    output_array = np.asarray(output, dtype=np.float32)
    staged = ctx.runtime.allocate_staging_buffer(
        tuple(int(v) for v in output_array.shape),
        dtype=output_array.dtype,
        name=node.outputs[0],
    )
    np.copyto(staged.data, output_array)
    ctx.runtime.sync_tensor_to_device(staged)
    return staged


def input_tensor(node: Node, ctx: GpuLoweringContext, index: int = 0) -> DeviceTensor:
    return ctx.tensors[node.inputs[index]]
