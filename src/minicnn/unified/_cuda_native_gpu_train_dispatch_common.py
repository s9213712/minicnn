from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.unified._cuda_native_context import NativeTrainingContext
from minicnn.unified._cuda_native_training_plan import _merge_gpu_native_step_runtime


def finalize_gpu_native_training_batch(
    ctx: NativeTrainingContext,
    *,
    optimizer_state: dict[str, Any],
    step: Any,
) -> tuple[float, int]:
    _merge_gpu_native_step_runtime(ctx, step.runtime_summary)
    optimizer_runtime = optimizer_state.setdefault('optimizer_runtime', {})
    optimizer_runtime['optimizer_type'] = ctx.optimizer_type
    optimizer_runtime['steps'] = int(optimizer_runtime.get('steps', 0)) + 1
    ctx.device_runtime.record_execution(
        'gpu_native_train_batch',
        input_name=ctx.graph.input_spec.name if ctx.graph.input_spec is not None else 'input',
        output_name=ctx.graph.output_spec.name if ctx.graph.output_spec is not None else 'output',
        node_count=len(ctx.graph.nodes),
    )
    ctx.device_runtime.synchronize('gpu-native-train-batch')
    return float(step.loss_mean), int(step.correct_count)
