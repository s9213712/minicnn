from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.unified._cuda_native_context import NativeTrainingContext
from minicnn.unified._cuda_native_gpu_train_dispatch_common import (
    finalize_gpu_native_training_batch,
)
from minicnn.unified._cuda_native_gpu_train_dispatch_conv import (
    run_gpu_native_conv_batch,
)
from minicnn.unified._cuda_native_gpu_train_dispatch_linear import (
    run_gpu_native_linear_or_pool_batch,
)
from minicnn.unified._cuda_native_gpu_train_dispatch_norm import (
    run_gpu_native_norm_batch,
)


def run_gpu_native_training_batch(
    ctx: NativeTrainingContext,
    *,
    optimizer_view: Any,
    optimizer_state: dict[str, Any],
    gpu_training_plan: dict[str, Any],
    params: dict[str, np.ndarray],
    xb: np.ndarray,
    yb: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, int]:
    batch_result = run_gpu_native_linear_or_pool_batch(
        ctx,
        optimizer_view=optimizer_view,
        optimizer_state=optimizer_state,
        gpu_training_plan=gpu_training_plan,
        params=params,
        xb=xb,
        yb=yb,
    )
    if batch_result is None:
        batch_result = run_gpu_native_norm_batch(
            ctx,
            optimizer_view=optimizer_view,
            optimizer_state=optimizer_state,
            gpu_training_plan=gpu_training_plan,
            params=params,
            xb=xb,
            yb=yb,
        )
    if batch_result is None:
        batch_result = run_gpu_native_conv_batch(
            ctx,
            optimizer_view=optimizer_view,
            optimizer_state=optimizer_state,
            gpu_training_plan=gpu_training_plan,
            params=params,
            xb=xb,
            yb=yb,
        )
    if batch_result is None:
        raise RuntimeError(
            f"unhandled gpu_native training plan kind: {gpu_training_plan['kind']!r}"
        )
    params, step = batch_result
    loss_mean, correct_count = finalize_gpu_native_training_batch(
        ctx,
        optimizer_state=optimizer_state,
        step=step,
    )
    return params, loss_mean, correct_count
