from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime


def _load_bound_lib(bound_lib: Any | None) -> Any:
    if bound_lib is not None:
        from minicnn.core._cuda_library import ensure_cuda_runtime_available

        ensure_cuda_runtime_available(bound_lib)
        return bound_lib
    from minicnn.core._cuda_library import bind_symbols, ensure_cuda_runtime_available, load_library

    lib = bind_symbols(load_library())
    ensure_cuda_runtime_available(lib)
    return lib


def _apply_global_grad_clip(
    runtime: DeviceRuntime,
    lib: Any,
    grad_tensors: tuple[tuple[Any, int], ...],
    max_norm: float,
) -> None:
    if float(max_norm) <= 0.0:
        return
    grad_norm_sumsq_t = runtime.allocate((1,), dtype='float32', name='grad_norm_sumsq')
    try:
        lib.gpu_memset(grad_norm_sumsq_t.device_ptr, 0, grad_norm_sumsq_t.nbytes)
        for grad_t, size in grad_tensors:
            lib.grad_l2_sumsq(grad_t.device_ptr, grad_norm_sumsq_t.device_ptr, int(size))
        grad_sumsq = float(runtime.stage_to_host(grad_norm_sumsq_t)[0])
        grad_norm = float(np.sqrt(max(grad_sumsq, 0.0)))
        if grad_norm > float(max_norm) and grad_norm > 0.0:
            clip_scale = float(max_norm) / (grad_norm + 1e-12)
            for grad_t, size in grad_tensors:
                lib.scale_inplace(grad_t.device_ptr, clip_scale, int(size))
        runtime.record_execution(
            'gpu_native_train:grad_clip_global',
            input_name='gradients',
            output_name='gradients',
            node_count=1,
        )
    finally:
        runtime.release_buffer(grad_norm_sumsq_t)


def _run_softmax_xent_loss(
    runtime: DeviceRuntime,
    lib: Any,
    logits_t: Any,
    labels_t: Any,
    probs_t: Any,
    grad_logits_t: Any,
    loss_sum_t: Any,
    correct_t: Any,
    n: int,
    out_f: int,
    *,
    label_smoothing: float = 0.0,
) -> str:
    if float(label_smoothing) > 0.0:
        lib.softmax_xent_smooth_grad_loss_acc(
            logits_t.device_ptr,
            labels_t.device_ptr,
            probs_t.device_ptr,
            grad_logits_t.device_ptr,
            loss_sum_t.device_ptr,
            correct_t.device_ptr,
            int(n),
            int(out_f),
            float(label_smoothing),
        )
        return 'gpu_native_train:softmax_xent_smooth_grad_loss_acc'
    lib.softmax_xent_grad_loss_acc(
        logits_t.device_ptr,
        labels_t.device_ptr,
        probs_t.device_ptr,
        grad_logits_t.device_ptr,
        loss_sum_t.device_ptr,
        correct_t.device_ptr,
        int(n),
        int(out_f),
    )
    return 'gpu_native_train:softmax_xent_grad_loss_acc'
