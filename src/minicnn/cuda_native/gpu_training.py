from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime


@dataclass(frozen=True)
class NativeGpuLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    grad_input: np.ndarray
    grad_weight: np.ndarray
    grad_bias: np.ndarray
    updated_weight: np.ndarray
    updated_bias: np.ndarray
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


def _load_bound_lib(bound_lib: Any | None) -> Any:
    if bound_lib is not None:
        return bound_lib
    from minicnn.core._cuda_library import bind_symbols, load_library

    return bind_symbols(load_library())


def native_gpu_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuLinearTrainingStepResult:
    """Run one native GPU Linear + SoftmaxCE + SGD training step.

    This is intentionally narrow: it proves the native C ABI can execute a
    complete forward/backward/update cycle without host-side gradient math.
    Full graph-level train-native integration remains a separate layer.
    """

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    weight_f32 = np.ascontiguousarray(weight, dtype=np.float32)
    bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
    if x_f32.ndim != 2:
        raise ValueError(f'native_gpu_linear_training_step expects x with shape (N, in_f), got {x_f32.shape}')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != x_f32.shape[0]:
        raise ValueError(
            'native_gpu_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if weight_f32.ndim != 2 or weight_f32.shape[1] != x_f32.shape[1]:
        raise ValueError(
            'native_gpu_linear_training_step expects weight with shape (out_f, in_f), '
            f'got weight={weight_f32.shape} for x={x_f32.shape}'
        )
    if bias_f32.shape != (weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_linear_training_step expects bias with shape (out_f,), '
            f'got bias={bias_f32.shape} for weight={weight_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    n, in_f = int(x_f32.shape[0]), int(x_f32.shape[1])
    out_f = int(weight_f32.shape[0])
    input_t = runtime.stage_to_device(x_f32, name='input')
    labels_t = runtime.stage_to_device(labels_i32, name='labels')
    weight_t = runtime.stage_to_device(weight_f32, name='weight')
    bias_t = runtime.stage_to_device(bias_f32, name='bias')
    logits_t = runtime.allocate((n, out_f), dtype='float32', name='logits')
    probs_t = runtime.allocate((n, out_f), dtype='float32', name='probs')
    grad_logits_t = runtime.allocate((n, out_f), dtype='float32', name='grad_logits')
    grad_input_t = runtime.allocate((n, in_f), dtype='float32', name='grad_input')
    grad_weight_t = runtime.allocate((out_f, in_f), dtype='float32', name='grad_weight')
    grad_bias_t = runtime.allocate((out_f,), dtype='float32', name='grad_bias')
    loss_sum_t = runtime.allocate((1,), dtype='float32', name='loss_sum')
    correct_t = runtime.allocate((1,), dtype='int32', name='correct_count')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.dense_forward(
            input_t.device_ptr,
            weight_t.device_ptr,
            bias_t.device_ptr,
            logits_t.device_ptr,
            n,
            in_f,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_forward', input_name='input', output_name='logits', node_count=1)
        lib.softmax_xent_grad_loss_acc(
            logits_t.device_ptr,
            labels_t.device_ptr,
            probs_t.device_ptr,
            grad_logits_t.device_ptr,
            loss_sum_t.device_ptr,
            correct_t.device_ptr,
            n,
            out_f,
        )
        runtime.record_execution(
            'gpu_native_train:softmax_xent_grad_loss_acc',
            input_name='logits',
            output_name='grad_logits',
            node_count=1,
        )
        lib.dense_backward_full(
            grad_logits_t.device_ptr,
            input_t.device_ptr,
            weight_t.device_ptr,
            grad_input_t.device_ptr,
            grad_weight_t.device_ptr,
            grad_bias_t.device_ptr,
            n,
            in_f,
            out_f,
        )
        runtime.record_execution(
            'gpu_native_train:dense_backward_full',
            input_name='grad_logits',
            output_name='grad_weight',
            node_count=1,
        )
        lib.apply_sgd_update(weight_t.device_ptr, grad_weight_t.device_ptr, float(lr), int(weight_f32.size))
        lib.apply_sgd_update(bias_t.device_ptr, grad_bias_t.device_ptr, float(lr), int(bias_f32.size))
        runtime.record_execution(
            'gpu_native_train:apply_sgd_update',
            input_name='grad_weight',
            output_name='weight',
            node_count=1,
        )

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        grad_logits = runtime.stage_to_host(grad_logits_t)
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_weight = runtime.stage_to_host(grad_weight_t)
        grad_bias = runtime.stage_to_host(grad_bias_t)
        updated_weight = runtime.stage_to_host(weight_t)
        updated_bias = runtime.stage_to_host(bias_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-linear-training-step')
        return NativeGpuLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            grad_input=grad_input,
            grad_weight=grad_weight,
            grad_bias=grad_bias,
            updated_weight=updated_weight,
            updated_bias=updated_bias,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in (
            input_t,
            labels_t,
            weight_t,
            bias_t,
            logits_t,
            probs_t,
            grad_logits_t,
            grad_input_t,
            grad_weight_t,
            grad_bias_t,
            loss_sum_t,
            correct_t,
        ):
            runtime.release_buffer(tensor)
