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
    updated_weight_velocity: np.ndarray | None
    updated_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuTwoLinearReluTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    grad_hidden: np.ndarray
    grad_input: np.ndarray
    grad_weight1: np.ndarray
    grad_bias1: np.ndarray
    grad_weight2: np.ndarray
    grad_bias2: np.ndarray
    updated_weight1: np.ndarray
    updated_bias1: np.ndarray
    updated_weight2: np.ndarray
    updated_bias2: np.ndarray
    updated_weight1_velocity: np.ndarray | None
    updated_bias1_velocity: np.ndarray | None
    updated_weight2_velocity: np.ndarray | None
    updated_bias2_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuPoolLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    grad_logits: np.ndarray
    pooled: np.ndarray
    grad_pooled: np.ndarray
    grad_input: np.ndarray
    grad_weight: np.ndarray
    grad_bias: np.ndarray
    updated_weight: np.ndarray
    updated_bias: np.ndarray
    updated_weight_velocity: np.ndarray | None
    updated_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeGpuConvLinearTrainingStepResult:
    logits: np.ndarray
    probabilities: np.ndarray
    conv_output: np.ndarray
    grad_logits: np.ndarray
    grad_conv_output: np.ndarray
    grad_input: np.ndarray
    grad_conv_weight: np.ndarray
    grad_linear_weight: np.ndarray
    grad_linear_bias: np.ndarray
    updated_conv_weight: np.ndarray
    updated_linear_weight: np.ndarray
    updated_linear_bias: np.ndarray
    updated_conv_weight_velocity: np.ndarray | None
    updated_linear_weight_velocity: np.ndarray | None
    updated_linear_bias_velocity: np.ndarray | None
    loss_sum: float
    loss_mean: float
    correct_count: int
    runtime_summary: dict[str, Any]
    pooled_output: np.ndarray | None = None
    grad_pooled: np.ndarray | None = None


def _load_bound_lib(bound_lib: Any | None) -> Any:
    if bound_lib is not None:
        from minicnn.core._cuda_library import ensure_cuda_runtime_available

        ensure_cuda_runtime_available(bound_lib)
        return bound_lib
    from minicnn.core._cuda_library import bind_symbols, ensure_cuda_runtime_available, load_library

    lib = bind_symbols(load_library())
    ensure_cuda_runtime_available(lib)
    return lib


def native_gpu_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
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
    weight_velocity_f32 = (
        np.zeros_like(weight_f32)
        if weight_velocity is None
        else np.ascontiguousarray(weight_velocity, dtype=np.float32)
    )
    bias_velocity_f32 = (
        np.zeros_like(bias_f32)
        if bias_velocity is None
        else np.ascontiguousarray(bias_velocity, dtype=np.float32)
    )
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
    if weight_velocity_f32.shape != weight_f32.shape:
        raise ValueError(
            'native_gpu_linear_training_step expects weight_velocity with same shape as weight, '
            f'got weight_velocity={weight_velocity_f32.shape} and weight={weight_f32.shape}'
        )
    if bias_velocity_f32.shape != bias_f32.shape:
        raise ValueError(
            'native_gpu_linear_training_step expects bias_velocity with same shape as bias, '
            f'got bias_velocity={bias_velocity_f32.shape} and bias={bias_f32.shape}'
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
    weight_velocity_t = runtime.stage_to_device(weight_velocity_f32, name='weight_velocity')
    bias_velocity_t = runtime.stage_to_device(bias_velocity_f32, name='bias_velocity')
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
        if float(momentum) != 0.0:
            lib.apply_momentum_update(
                weight_t.device_ptr,
                grad_weight_t.device_ptr,
                weight_velocity_t.device_ptr,
                float(lr),
                float(momentum),
                int(weight_f32.size),
            )
            lib.apply_momentum_update(
                bias_t.device_ptr,
                grad_bias_t.device_ptr,
                bias_velocity_t.device_ptr,
                float(lr),
                float(momentum),
                int(bias_f32.size),
            )
            update_kind = 'gpu_native_train:apply_momentum_update'
        else:
            lib.apply_sgd_update(weight_t.device_ptr, grad_weight_t.device_ptr, float(lr), int(weight_f32.size))
            lib.apply_sgd_update(bias_t.device_ptr, grad_bias_t.device_ptr, float(lr), int(bias_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(
            update_kind,
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
        updated_weight_velocity = runtime.stage_to_host(weight_velocity_t)
        updated_bias_velocity = runtime.stage_to_host(bias_velocity_t)
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
            updated_weight_velocity=updated_weight_velocity,
            updated_bias_velocity=updated_bias_velocity,
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
            weight_velocity_t,
            bias_velocity_t,
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


def native_gpu_two_linear_relu_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight1: np.ndarray,
    bias1: np.ndarray,
    weight2: np.ndarray,
    bias2: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    weight1_velocity: np.ndarray | None = None,
    bias1_velocity: np.ndarray | None = None,
    weight2_velocity: np.ndarray | None = None,
    bias2_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuTwoLinearReluTrainingStepResult:
    """Run one native GPU Linear + ReLU + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    w1_f32 = np.ascontiguousarray(weight1, dtype=np.float32)
    b1_f32 = np.ascontiguousarray(bias1, dtype=np.float32)
    w2_f32 = np.ascontiguousarray(weight2, dtype=np.float32)
    b2_f32 = np.ascontiguousarray(bias2, dtype=np.float32)
    w1v_f32 = np.zeros_like(w1_f32) if weight1_velocity is None else np.ascontiguousarray(weight1_velocity, dtype=np.float32)
    b1v_f32 = np.zeros_like(b1_f32) if bias1_velocity is None else np.ascontiguousarray(bias1_velocity, dtype=np.float32)
    w2v_f32 = np.zeros_like(w2_f32) if weight2_velocity is None else np.ascontiguousarray(weight2_velocity, dtype=np.float32)
    b2v_f32 = np.zeros_like(b2_f32) if bias2_velocity is None else np.ascontiguousarray(bias2_velocity, dtype=np.float32)
    if x_f32.ndim != 2:
        raise ValueError(f'native_gpu_two_linear_relu_training_step expects x with shape (N, in_f), got {x_f32.shape}')
    if w1_f32.ndim != 2 or w1_f32.shape[1] != x_f32.shape[1]:
        raise ValueError(
            'native_gpu_two_linear_relu_training_step expects weight1 with shape (hidden_f, in_f), '
            f'got weight1={w1_f32.shape} for x={x_f32.shape}'
        )
    if b1_f32.shape != (w1_f32.shape[0],):
        raise ValueError(
            'native_gpu_two_linear_relu_training_step expects bias1 with shape (hidden_f,), '
            f'got bias1={b1_f32.shape} for weight1={w1_f32.shape}'
        )
    if w2_f32.ndim != 2 or w2_f32.shape[1] != w1_f32.shape[0]:
        raise ValueError(
            'native_gpu_two_linear_relu_training_step expects weight2 with shape (out_f, hidden_f), '
            f'got weight2={w2_f32.shape} for hidden_f={w1_f32.shape[0]}'
        )
    if b2_f32.shape != (w2_f32.shape[0],):
        raise ValueError(
            'native_gpu_two_linear_relu_training_step expects bias2 with shape (out_f,), '
            f'got bias2={b2_f32.shape} for weight2={w2_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != x_f32.shape[0]:
        raise ValueError(
            'native_gpu_two_linear_relu_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= w2_f32.shape[0]):
        raise ValueError('native_gpu_two_linear_relu_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    n, in_f = int(x_f32.shape[0]), int(x_f32.shape[1])
    hidden_f = int(w1_f32.shape[0])
    out_f = int(w2_f32.shape[0])
    tensors = []

    def stage(array: np.ndarray, name: str):
        tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def alloc(shape: tuple[int, ...], name: str, dtype: str = 'float32'):
        tensor = runtime.allocate(shape, dtype=dtype, name=name)
        tensors.append(tensor)
        return tensor

    input_t = stage(x_f32, 'input')
    labels_t = stage(labels_i32, 'labels')
    w1_t = stage(w1_f32, 'weight1')
    b1_t = stage(b1_f32, 'bias1')
    w2_t = stage(w2_f32, 'weight2')
    b2_t = stage(b2_f32, 'bias2')
    w1v_t = stage(w1v_f32, 'weight1_velocity')
    b1v_t = stage(b1v_f32, 'bias1_velocity')
    w2v_t = stage(w2v_f32, 'weight2_velocity')
    b2v_t = stage(b2v_f32, 'bias2_velocity')
    hidden_t = alloc((n, hidden_f), 'hidden')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_hidden_t = alloc((n, hidden_f), 'grad_hidden')
    grad_input_t = alloc((n, in_f), 'grad_input')
    grad_w1_t = alloc((hidden_f, in_f), 'grad_weight1')
    grad_b1_t = alloc((hidden_f,), 'grad_bias1')
    grad_w2_t = alloc((out_f, hidden_f), 'grad_weight2')
    grad_b2_t = alloc((out_f,), 'grad_bias2')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.dense_forward(input_t.device_ptr, w1_t.device_ptr, b1_t.device_ptr, hidden_t.device_ptr, n, in_f, hidden_f)
        runtime.record_execution('gpu_native_train:dense_forward_1', input_name='input', output_name='hidden', node_count=1)
        lib.apply_relu(hidden_t.device_ptr, int(n * hidden_f))
        runtime.record_execution('gpu_native_train:apply_relu', input_name='hidden', output_name='hidden', node_count=1)
        lib.dense_forward(hidden_t.device_ptr, w2_t.device_ptr, b2_t.device_ptr, logits_t.device_ptr, n, hidden_f, out_f)
        runtime.record_execution('gpu_native_train:dense_forward_2', input_name='hidden', output_name='logits', node_count=1)
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
            hidden_t.device_ptr,
            w2_t.device_ptr,
            grad_hidden_t.device_ptr,
            grad_w2_t.device_ptr,
            grad_b2_t.device_ptr,
            n,
            hidden_f,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full_2', input_name='grad_logits', output_name='grad_weight2', node_count=1)
        lib.apply_relu_backward(hidden_t.device_ptr, grad_hidden_t.device_ptr, int(n * hidden_f))
        runtime.record_execution('gpu_native_train:apply_relu_backward', input_name='hidden', output_name='grad_hidden', node_count=1)
        lib.dense_backward_full(
            grad_hidden_t.device_ptr,
            input_t.device_ptr,
            w1_t.device_ptr,
            grad_input_t.device_ptr,
            grad_w1_t.device_ptr,
            grad_b1_t.device_ptr,
            n,
            in_f,
            hidden_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full_1', input_name='grad_hidden', output_name='grad_weight1', node_count=1)
        if float(momentum) != 0.0:
            updates = (
                (w1_t, grad_w1_t, w1v_t, int(w1_f32.size)),
                (b1_t, grad_b1_t, b1v_t, int(b1_f32.size)),
                (w2_t, grad_w2_t, w2v_t, int(w2_f32.size)),
                (b2_t, grad_b2_t, b2v_t, int(b2_f32.size)),
            )
            for value_t, grad_t, velocity_t, size in updates:
                lib.apply_momentum_update(
                    value_t.device_ptr,
                    grad_t.device_ptr,
                    velocity_t.device_ptr,
                    float(lr),
                    float(momentum),
                    size,
                )
            update_kind = 'gpu_native_train:apply_momentum_update'
        else:
            updates = (
                (w1_t, grad_w1_t, int(w1_f32.size)),
                (b1_t, grad_b1_t, int(b1_f32.size)),
                (w2_t, grad_w2_t, int(w2_f32.size)),
                (b2_t, grad_b2_t, int(b2_f32.size)),
            )
            for value_t, grad_t, size in updates:
                lib.apply_sgd_update(value_t.device_ptr, grad_t.device_ptr, float(lr), size)
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_weight1', output_name='weight1', node_count=1)

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        grad_logits = runtime.stage_to_host(grad_logits_t)
        grad_hidden = runtime.stage_to_host(grad_hidden_t)
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_weight1 = runtime.stage_to_host(grad_w1_t)
        grad_bias1 = runtime.stage_to_host(grad_b1_t)
        grad_weight2 = runtime.stage_to_host(grad_w2_t)
        grad_bias2 = runtime.stage_to_host(grad_b2_t)
        updated_weight1 = runtime.stage_to_host(w1_t)
        updated_bias1 = runtime.stage_to_host(b1_t)
        updated_weight2 = runtime.stage_to_host(w2_t)
        updated_bias2 = runtime.stage_to_host(b2_t)
        updated_weight1_velocity = runtime.stage_to_host(w1v_t)
        updated_bias1_velocity = runtime.stage_to_host(b1v_t)
        updated_weight2_velocity = runtime.stage_to_host(w2v_t)
        updated_bias2_velocity = runtime.stage_to_host(b2v_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-two-linear-relu-training-step')
        return NativeGpuTwoLinearReluTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            grad_hidden=grad_hidden,
            grad_input=grad_input,
            grad_weight1=grad_weight1,
            grad_bias1=grad_bias1,
            grad_weight2=grad_weight2,
            grad_bias2=grad_bias2,
            updated_weight1=updated_weight1,
            updated_bias1=updated_bias1,
            updated_weight2=updated_weight2,
            updated_bias2=updated_bias2,
            updated_weight1_velocity=updated_weight1_velocity,
            updated_bias1_velocity=updated_bias1_velocity,
            updated_weight2_velocity=updated_weight2_velocity,
            updated_bias2_velocity=updated_bias2_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_pool_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuPoolLinearTrainingStepResult:
    """Run one native GPU MaxPool2d(2,2) + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    weight_f32 = np.ascontiguousarray(weight, dtype=np.float32)
    bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
    weight_velocity_f32 = (
        np.zeros_like(weight_f32)
        if weight_velocity is None
        else np.ascontiguousarray(weight_velocity, dtype=np.float32)
    )
    bias_velocity_f32 = (
        np.zeros_like(bias_f32)
        if bias_velocity is None
        else np.ascontiguousarray(bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_pool_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError('native_gpu_pool_linear_training_step requires even H/W for 2x2 stride-2 MaxPool2d.')
    pool_h = height // 2
    pool_w = width // 2
    flat_features = channels * pool_h * pool_w
    if weight_f32.ndim != 2 or weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_pool_linear_training_step expects weight with shape (out_f, C*H/2*W/2), '
            f'got weight={weight_f32.shape} for pooled_features={flat_features}'
        )
    if bias_f32.shape != (weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_pool_linear_training_step expects bias with shape (out_f,), '
            f'got bias={bias_f32.shape} for weight={weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_pool_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_pool_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(weight_f32.shape[0])
    tensors = []

    def stage(array: np.ndarray, name: str):
        tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def alloc(shape: tuple[int, ...], name: str, dtype: str = 'float32'):
        tensor = runtime.allocate(shape, dtype=dtype, name=name)
        tensors.append(tensor)
        return tensor

    input_t = stage(x_f32, 'input')
    labels_t = stage(labels_i32, 'labels')
    weight_t = stage(weight_f32, 'weight')
    bias_t = stage(bias_f32, 'bias')
    weight_velocity_t = stage(weight_velocity_f32, 'weight_velocity')
    bias_velocity_t = stage(bias_velocity_f32, 'bias_velocity')
    pooled_t = alloc((n, channels, pool_h, pool_w), 'pooled')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pooled_t = alloc((n, channels, pool_h, pool_w), 'grad_pooled')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_weight_t = alloc((out_f, flat_features), 'grad_weight')
    grad_bias_t = alloc((out_f,), 'grad_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.apply_maxpool(input_t.device_ptr, pooled_t.device_ptr, n, channels, height, width)
        runtime.record_execution('gpu_native_train:apply_maxpool', input_name='input', output_name='pooled', node_count=1)
        lib.dense_forward(pooled_t.device_ptr, weight_t.device_ptr, bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pooled', output_name='logits', node_count=1)
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
            pooled_t.device_ptr,
            weight_t.device_ptr,
            grad_pooled_t.device_ptr,
            grad_weight_t.device_ptr,
            grad_bias_t.device_ptr,
            n,
            flat_features,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_weight', node_count=1)
        lib.maxpool_backward_nchw(
            grad_pooled_t.device_ptr,
            input_t.device_ptr,
            grad_input_t.device_ptr,
            n,
            channels,
            height,
            width,
            pool_h,
            pool_w,
        )
        runtime.record_execution('gpu_native_train:maxpool_backward_nchw', input_name='grad_pooled', output_name='grad_input', node_count=1)
        if float(momentum) != 0.0:
            lib.apply_momentum_update(weight_t.device_ptr, grad_weight_t.device_ptr, weight_velocity_t.device_ptr, float(lr), float(momentum), int(weight_f32.size))
            lib.apply_momentum_update(bias_t.device_ptr, grad_bias_t.device_ptr, bias_velocity_t.device_ptr, float(lr), float(momentum), int(bias_f32.size))
            update_kind = 'gpu_native_train:apply_momentum_update'
        else:
            lib.apply_sgd_update(weight_t.device_ptr, grad_weight_t.device_ptr, float(lr), int(weight_f32.size))
            lib.apply_sgd_update(bias_t.device_ptr, grad_bias_t.device_ptr, float(lr), int(bias_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_weight', output_name='weight', node_count=1)

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        grad_logits = runtime.stage_to_host(grad_logits_t)
        pooled = runtime.stage_to_host(pooled_t)
        grad_pooled = runtime.stage_to_host(grad_pooled_t)
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_weight = runtime.stage_to_host(grad_weight_t)
        grad_bias = runtime.stage_to_host(grad_bias_t)
        updated_weight = runtime.stage_to_host(weight_t)
        updated_bias = runtime.stage_to_host(bias_t)
        updated_weight_velocity = runtime.stage_to_host(weight_velocity_t)
        updated_bias_velocity = runtime.stage_to_host(bias_velocity_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-pool-linear-training-step')
        return NativeGpuPoolLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            pooled=pooled,
            grad_pooled=grad_pooled,
            grad_input=grad_input,
            grad_weight=grad_weight,
            grad_bias=grad_bias,
            updated_weight=updated_weight,
            updated_bias=updated_bias,
            updated_weight_velocity=updated_weight_velocity,
            updated_bias_velocity=updated_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_conv_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    conv_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    conv_weight_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    apply_relu_activation: bool = False,
    apply_maxpool: bool = False,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuConvLinearTrainingStepResult:
    """Run one native GPU Conv2d(valid, no bias) + optional ReLU/MaxPool + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    conv_w_f32 = np.ascontiguousarray(conv_weight, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    conv_wv_f32 = (
        np.zeros_like(conv_w_f32)
        if conv_weight_velocity is None
        else np.ascontiguousarray(conv_weight_velocity, dtype=np.float32)
    )
    linear_wv_f32 = (
        np.zeros_like(linear_w_f32)
        if linear_weight_velocity is None
        else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    )
    linear_bv_f32 = (
        np.zeros_like(linear_b_f32)
        if linear_bias_velocity is None
        else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_conv_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    if conv_w_f32.ndim != 4:
        raise ValueError(
            'native_gpu_conv_linear_training_step expects conv_weight with shape (out_c, in_c, kh, kw), '
            f'got {conv_w_f32.shape}'
        )
    n, in_c, height, width = [int(v) for v in x_f32.shape]
    out_c, conv_in_c, kh, kw = [int(v) for v in conv_w_f32.shape]
    if conv_in_c != in_c:
        raise ValueError(f'conv_weight input channels {conv_in_c} do not match x channels {in_c}')
    out_h = height - kh + 1
    out_w = width - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError('native_gpu_conv_linear_training_step requires valid Conv2d output dimensions.')
    conv_features = out_c * out_h * out_w
    if bool(apply_maxpool):
        if out_h % 2 != 0 or out_w % 2 != 0:
            raise ValueError('native_gpu_conv_linear_training_step requires even Conv2d H/W before 2x2 MaxPool2d.')
        pool_h = out_h // 2
        pool_w = out_w // 2
        dense_features = out_c * pool_h * pool_w
    else:
        pool_h = out_h
        pool_w = out_w
        dense_features = conv_features
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != dense_features:
        raise ValueError(
            'native_gpu_conv_linear_training_step expects linear_weight with shape (classes, flattened_features), '
            f'got linear_weight={linear_w_f32.shape} for flattened_features={dense_features}'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError(
            'native_gpu_conv_linear_training_step expects linear_bias with shape (classes,), '
            f'got {linear_b_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_conv_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_conv_linear_training_step labels must be in [0, classes)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    classes = int(linear_w_f32.shape[0])
    patch_size = in_c * kh * kw
    spatial_size = n * out_h * out_w
    tensors = []

    def stage(array: np.ndarray, name: str):
        tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def alloc(shape: tuple[int, ...], name: str, dtype: str = 'float32'):
        tensor = runtime.allocate(shape, dtype=dtype, name=name)
        tensors.append(tensor)
        return tensor

    input_t = stage(x_f32, 'input')
    labels_t = stage(labels_i32, 'labels')
    conv_w_t = stage(conv_w_f32, 'conv_weight')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    conv_wv_t = stage(conv_wv_f32, 'conv_weight_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    col_t = alloc((patch_size, spatial_size), 'conv_col')
    conv_raw_t = alloc((out_c, n, out_h, out_w), 'conv_raw_cnhw')
    conv_t = alloc((n, out_c, out_h, out_w), 'conv_output')
    pooled_t = alloc((n, out_c, pool_h, pool_w), 'pooled') if bool(apply_maxpool) else None
    logits_t = alloc((n, classes), 'logits')
    probs_t = alloc((n, classes), 'probs')
    grad_logits_t = alloc((n, classes), 'grad_logits')
    grad_conv_t = alloc((n, out_c, out_h, out_w), 'grad_conv_output')
    grad_pooled_t = alloc((n, out_c, pool_h, pool_w), 'grad_pooled') if bool(apply_maxpool) else None
    grad_conv_cnhw_t = alloc((out_c, n, out_h, out_w), 'grad_conv_cnhw')
    grad_input_t = alloc((n, in_c, height, width), 'grad_input')
    grad_conv_w_t = alloc((out_c, in_c, kh, kw), 'grad_conv_weight')
    grad_linear_w_t = alloc((classes, dense_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((classes,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.im2col_forward(input_t.device_ptr, col_t.device_ptr, n, in_c, height, width, kh, kw, out_h, out_w)
        lib.gemm_forward(conv_w_t.device_ptr, col_t.device_ptr, conv_raw_t.device_ptr, out_c, spatial_size, patch_size)
        lib.cnhw_to_nchw(conv_raw_t.device_ptr, conv_t.device_ptr, n, out_c, out_h, out_w)
        runtime.record_execution('gpu_native_train:conv2d_im2col_gemm', input_name='input', output_name='conv_output', node_count=1)
        if bool(apply_relu_activation):
            lib.apply_relu(conv_t.device_ptr, int(n * conv_features))
            runtime.record_execution('gpu_native_train:apply_relu', input_name='conv_output', output_name='conv_output', node_count=1)
        dense_input_t = conv_t
        dense_input_name = 'conv_output'
        if bool(apply_maxpool):
            assert pooled_t is not None
            lib.apply_maxpool(conv_t.device_ptr, pooled_t.device_ptr, n, out_c, out_h, out_w)
            runtime.record_execution('gpu_native_train:apply_maxpool', input_name='conv_output', output_name='pooled', node_count=1)
            dense_input_t = pooled_t
            dense_input_name = 'pooled'
        lib.dense_forward(dense_input_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, dense_features, classes)
        runtime.record_execution('gpu_native_train:dense_forward', input_name=dense_input_name, output_name='logits', node_count=1)
        lib.softmax_xent_grad_loss_acc(
            logits_t.device_ptr,
            labels_t.device_ptr,
            probs_t.device_ptr,
            grad_logits_t.device_ptr,
            loss_sum_t.device_ptr,
            correct_t.device_ptr,
            n,
            classes,
        )
        runtime.record_execution('gpu_native_train:softmax_xent_grad_loss_acc', input_name='logits', output_name='grad_logits', node_count=1)
        dense_grad_t = grad_pooled_t if bool(apply_maxpool) else grad_conv_t
        assert dense_grad_t is not None
        lib.dense_backward_full(
            grad_logits_t.device_ptr,
            dense_input_t.device_ptr,
            linear_w_t.device_ptr,
            dense_grad_t.device_ptr,
            grad_linear_w_t.device_ptr,
            grad_linear_b_t.device_ptr,
            n,
            dense_features,
            classes,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        if bool(apply_maxpool):
            assert grad_pooled_t is not None
            lib.maxpool_backward_nchw(
                grad_pooled_t.device_ptr,
                conv_t.device_ptr,
                grad_conv_t.device_ptr,
                n,
                out_c,
                out_h,
                out_w,
                pool_h,
                pool_w,
            )
            runtime.record_execution('gpu_native_train:maxpool_backward_nchw', input_name='grad_pooled', output_name='grad_conv_output', node_count=1)
        if bool(apply_relu_activation):
            lib.apply_relu_backward(conv_t.device_ptr, grad_conv_t.device_ptr, int(n * conv_features))
            runtime.record_execution('gpu_native_train:apply_relu_backward', input_name='conv_output', output_name='grad_conv_output', node_count=1)
        lib.nchw_to_cnhw(grad_conv_t.device_ptr, grad_conv_cnhw_t.device_ptr, n, out_c, out_h, out_w)
        lib.conv_backward(
            grad_conv_cnhw_t.device_ptr,
            input_t.device_ptr,
            conv_w_t.device_ptr,
            grad_conv_w_t.device_ptr,
            grad_input_t.device_ptr,
            n,
            in_c,
            height,
            width,
            kh,
            kw,
            out_h,
            out_w,
            out_c,
        )
        runtime.record_execution('gpu_native_train:conv_backward', input_name='grad_conv_output', output_name='grad_conv_weight', node_count=1)
        if float(momentum) != 0.0:
            lib.apply_momentum_update(conv_w_t.device_ptr, grad_conv_w_t.device_ptr, conv_wv_t.device_ptr, float(lr), float(momentum), int(conv_w_f32.size))
            lib.apply_momentum_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), int(linear_w_f32.size))
            lib.apply_momentum_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_momentum_update'
        else:
            lib.apply_sgd_update(conv_w_t.device_ptr, grad_conv_w_t.device_ptr, float(lr), int(conv_w_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_conv_weight', output_name='conv_weight', node_count=1)

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        conv_output = runtime.stage_to_host(conv_t)
        pooled_output = runtime.stage_to_host(pooled_t) if pooled_t is not None else None
        grad_logits = runtime.stage_to_host(grad_logits_t)
        grad_conv_output = runtime.stage_to_host(grad_conv_t)
        grad_pooled = runtime.stage_to_host(grad_pooled_t) if grad_pooled_t is not None else None
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_conv_weight = runtime.stage_to_host(grad_conv_w_t)
        grad_linear_weight = runtime.stage_to_host(grad_linear_w_t)
        grad_linear_bias = runtime.stage_to_host(grad_linear_b_t)
        updated_conv_weight = runtime.stage_to_host(conv_w_t)
        updated_linear_weight = runtime.stage_to_host(linear_w_t)
        updated_linear_bias = runtime.stage_to_host(linear_b_t)
        updated_conv_weight_velocity = runtime.stage_to_host(conv_wv_t)
        updated_linear_weight_velocity = runtime.stage_to_host(linear_wv_t)
        updated_linear_bias_velocity = runtime.stage_to_host(linear_bv_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-conv-linear-training-step')
        return NativeGpuConvLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            conv_output=conv_output,
            grad_logits=grad_logits,
            grad_conv_output=grad_conv_output,
            grad_input=grad_input,
            grad_conv_weight=grad_conv_weight,
            grad_linear_weight=grad_linear_weight,
            grad_linear_bias=grad_linear_bias,
            updated_conv_weight=updated_conv_weight,
            updated_linear_weight=updated_linear_weight,
            updated_linear_bias=updated_linear_bias,
            updated_conv_weight_velocity=updated_conv_weight_velocity,
            updated_linear_weight_velocity=updated_linear_weight_velocity,
            updated_linear_bias_velocity=updated_linear_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
            pooled_output=pooled_output,
            grad_pooled=grad_pooled,
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)
