from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_training_common import (
    _apply_global_grad_clip,
    _load_bound_lib,
    _run_softmax_xent_loss,
)
from minicnn.cuda_native.gpu_training_types import (
    NativeGpuBatchNormLinearTrainingStepResult,
    NativeGpuLinearTrainingStepResult,
    NativeGpuGroupNormLinearTrainingStepResult,
    NativeGpuLayerNorm2dLinearTrainingStepResult,
    NativeGpuPoolLinearTrainingStepResult,
    NativeGpuTwoLinearReluTrainingStepResult,
)


def native_gpu_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    loss_type: str = 'cross_entropy',
    label_smoothing: float = 0.0,
    optimizer_type: str = 'sgd',
    weight_decay: float = 0.0,
    grad_clip_value: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    step_index: int = 1,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
    weight_m: np.ndarray | None = None,
    weight_v: np.ndarray | None = None,
    bias_m: np.ndarray | None = None,
    bias_v: np.ndarray | None = None,
    weight_rmsprop_v: np.ndarray | None = None,
    weight_rmsprop_buf: np.ndarray | None = None,
    bias_rmsprop_v: np.ndarray | None = None,
    bias_rmsprop_buf: np.ndarray | None = None,
    bound_lib: Any | None = None,
    device_runtime: DeviceRuntime | None = None,
    persistent_device_state: bool = False,
    persistent_cache_prefix: str = 'linear',
    return_intermediates: bool = True,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuLinearTrainingStepResult:
    """Run one native GPU Linear training step.

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
    weight_m_f32 = np.zeros_like(weight_f32) if weight_m is None else np.ascontiguousarray(weight_m, dtype=np.float32)
    weight_v_f32 = np.zeros_like(weight_f32) if weight_v is None else np.ascontiguousarray(weight_v, dtype=np.float32)
    bias_m_f32 = np.zeros_like(bias_f32) if bias_m is None else np.ascontiguousarray(bias_m, dtype=np.float32)
    bias_v_f32 = np.zeros_like(bias_f32) if bias_v is None else np.ascontiguousarray(bias_v, dtype=np.float32)
    weight_rmsprop_v_f32 = (
        np.zeros_like(weight_f32)
        if weight_rmsprop_v is None
        else np.ascontiguousarray(weight_rmsprop_v, dtype=np.float32)
    )
    weight_rmsprop_buf_f32 = (
        np.zeros_like(weight_f32)
        if weight_rmsprop_buf is None
        else np.ascontiguousarray(weight_rmsprop_buf, dtype=np.float32)
    )
    bias_rmsprop_v_f32 = (
        np.zeros_like(bias_f32)
        if bias_rmsprop_v is None
        else np.ascontiguousarray(bias_rmsprop_v, dtype=np.float32)
    )
    bias_rmsprop_buf_f32 = (
        np.zeros_like(bias_f32)
        if bias_rmsprop_buf is None
        else np.ascontiguousarray(bias_rmsprop_buf, dtype=np.float32)
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
    normalized_optimizer_type = str(optimizer_type).lower()
    if normalized_optimizer_type not in {'sgd', 'adam', 'adamw', 'rmsprop'}:
        raise ValueError(
            'native_gpu_linear_training_step optimizer_type must be one of sgd, adam, adamw, rmsprop; '
            f'got {optimizer_type!r}'
        )
    if normalized_optimizer_type == 'adam' and float(weight_decay) != 0.0:
        raise ValueError('native_gpu_linear_training_step Adam currently requires weight_decay=0.0; use AdamW for decoupled weight decay.')
    if weight_m_f32.shape != weight_f32.shape or weight_v_f32.shape != weight_f32.shape:
        raise ValueError('native_gpu_linear_training_step expects weight Adam moments with same shape as weight.')
    if bias_m_f32.shape != bias_f32.shape or bias_v_f32.shape != bias_f32.shape:
        raise ValueError('native_gpu_linear_training_step expects bias Adam moments with same shape as bias.')
    if weight_rmsprop_v_f32.shape != weight_f32.shape or weight_rmsprop_buf_f32.shape != weight_f32.shape:
        raise ValueError('native_gpu_linear_training_step expects weight RMSprop state with same shape as weight.')
    if bias_rmsprop_v_f32.shape != bias_f32.shape or bias_rmsprop_buf_f32.shape != bias_f32.shape:
        raise ValueError('native_gpu_linear_training_step expects bias RMSprop state with same shape as bias.')
    normalized_loss_type = str(loss_type)
    if normalized_loss_type not in {'cross_entropy', 'mse', 'bce_with_logits'}:
        raise ValueError(
            'native_gpu_linear_training_step loss_type must be one of '
            f'cross_entropy, mse, bce_with_logits; got {loss_type!r}'
        )
    if normalized_loss_type == 'bce_with_logits':
        if weight_f32.shape[0] != 1:
            raise ValueError('native_gpu_linear_training_step BCEWithLogitsLoss requires out_f=1.')
        if np.any(labels_i32 < 0) or np.any(labels_i32 > 1):
            raise ValueError('native_gpu_linear_training_step BCEWithLogitsLoss labels must be in {0, 1}.')
    elif np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib if bound_lib is not None else (device_runtime.bound_lib if device_runtime is not None else None))
    runtime = device_runtime if device_runtime is not None else DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    runtime.bound_lib = lib
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    n, in_f = int(x_f32.shape[0]), int(x_f32.shape[1])
    out_f = int(weight_f32.shape[0])

    def stage_state(array: np.ndarray, name: str):
        if persistent_device_state:
            return runtime.stage_persistent_to_device(
                array,
                key=f'{persistent_cache_prefix}:{name}',
                name=name,
                update_on_reuse=False,
            )
        return runtime.stage_to_device(array, name=name)

    input_t = runtime.stage_to_device(x_f32, name='input')
    labels_t = runtime.stage_to_device(labels_i32, name='labels')
    weight_t = stage_state(weight_f32, 'weight')
    bias_t = stage_state(bias_f32, 'bias')
    weight_velocity_t = stage_state(weight_velocity_f32, 'weight_velocity')
    bias_velocity_t = stage_state(bias_velocity_f32, 'bias_velocity')
    weight_m_t = stage_state(weight_m_f32, 'weight_m')
    weight_v_t = stage_state(weight_v_f32, 'weight_v')
    bias_m_t = stage_state(bias_m_f32, 'bias_m')
    bias_v_t = stage_state(bias_v_f32, 'bias_v')
    weight_rmsprop_v_t = stage_state(weight_rmsprop_v_f32, 'weight_rmsprop_v')
    weight_rmsprop_buf_t = stage_state(weight_rmsprop_buf_f32, 'weight_rmsprop_buf')
    bias_rmsprop_v_t = stage_state(bias_rmsprop_v_f32, 'bias_rmsprop_v')
    bias_rmsprop_buf_t = stage_state(bias_rmsprop_buf_f32, 'bias_rmsprop_buf')
    logits_t = runtime.allocate((n, out_f), dtype='float32', name='logits')
    probs_t = runtime.allocate((n, out_f), dtype='float32', name='probs')
    grad_logits_t = runtime.allocate((n, out_f), dtype='float32', name='grad_logits')
    grad_input_t = runtime.allocate((n, in_f), dtype='float32', name='grad_input')
    grad_weight_t = runtime.allocate((out_f, in_f), dtype='float32', name='grad_weight')
    grad_bias_t = runtime.allocate((out_f,), dtype='float32', name='grad_bias')
    loss_sum_t = runtime.allocate((1,), dtype='float32', name='loss_sum')
    correct_t = runtime.allocate((1,), dtype='int32', name='correct_count')
    grad_norm_sumsq_t = runtime.allocate((1,), dtype='float32', name='grad_norm_sumsq')

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
        if normalized_loss_type == 'cross_entropy':
            loss_kind = _run_softmax_xent_loss(
                runtime,
                lib,
                logits_t,
                labels_t,
                probs_t,
                grad_logits_t,
                loss_sum_t,
                correct_t,
                n,
                out_f,
                label_smoothing=float(label_smoothing),
            )
        elif normalized_loss_type == 'mse':
            lib.mse_fwd_grad_loss_acc(
                logits_t.device_ptr,
                labels_t.device_ptr,
                grad_logits_t.device_ptr,
                loss_sum_t.device_ptr,
                correct_t.device_ptr,
                n,
                out_f,
            )
            loss_kind = 'gpu_native_train:mse_fwd_grad_loss_acc'
        else:
            lib.bce_fwd_grad_loss_acc(
                logits_t.device_ptr,
                labels_t.device_ptr,
                grad_logits_t.device_ptr,
                loss_sum_t.device_ptr,
                correct_t.device_ptr,
                n,
            )
            loss_kind = 'gpu_native_train:bce_fwd_grad_loss_acc'
        runtime.record_execution(
            loss_kind,
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
        if float(grad_clip_value) > 0.0:
            lib.gpu_memset(grad_norm_sumsq_t.device_ptr, 0, grad_norm_sumsq_t.nbytes)
            lib.grad_l2_sumsq(grad_weight_t.device_ptr, grad_norm_sumsq_t.device_ptr, int(weight_f32.size))
            lib.grad_l2_sumsq(grad_bias_t.device_ptr, grad_norm_sumsq_t.device_ptr, int(bias_f32.size))
            grad_sumsq = float(runtime.stage_to_host(grad_norm_sumsq_t)[0])
            grad_norm = float(np.sqrt(max(grad_sumsq, 0.0)))
            if grad_norm > float(grad_clip_value) and grad_norm > 0.0:
                clip_scale = float(grad_clip_value) / (grad_norm + 1e-12)
                lib.scale_inplace(grad_weight_t.device_ptr, clip_scale, int(weight_f32.size))
                lib.scale_inplace(grad_bias_t.device_ptr, clip_scale, int(bias_f32.size))
            runtime.record_execution(
                'gpu_native_train:grad_clip_global',
                input_name='grad_weight',
                output_name='grad_weight',
                node_count=1,
        )
        if normalized_optimizer_type in {'adam', 'adamw'}:
            if not hasattr(lib, 'adam_update_fused'):
                raise ValueError(
                    'native_gpu_linear_training_step: Adam/AdamW requires adam_update_fused '
                    'in the bound CUDA library, but the symbol is absent.'
                )
            bias_corr1 = 1.0 - float(beta1) ** int(step_index)
            bias_corr2 = 1.0 - float(beta2) ** int(step_index)
            adam_weight_decay = float(weight_decay) if normalized_optimizer_type == 'adamw' else 0.0
            lib.adam_update_fused(
                weight_t.device_ptr,
                grad_weight_t.device_ptr,
                weight_m_t.device_ptr,
                weight_v_t.device_ptr,
                float(lr),
                float(beta1),
                float(beta2),
                float(eps),
                adam_weight_decay,
                0.0,
                1.0,
                float(bias_corr1),
                float(bias_corr2),
                int(weight_f32.size),
            )
            lib.adam_update_fused(
                bias_t.device_ptr,
                grad_bias_t.device_ptr,
                bias_m_t.device_ptr,
                bias_v_t.device_ptr,
                float(lr),
                float(beta1),
                float(beta2),
                float(eps),
                adam_weight_decay,
                0.0,
                1.0,
                float(bias_corr1),
                float(bias_corr2),
                int(bias_f32.size),
            )
            update_kind = 'gpu_native_train:adam_update_fused'
        elif normalized_optimizer_type == 'rmsprop':
            if not hasattr(lib, 'rmsprop_update_fused'):
                raise ValueError(
                    'native_gpu_linear_training_step: RMSprop requires rmsprop_update_fused '
                    'in the bound CUDA library, but the symbol is absent.'
                )
            lib.rmsprop_update_fused(
                weight_t.device_ptr,
                grad_weight_t.device_ptr,
                weight_rmsprop_v_t.device_ptr,
                weight_rmsprop_buf_t.device_ptr,
                float(lr),
                float(rmsprop_alpha),
                float(eps),
                float(momentum),
                float(weight_decay),
                0.0,
                1.0,
                int(weight_f32.size),
            )
            lib.rmsprop_update_fused(
                bias_t.device_ptr,
                grad_bias_t.device_ptr,
                bias_rmsprop_v_t.device_ptr,
                bias_rmsprop_buf_t.device_ptr,
                float(lr),
                float(rmsprop_alpha),
                float(eps),
                float(momentum),
                float(weight_decay),
                0.0,
                1.0,
                int(bias_f32.size),
            )
            update_kind = 'gpu_native_train:rmsprop_update_fused'
        elif float(weight_decay) != 0.0 or float(grad_clip_value) > 0.0:
            lib.sgd_update_fused(
                weight_t.device_ptr,
                grad_weight_t.device_ptr,
                weight_velocity_t.device_ptr,
                float(lr),
                float(momentum),
                float(weight_decay),
                0.0,
                1.0,
                int(weight_f32.size),
            )
            lib.sgd_update_fused(
                bias_t.device_ptr,
                grad_bias_t.device_ptr,
                bias_velocity_t.device_ptr,
                float(lr),
                float(momentum),
                float(weight_decay),
                0.0,
                1.0,
                int(bias_f32.size),
            )
            update_kind = 'gpu_native_train:sgd_update_fused'
        elif float(momentum) != 0.0:
            if not hasattr(lib, 'apply_momentum_update'):
                raise ValueError(
                    'native_gpu_linear_training_step: momentum requires apply_momentum_update '
                    'in the bound CUDA library, but the symbol is absent.'
                )
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

        if return_intermediates:
            logits = runtime.stage_to_host(logits_t)
            probabilities = runtime.stage_to_host(probs_t)
            grad_logits = runtime.stage_to_host(grad_logits_t)
            grad_input = runtime.stage_to_host(grad_input_t)
            grad_weight = runtime.stage_to_host(grad_weight_t)
            grad_bias = runtime.stage_to_host(grad_bias_t)
        else:
            empty = np.empty((0,), dtype=np.float32)
            logits = empty
            probabilities = empty
            grad_logits = empty
            grad_input = empty
            grad_weight = empty
            grad_bias = empty
        updated_weight = runtime.stage_to_host(weight_t)
        updated_bias = runtime.stage_to_host(bias_t)
        copy_sgd_velocity = return_intermediates or (
            normalized_optimizer_type == 'sgd' and float(momentum) != 0.0
        )
        copy_adam_state = return_intermediates or normalized_optimizer_type in {'adam', 'adamw'}
        copy_rmsprop_state = return_intermediates or normalized_optimizer_type == 'rmsprop'
        updated_weight_velocity = runtime.stage_to_host(weight_velocity_t) if copy_sgd_velocity else None
        updated_bias_velocity = runtime.stage_to_host(bias_velocity_t) if copy_sgd_velocity else None
        updated_weight_m = runtime.stage_to_host(weight_m_t) if copy_adam_state else None
        updated_weight_v = runtime.stage_to_host(weight_v_t) if copy_adam_state else None
        updated_bias_m = runtime.stage_to_host(bias_m_t) if copy_adam_state else None
        updated_bias_v = runtime.stage_to_host(bias_v_t) if copy_adam_state else None
        updated_weight_rmsprop_v = runtime.stage_to_host(weight_rmsprop_v_t) if copy_rmsprop_state else None
        updated_weight_rmsprop_buf = runtime.stage_to_host(weight_rmsprop_buf_t) if copy_rmsprop_state else None
        updated_bias_rmsprop_v = runtime.stage_to_host(bias_rmsprop_v_t) if copy_rmsprop_state else None
        updated_bias_rmsprop_buf = runtime.stage_to_host(bias_rmsprop_buf_t) if copy_rmsprop_state else None
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
            updated_weight_m=updated_weight_m if normalized_optimizer_type in {'adam', 'adamw'} else None,
            updated_weight_v=updated_weight_v if normalized_optimizer_type in {'adam', 'adamw'} else None,
            updated_bias_m=updated_bias_m if normalized_optimizer_type in {'adam', 'adamw'} else None,
            updated_bias_v=updated_bias_v if normalized_optimizer_type in {'adam', 'adamw'} else None,
            updated_weight_rmsprop_v=updated_weight_rmsprop_v if normalized_optimizer_type == 'rmsprop' else None,
            updated_weight_rmsprop_buf=updated_weight_rmsprop_buf if normalized_optimizer_type == 'rmsprop' else None,
            updated_bias_rmsprop_v=updated_bias_rmsprop_v if normalized_optimizer_type == 'rmsprop' else None,
            updated_bias_rmsprop_buf=updated_bias_rmsprop_buf if normalized_optimizer_type == 'rmsprop' else None,
        )
    finally:
        for tensor in (
            input_t,
            labels_t,
            weight_t,
            bias_t,
            weight_velocity_t,
            bias_velocity_t,
            weight_m_t,
            weight_v_t,
            bias_m_t,
            bias_v_t,
            weight_rmsprop_v_t,
            weight_rmsprop_buf_t,
            bias_rmsprop_v_t,
            bias_rmsprop_buf_t,
            logits_t,
            probs_t,
            grad_logits_t,
            grad_input_t,
            grad_weight_t,
            grad_bias_t,
            loss_sum_t,
            correct_t,
            grad_norm_sumsq_t,
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
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    weight1_velocity: np.ndarray | None = None,
    bias1_velocity: np.ndarray | None = None,
    weight2_velocity: np.ndarray | None = None,
    bias2_velocity: np.ndarray | None = None,
    activation: str = 'ReLU',
    activation_alpha: float = 0.01,
    bound_lib: Any | None = None,
    return_intermediates: bool = True,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuTwoLinearReluTrainingStepResult:
    """Run one native GPU Linear + activation + Linear + SoftmaxCE + SGD step."""

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
    activation_name = str(activation)
    activation_key = activation_name.lower()
    activation_forward = {
        'relu': ('apply_relu', 'apply_relu'),
        'leakyrelu': ('leaky_relu_forward', 'leaky_relu_forward'),
        'gelu': ('gelu_forward', 'gelu_forward'),
        'silu': ('silu_forward', 'silu_forward'),
        'sigmoid': ('sigmoid_forward', 'sigmoid_forward'),
        'tanh': ('tanh_forward', 'tanh_forward'),
    }
    activation_backward = {
        'relu': ('apply_relu_backward', 'apply_relu_backward'),
        'leakyrelu': ('leaky_relu_backward', 'leaky_relu_backward'),
        'gelu': ('gelu_backward', 'gelu_backward'),
        'silu': ('silu_backward', 'silu_backward'),
        'sigmoid': ('sigmoid_backward', 'sigmoid_backward'),
        'tanh': ('tanh_backward', 'tanh_backward'),
    }
    if activation_key not in activation_forward:
        raise ValueError(f'native_gpu_two_linear_relu_training_step does not support activation={activation_name!r}')

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
    hidden_pre_t = alloc((n, hidden_f), 'hidden_pre')
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
        lib.dense_forward(input_t.device_ptr, w1_t.device_ptr, b1_t.device_ptr, hidden_pre_t.device_ptr, n, in_f, hidden_f)
        runtime.record_execution('gpu_native_train:dense_forward_1', input_name='input', output_name='hidden_pre', node_count=1)
        lib.gpu_memcpy_d2d(hidden_t.device_ptr, hidden_pre_t.device_ptr, hidden_pre_t.nbytes)
        forward_symbol, forward_kind = activation_forward[activation_key]
        if activation_key == 'leakyrelu':
            getattr(lib, forward_symbol)(hidden_t.device_ptr, float(activation_alpha), int(n * hidden_f))
        else:
            getattr(lib, forward_symbol)(hidden_t.device_ptr, int(n * hidden_f))
        runtime.record_execution(f'gpu_native_train:{forward_kind}', input_name='hidden_pre', output_name='hidden', node_count=1)
        lib.dense_forward(hidden_t.device_ptr, w2_t.device_ptr, b2_t.device_ptr, logits_t.device_ptr, n, hidden_f, out_f)
        runtime.record_execution('gpu_native_train:dense_forward_2', input_name='hidden', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(
            runtime,
            lib,
            logits_t,
            labels_t,
            probs_t,
            grad_logits_t,
            loss_sum_t,
            correct_t,
            n,
            out_f,
            label_smoothing=float(label_smoothing),
        )
        runtime.record_execution(
            loss_kind,
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
        backward_symbol, backward_kind = activation_backward[activation_key]
        if activation_key == 'leakyrelu':
            getattr(lib, backward_symbol)(hidden_pre_t.device_ptr, grad_hidden_t.device_ptr, float(activation_alpha), int(n * hidden_f))
        else:
            getattr(lib, backward_symbol)(hidden_pre_t.device_ptr, grad_hidden_t.device_ptr, int(n * hidden_f))
        runtime.record_execution(f'gpu_native_train:{backward_kind}', input_name='hidden_pre', output_name='grad_hidden', node_count=1)
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
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_w1_t, int(w1_f32.size)),
                (grad_b1_t, int(b1_f32.size)),
                (grad_w2_t, int(w2_f32.size)),
                (grad_b2_t, int(b2_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            updates = (
                (w1_t, grad_w1_t, w1v_t, int(w1_f32.size)),
                (b1_t, grad_b1_t, b1v_t, int(b1_f32.size)),
                (w2_t, grad_w2_t, w2v_t, int(w2_f32.size)),
                (b2_t, grad_b2_t, b2v_t, int(b2_f32.size)),
            )
            for value_t, grad_t, velocity_t, size in updates:
                lib.sgd_update_fused(
                    value_t.device_ptr,
                    grad_t.device_ptr,
                    velocity_t.device_ptr,
                    float(lr),
                    float(momentum),
                    float(weight_decay),
                    0.0,
                    1.0,
                    size,
                )
            update_kind = 'gpu_native_train:sgd_update_fused'
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

        if return_intermediates:
            logits = runtime.stage_to_host(logits_t)
            probabilities = runtime.stage_to_host(probs_t)
            grad_logits = runtime.stage_to_host(grad_logits_t)
            grad_hidden = runtime.stage_to_host(grad_hidden_t)
            grad_input = runtime.stage_to_host(grad_input_t)
            grad_weight1 = runtime.stage_to_host(grad_w1_t)
            grad_bias1 = runtime.stage_to_host(grad_b1_t)
            grad_weight2 = runtime.stage_to_host(grad_w2_t)
            grad_bias2 = runtime.stage_to_host(grad_b2_t)
        else:
            empty = np.empty((0,), dtype=np.float32)
            logits = empty
            probabilities = empty
            grad_logits = empty
            grad_hidden = empty
            grad_input = empty
            grad_weight1 = empty
            grad_bias1 = empty
            grad_weight2 = empty
            grad_bias2 = empty
        updated_weight1 = runtime.stage_to_host(w1_t)
        updated_bias1 = runtime.stage_to_host(b1_t)
        updated_weight2 = runtime.stage_to_host(w2_t)
        updated_bias2 = runtime.stage_to_host(b2_t)
        copy_velocity = return_intermediates or float(momentum) != 0.0
        updated_weight1_velocity = runtime.stage_to_host(w1v_t) if copy_velocity else None
        updated_bias1_velocity = runtime.stage_to_host(b1v_t) if copy_velocity else None
        updated_weight2_velocity = runtime.stage_to_host(w2v_t) if copy_velocity else None
        updated_bias2_velocity = runtime.stage_to_host(b2v_t) if copy_velocity else None
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
