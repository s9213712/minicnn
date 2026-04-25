from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_training_common import (
    _apply_global_grad_clip,
    _load_bound_lib,
    _run_softmax_xent_loss,
)
from minicnn.cuda_native.gpu_training_conv import (
    native_gpu_conv_linear_training_step,
    native_gpu_two_conv_relu_pool_linear_training_step,
)
from minicnn.cuda_native.gpu_training_types import (
    NativeGpuBatchNormLinearTrainingStepResult,
    NativeGpuConvLinearTrainingStepResult,
    NativeGpuDepthwiseLayerNorm2dLinearTrainingStepResult,
    NativeGpuDepthwiseLayerNorm2dPointwiseGeluPointwiseLinearTrainingStepResult,
    NativeGpuDepthwiseLayerNorm2dPointwiseLinearTrainingStepResult,
    NativeGpuGroupNormLinearTrainingStepResult,
    NativeGpuLayerNorm2dLinearTrainingStepResult,
    NativeGpuLinearTrainingStepResult,
    NativeGpuPoolLinearTrainingStepResult,
    NativeGpuTwoConvReluPoolLinearTrainingStepResult,
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
    weight_m_t = runtime.stage_to_device(weight_m_f32, name='weight_m')
    weight_v_t = runtime.stage_to_device(weight_v_f32, name='weight_v')
    bias_m_t = runtime.stage_to_device(bias_m_f32, name='bias_m')
    bias_v_t = runtime.stage_to_device(bias_v_f32, name='bias_v')
    weight_rmsprop_v_t = runtime.stage_to_device(weight_rmsprop_v_f32, name='weight_rmsprop_v')
    weight_rmsprop_buf_t = runtime.stage_to_device(weight_rmsprop_buf_f32, name='weight_rmsprop_buf')
    bias_rmsprop_v_t = runtime.stage_to_device(bias_rmsprop_v_f32, name='bias_rmsprop_v')
    bias_rmsprop_buf_t = runtime.stage_to_device(bias_rmsprop_buf_f32, name='bias_rmsprop_buf')
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
        updated_weight_m = runtime.stage_to_host(weight_m_t)
        updated_weight_v = runtime.stage_to_host(weight_v_t)
        updated_bias_m = runtime.stage_to_host(bias_m_t)
        updated_bias_v = runtime.stage_to_host(bias_v_t)
        updated_weight_rmsprop_v = runtime.stage_to_host(weight_rmsprop_v_t)
        updated_weight_rmsprop_buf = runtime.stage_to_host(weight_rmsprop_buf_t)
        updated_bias_rmsprop_v = runtime.stage_to_host(bias_rmsprop_v_t)
        updated_bias_rmsprop_buf = runtime.stage_to_host(bias_rmsprop_buf_t)
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
    bound_lib: Any | None = None,
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
        'gelu': ('gelu_forward', 'gelu_forward'),
        'silu': ('silu_forward', 'silu_forward'),
        'sigmoid': ('sigmoid_forward', 'sigmoid_forward'),
        'tanh': ('tanh_forward', 'tanh_forward'),
    }
    activation_backward = {
        'relu': ('apply_relu_backward', 'apply_relu_backward'),
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
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
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
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_weight_t, int(weight_f32.size)),
                (grad_bias_t, int(bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(weight_t.device_ptr, grad_weight_t.device_ptr, weight_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(weight_f32.size))
            lib.sgd_update_fused(bias_t.device_ptr, grad_bias_t.device_ptr, bias_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(bias_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
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


def native_gpu_avgpool_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuPoolLinearTrainingStepResult:
    """Run one native GPU AvgPool2d(2,2) + Linear + SoftmaxCE + SGD step."""

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
        raise ValueError(f'native_gpu_avgpool_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError('native_gpu_avgpool_linear_training_step requires even H/W for 2x2 stride-2 AvgPool2d.')
    pool_h = height // 2
    pool_w = width // 2
    flat_features = channels * pool_h * pool_w
    if weight_f32.ndim != 2 or weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_avgpool_linear_training_step expects weight with shape (out_f, C*H/2*W/2), '
            f'got weight={weight_f32.shape} for pooled_features={flat_features}'
        )
    if bias_f32.shape != (weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_avgpool_linear_training_step expects bias with shape (out_f,), '
            f'got bias={bias_f32.shape} for weight={weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_avgpool_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_avgpool_linear_training_step labels must be in [0, out_f)')

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
        lib.avgpool2d_forward(input_t.device_ptr, pooled_t.device_ptr, n, channels, height, width, pool_h, pool_w, 2, 2, 2, 2, 0, 0)
        runtime.record_execution('gpu_native_train:avgpool2d_forward', input_name='input', output_name='pooled', node_count=1)
        lib.dense_forward(pooled_t.device_ptr, weight_t.device_ptr, bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pooled', output_name='logits', node_count=1)
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
        lib.avgpool2d_backward(grad_pooled_t.device_ptr, grad_input_t.device_ptr, n, channels, height, width, pool_h, pool_w, 2, 2, 2, 2, 0, 0)
        runtime.record_execution('gpu_native_train:avgpool2d_backward', input_name='grad_pooled', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_weight_t, int(weight_f32.size)),
                (grad_bias_t, int(bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(weight_t.device_ptr, grad_weight_t.device_ptr, weight_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(weight_f32.size))
            lib.sgd_update_fused(bias_t.device_ptr, grad_bias_t.device_ptr, bias_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(bias_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
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
        runtime.synchronize('gpu-native-avgpool-linear-training-step')
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


def native_gpu_layernorm2d_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    norm_eps: float = 1e-6,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuLayerNorm2dLinearTrainingStepResult:
    """Run one native GPU LayerNorm2d + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_layernorm2d_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    flat_features = channels * height * width
    if norm_weight_f32.shape != (channels,) or norm_bias_f32.shape != (channels,):
        raise ValueError('native_gpu_layernorm2d_linear_training_step expects norm weight/bias with shape (C,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_layernorm2d_linear_training_step expects linear_weight with shape (out_f, C*H*W), '
            f'got linear_weight={linear_w_f32.shape} for flattened_features={flat_features}'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_layernorm2d_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_layernorm2d_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_layernorm2d_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_w_f32.shape[0])
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
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    norm_t = alloc((n, channels, height, width), 'norm_output')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_norm_t = alloc((n, channels, height, width), 'grad_norm_output')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_norm_weight_t = alloc((channels,), 'grad_norm_weight')
    grad_norm_bias_t = alloc((channels,), 'grad_norm_bias')
    grad_linear_w_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.layernorm2d_forward(input_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, channels, height, width, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_forward', input_name='input', output_name='norm_output', node_count=1)
        lib.dense_forward(norm_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='norm_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, norm_t.device_ptr, linear_w_t.device_ptr, grad_norm_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.layernorm2d_backward(grad_norm_t.device_ptr, input_t.device_ptr, norm_weight_t.device_ptr, grad_input_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, channels, height, width, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_backward', input_name='grad_norm_output', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_norm_weight', output_name='norm_weight', node_count=1)

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-layernorm2d-linear-training-step')
        return NativeGpuLayerNorm2dLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            norm_output=runtime.stage_to_host(norm_t),
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_norm_output=runtime.stage_to_host(grad_norm_t),
            grad_input=runtime.stage_to_host(grad_input_t),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_linear_weight_velocity=runtime.stage_to_host(linear_wv_t),
            updated_linear_bias_velocity=runtime.stage_to_host(linear_bv_t),
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_groupnorm_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    num_groups: int = 1,
    norm_eps: float = 1e-5,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuGroupNormLinearTrainingStepResult:
    """Run one native GPU GroupNorm + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_groupnorm_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    if int(num_groups) <= 0 or channels % int(num_groups) != 0:
        raise ValueError('native_gpu_groupnorm_linear_training_step requires num_groups to divide input channels.')
    flat_features = channels * height * width
    if norm_weight_f32.shape != (channels,) or norm_bias_f32.shape != (channels,):
        raise ValueError('native_gpu_groupnorm_linear_training_step expects norm weight/bias with shape (C,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_groupnorm_linear_training_step expects linear_weight with shape (out_f, C*H*W), '
            f'got linear_weight={linear_w_f32.shape} for flattened_features={flat_features}'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_groupnorm_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_groupnorm_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_groupnorm_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_w_f32.shape[0])
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
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    norm_t = alloc((n, channels, height, width), 'norm_output')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_norm_t = alloc((n, channels, height, width), 'grad_norm_output')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_norm_weight_t = alloc((channels,), 'grad_norm_weight')
    grad_norm_bias_t = alloc((channels,), 'grad_norm_bias')
    grad_linear_w_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.groupnorm_forward(input_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, channels, height, width, int(num_groups), float(norm_eps))
        runtime.record_execution('gpu_native_train:groupnorm_forward', input_name='input', output_name='norm_output', node_count=1)
        lib.dense_forward(norm_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='norm_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, norm_t.device_ptr, linear_w_t.device_ptr, grad_norm_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.groupnorm_backward(grad_norm_t.device_ptr, input_t.device_ptr, norm_weight_t.device_ptr, grad_input_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, channels, height, width, int(num_groups), float(norm_eps))
        runtime.record_execution('gpu_native_train:groupnorm_backward', input_name='grad_norm_output', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_norm_weight', output_name='norm_weight', node_count=1)

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-groupnorm-linear-training-step')
        return NativeGpuGroupNormLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            norm_output=runtime.stage_to_host(norm_t),
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_norm_output=runtime.stage_to_host(grad_norm_t),
            grad_input=runtime.stage_to_host(grad_input_t),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_linear_weight_velocity=runtime.stage_to_host(linear_wv_t),
            updated_linear_bias_velocity=runtime.stage_to_host(linear_bv_t),
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_batchnorm_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    bn_eps: float = 1e-5,
    bn_momentum: float = 0.1,
    bn_weight_velocity: np.ndarray | None = None,
    bn_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuBatchNormLinearTrainingStepResult:
    """Run one native GPU BatchNorm2d + Linear + SoftmaxCE + SGD step."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    bn_weight_f32 = np.ascontiguousarray(bn_weight, dtype=np.float32)
    bn_bias_f32 = np.ascontiguousarray(bn_bias, dtype=np.float32)
    running_mean_f32 = np.ascontiguousarray(running_mean, dtype=np.float32)
    running_var_f32 = np.ascontiguousarray(running_var, dtype=np.float32)
    linear_weight_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_bias_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    bn_weight_velocity_f32 = (
        np.zeros_like(bn_weight_f32)
        if bn_weight_velocity is None
        else np.ascontiguousarray(bn_weight_velocity, dtype=np.float32)
    )
    bn_bias_velocity_f32 = (
        np.zeros_like(bn_bias_f32)
        if bn_bias_velocity is None
        else np.ascontiguousarray(bn_bias_velocity, dtype=np.float32)
    )
    linear_weight_velocity_f32 = (
        np.zeros_like(linear_weight_f32)
        if linear_weight_velocity is None
        else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    )
    linear_bias_velocity_f32 = (
        np.zeros_like(linear_bias_f32)
        if linear_bias_velocity is None
        else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)
    )
    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_batchnorm_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    flat_features = channels * height * width
    for label, value in (
        ('bn_weight', bn_weight_f32),
        ('bn_bias', bn_bias_f32),
        ('running_mean', running_mean_f32),
        ('running_var', running_var_f32),
    ):
        if value.shape != (channels,):
            raise ValueError(
                f'native_gpu_batchnorm_linear_training_step expects {label} with shape {(channels,)}, got {value.shape}'
            )
    if linear_weight_f32.ndim != 2 or linear_weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_batchnorm_linear_training_step expects linear_weight with shape (out_f, C*H*W), '
            f'got linear_weight={linear_weight_f32.shape} for flat_features={flat_features}'
        )
    if linear_bias_f32.shape != (linear_weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_batchnorm_linear_training_step expects linear_bias with shape (out_f,), '
            f'got linear_bias={linear_bias_f32.shape} for linear_weight={linear_weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_batchnorm_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_weight_f32.shape[0]):
        raise ValueError('native_gpu_batchnorm_linear_training_step labels must be in [0, out_f)')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_weight_f32.shape[0])
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
    bn_weight_t = stage(bn_weight_f32, 'bn_weight')
    bn_bias_t = stage(bn_bias_f32, 'bn_bias')
    running_mean_t = stage(running_mean_f32, 'running_mean')
    running_var_t = stage(running_var_f32, 'running_var')
    linear_weight_t = stage(linear_weight_f32, 'linear_weight')
    linear_bias_t = stage(linear_bias_f32, 'linear_bias')
    bn_weight_velocity_t = stage(bn_weight_velocity_f32, 'bn_weight_velocity')
    bn_bias_velocity_t = stage(bn_bias_velocity_f32, 'bn_bias_velocity')
    linear_weight_velocity_t = stage(linear_weight_velocity_f32, 'linear_weight_velocity')
    linear_bias_velocity_t = stage(linear_bias_velocity_f32, 'linear_bias_velocity')
    bn_output_t = alloc((n, channels, height, width), 'bn_output')
    x_hat_t = alloc((n, channels, height, width), 'x_hat')
    batch_mean_t = alloc((channels,), 'batch_mean')
    batch_inv_std_t = alloc((channels,), 'batch_inv_std')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_bn_output_t = alloc((n, channels, height, width), 'grad_bn_output')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_bn_weight_t = alloc((channels,), 'grad_bn_weight')
    grad_bn_bias_t = alloc((channels,), 'grad_bn_bias')
    grad_linear_weight_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_bias_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.bn_train_forward(
            bn_output_t.device_ptr,
            input_t.device_ptr,
            x_hat_t.device_ptr,
            batch_mean_t.device_ptr,
            batch_inv_std_t.device_ptr,
            running_mean_t.device_ptr,
            running_var_t.device_ptr,
            bn_weight_t.device_ptr,
            bn_bias_t.device_ptr,
            n,
            channels,
            height,
            width,
            float(bn_eps),
            float(bn_momentum),
        )
        runtime.record_execution('gpu_native_train:bn_train_forward', input_name='input', output_name='bn_output', node_count=1)
        lib.dense_forward(bn_output_t.device_ptr, linear_weight_t.device_ptr, linear_bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='bn_output', output_name='logits', node_count=1)
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
            bn_output_t.device_ptr,
            linear_weight_t.device_ptr,
            grad_bn_output_t.device_ptr,
            grad_linear_weight_t.device_ptr,
            grad_linear_bias_t.device_ptr,
            n,
            flat_features,
            out_f,
        )
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.bn_backward(
            grad_input_t.device_ptr,
            grad_bn_weight_t.device_ptr,
            grad_bn_bias_t.device_ptr,
            grad_bn_output_t.device_ptr,
            x_hat_t.device_ptr,
            bn_weight_t.device_ptr,
            batch_inv_std_t.device_ptr,
            n,
            channels,
            height,
            width,
        )
        runtime.record_execution('gpu_native_train:bn_backward', input_name='grad_bn_output', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_bn_weight_t, int(bn_weight_f32.size)),
                (grad_bn_bias_t, int(bn_bias_f32.size)),
                (grad_linear_weight_t, int(linear_weight_f32.size)),
                (grad_linear_bias_t, int(linear_bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            updates = (
                (bn_weight_t, grad_bn_weight_t, bn_weight_velocity_t, int(bn_weight_f32.size)),
                (bn_bias_t, grad_bn_bias_t, bn_bias_velocity_t, int(bn_bias_f32.size)),
                (linear_weight_t, grad_linear_weight_t, linear_weight_velocity_t, int(linear_weight_f32.size)),
                (linear_bias_t, grad_linear_bias_t, linear_bias_velocity_t, int(linear_bias_f32.size)),
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
                (bn_weight_t, grad_bn_weight_t, int(bn_weight_f32.size)),
                (bn_bias_t, grad_bn_bias_t, int(bn_bias_f32.size)),
                (linear_weight_t, grad_linear_weight_t, int(linear_weight_f32.size)),
                (linear_bias_t, grad_linear_bias_t, int(linear_bias_f32.size)),
            )
            for value_t, grad_t, size in updates:
                lib.apply_sgd_update(value_t.device_ptr, grad_t.device_ptr, float(lr), size)
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_bn_weight', output_name='bn_weight', node_count=1)

        logits = runtime.stage_to_host(logits_t)
        probabilities = runtime.stage_to_host(probs_t)
        grad_logits = runtime.stage_to_host(grad_logits_t)
        bn_output = runtime.stage_to_host(bn_output_t)
        x_hat = runtime.stage_to_host(x_hat_t)
        batch_mean = runtime.stage_to_host(batch_mean_t)
        batch_inv_std = runtime.stage_to_host(batch_inv_std_t)
        grad_bn_output = runtime.stage_to_host(grad_bn_output_t)
        grad_input = runtime.stage_to_host(grad_input_t)
        grad_bn_weight = runtime.stage_to_host(grad_bn_weight_t)
        grad_bn_bias = runtime.stage_to_host(grad_bn_bias_t)
        grad_linear_weight = runtime.stage_to_host(grad_linear_weight_t)
        grad_linear_bias = runtime.stage_to_host(grad_linear_bias_t)
        updated_bn_weight = runtime.stage_to_host(bn_weight_t)
        updated_bn_bias = runtime.stage_to_host(bn_bias_t)
        updated_running_mean = runtime.stage_to_host(running_mean_t)
        updated_running_var = runtime.stage_to_host(running_var_t)
        updated_linear_weight = runtime.stage_to_host(linear_weight_t)
        updated_linear_bias = runtime.stage_to_host(linear_bias_t)
        updated_bn_weight_velocity = runtime.stage_to_host(bn_weight_velocity_t)
        updated_bn_bias_velocity = runtime.stage_to_host(bn_bias_velocity_t)
        updated_linear_weight_velocity = runtime.stage_to_host(linear_weight_velocity_t)
        updated_linear_bias_velocity = runtime.stage_to_host(linear_bias_velocity_t)
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-batchnorm-linear-training-step')
        return NativeGpuBatchNormLinearTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            bn_output=bn_output,
            x_hat=x_hat,
            batch_mean=batch_mean,
            batch_inv_std=batch_inv_std,
            grad_bn_output=grad_bn_output,
            grad_input=grad_input,
            grad_bn_weight=grad_bn_weight,
            grad_bn_bias=grad_bn_bias,
            grad_linear_weight=grad_linear_weight,
            grad_linear_bias=grad_linear_bias,
            updated_bn_weight=updated_bn_weight,
            updated_bn_bias=updated_bn_bias,
            updated_running_mean=updated_running_mean,
            updated_running_var=updated_running_var,
            updated_linear_weight=updated_linear_weight,
            updated_linear_bias=updated_linear_bias,
            updated_bn_weight_velocity=updated_bn_weight_velocity,
            updated_bn_bias_velocity=updated_bn_bias_velocity,
            updated_linear_weight_velocity=updated_linear_weight_velocity,
            updated_linear_bias_velocity=updated_linear_bias_velocity,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_global_avgpool_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    weight_velocity: np.ndarray | None = None,
    bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuPoolLinearTrainingStepResult:
    """Run one native GPU GlobalAvgPool2d + Linear + SoftmaxCE + SGD step."""

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
        raise ValueError(f'native_gpu_global_avgpool_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    n, channels, height, width = [int(v) for v in x_f32.shape]
    flat_features = channels
    if weight_f32.ndim != 2 or weight_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_global_avgpool_linear_training_step expects weight with shape (out_f, C), '
            f'got weight={weight_f32.shape} for channels={flat_features}'
        )
    if bias_f32.shape != (weight_f32.shape[0],):
        raise ValueError(
            'native_gpu_global_avgpool_linear_training_step expects bias with shape (out_f,), '
            f'got bias={bias_f32.shape} for weight={weight_f32.shape}'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError(
            'native_gpu_global_avgpool_linear_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= weight_f32.shape[0]):
        raise ValueError('native_gpu_global_avgpool_linear_training_step labels must be in [0, out_f)')

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
    pooled_t = alloc((n, channels, 1, 1), 'pooled')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pooled_t = alloc((n, channels, 1, 1), 'grad_pooled')
    grad_input_t = alloc((n, channels, height, width), 'grad_input')
    grad_weight_t = alloc((out_f, flat_features), 'grad_weight')
    grad_bias_t = alloc((out_f,), 'grad_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.global_avgpool2d_forward(input_t.device_ptr, pooled_t.device_ptr, n, channels, height, width)
        runtime.record_execution('gpu_native_train:global_avgpool2d_forward', input_name='input', output_name='pooled', node_count=1)
        lib.dense_forward(pooled_t.device_ptr, weight_t.device_ptr, bias_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pooled', output_name='logits', node_count=1)
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
        lib.global_avgpool2d_backward(grad_pooled_t.device_ptr, grad_input_t.device_ptr, n, channels, height, width)
        runtime.record_execution('gpu_native_train:global_avgpool2d_backward', input_name='grad_pooled', output_name='grad_input', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_weight_t, int(weight_f32.size)),
                (grad_bias_t, int(bias_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(weight_t.device_ptr, grad_weight_t.device_ptr, weight_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(weight_f32.size))
            lib.sgd_update_fused(bias_t.device_ptr, grad_bias_t.device_ptr, bias_velocity_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(bias_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
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
        runtime.synchronize('gpu-native-global-avgpool-linear-training-step')
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


def native_gpu_depthwise_layernorm2d_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    conv_weight: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    norm_eps: float = 1e-6,
    conv_weight_velocity: np.ndarray | None = None,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuDepthwiseLayerNorm2dLinearTrainingStepResult:
    """Run native GPU DepthwiseConv2d + LayerNorm2d + Linear + SoftmaxCE + SGD."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    conv_w_f32 = np.ascontiguousarray(conv_weight, dtype=np.float32)
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    conv_wv_f32 = np.zeros_like(conv_w_f32) if conv_weight_velocity is None else np.ascontiguousarray(conv_weight_velocity, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)

    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_depthwise_layernorm2d_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    if conv_w_f32.ndim != 4 or conv_w_f32.shape[1] != 1:
        raise ValueError('native_gpu_depthwise_layernorm2d_linear_training_step expects conv_weight shape (out_c, 1, kh, kw).')
    n, in_c, height, width = [int(v) for v in x_f32.shape]
    out_c, _conv_in_c, kh, kw = [int(v) for v in conv_w_f32.shape]
    if out_c % in_c != 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_linear_training_step requires out_c to be a multiple of input channels.')
    out_h = height - kh + 1
    out_w = width - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_linear_training_step requires valid DepthwiseConv2d output dimensions.')
    flat_features = out_c * out_h * out_w
    if norm_weight_f32.shape != (out_c,) or norm_bias_f32.shape != (out_c,):
        raise ValueError('native_gpu_depthwise_layernorm2d_linear_training_step expects norm weight/bias with shape (out_c,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_depthwise_layernorm2d_linear_training_step expects linear_weight with shape '
            f'(out_f, {flat_features}), got {linear_w_f32.shape}.'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_depthwise_layernorm2d_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_depthwise_layernorm2d_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_depthwise_layernorm2d_linear_training_step labels must be in [0, out_f).')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_w_f32.shape[0])
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
    depthwise_bias_t = stage(np.zeros((out_c,), dtype=np.float32), 'depthwise_bias')
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    conv_wv_t = stage(conv_wv_f32, 'conv_weight_velocity')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    conv_t = alloc((n, out_c, out_h, out_w), 'conv_output')
    norm_t = alloc((n, out_c, out_h, out_w), 'norm_output')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_norm_t = alloc((n, out_c, out_h, out_w), 'grad_norm_output')
    grad_conv_t = alloc((n, out_c, out_h, out_w), 'grad_conv_output')
    grad_input_t = alloc((n, in_c, height, width), 'grad_input')
    grad_conv_w_t = alloc(tuple(int(v) for v in conv_w_f32.shape), 'grad_conv_weight')
    grad_depthwise_bias_t = alloc((out_c,), 'grad_depthwise_bias')
    grad_norm_weight_t = alloc((out_c,), 'grad_norm_weight')
    grad_norm_bias_t = alloc((out_c,), 'grad_norm_bias')
    grad_linear_w_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.depthwise_conv2d_forward(
            input_t.device_ptr,
            conv_w_t.device_ptr,
            depthwise_bias_t.device_ptr,
            conv_t.device_ptr,
            n,
            in_c,
            height,
            width,
            out_c,
            kh,
            kw,
            out_h,
            out_w,
            1,
            1,
            0,
            0,
            0,
        )
        runtime.record_execution('gpu_native_train:depthwise_conv2d_forward', input_name='input', output_name='conv_output', node_count=1)
        lib.layernorm2d_forward(conv_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_forward', input_name='conv_output', output_name='norm_output', node_count=1)
        lib.dense_forward(norm_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='norm_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, norm_t.device_ptr, linear_w_t.device_ptr, grad_norm_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.layernorm2d_backward(grad_norm_t.device_ptr, conv_t.device_ptr, norm_weight_t.device_ptr, grad_conv_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_backward', input_name='grad_norm_output', output_name='grad_conv_output', node_count=1)
        lib.depthwise_conv2d_backward(
            grad_conv_t.device_ptr,
            input_t.device_ptr,
            conv_w_t.device_ptr,
            grad_input_t.device_ptr,
            grad_conv_w_t.device_ptr,
            grad_depthwise_bias_t.device_ptr,
            n,
            in_c,
            height,
            width,
            out_c,
            kh,
            kw,
            out_h,
            out_w,
            1,
            1,
            0,
            0,
            0,
        )
        runtime.record_execution('gpu_native_train:depthwise_conv2d_backward', input_name='grad_conv_output', output_name='grad_conv_weight', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_conv_w_t, int(conv_w_f32.size)),
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(conv_w_t.device_ptr, grad_conv_w_t.device_ptr, conv_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(conv_w_f32.size))
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(conv_w_t.device_ptr, grad_conv_w_t.device_ptr, float(lr), int(conv_w_f32.size))
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_conv_weight', output_name='conv_weight', node_count=1)

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-depthwise-layernorm2d-linear-training-step')
        return NativeGpuDepthwiseLayerNorm2dLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            conv_output=runtime.stage_to_host(conv_t),
            norm_output=runtime.stage_to_host(norm_t),
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_norm_output=runtime.stage_to_host(grad_norm_t),
            grad_conv_output=runtime.stage_to_host(grad_conv_t),
            grad_input=runtime.stage_to_host(grad_input_t),
            grad_conv_weight=runtime.stage_to_host(grad_conv_w_t),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_conv_weight=runtime.stage_to_host(conv_w_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_conv_weight_velocity=runtime.stage_to_host(conv_wv_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_linear_weight_velocity=runtime.stage_to_host(linear_wv_t),
            updated_linear_bias_velocity=runtime.stage_to_host(linear_bv_t),
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_depthwise_layernorm2d_pointwise_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    depthwise_weight: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    pointwise_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    norm_eps: float = 1e-6,
    depthwise_weight_velocity: np.ndarray | None = None,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    pointwise_weight_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuDepthwiseLayerNorm2dPointwiseLinearTrainingStepResult:
    """Run native GPU DepthwiseConv2d + LayerNorm2d + PointwiseConv2d + Linear."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    depthwise_w_f32 = np.ascontiguousarray(depthwise_weight, dtype=np.float32)
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    pointwise_w_f32 = np.ascontiguousarray(pointwise_weight, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    depthwise_wv_f32 = np.zeros_like(depthwise_w_f32) if depthwise_weight_velocity is None else np.ascontiguousarray(depthwise_weight_velocity, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    pointwise_wv_f32 = np.zeros_like(pointwise_w_f32) if pointwise_weight_velocity is None else np.ascontiguousarray(pointwise_weight_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)

    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_depthwise_layernorm2d_pointwise_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    if depthwise_w_f32.ndim != 4 or depthwise_w_f32.shape[1] != 1:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step expects depthwise_weight shape (out_c, 1, kh, kw).')
    if pointwise_w_f32.ndim != 4 or pointwise_w_f32.shape[2:] != (1, 1):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step expects pointwise_weight shape (out_c, in_c, 1, 1).')
    n, in_c, height, width = [int(v) for v in x_f32.shape]
    depthwise_out_c, _depthwise_in_c, kh, kw = [int(v) for v in depthwise_w_f32.shape]
    pointwise_out_c, pointwise_in_c, _p_kh, _p_kw = [int(v) for v in pointwise_w_f32.shape]
    if depthwise_out_c % in_c != 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step requires depthwise out_c to be a multiple of input channels.')
    if pointwise_in_c != depthwise_out_c:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step requires pointwise in_c to match depthwise out_c.')
    out_h = height - kh + 1
    out_w = width - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step requires valid DepthwiseConv2d output dimensions.')
    flat_features = pointwise_out_c * out_h * out_w
    if norm_weight_f32.shape != (depthwise_out_c,) or norm_bias_f32.shape != (depthwise_out_c,):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step expects norm weight/bias with shape (depthwise_out_c,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_depthwise_layernorm2d_pointwise_linear_training_step expects linear_weight with shape '
            f'(out_f, {flat_features}), got {linear_w_f32.shape}.'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_linear_training_step labels must be in [0, out_f).')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_w_f32.shape[0])
    pointwise_patch_size = depthwise_out_c
    pointwise_spatial_size = n * out_h * out_w
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
    depthwise_w_t = stage(depthwise_w_f32, 'depthwise_weight')
    depthwise_bias_t = stage(np.zeros((depthwise_out_c,), dtype=np.float32), 'depthwise_bias')
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    pointwise_w_t = stage(pointwise_w_f32, 'pointwise_weight')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    depthwise_wv_t = stage(depthwise_wv_f32, 'depthwise_weight_velocity')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    pointwise_wv_t = stage(pointwise_wv_f32, 'pointwise_weight_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    depthwise_t = alloc((n, depthwise_out_c, out_h, out_w), 'depthwise_output')
    norm_t = alloc((n, depthwise_out_c, out_h, out_w), 'norm_output')
    pointwise_col_t = alloc((pointwise_patch_size, pointwise_spatial_size), 'pointwise_col')
    pointwise_raw_t = alloc((pointwise_out_c, n, out_h, out_w), 'pointwise_raw_cnhw')
    pointwise_t = alloc((n, pointwise_out_c, out_h, out_w), 'pointwise_output')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pointwise_t = alloc((n, pointwise_out_c, out_h, out_w), 'grad_pointwise_output')
    grad_pointwise_cnhw_t = alloc((pointwise_out_c, n, out_h, out_w), 'grad_pointwise_cnhw')
    grad_norm_t = alloc((n, depthwise_out_c, out_h, out_w), 'grad_norm_output')
    grad_depthwise_t = alloc((n, depthwise_out_c, out_h, out_w), 'grad_depthwise_output')
    grad_input_t = alloc((n, in_c, height, width), 'grad_input')
    grad_depthwise_w_t = alloc(tuple(int(v) for v in depthwise_w_f32.shape), 'grad_depthwise_weight')
    grad_depthwise_bias_t = alloc((depthwise_out_c,), 'grad_depthwise_bias')
    grad_norm_weight_t = alloc((depthwise_out_c,), 'grad_norm_weight')
    grad_norm_bias_t = alloc((depthwise_out_c,), 'grad_norm_bias')
    grad_pointwise_w_t = alloc(tuple(int(v) for v in pointwise_w_f32.shape), 'grad_pointwise_weight')
    grad_linear_w_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.depthwise_conv2d_forward(
            input_t.device_ptr,
            depthwise_w_t.device_ptr,
            depthwise_bias_t.device_ptr,
            depthwise_t.device_ptr,
            n,
            in_c,
            height,
            width,
            depthwise_out_c,
            kh,
            kw,
            out_h,
            out_w,
            1,
            1,
            0,
            0,
            0,
        )
        runtime.record_execution('gpu_native_train:depthwise_conv2d_forward', input_name='input', output_name='depthwise_output', node_count=1)
        lib.layernorm2d_forward(depthwise_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, depthwise_out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_forward', input_name='depthwise_output', output_name='norm_output', node_count=1)
        lib.im2col_forward(norm_t.device_ptr, pointwise_col_t.device_ptr, n, depthwise_out_c, out_h, out_w, 1, 1, out_h, out_w)
        lib.gemm_forward(pointwise_w_t.device_ptr, pointwise_col_t.device_ptr, pointwise_raw_t.device_ptr, pointwise_out_c, pointwise_spatial_size, pointwise_patch_size)
        lib.cnhw_to_nchw(pointwise_raw_t.device_ptr, pointwise_t.device_ptr, n, pointwise_out_c, out_h, out_w)
        runtime.record_execution('gpu_native_train:pointwise_conv2d_im2col_gemm', input_name='norm_output', output_name='pointwise_output', node_count=1)
        lib.dense_forward(pointwise_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pointwise_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, pointwise_t.device_ptr, linear_w_t.device_ptr, grad_pointwise_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.nchw_to_cnhw(grad_pointwise_t.device_ptr, grad_pointwise_cnhw_t.device_ptr, n, pointwise_out_c, out_h, out_w)
        lib.conv_backward(
            grad_pointwise_cnhw_t.device_ptr,
            norm_t.device_ptr,
            pointwise_w_t.device_ptr,
            grad_pointwise_w_t.device_ptr,
            grad_norm_t.device_ptr,
            n,
            depthwise_out_c,
            out_h,
            out_w,
            1,
            1,
            out_h,
            out_w,
            pointwise_out_c,
        )
        runtime.record_execution('gpu_native_train:pointwise_conv2d_backward', input_name='grad_pointwise_output', output_name='grad_pointwise_weight', node_count=1)
        lib.layernorm2d_backward(grad_norm_t.device_ptr, depthwise_t.device_ptr, norm_weight_t.device_ptr, grad_depthwise_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, depthwise_out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_backward', input_name='grad_norm_output', output_name='grad_depthwise_output', node_count=1)
        lib.depthwise_conv2d_backward(
            grad_depthwise_t.device_ptr,
            input_t.device_ptr,
            depthwise_w_t.device_ptr,
            grad_input_t.device_ptr,
            grad_depthwise_w_t.device_ptr,
            grad_depthwise_bias_t.device_ptr,
            n,
            in_c,
            height,
            width,
            depthwise_out_c,
            kh,
            kw,
            out_h,
            out_w,
            1,
            1,
            0,
            0,
            0,
        )
        runtime.record_execution('gpu_native_train:depthwise_conv2d_backward', input_name='grad_depthwise_output', output_name='grad_depthwise_weight', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_depthwise_w_t, int(depthwise_w_f32.size)),
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_pointwise_w_t, int(pointwise_w_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(depthwise_w_t.device_ptr, grad_depthwise_w_t.device_ptr, depthwise_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(depthwise_w_f32.size))
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(pointwise_w_t.device_ptr, grad_pointwise_w_t.device_ptr, pointwise_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(pointwise_w_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(depthwise_w_t.device_ptr, grad_depthwise_w_t.device_ptr, float(lr), int(depthwise_w_f32.size))
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(pointwise_w_t.device_ptr, grad_pointwise_w_t.device_ptr, float(lr), int(pointwise_w_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_depthwise_weight', output_name='depthwise_weight', node_count=1)

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-depthwise-layernorm2d-pointwise-linear-training-step')
        return NativeGpuDepthwiseLayerNorm2dPointwiseLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            depthwise_output=runtime.stage_to_host(depthwise_t),
            norm_output=runtime.stage_to_host(norm_t),
            pointwise_output=runtime.stage_to_host(pointwise_t),
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_pointwise_output=runtime.stage_to_host(grad_pointwise_t),
            grad_norm_output=runtime.stage_to_host(grad_norm_t),
            grad_depthwise_output=runtime.stage_to_host(grad_depthwise_t),
            grad_input=runtime.stage_to_host(grad_input_t),
            grad_depthwise_weight=runtime.stage_to_host(grad_depthwise_w_t),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_pointwise_weight=runtime.stage_to_host(grad_pointwise_w_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_depthwise_weight=runtime.stage_to_host(depthwise_w_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_pointwise_weight=runtime.stage_to_host(pointwise_w_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_depthwise_weight_velocity=runtime.stage_to_host(depthwise_wv_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_pointwise_weight_velocity=runtime.stage_to_host(pointwise_wv_t),
            updated_linear_weight_velocity=runtime.stage_to_host(linear_wv_t),
            updated_linear_bias_velocity=runtime.stage_to_host(linear_bv_t),
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)


def native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    depthwise_weight: np.ndarray,
    norm_weight: np.ndarray,
    norm_bias: np.ndarray,
    pointwise1_weight: np.ndarray,
    pointwise2_weight: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: np.ndarray,
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    norm_eps: float = 1e-6,
    depthwise_weight_velocity: np.ndarray | None = None,
    norm_weight_velocity: np.ndarray | None = None,
    norm_bias_velocity: np.ndarray | None = None,
    pointwise1_weight_velocity: np.ndarray | None = None,
    pointwise2_weight_velocity: np.ndarray | None = None,
    linear_weight_velocity: np.ndarray | None = None,
    linear_bias_velocity: np.ndarray | None = None,
    bound_lib: Any | None = None,
    reserve_bytes: int = 0,
    reserve_buffers: int = 0,
) -> NativeGpuDepthwiseLayerNorm2dPointwiseGeluPointwiseLinearTrainingStepResult:
    """Run native GPU DepthwiseConv2d + LayerNorm2d + PointwiseConv2d + GELU + PointwiseConv2d + Linear."""

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    depthwise_w_f32 = np.ascontiguousarray(depthwise_weight, dtype=np.float32)
    norm_weight_f32 = np.ascontiguousarray(norm_weight, dtype=np.float32)
    norm_bias_f32 = np.ascontiguousarray(norm_bias, dtype=np.float32)
    pointwise1_w_f32 = np.ascontiguousarray(pointwise1_weight, dtype=np.float32)
    pointwise2_w_f32 = np.ascontiguousarray(pointwise2_weight, dtype=np.float32)
    linear_w_f32 = np.ascontiguousarray(linear_weight, dtype=np.float32)
    linear_b_f32 = np.ascontiguousarray(linear_bias, dtype=np.float32)
    depthwise_wv_f32 = np.zeros_like(depthwise_w_f32) if depthwise_weight_velocity is None else np.ascontiguousarray(depthwise_weight_velocity, dtype=np.float32)
    norm_wv_f32 = np.zeros_like(norm_weight_f32) if norm_weight_velocity is None else np.ascontiguousarray(norm_weight_velocity, dtype=np.float32)
    norm_bv_f32 = np.zeros_like(norm_bias_f32) if norm_bias_velocity is None else np.ascontiguousarray(norm_bias_velocity, dtype=np.float32)
    pointwise1_wv_f32 = np.zeros_like(pointwise1_w_f32) if pointwise1_weight_velocity is None else np.ascontiguousarray(pointwise1_weight_velocity, dtype=np.float32)
    pointwise2_wv_f32 = np.zeros_like(pointwise2_w_f32) if pointwise2_weight_velocity is None else np.ascontiguousarray(pointwise2_weight_velocity, dtype=np.float32)
    linear_wv_f32 = np.zeros_like(linear_w_f32) if linear_weight_velocity is None else np.ascontiguousarray(linear_weight_velocity, dtype=np.float32)
    linear_bv_f32 = np.zeros_like(linear_b_f32) if linear_bias_velocity is None else np.ascontiguousarray(linear_bias_velocity, dtype=np.float32)

    if x_f32.ndim != 4:
        raise ValueError(f'native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects x with shape (N, C, H, W), got {x_f32.shape}')
    if depthwise_w_f32.ndim != 4 or depthwise_w_f32.shape[1] != 1:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects depthwise_weight shape (out_c, 1, kh, kw).')
    if pointwise1_w_f32.ndim != 4 or pointwise1_w_f32.shape[2:] != (1, 1):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects pointwise1_weight shape (out_c, in_c, 1, 1).')
    if pointwise2_w_f32.ndim != 4 or pointwise2_w_f32.shape[2:] != (1, 1):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects pointwise2_weight shape (out_c, in_c, 1, 1).')
    n, in_c, height, width = [int(v) for v in x_f32.shape]
    depthwise_out_c, _depthwise_in_c, kh, kw = [int(v) for v in depthwise_w_f32.shape]
    hidden_c, pointwise1_in_c, _p1_kh, _p1_kw = [int(v) for v in pointwise1_w_f32.shape]
    pointwise_out_c, pointwise2_in_c, _p2_kh, _p2_kw = [int(v) for v in pointwise2_w_f32.shape]
    if depthwise_out_c % in_c != 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires depthwise out_c to be a multiple of input channels.')
    if pointwise1_in_c != depthwise_out_c:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires pointwise1 in_c to match depthwise out_c.')
    if pointwise2_in_c != hidden_c:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires pointwise2 in_c to match pointwise1 out_c.')
    out_h = height - kh + 1
    out_w = width - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step requires valid DepthwiseConv2d output dimensions.')
    flat_features = pointwise_out_c * out_h * out_w
    if norm_weight_f32.shape != (depthwise_out_c,) or norm_bias_f32.shape != (depthwise_out_c,):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects norm weight/bias with shape (depthwise_out_c,).')
    if linear_w_f32.ndim != 2 or linear_w_f32.shape[1] != flat_features:
        raise ValueError(
            'native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects linear_weight with shape '
            f'(out_f, {flat_features}), got {linear_w_f32.shape}.'
        )
    if linear_b_f32.shape != (linear_w_f32.shape[0],):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects linear_bias with shape (out_f,).')
    if labels_i32.ndim != 1 or labels_i32.shape[0] != n:
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step expects labels with shape (N,).')
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= linear_w_f32.shape[0]):
        raise ValueError('native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step labels must be in [0, out_f).')

    lib = _load_bound_lib(bound_lib)
    runtime = DeviceRuntime(execution_mode='gpu_native', tensor_execution_device='gpu', bound_lib=lib)
    if reserve_bytes > 0 or reserve_buffers > 0:
        runtime.reserve_from_planner(total_bytes=int(reserve_bytes), num_buffers=int(reserve_buffers))

    out_f = int(linear_w_f32.shape[0])
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
    depthwise_w_t = stage(depthwise_w_f32, 'depthwise_weight')
    depthwise_bias_t = stage(np.zeros((depthwise_out_c,), dtype=np.float32), 'depthwise_bias')
    norm_weight_t = stage(norm_weight_f32, 'norm_weight')
    norm_bias_t = stage(norm_bias_f32, 'norm_bias')
    pointwise1_w_t = stage(pointwise1_w_f32, 'pointwise1_weight')
    pointwise2_w_t = stage(pointwise2_w_f32, 'pointwise2_weight')
    linear_w_t = stage(linear_w_f32, 'linear_weight')
    linear_b_t = stage(linear_b_f32, 'linear_bias')
    depthwise_wv_t = stage(depthwise_wv_f32, 'depthwise_weight_velocity')
    norm_wv_t = stage(norm_wv_f32, 'norm_weight_velocity')
    norm_bv_t = stage(norm_bv_f32, 'norm_bias_velocity')
    pointwise1_wv_t = stage(pointwise1_wv_f32, 'pointwise1_weight_velocity')
    pointwise2_wv_t = stage(pointwise2_wv_f32, 'pointwise2_weight_velocity')
    linear_wv_t = stage(linear_wv_f32, 'linear_weight_velocity')
    linear_bv_t = stage(linear_bv_f32, 'linear_bias_velocity')
    depthwise_t = alloc((n, depthwise_out_c, out_h, out_w), 'depthwise_output')
    norm_t = alloc((n, depthwise_out_c, out_h, out_w), 'norm_output')
    pw1_col_t = alloc((depthwise_out_c, spatial_size), 'pointwise1_col')
    pw1_raw_t = alloc((hidden_c, n, out_h, out_w), 'pointwise1_raw_cnhw')
    pw1_t = alloc((n, hidden_c, out_h, out_w), 'pointwise1_output')
    activation_t = alloc((n, hidden_c, out_h, out_w), 'activation_output')
    pw2_col_t = alloc((hidden_c, spatial_size), 'pointwise2_col')
    pw2_raw_t = alloc((pointwise_out_c, n, out_h, out_w), 'pointwise2_raw_cnhw')
    pw2_t = alloc((n, pointwise_out_c, out_h, out_w), 'pointwise2_output')
    logits_t = alloc((n, out_f), 'logits')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    grad_pw2_t = alloc((n, pointwise_out_c, out_h, out_w), 'grad_pointwise2_output')
    grad_pw2_cnhw_t = alloc((pointwise_out_c, n, out_h, out_w), 'grad_pointwise2_cnhw')
    grad_activation_t = alloc((n, hidden_c, out_h, out_w), 'grad_activation_output')
    grad_activation_before_gelu_t = alloc((n, hidden_c, out_h, out_w), 'grad_activation_before_gelu')
    grad_activation_cnhw_t = alloc((hidden_c, n, out_h, out_w), 'grad_activation_cnhw')
    grad_pw1_t = alloc((n, hidden_c, out_h, out_w), 'grad_pointwise1_output')
    grad_norm_t = alloc((n, depthwise_out_c, out_h, out_w), 'grad_norm_output')
    grad_depthwise_t = alloc((n, depthwise_out_c, out_h, out_w), 'grad_depthwise_output')
    grad_input_t = alloc((n, in_c, height, width), 'grad_input')
    grad_depthwise_w_t = alloc(tuple(int(v) for v in depthwise_w_f32.shape), 'grad_depthwise_weight')
    grad_depthwise_bias_t = alloc((depthwise_out_c,), 'grad_depthwise_bias')
    grad_norm_weight_t = alloc((depthwise_out_c,), 'grad_norm_weight')
    grad_norm_bias_t = alloc((depthwise_out_c,), 'grad_norm_bias')
    grad_pw1_w_t = alloc(tuple(int(v) for v in pointwise1_w_f32.shape), 'grad_pointwise1_weight')
    grad_pw2_w_t = alloc(tuple(int(v) for v in pointwise2_w_f32.shape), 'grad_pointwise2_weight')
    grad_linear_w_t = alloc((out_f, flat_features), 'grad_linear_weight')
    grad_linear_b_t = alloc((out_f,), 'grad_linear_bias')
    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        lib.depthwise_conv2d_forward(
            input_t.device_ptr,
            depthwise_w_t.device_ptr,
            depthwise_bias_t.device_ptr,
            depthwise_t.device_ptr,
            n,
            in_c,
            height,
            width,
            depthwise_out_c,
            kh,
            kw,
            out_h,
            out_w,
            1,
            1,
            0,
            0,
            0,
        )
        runtime.record_execution('gpu_native_train:depthwise_conv2d_forward', input_name='input', output_name='depthwise_output', node_count=1)
        lib.layernorm2d_forward(depthwise_t.device_ptr, norm_weight_t.device_ptr, norm_bias_t.device_ptr, norm_t.device_ptr, n, depthwise_out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_forward', input_name='depthwise_output', output_name='norm_output', node_count=1)
        lib.im2col_forward(norm_t.device_ptr, pw1_col_t.device_ptr, n, depthwise_out_c, out_h, out_w, 1, 1, out_h, out_w)
        lib.gemm_forward(pointwise1_w_t.device_ptr, pw1_col_t.device_ptr, pw1_raw_t.device_ptr, hidden_c, spatial_size, depthwise_out_c)
        lib.cnhw_to_nchw(pw1_raw_t.device_ptr, pw1_t.device_ptr, n, hidden_c, out_h, out_w)
        runtime.record_execution('gpu_native_train:pointwise1_conv2d_im2col_gemm', input_name='norm_output', output_name='pointwise1_output', node_count=1)
        lib.gpu_memcpy_d2d(activation_t.device_ptr, pw1_t.device_ptr, activation_t.nbytes)
        lib.gelu_forward(activation_t.device_ptr, int(n * hidden_c * out_h * out_w))
        runtime.record_execution('gpu_native_train:gelu_forward', input_name='pointwise1_output', output_name='activation_output', node_count=1)
        lib.im2col_forward(activation_t.device_ptr, pw2_col_t.device_ptr, n, hidden_c, out_h, out_w, 1, 1, out_h, out_w)
        lib.gemm_forward(pointwise2_w_t.device_ptr, pw2_col_t.device_ptr, pw2_raw_t.device_ptr, pointwise_out_c, spatial_size, hidden_c)
        lib.cnhw_to_nchw(pw2_raw_t.device_ptr, pw2_t.device_ptr, n, pointwise_out_c, out_h, out_w)
        runtime.record_execution('gpu_native_train:pointwise2_conv2d_im2col_gemm', input_name='activation_output', output_name='pointwise2_output', node_count=1)
        lib.dense_forward(pw2_t.device_ptr, linear_w_t.device_ptr, linear_b_t.device_ptr, logits_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_forward', input_name='pointwise2_output', output_name='logits', node_count=1)
        loss_kind = _run_softmax_xent_loss(runtime, lib, logits_t, labels_t, probs_t, grad_logits_t, loss_sum_t, correct_t, n, out_f, label_smoothing=float(label_smoothing))
        runtime.record_execution(loss_kind, input_name='logits', output_name='grad_logits', node_count=1)
        lib.dense_backward_full(grad_logits_t.device_ptr, pw2_t.device_ptr, linear_w_t.device_ptr, grad_pw2_t.device_ptr, grad_linear_w_t.device_ptr, grad_linear_b_t.device_ptr, n, flat_features, out_f)
        runtime.record_execution('gpu_native_train:dense_backward_full', input_name='grad_logits', output_name='grad_linear_weight', node_count=1)
        lib.nchw_to_cnhw(grad_pw2_t.device_ptr, grad_pw2_cnhw_t.device_ptr, n, pointwise_out_c, out_h, out_w)
        lib.conv_backward(
            grad_pw2_cnhw_t.device_ptr,
            activation_t.device_ptr,
            pointwise2_w_t.device_ptr,
            grad_pw2_w_t.device_ptr,
            grad_activation_t.device_ptr,
            n,
            hidden_c,
            out_h,
            out_w,
            1,
            1,
            out_h,
            out_w,
            pointwise_out_c,
        )
        runtime.record_execution('gpu_native_train:pointwise2_conv2d_backward', input_name='grad_pointwise2_output', output_name='grad_pointwise2_weight', node_count=1)
        lib.gpu_memcpy_d2d(grad_activation_before_gelu_t.device_ptr, grad_activation_t.device_ptr, grad_activation_before_gelu_t.nbytes)
        lib.gelu_backward(pw1_t.device_ptr, grad_activation_t.device_ptr, int(n * hidden_c * out_h * out_w))
        runtime.record_execution('gpu_native_train:gelu_backward', input_name='pointwise1_output', output_name='grad_pointwise1_output', node_count=1)
        lib.nchw_to_cnhw(grad_activation_t.device_ptr, grad_activation_cnhw_t.device_ptr, n, hidden_c, out_h, out_w)
        lib.conv_backward(
            grad_activation_cnhw_t.device_ptr,
            norm_t.device_ptr,
            pointwise1_w_t.device_ptr,
            grad_pw1_w_t.device_ptr,
            grad_norm_t.device_ptr,
            n,
            depthwise_out_c,
            out_h,
            out_w,
            1,
            1,
            out_h,
            out_w,
            hidden_c,
        )
        runtime.record_execution('gpu_native_train:pointwise1_conv2d_backward', input_name='grad_pointwise1_output', output_name='grad_pointwise1_weight', node_count=1)
        lib.gpu_memcpy_d2d(grad_pw1_t.device_ptr, grad_activation_t.device_ptr, grad_pw1_t.nbytes)
        lib.layernorm2d_backward(grad_norm_t.device_ptr, depthwise_t.device_ptr, norm_weight_t.device_ptr, grad_depthwise_t.device_ptr, grad_norm_weight_t.device_ptr, grad_norm_bias_t.device_ptr, n, depthwise_out_c, out_h, out_w, float(norm_eps))
        runtime.record_execution('gpu_native_train:layernorm2d_backward', input_name='grad_norm_output', output_name='grad_depthwise_output', node_count=1)
        lib.depthwise_conv2d_backward(
            grad_depthwise_t.device_ptr,
            input_t.device_ptr,
            depthwise_w_t.device_ptr,
            grad_input_t.device_ptr,
            grad_depthwise_w_t.device_ptr,
            grad_depthwise_bias_t.device_ptr,
            n,
            in_c,
            height,
            width,
            depthwise_out_c,
            kh,
            kw,
            out_h,
            out_w,
            1,
            1,
            0,
            0,
            0,
        )
        runtime.record_execution('gpu_native_train:depthwise_conv2d_backward', input_name='grad_depthwise_output', output_name='grad_depthwise_weight', node_count=1)
        _apply_global_grad_clip(
            runtime,
            lib,
            (
                (grad_depthwise_w_t, int(depthwise_w_f32.size)),
                (grad_norm_weight_t, int(norm_weight_f32.size)),
                (grad_norm_bias_t, int(norm_bias_f32.size)),
                (grad_pw1_w_t, int(pointwise1_w_f32.size)),
                (grad_pw2_w_t, int(pointwise2_w_f32.size)),
                (grad_linear_w_t, int(linear_w_f32.size)),
                (grad_linear_b_t, int(linear_b_f32.size)),
            ),
            float(grad_clip_value),
        )
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            lib.sgd_update_fused(depthwise_w_t.device_ptr, grad_depthwise_w_t.device_ptr, depthwise_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(depthwise_w_f32.size))
            lib.sgd_update_fused(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, norm_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_weight_f32.size))
            lib.sgd_update_fused(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, norm_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(norm_bias_f32.size))
            lib.sgd_update_fused(pointwise1_w_t.device_ptr, grad_pw1_w_t.device_ptr, pointwise1_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(pointwise1_w_f32.size))
            lib.sgd_update_fused(pointwise2_w_t.device_ptr, grad_pw2_w_t.device_ptr, pointwise2_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(pointwise2_w_f32.size))
            lib.sgd_update_fused(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, linear_wv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_w_f32.size))
            lib.sgd_update_fused(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, linear_bv_t.device_ptr, float(lr), float(momentum), float(weight_decay), 0.0, 1.0, int(linear_b_f32.size))
            update_kind = 'gpu_native_train:sgd_update_fused'
        else:
            lib.apply_sgd_update(depthwise_w_t.device_ptr, grad_depthwise_w_t.device_ptr, float(lr), int(depthwise_w_f32.size))
            lib.apply_sgd_update(norm_weight_t.device_ptr, grad_norm_weight_t.device_ptr, float(lr), int(norm_weight_f32.size))
            lib.apply_sgd_update(norm_bias_t.device_ptr, grad_norm_bias_t.device_ptr, float(lr), int(norm_bias_f32.size))
            lib.apply_sgd_update(pointwise1_w_t.device_ptr, grad_pw1_w_t.device_ptr, float(lr), int(pointwise1_w_f32.size))
            lib.apply_sgd_update(pointwise2_w_t.device_ptr, grad_pw2_w_t.device_ptr, float(lr), int(pointwise2_w_f32.size))
            lib.apply_sgd_update(linear_w_t.device_ptr, grad_linear_w_t.device_ptr, float(lr), int(linear_w_f32.size))
            lib.apply_sgd_update(linear_b_t.device_ptr, grad_linear_b_t.device_ptr, float(lr), int(linear_b_f32.size))
            update_kind = 'gpu_native_train:apply_sgd_update'
        runtime.record_execution(update_kind, input_name='grad_depthwise_weight', output_name='depthwise_weight', node_count=1)

        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-depthwise-layernorm2d-pointwise-gelu-pointwise-linear-training-step')
        return NativeGpuDepthwiseLayerNorm2dPointwiseGeluPointwiseLinearTrainingStepResult(
            logits=runtime.stage_to_host(logits_t),
            probabilities=runtime.stage_to_host(probs_t),
            depthwise_output=runtime.stage_to_host(depthwise_t),
            norm_output=runtime.stage_to_host(norm_t),
            pointwise1_output=runtime.stage_to_host(pw1_t),
            activation_output=runtime.stage_to_host(activation_t),
            pointwise2_output=runtime.stage_to_host(pw2_t),
            grad_logits=runtime.stage_to_host(grad_logits_t),
            grad_pointwise2_output=runtime.stage_to_host(grad_pw2_t),
            grad_activation_output=runtime.stage_to_host(grad_activation_before_gelu_t),
            grad_pointwise1_output=runtime.stage_to_host(grad_pw1_t),
            grad_norm_output=runtime.stage_to_host(grad_norm_t),
            grad_depthwise_output=runtime.stage_to_host(grad_depthwise_t),
            grad_input=runtime.stage_to_host(grad_input_t),
            grad_depthwise_weight=runtime.stage_to_host(grad_depthwise_w_t),
            grad_norm_weight=runtime.stage_to_host(grad_norm_weight_t),
            grad_norm_bias=runtime.stage_to_host(grad_norm_bias_t),
            grad_pointwise1_weight=runtime.stage_to_host(grad_pw1_w_t),
            grad_pointwise2_weight=runtime.stage_to_host(grad_pw2_w_t),
            grad_linear_weight=runtime.stage_to_host(grad_linear_w_t),
            grad_linear_bias=runtime.stage_to_host(grad_linear_b_t),
            updated_depthwise_weight=runtime.stage_to_host(depthwise_w_t),
            updated_norm_weight=runtime.stage_to_host(norm_weight_t),
            updated_norm_bias=runtime.stage_to_host(norm_bias_t),
            updated_pointwise1_weight=runtime.stage_to_host(pointwise1_w_t),
            updated_pointwise2_weight=runtime.stage_to_host(pointwise2_w_t),
            updated_linear_weight=runtime.stage_to_host(linear_w_t),
            updated_linear_bias=runtime.stage_to_host(linear_b_t),
            updated_depthwise_weight_velocity=runtime.stage_to_host(depthwise_wv_t),
            updated_norm_weight_velocity=runtime.stage_to_host(norm_wv_t),
            updated_norm_bias_velocity=runtime.stage_to_host(norm_bv_t),
            updated_pointwise1_weight_velocity=runtime.stage_to_host(pointwise1_wv_t),
            updated_pointwise2_weight_velocity=runtime.stage_to_host(pointwise2_wv_t),
            updated_linear_weight_velocity=runtime.stage_to_host(linear_wv_t),
            updated_linear_bias_velocity=runtime.stage_to_host(linear_bv_t),
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)
