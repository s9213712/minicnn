from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_training_common import (
    _apply_global_grad_clip,
    _load_bound_lib,
    _run_softmax_xent_loss,
)
from minicnn.cuda_native.gpu_training_types import NativeGpuMlpTrainingStepResult


_ACTIVATION_FORWARD: dict[str, tuple[str, str]] = {
    'ReLU': ('apply_relu', 'apply_relu'),
    'LeakyReLU': ('leaky_relu_forward', 'leaky_relu_forward'),
    'GELU': ('gelu_forward', 'gelu_forward'),
    'SiLU': ('silu_forward', 'silu_forward'),
    'Sigmoid': ('sigmoid_forward', 'sigmoid_forward'),
    'Tanh': ('tanh_forward', 'tanh_forward'),
}

_ACTIVATION_BACKWARD: dict[str, tuple[str, str]] = {
    'ReLU': ('apply_relu_backward', 'apply_relu_backward'),
    'LeakyReLU': ('leaky_relu_backward', 'leaky_relu_backward'),
    'GELU': ('gelu_backward', 'gelu_backward'),
    'SiLU': ('silu_backward', 'silu_backward'),
    'Sigmoid': ('sigmoid_backward', 'sigmoid_backward'),
    'Tanh': ('tanh_backward', 'tanh_backward'),
}


def native_gpu_mlp_training_step(
    x: np.ndarray,
    labels: np.ndarray,
    linear_params: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    activations: tuple[tuple[str, float], ...],
    *,
    lr: float,
    momentum: float = 0.0,
    grad_clip_value: float = 0.0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    velocity: dict[str, np.ndarray] | None = None,
    bound_lib: Any | None = None,
    device_runtime: DeviceRuntime | None = None,
    persistent_device_state: bool = False,
    persistent_cache_prefix: str = 'mlp',
    return_intermediates: bool = True,
) -> NativeGpuMlpTrainingStepResult:
    """Run a sequential MLP GPU-native training step with per-op kernels."""

    x_f32 = np.ascontiguousarray(x.reshape(x.shape[0], -1), dtype=np.float32)
    labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
    if not linear_params:
        raise ValueError('native_gpu_mlp_training_step requires at least one Linear layer.')
    if len(activations) != len(linear_params) - 1:
        raise ValueError(
            'native_gpu_mlp_training_step expects one activation between each hidden Linear layer; '
            f'got {len(activations)} activations for {len(linear_params)} Linear layers.'
        )
    if labels_i32.ndim != 1 or labels_i32.shape[0] != x_f32.shape[0]:
        raise ValueError(
            'native_gpu_mlp_training_step expects labels with shape (N,), '
            f'got labels={labels_i32.shape} for x={x_f32.shape}'
        )

    names: list[str] = []
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    prev_features = int(x_f32.shape[1])
    for name, weight, bias in linear_params:
        weight_f32 = np.ascontiguousarray(weight, dtype=np.float32)
        bias_f32 = np.ascontiguousarray(bias, dtype=np.float32)
        if weight_f32.ndim != 2 or weight_f32.shape[1] != prev_features:
            raise ValueError(
                f'native_gpu_mlp_training_step expects {name} weight with shape '
                f'(out_f, {prev_features}), got {weight_f32.shape}.'
            )
        if bias_f32.shape != (weight_f32.shape[0],):
            raise ValueError(
                f'native_gpu_mlp_training_step expects {name} bias with shape '
                f'({weight_f32.shape[0]},), got {bias_f32.shape}.'
            )
        names.append(str(name))
        weights.append(weight_f32)
        biases.append(bias_f32)
        prev_features = int(weight_f32.shape[0])
    out_f = int(weights[-1].shape[0])
    if np.any(labels_i32 < 0) or np.any(labels_i32 >= out_f):
        raise ValueError('native_gpu_mlp_training_step labels must be in [0, out_f).')
    for activation_name, _alpha in activations:
        if activation_name not in _ACTIVATION_FORWARD:
            raise ValueError(f'native_gpu_mlp_training_step does not support activation={activation_name!r}.')

    velocity = dict(velocity or {})
    lib = _load_bound_lib(bound_lib if bound_lib is not None else (device_runtime.bound_lib if device_runtime is not None else None))
    runtime = device_runtime if device_runtime is not None else DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=lib,
    )
    runtime.bound_lib = lib

    tensors: list[Any] = []

    def stage(array: np.ndarray, name: str):
        tensor = runtime.stage_to_device(array, name=name)
        tensors.append(tensor)
        return tensor

    def stage_state(array: np.ndarray, cache_name: str):
        if persistent_device_state:
            tensor = runtime.stage_persistent_to_device(
                array,
                key=f'{persistent_cache_prefix}:{cache_name}',
                name=cache_name,
                update_on_reuse=False,
            )
        else:
            tensor = runtime.stage_to_device(array, name=cache_name)
        tensors.append(tensor)
        return tensor

    def alloc(shape: tuple[int, ...], name: str, dtype: str = 'float32'):
        tensor = runtime.allocate(shape, dtype=dtype, name=name)
        tensors.append(tensor)
        return tensor

    n = int(x_f32.shape[0])
    input_t = stage(x_f32, 'input')
    labels_t = stage(labels_i32, 'labels')
    weight_tensors = [stage_state(weight, f'{name}:weight') for name, weight in zip(names, weights)]
    bias_tensors = [stage_state(bias, f'{name}:bias') for name, bias in zip(names, biases)]
    weight_velocity_tensors = [
        stage_state(
            np.zeros_like(weight) if f'_w_{name}' not in velocity else np.ascontiguousarray(velocity[f'_w_{name}'], dtype=np.float32),
            f'{name}:weight_velocity',
        )
        for name, weight in zip(names, weights)
    ]
    bias_velocity_tensors = [
        stage_state(
            np.zeros_like(bias) if f'_b_{name}' not in velocity else np.ascontiguousarray(velocity[f'_b_{name}'], dtype=np.float32),
            f'{name}:bias_velocity',
        )
        for name, bias in zip(names, biases)
    ]

    loss_sum_t = alloc((1,), 'loss_sum')
    correct_t = alloc((1,), 'correct_count', dtype='int32')
    probs_t = alloc((n, out_f), 'probs')
    grad_logits_t = alloc((n, out_f), 'grad_logits')
    linear_inputs: list[Any] = []
    linear_outputs: list[Any] = []
    activation_inputs: list[Any] = []
    activation_outputs: list[Any] = []
    grad_param_tensors: dict[str, Any] = {}
    updated_params: dict[str, np.ndarray] = {}
    updated_velocity: dict[str, np.ndarray] = {}
    grad_params: dict[str, np.ndarray] = {}

    try:
        lib.gpu_memset(loss_sum_t.device_ptr, 0, loss_sum_t.nbytes)
        lib.gpu_memset(correct_t.device_ptr, 0, correct_t.nbytes)
        current_t = input_t
        current_features = int(x_f32.shape[1])
        for idx, (name, weight_t, bias_t, weight) in enumerate(zip(names, weight_tensors, bias_tensors, weights)):
            out_features = int(weight.shape[0])
            linear_inputs.append(current_t)
            out_t = alloc((n, out_features), f'{name}:linear_output')
            lib.dense_forward(
                current_t.device_ptr,
                weight_t.device_ptr,
                bias_t.device_ptr,
                out_t.device_ptr,
                n,
                current_features,
                out_features,
            )
            runtime.record_execution(
                f'gpu_native_train:generic_mlp:dense_forward_{idx + 1}',
                input_name=current_t.name or 'input',
                output_name=out_t.name,
                node_count=1,
            )
            linear_outputs.append(out_t)
            current_t = out_t
            current_features = out_features
            if idx < len(activations):
                activation_name, activation_alpha = activations[idx]
                activation_inputs.append(out_t)
                act_t = alloc((n, out_features), f'{name}:activation_output')
                lib.gpu_memcpy_d2d(act_t.device_ptr, out_t.device_ptr, out_t.nbytes)
                forward_symbol, forward_kind = _ACTIVATION_FORWARD[activation_name]
                if activation_name == 'LeakyReLU':
                    getattr(lib, forward_symbol)(act_t.device_ptr, float(activation_alpha), int(n * out_features))
                else:
                    getattr(lib, forward_symbol)(act_t.device_ptr, int(n * out_features))
                runtime.record_execution(
                    f'gpu_native_train:generic_mlp:{forward_kind}',
                    input_name=out_t.name,
                    output_name=act_t.name,
                    node_count=1,
                )
                activation_outputs.append(act_t)
                current_t = act_t
        logits_t = current_t
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
        runtime.record_execution(loss_kind, input_name=logits_t.name, output_name='grad_logits', node_count=1)

        grad_current_t = grad_logits_t
        grad_input_t = grad_logits_t
        grad_tensors_for_clip: list[tuple[Any, int]] = []
        for idx in range(len(names) - 1, -1, -1):
            input_for_linear_t = linear_inputs[idx]
            in_features = int(weights[idx].shape[1])
            out_features = int(weights[idx].shape[0])
            grad_input_t = alloc((n, in_features), f'{names[idx]}:grad_input')
            grad_weight_t = alloc(weights[idx].shape, f'{names[idx]}:grad_weight')
            grad_bias_t = alloc(biases[idx].shape, f'{names[idx]}:grad_bias')
            lib.dense_backward_full(
                grad_current_t.device_ptr,
                input_for_linear_t.device_ptr,
                weight_tensors[idx].device_ptr,
                grad_input_t.device_ptr,
                grad_weight_t.device_ptr,
                grad_bias_t.device_ptr,
                n,
                in_features,
                out_features,
            )
            runtime.record_execution(
                f'gpu_native_train:generic_mlp:dense_backward_full_{idx + 1}',
                input_name=grad_current_t.name,
                output_name=grad_weight_t.name,
                node_count=1,
            )
            grad_param_tensors[f'_w_{names[idx]}'] = grad_weight_t
            grad_param_tensors[f'_b_{names[idx]}'] = grad_bias_t
            grad_tensors_for_clip.append((grad_weight_t, int(weights[idx].size)))
            grad_tensors_for_clip.append((grad_bias_t, int(biases[idx].size)))
            grad_current_t = grad_input_t
            if idx > 0:
                activation_name, activation_alpha = activations[idx - 1]
                backward_symbol, backward_kind = _ACTIVATION_BACKWARD[activation_name]
                activation_input_t = activation_inputs[idx - 1]
                if activation_name == 'LeakyReLU':
                    getattr(lib, backward_symbol)(
                        activation_input_t.device_ptr,
                        grad_current_t.device_ptr,
                        float(activation_alpha),
                        int(n * in_features),
                    )
                else:
                    getattr(lib, backward_symbol)(
                        activation_input_t.device_ptr,
                        grad_current_t.device_ptr,
                        int(n * in_features),
                    )
                runtime.record_execution(
                    f'gpu_native_train:generic_mlp:{backward_kind}',
                    input_name=activation_input_t.name,
                    output_name=grad_current_t.name,
                    node_count=1,
                )
        _apply_global_grad_clip(runtime, lib, tuple(grad_tensors_for_clip), float(grad_clip_value))

        update_kind = 'gpu_native_train:generic_mlp:sgd_update_fused'
        if float(momentum) != 0.0 or float(weight_decay) != 0.0:
            for idx, name in enumerate(names):
                lib.sgd_update_fused(
                    weight_tensors[idx].device_ptr,
                    grad_param_tensors[f'_w_{name}'].device_ptr,
                    weight_velocity_tensors[idx].device_ptr,
                    float(lr),
                    float(momentum),
                    float(weight_decay),
                    0.0,
                    1.0,
                    int(weights[idx].size),
                )
                lib.sgd_update_fused(
                    bias_tensors[idx].device_ptr,
                    grad_param_tensors[f'_b_{name}'].device_ptr,
                    bias_velocity_tensors[idx].device_ptr,
                    float(lr),
                    float(momentum),
                    float(weight_decay),
                    0.0,
                    1.0,
                    int(biases[idx].size),
                )
        else:
            update_kind = 'gpu_native_train:generic_mlp:apply_sgd_update'
            for idx, name in enumerate(names):
                lib.apply_sgd_update(
                    weight_tensors[idx].device_ptr,
                    grad_param_tensors[f'_w_{name}'].device_ptr,
                    float(lr),
                    int(weights[idx].size),
                )
                lib.apply_sgd_update(
                    bias_tensors[idx].device_ptr,
                    grad_param_tensors[f'_b_{name}'].device_ptr,
                    float(lr),
                    int(biases[idx].size),
                )
        runtime.record_execution(update_kind, input_name='grad_params', output_name='params', node_count=1)

        if return_intermediates:
            logits = runtime.stage_to_host(logits_t)
            probabilities = runtime.stage_to_host(probs_t)
            grad_logits = runtime.stage_to_host(grad_logits_t)
            grad_input = runtime.stage_to_host(grad_input_t)
            grad_params = {
                key: runtime.stage_to_host(tensor)
                for key, tensor in grad_param_tensors.items()
            }
        else:
            empty = np.empty((0,), dtype=np.float32)
            logits = empty
            probabilities = empty
            grad_logits = empty
            grad_input = empty
            grad_params = {}
        for idx, name in enumerate(names):
            updated_params[f'_w_{name}'] = runtime.stage_to_host(weight_tensors[idx])
            updated_params[f'_b_{name}'] = runtime.stage_to_host(bias_tensors[idx])
            if return_intermediates or float(momentum) != 0.0:
                updated_velocity[f'_w_{name}'] = runtime.stage_to_host(weight_velocity_tensors[idx])
                updated_velocity[f'_b_{name}'] = runtime.stage_to_host(bias_velocity_tensors[idx])
        loss_sum = float(runtime.stage_to_host(loss_sum_t)[0])
        correct_count = int(runtime.stage_to_host(correct_t)[0])
        runtime.synchronize('gpu-native-generic-mlp-training-step')
        return NativeGpuMlpTrainingStepResult(
            logits=logits,
            probabilities=probabilities,
            grad_logits=grad_logits,
            grad_input=grad_input,
            grad_params=grad_params,
            updated_params=updated_params,
            updated_velocity=updated_velocity if updated_velocity else None,
            loss_sum=loss_sum,
            loss_mean=loss_sum / float(n),
            correct_count=correct_count,
            runtime_summary=runtime.summary(),
        )
    finally:
        for tensor in tensors:
            runtime.release_buffer(tensor)
