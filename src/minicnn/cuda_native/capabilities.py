"""Capability surface for the cuda_native backend.

This module is the single source of truth for what cuda_native currently
supports.  All other modules (api, validators, diagnostics) should read
from here rather than duplicating the list.
"""
from __future__ import annotations

from typing import Any

from minicnn.cuda_native.gpu_kernel_registry import list_gpu_kernel_specs

CAPABILITY_SCHEMA_VERSION = 1
GPU_NATIVE_BOOTSTRAP_OPS = [spec.op_name for spec in list_gpu_kernel_specs()]
GPU_NATIVE_BOOTSTRAP_BLOCKERS = [
    'gpu_graph_backward_generalization_pending',
    'gpu_composite_block_training_pending',
]
GPU_NATIVE_TRAINING_SUBSETS = [
    {
        'name': 'linear',
        'ops': ['Linear'],
        'losses': ['CrossEntropyLoss', 'MSELoss', 'BCEWithLogitsLoss'],
        'optimizers': ['SGD', 'Adam', 'AdamW', 'RMSprop'],
        'helper': 'native_gpu_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'flatten_linear',
        'ops': ['Flatten', 'Linear'],
        'losses': ['CrossEntropyLoss', 'MSELoss', 'BCEWithLogitsLoss'],
        'optimizers': ['SGD', 'Adam', 'AdamW', 'RMSprop'],
        'helper': 'native_gpu_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'linear_relu_linear',
        'ops': ['Linear', 'ReLU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'flatten_linear_relu_linear',
        'ops': ['Flatten', 'Linear', 'ReLU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'linear_leaky_relu_linear',
        'ops': ['Linear', 'LeakyReLU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'flatten_linear_leaky_relu_linear',
        'ops': ['Flatten', 'Linear', 'LeakyReLU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'linear_gelu_linear',
        'ops': ['Linear', 'GELU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'flatten_linear_gelu_linear',
        'ops': ['Flatten', 'Linear', 'GELU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'linear_silu_linear',
        'ops': ['Linear', 'SiLU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'flatten_linear_silu_linear',
        'ops': ['Flatten', 'Linear', 'SiLU', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'linear_sigmoid_linear',
        'ops': ['Linear', 'Sigmoid', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'flatten_linear_sigmoid_linear',
        'ops': ['Flatten', 'Linear', 'Sigmoid', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'linear_tanh_linear',
        'ops': ['Linear', 'Tanh', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'flatten_linear_tanh_linear',
        'ops': ['Flatten', 'Linear', 'Tanh', 'Linear'],
        'helper': 'native_gpu_two_linear_relu_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'maxpool_linear',
        'ops': ['MaxPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_pool_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'avgpool_linear',
        'ops': ['AvgPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_avgpool_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'batchnorm_linear',
        'ops': ['BatchNorm2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_batchnorm_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'layernorm2d_linear',
        'ops': ['LayerNorm2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_layernorm2d_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'groupnorm_linear',
        'ops': ['GroupNorm', 'Flatten', 'Linear'],
        'helper': 'native_gpu_groupnorm_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'depthwise_layernorm2d_linear',
        'ops': ['DepthwiseConv2d', 'LayerNorm2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_depthwise_layernorm2d_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'depthwise_layernorm2d_pointwise_linear',
        'ops': ['DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_depthwise_layernorm2d_pointwise_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'depthwise_layernorm2d_pointwise_gelu_pointwise_linear',
        'ops': ['DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'GELU', 'PointwiseConv2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'global_avgpool_linear',
        'ops': ['GlobalAvgPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_global_avgpool_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'adaptive_avgpool_linear',
        'ops': ['AdaptiveAvgPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_global_avgpool_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'conv_linear',
        'ops': ['Conv2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'conv_relu_linear',
        'ops': ['Conv2d', 'ReLU', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'pointwise_conv_linear',
        'ops': ['PointwiseConv2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'pointwise_conv_relu_linear',
        'ops': ['PointwiseConv2d', 'ReLU', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'depthwise_conv_linear',
        'ops': ['DepthwiseConv2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'depthwise_conv_relu_linear',
        'ops': ['DepthwiseConv2d', 'ReLU', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'depthwise_conv_pool_linear',
        'ops': ['DepthwiseConv2d', 'MaxPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'depthwise_conv_relu_pool_linear',
        'ops': ['DepthwiseConv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'conv_pool_linear',
        'ops': ['Conv2d', 'MaxPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'conv_relu_pool_linear',
        'ops': ['Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_conv_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
    {
        'name': 'two_conv_relu_pool_linear',
        'ops': ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'],
        'helper': 'native_gpu_two_conv_relu_pool_linear_training_step',
        'parity': 'hermetic_reference_math',
    },
]


def _training_subset_constraints(ops: list[str]) -> list[str]:
    constraints: list[str] = []
    if any(op in ops for op in ('Conv2d', 'DepthwiseConv2d', 'PointwiseConv2d')):
        constraints.append('Conv-family training helpers require bias=false, stride=1, padding=0, dilation=1')
    if 'PointwiseConv2d' in ops:
        constraints.append('PointwiseConv2d training helpers require kernel_size=1')
    if any(op in ops for op in ('AvgPool2d', 'MaxPool2d')):
        constraints.append('Pool training helpers require kernel_size=2, stride=2, padding=0')
    if 'AdaptiveAvgPool2d' in ops:
        constraints.append('AdaptiveAvgPool2d training helper requires output_size=1')
    return constraints


for _subset in GPU_NATIVE_TRAINING_SUBSETS:
    _subset.setdefault('losses', ['CrossEntropyLoss'])
    _subset.setdefault('optimizers', ['SGD'])
    _subset.setdefault('execution_mode', 'gpu_native')
    _subset.setdefault('fallback_execution_mode', 'reference_numpy')
    _subset.setdefault('fallback_policy', {
        'fallback_execution_mode': 'reference_numpy',
        'fallback_available': True,
        'fallback_active_when': 'gpu_native_lowering_not_ready',
    })
    _subset.setdefault('constraints', _training_subset_constraints(list(_subset.get('ops', []))))

for _conv_prefix, _subset_prefix in (
    ('Conv2d', 'conv'),
    ('PointwiseConv2d', 'pointwise_conv'),
    ('DepthwiseConv2d', 'depthwise_conv'),
):
    for _activation in ('LeakyReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh'):
        _activation_key = _activation.replace('ReLU', '_relu').lower()
        GPU_NATIVE_TRAINING_SUBSETS.append(
            {
                'name': f'{_subset_prefix}_{_activation_key}_linear',
                'ops': [_conv_prefix, _activation, 'Flatten', 'Linear'],
                'helper': 'native_gpu_conv_linear_training_step',
                'parity': 'hermetic_reference_math',
                'losses': ['CrossEntropyLoss'],
                'optimizers': ['SGD'],
                'execution_mode': 'gpu_native',
                'fallback_execution_mode': 'reference_numpy',
                'fallback_policy': {
                    'fallback_execution_mode': 'reference_numpy',
                    'fallback_available': True,
                    'fallback_active_when': 'gpu_native_lowering_not_ready',
                },
                'constraints': _training_subset_constraints([_conv_prefix, _activation, 'Flatten', 'Linear']),
            }
        )
        if _conv_prefix != 'PointwiseConv2d':
            GPU_NATIVE_TRAINING_SUBSETS.append(
                {
                    'name': f'{_subset_prefix}_{_activation_key}_pool_linear',
                    'ops': [_conv_prefix, _activation, 'MaxPool2d', 'Flatten', 'Linear'],
                    'helper': 'native_gpu_conv_linear_training_step',
                    'parity': 'hermetic_reference_math',
                    'losses': ['CrossEntropyLoss'],
                    'optimizers': ['SGD'],
                    'execution_mode': 'gpu_native',
                    'fallback_execution_mode': 'reference_numpy',
                    'fallback_policy': {
                        'fallback_execution_mode': 'reference_numpy',
                        'fallback_available': True,
                        'fallback_active_when': 'gpu_native_lowering_not_ready',
                    },
                    'constraints': _training_subset_constraints([_conv_prefix, _activation, 'MaxPool2d', 'Flatten', 'Linear']),
                }
            )

for _activation in ('LeakyReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh'):
    _activation_key = _activation.replace('ReLU', '_relu').lower()
    GPU_NATIVE_TRAINING_SUBSETS.append(
        {
            'name': f'two_conv_{_activation_key}_pool_linear',
            'ops': ['Conv2d', _activation, 'Conv2d', _activation, 'MaxPool2d', 'Flatten', 'Linear'],
            'helper': 'native_gpu_two_conv_relu_pool_linear_training_step',
            'parity': 'hermetic_reference_math',
            'losses': ['CrossEntropyLoss'],
            'optimizers': ['SGD'],
            'execution_mode': 'gpu_native',
            'fallback_execution_mode': 'reference_numpy',
            'fallback_policy': {
                'fallback_execution_mode': 'reference_numpy',
                'fallback_available': True,
                'fallback_active_when': 'gpu_native_lowering_not_ready',
            },
            'constraints': _training_subset_constraints(['Conv2d', _activation, 'Conv2d', _activation, 'MaxPool2d', 'Flatten', 'Linear']),
        }
    )

CUDA_NATIVE_SUPPORT_TIERS: dict[str, dict[str, list[str]]] = {
    'stable': {
        'ops': [
            'Add',
            'BatchNorm2d',
            'Concat',
            'Conv2d',
        'DepthwiseConv2d',
        'Flatten',
        'GroupNorm',
        'Identity',
        'LayerNorm',
            'LayerNorm2d',
            'Linear',
            'PointwiseConv2d',
        ],
        'optimizers': ['AdamW', 'SGD'],
        'losses': ['CrossEntropyLoss', 'MSELoss'],
        'features': [
            'artifact_contracts',
            'ordered_dag',
            'validation_contracts',
        ],
    },
    'beta': {
        'ops': [
        'AdaptiveAvgPool2d',
        'AvgPool2d',
        'ConvNeXtBlock',
        'DropPath',
        'Dropout',
        'ResidualBlock',
        'GELU',
        'GlobalAvgPool2d',
            'LeakyReLU',
            'MaxPool2d',
            'ReLU',
            'Sigmoid',
            'SiLU',
            'Tanh',
        ],
        'optimizers': ['Adam', 'RMSprop'],
        'losses': ['BCEWithLogitsLoss'],
        'features': [
            'amp',
            'branching_graph',
            'performance_report',
            'reproducibility_smoke',
            'tolerance_matrix',
        ],
    },
    'experimental': {
        'ops': [],
        'optimizers': [],
        'losses': [],
        'features': [],
    },
}

CUDA_NATIVE_GRADUATION_GATES: dict[str, object] = {
    'core_beta_subset': {
        'ready': True,
        'criteria': {
            'artifact_contract_frozen': True,
            'validation_contract_frozen': True,
            'smoke_matrix_present': True,
            'tolerance_matrix_present': True,
            'support_tiers_published': True,
            'support_tiers_machine_readable': True,
            'core_parity_baseline_present': True,
            'composite_parity_baseline_present': True,
        },
        'remaining_blockers': [],
    },
    'full_backend_non_experimental': {
        'ready': True,
        'criteria': {
            'artifact_contract_frozen': True,
            'validation_contract_frozen': True,
            'smoke_matrix_present': True,
            'tolerance_matrix_present': True,
            'support_tiers_published': True,
            'support_tiers_machine_readable': True,
            'core_parity_baseline_present': True,
            'composite_parity_baseline_present': True,
            'amp_tolerance_matrix_present': True,
            'amp_composite_tolerance_matrix_present': True,
            'amp_reproducible_smoke_present': True,
            'training_stable': True,
            'backward_stable': True,
            'amp_graduated': True,
        },
        'remaining_blockers': [],
    },
}

CUDA_NATIVE_CAPABILITIES: dict[str, object] = {
    'experimental': False,
    'production_ready': False,
    'numpy_reference': True,
    'sequential_only': False,
    'forward_only': False,
    'training': True,
    'training_stable': True,
    'backward': True,
    'backward_stable': True,
    'dynamic_shapes': False,
    'branching_graph': True,
    'amp': True,
    'supports_depthwise_conv': True,
    'supports_pointwise_conv': True,
    'supports_groupnorm': True,
    'supports_layernorm': True,
    'supports_layernorm2d': True,
    'supports_gelu': True,
    'supports_residual_add': True,
    'supports_convnext_block': True,
    'supported_datasets': ['random', 'cifar10', 'mnist'],
    'supported_losses': ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss'],
    'supported_optimizers': ['SGD', 'Adam', 'AdamW', 'RMSprop'],
    'supported_schedulers': [
        'StepLR',
        'CosineAnnealingLR',
        'ReduceLROnPlateau',
    ],
    'supported_ops': [
        'BatchNorm2d',
        'Concat',
        'Conv2d',
        'DepthwiseConv2d',
        'PointwiseConv2d',
        'GroupNorm',
        'LayerNorm',
        'LayerNorm2d',
        'DropPath',
        'Dropout',
        'ReLU',
        'LeakyReLU',
        'Sigmoid',
        'Tanh',
        'SiLU',
        'GELU',
        'Flatten',
        'Linear',
        'MaxPool2d',
        'AvgPool2d',
        'AdaptiveAvgPool2d',
        'Add',
        'GlobalAvgPool2d',
        'Identity',
        'ResidualBlock',
        'ConvNeXtBlock',
        'depthwise_conv2d',
        'pointwise_conv2d',
        'layernorm2d',
        'convnext_block',
    ],
    'planned_ops': [],
    'unsupported_ops': [
        'Embedding',
        'SelfAttention',
        'Upsample',
    ],
    'notes': [
        'Backward and training now meet the current beta graduation gate, but the backend is not yet production-ready.',
        'BatchNorm2d forward/backward exist within the beta training surface; gpu_native train-native covers BatchNorm2d+Flatten+Linear.',
        'LayerNorm uses numpy reference kernels; GroupNorm train-native now covers GroupNorm -> Flatten -> Linear through groupnorm forward/backward C ABI shims.',
        'cuda_native is GPU-first for the active enablement path; numpy kernels are retained as historical fallback and hermetic parity baselines.',
        'gpu_native two-Conv helper now also covers same-activation modern variants for LeakyReLU, GELU, SiLU, Sigmoid, and Tanh in addition to the original ReLU/ReLU path.',
        'ResidualBlock and ConvNeXtBlock still run through composite/reference numpy kernels; support tier is published separately.',
        'Identity plus Dropout/DropPath with p=0 are gpu_native no-op aliases; stochastic Dropout/DropPath training still requires native mask kernels.',
        'Explicit ordered DAG wiring is supported through named tensor outputs plus Add/Concat multi-input nodes.',
        'train-native supports SGD, Adam, AdamW, RMSprop, BCEWithLogitsLoss, label_smoothing for cross entropy, grad_accum_steps >= 1, and beta AMP with loss scaling / overflow backoff.',
        'gpu_native forward dispatch now includes BatchNorm2d; train-native still restricts BatchNorm2d to reference_numpy until a training helper or graph-backward lowering lands.',
        'gpu_native train-native includes GlobalAvgPool2d/AdaptiveAvgPool2d(output_size=1)+Flatten+Linear through native global_avgpool2d forward/backward C ABI shims.',
        'gpu_native train-native includes AvgPool2d(2x2 stride-2)+Flatten+Linear through native avgpool2d forward/backward C ABI shims.',
        'gpu_native forward dispatch now includes LeakyReLU, GELU, SiLU, Sigmoid, and Tanh elementwise activation C ABI shims; two-linear train-native helper coverage is active for these activations.',
        'gpu_native now exposes GELU, LeakyReLU, SiLU, Sigmoid, and Tanh activation backward C ABI shims as the prerequisite for modern activation train-native helpers.',
        'gpu_native train-native includes PointwiseConv2d(1x1,bias=false)+Flatten+Linear through the Conv2d helper path.',
        'gpu_native forward dispatch includes DepthwiseConv2d through a native depthwise_conv2d C ABI shim.',
        'gpu_native train-native now covers DepthwiseConv2d -> optional ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> optional MaxPool2d -> Flatten -> Linear through depthwise forward/backward C ABI shims.',
        'gpu_native train-native now covers DepthwiseConv2d -> LayerNorm2d -> Flatten -> Linear as a ConvNeXt-style native GPU bridge subset.',
        'gpu_native train-native now covers DepthwiseConv2d -> LayerNorm2d -> PointwiseConv2d -> Flatten -> Linear as a deeper ConvNeXt-style native GPU bridge subset.',
        'gpu_native train-native now covers DepthwiseConv2d -> LayerNorm2d -> PointwiseConv2d -> GELU -> PointwiseConv2d -> Flatten -> Linear as the deepest current ConvNeXt-style native GPU bridge subset.',
        'gpu_native forward dispatch includes LayerNorm2d through a native layernorm2d C ABI shim.',
        'gpu_native train-native now covers LayerNorm2d -> Flatten -> Linear through layernorm2d forward/backward C ABI shims.',
        'gpu_native train-native currently covers narrow Linear, Linear+ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh, MaxPool+Linear, Conv2d/PointwiseConv2d/DepthwiseConv2d(valid,bias=false)+optional ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh+optional MaxPool+Linear, and two-Conv ReLU+MaxPool+Linear subsets through native device-pointer helpers.',
        'gpu_native readiness diagnostics expose a training_lowering_plan that decomposes helper subsets into forward, loss, backward, and optimizer lowering steps.',
        'gpu_native Linear subsets support native CrossEntropyLoss with label_smoothing, MSELoss, and BCEWithLogitsLoss loss-gradient helpers; Conv-family subsets currently support CrossEntropyLoss with label_smoothing.',
        'gpu_native Linear subsets support native SGD, Adam, AdamW, and RMSprop update helpers; Conv-family subsets currently support SGD.',
        'gpu_native SGD helper subsets support native weight_decay through sgd_update_fused.',
        'gpu_native training subsets support native global-norm gradient clipping through grad_l2_sumsq plus scale_inplace.',
        'validate-cuda-native-config enforces the current train-native support boundary.',
    ],
}

CUDA_NATIVE_EXECUTION_MODE_READINESS: dict[str, dict[str, object]] = {
    'reference_numpy': {
        'status': 'active',
        'ready': True,
        'tensor_execution_device': 'cpu',
        'bootstrap_subset_ops': [],
        'kernel_readiness': {},
        'remaining_blockers': [],
    },
    'gpu_native_auto': {
        'status': 'gpu_first_with_reference_numpy_fallback',
        'ready': True,
        'tensor_execution_device': 'auto',
        'bootstrap_subset_ops': GPU_NATIVE_BOOTSTRAP_OPS,
        'training_subsets': GPU_NATIVE_TRAINING_SUBSETS,
        'kernel_readiness': {
            spec.op_name: spec.forward_status
            for spec in list_gpu_kernel_specs()
        },
        'fallback_execution_mode': 'reference_numpy',
        'fallback_available': True,
        'remaining_blockers': GPU_NATIVE_BOOTSTRAP_BLOCKERS,
    },
    'gpu_native': {
        'status': 'bootstrap_training_partial',
        'ready': True,
        'tensor_execution_device': 'gpu',
        'bootstrap_subset_ops': GPU_NATIVE_BOOTSTRAP_OPS,
        'training_subsets': GPU_NATIVE_TRAINING_SUBSETS,
        'kernel_readiness': {
            spec.op_name: spec.forward_status
            for spec in list_gpu_kernel_specs()
        },
        'remaining_blockers': GPU_NATIVE_BOOTSTRAP_BLOCKERS,
    },
}


def _sorted_unique_strings(items: object) -> list[str]:
    return sorted({str(item) for item in items if str(item)})


def _normalized_support_tiers() -> dict[str, dict[str, list[str]]]:
    return {
        tier: {
            bucket: _sorted_unique_strings(values)
            for bucket, values in buckets.items()
        }
        for tier, buckets in CUDA_NATIVE_SUPPORT_TIERS.items()
    }


def _normalized_graduation_gates() -> dict[str, object]:
    normalized: dict[str, object] = {}
    for gate_name, gate in CUDA_NATIVE_GRADUATION_GATES.items():
        gate_dict = dict(gate)
        gate_dict['criteria'] = {
            str(key): bool(value)
            for key, value in dict(gate_dict.get('criteria', {})).items()
        }
        gate_dict['remaining_blockers'] = [
            str(item)
            for item in gate_dict.get('remaining_blockers', [])
        ]
        gate_dict['ready'] = bool(gate_dict.get('ready', False))
        normalized[gate_name] = gate_dict
    return normalized


def _kernel_registry_surface() -> list[dict[str, str]]:
    from minicnn.cuda_native.kernels import DEFAULT_KERNEL_SPECS

    return [
        {
            'op_name': spec.op_name,
            'category': spec.category,
        }
        for spec in sorted(DEFAULT_KERNEL_SPECS, key=lambda spec: spec.op_name)
    ]


def _gpu_kernel_registry_surface() -> list[dict[str, str]]:
    return [
        {
            'op_name': spec.op_name,
            'category': spec.category,
            'forward_status': spec.forward_status,
            'backward_status': spec.backward_status,
        }
        for spec in list_gpu_kernel_specs()
    ]


def _normalized_execution_mode_readiness() -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for mode, readiness in CUDA_NATIVE_EXECUTION_MODE_READINESS.items():
        readiness_dict = dict(readiness)
        readiness_dict['status'] = str(readiness_dict.get('status', 'unknown'))
        readiness_dict['ready'] = bool(readiness_dict.get('ready', False))
        readiness_dict['tensor_execution_device'] = str(readiness_dict.get('tensor_execution_device', 'unknown'))
        readiness_dict['bootstrap_subset_ops'] = _sorted_unique_strings(
            readiness_dict.get('bootstrap_subset_ops', [])
        )
        readiness_dict['kernel_readiness'] = {
            str(key): str(value)
            for key, value in dict(readiness_dict.get('kernel_readiness', {})).items()
        }
        readiness_dict['remaining_blockers'] = [
            str(item)
            for item in readiness_dict.get('remaining_blockers', [])
        ]
        normalized[str(mode)] = readiness_dict
    return normalized


def get_cuda_native_capabilities() -> dict[str, Any]:
    """Return a versioned, machine-readable cuda_native capability descriptor."""
    caps = dict(CUDA_NATIVE_CAPABILITIES)
    caps['supported_ops'] = _sorted_unique_strings(caps['supported_ops'])
    caps['planned_ops'] = _sorted_unique_strings(caps['planned_ops'])
    caps['unsupported_ops'] = _sorted_unique_strings(caps['unsupported_ops'])
    caps['supported_datasets'] = _sorted_unique_strings(caps['supported_datasets'])
    caps['supported_losses'] = _sorted_unique_strings(caps['supported_losses'])
    caps['supported_optimizers'] = _sorted_unique_strings(caps['supported_optimizers'])
    caps['supported_schedulers'] = _sorted_unique_strings(caps['supported_schedulers'])
    caps['notes'] = [str(note) for note in caps['notes']]
    support_tiers = _normalized_support_tiers()
    graduation_gates = _normalized_graduation_gates()
    kernel_surface = _kernel_registry_surface()
    gpu_kernel_surface = _gpu_kernel_registry_surface()
    execution_mode_readiness = _normalized_execution_mode_readiness()
    caps.update({
        'schema_version': CAPABILITY_SCHEMA_VERSION,
        'backend': 'cuda_native',
        'status': 'ok',
        'summary_status': 'beta',
        'capability_kind': 'backend_capability_summary',
        'execution_modes_supported': ['reference_numpy', 'gpu_native_auto', 'gpu_native'],
        'execution_modes_planned': [],
        'default_execution_mode': 'gpu_native_auto',
        'preferred_gpu_first_execution_mode': 'gpu_native_auto',
        'default_execution_policy': 'gpu_first_with_reference_numpy_fallback',
        'default_tensor_execution_device': 'auto',
        'gpu_execution': False,
        'support_tiers': support_tiers,
        'support_tier_counts': {
            tier: {
                bucket: len(values)
                for bucket, values in buckets.items()
            }
            for tier, buckets in support_tiers.items()
        },
        'graduation_gates': graduation_gates,
        'execution_mode_readiness': execution_mode_readiness,
        'supported_op_categories': sorted({entry['category'] for entry in kernel_surface}),
        'kernel_registry_surface': kernel_surface,
        'gpu_kernel_registry_surface': gpu_kernel_surface,
    })
    return caps
