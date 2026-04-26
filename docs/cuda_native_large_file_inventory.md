# CUDA Native Large File Inventory

Last updated: 2026-04-26

This note tracks large MiniCNN files that can slow review and maintenance as `cuda_native` continues from beta toward stable-ready GPU-native execution.

## Current production-code candidates

| File | Approx. lines | Status | Suggested next action |
| --- | ---: | --- | --- |
| `src/minicnn/unified/_cuda_native_training_loop.py` | ~0.4k | Epoch loop, fallback training path, metrics, and profiling orchestration after GPU-native batch dispatch extraction. | Leave as-is unless epoch-state reporting starts growing materially again. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch_conv.py` | ~0.25k | Conv/depthwise gpu-native batch dispatch helpers extracted from the main dispatch router. | Leave as-is unless many new conv-family subsets land. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch_linear.py` | ~0.2k | Linear/pool gpu-native batch dispatch helpers extracted from the main dispatch router. | Leave as-is unless optimizer-family routing becomes materially more complex. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch_norm.py` | ~0.2k | Norm-family gpu-native batch dispatch helpers extracted from the main dispatch router. | Leave as-is unless normalization families multiply again. |
| `src/minicnn/cuda_native/gpu_training_depthwise_activation.py` | ~0.4k | Two-pointwise + activation bridge helper extracted from the bridge-family module. | Leave as-is unless another activation-heavy depthwise bridge topology is added. |
| `src/minicnn/cuda_native/gpu_training_depthwise_pointwise.py` | ~0.3k | Single-pointwise depthwise bridge helper extracted from the bridge-family module. | Leave as-is unless more single-pointwise variants appear. |
| `src/minicnn/cuda_native/gpu_lowering_conv.py` | ~0.3k | Conv/pool lowering helpers extracted from the main lowering module. | Leave as-is unless many more conv lowering variants land. |
| `src/minicnn/cuda_native/gpu_lowering_activation.py` | ~0.2k | Activation lowering helpers extracted from the main lowering module. | Leave as-is unless activation families expand again. |
| `src/minicnn/cuda_native/gpu_lowering_registry_build.py` | ~0.1k | Lowering registry builder and linear lowering extracted from the main lowering module. | Leave as-is unless registry construction grows materially. |
| `src/minicnn/cuda_native/gpu_training_norm.py` | ~0.5k | BatchNorm and GroupNorm helpers after LayerNorm-family extraction. | Leave as-is unless BatchNorm/GroupNorm logic grows materially. |
| `src/minicnn/cuda_native/gpu_training_layernorm.py` | ~0.5k | Dedicated LayerNorm/LayerNorm2d helper module extracted from `gpu_training_norm.py`. | Keep isolated so future LayerNorm-family growth does not re-inflate `gpu_training_norm.py`. |
| `src/minicnn/cuda_native/gpu_training_depthwise.py` | ~0.3k | Base depthwise + LayerNorm2d + Linear helper plus bridge-family compatibility re-exports. | Leave as-is unless another non-bridge depthwise family is added. |

## Extracted production modules

| File | Role |
| --- | --- |
| `src/minicnn/unified/_cuda_native_context.py` | Training context dataclass. |
| `src/minicnn/unified/_cuda_native_diagnostics.py` | Optimizer runtime snapshot, hotspot profiling, and hotspot diff summaries. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch.py` | Thin gpu-native batch dispatch router. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch_common.py` | Shared gpu-native batch finalize/runtime merge helper. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch_linear.py` | Linear/pool gpu-native batch dispatch helpers. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch_norm.py` | Normalization-family gpu-native batch dispatch helpers. |
| `src/minicnn/unified/_cuda_native_gpu_train_dispatch_conv.py` | Conv/depthwise gpu-native batch dispatch helpers. |
| `src/minicnn/unified/_cuda_native_training_loop.py` | Runtime epoch loop and training orchestration. |
| `src/minicnn/unified/_cuda_native_training_plan.py` | GPU-native training plan selection, validation, and runtime counter merge helpers. |
| `src/minicnn/cuda_native/gpu_training_types.py` | GPU training result dataclasses. |
| `src/minicnn/cuda_native/gpu_training_common.py` | Shared CUDA binding, loss, and gradient-clip helpers. |
| `src/minicnn/cuda_native/gpu_training_linear.py` | Linear and two-linear GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_pool.py` | Pool/avgpool/global-avgpool GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_norm.py` | BatchNorm/GroupNorm GPU training helpers plus compatibility re-exports for LayerNorm-family helpers. |
| `src/minicnn/cuda_native/gpu_training_layernorm.py` | LayerNorm/LayerNorm2d GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_conv.py` | Conv-family GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_depthwise.py` | Base depthwise + LayerNorm2d helper plus compatibility re-exports. |
| `src/minicnn/cuda_native/gpu_training_depthwise_bridge.py` | Compatibility facade for bridge-family depthwise helpers. |
| `src/minicnn/cuda_native/gpu_training_depthwise_pointwise.py` | Single-pointwise depthwise bridge GPU training helper. |
| `src/minicnn/cuda_native/gpu_training_depthwise_activation.py` | Two-pointwise + activation depthwise bridge GPU training helper. |
| `src/minicnn/cuda_native/gpu_lowering_registry.py` | GPU lowering registry/context/spec types. |
| `src/minicnn/cuda_native/gpu_lowering_registry_build.py` | Default lowering registry builder and linear lowering helper. |
| `src/minicnn/cuda_native/gpu_lowering_utils.py` | Shared GPU lowering tensor helpers. |
| `src/minicnn/cuda_native/gpu_lowering_norm.py` | BatchNorm/LayerNorm/LayerNorm2d/GroupNorm GPU lowering helpers. |
| `src/minicnn/cuda_native/gpu_lowering_shape.py` | Flatten/identity alias lowering helpers. |
| `src/minicnn/cuda_native/gpu_lowering_merge.py` | Add/Concat lowering helpers. |
| `src/minicnn/cuda_native/gpu_lowering_activation.py` | Activation lowering helpers. |
| `src/minicnn/cuda_native/gpu_lowering_conv.py` | Conv/pool lowering helpers. |

## Files intentionally deprioritized

Large test modules are acceptable for now because they encode regression coverage for CUDA-native behavior and are less risky than splitting production runtime code during GPU enablement.

| File | Approx. lines | Reason |
| --- | ---: | --- |
| `tests/test_cuda_native_gpu_training.py` | ~2.7k | CUDA-native training parity and regression coverage. |
| `tests/test_regressions_cli_runtime.py` | ~2.1k | CLI/runtime regression matrix coverage. |

## Completed cleanup in this pass

- Reduced `src/minicnn/cuda_native/gpu_training.py` from ~3.9k lines before cleanup to a thin compatibility facade, with depthwise-family helpers extracted to `gpu_training_depthwise.py`.
- Reduced `src/minicnn/cuda_native/gpu_training_norm.py` to ~500 lines by extracting LayerNorm-family helpers into `gpu_training_layernorm.py`.
- Reduced `src/minicnn/cuda_native/gpu_training_depthwise.py` to ~300 lines by extracting pointwise bridge-family helpers into `gpu_training_depthwise_bridge.py`.
- Reduced `src/minicnn/cuda_native/gpu_training_depthwise_bridge.py` to a tiny compatibility facade by extracting single-pointwise and two-pointwise+activation implementations into dedicated modules.
- Reduced `src/minicnn/cuda_native/gpu_lowering.py` to a thin facade by extracting normalization, shape, merge, activation, conv/pool, and registry-builder logic into dedicated modules.
- Reduced `src/minicnn/unified/_cuda_native_training_loop.py` to ~400 lines by extracting gpu-native per-plan batch dispatch into `src/minicnn/unified/_cuda_native_gpu_train_dispatch.py`.
- Reduced `src/minicnn/unified/_cuda_native_gpu_train_dispatch.py` to a thin router by extracting linear/pool, normalization, conv/depthwise, and shared finalize helpers into dedicated modules.
- Reduced `src/minicnn/unified/_cuda_native_runtime.py` to a small prepare/finalize facade by extracting context, diagnostics, training-loop, and training-plan modules.
- Kept public import surfaces compatible for existing callers.
