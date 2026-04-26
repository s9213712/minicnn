# CUDA Native Large File Inventory

Last updated: 2026-04-26

This note tracks large MiniCNN files that can slow review and maintenance as `cuda_native` moves from experimental toward beta/stable-ready GPU-native execution.

## Current production-code candidates

| File | Approx. lines | Status | Suggested next action |
| --- | ---: | --- | --- |
| `src/minicnn/unified/_cuda_native_training_loop.py` | ~1.0k | Extracted from `_cuda_native_runtime.py`; owns epoch loop, fallback training path, metrics, and profiling orchestration. GPU-native plan selection moved out. | Split epoch metrics/reporting construction if this file grows further. |
| `src/minicnn/cuda_native/gpu_training_depthwise.py` | ~0.9k | Depthwise bridge-family helpers were extracted from `gpu_training.py`, but the bridge family is still concentrated here. | Split bridge families by topology if this file grows again. |
| `src/minicnn/cuda_native/gpu_lowering.py` | ~0.8k | Lowering registry file after normalization-family extraction. | Next split target is shape/merge or pool-family lowering only if it starts growing again. |
| `src/minicnn/cuda_native/gpu_training_norm.py` | ~0.5k | BatchNorm and GroupNorm helpers after LayerNorm-family extraction. | Leave as-is unless BatchNorm/GroupNorm logic grows materially. |
| `src/minicnn/cuda_native/gpu_training_layernorm.py` | ~0.5k | Dedicated LayerNorm/LayerNorm2d helper module extracted from `gpu_training_norm.py`. | Keep isolated so future LayerNorm-family growth does not re-inflate `gpu_training_norm.py`. |

## Extracted production modules

| File | Role |
| --- | --- |
| `src/minicnn/unified/_cuda_native_context.py` | Training context dataclass. |
| `src/minicnn/unified/_cuda_native_diagnostics.py` | Optimizer runtime snapshot, hotspot profiling, and hotspot diff summaries. |
| `src/minicnn/unified/_cuda_native_training_loop.py` | Runtime epoch loop and training orchestration. |
| `src/minicnn/unified/_cuda_native_training_plan.py` | GPU-native training plan selection, validation, and runtime counter merge helpers. |
| `src/minicnn/cuda_native/gpu_training_types.py` | GPU training result dataclasses. |
| `src/minicnn/cuda_native/gpu_training_common.py` | Shared CUDA binding, loss, and gradient-clip helpers. |
| `src/minicnn/cuda_native/gpu_training_linear.py` | Linear and two-linear GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_pool.py` | Pool/avgpool/global-avgpool GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_norm.py` | BatchNorm/GroupNorm GPU training helpers plus compatibility re-exports for LayerNorm-family helpers. |
| `src/minicnn/cuda_native/gpu_training_layernorm.py` | LayerNorm/LayerNorm2d GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_conv.py` | Conv-family GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_depthwise.py` | Depthwise + LayerNorm2d bridge-family GPU training helpers. |
| `src/minicnn/cuda_native/gpu_lowering_registry.py` | GPU lowering registry/context/spec types. |
| `src/minicnn/cuda_native/gpu_lowering_utils.py` | Shared GPU lowering tensor helpers. |
| `src/minicnn/cuda_native/gpu_lowering_norm.py` | BatchNorm/LayerNorm/LayerNorm2d/GroupNorm GPU lowering helpers. |

## Files intentionally deprioritized

Large test modules are acceptable for now because they encode regression coverage for CUDA-native behavior and are less risky than splitting production runtime code during GPU enablement.

| File | Approx. lines | Reason |
| --- | ---: | --- |
| `tests/test_cuda_native_gpu_training.py` | ~2.7k | CUDA-native training parity and regression coverage. |
| `tests/test_regressions_cli_runtime.py` | ~2.1k | CLI/runtime regression matrix coverage. |

## Completed cleanup in this pass

- Reduced `src/minicnn/cuda_native/gpu_training.py` from ~3.9k lines before cleanup to a thin compatibility facade, with depthwise-family helpers extracted to `gpu_training_depthwise.py`.
- Reduced `src/minicnn/cuda_native/gpu_training_norm.py` to ~500 lines by extracting LayerNorm-family helpers into `gpu_training_layernorm.py`.
- Reduced `src/minicnn/cuda_native/gpu_lowering.py` to ~800 lines by extracting normalization-family lowering into `gpu_lowering_norm.py`.
- Reduced `src/minicnn/unified/_cuda_native_runtime.py` to a small prepare/finalize facade by extracting context, diagnostics, training-loop, and training-plan modules.
- Kept public import surfaces compatible for existing callers.
