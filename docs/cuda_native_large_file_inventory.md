# CUDA Native Large File Inventory

Last updated: 2026-04-25

This note tracks large MiniCNN files that can slow review and maintenance as `cuda_native` moves from experimental toward beta/stable-ready GPU-native execution.

## Current production-code candidates

| File | Approx. lines | Status | Suggested next action |
| --- | ---: | --- | --- |
| `src/minicnn/cuda_native/gpu_lowering.py` | ~980 | Partially split. Registry types and shared tensor helpers were moved out. | Split lowering implementations by op family once per-op coverage stabilizes. |
| `src/minicnn/cuda_native/gpu_training.py` | ~900 | Compatibility import surface plus remaining depthwise-family helpers. Linear, pool, norm, conv, result types, and shared helpers were moved out. | Move remaining depthwise-family helpers into a dedicated module, then leave this file as a thin compatibility facade. |
| `src/minicnn/unified/_cuda_native_training_loop.py` | ~800 | Extracted from `_cuda_native_runtime.py`; owns epoch loop, fallback training path, metrics, and profiling orchestration. GPU-native plan selection moved out. | Split epoch metrics construction only if this file grows again. |

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
| `src/minicnn/cuda_native/gpu_training_norm.py` | BatchNorm/LayerNorm/GroupNorm GPU training helpers. |
| `src/minicnn/cuda_native/gpu_training_conv.py` | Conv-family GPU training helpers. |
| `src/minicnn/cuda_native/gpu_lowering_registry.py` | GPU lowering registry/context/spec types. |
| `src/minicnn/cuda_native/gpu_lowering_utils.py` | Shared GPU lowering tensor helpers. |

## Files intentionally deprioritized

Large test modules are acceptable for now because they encode regression coverage for CUDA-native behavior and are less risky than splitting production runtime code during GPU enablement.

| File | Approx. lines | Reason |
| --- | ---: | --- |
| `tests/test_cuda_native_gpu_training.py` | ~2.7k | CUDA-native training parity and regression coverage. |
| `tests/test_regressions_cli_runtime.py` | ~2.1k | CLI/runtime regression matrix coverage. |

## Completed cleanup in this pass

- Reduced `src/minicnn/cuda_native/gpu_training.py` from ~3.9k lines before cleanup to ~900 lines.
- Reduced `src/minicnn/cuda_native/gpu_lowering.py` to below 1k lines.
- Reduced `src/minicnn/unified/_cuda_native_runtime.py` to a small prepare/finalize facade by extracting context, diagnostics, training-loop, and training-plan modules.
- Kept public import surfaces compatible for existing callers.
