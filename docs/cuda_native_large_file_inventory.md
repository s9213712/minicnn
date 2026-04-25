# CUDA Native Large File Inventory

Last updated: 2026-04-25

This note tracks large MiniCNN files that can slow review and maintenance as `cuda_native` moves from experimental toward beta/stable-ready GPU-native execution.

## Current production-code candidates

| File | Approx. lines | Status | Suggested next action |
| --- | ---: | --- | --- |
| `src/minicnn/cuda_native/gpu_training.py` | ~3.2k | Partially split. Public imports remain compatible. Dataclasses, shared CUDA helpers, and conv-family helpers were moved out. | Continue extracting optimizer-state handling and MLP/linear-family helpers. |
| `src/minicnn/unified/_cuda_native_runtime.py` | ~1.6k | Large orchestration file for support checks, execution selection, fallback, training, evaluation, and metrics. | Split into runtime planning, training dispatch, evaluation dispatch, and diagnostics modules. |
| `src/minicnn/cuda_native/gpu_lowering.py` | ~1.1k | Growing per-op lowering shim. | Split by op family once per-op coverage stabilizes: convolution/pooling, activation/loss, optimizer/update, tensor utilities. |

## Files intentionally deprioritized

Large test modules are acceptable for now because they encode regression coverage for CUDA-native behavior and are less risky than splitting production runtime code during GPU enablement.

| File | Approx. lines | Reason |
| --- | ---: | --- |
| `tests/test_cuda_native_gpu_training.py` | ~2.7k | CUDA-native training parity and regression coverage. |
| `tests/test_regressions_cli_runtime.py` | ~2.1k | CLI/runtime regression matrix coverage. |

## Completed cleanup in this pass

- Added `src/minicnn/cuda_native/gpu_training_types.py` for result dataclasses.
- Added `src/minicnn/cuda_native/gpu_training_common.py` for shared CUDA binding and loss/clip helpers.
- Added `src/minicnn/cuda_native/gpu_training_conv.py` for conv-family GPU training helpers.
- Kept `src/minicnn/cuda_native/gpu_training.py` as the compatibility import surface while further extraction continues.
