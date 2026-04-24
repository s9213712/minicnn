# cuda_native GPU Parity Matrix

Last updated: 2026-04-25

This matrix records the current `engine.execution_mode=gpu_native` training
surface. It covers hermetic reference-math parity for the narrow native GPU
training helpers. Real-hardware parity remains pending until the local CUDA
driver/runtime mismatch is resolved.

## Current training subset matrix

| Subset | Helper | Evidence | Hardware status |
|---|---|---|---|
| `Linear` | `native_gpu_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Flatten -> Linear` | `native_gpu_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Linear -> ReLU -> Linear` | `native_gpu_two_linear_relu_training_step` | Hermetic reference math | Pending real GPU run |
| `Flatten -> Linear -> ReLU -> Linear` | `native_gpu_two_linear_relu_training_step` | Hermetic reference math | Pending real GPU run |
| `MaxPool2d -> Flatten -> Linear` | `native_gpu_pool_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> ReLU -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> MaxPool2d -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |

## Current real-hardware blocker

The local machine reaches the native CUDA shared library but fails CUDA runtime
preflight with:

```text
CUDA driver version is insufficient for CUDA runtime version
```

Until that environment issue is fixed, `gpu_native` can be regression-tested
through hermetic fake-library parity, but not certified as real-hardware parity
complete.
