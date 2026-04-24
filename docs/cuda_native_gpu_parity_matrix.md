# cuda_native GPU Parity Matrix

Last updated: 2026-04-25

This matrix records the current `engine.execution_mode=gpu_native` training
surface. It covers hermetic reference-math parity for the narrow native GPU
training helpers. Real-hardware parity remains pending until the local CUDA
driver/runtime mismatch is resolved.

For the closure/status summary, see
[cuda_native_gpu_enablement_status.md](cuda_native_gpu_enablement_status.md).

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
| `Conv2d(valid, bias=false) -> ReLU -> Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear` | `native_gpu_two_conv_relu_pool_linear_training_step` | Hermetic reference math | Pending real GPU run |

## Current real-hardware blocker

The local machine reaches the native CUDA shared library but fails CUDA runtime
preflight with:

```text
CUDA driver version is insufficient for CUDA runtime version
```

Until that environment issue is fixed, `gpu_native` can be regression-tested
through hermetic fake-library parity, but not certified as real-hardware parity
complete.

## Real-data smoke entrypoint

Once CUDA driver/runtime compatibility is restored, run:

```bash
PYTHONPATH=src python3 examples/cuda_native_gpu_two_conv_training_cifar10_demo.py --batch-size 2
```

The script runs the repeated-Conv `gpu_native` training helper on a CIFAR-10
batch and compares updated weights against a NumPy reference step.
