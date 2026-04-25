# cuda_native GPU Parity Matrix

Last updated: 2026-04-25

This matrix records the current `engine.execution_mode=gpu_native` training
surface. It covers hermetic reference-math parity for the narrow native GPU
training helpers plus representative real-hardware smoke evidence.

For the closure/status summary, see
[cuda_native_gpu_enablement_status.md](cuda_native_gpu_enablement_status.md).

## Current training subset matrix

`BatchNorm2d`, `GlobalAvgPool2d`, `AdaptiveAvgPool2d(output_size=1)`, `GELU`,
`SiLU`, `Sigmoid`, and `Tanh` are covered as `gpu_native` forward dispatch
primitives, but they are not yet part of the helper-backed train-native subset
matrix.

| Subset | Helper | Evidence | Hardware status |
|---|---|---|---|
| `Linear` | `native_gpu_linear_training_step` | Hermetic reference math for CE/MSE/BCE, label smoothing, SGD/Adam/AdamW/RMSprop, global grad clip, grad accumulation, and execution trace order | Real CUDA smoke passed for SGD, RMSprop, and label smoothing; grad accumulation allocation smoke pending on a compatible host |
| `Flatten -> Linear` | `native_gpu_linear_training_step` | Hermetic reference math for CE/MSE/BCE, label smoothing, SGD/Adam/AdamW/RMSprop, global grad clip, grad accumulation, and execution trace order | Covered by Linear helper smoke; full CLI smoke pending |
| `Linear -> ReLU -> Linear` | `native_gpu_two_linear_relu_training_step` | Hermetic reference math | Pending real GPU run |
| `Flatten -> Linear -> ReLU -> Linear` | `native_gpu_two_linear_relu_training_step` | Hermetic reference math | Pending real GPU run |
| `MaxPool2d -> Flatten -> Linear` | `native_gpu_pool_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> ReLU -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> MaxPool2d -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> ReLU -> Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear` | `native_gpu_two_conv_relu_pool_linear_training_step` | Hermetic reference math | Real CIFAR-10 CUDA smoke passed |

## Current real-hardware status

Representative real CUDA smoke now passes on this machine:

- minimal Linear SGD smoke emits `gpu_native_train:dense_forward`,
  `softmax_xent_grad_loss_acc`, `dense_backward_full`, and `apply_sgd_update`
- minimal Linear RMSprop smoke emits `gpu_native_train:rmsprop_update_fused`
- minimal Linear label-smoothing smoke emits
  `gpu_native_train:softmax_xent_smooth_grad_loss_acc`
- minimal Linear `grad_accum_steps=2` validation keeps
  `engine.execution_mode=gpu_native`; allocation smoke is pending on a
  CUDA driver/runtime-compatible host
- minimal Linear global grad-clip smoke emits `gpu_native_train:grad_clip_global`
  and clips the reported gradient norm to the requested threshold
- minimal Conv+Linear global grad-clip smoke emits `gpu_native_train:grad_clip_global`
  and clips the combined Conv/Linear gradient norm to the requested threshold
- CIFAR-10 repeated-Conv smoke uses `official:cifar10:test_batch` and matches
  NumPy reference updated weights with max absolute diffs around `1e-9`

This is not yet a claim of full graph-level GPU backward generalization.

## Real-data smoke entrypoint

Once CUDA driver/runtime compatibility is restored, run:

```bash
PYTHONPATH=src python3 examples/cuda_native_gpu_two_conv_training_cifar10_demo.py --batch-size 2
```

The script runs the repeated-Conv `gpu_native` training helper on a CIFAR-10
batch and compares updated weights against a NumPy reference step.
