# cuda_native GPU Parity Matrix

Last updated: 2026-04-25

This matrix records the current `engine.execution_mode=gpu_native` training
surface. It covers hermetic reference-math parity for the narrow native GPU
training helpers plus representative real-hardware smoke evidence.

For the closure/status summary, see
[cuda_native_gpu_enablement_status.md](cuda_native_gpu_enablement_status.md).

## Current training subset matrix

`AvgPool2d`, `BatchNorm2d`, `DepthwiseConv2d`, `PointwiseConv2d`, `GroupNorm`, `LayerNorm2d`, `Identity`,
`Dropout(p=0)`, `DropPath(p=0)`, `GELU`, `SiLU`, `Sigmoid`, and `Tanh` are covered as
`gpu_native` forward dispatch primitives. `AvgPool2d(kernel_size=2,stride=2,padding=0)`
and `BatchNorm2d -> Flatten -> Linear` are also covered in the helper-backed
train-native subset matrix. `DepthwiseConv2d` now has helper-backed
train-native coverage for the same SGD/CrossEntropyLoss envelope as the narrow
conv-family helpers. `DepthwiseConv2d -> LayerNorm2d -> Flatten -> Linear` is
covered as the first ConvNeXt-style bridge subset, and
`DepthwiseConv2d -> LayerNorm2d -> PointwiseConv2d -> Flatten -> Linear`
extends that bridge through a native pointwise im2col/GEMM forward/backward.
`DepthwiseConv2d -> LayerNorm2d -> PointwiseConv2d -> GELU -> PointwiseConv2d
-> Flatten -> Linear` is the deepest current ConvNeXt-style helper-backed
training subset.

| Subset | Helper | Evidence | Hardware status |
|---|---|---|---|
| `Linear` | `native_gpu_linear_training_step` | Hermetic reference math for CE/MSE/BCE, label smoothing, SGD/Adam/AdamW/RMSprop, global grad clip, grad accumulation, and execution trace order | Real CUDA smoke passed for SGD, RMSprop, and label smoothing; grad accumulation allocation smoke pending on a compatible host |
| `Flatten -> Linear` | `native_gpu_linear_training_step` | Hermetic reference math for CE/MSE/BCE, label smoothing, SGD/Adam/AdamW/RMSprop, global grad clip, grad accumulation, and execution trace order | Covered by Linear helper smoke; full CLI smoke pending |
| `Linear -> ReLU -> Linear` | `native_gpu_two_linear_relu_training_step` | Hermetic reference math | Pending real GPU run |
| `Flatten -> Linear -> ReLU -> Linear` | `native_gpu_two_linear_relu_training_step` | Hermetic reference math | Pending real GPU run |
| `Linear -> GELU/SiLU/Sigmoid/Tanh -> Linear` | `native_gpu_two_linear_relu_training_step` | Hermetic reference math | Pending real GPU run |
| `Flatten -> Linear -> GELU/SiLU/Sigmoid/Tanh -> Linear` | `native_gpu_two_linear_relu_training_step` | Covered by two-linear activation helper math | Pending real GPU run |
| `MaxPool2d -> Flatten -> Linear` | `native_gpu_pool_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `AvgPool2d(kernel_size=2,stride=2,padding=0) -> Flatten -> Linear` | `native_gpu_avgpool_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `BatchNorm2d -> Flatten -> Linear` | `native_gpu_batchnorm_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `LayerNorm2d -> Flatten -> Linear` | `native_gpu_layernorm2d_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `GroupNorm -> Flatten -> Linear` | `native_gpu_groupnorm_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `GlobalAvgPool2d -> Flatten -> Linear` | `native_gpu_global_avgpool_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `AdaptiveAvgPool2d(output_size=1) -> Flatten -> Linear` | `native_gpu_global_avgpool_linear_training_step` | Covered by GlobalAvgPool helper math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `Conv2d(valid, bias=false) -> ReLU -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `PointwiseConv2d(bias=false) -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Covered by Conv2d helper math | Pending real GPU run |
| `PointwiseConv2d(bias=false) -> ReLU -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Covered by Conv2d+ReLU helper math | Pending real GPU run |
| `DepthwiseConv2d(bias=false) -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `DepthwiseConv2d(bias=false) -> ReLU -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Covered by depthwise helper routing | Pending real GPU run |
| `DepthwiseConv2d(bias=false) -> LayerNorm2d -> Flatten -> Linear` | `native_gpu_depthwise_layernorm2d_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `DepthwiseConv2d(bias=false) -> LayerNorm2d -> PointwiseConv2d(bias=false) -> Flatten -> Linear` | `native_gpu_depthwise_layernorm2d_pointwise_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `DepthwiseConv2d(bias=false) -> LayerNorm2d -> PointwiseConv2d(bias=false) -> GELU -> PointwiseConv2d(bias=false) -> Flatten -> Linear` | `native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step` | Hermetic reference math | Pending real GPU run |
| `DepthwiseConv2d(bias=false) -> MaxPool2d -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Covered by depthwise helper routing | Pending real GPU run |
| `DepthwiseConv2d(bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear` | `native_gpu_conv_linear_training_step` | Covered by depthwise helper routing | Pending real GPU run |
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
- ConvNeXt-style bridge CIFAR-10 smoke entrypoint exists; on the current host it
  stops at CUDA preflight with status 35 because the installed driver is older
  than the CUDA runtime selected by the native library

This is not yet a claim of full graph-level GPU backward generalization.

## Real-data smoke entrypoint

Once CUDA driver/runtime compatibility is restored, run:

```bash
PYTHONPATH=src python3 examples/cuda_native_gpu_two_conv_training_cifar10_demo.py --batch-size 2
PYTHONPATH=src python3 examples/cuda_native_gpu_convnext_bridge_training_cifar10_demo.py --batch-size 2
```

These scripts run representative `gpu_native` training helpers on CIFAR-10
batches and compare updated weights against NumPy reference steps.
