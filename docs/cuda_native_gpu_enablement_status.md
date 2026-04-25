# cuda_native GPU Enablement Status

Last updated: 2026-04-25

This is the current closure/status note for moving `cuda_native` from a
NumPy-reference-only backend toward a real `gpu_native` training backend.

## Repo-side status

Repo-side implementation for the first `gpu_native` training tier is complete,
and representative real CUDA smoke now passes on this machine.

Completed:

- explicit `reference_numpy` vs `gpu_native` execution-mode contract
- native device-pointer runtime substrate and CUDA runtime preflight
- native forward lowering for the bootstrap op set
- `BatchNorm2d` forward dispatch/lowering shim in the `gpu_native` bootstrap
  primitive set
- native `BatchNorm2d -> Flatten -> Linear` training helper through
  `bn_train_forward` and `bn_backward`
- `GlobalAvgPool2d` and `AdaptiveAvgPool2d(output_size=1)` forward dispatch
  through the `global_avgpool2d_forward` C ABI shim
- `AvgPool2d` forward/backward dispatch through the `avgpool2d_forward` and
  `avgpool2d_backward` C ABI shims
- `Identity`, `Dropout(p=0)`, and `DropPath(p=0)` forward dispatch as no-op
  GPU aliases
- `GELU`, `SiLU`, `Sigmoid`, and `Tanh` forward dispatch through native
  elementwise activation C ABI shims
- `GELU`, `SiLU`, `Sigmoid`, and `Tanh` backward C ABI shims for modern
  activation train-native helpers
- native `Linear -> GELU/SiLU/Sigmoid/Tanh -> Linear` and
  `Flatten -> Linear -> GELU/SiLU/Sigmoid/Tanh -> Linear` training helpers
- `PointwiseConv2d` forward dispatch through the native Conv2d im2col/GEMM
  lowering path
- `DepthwiseConv2d` forward dispatch through the native
  `depthwise_conv2d_forward` C ABI shim
- `DepthwiseConv2d` backward C ABI shim through
  `depthwise_conv2d_backward`
- native `DepthwiseConv2d -> Flatten -> Linear`,
  `DepthwiseConv2d -> ReLU -> Flatten -> Linear`,
  `DepthwiseConv2d -> MaxPool2d -> Flatten -> Linear`, and
  `DepthwiseConv2d -> ReLU -> MaxPool2d -> Flatten -> Linear` training helper
  routing through the depthwise forward/backward C ABI shims
- `GroupNorm` forward dispatch through the native `groupnorm_forward` C ABI shim
- `LayerNorm2d` forward dispatch through the native `layernorm2d_forward` C ABI
  shim
- `LayerNorm2d` backward C ABI shim through `layernorm2d_backward` as the
  prerequisite for LayerNorm2d helper-backed training subsets
- native training helpers for the current narrow training subsets
- native `GlobalAvgPool2d -> Flatten -> Linear` and
  `AdaptiveAvgPool2d(output_size=1) -> Flatten -> Linear` training helpers
- native `AvgPool2d(kernel_size=2,stride=2,padding=0) -> Flatten -> Linear`
  training helper
- stochastic `Dropout/DropPath` native mask kernels and graph-backward coverage;
  only `p=0` no-op aliases are currently part of the GPU-first dispatch path
- CLI validation/runtime routing for supported `gpu_native` subsets
- hermetic reference-math parity matrix
- readiness diagnostics with a `training_lowering_plan` that breaks helper
  subsets into forward/loss/backward/optimizer lowering steps
- runtime `execution_trace` telemetry that records the actual native
  forward/loss/backward/optimizer calls emitted by helper-backed training steps
- real-data CIFAR-10 smoke entrypoints for linear and repeated-conv native GPU training
- docs and capability payloads aligned with the implemented surface
- native `MSELoss` and `BCEWithLogitsLoss` loss-gradient helpers for Linear
  `gpu_native` training subsets
- native `Adam`, `AdamW`, and `RMSprop` update helpers for Linear
  `gpu_native` training subsets
- native SGD fused update helper for Linear `gpu_native` weight-decay parity
- native global-norm gradient clipping for supported `gpu_native` training subsets
- native CrossEntropyLoss `label_smoothing` helper for supported `gpu_native`
  training subsets
- `train.grad_accum_steps >= 1` for supported `gpu_native` subsets via native
  accumulated-batch training steps

## Current `gpu_native` training subsets

Supported through native GPU helper paths:

- `Linear`
- `Flatten -> Linear`
- `Linear -> ReLU -> Linear`
- `Flatten -> Linear -> ReLU -> Linear`
- `Linear -> GELU/SiLU/Sigmoid/Tanh -> Linear`
- `Flatten -> Linear -> GELU/SiLU/Sigmoid/Tanh -> Linear`
- `MaxPool2d -> Flatten -> Linear`
- `AvgPool2d(kernel_size=2,stride=2,padding=0) -> Flatten -> Linear`
- `BatchNorm2d -> Flatten -> Linear`
- `GlobalAvgPool2d -> Flatten -> Linear`
- `AdaptiveAvgPool2d(output_size=1) -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> ReLU -> Flatten -> Linear`
- `PointwiseConv2d(bias=false) -> Flatten -> Linear`
- `PointwiseConv2d(bias=false) -> ReLU -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> ReLU -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> MaxPool2d -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> MaxPool2d -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> ReLU -> Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear`

These subsets are intentionally narrow. They are the first usable native-GPU
training tier, not a claim of full graph backward generalization.

Loss support:

- Linear subsets: `CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`
- Conv-family subsets: `CrossEntropyLoss`

Optimizer support:

- Linear subsets: `SGD`, `Adam`, `AdamW`, `RMSprop`
- Conv-family subsets: `SGD`

## Explicitly out of scope for this tier

The following remain `reference_numpy`-only for training:

- `ResidualBlock`
- `ConvNeXtBlock`
- `convnext_block`
- arbitrary composite block lowering
- arbitrary ordered-DAG backward lowering

These are not silently treated as GPU-supported. `gpu_native` validation rejects
them outside the supported subsets.

`optimizer.grad_clip_global` is native for supported `gpu_native` subsets through
`grad_l2_sumsq` plus `scale_inplace`.

`train.grad_accum_steps >= 1` is supported for the same `gpu_native` helper
subsets by accumulating microbatches into a single native GPU helper step.
The Linear helper has hermetic mega-batch parity coverage for this path.

## Real-hardware smoke evidence

Current real CUDA evidence:

- minimal Linear SGD smoke passed and emitted native `gpu_native_train:*`
  execution kinds
- minimal Linear RMSprop smoke passed and emitted
  `gpu_native_train:rmsprop_update_fused`
- minimal Linear label-smoothing smoke passed and emitted
  `gpu_native_train:softmax_xent_smooth_grad_loss_acc`
- minimal Linear global grad-clip smoke passed and emitted
  `gpu_native_train:grad_clip_global` plus `gpu_native_train:sgd_update_fused`
- minimal Conv+Linear global grad-clip smoke passed and clipped the combined
  Conv/Linear gradient norm to the requested threshold
- minimal Conv+Linear `weight_decay` smoke passed and emitted
  `gpu_native_train:sgd_update_fused`
- minimal Linear `grad_accum_steps=2` validation now resolves to
  `engine.execution_mode=gpu_native`; real allocation smoke is pending on a
  CUDA driver/runtime-compatible host
- CIFAR-10 repeated-Conv smoke used `official:cifar10:test_batch`; updated
  Conv/Linear weights matched NumPy reference with max absolute diffs around
  `1e-9`

If a future machine fails with `CUDA runtime preflight failed`, the repo still
fails before allocation with a Python `RuntimeError` instead of aborting inside
`cudaMalloc`.

## Remaining structural blockers

Still not claimed as complete:

- full graph-level GPU backward generalization
- composite/block training lowering for residual and ConvNeXt-style models
- broader `BatchNorm2d` graph-level train-native coverage beyond the
  `BatchNorm2d -> Flatten -> Linear` helper subset
- broader modern elementwise activation graph-level train-native coverage beyond
  the two-linear helper subsets
- broader `PointwiseConv2d` graph-level train-native coverage beyond the
  `PointwiseConv2d -> Flatten -> Linear` helper subsets
- `GroupNorm` train-native helper coverage; current work is forward dispatch
  only
- `LayerNorm2d` train-native helper coverage; current work has native
  forward/backward C ABI shims but no helper-backed training subset yet

## Validation evidence

Current repo-side validation:

```text
Current validation count moves as GPU-native coverage expands; the current
cuda_native GPU enablement subset is expected to pass with a small number of
CUDA-runtime-preflight skips on hosts with incompatible driver/runtime pairs.
```

Covered test subset:

- `tests/test_cuda_native_device_runtime.py`
- `tests/test_cuda_native_gpu_dispatch.py`
- `tests/test_cuda_native_gpu_executor.py`
- `tests/test_cuda_native_gpu_training.py`
- `tests/test_cuda_native_execution_mode_contract.py`
- `tests/test_regressions_cli_runtime.py`

## Hardware smoke commands

Run after fixing CUDA driver/runtime compatibility:

```bash
PYTHONPATH=src python3 examples/cuda_native_gpu_linear_training_cifar10_demo.py --batch-size 8
PYTHONPATH=src python3 examples/cuda_native_gpu_two_conv_training_cifar10_demo.py --batch-size 2
```
