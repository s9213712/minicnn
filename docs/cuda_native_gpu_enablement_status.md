# cuda_native GPU Enablement Status

Last updated: 2026-04-25

This is the current closure/status note for moving `cuda_native` from a
NumPy-reference-only backend toward a real `gpu_native` training backend.

## Repo-side status

Repo-side implementation for the first `gpu_native` training tier is complete.

Completed:

- explicit `reference_numpy` vs `gpu_native` execution-mode contract
- native device-pointer runtime substrate and CUDA runtime preflight
- native forward lowering for the bootstrap op set
- native training helpers for the current narrow training subsets
- CLI validation/runtime routing for supported `gpu_native` subsets
- hermetic reference-math parity matrix
- real-data CIFAR-10 smoke entrypoints for linear and repeated-conv native GPU training
- docs and capability payloads aligned with the implemented surface
- native `MSELoss` and `BCEWithLogitsLoss` loss-gradient helpers for Linear
  `gpu_native` training subsets
- native `Adam`, `AdamW`, and `RMSprop` update helpers for Linear
  `gpu_native` training subsets
  subsets

## Current `gpu_native` training subsets

Supported through native GPU helper paths:

- `Linear`
- `Flatten -> Linear`
- `Linear -> ReLU -> Linear`
- `Flatten -> Linear -> ReLU -> Linear`
- `MaxPool2d -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> ReLU -> Flatten -> Linear`
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

## Remaining hard blocker

The local machine still fails real CUDA execution with:

```text
CUDA driver version is insufficient for CUDA runtime version
```

Because of that environment issue, real-hardware GPU smoke cannot be certified
from this machine. The repo now fails before allocation with a Python
`RuntimeError` instead of aborting inside `cudaMalloc`.

## Validation evidence

Current repo-side validation:

```text
129 passed
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
