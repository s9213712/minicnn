# ConvNeXt Explicit Path Delivery Summary

This document summarizes the explicit ConvNeXt primitive path that was added and
stabilized for MiniCNN, and the later `cuda_native` primitive expansion that
made a minimal native smoke path possible.

## What changed

### Explicit primitives

MiniCNN now exposes these torch/flex primitives for a ConvNeXt-like frontend:

- `DepthwiseConv2d` / `depthwise_conv2d`
- `PointwiseConv2d` / `pointwise_conv2d`
- `LayerNorm2d` / `layernorm2d`
- `ConvNeXtBlock` / `convnext_block`

The `ConvNeXtBlock` implementation was aligned to use channel-first
`LayerNorm2d` on NCHW tensors instead of relying on an implicit NHWC permutation
contract.

### Templates

- `templates/cifar10/convnext_like.yaml`
  - block-based ConvNeXt-like path
- `templates/cifar10/convnext_tiny_cuda_native_smoke.yaml`
  - smallest built-in cuda_native smoke config for the block-based named-model path
- `templates/cifar10/convnext_explicit.yaml`
  - explicit primitive path for direct editing
  - sequential primitive stack only; does not encode residual add / layer scale
- `templates/cifar10/convnext_explicit_smoke.yaml`
  - smallest built-in smoke training config
- `templates/cifar10/convnext_explicit_cuda_native_smoke.yaml`
  - smallest built-in cuda_native smoke config for the explicit primitive path

### Diagnostics and CLI UX

- `minicnn smoke` now reports:
  - `torch_available`
  - `cuda_available`
  - `native_available`
  - `flex_registry_ready`
- dataset split failures now use a structured user-facing format with:
  - `Cause`
  - `Fix`
  - `Example`

## Validation completed

### Smoke / diagnostics

- `minicnn smoke --format json`
  - confirmed `flex_registry_ready=true`
  - confirmed snake_case registry aliases appear in the registry surface
  - warning remained limited to missing native artifacts

### Training path

- `minicnn train-flex --config templates/cifar10/convnext_explicit.yaml ...`
  - completed a 1-epoch reduced-data training run
- `minicnn train-flex --config templates/cifar10/convnext_explicit_smoke.yaml`
  - completed the built-in smoke config run
- `templates/cifar10/convnext_explicit_cuda_native_smoke.yaml`
  - defines the hermetic cuda_native smoke path for explicit primitives
- `templates/cifar10/convnext_tiny_cuda_native_smoke.yaml`
  - defines the hermetic cuda_native smoke path for the block-based named model

The canonical smoke contract lives in:

- `docs/convnext_explicit_smoke_index.md`

### Tests

Validated coverage now includes:

- template parse + forward coverage
- primitive registry coverage
- `show-model`
- `show-graph`
- `validate-config`
- CLI training integration smoke
- dataset split CLI error formatting

## Files added or updated

### Code

- `src/minicnn/flex/components.py`
- `src/minicnn/flex/builder.py`
- `src/minicnn/_cli_readonly.py`
- `src/minicnn/_cli_errors.py`
- `src/minicnn/user_errors.py`
- `src/minicnn/flex/_datasets.py`
- `src/minicnn/data/mnist.py`
- `src/minicnn/data/cifar10.py`

### Templates

- `templates/cifar10/convnext_like.yaml`
- `templates/cifar10/convnext_explicit.yaml`
- `templates/cifar10/convnext_explicit_smoke.yaml`
- `templates/README.md`

### Docs

- `README.md`
- `USAGE.md`
- `docs/convnext_explicit_smoke_index.md`
- `docs/convnext_explicit_delivery_summary.md`

### Tests

- `tests/test_convnext_components.py`
- `tests/test_smoke_dependency_contract.py`
- `tests/test_convnext_explicit_cli.py`
- `tests/test_convnext_explicit_smoke.py`
- `tests/test_dataset_split_cli_errors.py`
- `tests/test_templates_validity.py`

## Scope boundary

This delivery does not claim:

- `cuda_legacy` support for ConvNeXt primitives
- repo-wide ConvNeXt support beyond the torch/flex experimental slice
- silent fallback from unsupported backends
- `DropPath` / stochastic depth support
