# Dual Backend Guide

This guide explains how the shared dual-backend config works on the current
branch.

The stable shared toggle is:

- `engine.backend: torch`
- `engine.backend: cuda_legacy`

## Same Frontend, Different Backend Boundary

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

The important nuance is:

- both commands read the same top-level config shape
- torch can use far more of that frontend surface
- `cuda_legacy` only accepts a narrow validated subset

## Validate Before Running CUDA Legacy

```bash
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
```

If validation fails, fix the config first. Do not assume the native backend is
supposed to accept the same full surface as torch.

## Inspect The Mapping

```bash
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
```

This is the fastest way to see how the shared config is compiled into the
legacy experiment config used by the handwritten CUDA trainer.

## What The Shared Config Actually Uses

For the public dual-backend path, architecture is described through
`model.layers[]`.

Torch consumes that list directly through the flex builder.

`cuda_legacy` does not. It validates the list against one fixed pattern and
then compiles the result into the older stage-oriented experiment config.

That means the same YAML key exists on both sides, but the accepted semantic
surface is different.

## Current `cuda_legacy` Contract

On this branch, `cuda_legacy` accepts:

- dataset type `cifar10`
- input shape `[3, 32, 32]`
- exactly four `Conv2d` layers
- `ReLU` or `LeakyReLU`
- exactly two `MaxPool2d` layers in fixed positions
- final `Flatten -> Linear`

This is why a config may be valid for torch but invalid for `cuda_legacy`.

## Variant Selection

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

In-process variant switching resets the cached native handle so repeated scripted
comparisons still load the requested `.so`.

## Changing Architecture

### Torch

Most torch-side architecture edits are YAML-only:

- add/remove/reorder `model.layers[]`
- use registered layers or torch module names
- use dotted-path custom components

PyTorch autograd owns backward for those components.

### CUDA Legacy

Most `cuda_legacy` architecture edits are only YAML-only if they stay inside the
existing validated pattern.

Examples of safe edits:

- channel widths
- `ReLU` vs `LeakyReLU`
- classifier output width, if still compatible with the fixed dataset/output contract

Examples that are not YAML-only:

- new layer types
- different pool structure
- branching/residual topology
- new normalization ops inside the training graph

Those require real backend work across:

- `src/minicnn/unified/cuda_legacy.py`
- `src/minicnn/training/`
- `src/minicnn/core/cuda_backend.py`
- `cpp/src/`

## Change-Impact Table

| Change | Torch path | CUDA legacy path |
|---|---|---|
| Change `Conv2d` widths or linear output inside supported shapes | YAML-only | YAML-only if still inside validator boundary |
| Reorder or add arbitrary layers | Usually YAML-only | Not supported without backend work |
| Add a torch-only custom block | Add/import it for flex | No effect unless CUDA must also support it |
| Add a new native training op | No torch change unless frontend parity is desired | Update validator, mapping, workspace, ctypes, and native kernels |
| Change native backward math | Usually no project change | Update native backward kernel and matching Python call sites |
| Change optimizer semantics | Torch optimizer/config changes | Native optimizer and CUDA batch path changes |

## Legacy Training Layout

The public command is still `minicnn train-dual`, but the legacy backend is
split internally:

- `src/minicnn/training/train_cuda.py`: orchestration
- `src/minicnn/training/cuda_batch.py`: per-batch CUDA forward/backward/update
- `src/minicnn/training/legacy_data.py`: CIFAR-10 load/normalize
- `src/minicnn/training/loop.py`: shared metrics and LR/early-stop helpers

## Custom Components

Custom dotted-path components are a torch-side feature.

They do not automatically become legal for `cuda_legacy`. If CUDA needs the
same capability, it must be added through validation, mapping, and native
execution support.
