# Dual Backend Guide

This guide explains how one config can target two execution paths:

- `engine.backend: torch`
- `engine.backend: cuda_legacy`

## Same config, different backend

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

## Validate before running handcrafted CUDA

```bash
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
```

## Inspect the mapping into the legacy CUDA trainer

```bash
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
```

## Training implementation layout

The public command stays `minicnn train-dual`, but the legacy trainers are split
internally:

- `src/minicnn/training/train_cuda.py`: CUDA legacy orchestration for data,
  epochs, validation, checkpointing, LR reduction, early stop, and final test.
- `src/minicnn/training/cuda_batch.py`: one CUDA batch of conv forward, FC
  forward, fused loss/accuracy, FC update, and conv backward/update.
- `src/minicnn/training/train_torch_baseline.py`: Torch baseline runtime,
  batch preparation, one-step training, epoch loop, checkpointing, and final
  evaluation.
- `src/minicnn/training/loop.py`: shared metrics, LR state, best/plateau state,
  epoch timing, LR plateau reduction, and epoch summary formatting.
- `src/minicnn/training/legacy_data.py`: shared CIFAR-10 load/normalize helper
  for CUDA legacy and Torch baseline.

## Changing network architecture

The two backends use separate config keys — no Python file changes are needed for either.

### CUDA backend

Edit `model.conv_layers` in `configs/train_cuda.yaml`. Each entry is
`{out_c: <channels>, pool: <bool>}`. `CudaNetGeometry` derives every buffer
size and shape from this list at startup.

```yaml
# Add a stage: append an entry
conv_layers:
  - {out_c: 32, pool: false}
  - {out_c: 32, pool: true}
  - {out_c: 64, pool: false}
  - {out_c: 64, pool: true}
  - {out_c: 128, pool: false}   # ← new stage
  - {out_c: 128, pool: true}    # ← new stage + pool
```

Validate after editing:

```bash
minicnn validate-dual-config --config configs/train_cuda.yaml
```

### Torch / Flex backend

Edit `model.layers` in `configs/dual_backend_cnn.yaml` (or your own config).
Add, remove, or reorder entries freely — `in_channels` and `in_features` are
inferred automatically by the flex builder.

## Custom components

Custom dotted-path components are intended for the Torch backend. The CUDA backend requires supported layer shapes and semantics listed in `src/minicnn/unified/cuda_legacy.py`.
