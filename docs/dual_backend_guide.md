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
