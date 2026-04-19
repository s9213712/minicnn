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

## Custom components

Custom dotted-path components are intended for the Torch backend. The CUDA backend requires supported layer shapes and semantics listed in `src/minicnn/unified/cuda_legacy.py`.
