# Dual Backend Guide

This guide explains how one config can target two execution paths:

- `engine.backend: torch`
- `engine.backend: cuda_legacy`

## Same config, different backend

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

For reproducible backend comparisons, keep `train.init_seed`, `dataset.seed`, and `train.seed` fixed. The torch/flex path seeds PyTorch before model construction, and CUDA legacy maps the same field into its legacy experiment config.

## Validate before running handcrafted CUDA

```bash
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
```

## Inspect the mapping into the legacy CUDA trainer

```bash
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
```

To compare native variants:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

In-process variant switching resets the cached native library handle, so scripted comparisons should load the requested `.so` for each run.

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

Dotted CLI overrides can patch a layer entry directly:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=torch model.layers.1.negative_slope=0.05
```

Boolean fields are parsed strictly, so string `"false"` is treated as false rather than as a truthy Python string.

## When Architecture Changes Require Code Changes

Most architecture edits should be YAML-only. Code changes are needed only when
the requested layer behavior is outside the supported backend contract.

| Change | Torch / Flex backend | CUDA legacy backend |
|---|---|---|
| Change layer count, channel width, pool placement, or classifier output within supported layers | Edit `model.layers`; PyTorch autograd handles backward. | Edit `model.conv_layers`; `CudaNetGeometry`, `BatchWorkspace`, and `cuda_batch.py` derive shapes from config. |
| Add a built-in PyTorch layer already registered by MiniCNN | Edit `model.layers`; no backward file changes. | Not supported unless the CUDA subset already maps that layer. |
| Add a new Torch-only layer or block | Add/register it in `src/minicnn/flex/builder.py` or `src/minicnn/flex/registry.py`; backward remains PyTorch autograd. | No change unless this layer must also run on CUDA legacy. |
| Add a new CUDA legacy layer type | Usually no Torch change unless config parity is required. | Update `src/minicnn/unified/cuda_legacy.py`, `src/minicnn/training/cuda_arch.py`, `src/minicnn/training/cuda_workspace.py`, `src/minicnn/training/cuda_batch.py`, native kernels under `cpp/src/`, and ctypes signatures in `src/minicnn/core/cuda_backend.py`. |
| Change backward math for an existing op | Usually no project code change; PyTorch autograd owns the gradient. | Update the relevant native backward kernel in `cpp/src/`, then update `src/minicnn/core/cuda_backend.py` and the call order in `src/minicnn/training/cuda_batch.py` if the ABI or buffers changed. |
| Change optimizer/update semantics | Update torch optimizer config or `src/minicnn/flex/trainer.py` for flex training. | Update `cpp/src/optimizer.cu` and the CUDA update calls in `src/minicnn/training/cuda_batch.py`. |

Malformed CUDA legacy config values are collected as validation errors before the legacy experiment is compiled. If validation fails, fix the YAML or CLI override first rather than editing native code.

Native CUDA backward files by operation:

| Operation | Native file | Python call site |
|---|---|---|
| Convolution backward | `cpp/src/conv_backward.cu` | `src/minicnn/training/cuda_batch.py` |
| Dense backward | `cpp/src/dense_layer.cu` | `src/minicnn/training/cuda_batch.py` |
| MaxPool backward | `cpp/src/maxpool_backward_use_idx.cu` or `cpp/src/maxpool_backward_nchw.cu` | `src/minicnn/training/cuda_batch.py` |
| LeakyReLU backward | `cpp/src/leaky_relu.cu` | `src/minicnn/training/cuda_batch.py` |
| Softmax / cross-entropy backward | `cpp/src/loss_layer.cu` | `src/minicnn/training/cuda_batch.py` |
| LayerNorm backward | `cpp/src/layer_norm.cu` | Add a Python call site only if LayerNorm becomes part of CUDA legacy training. |

`cpp/src/layer_norm.cu` is currently a tested native kernel asset, not a
supported `cuda_legacy` training layer. `tests/test_layer_norm.py` mirrors the
kernel math in NumPy and checks it against PyTorch. Wiring it into training
would still require config validation, ctypes bindings, workspace buffers, and
call-site integration.

Rule of thumb:

- If the change can be expressed with current `model.layers` or
  `model.conv_layers`, change YAML and validate.
- If a new CUDA op appears in the training graph, update native forward,
  native backward, ctypes bindings, CUDA workspace allocation, and
  `cuda_batch.py`.
- If only Torch needs the op, keep it in the flex builder/registry and let
  PyTorch autograd handle backward.

## Custom components

Custom dotted-path components are intended for the Torch backend. The CUDA backend requires supported layer shapes and semantics listed in `src/minicnn/unified/cuda_legacy.py`.

`maxpool_backward_nchw.cu` exports both the old void ABI and `maxpool_backward_nchw_status(...)`. Prefer the status-returning form from Python wrappers so invalid geometry becomes a catchable host-side error.
