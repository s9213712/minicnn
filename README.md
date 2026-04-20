# MiniCNN

[繁體中文 README](README.zh-TW.md)

A dual-backend mini deep learning framework that lets one shared model config drive either:

- **`engine.backend: torch`** for rapid experimentation and custom components
- **`engine.backend: cuda_legacy`** for the handcrafted CUDA CNN path already in this repository

The goal is simple: users should change **one option in the config file** to switch backend, while keeping the same layer definitions, optimizer section, and training parameters.

This repository is the consolidated MiniCNN line. Older exploratory snapshots were removed so this project has one clear development target.

## What is in this repo

- Handcrafted CUDA/C++ backend under `cpp/`
- Config-driven model builder under `src/minicnn/flex/`
- Dual-backend compiler and trainer under `src/minicnn/unified/`
- GitHub-ready project files, CI, docs, examples, and tests

## Quick start

```bash
git clone https://github.com/s9213712/minicnn.git
cd minicnn
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[torch,dev]
pytest
```

## Build the handcrafted CUDA library

```bash
minicnn build --legacy-make --check
```

Build both native variants for comparison:

```bash
minicnn build --legacy-make --variant both --check
```

This writes:

```text
cpp/libminimal_cuda_cnn_cublas.so
cpp/libminimal_cuda_cnn_handmade.so
```

On Windows, build native DLLs with PowerShell:

```powershell
.\scripts\build_windows_native.ps1 -Variant both
```

Or with the CMake path:

```bash
minicnn build --check
```

## Prepare Dataset

Download CIFAR-10 before running the training commands below:

```bash
minicnn prepare-data
```

## Shared-config training

### 1) PyTorch backend

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
```

### 2) Handcrafted CUDA backend

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

The only switch above is `engine.backend`.

For the handcrafted CUDA backend, choose the native `.so` variant with:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

The native library is lazy-loaded, so non-CUDA commands such as `--help`, `prepare-data`, `validate-dual-config`, and torch backend runs do not require a built `.so`.
When a Python process switches between `runtime.cuda_variant` values, MiniCNN resets the cached `ctypes` handle so the next CUDA call loads the requested library instead of reusing the previous `.so`.

For quick debug runs, use config overrides:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas \
  train.epochs=1 train.batch_size=32 \
  dataset.num_samples=128 dataset.val_samples=32
```

The legacy trainer also accepts environment overrides such as `MINICNN_EPOCHS`, `MINICNN_BATCH`, `MINICNN_N_TRAIN`, and `MINICNN_N_VAL`.

Best model files are always written under:

```text
src/minicnn/training/models/
```

PyTorch writes `*_best.pt`; CUDA legacy writes `*_best_model_split.npz`. Per-run metrics and summaries stay under `artifacts/`.

## Architecture Templates

Ready-to-edit YAML examples live under `templates/`:

```bash
minicnn train-flex --config templates/mnist/lenet_like.yaml
minicnn train-flex --config templates/mnist/mlp.yaml
minicnn train-flex --config templates/cifar10/vgg_mini.yaml
minicnn train-dual --config templates/cifar10/vgg_mini_cuda.yaml engine.backend=cuda_legacy
```

MNIST templates use `dataset.type: mnist` and can download the IDX gzip files into `data/mnist/` on first run. CIFAR-10 templates expect `minicnn prepare-data` unless `dataset.download: true` is enabled. See `templates/README.md` for the architecture list and backend compatibility.

## Local Smoke Test Results

Validated on 2026-04-19 with an RTX 3050 Laptop GPU. The latest quick verification used `features/backend-smoke-matrix/run_smoke_matrix.py` with `128` train samples, `32` validation samples, batch size `32`, and `1` epoch:

| Backend | Native variant | Train acc | Val acc | Test acc | Epoch time |
|---|---|---:|---:|---:|---:|
| `torch` | PyTorch CUDA | `11.72%` | `0.00%` | `10.36%` | `1.3s` |
| `cuda_legacy` | `cublas` | `7.03%` | `6.25%` | `12.97%` | `0.2s` |
| `cuda_legacy` | `handmade` | `7.03%` | `6.25%` | `12.97%` | `0.2s` |

The smoke run writes model files to `src/minicnn/training/models/` and run logs to `/tmp/minicnn_backend_compare` when using the commands in the docs.

Verification also covered `pytest`, CLI help without a native `.so`, config validation, Python compile checks, and `minicnn build --legacy-make --variant both --check`.

Latest maintenance verification on 2026-04-20: `99 passed, 4 warnings`, `compileall` clean, `git diff --check` clean, and both cuBLAS/handmade native variants rebuilt with required symbols OK.

## Why two backends?

This repo intentionally supports two workflows:

- **Torch backend**: broad layer coverage, fast experimentation, custom components via dotted-path imports
- **CUDA backend**: your hand-rolled CUDA CNN path for low-level control and backend ownership

## MiniCNN autograd core

MiniCNN also includes a small CPU/NumPy autograd stack in `src/minicnn/nn/tensor.py`, `src/minicnn/ops/`, and `src/minicnn/nn/layers.py`. It supports `Tensor.backward()` with topological reverse-mode autodiff for scalar/tensor arithmetic, broadcasting, matrix multiply, reductions, reshape, ReLU, `log_softmax`, `cross_entropy`, trainable `Parameter`, lightweight `SGD` and `Adam` optimizers, and small educational layers such as `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten`, and `ResidualBlock`.

This core is useful for framework-level tests and small educational examples. The Torch backend still uses PyTorch autograd, and the handcrafted CUDA backend still uses its explicit CUDA backward kernels.

### Robustness notes

- **`train.init_seed`**: torch/flex model construction now seeds PyTorch before `build_model()`, so repeated runs with the same config start from the same weights. CUDA legacy and CPU/NumPy autograd already use their init seed paths.
- **Config overrides**: dotted CLI overrides support list indexes such as `model.layers.1.out_features=7`; malformed CUDA legacy numeric fields are reported as validation errors instead of raw tracebacks.
- **Boolean parsing**: string values such as `"false"` and `"0"` are parsed as false for dataset download, augmentation, AMP, optimizer helper flags, and CUDA conv `pool` fields.
- **CUDA library switching**: in-process `cuda_legacy` runs reset the cached native handle when runtime library settings change.
- **CUDA cleanup**: legacy CUDA training now frees device weights and velocity buffers through an outer cleanup path even if training raises before final evaluation.
- **`maxpool_backward_nchw`**: native code exports a status-returning wrapper, while the old void ABI remains for compatibility.
- **SGD**: parameter updates that fail with a shape or type mismatch emit a `RuntimeWarning` (including the parameter name) instead of silently skipping. Truly unexpected exceptions are no longer swallowed.
- **`Tensor.__pow__` backward**: gradient computation at `base == 0` with a negative exponent now returns `0` instead of `NaN`, preventing silent NaN propagation through the compute graph.
- **`maxpool2d`**: forward pass is fully vectorized with `sliding_window_view`; the loop over spatial output positions is gone.
- **`flex/builder`**: after each layer is materialized, the inferred output shape is validated to have all-positive dimensions. A misconfigured kernel size or pooling stride raises `ValueError` immediately with a descriptive message.
- **`BatchWorkspace.__del__`**: GPU memory cleanup failures now emit a `ResourceWarning` instead of being silently discarded.

## Shared config contract

The same config file contains:

- `dataset`
- `model.layers`
- `train`
- `loss`
- `optimizer`
- `scheduler`
- `engine.backend`

Example snippet:

```yaml
engine:
  backend: torch

model:
  layers:
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: MaxPool2d
      kernel_size: 2
      stride: 2
    - type: Conv2d
      out_channels: 64
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: Conv2d
      out_channels: 64
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: MaxPool2d
      kernel_size: 2
      stride: 2
    - type: Flatten
    - type: Linear
      out_features: 10
```

## CUDA backend support boundary

The handcrafted CUDA route currently supports the subset compiled by `src/minicnn/unified/cuda_legacy.py`:

- dataset: `cifar10`
- layers: `Conv2d -> activation -> Conv2d -> activation -> MaxPool2d -> Conv2d -> activation -> Conv2d -> activation -> MaxPool2d -> Flatten -> Linear`
- activations: `ReLU` or `LeakyReLU` with a single shared negative slope
- optimizer: `SGD`
- loss: `CrossEntropyLoss`
- input shape: `[3, 32, 32]`
- classes: `10`

If a config goes outside this subset, `validate-dual-config` explains why.

## Changing network architecture

The two backends use separate config keys for architecture.
For a file-by-file decision table covering YAML-only changes versus Torch
registry changes versus native CUDA backward changes, see
[docs/dual_backend_guide.md](docs/dual_backend_guide.md#when-architecture-changes-require-code-changes).

### CUDA backend (`train-cuda` / `cuda_legacy`)

Edit `model.conv_layers` in `configs/train_cuda.yaml`. Each entry is a `{out_c, pool}` pair. All shapes are derived automatically by `CudaNetGeometry` — no other file needs to change.

```yaml
model:
  c_in: 3
  h: 32
  w: 32
  kh: 3
  kw: 3
  fc_out: 10
  conv_layers:
    - {out_c: 32, pool: false}   # conv1
    - {out_c: 32, pool: true}    # conv2 + pool
    - {out_c: 64, pool: false}   # conv3
    - {out_c: 64, pool: true}    # conv4 + pool
```

Rules:
- `pool: true` inserts a 2×2 max-pool after the convolution at that stage.
- The first stage must have `in_c == c_in`; subsequent stages infer `in_c` from the previous stage's `out_c`.
- The CUDA kernel only supports `kh == kw == 3` and input sizes divisible by pooling strides.
- Run `minicnn validate-dual-config --config configs/train_cuda.yaml` to check your config before training.

### Flex / Torch backend (`train-flex` / `torch`)

Edit `model.layers` in `configs/dual_backend_cnn.yaml` (or your own flex config). Add, remove, or rearrange layer entries freely. `in_channels` / `in_features` are inferred automatically.

```yaml
model:
  layers:
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
    - type: LeakyReLU
      negative_slope: 0.1
    - type: MaxPool2d
      kernel_size: 2
      stride: 2
    - type: Flatten
    - type: Linear
      out_features: 10
```

No Python changes are needed for either backend — only the YAML file.

## Useful commands

```bash
minicnn info
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
minicnn train --config configs/dual_backend_cnn.yaml engine.backend=torch train.epochs=1
minicnn train-torch --config configs/dual_backend_cnn.yaml train.epochs=1
minicnn train-cuda --config configs/dual_backend_cnn.yaml train.epochs=1
minicnn train-autograd --config configs/autograd_tiny.yaml train.epochs=1
minicnn compare --config configs/dual_backend_cnn.yaml train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
minicnn dual-config-template
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn validate-config --config configs/dual_backend_cnn.yaml
minicnn compile --config configs/autograd_tiny.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
```

## Custom components like PyTorch

For the Torch backend, users can point at dotted-path classes directly in the config.

Example:

```yaml
model:
  layers:
    - type: Flatten
    - type: Linear
      out_features: 32
    - type: examples.custom_block.CustomHead
      in_features: 32
      out_features: 10
```

Run it with:

```bash
minicnn train-dual --config configs/dual_backend_torch_custom.yaml
```

## Project layout

```text
minicnn/
├── configs/
│   ├── dual_backend_cnn.yaml          # main CIFAR-10 config; switch torch/cuda_legacy here
│   ├── dual_backend_torch_custom.yaml # custom dotted-path component example
│   ├── autograd_tiny.yaml             # tiny CPU/NumPy autograd training smoke config
│   ├── flex_*.yaml                    # PyTorch flex trainer examples
│   ├── train_cuda.yaml                # legacy CUDA compatibility config
│   └── train_torch.yaml               # Torch baseline compatibility config
├── cpp/
│   ├── Makefile                       # Linux native .so build, including cublas/handmade variants
│   ├── CMakeLists.txt                 # CMake build path used by Linux/Windows helpers
│   ├── include/                       # native public headers
│   └── src/                           # CUDA/C++ kernels and C API implementation
├── docs/                              # tutorials and reference docs
├── examples/                          # custom PyTorch component examples
├── features/
│   ├── README.md                      # rules for isolated prototype work
│   └── backend-smoke-matrix/          # example feature comparing torch/cublas/handmade smoke runs
├── scripts/
│   └── build_windows_native.ps1       # Windows CUDA DLL build helper
├── src/minicnn/
│   ├── cli.py                         # minicnn command entrypoint
│   ├── autograd/                      # compatibility namespace for Tensor, Function, Context
│   ├── compiler/                      # lightweight MiniCNN IR, tracer, passes, scheduler, lowering stubs
│   ├── config/                        # config schema, loader, legacy settings bridge
│   ├── core/                          # native build helpers and lazy ctypes CUDA binding
│   ├── data/                          # CIFAR-10 preparation/loading
│   ├── flex/                          # config-driven PyTorch model builder and trainer
│   ├── models/                        # MiniCNN CPU/NumPy model registry and config builder
│   ├── nn/                            # MiniCNN Tensor, Parameter, and CPU/NumPy autograd core
│   ├── ops/                           # CPU/NumPy differentiable layer ops
│   ├── optim/                         # MiniCNN SGD and Adam optimizers
│   ├── runtime/                       # graph executor, backend protocol, memory pool, profiler
│   ├── training/
│   │   ├── train_cuda.py              # legacy CUDA CIFAR-10 orchestration entrypoint
│   │   ├── cuda_batch.py              # CUDA batch forward/loss/backward/update steps
│   │   ├── train_torch_baseline.py    # PyTorch baseline orchestration entrypoint
│   │   ├── train_autograd.py          # CPU/NumPy autograd training entrypoint
│   │   ├── models/                    # fixed checkpoint output folder
│   │   ├── loop.py                    # shared metrics/LR/early-stop/epoch summary helpers
│   │   ├── legacy_data.py             # shared CIFAR-10 loading/normalization for legacy trainers
│   │   ├── cuda_ops.py                # CUDA copy/layout/forward helper wrappers
│   │   ├── cuda_workspace.py          # reusable per-batch GPU workspace
│   │   ├── evaluation.py              # CUDA eval forward/accuracy helpers
│   │   └── checkpoints.py             # CUDA checkpoint save/load/free helpers
│   └── unified/
│       ├── config.py                  # shared default config and override merge
│       ├── cuda_legacy.py             # maps shared config into legacy CUDA settings
│       └── trainer.py                 # dispatches train-dual to torch or cuda_legacy
└── tests/                             # unit/smoke tests for config, imports, and framework wiring
```

Key folder and file responsibilities:

| Path | Purpose |
|---|---|
| `configs/` | YAML configs for torch, cuda_legacy, flex, custom-component, AlexNet-like, and ResNet-like runs. |
| `configs/autograd_tiny.yaml` | Small random-data config for the CPU/NumPy autograd trainer. |
| `cpp/` | Native CUDA/C++ source, headers, Makefile, and CMake build files. |
| `cpp/include/cuda_check.h` | CUDA runtime and kernel launch checking; debug builds define `MINICNN_DEBUG_SYNC` for synchronizing checks. |
| `cpp/include/network.h` | Secondary C++ layer API using RAII-owned `std::unique_ptr<CudaTensor>` forward outputs; the default CLI path uses the flat C ABI through `ctypes`. |
| `cpp/src/cublas_context.cu` | Shared cuBLAS handle used by forward and backward CUDA code. |
| `cpp/src/core.cu` | GEMM forward path; switches between cuBLAS and handwritten CUDA with `USE_CUBLAS`. |
| `cpp/src/conv_backward.cu` | Convolution backward kernels and cuBLAS/handmade weight-gradient path. |
| `cpp/src/loss_layer.cu` | Softmax, fused softmax cross-entropy loss/gradient/accuracy, and GEMM backward helpers. |
| `cpp/src/network.cu` | C++ layer forward implementations; ConvLayer reuses an im2col cache and ReLU writes out-of-place. |
| `cpp/src/gpu_monitor.cu` | Lightweight GPU memory status helper using CUDA runtime APIs, without shelling out. |
| `docs/` | Build, C API, Python ctypes, C++ linking, layout/debug, and Windows build guides. |
| `examples/` | Minimal custom PyTorch component examples. |
| `features/` | Isolated prototypes that production code must not import by default; includes `backend-smoke-matrix/` as an example feature. |
| `scripts/build_windows_native.ps1` | PowerShell helper for building Windows CUDA DLL variants. |
| `src/minicnn/cli.py` | Main CLI entrypoint. |
| `src/minicnn/autograd/` | Compatibility namespace for MiniCNN `Tensor`, `Parameter`, `Function`, `Context`, `no_grad`, and `backward`. |
| `src/minicnn/compiler/` | Lightweight IR, model-config tracer, optimizer passes, scheduler, and explicit lowering boundary. |
| `src/minicnn/core/build.py` | Native build/check helper used by `minicnn build`. |
| `src/minicnn/core/cuda_backend.py` | Lazy ctypes loader and Python helpers for the native CUDA library. |
| `src/minicnn/core/fused_ops.py` | NumPy reference helper for Conv2d + BatchNorm2d + ReLU fusion semantics. |
| `src/minicnn/data/` | CIFAR-10 download/loading and random dataset helpers. |
| `src/minicnn/flex/` | PyTorch config-driven model/loss/optimizer/scheduler builder and trainer; includes torch-only `ResidualBlock` and `GlobalAvgPool2d`. |
| `src/minicnn/models/` | CPU/NumPy MiniCNN model registry, shape inference, config builder, and graph helpers. |
| `src/minicnn/nn/` | MiniCNN framework layer: `Module`, `Sequential`, `Tensor`, `Parameter`, and the CPU/NumPy autograd functions. |
| `src/minicnn/nn/tensor.py` | Reverse-mode autograd engine for scalar/tensor ops, broadcasting, matmul, reductions, ReLU, `log_softmax`, and `cross_entropy`. |
| `src/minicnn/nn/layers.py` | CPU/NumPy MiniCNN layers: `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten`, `ReLU`, and `ResidualBlock`. |
| `src/minicnn/ops/` | Differentiable NumPy ops used by MiniCNN layers. |
| `src/minicnn/optim/` | Lightweight optimizer interfaces; `SGD` and `Adam` update MiniCNN `Parameter` objects without requiring torch. |
| `src/minicnn/runtime/` | Small graph executor, backend protocol, tensor memory pool, and profiler utilities. |
| `src/minicnn/training/train_cuda.py` | Legacy CUDA CIFAR-10 orchestration entrypoint: data, epochs, validation, checkpointing, LR reduction, early stop, and final test evaluation. |
| `src/minicnn/training/cuda_batch.py` | Backend-specific CUDA batch steps: conv forward, FC forward, fused loss/accuracy, FC update, conv backward/update. |
| `src/minicnn/training/train_autograd.py` | Random-data CPU/NumPy autograd training loop that writes `*_autograd_best.npz`. |
| `src/minicnn/training/models/` | Fixed output folder for best model checkpoints; generated `*.pt` and `*.npz` files are git-ignored. |
| `src/minicnn/training/loop.py` | Shared training-loop state: running metrics, per-layer-group LR state, best/plateau/early-stop state, epoch timing, LR reduction, and epoch summary formatting. |
| `src/minicnn/training/legacy_data.py` | Shared CIFAR-10 load/normalize helper used by legacy CUDA and Torch baseline trainers. |
| `src/minicnn/training/cuda_ops.py` | Small CUDA operation wrappers used by the legacy training loop. |
| `src/minicnn/training/cuda_workspace.py` | Reusable per-batch GPU workspace with double-free protection. |
| `src/minicnn/training/evaluation.py` | CUDA evaluation forward path and accuracy helpers. |
| `src/minicnn/training/checkpoints.py` | CUDA checkpoint save/reload and GPU pointer cleanup. |
| `src/minicnn/training/train_torch_baseline.py` | PyTorch baseline orchestration and batch helpers mirroring the handcrafted CUDA update rules. |
| `src/minicnn/unified/` | Shared-config compiler and dispatcher for `torch` vs `cuda_legacy`. |
| `tests/` | Unit and smoke tests that avoid requiring GPU unless explicitly run through training commands. |

Windows native build notes are in [docs/07_windows_build.md](docs/07_windows_build.md).

## Development

```bash
python -m pip install -e .[torch,dev]
pytest
python -m compileall -q src
```

## Feature Isolation Workflow

Stable code lives under `src/minicnn/` and must keep `main` runnable. New or risky work starts in a Git branch and an isolated folder under `features/`.

```bash
git checkout -b feature/native-cuda-class-backend
mkdir -p features/native-cuda-class-backend
```

Use `features/<name>/` for prototypes, notes, and exploratory tests. Production code must not import from `features/` by default. Once a feature is stable, move the supported implementation into `src/minicnn/`, move tests into `tests/`, update docs, and run the full test suite before merging.

For larger experiments, use a separate worktree so the stable checkout remains available:

```bash
git worktree add ../minicnn-feature-native -b feature/native-backend
```

## Notes

- The **Torch path is the most flexible path**.
- The **CUDA path is the handcrafted path** and currently validates a supported subset before running.
- This keeps the config interface unified while staying honest about backend capabilities.


## Honest capability note

This package gives you one shared config interface for both backends. The Torch path is fully flexible. The handcrafted CUDA path is real, but it currently targets the supported CNN subset described above and compiles that config into the legacy CUDA trainer.
