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

Best model files are always written under:

```text
python/best_models/
```

PyTorch writes `*_best.pt`; CUDA legacy writes `*_best_model_split.npz`. Per-run metrics and summaries stay under `artifacts/`.

## Local Smoke Test Results

Validated on 2026-04-19 with an RTX 3050 Laptop GPU, CIFAR-10 smoke split, `256` train samples, `64` validation samples, batch size `64`, and `1` epoch:

| Backend | Native variant | Result |
|---|---|---|
| `torch` | PyTorch CUDA | train_acc `10.16%`, val_acc `12.50%` |
| `cuda_legacy` | `cublas` | train_acc `12.50%`, val_acc `20.31%`, test_acc `14.00%`, epoch time `0.1s` |
| `cuda_legacy` | `handmade` | train_acc `12.50%`, val_acc `20.31%`, test_acc `14.00%`, epoch time `0.3s` |

The smoke run writes model files to `python/best_models/` and run logs to `/tmp/minicnn_backend_compare` when using the commands in the docs.

## Why two backends?

This repo intentionally supports two workflows:

- **Torch backend**: broad layer coverage, fast experimentation, custom components via dotted-path imports
- **CUDA backend**: your hand-rolled CUDA CNN path for low-level control and backend ownership

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

## Useful commands

```bash
minicnn info
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
minicnn dual-config-template
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
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
configs/
cpp/
docs/
examples/
src/minicnn/
  config/
  core/
  data/
  engine/
  flex/
  framework/
  nn/
  optim/
  runtime/
  schedulers/
  training/
  unified/
tests/
```

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
