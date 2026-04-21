# MiniCNN Usage Guide

This file is the documentation index for the current repo.

If you are starting from scratch, read in this order.

## 1. Project Orientation

- [architecture.md](architecture.md): overall layout, stable paths, and branch-local `cuda_native` status
- [backend_capabilities.md](backend_capabilities.md): what each backend really supports
- [dual_backend_guide.md](dual_backend_guide.md): how one shared config maps into torch vs `cuda_legacy`

## 2. Native CUDA Library

- [01_project_files.md](01_project_files.md): file-by-file map of the native and Python sides
- [02_build_shared_library.md](02_build_shared_library.md): build the shared library
- [03_c_api_reference.md](03_c_api_reference.md): exported C API surface
- [06_layout_and_debug.md](06_layout_and_debug.md): layout rules and debugging workflow
- [07_windows_build.md](07_windows_build.md): Windows build path

## 3. Language Bindings / Examples

- [04_python_ctypes_mnist.md](04_python_ctypes_mnist.md): Python `ctypes` example
- [05_cpp_linking.md](05_cpp_linking.md): C++ linking path

## 4. Higher-Level Frontend / Autograd

- [08_autograd.md](08_autograd.md): NumPy autograd stack and `train-autograd`
- [09_feature_expansion.md](09_feature_expansion.md): broader feature notes
- [custom_components.md](custom_components.md): torch/flex dotted-path component extension points
- [generalization_roadmap.md](generalization_roadmap.md): how frontend generalization should relate to backend honesty

## Stable CLI Surface

The stable user-facing commands on this branch are:

```bash
minicnn build --legacy-make --check
minicnn prepare-data
minicnn train-flex --config configs/flex_cnn.yaml
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
minicnn train-autograd --config configs/autograd_tiny.yaml
minicnn compare --config configs/dual_backend_cnn.yaml
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
minicnn compile --config configs/autograd_tiny.yaml
```

Useful inspection commands:

```bash
minicnn info
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
```

## Backend Routing

Today, `train-dual` is the stable shared-config entry for:

- `engine.backend=torch`
- `engine.backend=cuda_legacy`

`train-autograd` remains a separate path with its own training loop and partly
overlapping config contract.

This branch also contains `src/minicnn/cuda_native/`, but that backend is still
experimental and is not yet part of the stable `train-dual` toggle surface.

## Quick Commands

Torch/flex:

```bash
minicnn train-flex --config configs/flex_broad.yaml
```

Torch through the shared dual config:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
```

Handwritten CUDA:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

Native CUDA variant selection:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

NumPy autograd:

```bash
minicnn train-autograd --config configs/autograd_enhanced.yaml
```

## Build / Runtime Notes

- The native `.so` is lazy-loaded, so `--help`, `prepare-data`, validation, and torch-only flows do not require a built library.
- `runtime.cuda_variant` switching resets the cached native library handle inside one Python process.
- Best model files are written under `src/minicnn/training/models/`.
- Per-run metrics and summaries are written under `artifacts/`.

## Templates And Example Configs

Ready-to-edit templates live under `templates/`:

```bash
minicnn train-flex --config templates/mnist/lenet_like.yaml
minicnn train-flex --config templates/mnist/mlp.yaml
minicnn train-flex --config templates/cifar10/vgg_mini.yaml
minicnn train-dual --config templates/cifar10/vgg_mini_cuda.yaml engine.backend=cuda_legacy
```

Other useful example configs:

- `configs/flex_broad.yaml`
- `configs/autograd_enhanced.yaml`
- `configs/cuda_legacy_strict.yaml`

## Benchmarking

Use [benchmark_report_template.md](benchmark_report_template.md) when recording
results.

If you want a repeatable smoke benchmark workflow, inspect
`features/backend-smoke-matrix/`.

## Rule Of Thumb

- If you want the broadest stable feature set, use torch/flex.
- If you want the handwritten CUDA path, stay inside the `cuda_legacy` validator boundary.
- If you want framework learning without torch, use `train-autograd`.
- If you want to push native backend generalization, treat `cuda_native` as experimental branch work until its CLI and capability surface are promoted together.
