# MiniCNN Documentation Guide

This page is the documentation entrypoint for the current repo.

If you only read one file inside `docs/`, make it this one.

## Start Here

Read these in order if you want the current, operational picture of the repo:

1. [architecture.md](architecture.md) — overall structure, execution paths, and module boundaries
2. [backend_capabilities.md](backend_capabilities.md) — what each backend really supports
3. [dual_backend_guide.md](dual_backend_guide.md) — how one shared config maps into `torch`, `cuda_legacy`, and `cuda_native`

## By Task

### I want to install MiniCNN and check that it works

- Run `minicnn smoke`
- See [../README.md](../README.md) for quick start
- See [architecture.md](architecture.md) and [backend_capabilities.md](backend_capabilities.md) if the result is unclear

### I want to train a model right now

- [backend_capabilities.md](backend_capabilities.md) — choose the right backend first
- [dual_backend_guide.md](dual_backend_guide.md) — shared-config path and backend routing
- [../templates/README.md](../templates/README.md) — ready-to-edit config templates

### I want to work on the handcrafted CUDA path

- [01_project_files.md](01_project_files.md) — repo map of the native and Python sides
- [02_build_shared_library.md](02_build_shared_library.md) — build the shared library
- [03_c_api_reference.md](03_c_api_reference.md) — exported C API reference
- [06_layout_and_debug.md](06_layout_and_debug.md) — layout rules and debugging workflow
- [07_windows_build.md](07_windows_build.md) — Windows-specific build path
- [cuda_batchnorm2d_evaluation.md](cuda_batchnorm2d_evaluation.md) — focused note on one unresolved `cuda_legacy` extension area

### I want Python `ctypes` or C++ embedding examples

- [04_python_ctypes_mnist.md](04_python_ctypes_mnist.md) — Python `ctypes` example against the native library
- [05_cpp_linking.md](05_cpp_linking.md) — C++ linking path for the secondary C++ API

### I want to work on autograd or the broader frontend

- [08_autograd.md](08_autograd.md) — NumPy autograd stack and `train-autograd`
- [09_feature_expansion.md](09_feature_expansion.md) — wider feature-surface notes
- [custom_components.md](custom_components.md) — dotted-path component and dataset extension points
- [generalization_roadmap.md](generalization_roadmap.md) — how frontend breadth should relate to backend honesty

### I want to work on `cuda_native`

- [cuda_native.md](cuda_native.md) — full guide to the current experimental backend
- [cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md) — future extension RFCs
- [backend_capabilities.md](backend_capabilities.md) — current validated contract

## CLI Surface

### Stable commands

```bash
minicnn build --legacy-make --check
minicnn smoke
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

### Experimental but public commands

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
minicnn cuda-native-capabilities
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_native
```

### Inspection commands

```bash
minicnn info
minicnn smoke
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
```

`minicnn smoke` is the fastest first-run check. It confirms that:

- the repo layout is intact
- built-in configs parse
- the compiler can trace a default flex model
- both `cuda_legacy` and `cuda_native` validators still accept their reference configs

Built-in config paths such as `configs/flex_cnn.yaml` and
`configs/dual_backend_cnn.yaml` fall back to project-root-relative resolution,
so they still work when `minicnn` is launched from outside the repo root.

## Document Roles

### Current operational docs

These describe the repo as it works today:

- [architecture.md](architecture.md)
- [backend_capabilities.md](backend_capabilities.md)
- [dual_backend_guide.md](dual_backend_guide.md)
- [cuda_native.md](cuda_native.md)
- [custom_components.md](custom_components.md)
- [01_project_files.md](01_project_files.md)
- [08_autograd.md](08_autograd.md)
- [09_feature_expansion.md](09_feature_expansion.md)

### Focused technical notes

These are narrower deep dives, still useful, but not the best first read:

- [06_layout_and_debug.md](06_layout_and_debug.md)
- [cuda_batchnorm2d_evaluation.md](cuda_batchnorm2d_evaluation.md)
- [cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)

### Historical or reporting documents

These are reference material, not primary onboarding docs:

- [comparison_report.md](comparison_report.md)
- [comparison_completion_report.md](comparison_completion_report.md)
- [optimization_progress.md](optimization_progress.md)
- [benchmark_report_template.md](benchmark_report_template.md)

## Quick Navigation

If you are unsure where to go next:

- Need the truth about support boundaries: [backend_capabilities.md](backend_capabilities.md)
- Need to understand one shared config across backends: [dual_backend_guide.md](dual_backend_guide.md)
- Need native build/debug details: [01_project_files.md](01_project_files.md) and [06_layout_and_debug.md](06_layout_and_debug.md)
- Need extension points: [custom_components.md](custom_components.md)
- Need experimental native graph backend context: [cuda_native.md](cuda_native.md)

## Rule Of Thumb

- If you want the broadest stable feature set, use torch/flex.
- If you want the handwritten CUDA path, stay inside the `cuda_legacy` validator boundary.
- If you want framework learning without torch, use `train-autograd`.
- If you want to push native backend generalization, treat `cuda_native` as an experimental backend with a narrow validated contract, not as a drop-in stable replacement.
