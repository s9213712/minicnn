# Comment Review Status

This file records the result of processing the actionable notes under `comments/`, excluding `comments/參考projects/`.

Processed local note files:

- `comments/minicnn_agent_ultimate_full_spec_v5.txt`
- `comments/minicnn_agent_ultimate_full_spec.txt`
- `comments/minicnn_agent_ultimate_prompt.txt`
- `comments/新增 文字文件.txt`
- `comments/TEST PLAN AND REQUIRED STANDARD.txt`
- `comments/AGENT SELF-CHECKLIST.txt`

## Completed

| Area | Status |
|---|---|
| Lazy CUDA loading | Implemented through `get_lib()` and `LazyCudaLibrary`; non-CUDA CLI paths do not require a built `.so`. `MINICNN_SO_PATH` is also accepted as a library override alias. |
| Settings/config overrides | Existing schema/YAML/CLI overrides remain in place; common train CLI flags are exposed across `train-flex`, `train-dual`, `train`, `train-cuda`, `train-torch`, `train-autograd`, and `compare`. |
| CLI introspection | `info`, `doctor`, `healthcheck`, component listing, `validate-dual-config`, `validate-config`, `show-cuda-mapping`, `compare`, and `compile` are available. |
| Unified train alias | Added `minicnn train` as an alias for `train-dual`. |
| Package hygiene | `.gitignore` covers caches, artifacts, native libs, data, model checkpoints, and `comments/`. Generated smoke checkpoints are kept out of git. |
| Fixed checkpoint location | Best model outputs are fixed under `src/minicnn/training/models/` for torch, CUDA legacy, and CPU/NumPy autograd paths. |
| CUDA training split | Existing `cuda_workspace.py` and `cuda_ops.py` are retained; `cuda_epoch.py` hosts lower-risk epoch helpers while preserving the legacy training loop behavior. |
| Explicit device state | Added `DeviceWeights` and `VelocityBuffers` dataclasses while keeping tuple iteration compatibility for legacy call sites. |
| Duplicated eval forward path | Centralized CUDA evaluation forward flow through `_forward_logits_ptr()`. |
| Checkpoint reload safety | `reload_weights_from_checkpoint()` uploads replacement weights before freeing existing runtime weights. |
| MiniCNN autograd core | Expanded CPU/NumPy autograd with `no_grad()`, `detach()`, module `__call__`, recursive module traversal, `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten`, `ReLU`, same-channel `ResidualBlock`, and differentiable NumPy ops. |
| Optimizers | `SGD` now applies weight decay; added `Adam` for MiniCNN `Parameter` objects. |
| Autograd namespace | Added `src/minicnn/autograd/` compatibility modules for `Tensor`, `Parameter`, `Context`, `Function`, `no_grad`, and `backward`. |
| MiniCNN model builder | Added CPU/NumPy model registry, shape inference, block factory, graph builder, and `build_model_from_config()`. |
| CPU/NumPy autograd training | Added `minicnn train-autograd` and `configs/autograd_tiny.yaml`; it writes `*_autograd_best.npz`, metrics, `test_acc`, and epoch timing. |
| Runtime utilities | Added graph executor, backend protocol, memory pool, and profiler utilities. |
| Compiler skeleton | Added IR, config tracer, Conv2d+BatchNorm2d+ReLU fusion annotation pass, optimizer, scheduler, and lowering boundary. |
| Fused op semantics | Added a NumPy reference helper for Conv2d + BatchNorm2d + ReLU fusion semantics and tests against the unfused path. |
| Data path handling | `src/minicnn/data/cifar10.py` now normalizes data roots through `Path`, reports missing batch files explicitly, and points users to `minicnn prepare-data` or `--data-dir`. |
| Builder validation | `src/minicnn/models/builder.py` now validates layer mappings, required `type`, rank assumptions, and non-positive inferred shapes with clear errors. |
| Windows build support | Added PowerShell/CMake Windows native build documentation and helper in earlier work; Windows artifacts cannot be tested on this Linux host. |
| Example feature folder | Added `features/backend-smoke-matrix/` in earlier work for isolated smoke comparison. |
| Docs | README, Traditional Chinese README, USAGE, project file guide, autograd guide, and this status report document current commands, folders, metrics, and capability boundaries. |
| Tests | Added coverage for CLI exposure, autograd layers, optimizer updates, runtime utilities, compiler fusion annotation, and fused-op fallback semantics. |

## Deferred

| Item | Reason |
|---|---|
| Full class-based CUDA trainer rewrite | The current legacy trainer is safer and more modular, but a complete behavior-preserving rewrite needs longer GPU regression coverage. |
| Full `train_one_epoch()` extraction | The inner CUDA forward/backward/update loop is still in `train_cuda.py` to avoid untested device-state transfer bugs; safer helpers were extracted first. |
| Native CUDA autograd bridge | Mapping MiniCNN `Function` graphs to existing CUDA kernels needs a separate design pass and GPU validation. The implemented autograd stack is CPU/NumPy. |
| Production native fused Conv+BN+ReLU kernel | Implemented as NumPy semantic reference and compiler annotation only. A real CUDA fused kernel requires C++/CUDA API work and benchmark validation. |
| Full compiler lowering to native execution | The compiler currently traces configs, annotates simple fusion patterns, schedules nodes, and marks unsupported CUDA direct lowering explicitly. It does not emit runnable CUDA kernels. |
| General residual projections | The CPU/NumPy `ResidualBlock` currently supports same-channel residuals. Projection shortcuts are deferred until the config and shape contract are expanded. |
| Full dataset support for `train-autograd` | `train-autograd` intentionally supports `dataset.type=random` only for deterministic CPU smoke tests. CIFAR-10 production training remains on torch or CUDA legacy. |
| Windows binary validation | Build scripts/docs are present, but this Linux environment cannot compile or run Windows DLLs. |

## Verification

Run for this review:

```bash
python3 -m compileall -q src
PYTHONPATH=src python3 -m pytest -q tests
PYTHONPATH=src python3 -m minicnn.cli info
PYTHONPATH=src python3 -m minicnn.cli doctor
PYTHONPATH=src python3 -m minicnn.cli validate-config --config configs/dual_backend_cnn.yaml
PYTHONPATH=src python3 -m minicnn.cli compile --config configs/autograd_tiny.yaml
PYTHONPATH=src python3 -m minicnn.cli train-autograd --config configs/autograd_tiny.yaml train.epochs=1 dataset.num_samples=8 dataset.val_samples=4 train.batch_size=4
git diff --check
```
