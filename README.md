# MiniCNN

[繁體中文 README](README.zh-TW.md)

![status](https://img.shields.io/badge/status-beta-yellow)
![frontend](https://img.shields.io/badge/frontend-YAML%20%2B%20CLI-blue)
![native](https://img.shields.io/badge/native-CUDA-green)

MiniCNN is a configuration-driven deep learning project for studying the gap
between a flexible frontend and backend-constrained execution paths.

Shell note:
Most multi-line command examples in this repo use Bash line continuation with
`\`. On Windows, replace that with PowerShell `` ` ``. If you are using
`cmd.exe` instead of PowerShell, use `^`.

Today, the repo gives you four backend roles:

- `torch` via `train-flex` / `train-dual` as the broad reference implementation
- `cuda_native` via `train-native` as the primary native backend direction
- `autograd` via `train-autograd` as the internal correctness oracle
- `cuda_legacy` via `train-dual` as the maintenance-only historical CUDA path

## Fast Start

If you want the shortest path from clone to a working command:

```bash
python3 -m pip install -e .
minicnn smoke
minicnn show-model --config configs/flex_cnn.yaml --format text
```

If those commands succeed, the repo layout, config parsing, and core CLI
surfaces are working. From there:

- use `minicnn train-flex --config configs/flex_cnn.yaml` for the broadest reference path
- use `minicnn train-autograd --config configs/autograd_tiny.yaml` for the smallest CPU-only reference path
- use `USAGE.md` as the doc index when you want task-based navigation

In JSON mode, `minicnn smoke` also reports:

- `torch_available`
- `cuda_available`
- `native_available`
- `flex_registry_ready`
- `warnings` / `errors`

## Why This Exists

Most frameworks intentionally hide kernel orchestration, memory handling, and
backend boundaries behind a smooth API.

MiniCNN is useful when you want to look at those boundaries directly:

- how one frontend interface maps into different backend realities
- where a narrow native backend needs strict validation instead of fake parity
- how a small autograd stack behaves without relying on torch internals
- how a future graph-based native backend can be prototyped in public

## Positioning

MiniCNN is not trying to replace PyTorch.

It is useful when you want one of these:

- a shared YAML/frontend interface that can target different backend roles
- a broad torch reference path for trying new ideas first
- a small NumPy autograd stack for correctness checks and framework-level experiments
- a place to grow a native backend in public without pretending every older path must keep up

## Backend Roles

| Backend | Role | Current status |
|---|---|---|
| `torch` | reference implementation | stable, broadest feature surface, first destination for new model work |
| `cuda_native` | primary native backend | beta, graph-based, ordered-DAG capable, reference mode plus partial real-CUDA `gpu_native` execution |
| `autograd` | correctness oracle | stable, CPU-only, useful for deterministic checks and framework learning |
| `cuda_legacy` | historical native backend | stable inside a narrow boundary, maintenance-only, not the target for new feature growth |

## Feature Rollout Order

When a new capability is added, the default rollout order is:

1. `torch/flex` first
2. `autograd` when a correctness reference adds value
3. `cuda_native` when the native graph path is ready for it
4. `cuda_legacy` only when maintenance requires it, not as the default expansion target

At a high level:

```text
shared YAML / CLI frontend -> torch [REFERENCE] | autograd [ORACLE]
                               \
                                -> cuda_native [PRIMARY NATIVE] (beta graph IR, planner, reference + gpu_native executor)
                               \
                                -> cuda_legacy [MAINTENANCE ONLY] (historical handwritten CUDA path)
```

## Recent Engineering Progress

The recent cleanup and refactor program is now part of `main`.

The current repo state includes:

- CLI dispatch split into dedicated parser, readonly, and training/compare helper modules
- `flex` training split across context setup, run orchestration, reporting, device resolution, and step helpers
- unified `cuda_native` training split across bridge, runtime-loop, and support/reporting helpers
- artifact inspection/export and checkpoint payload handling moved into dedicated helper layers
- JSON-friendly diagnostics and validation surfaces across `healthcheck`, `doctor`, `smoke`, `validate-*`, and inspection commands
- real `show-model` and `show-graph` introspection commands instead of placeholders
- `cuda_native` graph semantics broadened from strict sequential graphs to ordered DAG execution with named tensor wiring plus `Add` / `Concat`
- `cuda_native` training surface broadened to `SGD`, `Adam`, `AdamW`, `RMSprop`, `CrossEntropyLoss`, `BCEWithLogitsLoss`, `MSELoss`, `label_smoothing`, `grad_accum_steps`, and beta AMP
- `cuda_native` strict `gpu_native` now has a real CIFAR-10 two-Conv training runbook using native CUDA helper execution
- `summary.json` / `metrics.jsonl` now expose planner, AMP, and optimizer-state telemetry through stable reporting keys

The user-facing command surface is intentionally still small, but the internal
module boundaries are now narrower, the output contracts are more explicit, and
the backend role docs are aligned with the actual code.

## What You Can Run Today

### `torch`

- reference implementation for new frontend ideas
- broad `model.layers[]` support through the flex registry
- custom dotted-path components
- schedulers, regularization, and richer experimentation workflows

### `cuda_legacy`

- maintenance-only historical backend
- handcrafted CUDA / C++ backend in `cpp/`
- shared-config bridge from `engine.backend=cuda_legacy`
- strict validation instead of silent fallback
- narrow, honest support boundary centered on CIFAR-10 and the fixed Conv/Pool/Linear pattern

### `autograd`

- internal correctness oracle for CPU-side reference checks
- pure NumPy reverse-mode autodiff
- small optimizer/layer stack for learning and tests
- architecture tracing and CPU inference experiments without torch

### `cuda_native` (Primary Native Direction, Beta)

The active native growth path in the repo, built as a graph-based backend with:

- explicit graph IR (`graph.py`, `nodes.py`)
- strict validation (`validators.py`, `shapes.py`)
- conservative buffer planning (`planner.py`)
- numpy reference kernels (`kernels.py`, `executor.py`)
- backward prototype and SGD training loop
- layout validation (`layouts.py` — `validate_graph_layouts()`)
- memory estimation and reuse (`memory.py` — `memory_footprint()`, `BufferPool`)
- observability tooling (`debug.py` — `dump_graph()`, `dump_plan()`, `TracingForwardExecutor`)

Supported ops: `BatchNorm2d` (forward/backward prototype), `Conv2d`, `DepthwiseConv2d`, `PointwiseConv2d`, `GroupNorm`, `LayerNorm`, `LayerNorm2d`, `ResidualBlock`, `ConvNeXtBlock`, `Dropout`, `DropPath`, `Add`, `Concat`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GELU`, `Identity`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` (`output_size=(1,1)` only), `GlobalAvgPool2d`, `Flatten`, `Linear`.

Current validated support boundary:

- datasets: `random`, `cifar10`, `mnist`
- losses: `CrossEntropyLoss` (optional `label_smoothing`), `BCEWithLogitsLoss` (binary output only), `MSELoss`
- optimizer: `SGD`, `Adam`, `AdamW`, `RMSprop`, with optional global gradient clipping
- scheduler: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, or disabled
- `train.grad_accum_steps >= 1`
- `train.amp=true|false` with beta loss scaling / overflow backoff
- `summary.json` reports `amp_runtime`, `optimizer_runtime`, `planner`, and `performance_report`
- `metrics.jsonl` rows report per-epoch AMP, optimizer, and planner telemetry

`cuda_native` is now a beta-grade backend with stable artifact/validation contracts, `training_stable=true`, and `backward_stable=true`. The default path is `engine.execution_mode=gpu_native_auto`: it tries the supported real-CUDA helper subset first and falls back explicitly to `reference_numpy` when lowering or runtime preflight is not ready. Use `engine.execution_mode=gpu_native` when you want strict GPU-only failure semantics, or `engine.execution_mode=reference_numpy` for the historical CPU fallback path. It supports ordered DAG execution with explicit tensor wiring plus `Add` merge semantics; `cuda_legacy` remains a narrow maintenance path.

Strict real-CUDA CIFAR-10 training is available for the current two-Conv helper subset:

```bash
minicnn validate-cuda-native-config --config configs/cifar10_cuda_native_gpu_stronger.yaml

minicnn train-native --config configs/cifar10_cuda_native_gpu_stronger.yaml
```

See [docs/cuda_native_gpu_cifar10_runbook.md](docs/cuda_native_gpu_cifar10_runbook.md) for the current real-data GPU result and performance notes.

Hermetic native smoke examples now exist for:

- explicit ConvNeXt primitives: `templates/cifar10/convnext_explicit_cuda_native_smoke.yaml`
- named `ConvNeXtBlock` path: `templates/cifar10/convnext_tiny_cuda_native_smoke.yaml`
- `ResidualBlock` path: `templates/cifar10/resnet_like_cuda_native_smoke.yaml`

```bash
# Check what cuda_native supports
minicnn cuda-native-capabilities

# Check native CUDA library, driver/runtime, and WSL/device-node readiness
minicnn check-cuda-ready

# Validate your config
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5

# Run (research only)
minicnn train-native --config configs/dual_backend_cnn.yaml \
  dataset.type=random dataset.num_samples=128 dataset.val_samples=32 \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5

# GPU-first with NumPy fallback
minicnn train-native --config configs/dual_backend_cnn.yaml \
  engine.execution_mode=gpu_native_auto \
  train.device=cuda \
  dataset.type=random dataset.num_samples=128 dataset.val_samples=32
```

See [docs/cuda_native.md](docs/cuda_native.md) for the full guide.
See [docs/cuda_native_expansion_plan.md](docs/cuda_native_expansion_plan.md) for the staged expansion direction.
See [docs/cuda_native_productionization_plan.md](docs/cuda_native_productionization_plan.md) for the path from experimental backend to implementation-grade public contract.
See [docs/cuda_native_amp_graduation_checklist.md](docs/cuda_native_amp_graduation_checklist.md) for the AMP graduation gate that moved the backend to beta.
See [docs/cuda_native_gpu_enablement_plan.md](docs/cuda_native_gpu_enablement_plan.md) for the separate path from NumPy reference execution to real GPU execution.
See [docs/cuda_native_gpu_cifar10_runbook.md](docs/cuda_native_gpu_cifar10_runbook.md) for the current full CIFAR-10 strict `gpu_native` training runbook.

Real-dataset demo:

```bash
PYTHONPATH=src python3 examples/cuda_native_amp_cifar10_beta_demo.py \
  --data-root data/cifar-10-batches-py \
  --artifacts-root /tmp/minicnn_cuda_native_beta_demo
```

## Quick Start

# Linux / macOS

```bash
git clone https://github.com/s9213712/minicnn.git
cd minicnn
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[torch,dev]
minicnn smoke
pytest
```

# Windows PowerShell

```powershell
git clone https://github.com/s9213712/minicnn.git
cd minicnn
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .[dev]
minicnn smoke
pytest
```

If PowerShell blocks `.venv\Scripts\Activate.ps1`, run this in the same shell
first and then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

If `minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch`
still does not use the GPU on Windows, verify that the environment is really
using a CUDA-enabled PyTorch wheel:

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

`minicnn smoke` is the recommended first check after install. It verifies the
repo layout, parses built-in configs, runs a small compiler trace, and validates
both the `cuda_legacy` and `cuda_native` config validation boundaries.

If PyTorch is missing, `minicnn smoke` now returns a warning instead of
pretending the repo is broken. In JSON mode, use `torch_available=false` and
`flex_registry_ready=false` as the signal that the optional torch/flex path is
not ready yet.

## Minimum Dependency Matrix

| Command / feature | PyTorch | native library (`.so`/`.dll`) | CIFAR-10 data |
|---|---:|---:|---:|
| `minicnn --help` | no | no | no |
| `minicnn validate-dual-config` | no | no | no |
| `minicnn show-cuda-mapping` | no | no | no |
| `minicnn show-model` | no | no | no |
| `minicnn show-graph` | no | no | no |
| `minicnn compile` | no | no | no |
| `minicnn train-flex` | yes | no | depends on dataset |
| `minicnn train-dual engine.backend=torch` | yes | no | depends on dataset |
| `minicnn train-dual engine.backend=cuda_legacy` | no | yes | yes |
| `minicnn train-autograd` | no | no | depends on dataset |
| `minicnn train-native` | no | no | depends on dataset |

If a command needs PyTorch, the CLI now fails with a short dependency message
instead of an import-time traceback.

Config and override mistakes also fail with a short message and exit code `2`
instead of a Python traceback. `healthcheck`, `doctor`, `smoke`,
`validate-*`, `show-cuda-mapping`, `show-model`, `show-graph`, and `inspect-checkpoint` now emit
JSON-friendly output, and `train.device=cuda` fails early with guidance to use
`train.device=auto` or `train.device=cpu` when CUDA is unavailable.

Those inspection and validation commands also support `--format text` for
quick terminal use:

```bash
minicnn healthcheck --format json
minicnn doctor --format text
minicnn smoke --format json
minicnn validate-dual-config --format text
minicnn show-model --config configs/flex_cnn.yaml --format text
minicnn show-graph --config configs/flex_cnn.yaml --format json
minicnn inspect-checkpoint --path artifacts/models/example_best.pt --format text
```

`show-model` stays at the frontend/config view and preserves composite layer
names. `show-graph` shows the primitive graph traced by the compiler path after
basic optimization passes.

## Repo-First Resource Model

MiniCNN is still a repo-first tool. Built-in configs such as
`configs/flex_cnn.yaml` and `configs/dual_backend_cnn.yaml` resolve relative to
the project root when needed, but they are not packaged as a standalone
resource bundle yet.

That means:

- editable installs inside a repo checkout are the primary supported workflow
- `config-template` and `dual-config-template` are the portable built-ins
- if you need a fully packaged toolchain, use explicit config paths instead of
  assuming repo files exist in site-packages
- if a built-in config still cannot be found, the CLI now tells you to pass an
  explicit `--config` path instead of failing with a traceback

## Build The Native CUDA Library

```bash
minicnn build --legacy-make --check
```

Build both native variants:

```bash
minicnn build --legacy-make --variant both --check
```

Typical outputs:

```text
cpp/libminimal_cuda_cnn_cublas.so
cpp/libminimal_cuda_cnn_handmade.so
```

The native library is lazy-loaded, so commands such as `minicnn --help`,
`prepare-data`, `validate-dual-config`, and torch-only runs do not require a
built `.so`.

For the manually validated Windows path, see
[docs/guide_windows_build.md](docs/guide_windows_build.md). The recorded helper
workflow emits DLLs under `cpp\Release\`.

For a direct ctypes load check before training, use:

```bash
python3 -u examples/mnist_ctypes/check_native_library.py --variant handmade
```

## Prepare Data

Download CIFAR-10 for the handcrafted CUDA path:

```bash
minicnn prepare-data
```

MNIST-based flex/autograd configs can use `dataset.download=true` and download
their own files into `data/mnist/`.

## Common Training Commands

Train the flexible torch path:

```bash
minicnn train-flex --config configs/flex_cnn.yaml
```

Train with the shared dual-backend config on torch:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=torch
```

Train the handcrafted CUDA path:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

Select the native CUDA variant explicitly:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=cublas

minicnn train-dual --config configs/dual_backend_cnn.yaml \
  engine.backend=cuda_legacy runtime.cuda_variant=handmade
```

Windows users can verify a built DLL directly before training:

```powershell
python -u examples\mnist_ctypes\check_native_library.py --variant handmade
python -u examples\mnist_ctypes\check_native_library.py --path cpp\minimal_cuda_cnn_cublas.dll
```

Train the NumPy autograd path:

```bash
minicnn train-autograd --config configs/autograd_tiny.yaml
```

Compare backends:

```bash
minicnn compare --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

Inspect the current surface:

```bash
minicnn info
minicnn smoke
minicnn doctor
minicnn healthcheck
minicnn list-flex-components
minicnn list-dual-components
minicnn validate-dual-config --config configs/dual_backend_cnn.yaml
minicnn show-cuda-mapping --config configs/dual_backend_cnn.yaml
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn evaluate-checkpoint --config configs/dual_backend_cnn.yaml \
  --summary artifacts/example-run/summary.json
minicnn cuda-native-capabilities
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

Built-in config paths such as `configs/flex_cnn.yaml` and
`configs/dual_backend_cnn.yaml` are resolved relative to the project root when
needed, so they still work even if you launch the CLI from outside the repo
root. This is a repo-first convenience layer, not a full packaged-resource
system.

## Model Artifacts

Model files are not unified across all backends.

- torch paths save `.pt` checkpoints with `model_state`
- autograd saves `.npz` state dict arrays
- `cuda_native` saves flat `.npz` parameter dicts
- `cuda_legacy` saves handcrafted `.npz` runtime checkpoints

Use `summary.json` to find `best_model_path`, and use:

```bash
minicnn inspect-checkpoint --path artifacts/models/example_best.pt
minicnn evaluate-checkpoint --config configs/dual_backend_cnn.yaml \
  --summary artifacts/example-run/summary.json
minicnn export-torch-checkpoint --path artifacts/models/example_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/example_autograd_export.pt
```

for a quick schema view and a reproducible torch/flex test-set evaluation.

For real-image inference, use the repo example that resizes/crops large photos
into the configured input shape before running prediction:

```bash
python -u examples/inference/predict_image.py \
  --config configs/dual_backend_cnn.yaml \
  --summary artifacts/example-run/summary.json \
  --image path/to/photo.jpg \
  --topk 5
```

Full format and reuse guidance lives in [docs/model_artifacts.md](docs/model_artifacts.md).

Train the cuda_native path (GPU-first when `engine.cuda_native_execution=gpu_native`):

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

## Backend Boundary

The project-level frontend is broader than `cuda_legacy`.

That distinction matters:

- `torch` is the default place for new model ideas
- `torch/flex` is the reference implementation and first stop for new features
- `autograd` is the internal oracle for correctness-oriented checks
- `cuda_native` is the primary native direction, but tracked by backend readiness tiers
- `cuda_legacy` is kept for maintenance inside its validator-enforced boundary, not as the default place for new capability growth

See [docs/backend_capabilities.md](docs/backend_capabilities.md) for the
support matrix and [docs/generalization_roadmap.md](docs/generalization_roadmap.md)
for the longer-term direction.

## Config Interface

The main shared-config surface is:

- `dataset`
- `model.layers`
- `train`
- `loss`
- `optimizer`
- `scheduler`
- `engine.backend`

Minimal example:

```yaml
engine:
  backend: torch

dataset:
  type: cifar10
  input_shape: [3, 32, 32]
  num_classes: 10

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

If that same config does not fit `cuda_legacy`, use `minicnn validate-dual-config`
to see the exact compatibility errors instead of guessing.

## Extensibility

### Custom components

Torch/flex accepts dotted-path layer factories in `model.layers[].type`.

Example:

```yaml
model:
  layers:
    - type: Flatten
    - type: Linear
      out_features: 32
    - type: minicnn.extensions.custom_components.ConvBNReLU
      out_channels: 32
```

See [docs/custom_components.md](docs/custom_components.md).

## Documentation

Start here:

- [USAGE.md](USAGE.md): full documentation guide and reading order
- [docs/architecture.md](docs/architecture.md): overall architecture and module map
- [docs/backend_capabilities.md](docs/backend_capabilities.md): backend support matrix
- [docs/dual_backend_guide.md](docs/dual_backend_guide.md): shared-config routing and backend boundaries
- [docs/cuda_native.md](docs/cuda_native.md): primary native backend guide
- [docs/custom_components.md](docs/custom_components.md): dotted-path component extension points
- [docs/model_artifacts.md](docs/model_artifacts.md): checkpoint formats, reuse boundaries, and examples
- [templates/README.md](templates/README.md): ready-to-edit template configs
- [examples/README.md](examples/README.md): canonical example path and runnable example families

Background and reporting notes live under `docs/`, but [USAGE.md](USAGE.md)
now separates current operational docs from historical reports so they are not
all treated as the same kind of document.

## Repository Map

```text
minicnn/
├── cpp/                    # handcrafted CUDA / C++ backend
├── configs/                # example configs for flex, dual, and autograd paths
├── docs/                   # design notes, guides, and capability docs
├── examples/               # runnable examples, inference demos, and native ctypes samples
├── src/minicnn/
│   ├── flex/               # torch/flex frontend, registries, builder, trainer
│   ├── unified/            # shared-config dispatch and backend bridges
│   ├── training/           # cuda_legacy and autograd training code
│   ├── cuda_native/        # graph/planner/executor backend work
│   ├── nn/ ops/ optim/     # NumPy autograd stack
│   ├── compiler/ runtime/  # tracing, optimization, and CPU inference pipeline
│   └── core/               # native build helpers and ctypes CUDA binding
└── tests/                  # unit and smoke tests
```

## Philosophy

- explicit backend capability over vague parity claims
- one config frontend where that abstraction is actually honest
- fail fast on unsupported backend combinations
- keep backend readiness work visible without pretending it is stable

## License

MIT

## CUDA Native maintenance notes

- Full CIFAR-10 GPU-native smoke/benchmark runs should start from `configs/cifar10_cuda_native_gpu_stronger.yaml` and the runbook in `docs/cuda_native_gpu_cifar10_runbook.md`.
- `cuda_native` is now GPU-first for eligible `gpu_native` paths, while NumPy reference execution remains available as fallback/parity infrastructure.
- Large-file cleanup status is tracked in `docs/cuda_native_large_file_inventory.md`. Test modules are intentionally left intact for now because they preserve regression coverage.
