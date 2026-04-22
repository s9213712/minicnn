# MiniCNN

[繁體中文 README](README.zh-TW.md)

![status](https://img.shields.io/badge/status-experimental-orange)
![frontend](https://img.shields.io/badge/frontend-YAML%20%2B%20CLI-blue)
![native](https://img.shields.io/badge/native-CUDA-green)

MiniCNN is a configuration-driven deep learning project for studying the gap
between a flexible frontend and backend-constrained execution paths.

Today, the repo gives you four backend roles:

- `torch` via `train-flex` / `train-dual` as the broad reference implementation
- `cuda_native` via `train-native` as the primary native backend direction
- `autograd` via `train-autograd` as the internal correctness oracle
- `cuda_legacy` via `train-dual` as the maintenance-only historical CUDA path

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
| `cuda_native` | primary native backend | experimental, graph-based, sequential-only, active growth path |
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
                                -> cuda_native [PRIMARY NATIVE] (experimental graph IR, planner, numpy executor)
                               \
                                -> cuda_legacy [MAINTENANCE ONLY] (historical handwritten CUDA path)
```

## Recent Engineering Progress

The current cleanup and refactor work has pushed more orchestration logic out of
the hottest entrypoint files without changing the public CLI surface.

Recent passes include:

- CLI dispatch now splits parser construction, readonly commands, and
  training/compare commands into focused helper modules
- `flex` training now separates context setup, run orchestration, reporting,
  device resolution, and step-level execution
- `cuda_native` unified training now separates bridge helpers, runtime loop
  helpers, and support/reporting helpers
- artifact inspection/export and legacy checkpoint payload handling now live in
  dedicated helper layers instead of staying mixed into larger modules

The result is the same user-facing command set, but with narrower module
responsibilities and cleaner rollback points for deeper native optimization work.

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

### `cuda_native` (Primary Native Direction, Experimental)

The active native growth path in the repo, built as a graph-based backend with:

- explicit graph IR (`graph.py`, `nodes.py`)
- strict validation (`validators.py`, `shapes.py`)
- conservative buffer planning (`planner.py`)
- numpy reference kernels (`kernels.py`, `executor.py`)
- backward prototype and SGD training loop
- layout validation (`layouts.py` — `validate_graph_layouts()`)
- memory estimation and reuse (`memory.py` — `memory_footprint()`, `BufferPool`)
- observability tooling (`debug.py` — `dump_graph()`, `dump_plan()`, `TracingForwardExecutor`)

Supported ops: `BatchNorm2d` (forward/backward prototype), `Conv2d`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `SiLU`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `Linear`.

Current validated support boundary:

- datasets: `random`, `cifar10`, `mnist`
- losses: `CrossEntropyLoss`, `MSELoss`
- optimizer: `SGD` with optional momentum and global gradient clipping
- scheduler: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, or disabled
- `train.amp=false`, `train.grad_accum_steps=1`

Backward and training prototypes exist, but the backend is still experimental, sequential-only, and not production-ready yet. It is the backend that should grow next; `cuda_legacy` remains a narrow maintenance path.

```bash
# Check what cuda_native supports
minicnn cuda-native-capabilities

# Validate your config
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5

# Run (research only)
minicnn train-native --config configs/dual_backend_cnn.yaml \
  dataset.type=random dataset.num_samples=128 dataset.val_samples=32 \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5
```

See [docs/cuda_native.md](docs/cuda_native.md) for the full guide.

## Quick Start

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

`minicnn smoke` is the recommended first check after install. It verifies the
repo layout, parses built-in configs, runs a small compiler trace, and validates
both the `cuda_legacy` and `cuda_native` config validation boundaries.

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
minicnn export-torch-checkpoint --path artifacts/models/example_autograd_best.npz \
  --config configs/autograd_tiny.yaml \
  --output artifacts/models/example_autograd_export.pt
```

for a quick schema view.

Full format and reuse guidance lives in [docs/model_artifacts.md](docs/model_artifacts.md).

Train the experimental cuda_native path:

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
- `cuda_native` is the primary native direction, but still experimental
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

Background and reporting notes live under `docs/`, but [USAGE.md](USAGE.md)
now separates current operational docs from historical reports so they are not
all treated as the same kind of document.

## Repository Map

```text
minicnn/
├── cpp/                    # handcrafted CUDA / C++ backend
├── configs/                # example configs for flex, dual, and autograd paths
├── docs/                   # design notes, guides, and capability docs
├── examples/               # custom torch component examples
├── src/minicnn/
│   ├── flex/               # torch/flex frontend, registries, builder, trainer
│   ├── unified/            # shared-config dispatch and backend bridges
│   ├── training/           # cuda_legacy and autograd training code
│   ├── cuda_native/        # experimental graph/planner/executor backend work
│   ├── nn/ ops/ optim/     # NumPy autograd stack
│   ├── compiler/ runtime/  # tracing, optimization, and CPU inference pipeline
│   └── core/               # native build helpers and ctypes CUDA binding
└── tests/                  # unit and smoke tests
```

## Philosophy

- explicit backend capability over vague parity claims
- one config frontend where that abstraction is actually honest
- fail fast on unsupported backend combinations
- keep experimental backend work visible without pretending it is stable

## License

MIT
