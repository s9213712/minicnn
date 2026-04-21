# MiniCNN Architecture

MiniCNN has one broad frontend surface and multiple backend-oriented execution
paths. The important distinction is that not every backend accepts the full
frontend contract.

## Stable User-Facing Paths

| Path | Command | Backend | Purpose |
|---|---|---|---|
| flex | `train-flex` | PyTorch | broad experimentation, custom components |
| dual | `train-dual` | `torch` or `cuda_legacy` | compare one shared config across two backends |
| autograd | `train-autograd` | NumPy | learning and framework-level experiments |

`cuda_native` exists on this branch under `src/minicnn/cuda_native/`, but it is
not yet part of the stable `train-dual` backend toggle.

## High-Level Layout

```text
shared YAML / CLI frontend
        |
        +--> train-flex ------> torch
        |
        +--> train-dual ------> torch | cuda_legacy
        |
        +--> train-autograd --> NumPy autograd
        |
        +--> branch-local cuda_native modules (experimental backend work)
```

## Stable Training Flow

```text
YAML config
    |
    +--> flex/config.py --------------------------> flex/trainer.py
    |
    +--> unified/config.py --> engine.backend? --> unified/trainer.py
    |                                              |-> torch path
    |                                              `-> cuda_legacy path
    |
    `--> train_autograd.py
```

## Compiler / Runtime Inference Pipeline

Separate from training, MiniCNN also has a lightweight tracing + runtime path
for architecture inspection and CPU inference:

```text
model config
    |
    -> compiler/tracer.py
    -> compiler/optimizer.py
    -> runtime/pipeline.py
    -> runtime/executor.py
```

Use this when you want graph inspection or CPU inference without entering the
training loops.

## Branch-Local `cuda_native`

This branch contains experimental native-backend work under
`src/minicnn/cuda_native/`:

- capability descriptor
- validators
- sequential graph IR
- planner
- reference forward/backward executors
- minimal backend utilities

But the official capability descriptor still marks it as:

- experimental
- sequential-only
- not yet supported as a stable training backend

Treat it as active backend R&D, not as a finished public backend.

## Module Map

```text
src/minicnn/
├── cli.py                 # public CLI entrypoint
├── flex/                  # torch/flex frontend: config, builder, trainer, registry, data
├── unified/               # shared-config loader and dispatch to torch/cuda_legacy
├── training/              # cuda_legacy orchestration and NumPy autograd trainer
├── framework/             # healthcheck/list-dual-components registry surface on this branch
├── compiler/              # tracer, optimizer passes, lowering boundary
├── runtime/               # runtime graph, backend abstraction, executor, memory, profiler
├── cuda_native/           # experimental native graph/planner/backend work
├── nn/                    # NumPy autograd modules and layers
├── ops/                   # differentiable NumPy ops
├── optim/                 # NumPy-side optimizers
├── schedulers/            # NumPy-side schedulers
├── models/                # NumPy model registry, builder, shape inference
├── config/                # ExperimentConfig schema and legacy settings bridge
├── core/                  # native build helpers and ctypes CUDA binding
└── data/                  # CIFAR-10 and MNIST loaders
```

## Backend Boundaries

The project frontend is intentionally broader than `cuda_legacy`.

In practice:

- torch/flex is the default home for new layer ideas
- `cuda_legacy` is a narrow backend with validator-enforced limits
- the autograd path is for learning and tests, not throughput
- `cuda_native` should become its own backend, not a hidden extension of `cuda_legacy`

See [backend_capabilities.md](backend_capabilities.md) for the support matrix
and [dual_backend_guide.md](dual_backend_guide.md) for change-impact guidance.

## Adding New Functionality

### New torch/flex layer

Add or import the layer on the torch/flex side and keep the change scoped there
unless another backend also needs it.

### New NumPy autograd op

Implement the differentiable op in `src/minicnn/ops/`, add the corresponding
layer/module in `src/minicnn/nn/`, and register it through the NumPy model
builder path.

### New `cuda_legacy` training op

Expect to touch both Python and native code:

- `src/minicnn/unified/cuda_legacy.py`
- `src/minicnn/training/`
- `src/minicnn/core/cuda_backend.py`
- `cpp/src/`

### New `cuda_native` capability

Keep it inside the branch-local `src/minicnn/cuda_native/` pipeline until the
capability descriptor, validation surface, and CLI story are all coherent.
