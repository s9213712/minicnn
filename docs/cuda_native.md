# cuda_native Backend

`cuda_native` is an experimental graph-based backend for MiniCNN.

It is **not** a replacement for `cuda_legacy` and is **not** production-ready.
It is a research prototype for graph IR, explicit memory planning, and
backend-extensibility work.

## What cuda_native Is

A staged, modular backend structured in layers:

- **IR layer** (`graph.py`, `nodes.py`) — graph and tensor representation
- **Validation layer** (`validators.py`, `shapes.py`) — shape inference and legality checks
- **Planning layer** (`planner.py`) — conservative buffer allocation
- **Execution layer** (`executor.py`, `kernels.py`) — numpy reference kernels, dispatch table
- **Capability layer** (`capabilities.py`) — honest feature reporting

## What cuda_native Is Not

- Not a replacement for `cuda_legacy`
- Not a production training backend
- Not compatible with CUDA `.so` dispatch (uses numpy reference kernels)
- Not general-purpose (sequential graphs only, no branching)

## Current Status

| Feature | Status |
|---|---|
| Graph IR | Implemented |
| Shape inference | Basic |
| Forward execution | Basic (numpy) |
| Planner | Conservative / experimental |
| Pooling (MaxPool2d, AvgPool2d) | Partial |
| Backward prototype | Implemented but not stable |
| Training loop | Research prototype |
| Training enabled in MVP | No — disabled |
| Dynamic graph | Not supported |
| Mixed precision | Not supported |

## Supported Ops

Forward (sequential, validated):

- `Conv2d`
- `ReLU`
- `LeakyReLU`
- `MaxPool2d`
- `AvgPool2d`
- `Flatten`
- `Linear`

Unsupported (rejected at validation):

- `BatchNorm2d`
- `GroupNorm`
- `LayerNorm`
- `ResidualBlock`

## How It Differs From cuda_legacy

| | cuda_legacy | cuda_native |
|---|---|---|
| Kernel type | Real CUDA / cuBLAS | NumPy reference |
| Graph | Fixed handcrafted pipeline | Explicit graph IR |
| Validation | Strict contract check | Graph-level shape and op check |
| Planner | Implicit | Explicit buffer plan |
| Training | Stable | Research prototype |
| Extension model | Not extensible | Designed to grow |

## CLI Usage

Check capabilities:

```bash
minicnn cuda-native-capabilities
```

Validate a config:

```bash
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

Run (experimental, research only):

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml train.epochs=1 dataset.num_samples=128
```

Or via `train-dual`:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_native
```

## Architecture Overview

```text
Config
  └─ validators.py         (op legality, shape constraints)
  └─ graph.py + nodes.py   (IR: NativeGraph, Node, TensorSpec)
  └─ shapes.py             (shape inference per op)
  └─ planner.py            (buffer allocation plan)
  └─ executor.py           (forward dispatch)
  └─ kernels.py            (numpy reference kernels)
  └─ backward.py           (backward kernels — prototype)
  └─ loss.py               (cross_entropy, mse)
  └─ training.py           (train_step, sgd_update)
  └─ capabilities.py       (honest feature flags)
```

## Design Principles

1. Explicit over implicit — no hidden behavior
2. Fail fast — reject unsupported ops at validation time
3. Honest capability boundaries — never claim support beyond tested reality
4. Correctness before optimization — conservative planner, no clever tricks until stable
5. Separation of concerns — IR, planner, execution are distinct layers

## Roadmap Summary

- Phase 0: Scaffold (done)
- Phase 1: IR + forward execution (done)
- Phase 2: Planner + pooling (done)
- Phase 3: Backward prototype (done, not stable)
- Phase 4: MVP stabilization — CLI integration, honest capabilities, docs (current)
- Phase 5: Future evolution — autograd, optimizer stack, broader op coverage

See `comments/cuda_native/cuda_native_roadmap.md` for the full vision.
