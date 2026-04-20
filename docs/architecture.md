# MiniCNN Architecture

MiniCNN has three independent training paths that share a common YAML config format. Choose based on your goal:

| Path | Command | Backend | GPU | Purpose |
|---|---|---|---|---|
| **flex** | `train-flex` | PyTorch | Yes | Research, custom layers, production |
| **dual** | `train-dual` | PyTorch or hand-written CUDA | Yes | Compare backends, learn CUDA internals |
| **autograd** | `train-autograd` | Pure NumPy | No | Learn autodiff, no dependencies |

---

## System diagram

```
                         YAML config
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         train-flex     train-dual    train-autograd
              │              │              │
              │       engine.backend=?      │
              │         /         \         │
              │      torch    cuda_legacy   │
              │        │           │        │
              ▼        ▼           ▼        ▼
         PyTorch   PyTorch   hand-written  NumPy
          model     model    CUDA .so      model
            │         │         │            │
            └────┬────┘         │            │
                 │              │            │
                 ▼              ▼            ▼
          flex/trainer    train_cuda.py  train_autograd.py
                 │              │            │
                 └──────────────┴────────────┘
                                │
                         models/ checkpoint
```

---

## Compiler / Runtime pipeline (static analysis + inference)

Separate from the training paths above, MiniCNN has a lightweight compiler + runtime stack for architecture inspection and NumPy inference:

```
  YAML model config
        │
        ▼
  compiler/tracer.py          trace_model_config()  →  IRGraph
        │
        ▼
  compiler/optimizer.py       optimize()            →  IRGraph (fused)
        │
        ▼
  runtime/pipeline.py         ir_to_runtime_graph() →  runtime Graph
        │
        ▼
  runtime/executor.py         GraphExecutor.run(x)  →  {node: output}
```

Use this path for architecture inspection and fast CPU inference without training overhead:

```python
from minicnn.runtime.pipeline import InferencePipeline
import numpy as np

model_cfg = {
    'layers': [
        {'type': 'Linear', 'in_features': 4, 'out_features': 8},
        {'type': 'ReLU'},
        {'type': 'Linear', 'in_features': 8, 'out_features': 2},
    ]
}

pipeline = InferencePipeline.from_config(model_cfg)
x = np.random.randn(16, 4).astype('float32')
logits = pipeline.run_final(x)
```

Or trigger via CLI:

```bash
minicnn compile --config configs/autograd_tiny.yaml
```

---

## Module map

```
src/minicnn/
├── nn/                  # autograd primitives
│   ├── tensor.py        # Tensor, Parameter, backward(), all ops
│   ├── layers.py        # Linear, Conv2d, BatchNorm2d, ReLU, Sigmoid, Tanh, Dropout, …
│   └── modules.py       # Module, Sequential base classes
├── ops/
│   └── nn_ops.py        # functional forms of all layer ops
├── autograd/
│   ├── function.py      # Function API for custom differentiable ops
│   └── context.py       # Context (save_for_backward)
├── optim/
│   ├── sgd.py           # SGD (momentum, weight_decay, grad_clip)
│   └── adam.py          # Adam (weight_decay, grad_clip)
│
├── compiler/            # static analysis pipeline
│   ├── tracer.py        # YAML config → IRGraph
│   ├── passes.py        # Conv+BN+ReLU fusion detection
│   ├── optimizer.py     # compose passes
│   ├── lowering.py      # IRGraph → backend descriptor
│   └── ir.py            # IRGraph, IRNode dataclasses
├── runtime/             # execution pipeline
│   ├── pipeline.py      # InferencePipeline (compiler → executor)
│   ├── executor.py      # GraphExecutor (runs a runtime Graph)
│   ├── graph.py         # runtime Graph, Node
│   ├── backend.py       # Backend ABC (NumPy, Torch, Cuda stubs)
│   ├── memory.py        # MemoryPool (buffer reuse)
│   └── profiler.py      # Profiler (timing context manager)
│
├── flex/                # PyTorch train-flex path
│   ├── config.py        # load_flex_config(), CLI override parsing
│   ├── builder.py       # build_model() from YAML layers list
│   ├── trainer.py       # training loop, checkpointing
│   ├── registry.py      # REGISTRY (flex-path component registration)
│   └── data.py          # DataLoader helpers
├── unified/             # train-dual shared entry
│   ├── config.py        # load_unified_config()
│   ├── cuda_legacy.py   # compile_to_legacy_experiment() bridge
│   └── trainer.py       # dispatch to torch or cuda_legacy
├── training/            # CUDA legacy path internals
│   ├── train_cuda.py    # epoch loop, evaluation, checkpointing
│   ├── cuda_batch.py    # per-batch CUDA ops
│   ├── cuda_arch.py     # CudaNetGeometry (kernel dimensions)
│   └── train_autograd.py# NumPy autograd training loop
│
├── models/
│   ├── registry.py      # MODEL_REGISTRY (autograd layers)
│   ├── builder.py       # build_autograd_model() from YAML
│   └── shape_inference.py# Conv2d output shape helper
├── config/              # legacy ExperimentConfig loader
├── data/                # CIFAR-10, MNIST loaders
└── cli.py               # minicnn CLI entry point
```

---

## Adding a new layer

1. Implement the functional op in `ops/nn_ops.py` (forward + `_backward` closure).
2. Add a `Module` subclass in `nn/layers.py`.
3. Export from `nn/__init__.py`.
4. Register in `models/registry.py` `MODEL_REGISTRY` dict.
5. Add the config key `type: YourLayer` to any YAML config.

The layer is then usable in all three training paths without any other changes.

---

## Adding a custom differentiable op

Subclass `Function` and register in `MODEL_REGISTRY`:

```python
from minicnn.autograd.function import Function
from minicnn.nn.tensor import Tensor
import numpy as np

class Swish(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-x.data))
        ctx.save_for_backward(x)
        ctx.s = s
        return Tensor(x.data * s, requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        s = ctx.s
        return grad * (s + x.data * s * (1.0 - s))
```

See `docs/08_autograd.md` for the full Function API reference.
