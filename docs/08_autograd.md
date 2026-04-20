# MiniCNN Autograd

MiniCNN includes a compact CPU/NumPy autograd stack. The core graph engine is in `src/minicnn/nn/tensor.py`, differentiable layer ops live in `src/minicnn/ops/`, and small layer modules live in `src/minicnn/nn/layers.py`.

It is meant for framework-level tests, small examples, and educational experiments that should not depend on PyTorch.

It supports:

- `Tensor.backward()` with topological reverse-mode autodiff
- scalar and tensor arithmetic: `+`, `-`, `*`, `/`, `**`
- broadcasting-aware gradients
- matrix multiply with `@`
- `sum`, `mean`, and `reshape`
- `relu`, `log_softmax`, and `cross_entropy`
- trainable `Parameter`
- `no_grad()` and `Tensor.detach()`
- lightweight `SGD` and `Adam` optimizers with optional `grad_clip` and `weight_decay`
- `Linear`, `ReLU`, `Flatten`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, and same-channel `ResidualBlock`
- custom differentiable ops via the `Function` API
- training on random, CIFAR-10, or MNIST data through `minicnn train-autograd`
- lightweight config tracing and IR inspection through `minicnn compile`

## Minimal example

```python
import numpy as np

from minicnn.nn import Parameter, Tensor, cross_entropy
from minicnn.optim.sgd import SGD

w = Parameter([[0.1, -0.2], [0.3, 0.4]], name="w")
x = Tensor([[1.0, 2.0]])
target = np.array([1])

logits = x @ w
loss = cross_entropy(logits, target)
loss.backward()

SGD([w], lr=0.1).step()
```

## Tiny training run (random data)

```bash
minicnn train-autograd \
  --config configs/autograd_tiny.yaml \
  train.epochs=5 dataset.num_samples=64 dataset.val_samples=16 train.batch_size=8
```

## Training on real datasets

The `train-autograd` command supports three dataset types.  Note that the NumPy Conv2d implementation is slow — use `train-flex` or `train-dual` for production CIFAR-10 training.

### Random data (default, fast)

```yaml
dataset:
  type: random
  num_samples: 256
  val_samples: 64
  input_shape: [1, 4, 4]
  num_classes: 2
```

### CIFAR-10 (requires `minicnn prepare-data` first)

```yaml
dataset:
  type: cifar10
  data_root: data/cifar-10-batches-py
  num_samples: 1000
  val_samples: 200
```

```bash
minicnn train-autograd --config configs/autograd_tiny.yaml \
  dataset.type=cifar10 dataset.num_samples=1000 dataset.val_samples=200 \
  model.layers.0.type=Conv2d model.layers.0.out_channels=16
```

### MNIST (auto-downloads on first run)

```yaml
dataset:
  type: mnist
  download: true
  num_samples: 10000
  val_samples: 2000
  input_shape: [1, 28, 28]
```

## Custom differentiable operations (Function API)

Subclass `Function` and implement `forward` and `backward`.  Call via `MyOp.apply(*inputs)`:

```python
from minicnn.autograd.function import Function
from minicnn.nn.tensor import Tensor
import numpy as np

class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return Tensor(x.data ** 2, requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2.0 * x.data

x = Tensor([3.0, -2.0], requires_grad=True)
y = Square.apply(x)
y.sum().backward()
# x.grad == [6.0, -4.0]
```

`apply()` automatically wires the backward hook and sets `requires_grad=True` on the output when any input requires a gradient.

## Optimizers

Both `SGD` and `Adam` support `weight_decay` and `grad_clip`:

```python
from minicnn.optim.sgd import SGD
from minicnn.optim.adam import Adam

# SGD with momentum, weight decay, and gradient clipping
opt = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, grad_clip=5.0)

# Adam with weight decay and gradient clipping
opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4, grad_clip=1.0)
```

Gradient clipping clips each parameter's gradient to unit norm scaled by `grad_clip` before the update step.

In YAML:

```yaml
optimizer:
  type: SGD
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  grad_clip: 5.0
```

## LR scheduler (step decay)

The autograd trainer supports a simple step-decay scheduler via the `scheduler` config key:

```yaml
scheduler:
  enabled: true
  step_size: 10    # reduce every N epochs
  gamma: 0.5       # multiply lr by gamma
  min_lr: 1e-6     # floor
```

## Best model output

The trainer writes the best checkpoint (by validation accuracy) to:

```text
src/minicnn/training/models/<run_name>_autograd_best.npz
```

Load it back with `numpy.load()`:

```python
import numpy as np
data = np.load('path/to/best.npz')
# data contains named arrays matching model.state_dict() keys
```

## Layers available in the autograd path

| Layer | Config key `type` | Notes |
|---|---|---|
| `Linear` | `Linear` | |
| `Conv2d` | `Conv2d` | |
| `MaxPool2d` | `MaxPool2d` | |
| `BatchNorm2d` | `BatchNorm2d` | |
| `ReLU` | `ReLU` | |
| `Sigmoid` | `Sigmoid` | |
| `Tanh` | `Tanh` | |
| `Dropout` | `Dropout` | active in `training=True`, pass-through in `eval()` |
| `Flatten` | `Flatten` | |
| `ResidualBlock` | `ResidualBlock` | same-channel skip connection |

Add custom layers to `MODEL_REGISTRY` in `src/minicnn/models/registry.py`.

## Compile command and inference pipeline

```bash
minicnn compile --config configs/autograd_tiny.yaml
```

This traces the model config into a lightweight IR graph, applies optimization passes (identity removal, Conv+BN+ReLU fusion annotation), and prints a JSON summary.

The compiled graph is also runnable via `InferencePipeline` — no training loop required:

```python
from minicnn.runtime.pipeline import InferencePipeline
import numpy as np

model_cfg = {'layers': [
    {'type': 'Linear', 'in_features': 4, 'out_features': 16},
    {'type': 'ReLU'},
    {'type': 'Linear', 'in_features': 16, 'out_features': 3},
]}

pipeline = InferencePipeline.from_config(model_cfg, profile=True)
logits = pipeline.run_final(np.random.randn(8, 4).astype('float32'))
print(pipeline.profile_summary())
```

`from_config()` internally calls `trace_model_config → optimize → ir_to_runtime_graph → GraphExecutor`.

## Interactive tutorial

`notebooks/01_autograd_from_scratch.ipynb` walks through the engine from first principles — computation graph, backward(), Function API, training loop, and the pipeline — with no PyTorch dependency.

## Limitations

- Thread-safety: the global `_grad_enabled` flag in `nn/tensor.py` is not thread-safe.
- Performance: NumPy Conv2d is not optimised; large CIFAR-10 runs are much slower than PyTorch.
- No GPU support in the autograd path.
