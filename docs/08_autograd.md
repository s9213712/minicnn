# MiniCNN Autograd

MiniCNN includes a compact CPU/NumPy autograd path for learning, framework
tests, and small experiments that should not depend on PyTorch.

The core pieces live here:

- `src/minicnn/nn/tensor.py`: `Tensor`, reverse-mode autodiff, losses
- `src/minicnn/autograd/function.py`: custom differentiable `Function` API
- `src/minicnn/ops/`: NumPy reference ops
- `src/minicnn/nn/layers.py`: lightweight layer modules
- `src/minicnn/training/train_autograd.py`: CLI training loop

This path is intentionally educational. It is broad enough to demonstrate
framework behavior, but it is not intended to replace `torch` for performance.

## What It Supports

### Core tensor/autograd features

- `Tensor.backward()` with topological reverse-mode autodiff
- arithmetic: `+`, `-`, `*`, `/`, `**`, `@`
- broadcasting-aware gradients
- `sum`, `mean`, `reshape`
- `no_grad()` and `Tensor.detach()`
- custom differentiable ops through `Function.apply(...)`

### Layers

- `Linear`
- `Conv2d`
- `MaxPool2d`
- `AvgPool2d`
- `BatchNorm2d`
- `ResidualBlock`
- `Flatten`
- `ReLU`
- `LeakyReLU`
- `Sigmoid`
- `Tanh`
- `SiLU`
- `Dropout`

### Losses

- `CrossEntropyLoss`
- `MSELoss`
- `BCEWithLogitsLoss`

### Optimizers and scheduler support

- `SGD`
- `Adam`
- `AdamW`
- `RMSprop`
- step scheduler
- cosine scheduler
- per-parameter gradient clipping
- weight decay

## Running It

Train with the NumPy backend:

```bash
minicnn train-autograd --config configs/autograd_tiny.yaml
```

A broader example config lives at:

```text
configs/autograd_enhanced.yaml
```

## Dataset Support

`train-autograd` currently supports:

- `dataset.type=random`
- `dataset.type=cifar10`
- `dataset.type=mnist`

Examples:

```yaml
dataset:
  type: random
  num_samples: 256
  val_samples: 64
  input_shape: [1, 4, 4]
  num_classes: 2
```

```yaml
dataset:
  type: cifar10
  data_root: data/cifar-10-batches-py
  num_samples: 1000
  val_samples: 200
```

```yaml
dataset:
  type: mnist
  download: true
  num_samples: 10000
  val_samples: 2000
  input_shape: [1, 28, 28]
```

## Loss Contract Notes

`CrossEntropyLoss` expects class indices.

`MSELoss` converts labels into dense targets that match the output shape.

`BCEWithLogitsLoss` is currently treated as a binary classification path. The
output layer should produce one logit per example, and labels must be `0` or
`1`.

## Minimal Example

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

## Custom Differentiable Ops

Subclass `Function` and implement `forward` and `backward`. Call the op through
`MyOp.apply(...)` so the graph wiring happens automatically.

```python
from minicnn.autograd.function import Function
from minicnn.nn.tensor import Tensor


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return Tensor(x.data ** 2, requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2.0 * x.data
```

## Current Limits

- CPU/NumPy only
- no AMP or mixed precision
- much slower than `torch` for realistic CNN training
- useful for learning and parity-style tests, not for production throughput

If you need broad model experimentation, use `train-flex`.

If you need the handwritten CUDA path, stay inside the `cuda_legacy` validator
boundary exposed through `train-dual`.

## Related Docs

- [backend_capabilities.md](backend_capabilities.md)
- [architecture.md](architecture.md)
- [custom_components.md](custom_components.md)
