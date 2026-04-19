# MiniCNN Autograd

MiniCNN includes a compact CPU/NumPy autograd stack. The core graph engine is in `src/minicnn/nn/tensor.py`, differentiable layer ops live in `src/minicnn/ops/`, and small layer modules live in `src/minicnn/nn/layers.py`.

It is meant for framework-level tests, small examples, and educational experiments that should not depend on torch.

It supports:

- `Tensor.backward()` with topological reverse-mode autodiff
- scalar and tensor arithmetic: `+`, `-`, `*`, `/`, `**`
- broadcasting-aware gradients
- matrix multiply with `@`
- `sum`, `mean`, and `reshape`
- `relu`, `log_softmax`, and `cross_entropy`
- trainable `Parameter`
- `no_grad()` and `Tensor.detach()`
- lightweight `SGD` and `Adam` optimizers through `src/minicnn/optim/`
- `Linear`, `ReLU`, `Flatten`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, and same-channel `ResidualBlock`
- a small random-data training entrypoint through `minicnn train-autograd`
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

## Tiny training run

```bash
PYTHONPATH=src python -m minicnn.cli train-autograd \
  --config configs/autograd_tiny.yaml \
  train.epochs=1 dataset.num_samples=16 dataset.val_samples=8 train.batch_size=4
```

The trainer currently supports `dataset.type=random`. It writes the best model to:

```text
src/minicnn/training/models/
```

Autograd checkpoints use the `*_autograd_best.npz` suffix. Metrics and summaries stay under the configured `project.artifacts_root`.

Inspect the lightweight MiniCNN IR generated from the same config with:

```bash
PYTHONPATH=src python -m minicnn.cli compile --config configs/autograd_tiny.yaml
```

## Backend boundary

This autograd engine is separate from the two training backends:

- `engine.backend: torch` uses PyTorch autograd.
- `engine.backend: cuda_legacy` uses explicit CUDA/C++ backward kernels.
- `src/minicnn/nn/tensor.py` is the MiniCNN CPU/NumPy autograd layer for tests and small examples.
- Native CUDA lowering, a CUDA-backed autograd bridge, and production-grade fused CUDA kernels are not claimed by this CPU/NumPy stack.

Run the autograd tests with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_autograd.py tests/test_autograd_stack.py
```

## Robustness notes

### `Tensor.__pow__` and division by zero

`x ** power` delegates to NumPy, so `0.0 ** -1` produces `inf` in the forward pass (expected). The backward pass, however, would compute `power * (0 ** (power - 1))` which yields `inf * 0 = NaN` for `power < 1`. This is now guarded with `np.where`:

```python
# base == 0 with negative power → treat gradient as 0 (not NaN)
if power < 1.0:
    base_grad = np.where(self.data != 0.0, power * (self.data ** (power - 1.0)), 0.0)
```

This means `Tensor.__truediv__` (which uses `other ** -1.0`) also benefits from this fix.

### SGD silent failures

Previously, any exception inside `SGD.step()` was caught and silently ignored. The guard now only catches `ValueError` and `TypeError` (shape and type mismatches that can legitimately arise for metadata-only parameters), and emits a `RuntimeWarning` that includes the parameter index and name. All other exceptions propagate normally.

```python
except (ValueError, TypeError) as exc:
    warnings.warn(f"SGD.step(): skipped param {i} ({p.name!r}): {exc}", RuntimeWarning)
```
