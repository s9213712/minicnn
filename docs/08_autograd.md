# MiniCNN Autograd

MiniCNN includes a compact CPU/NumPy autograd core in `src/minicnn/nn/tensor.py`. It is meant for framework-level tests, small examples, and educational experiments that should not depend on torch.

It supports:

- `Tensor.backward()` with topological reverse-mode autodiff
- scalar and tensor arithmetic: `+`, `-`, `*`, `/`, `**`
- broadcasting-aware gradients
- matrix multiply with `@`
- `sum`, `mean`, and `reshape`
- `relu`, `log_softmax`, and `cross_entropy`
- trainable `Parameter`
- lightweight `SGD` updates through `src/minicnn/optim/sgd.py`

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

## Backend boundary

This autograd engine is separate from the two training backends:

- `engine.backend: torch` uses PyTorch autograd.
- `engine.backend: cuda_legacy` uses explicit CUDA/C++ backward kernels.
- `src/minicnn/nn/tensor.py` is the MiniCNN CPU/NumPy autograd layer for tests and small examples.

Run the autograd tests with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_autograd.py
```
