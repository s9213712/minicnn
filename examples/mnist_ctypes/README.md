# MNIST ctypes examples

These examples demonstrate training a small CNN on MNIST using
`cpp/libminimal_cuda_cnn.so` via Python `ctypes`. No PyTorch is required.

## Canonical entry point

**`train_mnist_so_full_cnn_frame.py`** — two-conv-layer CNN with clean Python
orchestration: `ConvBlock`, `DenseLayer`, dataclass caches, shape helpers, and
a standalone `SgdOptimizer`. This is the recommended starting point.

```bash
cd minicnn
make -C cpp
python3 -u examples/mnist_ctypes/train_mnist_so_full_cnn_frame.py --download
```

## Historical progression (`legacy/`)

The `legacy/` folder contains earlier iterations kept for reference:

| File | Description |
|------|-------------|
| `train_mnist_so.py` | Minimal: single conv layer, flat orchestration |
| `train_mnist_so_full.py` | Expanded: more complete training loop |
| `train_mnist_so_full_cnn.py` | Two conv layers, still flat orchestration |

Each file is self-contained (NumPy + stdlib only). Read them in order if you
want to see how the design evolved toward the canonical version.
