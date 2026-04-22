# MNIST ctypes examples

These examples demonstrate training a small CNN on MNIST using
MiniCNN's native library via Python `ctypes`. No PyTorch is required.

## Canonical entry point

**`train_mnist_so_full_cnn_frame.py`** — two-conv-layer CNN with clean Python
orchestration: `ConvBlock`, `DenseLayer`, dataclass caches, shape helpers, and
a standalone `SgdOptimizer`. This is the recommended starting point.

```bash
cd minicnn
make -C cpp
python3 -u examples/mnist_ctypes/train_mnist_so_full_cnn_frame.py --download
```

On Windows, the same script resolves `cpp/minimal_cuda_cnn_handmade.dll` by
default instead of the Linux `.so`.

## Minimal native-library smoke test

**`check_native_library.py`** — load the resolved native library through the
repo's ctypes bindings, verify required symbols, then run a GPU
upload/download round-trip.

Linux:

```bash
python3 -u examples/mnist_ctypes/check_native_library.py --variant handmade
```

Windows PowerShell:

```powershell
python -u examples\mnist_ctypes\check_native_library.py --variant handmade
python -u examples\mnist_ctypes\check_native_library.py --path cpp\minimal_cuda_cnn_cublas.dll
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
