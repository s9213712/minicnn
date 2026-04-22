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

`train-autograd` config and override mistakes now fail with short CLI messages
and exit code `2`, matching the torch and dual-backend paths.

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

---

# MiniCNN Autograd（中文）

MiniCNN 內建一條精簡的 CPU/NumPy autograd 路徑，適合學習、框架層級測試，以及不希望依賴 PyTorch 的小型實驗。

核心模組位置：

- `src/minicnn/nn/tensor.py`：`Tensor`、reverse-mode autodiff、loss 函數
- `src/minicnn/autograd/function.py`：自訂可微分 `Function` API
- `src/minicnn/ops/`：NumPy 參考 op
- `src/minicnn/nn/layers.py`：輕量 layer module
- `src/minicnn/training/train_autograd.py`：CLI 訓練迴圈

這條路徑刻意設計為教學取向，能展示框架行為，但不以效能為目標。

## 支援功能

### 核心 tensor/autograd 能力

- `Tensor.backward()`：拓撲排序 reverse-mode autodiff
- 算術運算：`+`、`-`、`*`、`/`、`**`、`@`
- 支援 broadcast 的梯度
- `sum`、`mean`、`reshape`
- `no_grad()` 與 `Tensor.detach()`
- 透過 `Function.apply(...)` 自訂可微分 op

### Layer

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

### Loss 函數

- `CrossEntropyLoss`
- `MSELoss`
- `BCEWithLogitsLoss`

### Optimizer 與 scheduler

- `SGD`
- `Adam`
- `AdamW`
- `RMSprop`
- Step scheduler
- Cosine scheduler
- Per-parameter gradient clipping
- Weight decay

## 執行方式

使用 NumPy backend 訓練：

```bash
minicnn train-autograd --config configs/autograd_tiny.yaml
```

更完整的範例 config：

```text
configs/autograd_enhanced.yaml
```

`train-autograd` 的 config 或 override 錯誤，現在也會和 torch / dual-backend
路徑一樣，以簡短 CLI 訊息和 exit code `2` 失敗。

## 支援資料集

`train-autograd` 目前支援：

- `dataset.type=random`
- `dataset.type=cifar10`
- `dataset.type=mnist`

## Loss 介面說明

`CrossEntropyLoss` 期望傳入 class index。

`MSELoss` 會把 label 轉換成與輸出 shape 對應的 dense target。

`BCEWithLogitsLoss` 目前當作 binary classification 路徑，輸出層應為每個樣本輸出一個 logit，label 必須為 `0` 或 `1`。

## 最小範例

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

## 自訂可微分 Op

繼承 `Function` 並實作 `forward` 與 `backward`，透過 `MyOp.apply(...)` 呼叫，graph 連接會自動完成。

## 目前限制

- 僅支援 CPU/NumPy
- 不支援 AMP 或 mixed precision
- 比 `torch` 慢很多，不適合正式 CNN 訓練
- 適合學習與 parity-style 測試

## 相關文件

- [backend_capabilities.md](backend_capabilities.md)
- [architecture.md](architecture.md)
- [custom_components.md](custom_components.md)
