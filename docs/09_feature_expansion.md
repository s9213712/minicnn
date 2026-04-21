# 第09章：功能擴充指南（Phases 0–12）

本章說明 MiniCNN 在 phases 0–12 所新增的功能，並提供 YAML 設定範例與執行指令。

---

## 概覽

新增功能分為六個主要類別：

| 類別 | 新增項目 |
|---|---|
| 激活函數 | LeakyReLU、SiLU（含原有 ReLU、Sigmoid、Tanh） |
| 優化器 | AdamW、RMSprop |
| 學習率排程器 | StepLR、CosineAnnealingLR |
| 權重初始化 | kaiming_uniform/normal、xavier_uniform/normal、normal_init、zeros_init |
| 資料增強 | random_crop、horizontal_flip（flex 路徑） |
| Block Preset | conv_relu、conv_bn_relu、conv_bn_silu（flex 路徑） |

---

## 1. 激活函數

### Python 使用

```python
from minicnn.nn.layers import LeakyReLU, SiLU
from minicnn.nn.tensor import Tensor
import numpy as np

x = Tensor(np.array([[-1.0, 0.5, 2.0]], dtype=np.float32), requires_grad=True)
y = LeakyReLU(negative_slope=0.01)(x)
y.sum().backward()
# x.grad 包含 LeakyReLU 的梯度
```

### YAML 模型設定

```yaml
model:
  layers:
    - type: Conv2d
      out_channels: 16
      kernel_size: 3
    - type: LeakyReLU
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
    - type: SiLU
    - type: Flatten
    - type: Linear
      out_features: 10
```

### 範例腳本

```bash
python examples/feature_expansion/01_activations.py
```

---

## 2. 優化器

### Python 使用

```python
from minicnn.optim.adamw import AdamW
from minicnn.optim.rmsprop import RMSprop
```

### YAML 設定

AdamW：

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.01
  grad_clip: 1.0
```

RMSprop：

```yaml
optimizer:
  type: RMSprop
  lr: 0.01
  alpha: 0.99
  weight_decay: 0.0
  momentum: 0.0
```

### CLI 指令

```bash
minicnn train-autograd --config configs/autograd_enhanced.yaml \
  optimizer.type=AdamW optimizer.lr=0.001 optimizer.weight_decay=0.01
```

### 範例腳本

```bash
python examples/feature_expansion/02_optimizers.py
```

---

## 3. 學習率排程器

### YAML 設定

StepLR（每 N 個 epoch 乘以 gamma）：

```yaml
scheduler:
  enabled: true
  type: step
  step_size: 10
  gamma: 0.5
  min_lr: 1.0e-6
```

CosineAnnealingLR（餘弦退火）：

```yaml
scheduler:
  enabled: true
  type: cosine
  T_max: 30
  min_lr: 1.0e-5
```

### CLI 指令

```bash
minicnn train-autograd --config configs/autograd_enhanced.yaml \
  scheduler.enabled=true scheduler.type=cosine scheduler.T_max=20
```

### 範例腳本

```bash
python examples/feature_expansion/03_schedulers.py
```

---

## 4. 權重初始化

### Python 使用

```python
from minicnn.models.initialization import get_initializer

init_fn = get_initializer('kaiming_uniform')
w = init_fn((64, 32, 3, 3))   # conv 層權重

init_fn2 = get_initializer('xavier_normal')
w2 = init_fn2((256, 128))     # linear 層權重
```

支援策略名稱：`kaiming_uniform`、`kaiming_normal`、`xavier_uniform`、`xavier_normal`、`normal`、`zeros`、`he`（`kaiming_uniform` 的別名）。

### 範例腳本

```bash
python examples/feature_expansion/04_initialization.py
```

---

## 5. Label Smoothing

`cross_entropy` 函數支援 `label_smoothing` 參數，可緩解過度自信問題。

### Python 使用

```python
from minicnn.nn.tensor import Tensor, cross_entropy
import numpy as np

logits = Tensor(np.array([[2.0, 1.0, 0.5]], dtype=np.float32), requires_grad=True)
loss = cross_entropy(logits, np.array([0]), label_smoothing=0.1)
loss.backward()
```

### YAML 設定

```yaml
loss:
  type: CrossEntropyLoss
  label_smoothing: 0.1
```

### 範例腳本

```bash
python examples/feature_expansion/05_label_smoothing.py
```

---

## 6. 完整訓練範例（autograd 路徑）

使用新功能進行端對端訓練：

```bash
minicnn train-autograd --config configs/autograd_enhanced.yaml
```

或參閱範例腳本（不依賴 PyTorch）：

```bash
python examples/feature_expansion/06_train_autograd_enhanced.py
```

---

## 7. Block Presets（flex 路徑，需要 PyTorch）

Flex builder 支援以下 preset 類型，一行設定自動展開為多個子層：

| Preset | 展開為 |
|---|---|
| `conv_relu` | Conv2d → ReLU |
| `conv_bn_relu` | Conv2d → BatchNorm2d → ReLU |
| `conv_bn_silu` | Conv2d → BatchNorm2d → SiLU |

### YAML 設定

```yaml
model:
  layers:
    - type: conv_bn_relu
      out_channels: 32
      kernel_size: 3
      padding: 1
    - type: conv_bn_silu
      out_channels: 64
      kernel_size: 3
      padding: 1
    - type: MaxPool2d
      kernel_size: 2
    - type: Flatten
    - type: Linear
      out_features: 10
```

### 範例腳本

```bash
python examples/feature_expansion/07_flex_presets.py
```

---

## 8. 資料增強（flex 路徑，需要 PyTorch）

在 `configs/flex_broad.yaml` 中加入 `augmentation:` 區段，或透過 API 傳入：

```yaml
augmentation:
  normalize: true
  random_crop: true
  random_crop_padding: 4
  horizontal_flip: true
```

### Python API

```python
from minicnn.flex.data import create_dataloaders

train_loader, val_loader = create_dataloaders(
    dataset_cfg,
    train_cfg,
    augmentation_cfg={'random_crop': True, 'random_crop_padding': 4, 'horizontal_flip': True},
)
```

### 範例腳本

```bash
python examples/feature_expansion/08_augmentation.py
```

---

## 9. AvgPool2d

可在 autograd 與 flex 路徑中使用：

```yaml
model:
  layers:
    - type: AvgPool2d
      kernel_size: 2
```

Python：

```python
from minicnn.nn.layers import AvgPool2d
pool = AvgPool2d(kernel_size=2)
```

---

## 10. AMP 拒絕（autograd 路徑）

若在 autograd 路徑設定 `train.amp=true`，會立即拋出清晰錯誤：

```yaml
train:
  amp: true   # 會觸發 ValueError
```

錯誤訊息：
```
ValueError: train-autograd (NumPy backend) does not support amp=true;
use engine.backend=torch for mixed-precision training
```

---

## 相關文件

- 支援矩陣：[`docs/backend_capabilities.md`](backend_capabilities.md)
- Autograd 核心：[`docs/08_autograd.md`](08_autograd.md)
- 範例腳本索引：[`examples/feature_expansion/README.md`](../examples/feature_expansion/README.md)
- 設定範例：`configs/flex_broad.yaml`、`configs/autograd_enhanced.yaml`、`configs/cuda_legacy_strict.yaml`
