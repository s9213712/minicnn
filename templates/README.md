# MiniCNN Templates

可直接執行的網路架構範本，涵蓋 CIFAR-10 與 MNIST 兩個資料集。
每個 YAML 檔都可不修改任何 Python 直接餵給 `minicnn train-flex` 或 `minicnn train-dual`。

## 建議起手順序

如果你只想最快跑通一條訓練路徑，建議順序是：

1. `minicnn smoke`
2. `minicnn show-model --config configs/flex_cnn.yaml --format text`
3. `minicnn train-flex --config templates/mnist/lenet_like.yaml`
4. `minicnn inspect-checkpoint --path <best-checkpoint>`

最推薦的第一個 template 是：

- `templates/mnist/lenet_like.yaml`

原因：

- MNIST 可自動下載，不需要先準備 CIFAR-10
- 架構小，第一次跑比較快
- 使用 `train-flex`，不會先把新使用者推進 native backend 限制

第二個建議 template 才是：

- `templates/cifar10/vgg_mini.yaml`

這個 template 比較適合在你已經理解基本 config 與 checkpoint 流程後，再開始接觸 CIFAR-10 和雙 backend 邊界。

## 目錄結構

```
templates/
├── README.md               ← 本檔案
├── cifar10/
│   ├── vgg_mini.yaml       # 4-conv VGG-mini（Torch + CUDA 雙後端）
│   ├── vgg_mini_cuda.yaml  # 同上，CUDA backend 專用 conv_layers 格式
│   ├── alexnet_like.yaml   # 更深的 AlexNet-like（僅 Torch）
│   ├── resnet_like.yaml    # 帶殘差連結的 ResNet-like（僅 Torch）
│   └── convnext_like.yaml  # 最小 ConvNeXt-like（僅 Torch，實驗性）
│   └── convnext_explicit.yaml # 顯式 primitive 版 ConvNeXt-like（僅 Torch，實驗性）
│   └── convnext_explicit_smoke.yaml # 顯式 primitive 最小 smoke 訓練版
└── mnist/
    ├── lenet_like.yaml     # LeNet-like 2-conv CNN
    └── mlp.yaml            # MLP baseline
```

---

## 前置步驟

### 安裝

```bash
python -m pip install -e .[torch,dev]
```

### 下載 CIFAR-10

```bash
minicnn prepare-data
```

MNIST 的範本設有 `dataset.download: true`，第一次執行時會自動下載（約 11 MB）。

資料切分規則：

- MNIST：60000 筆 train pool + 10000 筆獨立 test split
- CIFAR-10：50000 筆 train pool + 10000 筆獨立 test split
- `dataset.val_samples` 永遠是從 train pool 切出，不是額外資料
- 因此：
  `MNIST -> num_samples + val_samples <= 60000`
  `CIFAR-10 -> num_samples + val_samples <= 50000`

## 訓練程式位置

範本只需要改 YAML；訓練程式已拆成固定模組：

| 路徑 | 用途 |
|---|---|
| `src/minicnn/training/train_cuda.py` | CUDA legacy orchestration：資料、epoch、validation、checkpoint、LR reduction、early stop、final test。 |
| `src/minicnn/training/cuda_batch.py` | CUDA batch 級 forward/loss/backward/update。 |
| `src/minicnn/training/train_torch_baseline.py` | 對齊 CUDA update 規則的 Torch baseline orchestration 與 batch helper。 |
| `src/minicnn/training/loop.py` | CUDA legacy 與 Torch baseline 共用的 metrics、LR、best/plateau/early-stop、epoch summary helper。 |
| `src/minicnn/training/legacy_data.py` | CUDA legacy 與 Torch baseline 共用的 CIFAR-10 載入與 normalize helper。 |

最佳模型一律輸出到 `artifacts/models/`；template 的 `project.artifacts_root` 只影響 metrics、summary 與其他 run artifacts。

---

## CIFAR-10 範本

### vgg_mini — 4-conv VGG-mini（雙後端）

最基礎的 VGG-style 架構，與 MiniCNN 的手寫 CUDA backend 相容。

**架構**

```
Input 3×32×32
→ Conv(32, 3×3) → LeakyReLU
→ Conv(32, 3×3) → LeakyReLU → MaxPool(2)
→ Conv(64, 3×3) → LeakyReLU
→ Conv(64, 3×3) → LeakyReLU → MaxPool(2)
→ Flatten → Linear(10)
```

**執行（Torch backend）**

```bash
minicnn train-flex --config templates/cifar10/vgg_mini.yaml
```

**執行（CUDA legacy backend，走 `model.layers`）**

```bash
minicnn train-dual --config templates/cifar10/vgg_mini.yaml engine.backend=cuda_legacy
```

**修改架構**：直接編輯 `model.layers` 區塊，增減 Conv2d / MaxPool2d 項目即可。`in_channels` 會自動推算。

---

### vgg_mini_cuda — CUDA backend 專用格式

使用 `model.conv_layers` 清單，由 `CudaNetGeometry` 自動推算所有形狀與 buffer 大小。

**架構**（同上）

```
Input 3×32×32
→ Conv(32) → Conv(32)+Pool(2)
→ Conv(64) → Conv(64)+Pool(2)
→ FC(1600→10)
```

**執行**

```bash
minicnn train-cuda --config templates/cifar10/vgg_mini_cuda.yaml
```

**修改架構**：編輯 `model.conv_layers`，每個 entry 只需填 `out_c` 和 `pool`。

```yaml
conv_layers:
  - {out_c: 32, pool: false}  # 加寬：改 out_c
  - {out_c: 32, pool: true}
  - {out_c: 64, pool: false}
  - {out_c: 64, pool: true}
  - {out_c: 128, pool: false} # 加層：新增 entry
  - {out_c: 128, pool: true}
```

修改後驗證：

```bash
minicnn validate-dual-config --config templates/cifar10/vgg_mini_cuda.yaml
```

---

### alexnet_like — AlexNet-like（Torch only）

較深的三段式卷積架構，附 Dropout。

**架構**

```
Input 3×32×32
→ Conv(64, 3×3, pad=1) → ReLU → MaxPool(2)   # 16×16
→ Conv(128, 3×3, pad=1) → ReLU → MaxPool(2)  # 8×8
→ Conv(256, 3×3, pad=1) → ReLU
→ Conv(256, 3×3, pad=1) → ReLU → MaxPool(2)  # 4×4
→ Flatten → Linear(512) → ReLU → Dropout(0.5) → Linear(10)
```

**執行**

```bash
minicnn train-flex --config templates/cifar10/alexnet_like.yaml
```

> 使用 `padding: 1` 保留空間解析度，所有 MaxPool 後的特徵圖仍為整數尺寸。
> CUDA legacy backend 不支援此架構（帶 padding 的 conv、三段式 pool）。

---

### resnet_like — ResNet-like（Torch only）

帶跳躍連結的殘差架構，適合理解 residual learning。

**架構**

```
Input 3×32×32
→ Stem: Conv(64, 3×3, pad=1)/BN/ReLU
→ Stage 1 (32×32): ResBlock(64) × 2
→ Stage 2 (16×16): ResBlock(128, stride=2) + ResBlock(128)
→ GlobalAvgPool → Flatten → Linear(10)
```

**執行**

```bash
minicnn train-flex --config templates/cifar10/resnet_like.yaml
```

> `ResidualBlock` 和 `GlobalAvgPool2d` 是 MiniCNN torch flex 專用元件，
> 位於 `src/minicnn/flex/builder.py`。CUDA legacy backend 不支援此架構。

---

### convnext_like — ConvNeXt-like（Torch only, experimental）

最小、刻意收斂的 ConvNeXt-like 路徑，用來驗證 `torch/flex` 前端是否能承載
depthwise conv、LayerNorm、GELU、MLP 與 residual add 的組合。

**架構**

```
Input 3×32×32
→ Stem: Conv(64, 3×3, pad=1)
→ ConvNeXtBlock(64) × 2
→ Downsample: Conv(128, 2×2, stride=2)   # 16×16
→ ConvNeXtBlock(128) × 2
→ GlobalAvgPool → Flatten → Linear(10)
```

**執行**

```bash
minicnn train-flex --config templates/cifar10/convnext_like.yaml
```

> `ConvNeXtBlock` 是 MiniCNN 的 torch/flex 專用元件，屬於實驗性 frontend
> 擴充，不代表 `cuda_native` 或 `cuda_legacy` 已支援 ConvNeXt。
> 目前 block 內部會明確使用 depthwise conv、channel-first `LayerNorm2d`、
> pointwise conv、`GELU` 與 residual add，不再依賴隱含的 NHWC 轉換語意。

### convnext_explicit — ConvNeXt-like explicit primitives（Torch only, experimental）

與 `convnext_like` 不同，這份 template 不使用封裝好的 `ConvNeXtBlock`，
而是把 block 直接展開成顯式 primitive：

```text
DepthwiseConv2d -> LayerNorm2d -> PointwiseConv2d(expand) -> GELU -> PointwiseConv2d(shrink)
```

**執行**

```bash
minicnn train-flex --config templates/cifar10/convnext_explicit.yaml
```

適合用來：

- 驗證 registry 裡的 ConvNeXt primitives 是否可直接組裝
- 做更細的 block 級實驗，而不先改 `ConvNeXtBlock` 類別本身

> 這仍然是 `torch/flex` 專用的實驗性路徑，不代表其他 backend 已具備
> depthwise conv / LayerNorm2d / residual add 的對應能力。

### convnext_explicit_smoke — explicit primitives smoke config

如果你只想最快驗證顯式 ConvNeXt primitives 能否完成一次最小訓練啟動，
使用這份較小的 smoke config：

```bash
minicnn train-flex --config templates/cifar10/convnext_explicit_smoke.yaml
```

這份設定固定為：

- `dataset.num_samples: 64`
- `dataset.val_samples: 16`
- `train.epochs: 1`
- `train.batch_size: 16`
- `train.device: cpu`

---

## MNIST 範本

對第一次接觸 MiniCNN 的使用者，這一節通常比 CIFAR-10 範本更適合作為第一站。

第一次執行時 `dataset.download: true` 會自動抓取資料（約 11 MB）。
之後可將 `download` 改為 `false` 加快啟動。

資料儲存路徑：`data/mnist/`（可透過 `dataset.data_root` 修改）

MNIST split 規則：

- 60000 筆 training images 是 train/val 共用的 pool
- `dataset.val_samples` 是從這 60000 裡切出來，不是額外資料
- 因此 `dataset.num_samples + dataset.val_samples` 必須小於等於 `60000`
- 目前內建 MNIST templates 採用 `55000 + 5000`

---

### lenet_like — LeNet-like CNN

輕量的兩層卷積網路，對 MNIST 通常可達 99% 以上準確率。

**架構**

```
Input 1×28×28
→ Conv(8, 3×3) → ReLU → MaxPool(2)    # 8×13×13
→ Conv(16, 3×3) → ReLU → MaxPool(2)   # 16×5×5
→ Flatten(400) → Linear(10)
```

**執行**

```bash
minicnn train-flex --config templates/mnist/lenet_like.yaml
```

**修改建議**：
- 增加 filter 數量（`out_channels: 16/32`）可小幅提升準確率。
- 在 Flatten 後加一層 `Linear(out_features: 64)` + `ReLU` 可加深分類頭。

---

### mlp — MLP baseline

純全連接網路，不含卷積，適合作為效能下限參考。

**架構**

```
Input 1×28×28
→ Flatten(784)
→ Linear(256) → GELU
→ Dropout(0.1)
→ Linear(128) → ReLU
→ Linear(10)
```

**執行**

```bash
minicnn train-flex --config templates/mnist/mlp.yaml
```

---

## 常見調整

| 目標 | 修改位置 |
|---|---|
| 更換 optimizer | `optimizer.type: Adam / SGD / AdamW` |
| 調整學習率 | `optimizer.lr` |
| 開啟資料增強 | `dataset.random_crop_padding: 4`、`dataset.horizontal_flip: true` |
| 減少訓練量（快速驗證） | `train.epochs: 1`、`dataset.num_samples: 1024`、`dataset.val_samples: 256` |
| CLI 臨時覆蓋參數 | `minicnn train-flex --config ... train.epochs=3 optimizer.lr=0.001` |
| 固定初始權重 | `train.init_seed: 42`，torch/flex、CUDA legacy、autograd 都會使用各自的 init seed 路徑 |
| CLI 覆蓋 layer list | `model.layers.1.out_features=7`，數字段會當成 list index |

布林欄位請使用 YAML boolean 或可解析字串，例如 `true`、`false`、`1`、`0`。MiniCNN 會用 strict parser 處理 `dataset.download`、`dataset.horizontal_flip`、`train.amp`、`optimizer.exclude_bias_norm_weight_decay` 與 CUDA `conv_layers[].pool`，避免 `"false"` 被 Python `bool()` 誤判為 true。

## 以 template 為起點建立新架構

```bash
cp templates/cifar10/vgg_mini.yaml configs/my_arch.yaml
# 編輯 configs/my_arch.yaml 的 model.layers
minicnn train-flex --config configs/my_arch.yaml
```
