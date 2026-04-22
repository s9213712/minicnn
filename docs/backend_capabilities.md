# Backend Capabilities

Read MiniCNN capability by backend, not as one global checklist.

The frontend surface is broader than the narrowest backend. That is expected.

---

## What cuda_native Adds Over cuda_legacy (✗ → ✓)

These features were **not** supported in `cuda_legacy` and are now supported in `cuda_native`:

| Feature | cuda_legacy | cuda_native |
|---|:---:|:---:|
| **Datasets** | | |
| MNIST dataset | ✗ | ✓ |
| Random toy data | ✗ | ✓ |
| **Layers** | | |
| AvgPool2d | ✗ | ✓ numpy ref |
| **Losses** | | |
| MSELoss | Experimental | ✓ numpy |
| **Developer tooling** | | |
| Graph dump (`dump_graph`) | ✗ | ✓ |
| Plan dump (`dump_plan`) | ✗ | ✓ |
| Execution trace (`TracingForwardExecutor`) | ✗ | ✓ |
| Layout validation (`validate_graph_layouts`) | ✗ | ✓ |
| Memory footprint estimate (`memory_footprint`) | ✗ | ✓ |
| Buffer pool pre-allocation (`BufferPool`) | ✗ | ✓ |

Note: `cuda_native` uses numpy reference kernels, not real CUDA. It is experimental and not production-ready.

---

## Full Capability Matrix

| Capability | Torch/flex | CPU/NumPy autograd | CUDA legacy | cuda_native (experimental) |
|---|:---:|:---:|:---:|:---:|
| **Datasets** | | | | |
| CIFAR-10 | ✓ | ✓ slow | ✓ | ✓ |
| MNIST | ✓ | ✓ slow | ✗ | **✓** |
| Random toy data | ✓ | ✓ | ✗ | **✓** |
| **Layers** | | | | |
| Conv2d | ✓ | ✓ | ✓ fixed 3×3 s1 p0 | ✓ numpy ref |
| Linear | ✓ | ✓ | ✓ | ✓ numpy ref |
| MaxPool2d | ✓ | ✓ | ✓ fixed 2×2 | ✓ numpy ref |
| AvgPool2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| BatchNorm2d | ✓ | ✓ | ✗ | ✓ forward/backward prototype |
| LayerNorm | ✓ | ✗ | ✗ | ✗ rejected |
| GroupNorm | ✓ | ✗ | ✗ | ✗ rejected |
| ResidualBlock | ✓ | ✓ | ✗ | ✗ rejected |
| Dropout | ✓ | ✓ | ✗ | ✗ |
| **Activations** | | | | |
| ReLU | ✓ | ✓ | ✓ | ✓ numpy ref |
| LeakyReLU | ✓ | ✓ | ✓ | ✓ numpy ref |
| SiLU | ✓ | ✓ | ✗ | ✓ numpy ref |
| Sigmoid | ✓ | ✓ | ✗ | ✓ numpy ref |
| Tanh | ✓ | ✓ | ✗ | ✓ numpy ref |
| GELU | ✓ | ✗ | ✗ | ✗ |
| **Losses** | | | | |
| CrossEntropyLoss | ✓ | ✓ | ✓ | ✓ numpy |
| MSELoss | ✓ | ✓ | Experimental | **✓** numpy |
| BCEWithLogitsLoss | ✓ binary | ✓ binary | ✗ | ✗ |
| label_smoothing | ✓ | ✓ | ✗ | ✗ |
| **Optimizers** | | | | |
| SGD | ✓ | ✓ | ✓ | ✓ numpy prototype |
| Momentum SGD | ✓ | ✓ | ✓ | ✓ numpy prototype |
| Adam | ✓ | ✓ | Experimental | ✗ |
| AdamW | ✓ | ✓ | ✗ | ✗ |
| RMSprop | ✓ | ✓ | ✗ | ✗ |
| **Schedulers** | | | | |
| None / disabled | ✓ | ✓ | ✓ | ✓ |
| StepLR | ✓ | ✓ | ✗ | ✓ |
| CosineAnnealingLR | ✓ | ✓ | ✗ | ✓ |
| ReduceLROnPlateau | ✓ | ✓ | partial | ✓ |
| **Regularization** | | | | |
| weight_decay | ✓ | ✓ | ✓ | ✓ in SGD |
| gradient clipping | ✓ | ✓ | ✓ | ✓ global norm |
| AMP | ✓ CUDA only | ✗ | ✗ | ✗ |
| **Frontend** | | | | |
| `model.layers[]` YAML | ✓ | ✓ | ✓ fixed pattern | ✓ sequential only |
| dotted-path components | ✓ | ✗ | ✗ | ✗ |
| block presets | ✓ | ✗ | ✗ | ✗ |
| **Training** | | | | |
| Forward pass | ✓ | ✓ | ✓ | ✓ |
| Backward / gradients | ✓ | ✓ | ✓ | Prototype |
| Full training loop | ✓ | ✓ | ✓ | Prototype |
| Production-ready | ✓ | ✓ | ✓ | ✗ experimental |
| **Developer tooling** | | | | |
| Graph dump | ✗ | ✗ | ✗ | **✓** `dump_graph()` |
| Plan dump | ✗ | ✗ | ✗ | **✓** `dump_plan()` |
| Execution trace | ✗ | ✗ | ✗ | **✓** `TracingForwardExecutor` |
| Layout validation | ✗ | ✗ | ✗ | **✓** `validate_graph_layouts()` |
| Memory footprint | ✗ | ✗ | ✗ | **✓** `memory_footprint()` |
| Buffer pool | ✗ | ✗ | ✗ | **✓** `BufferPool` |

**Bold** = changed from ✗ in cuda_legacy (or new capability not in any other backend).

---

## Torch/Flex

Broadest stable path. Use it for new model ideas, custom Python components, fast iteration, and most experiments that do not specifically need a handcrafted CUDA path.

Accepts torch module names beyond the built-in registry through the flex builder fallback to `torch.nn`.

## CPU/NumPy Autograd

Intentionally educational and CPU-only.

Use it for framework learning, deterministic tests, and small experiments without torch dependency.

Limitations: Conv2d is much slower than torch; no AMP; no LayerNorm / GroupNorm; no dotted-path custom components.

## CUDA Legacy

Real training backend, intentionally narrow.

Stable support boundary:

- dataset: `cifar10`, input shape `[3, 32, 32]`
- layer pattern: `Conv2d → activation → Conv2d → activation → MaxPool2d → Conv2d → activation → Conv2d → activation → MaxPool2d → Flatten → Linear`
- activations: `ReLU` or `LeakyReLU`
- optimizer: `SGD` or `Adam`
- loss: `CrossEntropyLoss`, `MSELoss`

Use `minicnn validate-dual-config` before running.
Validation failures now return short CLI messages or JSON payloads instead of raw tracebacks.

## cuda_native (Experimental)

Opt-in via `engine.backend=cuda_native` or `train-native`. Not the default. Not a replacement for `cuda_legacy`.

Supported ops: `BatchNorm2d` (forward/backward prototype), `Conv2d`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `SiLU`, `Flatten`, `Linear`, `MaxPool2d`, `AvgPool2d`.

Validated train-native support boundary:

- dataset: `random`, `cifar10`, `mnist`
- loss: `CrossEntropyLoss`, `MSELoss`
- optimizer: `SGD` with optional momentum and global gradient clipping
- scheduler: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, or disabled
- `train.amp=false`, `train.grad_accum_steps=1`

Unsupported (rejected at validation): `GroupNorm`, `LayerNorm`, `ResidualBlock`.

Note: backward and training prototypes exist, and `BatchNorm2d` now has a
prototype backward path too. The overall backend remains experimental and not
production-ready.

Developer tooling (unique to cuda_native):

```python
from minicnn.cuda_native.debug import dump_graph, dump_plan, TracingForwardExecutor
from minicnn.cuda_native.layouts import validate_graph_layouts
from minicnn.cuda_native.memory import memory_footprint, BufferPool

# Inspect a graph
print(dump_graph(graph))

# Inspect a plan
plan = make_naive_plan(graph)
print(dump_plan(plan))

# Trace execution with per-node timing
ctx, trace = TracingForwardExecutor().run(graph, feeds, params)
trace.print()

# Check memory usage
print(memory_footprint(graph))
```

CLI:

```bash
minicnn cuda-native-capabilities
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

`validate-*`, `doctor`, `healthcheck`, and `smoke` all emit JSON-friendly output.
`healthcheck`, `doctor`, and `smoke` also accept `--format text`.

See [docs/cuda_native.md](cuda_native.md) for the full guide.
See [docs/cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md) for future extension RFCs.

## Reading Validation Errors

If a config runs on `torch` but fails on `cuda_legacy`, that is an expected backend boundary, not a parser bug.

Debugging order:
1. Check this matrix.
2. Run `minicnn validate-dual-config` or `minicnn validate-cuda-native-config`.
3. Decide whether you need a torch-only change, a native backend change, or a separate experimental branch.

---

# Backend 能力對照表（中文）

依 backend 閱讀 MiniCNN 的能力範圍，不要把它當成一份全局清單。

前端支援的功能本來就比最窄的 backend 更廣，這是預期中的設計。

---

## cuda_native 比 cuda_legacy 多了什麼（✗ → ✓）

以下功能在 `cuda_legacy` 不支援或受限，在 `cuda_native` 已支援：

| 功能 | cuda_legacy | cuda_native |
|---|:---:|:---:|
| **資料集** | | |
| MNIST 資料集 | ✗ | ✓ |
| 隨機假資料 | ✗ | ✓ |
| **層** | | |
| AvgPool2d | ✗ | ✓ numpy ref |
| **損失函數** | | |
| MSELoss | 實驗中 | ✓ numpy |
| **開發者工具** | | |
| Graph dump（`dump_graph`） | ✗ | ✓ |
| Plan dump（`dump_plan`） | ✗ | ✓ |
| Execution trace（`TracingForwardExecutor`） | ✗ | ✓ |
| Layout 驗證（`validate_graph_layouts`） | ✗ | ✓ |
| 記憶體估算（`memory_footprint`） | ✗ | ✓ |
| Buffer pool 預分配（`BufferPool`） | ✗ | ✓ |

注意：`cuda_native` 使用 numpy 參考 kernel，不是真正的 CUDA，屬於實驗性 backend。

---

## 完整能力對照表

| 功能 | Torch/flex | CPU/NumPy autograd | CUDA legacy | cuda_native（實驗） |
|---|:---:|:---:|:---:|:---:|
| **資料集** | | | | |
| CIFAR-10 | ✓ | ✓ 較慢 | ✓ | ✓ |
| MNIST | ✓ | ✓ 較慢 | ✗ | **✓** |
| 隨機假資料 | ✓ | ✓ | ✗ | **✓** |
| **層 (Layers)** | | | | |
| Conv2d | ✓ | ✓ | ✓ 固定 3×3 s1 p0 | ✓ numpy ref |
| Linear | ✓ | ✓ | ✓ | ✓ numpy ref |
| MaxPool2d | ✓ | ✓ | ✓ 固定 2×2 | ✓ numpy ref |
| AvgPool2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| BatchNorm2d | ✓ | ✓ | ✗ | ✓ forward/backward prototype |
| LayerNorm | ✓ | ✗ | ✗ | ✗ 拒絕 |
| GroupNorm | ✓ | ✗ | ✗ | ✗ 拒絕 |
| ResidualBlock | ✓ | ✓ | ✗ | ✗ 拒絕 |
| Dropout | ✓ | ✓ | ✗ | ✗ |
| **激活函數** | | | | |
| ReLU | ✓ | ✓ | ✓ | ✓ numpy ref |
| LeakyReLU | ✓ | ✓ | ✓ | ✓ numpy ref |
| SiLU | ✓ | ✓ | ✗ | ✓ numpy ref |
| Sigmoid | ✓ | ✓ | ✗ | ✓ numpy ref |
| Tanh | ✓ | ✓ | ✗ | ✓ numpy ref |
| GELU | ✓ | ✗ | ✗ | ✗ |
| **損失函數** | | | | |
| CrossEntropyLoss | ✓ | ✓ | ✓ | ✓ numpy |
| MSELoss | ✓ | ✓ | 實驗中 | **✓** numpy |
| BCEWithLogitsLoss | ✓ binary | ✓ binary | ✗ | ✗ |
| label_smoothing | ✓ | ✓ | ✗ | ✗ |
| **優化器** | | | | |
| SGD | ✓ | ✓ | ✓ | ✓ numpy prototype |
| Momentum SGD | ✓ | ✓ | ✓ | ✓ numpy prototype |
| Adam | ✓ | ✓ | 實驗中 | ✗ |
| AdamW | ✓ | ✓ | ✗ | ✗ |
| RMSprop | ✓ | ✓ | ✗ | ✗ |
| **Scheduler** | | | | |
| 無 / 停用 | ✓ | ✓ | ✓ | ✓ |
| StepLR | ✓ | ✓ | ✗ | ✓ |
| CosineAnnealingLR | ✓ | ✓ | ✗ | ✓ |
| ReduceLROnPlateau | ✓ | ✓ | 部分支援 | ✓ |
| **正則化** | | | | |
| weight_decay | ✓ | ✓ | ✓ | ✓ SGD 內建 |
| gradient clipping | ✓ | ✓ | ✓ | ✓ global norm |
| AMP | ✓ CUDA 限定 | ✗ | ✗ | ✗ |
| **前端便利功能** | | | | |
| `model.layers[]` YAML | ✓ | ✓ | ✓ 固定 pattern | ✓ sequential only |
| dotted-path 自訂元件 | ✓ | ✗ | ✗ | ✗ |
| block presets | ✓ | ✗ | ✗ | ✗ |
| **訓練** | | | | |
| Forward pass | ✓ | ✓ | ✓ | ✓ |
| Backward / 梯度 | ✓ | ✓ | ✓ | Prototype |
| 完整訓練迴圈 | ✓ | ✓ | ✓ | Prototype |
| 正式環境可用 | ✓ | ✓ | ✓ | ✗ 實驗中 |
| **開發者工具（cuda_native 獨有）** | | | | |
| Graph dump | ✗ | ✗ | ✗ | **✓** `dump_graph()` |
| Plan dump | ✗ | ✗ | ✗ | **✓** `dump_plan()` |
| Execution trace | ✗ | ✗ | ✗ | **✓** `TracingForwardExecutor` |
| Layout 驗證 | ✗ | ✗ | ✗ | **✓** `validate_graph_layouts()` |
| 記憶體估算 | ✗ | ✗ | ✗ | **✓** `memory_footprint()` |
| Buffer pool | ✗ | ✗ | ✗ | **✓** `BufferPool` |

**粗體** = 從 cuda_legacy 的 ✗ 變為 ✓，或其他 backend 都沒有的新能力。

---

## Torch/Flex

最廣泛的穩定路徑。適合新模型想法、自訂 Python 元件、快速迭代，以及大多數不需要手寫 CUDA 的實驗。

透過 flex builder fallback 到 `torch.nn`，可使用 built-in registry 以外的 torch module 名稱。

## CPU/NumPy Autograd

刻意設計為教學用途，CPU-only。

適合框架學習、可重現測試，以及不依賴 torch 的小型實驗。

限制：Conv2d 比 torch 慢很多；不支援 AMP、LayerNorm、GroupNorm、dotted-path 自訂元件。

## CUDA Legacy

真正的訓練 backend，刻意維持狹窄。

穩定支援範圍：
- 資料集：`cifar10`，input shape `[3, 32, 32]`
- 層 pattern：`Conv2d → activation → Conv2d → activation → MaxPool2d → Conv2d → activation → Conv2d → activation → MaxPool2d → Flatten → Linear`
- 激活：`ReLU` 或 `LeakyReLU`
- 優化器：`SGD` 或 `Adam`
- 損失：`CrossEntropyLoss`、`MSELoss`

訓練前請先執行 `minicnn validate-dual-config`。
驗證失敗現在會回傳簡短 CLI 訊息或 JSON payload，而不是直接丟出 raw traceback。

## cuda_native（實驗）

透過 `engine.backend=cuda_native` 或 `train-native` 明確啟用。不是預設 backend，不取代 `cuda_legacy`。

目前通過驗證的 train-native 支援範圍：

- dataset：`random`、`cifar10`、`mnist`
- loss：`CrossEntropyLoss`、`MSELoss`
- optimizer：支援 `SGD`，可選 momentum 與 global gradient clipping
- scheduler：支援 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`，也可停用
- `train.amp=false`、`train.grad_accum_steps=1`

支援 op：`BatchNorm2d`（forward/backward prototype）、`Conv2d`、`ReLU`、`LeakyReLU`、`Sigmoid`、`Tanh`、`SiLU`、`Flatten`、`Linear`、`MaxPool2d`、`AvgPool2d`。

驗證時拒絕的 op：`GroupNorm`、`LayerNorm`、`ResidualBlock`。

注意：雖然已有 backward 與 training prototype，且 `BatchNorm2d` 也已有
prototype 級的 backward，但整體 backend 仍屬實驗性，不是正式訓練後端。

開發者工具（cuda_native 獨有）：

```python
from minicnn.cuda_native.debug import dump_graph, dump_plan, TracingForwardExecutor
from minicnn.cuda_native.layouts import validate_graph_layouts
from minicnn.cuda_native.memory import memory_footprint, BufferPool

# 查看 graph 結構
print(dump_graph(graph))

# 查看 buffer 分配計劃
plan = make_naive_plan(graph)
print(dump_plan(plan))

# 帶 per-node 時序的 trace 執行
ctx, trace = TracingForwardExecutor().run(graph, feeds, params)
trace.print()

# 估算記憶體用量
print(memory_footprint(graph))
```

CLI：

```bash
minicnn cuda-native-capabilities
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

`validate-*`、`doctor`、`healthcheck`、`smoke` 現在都會輸出 JSON-friendly 結果。
`healthcheck`、`doctor`、`smoke` 也支援 `--format text`。

完整說明見 [docs/cuda_native.md](cuda_native.md)。
Phase 5 擴充 RFC 見 [docs/cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)。

## 閱讀 Validation 錯誤

如果 config 在 `torch` 能跑但在 `cuda_legacy` 失敗，那通常是預期中的 backend 邊界，不是 parser bug。

除錯順序：
1. 查看這份對照表。
2. 執行 `minicnn validate-dual-config` 或 `minicnn validate-cuda-native-config`。
3. 再決定是否需要 torch-only 修改、native backend 修改，或是放進獨立的實驗 branch。
