# Backend Capabilities

Read MiniCNN capability by backend, not as one global checklist.

The frontend surface is broader than the narrowest backend. That is expected.

## Backend Roles

| Backend | Role | Practical meaning |
|---|---|---|
| `torch/flex` | reference implementation | first destination for new frontend features, broadest stable surface |
| `cuda_native` | primary native backend | the native path that should grow next; beta-grade reference mode plus partial real-CUDA `gpu_native` execution |
| `autograd` | correctness oracle | CPU-side reference path for deterministic checks and framework learning |
| `cuda_legacy` | maintenance-only historical backend | narrow stable path, kept for compatibility and maintenance, not the default feature-expansion target |

## Feature Rollout Policy

Default order for new capability work:

1. `torch/flex`
2. `autograd` when a correctness reference is useful
3. `cuda_native`
4. `cuda_legacy` only for maintenance or compatibility needs

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

Note: default `cuda_native` execution uses NumPy reference kernels. Opt-in
`execution_mode=gpu_native` uses real device-pointer CUDA kernels for the
supported forward/training subset. The backend is beta-grade, not experimental,
but it is still not production-ready.

---

## Full Capability Matrix

| Capability | Torch/flex | CPU/NumPy autograd | CUDA legacy | cuda_native (beta) |
|---|:---:|:---:|:---:|:---:|
| **Datasets** | | | | |
| CIFAR-10 | ✓ | ✓ slow | ✓ | ✓ |
| MNIST | ✓ | ✓ slow | ✗ | **✓** |
| Random toy data | ✓ | ✓ | ✗ | **✓** |
| **Layers** | | | | |
| Conv2d | ✓ | ✓ | ✓ fixed 3×3 s1 p0 | ✓ numpy ref + partial `gpu_native` |
| Linear | ✓ | ✓ | ✓ | ✓ numpy ref + partial `gpu_native` |
| MaxPool2d | ✓ | ✓ | ✓ fixed 2×2 | ✓ numpy ref + partial `gpu_native` |
| AvgPool2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| BatchNorm2d | ✓ | ✓ | ✗ | ✓ forward/backward prototype |
| GroupNorm | ✓ | ✗ | ✗ | **✓** prototype |
| LayerNorm2d | ✓ | ✗ | ✗ | **✓** prototype |
| LayerNorm | ✓ | ✗ | ✗ | **✓** prototype |
| DepthwiseConv2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| PointwiseConv2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| GlobalAvgPool2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| AdaptiveAvgPool2d | ✓ | ✓ | ✗ | **✓** `(1,1)` only |
| Identity | ✓ | ✓ | ✗ | **✓** numpy ref |
| ResidualBlock | ✓ | ✓ | ✗ | **✓** composite prototype |
| ConvNeXtBlock | ✓ experimental | ✗ | ✗ | **✓** composite prototype |
| Dropout | ✓ | ✓ | ✗ | **✓** prototype |
| **Activations** | | | | |
| ReLU | ✓ | ✓ | ✓ | ✓ numpy ref + partial `gpu_native` |
| LeakyReLU | ✓ | ✓ | ✓ | ✓ numpy ref + partial `gpu_native` |
| SiLU | ✓ | ✓ | ✗ | ✓ numpy ref + partial `gpu_native` |
| Sigmoid | ✓ | ✓ | ✗ | ✓ numpy ref + partial `gpu_native` |
| Tanh | ✓ | ✓ | ✗ | ✓ numpy ref + partial `gpu_native` |
| GELU | ✓ | ✗ | ✗ | **✓** numpy ref + partial `gpu_native` |
| **Losses** | | | | |
| CrossEntropyLoss | ✓ | ✓ | ✓ | ✓ numpy + partial `gpu_native` |
| MSELoss | ✓ | ✓ | Experimental | **✓** numpy |
| BCEWithLogitsLoss | ✓ binary | ✓ binary | ✗ | **✓** binary |
| label_smoothing | ✓ | ✓ | ✗ | **✓** cross-entropy prototype |
| **Optimizers** | | | | |
| SGD | ✓ | ✓ | ✓ | ✓ numpy + partial `gpu_native` |
| Momentum SGD | ✓ | ✓ | ✓ | ✓ numpy + gpu_native |
| Adam | ✓ | ✓ | Experimental | **✓** numpy + gpu_native Linear |
| AdamW | ✓ | ✓ | ✗ | **✓** numpy + gpu_native Linear |
| RMSprop | ✓ | ✓ | ✗ | **✓** numpy + gpu_native Linear |
| **Schedulers** | | | | |
| None / disabled | ✓ | ✓ | ✓ | ✓ |
| StepLR | ✓ | ✓ | ✗ | ✓ |
| CosineAnnealingLR | ✓ | ✓ | ✗ | ✓ |
| ReduceLROnPlateau | ✓ | ✓ | partial | ✓ |
| **Regularization** | | | | |
| weight_decay | ✓ | ✓ | ✓ | ✓ |
| gradient clipping | ✓ | ✓ | ✓ | ✓ global norm |
| AMP | ✓ CUDA only | ✗ | ✗ | ✓ beta |
| **Frontend** | | | | |
| `model.layers[]` YAML | ✓ | ✓ | ✓ fixed pattern | ✓ ordered DAG with named tensor wiring |
| dotted-path components | ✓ | ✗ | ✗ | ✗ |
| block presets | ✓ | ✗ | ✗ | ✗ |
| **Training** | | | | |
| Forward pass | ✓ | ✓ | ✓ | ✓ reference + partial `gpu_native` |
| Backward / gradients | ✓ | ✓ | ✓ | ✓ beta within support boundary |
| Full training loop | ✓ | ✓ | ✓ | ✓ beta within support boundary |
| Production-ready | ✓ | ✓ | ✓ | ✗ beta, not production-ready |
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

Broadest stable path and the repo's reference implementation. Use it for new model ideas, custom Python components, fast iteration, and most experiments that do not specifically need a native backend constraint.

Accepts torch module names beyond the built-in registry through the flex builder fallback to `torch.nn`.

## CPU/NumPy Autograd

Intentionally educational, CPU-only, and used as the internal correctness oracle.

Use it for framework learning, deterministic tests, parity checks, and small experiments without torch dependency.

Limitations: Conv2d is much slower than torch; no full-graph AMP; `LayerNorm` and `GroupNorm` train-native coverage remain narrow/helper-scoped; no dotted-path custom components.

## CUDA Legacy

Historical native backend, intentionally narrow and maintenance-only.

Stable support boundary:

- dataset: `cifar10`, input shape `[3, 32, 32]`
- layer pattern: `Conv2d → activation → Conv2d → activation → MaxPool2d → Conv2d → activation → Conv2d → activation → MaxPool2d → Flatten → Linear`
- activations: `ReLU` or `LeakyReLU`
- optimizer: `SGD` or `Adam`
- loss: `CrossEntropyLoss`, `MSELoss`

Use `minicnn validate-dual-config` before running.
Validation failures now return short CLI messages or JSON payloads instead of raw tracebacks.

## cuda_native (Primary Native Direction, Beta)

Opt-in via `engine.backend=cuda_native` or `train-native`. This is the main native direction for future work. It is now beta-grade, but still not production-ready. The default execution mode is GPU-first auto execution when a CUDA-native lowering is eligible, with `reference_numpy` retained as explicit fallback and parity baseline; opt-in `execution_mode=gpu_native` runs a growing subset through real CUDA device-pointer kernels and native training helpers.

Real-data strict GPU training evidence exists for the current repeated-Conv
subset. Use `configs/cifar10_cuda_native_gpu_stronger.yaml` for the full
CIFAR-10 command path; see
[cuda_native_gpu_cifar10_runbook.md](cuda_native_gpu_cifar10_runbook.md).

Supported reference-mode ops: `BatchNorm2d` (forward/backward prototype), `Concat`, `Conv2d`, `DepthwiseConv2d`, `PointwiseConv2d`, `GroupNorm`, `LayerNorm`, `LayerNorm2d`, `ResidualBlock`, `ConvNeXtBlock`, `Dropout`, `DropPath`, `Add`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GELU`, `Identity`, `Flatten`, `Linear`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` (`output_size=(1,1)` only), `GlobalAvgPool2d`.

Supported `gpu_native` forward kernel surface currently includes: `Flatten`,
`Identity`, `Dropout(p=0)`, `DropPath(p=0)`, `Linear`, `ReLU`, `LeakyReLU`,
`Sigmoid`, `Tanh`, `SiLU`, `GELU`, `Add`, `Concat`, `Conv2d`,
`PointwiseConv2d`, `DepthwiseConv2d`, `MaxPool2d`, `AvgPool2d`,
`GlobalAvgPool2d`, `AdaptiveAvgPool2d(output_size=(1,1))`, `BatchNorm2d`
eval forward, `LayerNorm2d`, and `GroupNorm`.

Graph semantics:

- ordered DAG execution, not just a strict linear chain
- explicit `inputs: [...]` and `output: ...` tensor wiring in `model.layers[]`
- generic `Add` / `Concat` merge support for residual-style and channel-join paths

Validated train-native support boundary:

- dataset: `random`, `cifar10`, `mnist`
- loss: `CrossEntropyLoss` (with optional `label_smoothing`), `BCEWithLogitsLoss` (binary output only), `MSELoss`
- optimizer: `SGD`, `Adam`, `AdamW`, or `RMSprop`, with global gradient clipping
- scheduler: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, or disabled
- `train.amp=true|false` (beta mixed-precision path with loss scaling / overflow backoff)
- `train.grad_accum_steps >= 1`
- `summary.json` exposes `amp_runtime`, `optimizer_runtime`, `planner`, and `performance_report`
- `metrics.jsonl` exposes per-epoch AMP, optimizer, and planner telemetry

Supported `gpu_native` training subsets use native `optimizer.grad_clip_global`
through `grad_l2_sumsq` plus `scale_inplace`.
Supported SGD `gpu_native` helper subsets use native `optimizer.weight_decay`
through `sgd_update_fused`.

Still rejected at validation or train-native gating: unsupported optimizers outside `SGD` / `Adam` / `AdamW` / `RMSprop`.

Note: backward and training now meet the current beta graduation gate, and
`BatchNorm2d` now has a prototype backward path too. The overall backend is
beta, not production-ready. New native capability work should usually land here,
not in `cuda_legacy`.

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

## cuda_native Support Tiers

Use these tiers as the current public positioning, not as a promise that every
listed item is production-ready.

### Stable

- ordered DAG graph execution
- named tensor wiring
- `Add`
- `Concat`
- `Conv2d`
- `Linear`
- `Flatten`
- `ReLU`
- `CrossEntropyLoss`
- `SGD`
- `AdamW`
- `grad_accum_steps`
- artifact reporting contracts

### Beta

- `GroupNorm`
- `LayerNorm`
- `LayerNorm2d`
- `BCEWithLogitsLoss`
- `RMSprop`
- AMP
- planner reuse heuristics

### Experimental

- no public op/optimizer/loss surface currently remains in `experimental`
- full-graph GPU execution, generalized GPU backward, and composite-block GPU
  training composition still live outside the current beta contract

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

## Backend 角色

| Backend | 角色 | 實際意義 |
|---|---|---|
| `torch/flex` | reference implementation | 新 frontend 功能的第一站，也是最廣、最穩的路徑 |
| `cuda_native` | 主要 native backend | 後續 native 能力應優先長在這裡，但目前仍屬實驗性 |
| `autograd` | correctness oracle | CPU 側的參考路徑，適合 deterministic 檢查與框架學習 |
| `cuda_legacy` | 歷史維護 backend | 在窄邊界內穩定，但主要用途是相容與維護，不是預設擴充目標 |

## 新功能 rollout 順序

新增能力時，預設順序是：

1. `torch/flex`
2. 若 correctness 參考有價值，再補 `autograd`
3. 再推進 `cuda_native`
4. `cuda_legacy` 只在維護或相容需求下補修

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
| LayerNorm2d | ✓ | ✗ | ✗ | **✓** prototype |
| LayerNorm | ✓ | ✗ | ✗ | **✓** prototype |
| DepthwiseConv2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| PointwiseConv2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| GlobalAvgPool2d | ✓ | ✓ | ✗ | **✓** numpy ref |
| AdaptiveAvgPool2d | ✓ | ✓ | ✗ | **✓** 僅 `(1,1)` |
| Identity | ✓ | ✓ | ✗ | **✓** numpy ref |
| GroupNorm | ✓ | ✗ | ✗ | **✓** prototype |
| ResidualBlock | ✓ | ✓ | ✗ | **✓** composite prototype |
| ConvNeXtBlock | ✓ 實驗性 | ✗ | ✗ | **✓** composite prototype |
| Dropout | ✓ | ✓ | ✗ | **✓** prototype |
| **激活函數** | | | | |
| ReLU | ✓ | ✓ | ✓ | ✓ numpy ref |
| LeakyReLU | ✓ | ✓ | ✓ | ✓ numpy ref + partial `gpu_native` |
| SiLU | ✓ | ✓ | ✗ | ✓ numpy ref + partial `gpu_native` |
| Sigmoid | ✓ | ✓ | ✗ | ✓ numpy ref + partial `gpu_native` |
| Tanh | ✓ | ✓ | ✗ | ✓ numpy ref + partial `gpu_native` |
| GELU | ✓ | ✗ | ✗ | **✓** numpy ref + partial `gpu_native` |
| **損失函數** | | | | |
| CrossEntropyLoss | ✓ | ✓ | ✓ | ✓ numpy |
| MSELoss | ✓ | ✓ | 實驗中 | **✓** numpy |
| BCEWithLogitsLoss | ✓ binary | ✓ binary | ✗ | **✓** binary |
| label_smoothing | ✓ | ✓ | ✗ | **✓** cross-entropy prototype |
| **優化器** | | | | |
| SGD | ✓ | ✓ | ✓ | ✓ numpy + gpu_native |
| Momentum SGD | ✓ | ✓ | ✓ | ✓ numpy + gpu_native |
| Adam | ✓ | ✓ | 實驗中 | **✓** numpy + gpu_native Linear |
| AdamW | ✓ | ✓ | ✗ | **✓** numpy + gpu_native Linear |
| RMSprop | ✓ | ✓ | ✗ | **✓** numpy + gpu_native Linear |
| **Scheduler** | | | | |
| 無 / 停用 | ✓ | ✓ | ✓ | ✓ |
| StepLR | ✓ | ✓ | ✗ | ✓ |
| CosineAnnealingLR | ✓ | ✓ | ✗ | ✓ |
| ReduceLROnPlateau | ✓ | ✓ | 部分支援 | ✓ |
| **正則化** | | | | |
| weight_decay | ✓ | ✓ | ✓ | ✓ |
| gradient clipping | ✓ | ✓ | ✓ | ✓ global norm |
| AMP | ✓ CUDA 限定 | ✗ | ✗ | ⚠ 實驗性 |
| **前端便利功能** | | | | |
| `model.layers[]` YAML | ✓ | ✓ | ✓ 固定 pattern | ✓ 具名 tensor wiring 的 ordered DAG |
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

最廣泛的穩定路徑，也是這個 repo 的 reference implementation。適合新模型想法、自訂 Python 元件、快速迭代，以及大多數不需要 native backend 約束的實驗。

透過 flex builder fallback 到 `torch.nn`，可使用 built-in registry 以外的 torch module 名稱。

## CPU/NumPy Autograd

刻意維持教學取向、CPU-only，並作為內部 correctness oracle。

適合框架學習、可重現測試、對照驗證，以及不依賴 torch 的小型實驗。

限制：Conv2d 比 torch 慢很多；不支援完整圖級 AMP；`LayerNorm` 與 `GroupNorm` 的 train-native 覆蓋仍偏窄、以 helper 子集為主；不支援 dotted-path 自訂元件。

## CUDA Legacy

歷史 native backend，刻意維持狹窄，定位為 maintenance-only。

穩定支援範圍：
- 資料集：`cifar10`，input shape `[3, 32, 32]`
- 層 pattern：`Conv2d → activation → Conv2d → activation → MaxPool2d → Conv2d → activation → Conv2d → activation → MaxPool2d → Flatten → Linear`
- 激活：`ReLU` 或 `LeakyReLU`
- 優化器：`SGD` 或 `Adam`
- 損失：`CrossEntropyLoss`、`MSELoss`

訓練前請先執行 `minicnn validate-dual-config`。
驗證失敗現在會回傳簡短 CLI 訊息或 JSON payload，而不是直接丟出 raw traceback。

## cuda_native（主要 native 方向，仍屬實驗）

透過 `engine.backend=cuda_native` 或 `train-native` 明確啟用。這是目前 repo 裡主要的 native 發展方向，但仍不適合正式環境。

目前通過驗證的 train-native 支援範圍：

- dataset：`random`、`cifar10`、`mnist`
- loss：`CrossEntropyLoss`、`MSELoss`
- optimizer：支援 `SGD`、`Adam`、`AdamW`、`RMSprop`，並支援 global gradient clipping
- scheduler：支援 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`，也可停用
- `train.amp=true|false`（帶 loss scaling / overflow backoff 的實驗性 mixed-precision prototype）

支援的 `gpu_native` training subsets 已透過 `grad_l2_sumsq` 加
`scale_inplace` 支援 native `optimizer.grad_clip_global`。
支援的 SGD `gpu_native` helper subsets 已透過 `sgd_update_fused`
支援 native `optimizer.weight_decay`。

支援 op：`BatchNorm2d`（forward/backward prototype）、`Concat`、`Conv2d`、`DepthwiseConv2d`、`PointwiseConv2d`、`GroupNorm`、`LayerNorm`、`LayerNorm2d`、`ResidualBlock`、`ConvNeXtBlock`、`Dropout`、`DropPath`、`Add`、`ReLU`、`LeakyReLU`、`Sigmoid`、`Tanh`、`SiLU`、`GELU`、`Identity`、`Flatten`、`Linear`、`MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d`（僅 `output_size=(1,1)`）、`GlobalAvgPool2d`。

目前驗證或 `train-native` gate 仍拒絕：不在 `SGD` / `Adam` / `AdamW` / `RMSprop` 內的 optimizer。

注意：雖然已有 backward 與 training prototype，且 `BatchNorm2d` 也已有 prototype 級的 backward，但整體 backend 仍屬實驗性，不是正式訓練後端。後續 native 能力通常也應優先長在這裡，而不是回填到 `cuda_legacy`。

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

`minicnn cuda-native-capabilities` 目前也會回傳 machine-readable
`support_tiers` / `support_tier_counts`，讓 `Stable` / `Beta` /
`Experimental` 分級可被測試與工具直接消費。

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
