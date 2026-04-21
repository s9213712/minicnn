# cuda_native Backend

`cuda_native` is an experimental graph-based backend for MiniCNN.

It is **not** a replacement for `cuda_legacy` and is **not** production-ready. It is a research prototype for graph IR, explicit memory planning, and backend-extensibility work.

## What cuda_native Is

A staged, modular backend structured in layers:

- **IR layer** (`graph.py`, `nodes.py`) — graph and tensor representation
- **Validation layer** (`validators.py`, `shapes.py`) — shape inference and legality checks
- **Planning layer** (`planner.py`) — conservative buffer allocation
- **Execution layer** (`executor.py`, `kernels.py`) — numpy reference kernels, dispatch table
- **Capability layer** (`capabilities.py`) — honest feature flags

## What cuda_native Is Not

- Not a replacement for `cuda_legacy`
- Not a production training backend
- Not backed by real CUDA kernels (uses numpy reference implementations)
- Not general-purpose (sequential graphs only, no branching)

## Current Status

| Feature | Status |
|---|---|
| Graph IR | Implemented |
| Shape inference | Basic |
| Forward execution | Basic (numpy) |
| Planner | Conservative / experimental |
| MaxPool2d, AvgPool2d | Supported (numpy ref) |
| Backward prototype | Implemented, not stable |
| Training loop | Research prototype |
| Training in production | No — not enabled |
| Dynamic graph | Not supported |
| Mixed precision | Not supported |

## Supported Ops

| Op | Forward | Backward |
|---|:---:|:---:|
| Conv2d | ✓ | Prototype |
| ReLU | ✓ | Prototype |
| LeakyReLU | ✓ | Prototype |
| MaxPool2d | ✓ | Prototype |
| AvgPool2d | ✓ | Prototype |
| Flatten | ✓ | Prototype |
| Linear | ✓ | Prototype |
| BatchNorm2d | ✗ rejected | — |
| GroupNorm | ✗ rejected | — |
| LayerNorm | ✗ rejected | — |
| ResidualBlock | ✗ rejected | — |

## How It Differs From cuda_legacy

| | cuda_legacy | cuda_native |
|---|---|---|
| Kernel type | Real CUDA / cuBLAS | NumPy reference |
| Graph | Fixed handcrafted pipeline | Explicit graph IR |
| Validation | Strict contract check | Graph-level shape and op check |
| Planner | Implicit | Explicit buffer plan |
| Dataset | CIFAR-10 only | CIFAR-10, MNIST, random |
| AvgPool2d | ✗ | ✓ |
| Training | Stable | Research prototype |
| Extension model | Not extensible | Designed to grow |

## CLI Usage

Check capabilities:

```bash
minicnn cuda-native-capabilities
```

Validate a config:

```bash
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

Run (experimental, research only):

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

Or via `train-dual`:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_native
```

## Architecture Overview

```text
Config
  └─ validators.py         (op legality, shape constraints)
  └─ graph.py + nodes.py   (IR: NativeGraph, Node, TensorSpec)
  └─ shapes.py             (shape inference per op)
  └─ planner.py            (buffer allocation plan)
  └─ executor.py           (forward dispatch)
  └─ kernels.py            (numpy reference kernels)
  └─ backward.py           (backward kernels — prototype)
  └─ loss.py               (cross_entropy, mse)
  └─ training.py           (train_step, sgd_update)
  └─ capabilities.py       (honest feature flags)
```

## Design Principles

1. Explicit over implicit — no hidden behavior
2. Fail fast — reject unsupported ops at validation time, with clear error messages
3. Honest capability boundaries — never claim support beyond tested reality
4. Correctness before optimization — conservative planner, no clever tricks until stable
5. Separation of concerns — IR, planner, and execution are distinct layers

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| Phase 0 | Scaffold, capabilities, stub API | Done |
| Phase 1 | Graph IR, shape inference, forward execution | Done |
| Phase 2 | Planner, pooling support | Done |
| Phase 3 | Backward prototype, loss, training loop | Done |
| Phase 4 | MVP stabilization, CLI, doctor, docs | Done |
| Phase 5 | BatchNorm/Residual/Concat/Memory reuse RFCs | RFC written |
| Phase 6 | Autograd, optimizer stack, broader op coverage | Future |

---

# cuda_native Backend（中文）

`cuda_native` 是 MiniCNN 的一個實驗性 graph-based backend。

它**不是** `cuda_legacy` 的替代品，也**不適合**正式環境使用。它是一個用於研究 graph IR、顯式記憶體規劃和 backend 擴展性的 prototype。

## cuda_native 是什麼

分層設計的模組化 backend：

- **IR 層**（`graph.py`, `nodes.py`）— graph 與 tensor 表示
- **驗證層**（`validators.py`, `shapes.py`）— shape inference 與合法性檢查
- **規劃層**（`planner.py`）— 保守的 buffer 分配
- **執行層**（`executor.py`, `kernels.py`）— numpy 參考 kernel，dispatch table
- **能力層**（`capabilities.py`）— 誠實的功能旗標

## cuda_native 不是什麼

- 不是 `cuda_legacy` 的替代品
- 不是正式環境的訓練 backend
- 不使用真正的 CUDA kernel（使用 numpy 參考實作）
- 不支援通用 graph（僅限 sequential graph，不支援 branching）

## 目前狀態

| 功能 | 狀態 |
|---|---|
| Graph IR | 已實作 |
| Shape inference | 基本實作 |
| Forward execution | 基本實作（numpy） |
| Planner | 保守 / 實驗中 |
| MaxPool2d、AvgPool2d | 支援（numpy ref） |
| Backward prototype | 已實作，不穩定 |
| 訓練迴圈 | 研究 prototype |
| 正式環境訓練 | 否 — 未啟用 |
| Dynamic graph | 不支援 |
| Mixed precision | 不支援 |

## 支援的 Op

| Op | Forward | Backward |
|---|:---:|:---:|
| Conv2d | ✓ | Prototype |
| ReLU | ✓ | Prototype |
| LeakyReLU | ✓ | Prototype |
| MaxPool2d | ✓ | Prototype |
| AvgPool2d | ✓ | Prototype |
| Flatten | ✓ | Prototype |
| Linear | ✓ | Prototype |
| BatchNorm2d | ✗ 拒絕 | — |
| GroupNorm | ✗ 拒絕 | — |
| LayerNorm | ✗ 拒絕 | — |
| ResidualBlock | ✗ 拒絕 | — |

## 與 cuda_legacy 的比較

| | cuda_legacy | cuda_native |
|---|---|---|
| Kernel 類型 | 真正 CUDA / cuBLAS | NumPy 參考實作 |
| Graph | 固定手寫流水線 | 顯式 graph IR |
| 驗證 | 嚴格合約檢查 | Graph 層級 shape 與 op 檢查 |
| Planner | 隱式 | 顯式 buffer 規劃 |
| 支援資料集 | 僅 CIFAR-10 | CIFAR-10、MNIST、隨機假資料 |
| AvgPool2d | ✗ | ✓ |
| 訓練 | 穩定 | 研究 prototype |
| 擴展設計 | 不易擴展 | 設計上可延伸 |

## CLI 使用方式

查看支援能力：

```bash
minicnn cuda-native-capabilities
```

驗證 config：

```bash
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

執行（僅供研究）：

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

或透過 `train-dual`：

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_native
```

## 架構概覽

```text
Config
  └─ validators.py         (op 合法性、shape 限制)
  └─ graph.py + nodes.py   (IR：NativeGraph、Node、TensorSpec)
  └─ shapes.py             (各 op 的 shape inference)
  └─ planner.py            (buffer 分配計劃)
  └─ executor.py           (forward dispatch)
  └─ kernels.py            (numpy 參考 kernel)
  └─ backward.py           (backward kernel — prototype)
  └─ loss.py               (cross_entropy、mse)
  └─ training.py           (train_step、sgd_update)
  └─ capabilities.py       (誠實的功能旗標)
```

## 設計原則

1. 顯式優於隱式 — 不隱藏行為
2. 快速失敗 — 在驗證時拒絕不支援的 op，並給出清楚的錯誤訊息
3. 誠實的能力邊界 — 不宣稱未經測試的支援
4. 正確性優先於最佳化 — 保守的 planner，穩定前不做取巧設計
5. 關注點分離 — IR、planner、execution 是獨立的層

## 開發路線圖

| 階段 | 目標 | 狀態 |
|---|---|---|
| Phase 0 | Scaffold、capabilities、stub API | 完成 |
| Phase 1 | Graph IR、shape inference、forward execution | 完成 |
| Phase 2 | Planner、pooling 支援 | 完成 |
| Phase 3 | Backward prototype、loss、training loop | 完成 |
| Phase 4 | MVP 穩定化、CLI、doctor、docs | 完成 |
| Phase 5 | BatchNorm/Residual/Concat/Memory reuse RFC | RFC 已完成 |
| Phase 6 | Autograd、optimizer stack、更廣的 op 覆蓋 | 未來 |
