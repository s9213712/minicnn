# cuda_native Backend

`cuda_native` is MiniCNN's primary native backend direction.

It is still experimental and not production-ready. The purpose of this backend
is to grow the native graph/planner/executor stack in public, while
`cuda_legacy` remains the narrow maintenance-only historical path.

## What cuda_native Is

A staged, modular backend structured in layers:

- **IR layer** (`graph.py`, `nodes.py`) — graph and tensor representation
- **Validation layer** (`validators.py`, `shapes.py`) — shape inference and legality checks
- **Planning layer** (`planner.py`) — conservative buffer allocation
- **Execution layer** (`executor.py`, `kernels.py`) — numpy reference kernels, dispatch table
- **Backward layer** (`backward.py`) — gradient kernels prototype
- **Training layer** (`loss.py`, `training.py`) — loss functions and SGD training loop
- **Capability layer** (`capabilities.py`) — honest feature flags
- **Layout layer** (`layouts.py`) — layout constants, per-op layout rules, validation
- **Memory layer** (`memory.py`) — buffer allocator and pool abstraction
- **Debug layer** (`debug.py`) — graph dump, plan dump, execution trace

## What cuda_native Is Not

- Not a production training backend
- Not backed by real CUDA kernels (uses numpy reference implementations)
- Not general-purpose (sequential graphs only, no branching)

## Current Status

| Feature | Status |
|---|---|
| Graph IR | ✓ Implemented |
| Shape inference | ✓ Basic |
| Forward execution | ✓ Basic (numpy) |
| Planner | ✓ Conservative / experimental |
| MaxPool2d, AvgPool2d | ✓ Supported (numpy ref) |
| Layout validation | ✓ `validate_graph_layouts()` |
| Memory footprint / pool | ✓ `memory_footprint()`, `BufferPool` |
| Graph / plan dump | ✓ `dump_graph()`, `dump_plan()` |
| Execution trace | ✓ `TracingForwardExecutor` |
| Backward prototype | ⚠ Implemented, not stable |
| Training loop | ⚠ Research prototype |
| Training in production | ✗ Not enabled |
| Dynamic graph | ✗ Not supported |
| Mixed precision | ✗ Not supported |

## Supported Ops

| Op | Forward | Backward |
|---|:---:|:---:|
| Conv2d | ✓ | Prototype |
| ReLU | ✓ | Prototype |
| LeakyReLU | ✓ | Prototype |
| Sigmoid | ✓ | Prototype |
| Tanh | ✓ | Prototype |
| SiLU | ✓ | Prototype |
| MaxPool2d | ✓ | Prototype |
| AvgPool2d | ✓ | Prototype |
| Flatten | ✓ | Prototype |
| Linear | ✓ | Prototype |
| BatchNorm2d | ✓ prototype (eval + train-state update) | ✓ prototype |
| GroupNorm | ✗ rejected | — |
| LayerNorm | ✗ rejected | — |
| ResidualBlock | ✗ rejected | — |

`BatchNorm2d` now has forward/backward prototype support. It is part of the
experimental training path, but remains prototype-level rather than stable.

Validated `train-native` support boundary today:

- dataset: `random`, `cifar10`, `mnist`
- loss: `CrossEntropyLoss`, `MSELoss`
- optimizer: `SGD` with optional momentum and global gradient clipping
- scheduler: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, or disabled
- `train.amp=false`, `train.grad_accum_steps=1`

## How It Differs From cuda_legacy

| | cuda_legacy | cuda_native |
|---|---|---|
| Role | maintenance-only historical backend | primary native backend direction |
| Kernel type | Real CUDA / cuBLAS | NumPy reference |
| Graph | Fixed handcrafted pipeline | Explicit graph IR |
| Validation | Strict boundary check | Graph-level shape and op check |
| Planner | Implicit | Explicit buffer plan |
| Dataset | CIFAR-10 only | CIFAR-10, MNIST, random |
| AvgPool2d | ✗ | ✓ |
| MSELoss | Experimental | ✓ |
| Layout validation | ✗ | ✓ |
| Memory planning | ✗ | ✓ |
| Graph / plan dump | ✗ | ✓ |
| Execution trace | ✗ | ✓ |
| Training | Stable | Research prototype |
| Extension model | Not extensible | Designed to grow |

## CLI Usage

Check capabilities:

```bash
minicnn cuda-native-capabilities
```

Validate a config:

```bash
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5
```

Run (experimental, research only):

```bash
minicnn train-native --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.type=random dataset.num_samples=128 dataset.val_samples=32 \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5
```

Or via `train-dual`:

```bash
minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_native
```

## Developer Tooling

```python
from minicnn.cuda_native.graph import build_graph
from minicnn.cuda_native.planner import make_naive_plan
from minicnn.cuda_native.debug import dump_graph, dump_plan, TracingForwardExecutor
from minicnn.cuda_native.layouts import validate_graph_layouts, infer_layout
from minicnn.cuda_native.memory import memory_footprint, BufferPool

graph = build_graph(layers, input_shape=(8, 3, 32, 32))

# Inspect graph structure
print(dump_graph(graph))
# NativeGraph  input=(8,3,32,32)  output=(8,10)  nodes=5
#   [0] conv2d_0    Conv2d    (8,3,32,32) -> (8,16,30,30)  out_channels=16  kernel_size=3
#   ...

# Inspect memory plan
plan = make_naive_plan(graph)
print(dump_plan(plan))
# ExecutionPlan  buffers=6  total=880.0 KB
#   step  0  conv2d_0    Conv2d    [buf_0] -> [buf_1]
#   ...

# Validate layout correctness
errors = validate_graph_layouts(graph)
assert errors == []

# Estimate memory usage
fp = memory_footprint(graph)
print(f"Total: {fp['total_kb']} KB across {fp['num_buffers']} buffers")

# Trace execution with per-node timing
ctx, trace = TracingForwardExecutor().run(graph, {'input': x}, params)
trace.print()
# ExecutionTrace  steps=5  total=2.3ms
#   conv2d_0    Conv2d    in=[(8,3,32,32)]  out=[(8,16,30,30)]  1.2ms
#   ...

# Pre-allocate a reusable buffer pool
pool = BufferPool.build(plan, graph)
ctx = pool.make_ctx(feeds={'input': x}, params=params)
pool.reset()  # zero for next call
```

## Architecture Overview

```text
Config / YAML
  └─ validators.py     (op legality, attrs, shape constraints → fail fast)
  └─ shapes.py         (per-op shape inference)
  └─ nodes.py          (TensorSpec, Node dataclasses)
  └─ graph.py          (NativeGraph, build_graph)
  └─ layouts.py        (NCHW/NC layout constants, OP_LAYOUT_RULES, validate_graph_layouts)
  └─ planner.py        (BufferPlan, ExecutionPlan, make_naive_plan)
  └─ memory.py         (BufferAllocator, BufferPool, memory_footprint)
  └─ kernels.py        (KernelRegistry, numpy reference kernels)
  └─ executor.py       (ForwardExecutor, run / run_inference / run_with_cache)
  └─ debug.py          (dump_graph, dump_plan, TracingForwardExecutor, ExecutionTrace)
  └─ backward.py       (BackwardRegistry, backward kernels — prototype)
  └─ loss.py           (cross_entropy_loss, mse_loss)
  └─ training.py       (train_step, sgd_update)
  └─ capabilities.py   (CUDA_NATIVE_CAPABILITIES, get_cuda_native_capabilities)
  └─ api.py            (validate_cuda_native_config, build_cuda_native_graph, get_capability_summary)
```

## Design Principles

1. Explicit over implicit — no hidden behavior
2. Fail fast — reject unsupported ops at validation time, with clear error messages
3. Honest capability boundaries — never claim support beyond tested reality
4. Correctness before optimization — conservative planner, no clever tricks until stable
5. Separation of concerns — IR, planner, execution, debug are distinct layers

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| Phase 0 | Scaffold, capabilities, stub API | ✓ Done |
| Phase 1 | Graph IR, shape inference, forward execution | ✓ Done |
| Phase 2 | Planner, pooling support | ✓ Done |
| Phase 3 | Backward prototype, loss, training loop | ✓ Done |
| Phase 4 | MVP stabilization, CLI, doctor, docs | ✓ Done |
| Phase 4b | Debug observability, layouts, memory layer | ✓ Done |
| Phase 5 | BatchNorm/Residual/Concat/Memory reuse RFCs | ✓ RFC written |
| Phase 6 | Autograd, optimizer stack, broader op coverage | Future |

Phase 5 RFCs: [docs/cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)

---

# cuda_native Backend（中文）

`cuda_native` 是 MiniCNN 目前主要的 native backend 方向。

它目前仍屬實驗性，也**不適合**正式環境使用。這條 backend 的目的是把
native graph/planner/executor 能力公開地逐步長出來；`cuda_legacy`
則維持為窄邊界的歷史維護路徑。

## cuda_native 是什麼

分層設計的模組化 backend：

- **IR 層**（`graph.py`, `nodes.py`）— graph 與 tensor 表示
- **驗證層**（`validators.py`, `shapes.py`）— shape inference 與合法性檢查
- **規劃層**（`planner.py`）— 保守的 buffer 分配
- **執行層**（`executor.py`, `kernels.py`）— numpy 參考 kernel，dispatch table
- **Backward 層**（`backward.py`）— 梯度 kernel prototype
- **訓練層**（`loss.py`, `training.py`）— 損失函數與 SGD 訓練迴圈
- **能力層**（`capabilities.py`）— 誠實的功能旗標
- **Layout 層**（`layouts.py`）— layout 常數、各 op 的輸入輸出規則、驗證
- **Memory 層**（`memory.py`）— buffer 分配器與 pool 抽象
- **Debug 層**（`debug.py`）— graph dump、plan dump、execution trace

## cuda_native 不是什麼

- 不是正式環境的訓練 backend
- 不使用真正的 CUDA kernel（使用 numpy 參考實作）
- 不支援通用 graph（僅限 sequential graph，不支援 branching）

## 目前狀態

| 功能 | 狀態 |
|---|---|
| Graph IR | ✓ 已實作 |
| Shape inference | ✓ 基本實作 |
| Forward execution | ✓ 基本實作（numpy） |
| Planner | ✓ 保守 / 實驗中 |
| MaxPool2d、AvgPool2d | ✓ 支援（numpy ref） |
| Layout 驗證 | ✓ `validate_graph_layouts()` |
| 記憶體估算 / pool | ✓ `memory_footprint()`、`BufferPool` |
| Graph / plan dump | ✓ `dump_graph()`、`dump_plan()` |
| Execution trace | ✓ `TracingForwardExecutor` |
| Backward prototype | ⚠ 已實作，不穩定 |
| 訓練迴圈 | ⚠ 研究 prototype |
| 正式環境訓練 | ✗ 未啟用 |
| Dynamic graph | ✗ 不支援 |
| Mixed precision | ✗ 不支援 |

## 支援的 Op

| Op | Forward | Backward |
|---|:---:|:---:|
| Conv2d | ✓ | Prototype |
| ReLU | ✓ | Prototype |
| LeakyReLU | ✓ | Prototype |
| Sigmoid | ✓ | Prototype |
| Tanh | ✓ | Prototype |
| SiLU | ✓ | Prototype |
| MaxPool2d | ✓ | Prototype |
| AvgPool2d | ✓ | Prototype |
| Flatten | ✓ | Prototype |
| Linear | ✓ | Prototype |
| BatchNorm2d | ✓ prototype（eval + train 狀態更新） | ✓ prototype |
| GroupNorm | ✗ 拒絕 | — |
| LayerNorm | ✗ 拒絕 | — |
| ResidualBlock | ✗ 拒絕 | — |

`BatchNorm2d` 現在已有 forward/backward prototype，已可進入實驗性訓練路徑，
但整體仍屬 prototype 層級，不能視為穩定支援。

目前通過驗證的 `train-native` 支援範圍：

- dataset：`random`、`cifar10`、`mnist`
- loss：`CrossEntropyLoss`、`MSELoss`
- optimizer：支援 `SGD`，可選 momentum 與 global gradient clipping
- scheduler：支援 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`，也可停用
- `train.amp=false`、`train.grad_accum_steps=1`

## 與 cuda_legacy 的比較

| | cuda_legacy | cuda_native |
|---|---|---|
| 角色 | 歷史維護 backend | 主要 native backend 方向 |
| Kernel 類型 | 真正 CUDA / cuBLAS | NumPy 參考實作 |
| Graph | 固定手寫流水線 | 顯式 graph IR |
| 驗證 | 嚴格邊界檢查 | Graph 層級 shape 與 op 檢查 |
| Planner | 隱式 | 顯式 buffer 規劃 |
| 支援資料集 | 僅 CIFAR-10 | CIFAR-10、MNIST、隨機假資料 |
| AvgPool2d | ✗ | ✓ |
| MSELoss | 實驗中 | ✓ |
| Layout 驗證 | ✗ | ✓ |
| 記憶體規劃 | ✗ | ✓ |
| Graph / plan dump | ✗ | ✓ |
| Execution trace | ✗ | ✓ |
| 訓練 | 穩定 | 研究 prototype |
| 擴展設計 | 不易擴展 | 設計上可延伸 |

## CLI 使用方式

```bash
minicnn cuda-native-capabilities
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
minicnn train-native --config configs/dual_backend_cnn.yaml \
  train.epochs=1 dataset.num_samples=128 dataset.val_samples=32
```

## 開發者工具

```python
from minicnn.cuda_native.debug import dump_graph, dump_plan, TracingForwardExecutor
from minicnn.cuda_native.layouts import validate_graph_layouts
from minicnn.cuda_native.memory import memory_footprint, BufferPool

# 查看 graph 結構
print(dump_graph(graph))

# 查看 buffer 分配計劃
plan = make_naive_plan(graph)
print(dump_plan(plan))

# 驗證 layout 正確性
errors = validate_graph_layouts(graph)

# 估算記憶體用量
fp = memory_footprint(graph)
print(f"Total: {fp['total_kb']} KB, {fp['num_buffers']} buffers")

# 帶 per-node 時序的 trace 執行
ctx, trace = TracingForwardExecutor().run(graph, {'input': x}, params)
trace.print()

# 預分配可重用 buffer pool
pool = BufferPool.build(plan, graph)
ctx = pool.make_ctx(feeds={'input': x}, params=params)
pool.reset()
```

## 架構概覽

```text
Config / YAML
  └─ validators.py   （op 合法性、attrs、shape 限制 → 快速失敗）
  └─ shapes.py       （各 op 的 shape inference）
  └─ nodes.py        （TensorSpec、Node dataclasses）
  └─ graph.py        （NativeGraph、build_graph）
  └─ layouts.py      （NCHW/NC 常數、OP_LAYOUT_RULES、validate_graph_layouts）
  └─ planner.py      （BufferPlan、ExecutionPlan、make_naive_plan）
  └─ memory.py       （BufferAllocator、BufferPool、memory_footprint）
  └─ kernels.py      （KernelRegistry、numpy 參考 kernel）
  └─ executor.py     （ForwardExecutor：run / run_inference / run_with_cache）
  └─ debug.py        （dump_graph、dump_plan、TracingForwardExecutor、ExecutionTrace）
  └─ backward.py     （BackwardRegistry、backward kernel — prototype）
  └─ loss.py         （cross_entropy_loss、mse_loss）
  └─ training.py     （train_step、sgd_update）
  └─ capabilities.py （CUDA_NATIVE_CAPABILITIES、get_cuda_native_capabilities）
  └─ api.py          （validate_cuda_native_config、build_cuda_native_graph）
```

## 設計原則

1. 顯式優於隱式 — 不隱藏行為
2. 快速失敗 — 在驗證時拒絕不支援的 op，並給出清楚的錯誤訊息
3. 誠實的能力邊界 — 不宣稱未經測試的支援
4. 正確性優先於最佳化 — 保守的 planner，穩定前不做取巧設計
5. 關注點分離 — IR、planner、execution、debug 是獨立的層

## 開發路線圖

| 階段 | 目標 | 狀態 |
|---|---|---|
| Phase 0 | Scaffold、capabilities、stub API | ✓ 完成 |
| Phase 1 | Graph IR、shape inference、forward execution | ✓ 完成 |
| Phase 2 | Planner、pooling 支援 | ✓ 完成 |
| Phase 3 | Backward prototype、loss、training loop | ✓ 完成 |
| Phase 4 | MVP 穩定化、CLI、doctor、docs | ✓ 完成 |
| Phase 4b | Debug observability、layouts、memory 層 | ✓ 完成 |
| Phase 5 | BatchNorm/Residual/Concat/Memory reuse RFC | ✓ RFC 已完成 |
| Phase 6 | Autograd、optimizer stack、更廣的 op 覆蓋 | 未來 |

Phase 5 RFC：[docs/cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)
