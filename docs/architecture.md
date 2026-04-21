# MiniCNN Architecture

MiniCNN has one broad frontend surface and multiple backend-oriented execution paths. Not every backend accepts the full frontend contract.

## Execution Paths

| Path | Command | Backend | Purpose |
|---|---|---|---|
| flex | `train-flex` | PyTorch | broad experimentation, custom components |
| dual | `train-dual` | `torch` or `cuda_legacy` | compare one shared config across two backends |
| autograd | `train-autograd` | NumPy | learning and framework-level experiments |
| native | `train-native` | `cuda_native` | experimental graph-based backend prototype |

## High-Level Layout

```text
shared YAML / CLI frontend
        |
        +--> train-flex -------> torch
        |
        +--> train-dual -------> torch | cuda_legacy
        |
        +--> train-autograd ---> NumPy autograd
        |
        +--> train-native -----> cuda_native [EXPERIMENTAL]
```

## Training Flow

```text
YAML config
    |
    +--> flex/config.py -------------------------> flex/trainer.py (torch)
    |
    +--> unified/config.py --> engine.backend? --> unified/trainer.py
    |                                              |-> torch path
    |                                              |-> cuda_legacy path
    |                                              `-> cuda_native path [experimental]
    |
    `--> train_autograd.py (NumPy)
```

## Compiler / Runtime Inference Pipeline

Separate from training — for architecture inspection and CPU inference:

```text
model config
    |
    -> compiler/tracer.py
    -> compiler/optimizer.py
    -> runtime/pipeline.py
    -> runtime/executor.py
```

## cuda_native Backend

A staged, modular experimental backend under `src/minicnn/cuda_native/`:

- `graph.py`, `nodes.py` — graph IR (NativeGraph, Node, TensorSpec)
- `validators.py`, `shapes.py` — shape inference and legality checks
- `planner.py` — conservative buffer allocation
- `executor.py`, `kernels.py` — numpy reference kernels and dispatch
- `backward.py` — backward pass prototype
- `loss.py`, `training.py` — loss and SGD training loop prototype
- `capabilities.py` — honest feature flags

Capability descriptor marks it as: experimental, sequential-only, numpy-only, not production-ready.

See [backend_capabilities.md](backend_capabilities.md) for the full support matrix.

## Module Map

```text
src/minicnn/
├── cli.py                 # public CLI entrypoint
├── flex/                  # torch/flex frontend: config, builder, trainer, registry, data
├── unified/               # shared-config loader and dispatch to torch/cuda_legacy/cuda_native
├── training/              # cuda_legacy orchestration and NumPy autograd trainer
├── framework/             # healthcheck / diagnostics / registry surface
├── compiler/              # tracer, optimizer passes, lowering boundary
├── runtime/               # runtime graph, backend abstraction, executor, memory, profiler
├── cuda_native/           # experimental graph/planner/numpy-executor backend
├── nn/                    # NumPy autograd modules and layers
├── ops/                   # differentiable NumPy ops
├── optim/                 # NumPy-side optimizers
├── schedulers/            # NumPy-side schedulers
├── models/                # NumPy model registry, builder, shape inference
├── config/                # ExperimentConfig schema and legacy settings bridge
├── core/                  # native build helpers and ctypes CUDA binding
└── data/                  # CIFAR-10 and MNIST loaders
```

## Backend Boundaries

- `torch/flex` is the default home for new layer ideas
- `cuda_legacy` is a narrow backend with validator-enforced limits
- `autograd` is for learning and tests, not throughput
- `cuda_native` is a graph-based research prototype, not a drop-in replacement for any stable backend

See [backend_capabilities.md](backend_capabilities.md) for the support matrix and [dual_backend_guide.md](dual_backend_guide.md) for change-impact guidance.

## Adding New Functionality

### New torch/flex layer

Add or import the layer on the torch/flex side and keep the change scoped there unless another backend also needs it.

### New NumPy autograd op

Implement the differentiable op in `src/minicnn/ops/`, add the corresponding layer/module in `src/minicnn/nn/`, and register it through the NumPy model builder path.

### New cuda_legacy training op

Expect to touch both Python and native code:

- `src/minicnn/unified/cuda_legacy.py`
- `src/minicnn/training/`
- `src/minicnn/core/cuda_backend.py`
- `cpp/src/`

### New cuda_native capability

Keep it inside `src/minicnn/cuda_native/`. The capability descriptor, validator, and CLI surface must all be coherent before claiming support.

---

# MiniCNN 架構（中文）

MiniCNN 有一個廣泛的前端介面，對應多條以 backend 為導向的執行路徑。並非每個 backend 都接受完整的前端合約。

## 執行路徑

| 路徑 | 指令 | Backend | 用途 |
|---|---|---|---|
| flex | `train-flex` | PyTorch | 廣泛模型實驗、自訂元件 |
| dual | `train-dual` | `torch` 或 `cuda_legacy` | 同一份 config 跨兩個 backend 比較 |
| autograd | `train-autograd` | NumPy | 框架學習與實驗 |
| native | `train-native` | `cuda_native` | 實驗性 graph-based backend prototype |

## 高層次架構圖

```text
shared YAML / CLI frontend
        |
        +--> train-flex -------> torch
        |
        +--> train-dual -------> torch | cuda_legacy
        |
        +--> train-autograd ---> NumPy autograd
        |
        +--> train-native -----> cuda_native [實驗]
```

## 訓練流程

```text
YAML config
    |
    +--> flex/config.py -------------------------> flex/trainer.py (torch)
    |
    +--> unified/config.py --> engine.backend? --> unified/trainer.py
    |                                              |-> torch 路徑
    |                                              |-> cuda_legacy 路徑
    |                                              `-> cuda_native 路徑 [實驗]
    |
    `--> train_autograd.py (NumPy)
```

## Compiler / Runtime 推論流水線

獨立於訓練之外，用於架構檢查和 CPU 推論：

```text
model config
    |
    -> compiler/tracer.py
    -> compiler/optimizer.py
    -> runtime/pipeline.py
    -> runtime/executor.py
```

## cuda_native Backend

分階段設計的實驗性 backend，位於 `src/minicnn/cuda_native/`：

- `graph.py`, `nodes.py` — graph IR（NativeGraph、Node、TensorSpec）
- `validators.py`, `shapes.py` — shape inference 與合法性檢查
- `planner.py` — 保守的 buffer 分配計劃
- `executor.py`, `kernels.py` — numpy 參考 kernel 與 dispatch
- `backward.py` — backward pass prototype
- `loss.py`, `training.py` — loss 與 SGD 訓練迴圈 prototype
- `capabilities.py` — 誠實的功能旗標

Capability descriptor 標記為：實驗性、僅支援 sequential graph、numpy-only、非正式環境可用。

完整支援矩陣見 [backend_capabilities.md](backend_capabilities.md)。

## 模組對應圖

```text
src/minicnn/
├── cli.py                 # 公開 CLI 入口
├── flex/                  # torch/flex 前端：config、builder、trainer、registry、data
├── unified/               # shared-config 載入器，分發至 torch/cuda_legacy/cuda_native
├── training/              # cuda_legacy orchestration 與 NumPy autograd trainer
├── framework/             # healthcheck / diagnostics / registry surface
├── compiler/              # tracer、optimizer passes、lowering boundary
├── runtime/               # runtime graph、backend 抽象、executor、memory、profiler
├── cuda_native/           # 實驗性 graph/planner/numpy-executor backend
├── nn/                    # NumPy autograd modules 與 layers
├── ops/                   # 可微分 NumPy ops
├── optim/                 # NumPy 端 optimizers
├── schedulers/            # NumPy 端 schedulers
├── models/                # NumPy model registry、builder、shape inference
├── config/                # ExperimentConfig schema 與 legacy settings bridge
├── core/                  # native build helpers 與 ctypes CUDA binding
└── data/                  # CIFAR-10 與 MNIST loaders
```

## Backend 邊界

- `torch/flex` 是新層想法的預設家
- `cuda_legacy` 是有 validator 強制限制的窄 backend
- `autograd` 用於學習和測試，不追求吞吐量
- `cuda_native` 是 graph-based 研究 prototype，不是任何穩定 backend 的直接替代品

## 新增功能指引

### 新的 torch/flex 層

在 torch/flex 端新增或匯入，除非另一個 backend 也需要，否則不要跨越邊界。

### 新的 NumPy autograd op

在 `src/minicnn/ops/` 實作可微分 op，在 `src/minicnn/nn/` 新增對應 layer/module，並透過 NumPy model builder 路徑註冊。

### 新的 cuda_legacy 訓練 op

需要同時動 Python 和 native 程式碼：

- `src/minicnn/unified/cuda_legacy.py`
- `src/minicnn/training/`
- `src/minicnn/core/cuda_backend.py`
- `cpp/src/`

### 新的 cuda_native 能力

限制在 `src/minicnn/cuda_native/` 內部。capability descriptor、validator、CLI surface 必須都一致，才能宣告支援。
