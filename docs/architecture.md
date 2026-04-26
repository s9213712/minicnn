# MiniCNN Architecture

MiniCNN has one broad frontend surface and multiple backend-oriented execution paths. Not every backend accepts the full frontend interface.

## Execution Paths

| Path | Command | Backend | Purpose |
|---|---|---|---|
| flex | `train-flex` | PyTorch | reference implementation and first stop for new features |
| dual | `train-dual` | `torch` or `cuda_legacy` | compare shared configs against the historical native path |
| autograd | `train-autograd` | NumPy | correctness oracle and framework-level experiments |
| native | `train-native` | `cuda_native` | primary native backend direction, now beta-grade |

## Backend Roles

- `torch/flex` is the reference implementation
- `cuda_native` is the primary native backend direction
- `autograd` is the internal correctness oracle
- `cuda_legacy` is the maintenance-only historical backend

## Feature Rollout Order

Default order for new capability work:

1. `torch/flex`
2. `autograd` when a correctness reference is useful
3. `cuda_native`
4. `cuda_legacy` only for maintenance and compatibility fixes

## High-Level Layout

```text
shared YAML / CLI frontend
        |
        +--> train-flex -------> torch [REFERENCE]
        |
        +--> train-dual -------> torch | cuda_legacy [HISTORICAL]
        |
        +--> train-autograd ---> NumPy autograd [ORACLE]
        |
        +--> train-native -----> cuda_native [PRIMARY NATIVE, BETA]
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
    |                                              `-> cuda_native path [beta]
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

A staged, modular beta-grade backend under `src/minicnn/cuda_native/`. This is the main native growth path in the repo:

- `graph.py`, `nodes.py` — graph IR (NativeGraph, Node, TensorSpec)
- `validators.py`, `shapes.py` — shape inference and legality checks
- `planner.py` — conservative buffer allocation
- `executor.py`, `kernels.py` — numpy reference kernels and dispatch
- `backward.py` — backward pass prototype
- `loss.py`, `training.py` — loss and SGD training loop prototype
- `capabilities.py` — honest feature flags
- `layouts.py` — layout constants (NCHW/NC), per-op layout rules, `validate_graph_layouts()`
- `memory.py` — `BufferAllocator`, `BufferPool`, `memory_footprint()`
- `debug.py` — `dump_graph()`, `dump_plan()`, `TracingForwardExecutor`, `ExecutionTrace`

Capability descriptor marks it as: beta-grade, GPU-first for eligible `gpu_native` subsets, with `reference_numpy` retained as explicit fallback/parity execution.

See [backend_capabilities.md](backend_capabilities.md) for the full support matrix.

## Module Map

```text
src/minicnn/
├── cli.py                 # public CLI entrypoint
├── flex/                  # torch/flex reference implementation: config, builder, trainer, registry, data
├── unified/               # shared-config loader and dispatch to torch/cuda_legacy/cuda_native
├── training/              # cuda_legacy orchestration and NumPy autograd oracle trainer
├── framework/             # healthcheck / diagnostics surface
├── compiler/              # tracer and optimizer passes
├── runtime/               # runtime graph, executor, memory, profiler
├── cuda_native/           # primary native backend direction: graph/planner/reference+gpu-native executor
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

- `torch/flex` is the default home for new layer ideas and backend-agnostic feature work
- `autograd` is the oracle path for deterministic checks and learning, not throughput
- `cuda_native` is the primary native direction and should absorb future native growth
- `cuda_legacy` is a narrow historical backend with validator-enforced limits and maintenance-only scope
- `healthcheck`, `doctor`, and `smoke` are JSON-friendly CLI surfaces for automation
- those diagnostics also support `--format text` while keeping `json` as the default
- config and override mistakes fail at the CLI boundary with short exit-code-2 messages instead of Python tracebacks

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

MiniCNN 有一個廣泛的前端介面，對應多條以 backend 為導向的執行路徑。並非每個 backend 都接受完整的前端設定介面。

## 執行路徑

| 路徑 | 指令 | Backend | 用途 |
|---|---|---|---|
| flex | `train-flex` | PyTorch | reference implementation，也是新功能第一站 |
| dual | `train-dual` | `torch` 或 `cuda_legacy` | 用 shared config 對照歷史 native 路徑 |
| autograd | `train-autograd` | NumPy | correctness oracle 與框架實驗 |
| native | `train-native` | `cuda_native` | 主要 native backend 方向，現為 beta |

## Backend 角色

- `torch/flex` 是 reference implementation
- `cuda_native` 是主要 native backend 方向
- `autograd` 是內部 correctness oracle
- `cuda_legacy` 是 maintenance-only 的歷史 backend

## 新功能 rollout 順序

新增能力時，預設順序是：

1. `torch/flex`
2. 若需要 correctness 對照，再補 `autograd`
3. 再推進 `cuda_native`
4. `cuda_legacy` 只做維護與相容性修補

## 高層次架構圖

```text
shared YAML / CLI frontend
        |
        +--> train-flex -------> torch [REFERENCE]
        |
        +--> train-dual -------> torch | cuda_legacy [HISTORICAL]
        |
        +--> train-autograd ---> NumPy autograd [ORACLE]
        |
        +--> train-native -----> cuda_native [PRIMARY NATIVE, BETA]
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
    |                                              `-> cuda_native 路徑 [beta]
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

分階段設計的 beta 級 backend，位於 `src/minicnn/cuda_native/`，也是目前 repo 裡主要的 native 成長方向：

- `graph.py`, `nodes.py` — graph IR（NativeGraph、Node、TensorSpec）
- `validators.py`, `shapes.py` — shape inference 與合法性檢查
- `planner.py` — 保守的 buffer 分配計劃
- `executor.py`, `kernels.py` — GPU-first executor 與 reference/native dispatch
- `backward.py` — backward pass execution
- `loss.py`, `training.py` — loss 與 training loop
- `capabilities.py` — 誠實的功能旗標
- `layouts.py` — layout 常數（NCHW/NC）、各 op 的規則、`validate_graph_layouts()`
- `memory.py` — `BufferAllocator`、`BufferPool`、`memory_footprint()`
- `debug.py` — `dump_graph()`、`dump_plan()`、`TracingForwardExecutor`、`ExecutionTrace`

Capability descriptor 標記為：beta、ordered DAG、符合條件時 GPU-first `gpu_native`、`reference_numpy` 明確保留為 fallback/parity path、仍非 production-ready。

完整支援矩陣見 [backend_capabilities.md](backend_capabilities.md)。

## 模組對應圖

```text
src/minicnn/
├── cli.py                 # 公開 CLI 入口
├── flex/                  # torch/flex reference implementation：config、builder、trainer、registry、data
├── unified/               # shared-config 載入器，分發至 torch/cuda_legacy/cuda_native
├── training/              # cuda_legacy orchestration 與 NumPy autograd oracle trainer
├── framework/             # healthcheck / diagnostics surface
├── compiler/              # tracer 與 optimizer pass
├── runtime/               # runtime graph、executor、memory、profiler
├── cuda_native/           # 主要 native backend 方向：graph/planner/reference+gpu-native executor
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

- `torch/flex` 是新 layer 想法與 backend-agnostic 功能的預設落點
- `autograd` 是 deterministic 檢查與學習用途的 oracle 路徑，不追求吞吐量
- `cuda_native` 是主要 native 成長方向，後續 native 能力應優先長在這裡
- `cuda_legacy` 是有 validator 強制限制的歷史 backend，定位為 maintenance-only
- `healthcheck`、`doctor`、`smoke` 目前都是適合 automation 的 JSON-friendly CLI 介面
- 這些診斷命令也支援 `--format text`，但仍以 `json` 作為預設格式
- config 或 override 錯誤會在 CLI 邊界以簡短訊息和 exit code 2 失敗，不再直接吐出 Python traceback

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

## CUDA Native source layout note

`src/minicnn/cuda_native/gpu_training.py` remains the compatibility import surface for GPU training helpers. Result dataclasses, shared CUDA helpers, linear, pool, norm, conv-family, base depthwise, single-pointwise bridge, and two-pointwise activation bridge helpers now live in focused `gpu_training_*` modules. Runtime context, diagnostics, training-loop execution, GPU-native batch dispatch routing/family helpers, and GPU-native plan selection now live in focused `src/minicnn/unified/_cuda_native_*` modules. Lowering is also split into focused `gpu_lowering_*` modules for normalization, shape aliases, merge ops, activations, conv/pool ops, and registry assembly.
