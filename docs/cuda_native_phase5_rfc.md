# cuda_native Phase 5 — Extension RFCs

These RFCs describe how four high-risk feature areas could enter `cuda_native` in future phases.

None of these are implemented in the current MVP. Each RFC defines:
- what problem it solves
- what architecture changes it requires
- what risks it introduces
- what the recommended implementation sequence is

The goal is to establish clear design constraints **before** coding starts, so that future implementation work does not break the existing forward-only MVP.

---

## RFC-01: Normalization Layers (BatchNorm2d / GroupNorm / LayerNorm)

### Problem

Normalization layers are among the most commonly used in modern CNNs. Currently `cuda_native` rejects them at validation time. This RFC defines what it would take to support them correctly.

### Why They Are Hard

Normalization layers have **stateful behavior** that depends on execution mode:

| Behavior | Training mode | Inference mode |
|---|---|---|
| Statistics source | Computed from current batch | Uses running mean/var |
| Running stats | Updated (momentum) | Read-only |
| Backward | Non-trivial gradient through normalization | N/A |

This means a normalization layer is not a pure function of its inputs. It requires:
- an explicit **mode flag** (train vs eval) propagated through the executor
- **persistent state buffers** (running_mean, running_var) separate from activation buffers
- a planner that can distinguish **activation buffers** from **parameter buffers** from **statistic buffers**

### Required Architecture Changes

**1. Node metadata extension**

`Node` needs a `trainable_state` field to distinguish nodes that carry persistent mutable state:

```python
@dataclass
class Node:
    ...
    trainable_state: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # e.g. {'running_mean': (C,), 'running_var': (C,)}
```

**2. ExecutionContext mode**

The executor needs an explicit mode:

```python
class ForwardExecutor:
    def run(self, graph, feeds, params, mode: str = 'eval'):
        ...
```

In `'train'` mode, normalization kernels update running stats in-place.
In `'eval'` mode, they read running stats without updating.

**3. Planner buffer classification**

`BufferPlan` needs to distinguish buffer types:

```python
class BufferType(enum.Enum):
    ACTIVATION = 'activation'
    PARAMETER  = 'parameter'
    STATISTIC  = 'statistic'   # running_mean, running_var
    GRADIENT   = 'gradient'
```

**4. Kernel implementation**

BatchNorm2d numpy reference kernel:

```python
def _kernel_batchnorm2d(node, ctx):
    x = ctx[node.inputs[0]]
    gamma = ctx[f'_gamma_{node.name}']
    beta  = ctx[f'_beta_{node.name}']
    mode  = ctx.get('__mode__', 'eval')
    eps   = node.attrs.get('eps', 1e-5)
    momentum = node.attrs.get('momentum', 0.1)

    if mode == 'train':
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var  = x.var(axis=(0, 2, 3), keepdims=True)
        # update running stats
        ctx[f'_rm_{node.name}'] = (1 - momentum) * ctx[f'_rm_{node.name}'] + momentum * mean.squeeze()
        ctx[f'_rv_{node.name}'] = (1 - momentum) * ctx[f'_rv_{node.name}'] + momentum * var.squeeze()
    else:
        mean = ctx[f'_rm_{node.name}'].reshape(1, -1, 1, 1)
        var  = ctx[f'_rv_{node.name}'].reshape(1, -1, 1, 1)

    x_hat = (x - mean) / np.sqrt(var + eps)
    ctx[node.outputs[0]] = gamma.reshape(1, -1, 1, 1) * x_hat + beta.reshape(1, -1, 1, 1)
```

### Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Silent train/eval mode mismatch | High | Executor must fail loudly if mode is not set |
| Running stats not initialized | High | Validator must check stats buffers exist before execution |
| Backward through normalization is complex | Medium | Implement forward-only first; backward in a later sub-phase |
| GroupNorm / LayerNorm share similar issues | Low | Use the same buffer-classification design |

### Recommended Sequence

1. Extend `Node` with `trainable_state`
2. Add `BufferType` enum to planner
3. Implement `BatchNorm2d` forward (eval mode only) with tests
4. Implement `BatchNorm2d` forward (train mode) with running stat update
5. Extend validator to check that required stat buffers exist
6. Implement backward (separate sub-phase)
7. Add `GroupNorm` and `LayerNorm` using the same pattern

### Not In Scope For This RFC

- Synchronized BatchNorm (multi-GPU)
- FP16 / AMP normalization paths
- Fused BN+ReLU kernel

---

## RFC-02: ResidualBlock and Branching Graph

### Problem

ResidualBlock requires a **branch-merge graph topology**: one input feeds two paths, whose outputs are added elementwise before continuing. The current `NativeGraph` is strictly sequential (one input → one output per step). Supporting residual connections requires DAG execution.

### Why This Is Hard

The current executor assumes:

```
node_0 → node_1 → node_2 → ...
```

Residual requires:

```
input
  ├── node_A (main path: Conv → BN → ReLU → Conv → BN)
  └── node_B (shortcut: Identity or 1×1 Conv)
        ↓
      Add (elementwise)
        ↓
      ReLU
```

This means:
- `NativeGraph` must support multiple input edges per node
- The executor must track which tensors are still alive (not yet consumed)
- The planner must not release a buffer until all its consumers have run
- Shape inference must handle `Add` nodes with two inputs

### Required Architecture Changes

**1. Graph IR: remove sequential-only assumption**

`NativeGraph` currently only supports `node.inputs: list[str]` of length 1. Supporting branching requires nodes with multiple named inputs:

```python
# Add node example
Node(
    name='add_0',
    op_type='Add',
    inputs=['main_out', 'shortcut_out'],
    outputs=['added'],
    ...
)
```

The `build_graph()` function would need to be replaced or extended with a DAG builder that accepts explicit edge declarations.

**2. Executor: topological sort + live tensor tracking**

The executor must run nodes in topological order (not insertion order) and track tensor liveness:

```python
def run(self, graph, feeds, params):
    ctx = dict(feeds)
    ctx.update(params)
    for node in graph.topological_order():   # real topo sort, not insertion order
        kernel = self.registry.get(node.op_type)
        kernel(node, ctx)
        # future: release ctx[tensor] when last consumer has run
    return ctx
```

`topological_order()` must implement a real Kahn's algorithm or DFS-based sort.

**3. Planner: consumer count tracking**

The planner must record how many nodes consume each tensor, so buffers are not freed prematurely:

```python
@dataclass
class BufferPlan:
    ...
    consumer_count: dict[str, int]  # tensor_name -> number of consumers
```

**4. Shape inference: Add node**

```python
def infer_add(input_shape_a, input_shape_b, node_name):
    if input_shape_a != input_shape_b:
        raise ValueError(
            f'Add node {node_name}: shape mismatch {input_shape_a} vs {input_shape_b}'
        )
    return input_shape_a
```

**5. Shortcut projection (1×1 Conv)**

When input and output channels differ in a residual block, the shortcut path needs a 1×1 Conv projection. This is a standard residual pattern and requires no new op types — just a Conv2d with `kernel_size=1`.

### Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Breaking the sequential fast-path | High | Keep sequential path as a special case; don't force DAG overhead on linear graphs |
| Incorrect topological sort | High | Comprehensive tests with cycle detection |
| Buffer freed too early | High | Consumer-count tracking in planner |
| Shape mismatch at Add node | Medium | Validator catches this at build time |
| Backward through Add (gradient fan-out) | Medium | Handle in a separate backward RFC |

### Recommended Sequence

1. Add `Add` op to `kernels.py` and `shapes.py`
2. Implement real topological sort in `NativeGraph`
3. Add consumer-count tracking to planner
4. Build a minimal 2-branch test graph (no BN, no projection)
5. Add projection shortcut support (1×1 Conv path)
6. Build full ResidualBlock test
7. Add backward support (separate sub-phase)

### Not In Scope For This RFC

- Multi-branch fan-out beyond 2 paths
- Attention / cross-attention (requires more complex routing)
- Dynamic branching (branch condition at runtime)

---

## RFC-03: Concat and Elementwise Ops

### Problem

Many architectures require tensor combination ops beyond `Add`:
- `Concat` along the channel dimension (used in DenseNet, U-Net skip connections)
- `Mul` (used in gating, attention)
- `Sub`, `Div` (used in normalization, loss, etc.)

These are prerequisites for supporting architectures beyond simple residual CNNs.

### Required Architecture Changes

**1. Shape inference for Concat**

```python
def infer_concat(input_shapes, dim, node_name):
    # All shapes must match except along `dim`
    ref = list(input_shapes[0])
    for s in input_shapes[1:]:
        if len(s) != len(ref):
            raise ValueError(f'Concat {node_name}: rank mismatch')
        for i, (a, b) in enumerate(zip(ref, s)):
            if i == dim:
                ref[i] += b
            elif a != b:
                raise ValueError(f'Concat {node_name}: shape mismatch on axis {i}: {a} vs {b}')
    return tuple(ref)
```

**2. Kernel implementations**

```python
def _kernel_concat(node, ctx):
    arrays = [ctx[inp] for inp in node.inputs]
    dim = node.attrs.get('dim', 1)
    ctx[node.outputs[0]] = np.concatenate(arrays, axis=dim)

def _kernel_mul(node, ctx):
    a = ctx[node.inputs[0]]
    b = ctx[node.inputs[1]]
    ctx[node.outputs[0]] = a * b
```

**3. Multi-input node support**

`Concat` and `Mul` both have multiple inputs. This requires the same DAG infrastructure described in RFC-02. RFC-03 should be implemented **after** RFC-02 DAG support is in place.

**4. Validator rules**

- `Concat`: all inputs must have the same rank; all dims except concat dim must match
- `Mul`, `Add`, `Sub`: all inputs must have identical shapes (or be broadcastable — defer broadcasting to a later phase)
- Broadcasting support should be explicitly gated behind a capability flag until tested

### Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Depends on RFC-02 DAG infrastructure | High | Do not implement before RFC-02 is merged |
| Broadcasting complexity | Medium | Start with exact-shape-only; add broadcasting later |
| Backward for Concat (gradient splitting) | Medium | Implement forward-only first |

### Recommended Sequence

1. Complete RFC-02 (DAG executor)
2. Add `Add` kernel (already needed for RFC-02)
3. Add `Concat` kernel and shape inference
4. Add `Mul` kernel
5. Add validator rules for all three
6. Add backward for `Add` (gradient fan-out: copy grad to both inputs)
7. Add backward for `Concat` (gradient split by original sizes)
8. Add `Mul` backward separately

### Not In Scope For This RFC

- Broadcasting (defer until after basic exact-shape ops are tested)
- Attention / softmax (separate RFC)
- Reduction ops (sum, mean — separate RFC)

---

## RFC-04: Memory Reuse Optimization

### Problem

The current `make_naive_plan()` allocates one unique buffer per tensor with no reuse. For deep networks, this wastes memory proportional to the number of layers. A smarter planner can reuse buffers whose tensors have disjoint lifetimes.

### Current State

The naive plan for a 5-layer sequential network:

```
buf_0: input       2048 B
buf_1: after_conv  4608 B
buf_2: after_relu  4608 B
buf_3: after_flat   576 B
buf_4: output        40 B
```

Total: 5 buffers, 11880 B. With reuse, buffers 1 and 3 could share memory once buffer 1 is no longer needed.

### Lifetime Analysis

A tensor's **live range** is `[first_write_step, last_read_step]`. Two tensors can share a buffer if their live ranges do not overlap.

For a sequential graph, the live ranges are trivial:
- tensor `t_i` is written at step `i` and read at step `i+1`
- so `t_i` and `t_j` can share a buffer if `|i - j| >= 2`

For a DAG graph (RFC-02), lifetimes require proper liveness analysis.

### Proposed Planner API Extension

The planner should expose a `strategy` parameter without changing the existing `make_naive_plan()` interface:

```python
def make_plan(graph, strategy: str = 'naive') -> ExecutionPlan:
    if strategy == 'naive':
        return make_naive_plan(graph)
    if strategy == 'linear_reuse':
        return make_linear_reuse_plan(graph)
    raise ValueError(f'Unknown planner strategy: {strategy!r}')
```

This ensures callers of `make_naive_plan()` are not broken.

### Linear Reuse Algorithm (Sequential Graphs Only)

```python
def make_linear_reuse_plan(graph):
    # For sequential graphs only:
    # Odd-indexed tensors share one buffer pool, even-indexed share another.
    # This is safe because each tensor is consumed exactly once in order.
    pool_a = 'reuse_buf_A'
    pool_b = 'reuse_buf_B'
    tensor_to_buffer = {}
    for i, node in enumerate(graph.topological_order()):
        for spec in node.output_specs:
            tensor_to_buffer[spec.name] = pool_a if i % 2 == 0 else pool_b
    # Buffer sizes are the max of all tensors assigned to each pool
    ...
```

This reduces a sequential N-layer network from N+1 buffers to 3 (input + pool_A + pool_B).

### Full Lifetime-Based Reuse (DAG Graphs)

For general DAG graphs, use an interval graph coloring approach:
1. Compute live ranges for all tensors via topological traversal
2. Sort tensors by live range start
3. Greedily assign the smallest compatible free buffer, or allocate a new one

This is the standard approach used by compilers (LLVM register allocation).

### Correctness Requirements

Before enabling any reuse:
- Reuse must be proven safe via live range non-overlap
- Tests must compare naive vs reuse plan outputs on the same inputs (outputs must be identical)
- A `validate_plan_correctness(graph, plan, feeds, params)` helper should be added to `debug.py`

### Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Incorrect reuse causing value corruption | Critical | Always compare against naive plan output before merging |
| Reuse breaking DAG graphs where lifetimes are non-trivial | High | Start with sequential-only reuse; DAG reuse is a separate phase |
| Planner API change breaking existing tests | Medium | Keep `make_naive_plan()` unchanged; add `make_plan()` as the new entry point |
| In-place kernel aliasing (input == output buffer) | High | Mark ops as in-place-safe or not; ReLU may be in-place but Conv2d is not |

### Recommended Sequence

1. Add live range computation to `planner.py` (pure analysis, no behavior change)
2. Implement `make_linear_reuse_plan()` for sequential graphs
3. Add `validate_plan_correctness()` to `debug.py`
4. Test that reuse and naive plans produce identical outputs
5. Add `make_plan(strategy=)` entry point
6. Implement full interval-coloring reuse for DAG graphs (after RFC-02)

### Not In Scope For This RFC

- In-place ops (ops that write into their input buffer) — requires explicit in-place annotation on kernels
- Memory arena / pool allocator (C-level concern, not numpy-level)
- Gradient buffer reuse (requires backward liveness analysis)

---

## Summary Table

| RFC | Feature | Depends On | Complexity | Priority |
|---|---|---|---|---|
| RFC-01 | BatchNorm2d / GroupNorm / LayerNorm | Mode flag, buffer classification | Medium | Medium |
| RFC-02 | ResidualBlock / branching graph | DAG executor, topo sort | High | High |
| RFC-03 | Concat / Mul / elementwise ops | RFC-02 DAG executor | Medium | Medium |
| RFC-04 | Memory reuse optimization | RFC-02 (for DAG reuse) | Medium | Low |

Recommended implementation order: **RFC-02 → RFC-01 → RFC-03 → RFC-04**

RFC-02 unblocks the others because it replaces the sequential-only assumption with a proper DAG executor. Everything else builds on top of that foundation.

---

## What Must Not Change

Regardless of which RFC is implemented first, the following invariants must be preserved:

1. `make_naive_plan()` must remain unchanged and always produce a correct result
2. `validate_cuda_native_config()` must reject all RFC features until they are fully implemented and tested
3. `capabilities.py` must not claim support for any RFC feature until it passes its full test matrix
4. No RFC implementation may introduce silent fallback to torch or cuda_legacy

---

# cuda_native Phase 5 擴充 RFC（中文）

本文件描述四個高風險功能區域未來進入 `cuda_native` 的設計方向。

目前 MVP 均未實作。每份 RFC 定義問題、架構變更需求、風險分析與建議實作順序。目標是在寫程式碼之前建立清楚的設計約束，避免未來實作工作破壞現有的 forward-only MVP。

---

## RFC-01：正規化層（BatchNorm2d / GroupNorm / LayerNorm）

### 問題

正規化層是現代 CNN 最常用的模組之一。目前 `cuda_native` 在驗證時直接拒絕它們。本 RFC 定義正確支援所需的架構條件。

### 為什麼難

正規化層有**依執行模式而異的狀態行為**：

| 行為 | 訓練模式 | 推論模式 |
|---|---|---|
| 統計來源 | 從當前 batch 計算 | 使用 running mean/var |
| Running stats | 更新（momentum） | 唯讀 |
| Backward | 非平凡梯度計算 | N/A |

這表示正規化層不是輸入的純函數，需要：
- 顯式的 **mode flag**（train vs eval）透過 executor 傳遞
- 獨立的 **persistent state buffer**（running_mean, running_var）
- 能區分 activation / parameter / statistic buffer 的 planner

### 建議實作順序

1. 擴充 `Node` 加入 `trainable_state`
2. 在 planner 加入 `BufferType` enum
3. 實作 `BatchNorm2d` forward（eval 模式）並補測試
4. 實作 `BatchNorm2d` forward（train 模式）含 running stat 更新
5. 擴充 validator 確認 stat buffer 存在
6. 實作 backward（獨立子階段）
7. 用同樣模式加入 `GroupNorm`、`LayerNorm`

---

## RFC-02：ResidualBlock 與分支 Graph

### 問題

ResidualBlock 需要**分支-合併 graph 拓撲**：一個輸入分兩條路徑，輸出 elementwise 相加後繼續。現有 `NativeGraph` 是嚴格 sequential，不支援分支。

### 核心架構變更

1. `NativeGraph` 移除 sequential-only 假設，允許 node 有多個 input tensor
2. `topological_order()` 改為真正的拓撲排序（Kahn's algorithm 或 DFS）
3. Planner 加入 consumer count 追蹤，避免 buffer 提早釋放
4. Shape inference 加入 `Add` node（需驗證兩輸入 shape 相同）
5. 加入 shortcut projection 支援（1×1 Conv）

### 建議實作順序

1. 加入 `Add` kernel 與 shape inference
2. 實作真正的拓撲排序
3. Planner 加入 consumer-count 追蹤
4. 建立最小 2-branch 測試 graph（無 BN、無 projection）
5. 加入 projection shortcut 支援
6. 建立完整 ResidualBlock 測試
7. Backward 支援（獨立子階段）

---

## RFC-03：Concat 與 Elementwise Ops

### 問題

許多架構需要 `Concat`（DenseNet、U-Net skip connection）、`Mul`（gating）等張量合併操作，這些都需要 RFC-02 的 DAG executor 基礎。

### 建議實作順序

1. 完成 RFC-02（DAG executor）
2. 加入 `Concat` kernel 與 shape inference（需驗證所有輸入除 concat dim 外 shape 相同）
3. 加入 `Mul` kernel
4. 補齊 validator 規則
5. 加入 `Add` / `Concat` backward
6. 延遲 broadcasting 支援（先只做 exact-shape）

---

## RFC-04：Memory Reuse 最佳化

### 問題

現有 `make_naive_plan()` 每個 tensor 分配一個獨立 buffer，沒有任何重用。對深層網路來說，記憶體使用量正比於層數。

### 核心設計

加入 `make_plan(strategy=)` 入口點，保留 `make_naive_plan()` 不變：

```python
def make_plan(graph, strategy: str = 'naive') -> ExecutionPlan:
    if strategy == 'naive':       return make_naive_plan(graph)
    if strategy == 'linear_reuse': return make_linear_reuse_plan(graph)
```

Sequential graph 可用 odd/even pool 方式將 N+1 個 buffer 減為 3 個。DAG graph 需要 interval graph coloring（register allocation 標準演算法）。

任何 reuse 策略在合併前必須透過 `validate_plan_correctness()` 與 naive plan 輸出比對驗證。

---

## 摘要

| RFC | 功能 | 依賴 | 複雜度 | 優先度 |
|---|---|---|---|---|
| RFC-01 | BatchNorm2d / GroupNorm / LayerNorm | mode flag、buffer 分類 | 中 | 中 |
| RFC-02 | ResidualBlock / branching graph | DAG executor、拓撲排序 | 高 | 高 |
| RFC-03 | Concat / Mul / elementwise ops | RFC-02 | 中 | 中 |
| RFC-04 | Memory reuse 最佳化 | RFC-02（DAG reuse） | 中 | 低 |

建議實作順序：**RFC-02 → RFC-01 → RFC-03 → RFC-04**

RFC-02 解除其他所有 RFC 的阻擋，因為它把 sequential-only 假設換成正確的 DAG executor。
