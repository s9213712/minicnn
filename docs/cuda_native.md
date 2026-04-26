# cuda_native Backend

`cuda_native` is MiniCNN's primary native backend direction.

It is now beta-grade and still not production-ready. The purpose of this backend
is to grow the native graph/planner/executor stack in public, while
`cuda_legacy` remains the narrow maintenance-only historical path.

Related planning docs:

- [cuda_native_productionization_plan.md](cuda_native_productionization_plan.md)
- [cuda_native_amp_graduation_checklist.md](cuda_native_amp_graduation_checklist.md)
- [cuda_native_gpu_enablement_plan.md](cuda_native_gpu_enablement_plan.md)
- [cuda_native_gpu_parity_matrix.md](cuda_native_gpu_parity_matrix.md)
- [cuda_native_gpu_enablement_status.md](cuda_native_gpu_enablement_status.md)
- [cuda_native_gpu_cifar10_runbook.md](cuda_native_gpu_cifar10_runbook.md)

## What cuda_native Is

A staged, modular backend structured in layers:

- **IR layer** (`graph.py`, `nodes.py`) — graph and tensor representation
- **Validation layer** (`validators.py`, `shapes.py`) — shape inference and legality checks
- **Planning layer** (`planner.py`) — conservative or reuse-aware buffer allocation
- **Execution layer** (`executor.py`, `kernels.py`) — numpy reference kernels, dispatch table
- **Backward layer** (`backward.py`) — gradient kernels prototype
- **Training layer** (`loss.py`, `training.py`) — loss functions and SGD training loop
- **Capability layer** (`capabilities.py`) — honest feature flags
- **Layout layer** (`layouts.py`) — layout constants, per-op layout rules, validation
- **Memory layer** (`memory.py`) — buffer allocator and pool abstraction
- **Debug layer** (`debug.py`) — graph dump, plan dump, execution trace

## What cuda_native Is Not

- Not a production-ready training backend
- Default execution is GPU-first auto mode where supported; `engine.execution_mode=reference_numpy` remains the explicit CPU reference path and `engine.execution_mode=gpu_native` is strict real-CUDA mode for supported helper subsets
- Not a full general-purpose graph backend yet (`Add`-based ordered DAG support exists, but richer merge ops are still missing)

For the staged plan to move from NumPy reference execution to real GPU
execution, see [cuda_native_gpu_enablement_plan.md](cuda_native_gpu_enablement_plan.md).

## Current Status

| Feature | Status |
|---|---|
| Graph IR | ✓ Implemented |
| Shape inference | ✓ Basic |
| Forward execution | ✓ Reference mode plus partial real-CUDA GPU lowering |
| Planner | ✓ Conservative / beta-grade |
| Reuse-aware planning | ✓ Experimental (`make_reuse_plan`, `make_plan(..., strategy="reuse")`) |
| Liveness analysis | ✓ Experimental (`analyze_live_ranges`, `analyze_live_tensor_sets`, `estimate_peak_live_bytes`) |
| Reuse cost metrics | ✓ Experimental (`reuse_events`, `release_events`, `allocation_events`, `reuse_slack_bytes`) |
| Pressure-aware reuse scoring | ✓ Experimental (`max_reuse_slack_ratio`, `pressure_reuse_threshold`) |
| MaxPool2d, AvgPool2d | ✓ Supported (numpy ref) |
| Layout validation | ✓ `validate_graph_layouts()` |
| Memory footprint / pool | ✓ `memory_footprint()`, `BufferPool` |
| Graph / plan dump | ✓ `dump_graph()`, `dump_plan()` |
| Execution trace | ✓ `TracingForwardExecutor` |
| Backward prototype | ✓ Beta-grade within validated support boundary |
| Training loop | ✓ Beta-grade within validated support boundary |
| Training in production | ✗ Not enabled |
| Dynamic graph | ✗ Not supported |
| Mixed precision | ✓ Beta AMP |
| `gpu_native` training | ⚠ Partial helper-backed subsets, including full CIFAR-10 two-Conv strict GPU training |

## Supported Ops

| Op | Forward | Backward |
|---|:---:|:---:|
| Conv2d | ✓ | Prototype |
| DepthwiseConv2d | ✓ | Prototype |
| PointwiseConv2d | ✓ | Prototype |
| ReLU | ✓ | Prototype |
| LeakyReLU | ✓ | Prototype |
| Sigmoid | ✓ | Prototype |
| Tanh | ✓ | Prototype |
| SiLU | ✓ | Prototype |
| GELU | ✓ | Prototype |
| Identity | ✓ | Prototype |
| MaxPool2d | ✓ | Prototype |
| AvgPool2d | ✓ | Prototype |
| AdaptiveAvgPool2d (`output_size=(1,1)` only) | ✓ | Prototype |
| Add | ✓ | Prototype |
| Concat | ✓ | Prototype |
| GlobalAvgPool2d | ✓ | Prototype |
| Flatten | ✓ | Prototype |
| Linear | ✓ | Prototype |
| Dropout | ✓ prototype | ✓ prototype |
| BatchNorm2d | ✓ prototype (eval + train-state update) | ✓ prototype |
| GroupNorm | ✓ prototype | ✓ prototype |
| LayerNorm | ✓ prototype | ✓ prototype |
| LayerNorm2d | ✓ prototype | ✓ prototype |
| ConvNeXtBlock | ✓ composite prototype | ✓ composite prototype |
| ResidualBlock | ✓ composite prototype | ✓ composite prototype |
| DropPath | ✓ prototype | ✓ prototype |

`BatchNorm2d` now has forward/backward prototype support. It is part of the
beta-grade training path, with explicit graduation evidence and current training/backward stability gates flipped on.

`ResidualBlock`, `ConvNeXtBlock`, and `Dropout` also run through experimental
composite / reference-kernel paths. They are validation-backed and runnable,
but remain research-quality rather than production-ready.

`Add` and `Concat` are the first generic merge ops in the backend. Together
with explicit `inputs: [...]` and `output: ...` tensor wiring in `model.layers[]`,
they enable ordered DAG execution for residual-style paths and simple channel-join
paths without requiring every merge to be hidden inside a composite block.

Validated `train-native` support boundary today:

- dataset: `random`, `cifar10`, `mnist`
- loss: `CrossEntropyLoss` (optional `label_smoothing`), `BCEWithLogitsLoss` (binary output only), `MSELoss`
- optimizer: `SGD`, `Adam`, `AdamW`, or `RMSprop`, with optional global gradient clipping
- scheduler: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, or disabled
- `train.amp=true|false` (beta mixed-precision path with loss scaling / overflow backoff and graduation evidence)
- `summary.json` now records `amp_config` and `amp_runtime` telemetry for AMP runs
- `metrics.jsonl` rows now include per-epoch AMP telemetry (`loss_scale`, skipped/overflow steps, cache hits/updates/allocations)
- `summary.json` now also records `optimizer_runtime` telemetry for optimizer state tensors
- `metrics.jsonl` rows now include per-epoch optimizer telemetry (`steps_epoch`, state tensor allocations/updates, state tensor bytes)
- optimizer telemetry now also tracks grad-buffer allocation/reuse/reset behavior for accumulation-heavy runs
- `metrics.jsonl` rows now also include static planner memory telemetry (`strategy`, `peak_live_bytes`, `reuse_events`, `reuse_slack_bytes`)
- `summary.json` also records static planner/memory telemetry under `planner`
- `summary.json` now includes `performance_report`, which bundles planner, AMP, optimizer, and training knobs in one place
- `performance_report.efficiency` now adds directly readable derived metrics such as cache-hit ratio, grad-buffer reuse ratio, grad-buffer active/capacity fractions, and planner peak-live fraction
- `performance_report.runtime` now summarizes epoch-level timing and estimated training throughput
- `performance_report.runtime` now records both `train_hotspots` and `eval_hotspots`; the legacy `hotspots` field remains as an `eval_hotspots` compatibility alias, and `hotspot_diff` summarizes train/eval timing deltas plus a lightweight bottleneck summary
- `performance_report.runtime.*hotspots` records representative traced forward-pass hotspot summaries (`top_nodes`, `top_ops`, `top_categories`) and includes per-op call counts / average time; `hotspot_diff.top_op_deltas`, `hotspot_diff.top_node_deltas`, and `hotspot_diff.top_category_deltas` highlight the largest train/eval timing differences
- `performance_report.bottlenecks` folds planner/AMP/grad-buffer/hotspot signals into a direct runtime bottleneck summary so the main issues are readable without manually inspecting every telemetry ratio
- `metrics.jsonl` epoch rows now also carry an `efficiency` block so long-running jobs can be inspected incrementally instead of relying only on final `summary.json`
- optimizer runtime telemetry now also tracks reusable scratch buffers so Adam/AdamW/RMSprop temporary-array churn is observable in both `summary.json` and `metrics.jsonl`
- `summary.json` and `metrics.jsonl` now both expose explicit `schema_name` / `schema_version`
- `summary.json` now includes `checkpoint_contract` metadata instead of silently implying the checkpoint format
- `validate-cuda-native-config` now has an explicit validation-result schema contract (`schema_name`, `schema_version`, `artifact_kind`)
- `train-native` user-facing failures now expose stable category labels such as `unsupported_config` and `missing_resource`, while keeping exit code `2`

Hermetic smoke configs:

- `templates/cifar10/convnext_explicit_cuda_native_smoke.yaml`
- `templates/cifar10/convnext_tiny_cuda_native_smoke.yaml`
- `templates/cifar10/resnet_like_cuda_native_smoke.yaml`

## Support Tiers

`cuda_native` is now beta as a whole backend, with GPU-first `gpu_native` execution for eligible subsets and `reference_numpy` retained as explicit fallback/parity infrastructure rather than the only runtime reality. Its public surface
is no longer one undifferentiated bucket.

Current tiering:

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
- `Flatten -> LayerNorm -> Linear`
- `DepthwiseConv2d -> LayerNorm2d -> PointwiseConv2d -> LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> PointwiseConv2d -> Flatten -> Linear`
- `BCEWithLogitsLoss`
- `RMSprop`
- AMP (beta)
- planner reuse heuristics

### Experimental

- `ResidualBlock`
- `ConvNeXtBlock`
- `DropPath`
- composite lowering policies
- aggressive planner heuristics

## How It Differs From cuda_legacy

| | cuda_legacy | cuda_native |
|---|---|---|
| Role | maintenance-only historical backend | primary native backend direction |
| Kernel type | Real CUDA / cuBLAS | NumPy reference |
| Graph | Fixed handcrafted pipeline | Explicit ordered graph IR |
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

The capability payload now includes machine-readable `support_tiers` and
`support_tier_counts` so `Stable` / `Beta` / `Experimental` claims are not
only documentation labels.

For the explicit graduation checklist behind those markers, see
[cuda_native_contract.md](cuda_native_contract.md).

The same payload now also includes machine-readable `graduation_gates`, so the
current state of:

- `core_beta_subset`
- `full_backend_non_experimental`

is visible without manually reading the roadmap.

For GPU enablement work, the same capability payload now also exposes
`execution_mode_readiness`, which answers:

- which execution modes are active vs partial-forward vs planned
- which ops make up the first `gpu_native` bootstrap subset
- which blockers still prevent `gpu_native` from covering the full cuda_native training surface

The same payload also exposes `gpu_kernel_registry_surface`, which is the
bootstrap kernel table for `gpu_native` with per-op forward/backward
status markers.

Validation payloads and the `train-native` preamble now also include
`execution_readiness_assessment`, so a concrete config can report:

- which execution mode was requested
- whether that mode is actually ready
- which requested ops are already inside the GPU bootstrap subset
- which requested ops still fall outside that subset

Training artifacts preserve the same distinction:

- `execution_mode` and `effective_execution_mode` report what actually ran
- `selected_execution_mode` reports what the user requested
- `execution_mode_policy` records fallback metadata such as `fallback_active`,
  `fallback_reason`, `gpu_native_lowering_ready`, and
  `gpu_native_runtime_ready`

Execution-mode guidance:

- default `engine.execution_mode=gpu_native_auto` is the broad GPU-first path:
  it selects `gpu_native` when the training lowering plan and CUDA runtime
  readiness both pass, otherwise it explicitly falls back to `reference_numpy`
- `engine.execution_mode=gpu_native` is strict GPU mode for the supported native
  helper subsets and fails when the graph/runtime is outside that boundary
- `engine.execution_mode=reference_numpy` remains available as the historical
  CPU fallback path
- `train.device=cuda` or `train.device=gpu` is accepted only with
  `gpu_native` / `gpu_native_auto`; the actual execution device is still
  reported by `effective_execution_mode` and `tensor_execution_device`
- `minicnn check-cuda-ready` probes the selected native library, required
  symbols, CUDA runtime preflight, driver/runtime mismatch, and WSL device-node
  state. On WSL, a missing `/dev/dxg` means CUDA Driver API initialization is
  blocked before `cuda_native` can run real GPU kernels.

Current `train-native engine.execution_mode=gpu_native` training subsets:

- `Flatten -> Linear`
- `Linear -> ReLU -> Linear`
- `Flatten -> Linear -> ReLU -> Linear`
- `Linear -> LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> Linear`
- `Flatten -> Linear -> LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> Linear`
- `MaxPool2d -> Flatten -> Linear`
- `AvgPool2d(kernel_size=2,stride=2,padding=0) -> Flatten -> Linear`
- `BatchNorm2d -> Flatten -> Linear`
- `GlobalAvgPool2d -> Flatten -> Linear`
- `AdaptiveAvgPool2d(output_size=1) -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> Flatten -> Linear`
- `PointwiseConv2d(bias=false) -> Flatten -> Linear`
- `PointwiseConv2d(bias=false) -> ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> MaxPool2d -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> MaxPool2d -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> MaxPool2d -> Flatten -> Linear`
- `DepthwiseConv2d(bias=false) -> ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> MaxPool2d -> Flatten -> Linear`
- `Conv2d(valid, bias=false) -> ReLU/LeakyReLU/GELU/SiLU/Sigmoid/Tanh -> Conv2d(valid, bias=false) -> same activation -> MaxPool2d -> Flatten -> Linear`

These subsets execute through native GPU helper paths for forward, loss-gradient,
covered backward kernels, and supported optimizer updates. General graph-level
GPU backward lowering is still pending.
`BatchNorm2d` is now part of the `gpu_native` train-native helper subset for
`BatchNorm2d -> Flatten -> Linear` through `bn_train_forward` and
`bn_backward`.
`GlobalAvgPool2d` and `AdaptiveAvgPool2d(output_size=1)` are also covered by
helper-backed train-native subsets through `global_avgpool2d_forward` and
`global_avgpool2d_backward`.
`AvgPool2d(kernel_size=2,stride=2,padding=0)` is also covered by a
helper-backed train-native subset through `avgpool2d_forward` and
`avgpool2d_backward`.
`Identity`, `Dropout(p=0)`, and `DropPath(p=0)` are part of the forward
dispatch/bootstrap primitive set as no-op GPU aliases. Stochastic
`Dropout/DropPath` training remains outside the GPU-first path until native mask
kernels and graph backward lowering land.
`LeakyReLU`, `GELU`, `SiLU`, `Sigmoid`, and `Tanh` are part of the forward
dispatch/bootstrap primitive set through native elementwise activation shims,
and `Linear -> activation -> Linear` train-native helper subsets, including
`LeakyReLU`, now use their native backward C ABI shims.
`PointwiseConv2d` is also part of the forward dispatch/bootstrap primitive set
through the native Conv2d im2col/GEMM lowering path, and single-stage
`PointwiseConv2d -> activation -> Flatten -> Linear` helper subsets are now
covered for `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Sigmoid`, and `Tanh`.
`DepthwiseConv2d` is part of the forward dispatch/bootstrap primitive set
through `depthwise_conv2d_forward`, and single-stage
`DepthwiseConv2d -> optional activation -> optional MaxPool2d -> Flatten -> Linear`
helper subsets are now covered for the same activation family.
`GroupNorm` is part of the forward dispatch/bootstrap primitive set through
`groupnorm_forward`; train-native helper coverage is still pending.
`LayerNorm2d` is part of the forward dispatch/bootstrap primitive set through
`layernorm2d_forward`; train-native helper coverage is still pending.

`validate-cuda-native-config` now emits a `training_lowering_plan` for
`gpu_native`. The plan decomposes each accepted helper subset into explicit
forward, loss, backward, and optimizer lowering steps so diagnostics no longer
only report the coarse helper-pattern name.
At runtime, native training helpers also expose an `execution_trace` that records
the actual emitted GPU training calls in order.

`gpu_native` loss support is currently:

- Linear subsets: `CrossEntropyLoss` with `label_smoothing`, `MSELoss`, `BCEWithLogitsLoss`
- Conv-family subsets: `CrossEntropyLoss` with `label_smoothing`

`gpu_native` optimizer support is currently:

- Linear subsets: `SGD`, `Adam`, `AdamW`, `RMSprop`
- Conv-family subsets: `SGD`
- Supported SGD helper subsets support native `optimizer.weight_decay` through
  `sgd_update_fused`.
- Supported `gpu_native` training subsets use native `optimizer.grad_clip_global`
  through `grad_l2_sumsq` plus `scale_inplace`.
- Supported `gpu_native` training subsets accept `train.grad_accum_steps >= 1`
  by accumulating microbatches into one native GPU helper step.

If `gpu_native` fails with `CUDA runtime preflight failed`, the Python runtime
reached the real CUDA library but the installed NVIDIA driver/runtime pair is
not compatible. Update the driver, rebuild against a compatible CUDA toolkit, or
select a compatible `MINICNN_CUDA_VARIANT` before treating the machine as a valid
GPU smoke environment.

Repeated-Conv real-data GPU smoke entrypoint:

```bash
PYTHONPATH=src python3 examples/cuda_native_gpu_two_conv_training_cifar10_demo.py --batch-size 2
```

Current evidence: representative real CUDA smoke passes for minimal Linear SGD,
minimal Linear RMSprop, and the CIFAR-10 repeated-Conv helper; the repeated-Conv
smoke uses `official:cifar10:test_batch` when available and compares updated
weights against NumPy reference.

Full CIFAR-10 strict GPU training runbook:

```bash
PYTHONPATH=src python3 -m minicnn.cli validate-cuda-native-config \
  --config configs/cifar10_cuda_native_gpu_stronger.yaml

PYTHONPATH=src timeout 7200s python3 -m minicnn.cli train-native \
  --config configs/cifar10_cuda_native_gpu_stronger.yaml
```

Representative real-data result: the stronger two-Conv helper model reached
low-to-mid 60% validation accuracy in early epochs while train accuracy kept
rising, confirming real GPU training/eval execution and showing that the next
accuracy bottleneck is model capacity / regularization. See
[cuda_native_gpu_cifar10_runbook.md](cuda_native_gpu_cifar10_runbook.md).

Validate a config:

```bash
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml \
  optimizer.momentum=0.9 optimizer.grad_clip_global=1.0 \
  scheduler.enabled=true scheduler.type=StepLR scheduler.step_size=5
```

Create a partial native-forward GPU executor from Python:

```python
import numpy as np

from minicnn.cuda_native import build_cuda_native_graph, make_native_gpu_forward_executor

graph = build_cuda_native_graph(
    {
        "layers": [
            {"type": "Flatten", "output": "flat"},
            {"type": "Add", "inputs": ["flat", "flat"], "output": "sum"},
        ],
    },
    (1, 4),
)

executor = make_native_gpu_forward_executor(reserve_bytes=4096, reserve_buffers=4)
result = executor.run(graph, np.asarray([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32))
print(result.output)
```

Run one narrow native GPU training step for `Linear + SoftmaxCE + SGD`:

```python
import numpy as np

from minicnn.cuda_native import native_gpu_linear_training_step

step = native_gpu_linear_training_step(
    np.asarray([[1.0, 2.0, -1.0]], dtype=np.float32),
    np.asarray([2], dtype=np.int32),
    np.asarray([[0.1, 0.0, -0.1], [0.0, 0.2, 0.1], [0.3, -0.2, 0.0]], dtype=np.float32),
    np.zeros((3,), dtype=np.float32),
    lr=0.01,
)
print(step.loss_mean, step.runtime_summary["execution_kinds"])
```

The same GPU training substrate also supports the narrow
`Linear -> ReLU -> Linear` path through `native_gpu_two_linear_relu_training_step`
and through `train-native engine.execution_mode=gpu_native` when the model graph
is `Flatten -> Linear -> ReLU -> Linear`.
MaxPool backward is also covered for the narrow
`MaxPool2d -> Flatten -> Linear` graph through `native_gpu_pool_linear_training_step`
and the same `train-native` execution mode.

Validation payloads now also include `support_tier_assessment`, so a config can
be accepted while still being marked as touching `beta`
surfaces such as `amp` or composite blocks.

The same `support_tier_assessment` is now persisted into `summary.json` and
`metrics.jsonl`, so artifact consumers can tell whether a successful run stayed
on `stable` surfaces or crossed into `beta` ones.

`train-native` now also emits the same `support_tier_assessment` in its initial
JSON preamble, so the requested config's tier is visible before training starts.

Run (beta-grade NumPy-reference backend, still research-oriented):

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
from minicnn.cuda_native.planner import (
    make_naive_plan,
    make_reuse_plan,
    analyze_live_ranges,
    analyze_live_tensor_sets,
    estimate_peak_live_bytes,
)
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

# Or build a beta-grade topology-aware reuse plan
reuse_plan = make_reuse_plan(graph)
print(dump_plan(reuse_plan))

# Inspect logical liveness
print(analyze_live_ranges(graph))
print(analyze_live_tensor_sets(graph))
print(estimate_peak_live_bytes(graph))

# Reuse plans expose allocator/release behavior
print(reuse_plan.summary())

# Reuse policy can trade off slack waste against allocation pressure
reuse_plan = make_reuse_plan(
    graph,
    max_reuse_slack_ratio=2.0,
    pressure_reuse_threshold=0.9,
)

# Validate layout correctness
errors = validate_graph_layouts(graph)
assert errors == []

# Explicit merge wiring is supported
graph = build_graph([
    {'type': 'Identity', 'output': 'stem'},
    {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
    {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
    {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'sum'},
], input_shape=(8, 16, 32, 32))

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
  └─ graph.py          (NativeGraph, build_graph, ordered DAG tensor wiring)
  └─ layouts.py        (NCHW/NC layout constants, OP_LAYOUT_RULES, validate_graph_layouts)
  └─ planner.py        (BufferPlan, ExecutionPlan, liveness analysis, make_naive_plan, make_reuse_plan, make_plan)
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

Public executor contract:

- `ForwardExecutor` is stateless
- use `run(graph, feeds, params=None, mode='eval')`
- or the wrappers `run_inference(graph, x, ...)` / `run_with_cache(graph, feeds, ...)`
- do not instantiate it with a graph object

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
| Phase G1 | Partial cuda_native native-forward GPU execution | In progress |
| Phase 6 | Autograd, optimizer stack, broader op coverage | Future |

AMP graduation checklist: [docs/cuda_native_amp_graduation_checklist.md](cuda_native_amp_graduation_checklist.md)

Phase 5 RFCs: [docs/cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)

---

# cuda_native Backend（中文）

`cuda_native` 是 MiniCNN 目前主要的 native backend 方向，現況已提升到 beta。

它目前仍屬實驗性，也**不適合**正式環境使用。這條 backend 的目的是把
native graph/planner/executor 能力公開地逐步長出來；`cuda_legacy`
則維持為窄邊界的歷史維護路徑。

目前它已是 beta-grade，但仍不是 production-ready，也還不是實際 GPU kernel runtime。

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
| DepthwiseConv2d | ✓ | Prototype |
| PointwiseConv2d | ✓ | Prototype |
| ReLU | ✓ | Prototype |
| LeakyReLU | ✓ | Prototype |
| Sigmoid | ✓ | Prototype |
| Tanh | ✓ | Prototype |
| SiLU | ✓ | Prototype |
| GELU | ✓ | Prototype |
| Identity | ✓ | Prototype |
| MaxPool2d | ✓ | Prototype |
| AvgPool2d | ✓ | Prototype |
| AdaptiveAvgPool2d（僅 `output_size=(1,1)`） | ✓ | Prototype |
| GlobalAvgPool2d | ✓ | Prototype |
| Flatten | ✓ | Prototype |
| Linear | ✓ | Prototype |
| Dropout | ✓ prototype | ✓ prototype |
| BatchNorm2d | ✓ prototype（eval + train 狀態更新） | ✓ prototype |
| LayerNorm | ✓ prototype | ✓ prototype |
| LayerNorm2d | ✓ prototype | ✓ prototype |
| ConvNeXtBlock | ✓ composite prototype | ✓ composite prototype |
| GroupNorm | ✓ prototype | ✓ prototype |
| ResidualBlock | ✓ composite prototype | ✓ composite prototype |
| DropPath | ✓ prototype | ✓ prototype |

`BatchNorm2d` 現在已有 forward/backward prototype，已可進入實驗性訓練路徑，
但整體仍屬 prototype 層級，不能視為穩定支援。

`ResidualBlock`、`ConvNeXtBlock`、`Dropout` 也已透過實驗性 composite /
reference-kernel 路徑接通，可驗證、可執行，但仍不是正式穩定能力。

目前通過驗證的 `train-native` 支援範圍：

- dataset：`random`、`cifar10`、`mnist`
- loss：`CrossEntropyLoss`（可搭配 `label_smoothing`）、`BCEWithLogitsLoss`（僅 binary output）、`MSELoss`
- optimizer：支援 `SGD`、`Adam`、`AdamW`、`RMSprop`，可選 global gradient clipping
- scheduler：支援 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`，也可停用
- `train.amp=true|false`（帶 loss scaling / overflow backoff 的實驗性 mixed-precision prototype）、`train.grad_accum_steps>=1`
- `summary.json` 會額外記錄 `amp_config` 與 `amp_runtime` telemetry
- `metrics.jsonl` 每個 epoch row 也會記錄 AMP telemetry（`loss_scale`、skip/overflow、cache hit/update/allocation）
- `summary.json` 也會額外記錄 `optimizer_runtime` telemetry（optimizer state tensors）
- `metrics.jsonl` 每個 epoch row 也會記錄 optimizer telemetry（`steps_epoch`、state tensor allocation/update、state tensor bytes）
- optimizer telemetry 現在也會追蹤 grad buffer 的 allocation / reuse / reset 行為，方便觀察 accumulation-heavy 路徑
- `metrics.jsonl` 每個 epoch row 也會額外帶 planner/memory telemetry（`strategy`、`peak_live_bytes`、`reuse_events`、`reuse_slack_bytes`）
- `summary.json` 也會額外記錄靜態 planner/memory telemetry（`planner`）
- `summary.json` 也包含 `performance_report`，把 planner / AMP / optimizer / training knobs 集中整理
- `performance_report.efficiency` 也會提供較直接可讀的衍生效率指標，例如 cache-hit ratio、grad-buffer reuse ratio、grad-buffer 的 active/capacity fraction、planner peak-live fraction
- `performance_report.runtime` 也會整理 epoch-level timing 與估算的 training throughput
- `performance_report.runtime` 現在會同時記錄 `train_hotspots` 與 `eval_hotspots`；舊的 `hotspots` 欄位暫時保留為 `eval_hotspots` 的相容 alias，另外 `hotspot_diff` 會整理 train/eval 的 timing delta 摘要與輕量 bottleneck summary
- `performance_report.runtime.*hotspots` 也會記錄代表性 traced forward pass 的 hotspot 摘要（`top_nodes`、`top_ops`、`top_categories`），並提供 per-op call count / average time；`hotspot_diff.top_op_deltas`、`hotspot_diff.top_node_deltas`、`hotspot_diff.top_category_deltas` 會標出 train/eval 差異最大的 op、node、category
- `performance_report.bottlenecks` 會把 planner/AMP/grad-buffer/hotspot 訊號收斂成可直接閱讀的 runtime bottleneck 摘要，避免每次都要人工比對所有 telemetry ratio
- `metrics.jsonl` 的 epoch row 現在也會帶 `efficiency` 區塊，讓長時間訓練可以逐 epoch 觀察效率，而不是只能看最後的 `summary.json`
- optimizer runtime telemetry 現在也會追蹤可重用的 scratch buffers，讓 Adam/AdamW/RMSprop 的 temporary-array churn 可以在 `summary.json` 與 `metrics.jsonl` 中被直接觀察
- `summary.json` 與 `metrics.jsonl` 現在都會帶明確的 `schema_name` / `schema_version`
- `summary.json` 也會帶 `checkpoint_contract` metadata，而不是把 checkpoint 格式隱含在實作裡
- `validate-cuda-native-config` 也已有明確的 validation-result schema contract（`schema_name`、`schema_version`、`artifact_kind`）
- `train-native` 的 user-facing failure 也已有穩定 category label，例如 `unsupported_config`、`missing_resource`，exit code 維持 `2`

Hermetic smoke config：

- `templates/cifar10/convnext_explicit_cuda_native_smoke.yaml`
- `templates/cifar10/convnext_tiny_cuda_native_smoke.yaml`
- `templates/cifar10/resnet_like_cuda_native_smoke.yaml`

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
