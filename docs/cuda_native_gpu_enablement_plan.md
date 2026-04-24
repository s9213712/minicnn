# cuda_native GPU Enablement Plan

Last updated: 2026-04-25

This document defines how `cuda_native` should move from a NumPy reference
backend into a real GPU execution backend.

It is intentionally separate from `cuda_native_productionization_plan.md`.

- `cuda_native_productionization_plan.md` answers:
  - how the backend becomes implementation-grade within its current reference
    execution model
- this document answers:
  - how the backend stops being CPU/NumPy execution and starts running real GPU
    kernels

See also:

- [cuda_native.md](cuda_native.md)
- [cuda_native_contract.md](cuda_native_contract.md)
- [cuda_native_productionization_plan.md](cuda_native_productionization_plan.md)
- [cuda_native_expansion_plan.md](cuda_native_expansion_plan.md)
- [backend_capabilities.md](backend_capabilities.md)
- [cuda_native_gpu_parity_matrix.md](cuda_native_gpu_parity_matrix.md)
- [cuda_native_gpu_enablement_status.md](cuda_native_gpu_enablement_status.md)

## Current State

Today, `cuda_native` is:

- a native graph/training contract backend
- a NumPy reference execution backend
- useful for:
  - graph semantics
  - validator/runtime agreement
  - artifact contracts
  - parity/tolerance work
  - planner/telemetry experimentation

Today, `cuda_native` is **not**:

- a real CUDA-kernel backend
- a GPU execution backend
- a throughput-oriented runtime

That means:

- `engine.backend=cuda_native` currently executes on CPU through NumPy
- support for `AMP`, ordered DAG, `Add`, `Concat`, `ResidualBlock`,
  `ConvNeXtBlock`, and related ops is about semantics and correctness, not GPU
  acceleration

## Why GPU Enablement Is A Separate Phase

Moving `cuda_native` to GPU is not a naming cleanup.

It requires new implementation layers:

1. Device execution model
- host tensors vs device tensors
- explicit transfers
- synchronization boundaries

2. Kernel lowering
- map graph ops to GPU kernels
- define per-op launch/runtime constraints

3. Memory/runtime policy
- device allocation
- buffer reuse on device
- workspace/scratch planning

4. Correctness re-validation
- GPU kernels must re-pass parity/tolerance gates
- existing NumPy parity is necessary but not sufficient

If these are mixed into the current productionization work without separation,
the project loses clarity:

- reference correctness bugs
- runtime contract bugs
- GPU kernel bugs

would all blur together.

## Non-Goals

This plan does not assume:

- immediate replacement of `cuda_legacy`
- full feature parity with `torch`
- fusion/compilation research before baseline GPU correctness exists
- aggressive kernel autotuning before stable kernel coverage exists
- that every existing `beta` op in reference mode must be GPU-enabled in the
  first pass

## GPU Enablement Principles

1. Preserve contract honesty
- do not claim GPU support in `cuda_native` until the runtime actually lowers
  to GPU kernels

2. Reuse the current graph/validator surface where possible
- graph semantics should stay stable
- execution backend should change underneath, not the public model surface

3. Graduate by subset, not by whole-backend optimism
- bring a small `stable_gpu_subset` up first
- widen only after parity and runtime evidence exist

4. Keep reference and GPU paths comparable
- reference NumPy path remains the oracle
- GPU path is validated against it and against PyTorch

## Phase G0: Execution Boundary Freeze

Current status:

- initial execution-mode contract slice landed
- successful `cuda_native` runs now explicitly report `execution_mode=reference_numpy` and `tensor_execution_device=cpu` in CLI, `summary.json`, and `metrics.jsonl`
- `gpu_native` is now an explicit partial-forward execution track, not a silently implied full-training mode
- initial device-runtime substrate landed:
  - `DeviceTensor`
  - `DeviceRuntime`
  - staging / allocation / synchronization accounting
  - artifact telemetry seam for future GPU-runtime wiring
- planner reservation accounting now lands in `device_runtime`
- eval forward now runs through an explicit staged helper with execution telemetry
- reserved planner buffers now back staged output allocation/release telemetry
- train/eval input staging now also prefers reserved buffers, so the pool models a real staged IO lifecycle
- capability/validation surfaces now expose machine-readable GPU readiness:
  - `execution_mode_readiness` in `cuda-native-capabilities`
  - `execution_readiness_assessment` in validation and train preambles
  - `gpu_kernel_registry_surface` in `cuda-native-capabilities`
- initial GPU dispatch seam now exists:
  - `gpu_dispatch.py`
  - per-node bootstrap dispatch plans for supported subset graphs
  - explicit unsupported-op reporting for graphs outside the bootstrap subset
  - partial plans no longer silently drop unsupported nodes; unsupported steps stay visible in the plan
  - per-step parameter-binding manifests for future kernel lowering ABI
- native device-pointer forward execution now exists for the bootstrap subset:
  - `Flatten` uses device-pointer aliasing
  - `Linear`, `ReLU`, `LeakyReLU`, `Add`, `Concat`, `MaxPool2d`, and a constrained `Conv2d` path lower to native CUDA symbols when a bound library is attached
  - `Linear + SoftmaxCE + SGD` now has a narrow native GPU training-step helper that exercises forward, loss gradient, dense backward, and optimizer update through the C ABI
  - `train-native engine.execution_mode=gpu_native` now accepts the narrow `Flatten -> Linear`, `Flatten -> Linear -> ReLU -> Linear`, `MaxPool2d -> Flatten -> Linear`, `Conv2d(valid, bias=false) -> Flatten -> Linear`, `Conv2d(valid, bias=false) -> ReLU -> Flatten -> Linear`, `Conv2d(valid, bias=false) -> MaxPool2d -> Flatten -> Linear`, `Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear`, and `Conv2d(valid, bias=false) -> ReLU -> Conv2d(valid, bias=false) -> ReLU -> MaxPool2d -> Flatten -> Linear` / `CrossEntropyLoss` / `SGD` subset
  - CUDA runtime preflight now fails before allocation when the installed driver/runtime pair is incompatible, instead of aborting inside `cudaMalloc`
  - hermetic GPU training parity matrix now exists in `cuda_native_gpu_parity_matrix.md`
  - remaining blockers are composite-block GPU training composition and real-hardware parity after CUDA driver/runtime compatibility is restored

Goal:

- freeze what "GPU-enabled cuda_native" means before implementing kernels

Deliverables:

- explicit backend mode split, for example:
  - `execution_mode: reference_numpy`
  - `execution_mode: gpu_native`
- stable artifact field indicating actual execution mode
- clear CLI/runtime reporting of:
  - selected backend
  - effective execution mode
  - whether tensors ran on CPU or GPU

Required tests:

- execution-mode contract regression tests
- artifact regression tests for mode reporting

Exit criteria:

- no successful run can silently claim GPU execution while still using NumPy

## Phase G1: Device Runtime Substrate

Goal:

- introduce the minimum GPU runtime substrate without widening supported ops yet

Deliverables:

- device tensor abstraction for `cuda_native`
- explicit host->device and device->host staging helpers
- device allocator surface
- synchronization/debug helpers
- runtime telemetry for:
  - transfers
  - allocations
  - synchronization events

Initial files likely affected:

- `src/minicnn/cuda_native/executor.py`
- `src/minicnn/cuda_native/backward.py`
- `src/minicnn/cuda_native/memory.py`
- new GPU runtime modules under `src/minicnn/cuda_native/`

Required tests:

- tensor staging tests
- allocator accounting tests
- execution-mode correctness tests

Exit criteria:

- simple graph execution can move data into a GPU runtime and back without
  changing public graph semantics

## Phase G2: Stable GPU Kernel Bootstrap

Goal:

- bring up a minimal GPU kernel subset that is worth benchmarking and relying
  on

Recommended first subset:

- `Conv2d`
- `Linear`
- `ReLU`
- `LeakyReLU`
- `Flatten`
- `MaxPool2d`
- `Add`
- `Concat`

Why this subset:

- it covers simple CNN baselines
- it avoids starting with complex composite semantics
- it is enough to validate real GPU lowering against current contracts

Deliverables:

- forward GPU kernels
- backward GPU kernels
- kernel registry/lowering table
- shape/layout preconditions kept aligned with validator output

Required tests:

- per-op forward parity vs reference NumPy and PyTorch
- per-op backward parity
- simple sequential CNN end-to-end smoke on GPU
- ordered DAG smoke with `Add`/`Concat`

Exit criteria:

- a simple `stable_gpu_subset` can train and evaluate on a real dataset through
  `cuda_native` GPU execution

## Phase G3: Device Memory And Planner Integration

Goal:

- make the current planner useful for GPU execution instead of only for
  reference telemetry

Deliverables:

- device buffer allocation policy
- real device-buffer reuse
- workspace/scratch policy for kernels
- planner summaries extended with GPU metrics:
  - device bytes reserved
  - peak device live bytes
  - host/device transfer bytes
  - workspace reuse

Required tests:

- planner/device allocation regression tests
- reuse-plan stability tests
- no-leak / bounded-allocation smoke tests

Exit criteria:

- planner metrics correspond to actual device allocation behavior, not only
  static reference estimates

## Phase G4: GPU Training Correctness

Goal:

- prove the GPU path is correct, not just executable

Required work:

- optimizer-step parity for:
  - `SGD`
  - `AdamW`
- loss parity for:
  - `CrossEntropyLoss`
  - `MSELoss`
- batch-size and grad-accum tolerance matrix
- fixed-seed reproducibility slice for GPU mode

Required tests:

- reference NumPy vs GPU parity
- PyTorch vs GPU parity
- tolerance matrix:
  - fp32
  - `grad_accum_steps`
  - small/medium batch sizes

Exit criteria:

- the first GPU subset passes deterministic regression and tolerance gates

## Phase G5: Composite And Modern GPU Coverage

Goal:

- expand GPU support from primitive CNNs into the modern `beta` subset

Next candidates:

- `BatchNorm2d`
- `GroupNorm`
- `LayerNorm`
- `LayerNorm2d`
- `GlobalAvgPool2d`
- `ResidualBlock`
- `ConvNeXtBlock`
- `DropPath`

Rules:

- composite ops should only graduate after their primitive dependencies already
  have GPU parity
- do not mark composite GPU support as stable based only on forward-pass smoke

Required tests:

- block forward/backward parity
- composite tolerance matrix
- composite real-dataset smoke

Exit criteria:

- selected composite/model slices can run on GPU with parity evidence, not only
  feature flags

## Phase G6: AMP On GPU

Goal:

- graduate `AMP` only after the GPU path exists and baseline fp32 correctness is
  already stable

Deliverables:

- actual mixed-precision GPU execution
- loss scaling on GPU
- overflow handling on GPU
- AMP-specific telemetry still preserved in artifacts

Required tests:

- fp32 vs AMP tolerance matrix on GPU
- composite AMP tolerance matrix on GPU
- fixed-seed AMP reproducibility smoke on GPU

Exit criteria:

- `AMP` is no longer the last backend-wide experimental feature

## Phase G7: Performance Hardening

Goal:

- make GPU mode worth using beyond correctness demos

Deliverables:

- kernel-level hotspot reporting
- transfer overhead reporting
- train/eval runtime comparison in GPU mode
- benchmark suite for:
  - reference NumPy
  - `cuda_native` GPU mode
  - `cuda_legacy`
  - `torch + cuda`

Success metrics:

- GPU mode materially outperforms reference NumPy mode
- bottleneck summaries are actionable
- performance regressions are detectable in CI or benchmark workflows

## Suggested First GPU Graduation Target

The first target should not be "all of cuda_native on GPU".

It should be:

### `cuda_native stable_gpu_subset`

Scope:

- execution mode reporting is frozen
- primitive CNN subset runs on GPU
- real CIFAR-10 train/val/test smoke exists
- parity and tolerance gates pass for:
  - `Conv2d`
  - `Linear`
  - `ReLU`
  - `MaxPool2d`
  - `Add`
  - `Concat`
  - `CrossEntropyLoss`
  - `SGD`
  - `AdamW`

Only after that should the project widen to:

- normalization-heavy models
- residual/composite blocks
- ConvNeXt paths
- AMP graduation

## Concrete Blockers Before `cuda_native` Can Honestly Claim GPU Execution

Today the blockers are structural, not cosmetic:

- no GPU execution mode exists
- no device tensor/runtime layer exists
- no `cuda_native` kernel-lowering path exists
- planner metrics are not yet tied to actual device memory behavior
- parity/tolerance evidence is for reference execution, not GPU execution

Until those blockers are removed, `cuda_native` should remain described as:

- a reference backend with native graph/training semantics

not:

- a true GPU-native runtime

## Recommended Immediate Next Steps

1. Add explicit execution-mode reporting
- make `reference_numpy` vs future `gpu_native` machine-readable

2. Build the smallest GPU primitive path
- `Conv2d`
- `Linear`
- `ReLU`
- `MaxPool2d`
- `Add`

3. Add GPU parity gates before widening coverage
- do not start with `ConvNeXtBlock`
- do not start with `AMP`

4. Add one real-dataset benchmark lane
- CIFAR-10
- one epoch
- compare:
  - `cuda_native` reference
  - `cuda_native` GPU mode
  - `cuda_legacy`
  - `torch + cuda`

That benchmark lane should become the baseline for future GPU enablement work.
