# cuda_native Productionization Plan

Last updated: 2026-04-24

This document defines how `cuda_native` should move from an experimental
research backend into an implementation-grade backend with a stable public
contract.

It does **not** redefine `cuda_native` as production-ready today.

It defines the work required to stop calling it "just experimental" and start
calling parts of it implementation-grade with explicit support tiers.

See also:

- [cuda_native.md](cuda_native.md)
- [cuda_native_contract.md](cuda_native_contract.md)
- [cuda_native_expansion_plan.md](cuda_native_expansion_plan.md)
- [backend_capabilities.md](backend_capabilities.md)
- [dual_backend_guide.md](dual_backend_guide.md)
- [optimization_progress.md](optimization_progress.md)

## Goal

Move `cuda_native` from:

- broad experimental capability growth

to:

- stable public contracts
- predictable runtime behavior
- measurable correctness and performance
- explicit support tiers

## Exit Criteria For "Not Just Experimental"

`cuda_native` should not be described as more than experimental until all of
these hold at the same time:

1. Capability contract is stable
- validator, runtime, and docs agree
- supported / unsupported behavior is predictable

2. Artifact contract is stable
- `summary.json`
- `metrics.jsonl`
- checkpoint payload format
- CLI error / exit-code behavior

3. Core training behavior is reproducible
- canonical smoke configs remain stable
- fixed-seed runs stay inside documented tolerances

4. Runtime behavior is observable
- planner, AMP, and optimizer telemetry are sufficient to explain churn, peak
  memory, and skipped steps

5. Support tiers are explicit
- `stable`
- `beta`
- `experimental`

Without these five, `cuda_native` remains a prototype, even if it supports many
ops.

## Non-Goals

These are not the immediate goal of productionization:

- turning `cuda_native` into a real CUDA-kernel backend right now
- matching all `torch/flex` frontend breadth
- stretching `cuda_legacy`
- claiming production training quality across all supported ops

## Phase 1: Contract Freeze

Goal:

- freeze the public contract before more breadth is added

Deliverables:

- define stable schema/versioning for:
  - `summary.json`
  - `metrics.jsonl`
  - checkpoint metadata
- freeze `validate-cuda-native-config` output structure
- freeze `train-native` error categories and exit-code behavior
- document canonical smoke configs and their intent

Required tests:

- artifact schema regression tests
- CLI contract regression tests
- backward-compatibility tests for summary/checkpoint readers

Phase 1 done means external tooling can depend on the output shape.

Current status:

- first slice started
- `summary.json` / `metrics.jsonl` now expose explicit schema identifiers
- `summary.json` now carries explicit checkpoint-contract metadata
- artifact schema regression coverage exists for the current contract
- `validate-cuda-native-config` now exposes explicit validation-result schema metadata
- `train-native` now exposes stable user-facing failure categories for contract-level regression tests
- canonical smoke matrix is now documented separately in `cuda_native_smoke_matrix.md`
- support tiers are now published in `cuda_native.md` and `backend_capabilities.md`

## Phase 2: Runtime Hardening

Goal:

- make current supported capability run more predictably and with less churn

Priority order:

1. Planner / memory reuse
- reduce activation buffer churn
- improve buffer reuse quality
- expose peak/reserved/reuse efficiency clearly

2. AMP quality
- improve scale policy behavior
- tighten overflow handling
- reduce repeated cast/cache overhead

3. Optimizer efficiency
- make state tensor updates fully in-place
- reduce optimizer-state allocation churn
- expose step-cost / state-cost telemetry

4. Execution profiling
- node-level timing
- epoch-level runtime summary
- hotspot reporting

Phase 2 is about runtime quality, not adding more ops.

Current status:

- planner/static telemetry is already exposed through `summary.json` and `metrics.jsonl`
- AMP runtime now includes loss-scaling / overflow telemetry plus cache hit/update/allocation counters
- optimizer runtime now includes state-churn, reusable scratch-buffer telemetry, and grad-buffer reuse metrics
- `performance_report.runtime` now includes:
  - epoch timing / throughput summary
  - `train_hotspots`
  - `eval_hotspots`
  - `hotspot_diff`
- `performance_report.bottlenecks` now condenses planner / AMP / grad-buffer / hotspot signals into a direct runtime summary

Remaining Phase 2 emphasis:

- parity between runtime telemetry usefulness and actual runtime cost
- further reduction of temporary tensor churn in optimizer/update hot paths
- deeper profiling only where it materially improves actionability

## Phase 3: Correctness Hardening

Goal:

- establish that `cuda_native` can be trusted within its declared boundary

Required work:

- torch parity tests for:
  - forward
  - backward
  - optimizer step
- numeric tolerance matrices for:
  - fp32
  - AMP
  - grad accumulation
  - different batch sizes
- focused gradient checks for sensitive ops:
  - `Add`
  - `Concat`
  - `GroupNorm`
  - `LayerNorm`
  - `DropPath`
  - `ConvNeXtBlock`

Without parity and numeric regression, the backend remains research-grade.

Current status:

- initial torch parity coverage now exists for selected forward/backward-sensitive ops
- current baseline covers:
  - `Add`
  - `Concat`
  - `Linear`
  - `Conv2d` (including grouped/depthwise)
  - `BatchNorm2d`
  - `LayerNorm`
  - `LayerNorm2d`
  - `GroupNorm`
- this creates a first reference-backed slice before widening to larger composite blocks
- `BatchNorm2d` training-mode running-stat semantics now match PyTorch's biased-for-output / unbiased-for-running-var split
- composite parity now covers:
  - `ResidualBlock` eval forward/backward
  - `ResidualBlock` train forward/backward with shortcut projection
  - `ConvNeXtBlock` forward/backward
  - `ConvNeXtBlock` forward/backward with `layer_scale`
- fixed-seed smoke reproducibility is now regression-tested for a minimal `cuda_native` training path
- canonical `fp32`, `AMP`, and `grad_accum` native variants are now covered by an explicit tolerance-matrix regression test
- `ResidualBlock` hermetic smoke variants are now covered by an additional `fp32` vs `grad_accum` tolerance gate
- `ConvNeXtBlock` hermetic smoke variants are now covered by an additional `fp32` vs `grad_accum` tolerance gate
- named-model resolution now prefers the registered model spec over loader placeholder `model.layers`, so `convnext_tiny` support-tier assessment and train-path artifacts reflect the real expanded graph
- `cuda-native-capabilities` and `validate-cuda-native-config` now expose machine-readable support-tier metadata instead of leaving `Stable` / `Beta` / `Experimental` only in prose
- `summary.json` and `metrics.jsonl` now persist the same support-tier assessment, so successful runs keep their tier boundary in artifact form
- `cuda-native-capabilities` now also exposes machine-readable `graduation_gates`, separating the ready `core_beta_subset` from the still-blocked full-backend graduation path

## Phase 4: Support Tiers

Goal:

- publish honest support tiers instead of one global "experimental" label

Recommended initial tiering:

### Stable candidates

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

### Beta candidates

- `GroupNorm`
- `LayerNorm`
- `LayerNorm2d`
- `BCEWithLogitsLoss`
- `RMSprop`
- AMP
- planner reuse heuristics

### Experimental candidates

- `ResidualBlock`
- `ConvNeXtBlock`
- `DropPath`
- composite lowering policies
- aggressive planner heuristics

These tiers should be reflected in:

- docs
- validator messages
- capability summaries
- release notes / change summaries

## Recommended Milestone

The next serious milestone should be:

## `cuda_native beta-0`

Exit criteria:

- public artifact schema frozen
- canonical smoke matrix stable
- planner / AMP / optimizer telemetry stable
- core parity tests in place
- support tiers published

`beta-0` does **not** mean production-ready.

It means the backend has a trustworthy public boundary.

## Phase 1 TODO

This is the immediate execution checklist.

1. Freeze artifact schemas
- add explicit schema/version fields for `summary.json`
- add explicit schema/version fields for `metrics.jsonl`
- document required and optional keys

2. Freeze CLI contracts
- lock `validate-cuda-native-config` JSON/text shape
- lock `train-native` failure categories and exit codes
- add regression coverage for unsupported-config paths

3. Freeze canonical smoke set
- define a minimal smoke matrix:
  - sequential classifier
  - ordered DAG + `Add`
  - ordered DAG + `Concat`
  - AMP + grad accumulation
- document expected artifact outputs

4. Lock checkpoint compatibility rules
- define what metadata is guaranteed
- add summary/checkpoint compatibility tests

5. Publish support tiers
- mark `stable` / `beta` / `experimental` in capability docs
- keep validator/runtime/docs consistent

## Practical Rule

For every future `cuda_native` feature, answer these questions before merging:

1. Is this widening the public contract or only improving implementation quality?
2. Which artifact/reporting surfaces change?
3. Which regression tests lock the new contract?
4. Which support tier does this belong to?

If those answers are unclear, the change is not ready.
