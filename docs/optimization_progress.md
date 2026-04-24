# Optimization And Backend Progress

Last updated: 2026-04-24

This file tracks the current technical direction of MiniCNN without binding the
status to a specific historic PR number or branch name.

See also:

- [../USAGE.md](../USAGE.md)
- [master_roadmap.md](master_roadmap.md)
- [backend_capabilities.md](backend_capabilities.md)
- [dual_backend_guide.md](dual_backend_guide.md)

`master_roadmap.md` is the canonical planning document. This file stays focused
on the current technical direction and progress framing.

## Current Project Shape

MiniCNN currently has four relevant paths:

- `train-flex`: broad torch-backed experimentation
- `train-dual`: shared-config routing to `torch` or `cuda_legacy`
- `train-autograd`: CPU/NumPy educational training path
- `train-native` / experimental `engine.backend=cuda_native`

The stable default backend toggle is still:

- `engine.backend=torch`
- `engine.backend=cuda_legacy`

`cuda_native` is already public and CLI-visible, but it remains experimental and
prototype-level rather than a stable training backend.

## What Is In Good Shape

- shared YAML/CLI frontend
- torch/flex model-building path
- NumPy autograd teaching path
- validator-driven `cuda_legacy` bridge
- documentation for backend boundaries and capability differences
- `cuda_native` ordered DAG graph path with named tensor wiring plus generic `Add` / `Concat`
- `cuda_native` normalization/regularization slice: `GroupNorm`, `LayerNorm`, `LayerNorm2d`, `Dropout`, `DropPath`
- `cuda_native` modern training slice: `Adam`, `AdamW`, `RMSprop`, `BCEWithLogitsLoss`, `label_smoothing`, `grad_accum_steps`, experimental AMP
- planner / AMP / optimizer-state telemetry now wired into `summary.json` and `metrics.jsonl`

## What Is Still Constrained

`cuda_legacy` remains the narrowest part of the system.

Current limits include:

- CIFAR-10-centered training boundary
- fixed validated Conv/Pool/Linear pattern
- no production-grade native training quality or performance guarantees
- no shared scheduler bridge
- no shared augmentation bridge
- no `BCEWithLogitsLoss`
- no label smoothing

These are not project-wide limitations. They are backend-specific limits.

## Current Backend Priorities

### Near-term

- keep docs and capability tables aligned with actual code
- keep frontend surfaces broader than native backends
- consolidate runtime observability around stable reporting keys
- expand tests that compare native behavior against reference paths

### Medium-term

- improve runtime efficiency of the `cuda_native` research stack
- reduce tensor/state churn in planner, executor, and optimizer paths
- promote only tested native capabilities into the public backend surface

### Long-term

- generalize native execution through a backend designed for that purpose
- avoid endlessly stretching `cuda_legacy` beyond its original design

## Practical Rule

When a feature is added, the project should answer three questions explicitly:

1. Which frontend paths can declare it?
2. Which backend paths can execute it?
3. Which docs and validators make that boundary obvious?

If those three answers do not line up, the repo accumulates fake parity.

## Related Docs

- [backend_capabilities.md](backend_capabilities.md)
- [comparison_completion_report.md](comparison_completion_report.md)
- [generalization_roadmap.md](generalization_roadmap.md)
- [dual_backend_guide.md](dual_backend_guide.md)

## Current Phase

The last major expansion phase is functionally complete:

- graph semantics: ordered DAG + `Add` / `Concat`
- normalization / regularization: `GroupNorm`, `LayerNorm`, `DropPath`
- training surface: modern optimizers, richer losses, grad accumulation, experimental AMP
- reporting: planner / AMP / optimizer telemetry in artifacts
- runtime hardening slice now includes:
  - persistent grad-buffer reuse with active/capacity telemetry
  - AMP cache telemetry and reduced refresh churn
  - reusable optimizer scratch buffers with telemetry for Adam/AdamW/RMSprop
  - epoch-level efficiency telemetry in `metrics.jsonl`
  - train/eval hotspot summaries plus train/eval diff summaries
  - bottleneck-oriented runtime summaries in `summary.json`

The next phase is narrower and more technical:

- runtime efficiency
- memory / state reuse quality
- stronger performance-oriented reporting without overstating production readiness
- initial torch parity coverage for selected sensitive ops
- machine-readable `support_tiers` / `support_tier_assessment` now expose which configs stay on `stable` surfaces and which touch `beta` / `experimental` areas
  - the same `support_tier_assessment` is now persisted into `summary.json` and `metrics.jsonl`
  - trivial `Flatten -> Linear` native training paths now stay on the `stable` tier instead of being artificially forced into `beta`
  - `ResidualBlock` is now published as `beta` instead of `experimental`, backed by train/eval parity and hermetic smoke coverage
  - `ConvNeXtBlock` is now published as `beta`, backed by forward/backward parity, `layer_scale` parity, and hermetic train-path tolerance coverage
  - named-model resolution now ignores loader placeholder `model.layers`, so `model.name=convnext_tiny` consistently expands to the intended `ConvNeXtBlock` graph
  - `DropPath` is now published as `beta`, backed by deterministic train/eval correctness checks and a dedicated train smoke path
  - `AMP` is now the only remaining explicitly-published `experimental` surface in `support_tiers`
  - `train-native` preamble JSON now includes `support_tier_assessment`, so the entrypoint tells you whether the requested config is on `stable`, `beta`, or `experimental` surfaces before training starts
  - fixed-seed AMP smoke is now reproducibility-tested, so the remaining AMP blockers are graduation/stability, not lack of basic determinism evidence
  - current parity baseline: `Add`, `Concat`, `Linear`, `Conv2d` (including grouped/depthwise), `BatchNorm2d`, `LayerNorm`, `LayerNorm2d`, `GroupNorm` forward/backward
  - `BatchNorm2d` train-mode running-stat semantics are now aligned with PyTorch (`running_var` uses unbiased batch variance)
  - composite parity now covers:
    - `ResidualBlock` eval forward/backward
    - `ResidualBlock` train forward/backward with shortcut projection
    - `ConvNeXtBlock` forward/backward
    - `ConvNeXtBlock` forward/backward with `layer_scale`
  - fixed-seed smoke reproducibility is now locked for a minimal `cuda_native` training path
  - canonical `fp32`, `AMP`, and `grad_accum` native runs now have an explicit tolerance-matrix regression gate
  - `ResidualBlock` smoke variants now also have a dedicated `fp32` vs `grad_accum` tolerance gate
  - `ConvNeXtBlock` smoke variants now also have a dedicated `fp32` vs `grad_accum` tolerance gate
  - `DropPath` now has deterministic train/eval correctness checks plus a dedicated smoke path
  - `ConvNeXtBlock` now also has an `fp32` vs `AMP` tolerance gate, so `AMP` stability is isolated from block-support maturity
  - `ResidualBlock` now also has an `fp32` vs `AMP` tolerance gate, so AMP evidence covers both current beta composite block paths

For the formal productionization path, see [cuda_native_productionization_plan.md](cuda_native_productionization_plan.md).
