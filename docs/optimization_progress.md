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
  - epoch-level efficiency telemetry in `metrics.jsonl`
  - train/eval hotspot summaries plus train/eval diff summaries
  - bottleneck-oriented runtime summaries in `summary.json`

The next phase is narrower and more technical:

- runtime efficiency
- memory / state reuse quality
- stronger performance-oriented reporting without overstating production readiness

For the formal productionization path, see [cuda_native_productionization_plan.md](cuda_native_productionization_plan.md).
