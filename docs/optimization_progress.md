# Optimization And Backend Progress

Last updated: 2026-04-21

This file tracks the current technical direction of MiniCNN without binding the
status to a specific historic PR number or branch name.

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

## What Is Still Constrained

`cuda_legacy` remains the narrowest part of the system.

Current limits include:

- CIFAR-10-centered training contract
- fixed validated Conv/Pool/Linear pattern
- no `BatchNorm2d`, `LayerNorm`, `GroupNorm`, or `ResidualBlock`
- no shared scheduler bridge
- no shared augmentation bridge
- no `BCEWithLogitsLoss`
- no label smoothing

These are not project-wide limitations. They are backend-specific limits.

## Current Backend Priorities

### Near-term

- keep docs and capability tables aligned with actual code
- keep frontend surfaces broader than native backends
- expand tests that compare native behavior against reference paths

### Medium-term

- broaden native backend coverage where the value is clear
- avoid silent config fallbacks
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
