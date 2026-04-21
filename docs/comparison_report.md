# Historical Comparison Draft

This file is kept as historical context only.

It originally compared MiniCNN with:

- `llm.c`
- `tiny-cuda-nn`
- `neural-network-cuda`

The problem with the original draft was scope: it mixed together features from
three different MiniCNN layers:

- the torch/flex frontend path
- the CPU/NumPy autograd path
- the narrow `cuda_legacy` backend

That made several statements sound broader than they really were. For example,
some features described as "missing from MiniCNN" were only missing from
`cuda_legacy`, while already available in torch/flex or autograd.

## Use This Instead

For current, backend-scoped information, use:

- [comparison_completion_report.md](comparison_completion_report.md)
- [backend_capabilities.md](backend_capabilities.md)
- [generalization_roadmap.md](generalization_roadmap.md)

## Core Takeaway From The Original Draft

The original comparison still pointed at one useful truth:

MiniCNN is strongest as a small educational framework that exposes frontend and
backend boundaries directly. Its most obvious gap is not project structure. Its
main gap is broader native backend coverage beyond the narrow `cuda_legacy`
contract.

That conclusion still holds. The current roadmap question is how to generalize
the native path without pretending the fixed `cuda_legacy` training graph is
already a broadly reusable backend.
