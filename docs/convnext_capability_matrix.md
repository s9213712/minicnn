# ConvNeXt Capability Matrix

Last updated: 2026-04-23

This document answers one narrow question:

Can MiniCNN support ConvNeXt, and if so, on which backend path first?

The short answer is:

- `torch/flex`: one minimal built-in ConvNeXt-like block path now exists as an
  experimental frontend slice
- `autograd`: not a realistic near-term target
- `cuda_legacy`: not a target
- `cuda_native`: not a near-term target until graph and normalization support
  grow substantially

See also:

- [backend_capabilities.md](backend_capabilities.md)
- [convnext_support_blueprint.md](convnext_support_blueprint.md)
- [convnext_yaml_ir_decision.md](convnext_yaml_ir_decision.md)
- [master_roadmap.md](master_roadmap.md)
- [capability_expansion_slices.md](capability_expansion_slices.md)
- [cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)

## What A ConvNeXt-Like Path Needs

A practical ConvNeXt implementation typically needs most of these pieces:

1. depthwise convolution
2. pointwise expansion / projection
3. GELU
4. LayerNorm or an equivalent normalization choice
5. residual block semantics
6. stage transitions / downsampling
7. global pooling + classifier head
8. optional layer scale
9. optional stochastic depth / DropPath

MiniCNN should evaluate support at this level, not at the "one layer exists"
level.

## Backend Matrix

| Requirement | `torch/flex` | `autograd` | `cuda_legacy` | `cuda_native` |
|---|---|---:|---:|---:|
| Depthwise Conv2d | Partial | Unclear / not a target | No | No documented support |
| Pointwise Conv / Linear MLP head | Yes | Partial | No practical fit | Partial |
| GELU | Yes | No | No | No |
| LayerNorm | Torch fallback possible | No | No | Rejected at validation |
| Residual block semantics | Partial | Partial | No | Rejected / no branching graph |
| Global pooling classifier tail | Yes | Partial | No practical fit | Partial |
| Layer scale | Minimal built-in support | No | No | No |
| Stochastic depth / DropPath | No built-in support | No | No | No |
| Minimal ConvNeXt-like built-in path | Experimental | No | No | No |

## Interpretation

### `torch/flex`

This is the only backend that is even a reasonable first destination.

Why:

- `Conv2d` already exists
- `GELU` exists
- torch fallback makes richer layers possible in principle
- the frontend is broad enough to host a ConvNeXt-like configuration path

What now exists:

- a built-in `ConvNeXtBlock`
- optional layer-scale parameter inside that block
- a tracked `templates/cifar10/convnext_like.yaml`
- tests proving the declared torch/flex subset

Why this is still not "full ConvNeXt support":

- there is still no `DropPath` / stochastic-depth support
- there is still no staged architecture family beyond one minimal example
- there is still no autograd support
- there is still no `cuda_native` or `cuda_legacy` support

Conclusion:

- the first safe ConvNeXt work should land in `torch/flex`
- but it should be framed as **ConvNeXt-like experimental frontend work**, not
  repo-wide ConvNeXt support

### `autograd`

This is not the right first destination.

Why:

- no documented `LayerNorm`
- no `GELU`
- the path is better used as a correctness oracle than as the first home for a
  large architecture expansion

Conclusion:

- autograd should only follow later if a smaller correctness slice becomes valuable

### `cuda_legacy`

This should not be a ConvNeXt target.

Why:

- validator-enforced fixed graph
- no `LayerNorm`
- no residual / modern block generalization
- no strategic value in stretching the historical maintenance backend this far

Conclusion:

- explicit non-target

### `cuda_native`

This is also not the first destination, even though it is the long-term native direction.

Why:

- `LayerNorm` is currently rejected
- residual / branching graph support is not yet available
- graph-generalization RFC work is still ahead
- promoting an architecture-specific target now would outrun current native reality

Conclusion:

- ConvNeXt-specific native work should wait until the generic graph and normalization slices advance

## First Safe ConvNeXt Slice

The first safe slice is now implemented as:

1. define ConvNeXt as a `torch/flex`-only experimental frontend target
2. add one explicit capability note saying other backends do not support it
3. choose one minimal ConvNeXt-like block shape
4. add one example or template, not a whole architecture family
5. add tests that prove the exact declared subset

What this slice should **not** do:

- claim autograd support
- claim `cuda_native` support
- claim `cuda_legacy` support
- introduce a broad new YAML schema and new native promises in one patch

## Recommended Follow-Up Order

1. keep ConvNeXt explicitly scoped to `torch/flex`
2. avoid broad YAML / IR expansion until more than one minimal example is justified
3. only then evaluate whether any generic pieces should be promoted more broadly

## Practical Positioning

MiniCNN should currently talk about ConvNeXt this way:

- valid as a frontend-expansion exploration on `torch/flex`
- includes one built-in minimal `ConvNeXtBlock` path on `torch/flex`
- not yet a built-in repo-wide supported architecture family
- not a `cuda_legacy` target
- not a current `cuda_native` target until generic native graph capabilities improve
