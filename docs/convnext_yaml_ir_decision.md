# ConvNeXt YAML / IR Scope Decision

Last updated: 2026-04-23

This document answers one question:

Does MiniCNN need a new YAML schema or a ConvNeXt-specific IR draft in order to
support its current ConvNeXt-like slice?

The answer is:

- no new YAML schema is needed for the current slice
- no ConvNeXt-specific IR schema is needed for the current slice
- `model.layers[]` remains the correct declaration surface
- the current compiler IR may record `ConvNeXtBlock` as an opaque sequential op,
  but that does not imply native lowering support

See also:

- [convnext_capability_matrix.md](convnext_capability_matrix.md)
- [backend_capabilities.md](backend_capabilities.md)
- [custom_components.md](custom_components.md)
- [master_roadmap.md](master_roadmap.md)

## Current ConvNeXt Scope

The currently implemented ConvNeXt work is intentionally narrow:

- one built-in `ConvNeXtBlock`
- one tracked example template: `templates/cifar10/convnext_like.yaml`
- `torch/flex` only
- no claim of `autograd`, `cuda_legacy`, or `cuda_native` support

That scope matters because schema work should follow real pressure, not lead it.

## YAML Decision

MiniCNN should keep ConvNeXt declarations inside the existing shared frontend:

```yaml
model:
  layers:
    - type: Conv2d
      out_channels: 64
      kernel_size: 3
      stride: 1
      padding: 1
    - type: ConvNeXtBlock
    - type: ConvNeXtBlock
```

Why this is enough right now:

- `model.layers[]` already supports YAML-friendly constructor kwargs
- the flex builder already infers `channels` from the previous layer when omitted
- the current slice does not require branching syntax, stage macros, or nested block graphs
- adding a new top-level `model.convnext` or similar key would create a second
  architecture declaration style without enough payoff

Decision:

- keep ConvNeXt inside `model.layers[]`
- do not add a dedicated ConvNeXt YAML family yet

## IR Decision

MiniCNN also does not need a ConvNeXt-specific IR schema today.

Why:

- the current compiler tracer already records arbitrary `model.layers[]` entries
  as sequential IR nodes
- the current ConvNeXt slice is still a torch/flex frontend feature, not a
  native execution target
- a richer IR would only become justified when MiniCNN needs to preserve or
  lower ConvNeXt-specific graph semantics for another backend

Current acceptable behavior:

- `ConvNeXtBlock` may appear in IR summaries as one opaque sequential op
- this is sufficient for introspection and compile-path visibility
- this is not the same as decomposing the block into depthwise/norm/MLP/residual
  nodes

Decision:

- do not add a ConvNeXt-specific IR node family yet
- do not promise native lowering for `ConvNeXtBlock`

## What Would Justify Schema Expansion Later

Schema or IR expansion becomes reasonable only if at least one of these becomes
true:

1. MiniCNN needs more than one ConvNeXt-like architecture family and the YAML
   duplication becomes a real maintenance problem.
2. A backend other than `torch/flex` needs to understand ConvNeXt structure
   instead of treating it as an opaque module.
3. The project wants reusable stage-level macros, block stacks, or branching
   declarations that help multiple architecture families, not just ConvNeXt.
4. The compiler pipeline grows beyond simple sequential tracing and needs
   structure-preserving lowering decisions.

If that happens, the preferred order is:

1. define a backend-agnostic problem statement
2. prove the frontend value across more than one architecture family
3. design the smallest reusable schema change
4. update validators, docs, and tests in the same patch

## What Should Not Happen

MiniCNN should avoid these moves for now:

- adding `model.convnext`
- adding ConvNeXt-only stage DSL syntax
- adding native promises to `cuda_native` because a torch template exists
- expanding the IR just to mirror a single experimental frontend block

Those changes would make the project look more general than it currently is.

## Practical Rule

For now, talk about ConvNeXt this way:

- supported as a minimal experimental `torch/flex` block path
- declared through normal `model.layers[]`
- not a reason to widen the shared schema yet
- not a reason to widen native IR promises yet
