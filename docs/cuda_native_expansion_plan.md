# cuda_native Expansion Plan

Last updated: 2026-04-23

This document defines the next expansion direction for `cuda_native`.

It is not a wish list. It is the working plan for turning `cuda_native` from a
prototype backend that can run some modern blocks into a broader, more honest,
more reusable native execution path.

See also:

- [backend_capabilities.md](backend_capabilities.md)
- [cuda_native.md](cuda_native.md)
- [dual_backend_guide.md](dual_backend_guide.md)
- [convnext_capability_matrix.md](convnext_capability_matrix.md)
- [capability_expansion_slices.md](capability_expansion_slices.md)

## Current Position

Today `cuda_native` already has:

- strict config validation
- ordered graph IR with named tensor wiring
- numpy reference kernels
- backward and training prototypes
- experimental support for:
  - `Add`
  - `DepthwiseConv2d`
  - `PointwiseConv2d`
  - `LayerNorm2d`
  - `GELU`
  - `Dropout`
  - `ResidualBlock`
  - `ConvNeXtBlock`

This is enough to support hermetic smoke paths for modern CNN-style blocks.

This is not yet enough to call `cuda_native` a broadly extensible native backend.

## Core Diagnosis

The main bottleneck is no longer “missing one more op”.

The main bottlenecks are:

1. graph semantics are still too narrow
2. training surface is still too prototype-oriented
3. normalization / regularization coverage is still incomplete
4. capability claims still need to be driven by reusable backend primitives, not only composite shortcuts

## Primary Goal

Expand `cuda_native` by backend capability layer, not by architecture label.

In practice, that means:

1. generic graph semantics first
2. reusable normalization / regularization next
3. broader training contract after that
4. performance work only after the capability surface is honest and stable

## Non-Goals

This plan does not target:

- broadening `cuda_legacy`
- claiming production-ready native training
- filling capability tables ahead of implementation
- introducing model-specific YAML branches just to serve one architecture
- pretending composite support is equivalent to full generic graph support

## Expansion Order

## Phase A: Generic Graph Semantics

This is the highest-priority phase.

### Why

Current composite support for `ResidualBlock` and `ConvNeXtBlock` is useful, but
it is still a stopgap. The backend cannot honestly claim broad residual / merge
support until graph composition is generic.

### Target capabilities

- multi-input nodes
- explicit merge ops
- reusable residual-add semantics
- planner / layout / memory support for non-trivial graph topology

### Minimum scope

- add a backend-level `Add` op
- teach graph building and shape inference to represent multi-input nodes
- extend layout validation for multi-input graph edges
- extend planner / memory assumptions beyond purely linear chains

### Current status

- `Add` exists as a backend op
- graph building supports explicit `inputs: [...]` and `output: ...`
- backward supports gradient fan-out / accumulation by tensor name
- planner already carries multi-input step metadata correctly
- planner now has an experimental reuse-aware strategy driven by tensor last-use analysis
- planner/debug surfaces expose liveness-oriented metrics (`peak_live_bytes`, `reuse_events`)
- remaining follow-up work should focus on deeper planner / memory reuse analysis, not basic DAG correctness

### Acceptance criteria

- a config can express at least one real residual merge without requiring a composite block
- `branching_graph` can move from `false` to `true`
- `ResidualBlock` can be represented either as a composite convenience or via generic graph lowering
- graph / planner / debug tooling still produce honest structured output

## Phase B: Reusable Normalization and Regularization

### Why

Modern CNN support should not depend on one architecture-specific block type.
It should depend on reusable primitives.

### Target capabilities

- `GroupNorm`
- generic `LayerNorm`
- `DropPath` / stochastic depth

### Acceptance criteria

- validator, kernel, backward, capability summary, CLI contract, and docs all agree
- at least one hermetic smoke path uses the new capability
- support is described as prototype or stable honestly, not promoted early

## Phase C: Training Surface Expansion

### Why

Even with better ops, the backend remains narrow if the training contract is too
small for realistic experimentation.

### Target capabilities

- `AdamW`
- `BCEWithLogitsLoss`
- `label_smoothing`
- `grad_accum_steps > 1`
- later: `AMP`

### Acceptance criteria

- config validation reflects the real boundary
- summary / diagnostics surfaces expose the expanded support
- hermetic smoke coverage exists for each newly public training feature

## Phase D: Lowering and Model-Spec Discipline

### Why

As the frontend grows, `cuda_native` must not regress into “validation says yes,
runtime says something else”.

### Target capabilities

- explicit lowering rules from resolved frontend model config into native graph
- consistent handling for `model.name=...`
- fewer frontend/runtime drift opportunities

### Acceptance criteria

- `validate_cuda_native_config()` and `build_cuda_native_graph()` operate on the same resolved model surface
- at least one named-model smoke path remains covered

## Phase E: Native Quality and Performance

### Why

Only after capability honesty and reusable graph semantics are in place does it
make sense to push performance or real native execution quality harder.

### Target capabilities

- performance baselines
- stability baselines
- replacement of selected numpy reference paths with more native execution paths

### Acceptance criteria

- no capability regression while improving execution quality
- benchmark and stability claims are documented with explicit scope

## Work Rules

Every new public `cuda_native` claim should ship with all of these in the same patch line:

1. validator support
2. runtime support
3. capability summary update
4. at least one regression test
5. at least one hermetic smoke path when practical
6. docs update

If one of those is missing, the capability should not be promoted in docs.

## Immediate Next Slice

The next concrete implementation slice should be:

1. keep `Add` / ordered-DAG semantics covered by targeted tests
2. extend planner / memory analysis from “basic last-use reuse exists” to “buffer reuse decisions are more topology-aware and cost-aware”
3. preserve composite block support while lowering more paths onto generic graph primitives
4. defer `Concat` / richer merge ops until `Add` semantics are stable

This is the cleanest path from “composite-only residual semantics” to “real
backend graph capability”.

## Stop Conditions

Pause expansion and reassess if:

1. composite shortcuts start growing faster than generic graph capability
2. capability docs require repeated special-casing to stay honest
3. training feature additions begin to outpace runtime correctness coverage
4. `cuda_legacy` work starts leaking into this roadmap

## Success Criteria

This expansion plan is succeeding if, over time:

- `cuda_native` capability growth is driven by reusable backend primitives
- docs and runtime stay aligned
- smoke paths cover every new public claim
- model support broadens without architecture-specific schema sprawl
- the backend becomes more general without becoming less honest
