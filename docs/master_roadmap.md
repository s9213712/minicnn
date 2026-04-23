# MiniCNN Master Roadmap

Last updated: 2026-04-23

This is the canonical planning document for MiniCNN.

Use this file when you want the answer to:

- what should the project do next
- what should not be prioritized yet
- which work belongs to `torch/flex`, `cuda_native`, `autograd`, or `cuda_legacy`
- how feature growth should stay honest relative to backend support

This roadmap intentionally replaces scattered local planning notes as the
tracked source of truth.

See also:

- [optimization_progress.md](optimization_progress.md)
- [generalization_roadmap.md](generalization_roadmap.md)
- [capability_expansion_slices.md](capability_expansion_slices.md)
- [example_expansion_plan.md](example_expansion_plan.md)
- [planning_consolidation.md](planning_consolidation.md)
- [convnext_support_blueprint.md](convnext_support_blueprint.md)
- [convnext_yaml_ir_decision.md](convnext_yaml_ir_decision.md)
- [backend_capabilities.md](backend_capabilities.md)
- [cuda_native.md](cuda_native.md)
- [cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)

## Current Baseline

MiniCNN already has a coherent four-path shape:

- `torch/flex`: broad reference implementation and first landing zone for new frontend work
- `autograd`: CPU-side correctness oracle and educational path
- `cuda_legacy`: narrow maintenance-only historical CUDA backend
- `cuda_native`: experimental primary native direction with graph/planner/executor structure

The project is not blocked on "becoming a framework" first. The main remaining
work is about promoting the right features through the right backend path
without creating fake parity.

## Core Decision Rule

For any proposed change, decide which bucket it belongs to before coding:

1. frontend broadening
2. `cuda_native` capability growth
3. `cuda_legacy` maintenance
4. docs / validation / parity tightening

If that decision is unclear, the change is usually too vague to land safely.

## Priority Order

### Near-Term

#### 1. Keep The Public Surface Honest

This is the highest-priority recurring work:

- keep docs, examples, capability tables, and CLI validation aligned
- reject unsupported backend configs early
- avoid silent fallback across backend boundaries
- only promote native capabilities when CLI, tests, and docs all agree

Why first:

- it protects the repo from fake parity
- it improves onboarding immediately
- it lowers the risk of every later feature addition

#### 2. Improve The Canonical Example Path

The next highest-ROI product work is not a giant feature jump. It is a cleaner
path for users to go from:

1. install
2. smoke check
3. inspect a model/graph
4. run one reference training path
5. understand which backend to choose next

Concrete focus areas:

- tighten the example inventory
- define a beginner-safe example order
- make `README.md`, `USAGE.md`, `templates/`, and `examples/` point at the same path
- prefer a few canonical examples over many half-explained ones

#### 3. Expand Capability In Stages, Not In One Giant Wave

Future capability expansion should be split into narrow slices with explicit
validator, test, and doc boundaries.

The tracked slice breakdown now lives in
[capability_expansion_slices.md](capability_expansion_slices.md).

The preferred rollout order remains:

1. `torch/flex`
2. `autograd` when correctness coverage matters
3. `cuda_native`
4. `cuda_legacy` only when maintenance requires it

Good near-term expansion candidates:

- better component registries and extension points
- clearer optimizer/loss/scheduler configuration
- additional parity tests against reference paths
- selective frontend broadening that does not imply unsupported native parity

### Medium-Term

#### 4. Promote `cuda_native` Carefully

`cuda_native` is the main place where future native generalization should grow,
but promotion must stay conservative.

Medium-term goals:

- broaden validated op and training coverage where the value is clear
- tighten promotion criteria from prototype to documented capability
- expand parity and correctness checks around native execution
- keep experimental tooling public without overstating stability

This does **not** mean "make `cuda_native` look production-ready quickly."

#### 5. Strengthen Benchmark And Comparison Discipline

The repo should improve its ability to explain backend tradeoffs clearly.

Useful medium-term work:

- benchmark reports that compare backend behavior honestly
- more explicit parity / validation evidence for native paths
- clearer distinction between educational value and production claims

### Long-Term

#### 6. Put True Native Generalization In `cuda_native`

If the project wants:

- branching graphs
- residual connections
- concat / elementwise graph ops
- broader shape support
- richer memory planning and reuse

then that work belongs in `cuda_native`, not in an ever-growing exception list
inside `cuda_legacy`.

The Phase 5 RFCs already define the design guardrails for this direction.

#### 7. Keep `cuda_legacy` Narrow

Long-term success does not require turning `cuda_legacy` into a general backend.

The intended role is:

- stable inside a small validated boundary
- useful for low-level study
- maintained honestly
- extended only when there is a strong compatibility or maintenance reason

## Workstreams

### Workstream A: Docs / Validation / UX

Priority: highest

Includes:

- capability table sync
- validation message quality
- example ordering
- doc index clarity
- diagnostic output consistency

### Workstream B: Frontend Breadth

Priority: high

Includes:

- reusable `model.layers[]` declarations
- cleaner presets / registries / extension hooks
- dataset interface improvements
- configuration clarity across training paths

Constraint:

- do not imply unsupported native execution just because the frontend can
  declare something

### Workstream C: `cuda_native`

Priority: high, but staged

Includes:

- broader op coverage
- stronger parity tests
- promotion criteria for documented support
- eventual graph-generalization work

Constraint:

- research velocity is useful, but promotion to public capability must stay strict

### Workstream D: `cuda_legacy`

Priority: low unless maintenance requires it

Includes:

- compatibility fixes
- narrow validator/documentation updates
- selective high-value maintenance work such as explicitly scoped BatchNorm2d evaluation

Constraint:

- do not use this backend as the default home for future generalization

## Not The Priority Right Now

These are explicitly lower-value than the roadmap items above:

- transforming MiniCNN into a PyTorch replacement
- attention / transformer expansion
- multi-GPU or distributed infrastructure
- tiny-cuda-nn-style fused performance chasing
- broad `cuda_legacy` graph semantics
- shipping `cuda_native` as if it were production-ready

## Definition Of A Good Next Slice

A good next implementation slice should:

- fit one roadmap bucket clearly
- have explicit validator boundaries
- include the doc update in the same change
- include tests that prove the supported surface
- avoid expanding claims faster than real behavior

## Practical Sequence

1. keep the docs and capability boundaries honest
2. improve the canonical example path
3. stage broader capability work into smaller tracked slices
4. promote `cuda_native` carefully
5. reserve deep graph-native generalization for `cuda_native`, not `cuda_legacy`
