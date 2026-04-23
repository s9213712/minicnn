# Capability Expansion Slices

Last updated: 2026-04-23

This document turns the broad "capability expansion" idea into staged,
trackable implementation slices.

Use it when you need to answer:

- what should be implemented next
- what counts as "supported" for a new capability
- which validator, tests, and docs must change in the same patch
- which backend should own the work

See also:

- [master_roadmap.md](master_roadmap.md)
- [backend_capabilities.md](backend_capabilities.md)
- [generalization_roadmap.md](generalization_roadmap.md)
- [cuda_native_phase5_rfc.md](cuda_native_phase5_rfc.md)

## Expansion Rules

Every capability slice must answer all four questions before it is considered
done:

1. Which frontend paths can declare it?
2. Which backend paths can execute it?
3. Which validators reject unsupported combinations?
4. Which tests and docs prove the supported surface?

If one of those answers is missing, the capability is not ready to be promoted.

## Promotion Standard

A capability should only move from "experimental/prototype" to "documented
support" when the same patch set includes:

- validator coverage
- regression or parity tests
- capability-table update
- user-facing doc update

## Slice Order

### Slice A: Capability Table And Validator Tightening

Goal:

- reduce ambiguity before adding new behavior

Scope:

- tighten validator messages where support boundaries are already known
- align CLI validation, capability tables, and docs wording
- add missing regression tests for accepted vs rejected configs

Primary backend ownership:

- all backends, but especially `cuda_legacy` and `cuda_native`

Required validation work:

- explicit acceptance/rejection coverage for backend-specific config mixes
- short failure messages for unsupported layer/loss/scheduler combinations

Required tests:

- validator regression tests
- docs/config smoke tests where examples claim support

Required docs:

- `backend_capabilities.md`
- `dual_backend_guide.md`
- any example/template that references the affected feature

Why first:

- it lowers the risk of every later slice

### Slice B: Frontend Breadth Without Fake Native Parity

Goal:

- broaden the shared frontend and reference path while keeping backend claims honest

Scope:

- reusable `model.layers[]` declarations
- clearer presets / registries / extension hooks
- dataset interface improvements
- optimizer/loss/scheduler config clarity

Primary backend ownership:

- `torch/flex` first
- `autograd` only where a correctness reference adds real value

Out of scope:

- implying that `cuda_legacy` or `cuda_native` support the same declaration by default

Required validation work:

- explicit rejection where native backends cannot run the new frontend declaration

Required tests:

- flex builder / registry tests
- example or template smoke tests
- autograd tests if that path also claims support

Required docs:

- `backend_capabilities.md`
- `guide_feature_expansion.md`
- `templates/README.md`
- relevant README / USAGE entrypoints

### Slice C: Reference-Parity Expansion

Goal:

- strengthen correctness confidence before broadening native claims

Scope:

- forward/backward parity against torch or NumPy references
- training-summary and artifact schema consistency
- deterministic small-case tests for promoted behavior

Primary backend ownership:

- `autograd`
- `cuda_native`
- selectively `cuda_legacy` for maintenance work

Required validation work:

- none beyond existing config boundaries unless the slice changes supported configs

Required tests:

- parity tests
- regression tests
- artifact/summary contract tests when outputs change

Required docs:

- capability notes where promotion status changes
- benchmark/comparison notes if user-facing positioning changes

Why this slice matters:

- it is the main guardrail against promoting "works once" behavior

### Slice D: `cuda_native` Stabilization And Promotion

Goal:

- promote only the parts of `cuda_native` that are truly ready

Scope:

- broader validated op coverage where the value is clear
- tighter promotion criteria from prototype to documented support
- stronger training-path and parity evidence

Primary backend ownership:

- `cuda_native`

Near-term candidates:

- stabilize currently prototype-level BatchNorm semantics
- tighten documented support around scheduler/optimizer combinations
- improve execution/debug/parity confidence for already exposed ops

Required validation work:

- `validate-cuda-native-config` coverage for the exact supported subset
- explicit rejection of unsupported graph semantics and unsupported norms/blocks

Required tests:

- phase-style `cuda_native` tests
- parity tests against reference behavior
- CLI/validator regression tests for the promoted subset

Required docs:

- `backend_capabilities.md`
- `cuda_native.md`
- any related RFC/progress note when status changes

Not the same as:

- immediately implementing all Phase 5 RFCs

### Slice E: `cuda_native` Graph-Generalization Work

Goal:

- move from sequential experimental graphs toward broader graph-native execution

Scope:

- DAG execution
- residual connections
- concat / elementwise ops
- more general memory planning and reuse

Primary backend ownership:

- `cuda_native` only

Dependency:

- this slice should follow Slice D, not replace it

Required validation work:

- shape and topology validation for new graph structures
- failure paths for unsupported DAG or merge semantics

Required tests:

- graph build tests
- planner / liveness tests
- end-to-end graph execution tests
- parity tests where reference behavior exists

Required docs:

- `cuda_native.md`
- `cuda_native_phase5_rfc.md`
- `backend_capabilities.md`

### Slice F: `cuda_legacy` Maintenance-Only Extensions

Goal:

- allow narrowly scoped maintenance work without confusing the roadmap

Scope:

- compatibility fixes
- validator or docs tightening
- explicitly justified maintenance features such as the scoped BatchNorm2d evaluation path

Primary backend ownership:

- `cuda_legacy`

Constraint:

- no "general backend" rewrite

Required validation work:

- exact placement/shape rules for any newly allowed pattern

Required tests:

- maintenance-targeted regression tests
- parity tests for the exact promoted subset

Required docs:

- `backend_capabilities.md`
- the specific maintenance note, such as `cuda_batchnorm2d_evaluation.md`

## Suggested Execution Sequence

1. Slice A
2. Slice B
3. Slice C
4. Slice D
5. Slice E
6. Slice F only when maintenance justifies it

## What To Avoid

Avoid capability patches that:

- add frontend declarations without native rejection rules
- update code but not capability docs
- update docs but not tests
- promote `cuda_native` support from one successful run
- use `cuda_legacy` as the default destination for future generalization

