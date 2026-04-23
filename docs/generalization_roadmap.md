# Generalization Roadmap

Canonical planning note:
[master_roadmap.md](master_roadmap.md) is the tracked source of truth for the
overall project roadmap. This file stays narrower: it explains how frontend
breadth should relate to backend-specific execution boundaries.

MiniCNN already has a reasonably general frontend compared with its narrowest
native backend.

That distinction matters. The project does not need to force every backend to
accept every config immediately. It needs a clearer separation between:

- broad frontend declaration
- backend-specific execution capability

## Current State

Today the repo already has:

- a broad torch/flex path
- a useful NumPy autograd path
- a shared-config `train-dual` entrypoint
- a validator that makes `cuda_legacy` limits explicit
- an experimental `cuda_native` backend on the public CLI surface

The narrow part is mainly `cuda_legacy`, not the entire project.

## Recommended Architecture Direction

The clean long-term shape is:

1. stable general frontend
2. explicit backend capability mapping
3. native backend generalization in a backend designed for it

In practice, that means the project should treat these as separate concerns.

### 1. Keep The Frontend Broad

The frontend should keep moving toward:

- more reusable `model.layers[]` declarations
- better dataset interfaces
- clearer optimizer/loss/scheduler configuration
- stronger component registries and extension points
- consistent CLI behavior across paths

This is the layer that should feel closer to a modern framework interface.

### 2. Keep `cuda_legacy` Honest

`cuda_legacy` should stay:

- validator-driven
- explicit about unsupported features
- optimized for the training graph it actually owns
- narrow when narrowness keeps the code comprehensible

That backend can still grow, but it should not pretend to be an infinitely
general native runtime.

### 3. Put True Native Generalization In `cuda_native`

If the goal becomes:

- arbitrary layer order
- branching graphs
- skip connections
- broader shape support
- more general memory planning and graph execution

then the right home is not "more exceptions inside `cuda_legacy`." The right
home is a backend like `cuda_native` that is built around graph execution,
planning, and backend-level capability promotion.

## Practical Priorities

### Lower-risk work

- improve capability reporting
- tighten validators
- keep docs synced with code
- broaden frontend configuration where it does not create fake native parity
- add more reference and parity tests

### Higher-risk work

- stretching `cuda_legacy` far beyond its fixed training graph
- adding broad graph semantics into a backend that was designed around staged
  CIFAR-10 training
- promoting native experiments before the capability table,
  validators, and CLI all agree

## Decision Rule

When a feature request arrives, first decide which of these it is:

- frontend broadening
- `cuda_legacy` extension
- `cuda_native` backend work

That one decision prevents a lot of architectural confusion.

## Final Direction

The project should become more general over time, but not by hiding backend
limits.

The right path is:

- broaden the shared frontend
- keep backend capability tables honest
- grow true native generality in a backend that is built for it

That gets MiniCNN closer to torch-like usability at the frontend while still
preserving the educational value of explicit backend boundaries.
