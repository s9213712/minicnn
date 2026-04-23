# cuda_native Smoke Matrix

Last updated: 2026-04-24

This document defines the canonical `cuda_native` smoke matrix used for
contract-level validation.

These smokes are intentionally small, hermetic, and random-data-based.

They are not benchmark runs and they are not accuracy claims.

Their job is to prove that the declared public contract still holds.

See also:

- [cuda_native.md](cuda_native.md)
- [cuda_native_productionization_plan.md](cuda_native_productionization_plan.md)

## Canonical Smoke Set

### 1. Sequential classifier smoke

Intent:

- prove the minimum training success path still works
- prove the minimum artifact contract is emitted

Structure:

- `Flatten`
- `Linear`

Guarantees:

- `summary.json` exists
- `metrics.jsonl` exists
- checkpoint exists
- summary / metrics schema fields exist

### 2. Ordered DAG + `Add` smoke

Intent:

- prove named tensor wiring + merge execution still works

Structure:

- `Identity` stem
- two branches
- `Add`
- `Flatten`
- `Linear`

Guarantees:

- ordered DAG build succeeds
- training succeeds
- artifact minimum contract still holds

### 3. Ordered DAG + `Concat` smoke

Intent:

- prove channel-join / feature-join merge execution still works

Structure:

- `Identity` stem
- two branches
- `Concat`
- `Flatten`
- `Linear`

Guarantees:

- concat shape inference still matches runtime behavior
- training succeeds
- artifact minimum contract still holds

### 4. AMP + grad accumulation smoke

Intent:

- prove the advanced training contract still works

Structure:

- simple classifier
- `train.amp=true`
- `train.grad_accum_steps=2`
- optimizer using a stateful path

Guarantees:

- AMP telemetry exists
- optimizer telemetry exists
- planner telemetry exists
- artifact minimum contract still holds

## Minimum Artifact Contract

Every canonical smoke must prove at least these:

### `summary.json`

- `schema_name`
- `schema_version`
- `artifact_kind`
- `best_model_path`
- `checkpoint_contract`
- `planner`

### `metrics.jsonl`

- at least one row exists
- last row contains:
  - `schema_name`
  - `schema_version`
  - `artifact_kind`
  - `epoch`
  - `train_loss`
  - `val_loss`
  - `val_acc`
  - `epoch_time_s`

## Compatibility Check

At least one canonical smoke must also verify:

- `summary.json` can still resolve `best_model_path`
- the referenced checkpoint file exists
- the checkpoint format matches the declared `checkpoint_contract`

## Rule

New `cuda_native` public-contract work is not complete unless it either:

1. fits inside the current smoke matrix unchanged, or
2. updates this matrix and adds the corresponding regression coverage
