# cuda_native AMP Graduation Checklist

Last updated: 2026-04-24

This checklist defines the minimum evidence required before `AMP` stops being
the backend-wide experimental blocker for `cuda_native`.

It is narrower than full production readiness.

The target outcome is:

- `AMP` graduates into the `beta` surface
- `training_stable = true`
- `backward_stable = true`
- `graduation_gates.full_backend_non_experimental.ready = true`
- top-level `summary_status` moves from `experimental` to `beta`

See also:

- [cuda_native_contract.md](cuda_native_contract.md)
- [cuda_native_productionization_plan.md](cuda_native_productionization_plan.md)
- [cuda_native.md](cuda_native.md)

## Required Checklist

### 1. Forward evidence

- AMP forward path exists for the committed training surface
- AMP forward stays within documented tolerance of the fp32 baseline
- AMP forward is compared against a standard mixed-precision reference path
  (`torch.amp.autocast` where available)

Current status:

- done

### 2. Backward evidence

- AMP backward path exists for the committed training surface
- unscaled backward gradients remain within documented tolerance vs fp32
- gradients are compared against a torch reference baseline for representative
  primitive stacks

Current status:

- done

### 3. Loss-scaling behavior

- finite-step growth policy is regression-tested
- overflow backoff policy is regression-tested
- repeated backoff never drops below the minimum allowed scale floor
- behavior is compared against `torch.amp.GradScaler` policy where available

Current status:

- done

### 4. Training smoke

- fixed-seed AMP smoke exists
- fixed-seed AMP smoke is reproducible
- AMP smoke writes the expected telemetry:
  - `loss_scale`
  - skipped/overflow counters
  - cache counters

Current status:

- done

### 5. Composite coverage

- AMP tolerance evidence exists for current beta composite block paths
- at minimum:
  - `ResidualBlock`
  - `ConvNeXtBlock`

Current status:

- done

### 6. Convergence sanity

- mixed precision does not obviously break training convergence on a tiny
  learnable problem
- AMP final loss remains within documented tolerance of fp32

Current status:

- done

## Exit Rule

`AMP` may graduate from `experimental` only when all checklist items above are
true at the same time.

Once they are true:

- move `amp` from `support_tiers.experimental.features` to `support_tiers.beta.features`
- set:
  - `training_stable = true`
  - `backward_stable = true`
  - `amp_graduated = true`
- clear `full_backend_non_experimental.remaining_blockers`
- move top-level `summary_status` from `experimental` to `beta`

## What This Does Not Mean

This checklist does **not** imply:

- `cuda_native` is production-ready
- `cuda_native` is a real GPU execution backend
- AMP is ready to be called `stable`

It only means the backend-wide AMP blocker is no longer large enough to keep
the whole backend at `experimental`.
