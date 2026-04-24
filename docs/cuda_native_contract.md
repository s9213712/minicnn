# cuda_native Contract

`cuda_native` is the primary native-backend direction for `minicnn`, but it is
not yet globally production-ready. This document defines the current beta
support boundary, the meaning of support tiers, and the graduation checklist
that moved the backend-wide status out of `experimental`.

## Support Tiers

The machine-readable source of truth lives in:

- [capabilities.py](/home/s92137/NN/minicnn/src/minicnn/cuda_native/capabilities.py)
- `minicnn cuda-native-capabilities`
- `validate-cuda-native-config`
- `summary.json`
- `metrics.jsonl`

The top-level backend now reports `summary_status = "beta"`. Individual
surfaces are still split into `stable`, `beta`, and `experimental`.

## Stable Surface

Current `stable` surfaces are the ones we are willing to treat as committed
behavioral contracts:

- artifact contracts:
  - `summary.json`
  - `metrics.jsonl`
  - validation result shape / exit code contract
- graph semantics:
  - ordered DAG wiring
  - `Add`
  - `Concat`
- core ops:
  - `Conv2d`
  - grouped / depthwise / pointwise `Conv2d`
  - `Linear`
  - `Flatten`
  - `BatchNorm2d`
  - `LayerNorm`
  - `LayerNorm2d`
  - `GroupNorm`
  - `Identity`
- optimizers:
  - `SGD`
  - `AdamW`
- losses:
  - `CrossEntropyLoss`
  - `MSELoss`

## Beta Surface

`beta` means the surface is supported and tested, but we still expect contract
or implementation details to tighten as runtime hardening continues.

Current `beta` surfaces include:

- activations / pool ops:
  - `ReLU`
  - `LeakyReLU`
  - `Sigmoid`
  - `Tanh`
  - `SiLU`
  - `GELU`
  - `AvgPool2d`
  - `MaxPool2d`
  - `AdaptiveAvgPool2d`
  - `GlobalAvgPool2d`
- regularization:
  - `Dropout`
- optimizers:
  - `Adam`
  - `RMSprop`
- losses:
  - `BCEWithLogitsLoss`
- training-wide features:
  - `AMP`
- composite/block surfaces:
  - `ResidualBlock`
  - `ConvNeXtBlock`
  - `DropPath`
- runtime/reporting:
  - performance report
  - reproducibility smoke
  - tolerance matrix

## Experimental Surface

There is currently no public op / optimizer / loss / feature bucket left in
`support_tiers.experimental`.

The next major risk surface is not a published experimental feature bucket, but
the future transition from NumPy reference execution to a true GPU execution
backend. That future path is tracked separately in
[cuda_native_gpu_enablement_plan.md](cuda_native_gpu_enablement_plan.md).

## Graduation Checklist

Before a surface moves from `experimental` to `beta`, or from `beta` to
`stable`, the following must be true.

### For any op / feature

- validator support exists
- capability metadata is published
- smoke coverage exists when the surface is train-path visible
- regression coverage exists in `tests/`

### For `backward_stable = True`

- forward parity vs torch reference within documented tolerance
- backward parity vs torch reference within documented tolerance
- tolerance matrix coverage across:
  - fp32
  - AMP when applicable
  - `grad_accum_steps` when applicable
- no unresolved contract drift between:
  - capability summary
  - validation payload
  - training artifacts

### For `training_stable = True`

- all `backward_stable` requirements hold for the committed training surface
- fixed-seed smoke is reproducible within documented tolerances
- canonical smoke matrix passes
- artifact schema / validation / failure contracts are frozen
- runtime bottleneck reporting and memory telemetry remain present and tested

## AMP Graduation Checklist

See [cuda_native_amp_graduation_checklist.md](cuda_native_amp_graduation_checklist.md).

## Meaning of the Current Flags

In [capabilities.py](/home/s92137/NN/minicnn/src/minicnn/cuda_native/capabilities.py):

- `experimental = False`
  - the backend as a whole is no longer blocked at the top-level experimental label
- `training_stable = True`
  - end-to-end training now meets the current beta graduation gate
- `backward_stable = True`
  - backward coverage now meets the current beta graduation gate

These flags only flipped after the AMP checklist, parity/tolerance evidence,
and smoke gates were satisfied for the claimed surface.
