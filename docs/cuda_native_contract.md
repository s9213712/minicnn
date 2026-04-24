# cuda_native Contract

`cuda_native` is the primary native-backend direction for `minicnn`, but it is
not yet globally production-ready. This document defines the current support
boundary, the meaning of support tiers, and the graduation checklist for moving
surfaces out of `experimental`.

## Support Tiers

The machine-readable source of truth lives in:

- [capabilities.py](/home/s92137/NN/minicnn/src/minicnn/cuda_native/capabilities.py)
- `minicnn cuda-native-capabilities`
- `validate-cuda-native-config`
- `summary.json`
- `metrics.jsonl`

The top-level backend still reports `summary_status = "experimental"`, but
individual surfaces are split into `stable`, `beta`, and `experimental`.

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
- runtime/reporting:
  - performance report
  - reproducibility smoke
  - tolerance matrix

## Experimental Surface

`experimental` means the feature works, but we do not yet claim a stable
behavioral boundary.

Current `experimental` surfaces include:

- training-wide features:
  - `AMP`
- composite block paths:
  - `ResidualBlock`
  - `ConvNeXtBlock`
- stochastic regularization:
  - `DropPath`

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

## Meaning of the Current Flags

In [capabilities.py](/home/s92137/NN/minicnn/src/minicnn/cuda_native/capabilities.py):

- `experimental = True`
  - the backend as a whole is still under active graduation
- `training_stable = False`
  - end-to-end training is not yet globally committed as stable
- `backward_stable = False`
  - backward coverage exists, but not yet for the full committed surface

These flags should only flip after the checklist above is satisfied for the
claimed surface, not because a feature merely exists.
