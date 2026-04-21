# Backend Capabilities

Read MiniCNN capability by backend, not as one global checklist.

The frontend surface is broader than the narrowest backend. That is expected.

## Stable Execution Paths

| Capability | Torch/flex | CPU/NumPy autograd | CUDA legacy |
|---|---:|---:|---:|
| **Datasets** | | | |
| CIFAR-10 | ✓ | ✓ slow | ✓ |
| MNIST | ✓ | ✓ slow | ✗ |
| Random toy data | ✓ | ✓ | ✗ |
| **Layers** | | | |
| Conv2d | ✓ | ✓ | ✓ fixed 3x3, stride 1, pad 0 |
| Linear | ✓ | ✓ | ✓ |
| MaxPool2d | ✓ | ✓ | ✓ fixed 2x2 |
| AvgPool2d | ✓ | ✓ | ✗ |
| BatchNorm2d | ✓ | ✓ | ✗ training graph |
| LayerNorm | ✓ via torch module name | ✗ | ✗ |
| GroupNorm | ✓ via torch module name | ✗ | ✗ |
| ResidualBlock | ✓ built-in | ✓ same-channel path | ✗ |
| Dropout | ✓ | ✓ | ✗ |
| **Activations** | | | |
| ReLU | ✓ | ✓ | ✓ |
| LeakyReLU | ✓ | ✓ | ✓ shared slope across conv blocks |
| SiLU | ✓ | ✓ | ✗ |
| Sigmoid | ✓ | ✓ | ✗ |
| Tanh | ✓ | ✓ | ✗ |
| GELU | ✓ | ✗ | ✗ |
| **Losses** | | | |
| CrossEntropyLoss | ✓ | ✓ | ✓ |
| MSELoss | ✓ | ✓ | Experimental |
| BCEWithLogitsLoss | ✓ binary only | ✓ binary only | ✗ |
| label_smoothing | ✓ | ✓ | ✗ |
| **Optimizers** | | | |
| SGD | ✓ | ✓ | ✓ |
| Momentum SGD | ✓ | ✓ | ✓ |
| Adam | ✓ | ✓ | Experimental |
| AdamW | ✓ | ✓ | ✗ |
| RMSprop | ✓ | ✓ | ✗ |
| **Schedulers** | | | |
| None / disabled | ✓ | ✓ | ✓ |
| StepLR | ✓ | ✓ | ✗ shared-config bridge |
| CosineAnnealingLR | ✓ | ✓ | ✗ shared-config bridge |
| ReduceLROnPlateau | ✓ | ✓ | legacy internal LR reduction only |
| **Regularization / precision** | | | |
| weight_decay | ✓ | ✓ | ✓ |
| gradient clipping | ✓ | ✓ per-parameter | ✓ global + per-buffer rules |
| AMP | ✓ on CUDA | ✗ | ✗ |
| **Frontend conveniences** | | | |
| `model.layers[]` YAML | ✓ | ✓ | ✓ validated fixed pattern only |
| dotted-path custom components | ✓ | ✗ | ✗ |
| block presets (`conv_relu`, `conv_bn_relu`, `conv_bn_silu`) | ✓ | ✗ | ✗ |

## Torch/Flex

This is the broadest stable path.

Use it for:

- new model ideas
- custom Python components
- fast iteration
- most experiments that do not specifically need the handcrafted CUDA path

It also accepts torch module names beyond the small built-in registry through
the builder fallback to `torch.nn`.

## CPU/NumPy Autograd

This path is intentionally educational and CPU-only.

Use it for:

- framework learning
- deterministic tests
- small experiments without torch dependency

Limitations:

- Conv2d is much slower than torch
- no AMP
- no LayerNorm / GroupNorm
- no dotted-path custom component surface

## CUDA Legacy

This is a real training backend, but it is intentionally narrow.

The stable contract is:

- dataset type: `cifar10`
- input shape: `[3, 32, 32]`
- fixed layer pattern:
  `Conv2d -> activation -> Conv2d -> activation -> MaxPool2d -> Conv2d -> activation -> Conv2d -> activation -> MaxPool2d -> Flatten -> Linear`
- activations: `ReLU` or `LeakyReLU`
- optimizer: `SGD` or `Adam`
- losses: `CrossEntropyLoss`, `MSELoss`

Important constraints:

- shared-config scheduler parity is not there yet
- augmentation parity is not there yet
- `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `ResidualBlock`, `AvgPool2d` are not supported in the training graph
- `validate-dual-config` is the right way to check a config before trying to train

### Why some docs mention more capability

This repo contains native kernels and schema pieces that are broader than the
currently supported `cuda_legacy` training contract. For example:

- `layer_norm.cu` exists
- scheduler dataclasses exist
- `cuda_native` experiments exist on this branch

Those do not automatically mean the stable `cuda_legacy` training path supports
them. The validator and the actual training bridge are the source of truth.

## Branch-Local `cuda_native`

`cuda_native` is not a stable public backend yet, but this branch does include a
capability descriptor in `src/minicnn/cuda_native/capabilities.py`.

Current descriptor summary:

| Capability | `cuda_native` status |
|---|---|
| status | experimental |
| graph shape | sequential only |
| forward | descriptor says yes |
| backward | descriptor says not yet supported |
| training | descriptor says not yet supported |
| dynamic shapes | no |
| branching graph | no |
| supported ops | `Conv2d`, `ReLU`, `LeakyReLU`, `Flatten`, `Linear`, `MaxPool2d`, `AvgPool2d` |

Treat the modules under `src/minicnn/cuda_native/` as backend-development work
until the CLI surface and capability descriptor are promoted together.

## Reading Validation Errors

If a config runs on `engine.backend=torch` but fails on `engine.backend=cuda_legacy`,
that is usually an expected backend boundary, not a parser bug.

The correct debugging order is:

1. Check this matrix.
2. Run `minicnn validate-dual-config --config ...`.
3. Only then decide whether you need a torch-only change, a real native backend change, or a separate experimental branch.
