# Backend Capabilities

Read MiniCNN capability by backend, not as one global checklist.

The frontend surface is broader than the narrowest backend. That is expected.

## Execution Paths

| Capability | Torch/flex | CPU/NumPy autograd | CUDA legacy | cuda_native (experimental) |
|---|---:|---:|---:|---:|
| **Datasets** | | | | |
| CIFAR-10 | ✓ | ✓ slow | ✓ | ✓ |
| MNIST | ✓ | ✓ slow | ✗ | ✓ |
| Random toy data | ✓ | ✓ | ✗ | ✓ |
| **Layers** | | | | |
| Conv2d | ✓ | ✓ | ✓ fixed 3x3, stride 1, pad 0 | ✓ numpy ref |
| Linear | ✓ | ✓ | ✓ | ✓ numpy ref |
| MaxPool2d | ✓ | ✓ | ✓ fixed 2x2 | ✓ numpy ref |
| AvgPool2d | ✓ | ✓ | ✗ | ✓ numpy ref |
| BatchNorm2d | ✓ | ✓ | ✗ training graph | ✗ rejected |
| LayerNorm | ✓ via torch module name | ✗ | ✗ | ✗ rejected |
| GroupNorm | ✓ via torch module name | ✗ | ✗ | ✗ rejected |
| ResidualBlock | ✓ built-in | ✓ same-channel path | ✗ | ✗ rejected |
| Dropout | ✓ | ✓ | ✗ | ✗ |
| **Activations** | | | | |
| ReLU | ✓ | ✓ | ✓ | ✓ numpy ref |
| LeakyReLU | ✓ | ✓ | ✓ shared slope across conv blocks | ✓ numpy ref |
| SiLU | ✓ | ✓ | ✗ | ✗ |
| Sigmoid | ✓ | ✓ | ✗ | ✗ |
| Tanh | ✓ | ✓ | ✗ | ✗ |
| GELU | ✓ | ✗ | ✗ | ✗ |
| **Losses** | | | | |
| CrossEntropyLoss | ✓ | ✓ | ✓ | ✓ numpy |
| MSELoss | ✓ | ✓ | Experimental | ✓ numpy |
| BCEWithLogitsLoss | ✓ binary only | ✓ binary only | ✗ | ✗ |
| label_smoothing | ✓ | ✓ | ✗ | ✗ |
| **Optimizers** | | | | |
| SGD | ✓ | ✓ | ✓ | ✓ numpy (prototype) |
| Momentum SGD | ✓ | ✓ | ✓ | ✗ |
| Adam | ✓ | ✓ | Experimental | ✗ |
| AdamW | ✓ | ✓ | ✗ | ✗ |
| RMSprop | ✓ | ✓ | ✗ | ✗ |
| **Schedulers** | | | | |
| None / disabled | ✓ | ✓ | ✓ | ✓ |
| StepLR | ✓ | ✓ | ✗ shared-config bridge | ✗ |
| CosineAnnealingLR | ✓ | ✓ | ✗ shared-config bridge | ✗ |
| ReduceLROnPlateau | ✓ | ✓ | legacy internal LR reduction only | ✗ |
| **Regularization / precision** | | | | |
| weight_decay | ✓ | ✓ | ✓ | ✓ in SGD update |
| gradient clipping | ✓ | ✓ per-parameter | ✓ global + per-buffer rules | ✗ |
| AMP | ✓ on CUDA | ✗ | ✗ | ✗ |
| **Frontend conveniences** | | | | |
| `model.layers[]` YAML | ✓ | ✓ | ✓ validated fixed pattern only | ✓ validated sequential only |
| dotted-path custom components | ✓ | ✗ | ✗ | ✗ |
| block presets (`conv_relu`, `conv_bn_relu`, `conv_bn_silu`) | ✓ | ✗ | ✗ | ✗ |
| **Training** | | | | |
| Forward pass | ✓ | ✓ | ✓ | ✓ |
| Backward / gradients | ✓ | ✓ | ✓ | Prototype only |
| Full training loop | ✓ | ✓ | ✓ | Prototype only |
| Production-ready | ✓ | ✓ | ✓ | ✗ experimental |

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

## cuda_native (Experimental)

`cuda_native` is an experimental graph-based backend.

It is explicitly opt-in via `engine.backend=cuda_native` or the `train-native` CLI command.
It must not become the default backend and is not a replacement for `cuda_legacy`.

Current capability descriptor:

| Capability | `cuda_native` status |
|---|---|
| status | experimental |
| graph shape | sequential only |
| forward | yes (numpy reference) |
| backward | prototype — not stable |
| training | prototype — not production-ready |
| dynamic shapes | no |
| branching graph | no |
| supported ops | `Conv2d`, `ReLU`, `LeakyReLU`, `Flatten`, `Linear`, `MaxPool2d`, `AvgPool2d` |
| unsupported ops | `BatchNorm2d`, `GroupNorm`, `LayerNorm`, `ResidualBlock` |

Check current state:

```bash
minicnn cuda-native-capabilities
minicnn validate-cuda-native-config --config configs/dual_backend_cnn.yaml
```

See [docs/cuda_native.md](cuda_native.md) for the full guide.

## Reading Validation Errors

If a config runs on `engine.backend=torch` but fails on `engine.backend=cuda_legacy`,
that is usually an expected backend boundary, not a parser bug.

The correct debugging order is:

1. Check this matrix.
2. Run `minicnn validate-dual-config --config ...`.
3. Only then decide whether you need a torch-only change, a real native backend change, or a separate experimental branch.
