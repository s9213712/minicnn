# Backend Capabilities

MiniCNN has three execution paths. Feature support should be read by backend,
not as one global project-level checklist.

Flex (PyTorch) is the broadest stack; autograd (NumPy) is educational with a
real but smaller feature set; CUDA legacy is narrow, validated, and intentionally
conservative.

## Summary Matrix

| Capability | Torch/flex | CPU/NumPy autograd | CUDA legacy |
|---|---:|---:|---:|
| **Datasets** | | | |
| CIFAR-10 | ✓ | ✓ slow | ✓ |
| MNIST | ✓ | ✓ slow | ✗ |
| Random toy data | ✓ | ✓ | ✗ |
| **Layers** | | | |
| Conv2d | ✓ | ✓ | ✓ fixed 3×3, stride 1, pad 0 |
| Linear | ✓ | ✓ | ✓ |
| MaxPool2d | ✓ | ✓ | ✓ fixed 2×2 |
| AvgPool2d | ✓ | ✓ | ✗ |
| BatchNorm2d | ✓ | ✓ | Partial (via batch_norm flag on ConvStage only) |
| LayerNorm | ✓ | ✗ | Kernel exists, not in training graph |
| GroupNorm | ✓ | ✗ | ✗ |
| ResidualBlock | ✓ | ✓ same-channel | ✗ |
| Dropout | ✓ | ✓ | ✗ |
| **Activations** | | | |
| ReLU | ✓ | ✓ | ✓ |
| LeakyReLU | ✓ | ✓ | ✓ same slope |
| SiLU | ✓ | ✓ | ✗ |
| Tanh | ✓ | ✓ | ✗ |
| Sigmoid | ✓ | ✓ | ✗ |
| GELU | ✓ | ✗ | ✗ |
| **Losses** | | | |
| CrossEntropyLoss | ✓ | ✓ | ✓ |
| MSELoss | ✓ | ✓ | Experimental |
| BCEWithLogitsLoss | ✓ (binary only) | ✓ (binary only) | ✗ |
| label_smoothing | ✓ PyTorch built-in | ✓ custom impl | ✗ |
| **Optimizers** | | | |
| SGD | ✓ | ✓ | ✓ |
| Momentum SGD | ✓ | ✓ | ✓ |
| Adam | ✓ | ✓ | Experimental |
| AdamW | ✓ | ✓ | ✗ |
| RMSprop | ✓ | ✓ | ✗ |
| **Schedulers** | | | |
| None / disabled | ✓ | ✓ | ✓ |
| StepLR | ✓ | ✓ | ✗ |
| CosineAnnealingLR | ✓ | ✓ | ✗ |
| ReduceLROnPlateau | ✓ | ✓ | Partial (config key, no auto trigger) |
| **Initialization** | | | |
| kaiming_uniform | ✓ | ✓ | ✓ (fixed He) |
| kaiming_normal | ✓ | ✓ | ✗ |
| xavier_uniform | ✓ | ✓ | ✗ |
| xavier_normal | ✓ | ✓ | ✗ |
| normal | ✓ | ✓ | ✗ |
| zeros | ✓ | ✓ | ✗ |
| **Regularization** | | | |
| weight_decay | ✓ | ✓ | ✓ |
| Dropout | ✓ | ✓ | ✗ |
| Gradient clipping | ✓ | ✓ per-param norm | ✓ global + elementwise |
| **Augmentation** | | | |
| normalize | ✓ | ✓ | Partial (fixed CIFAR norm) |
| random_crop | ✓ | ✗ | ✗ |
| horizontal_flip | ✓ | ✗ | ✗ |
| **Precision** | | | |
| fp32 (default) | ✓ | ✓ | ✓ |
| fp16 / AMP | ✓ on CUDA | ✗ | ✗ |
| **Block presets** | | | |
| conv_relu | ✓ | ✗ | ✗ |
| conv_bn_relu | ✓ | ✗ | ✗ |
| conv_bn_silu | ✓ | ✗ | ✗ |
| **Misc** | | | |
| YAML model config | ✓ | ✓ | ✓ fixed pattern |
| Custom Python components | ✓ | Registry-based | ✗ |
| Gradient accumulation | ✓ | ✗ | ✗ |
| Native CUDA kernels | PyTorch-managed | ✗ | ✓ |
| cuBLAS path | PyTorch-managed | ✗ | ✓ |

## Torch/flex Backend

Use this path for the broadest component coverage and fastest iteration. It is
the right default for new model ideas, custom components, AMP experiments, and
training workflows that do not need handwritten CUDA internals.

## CPU/NumPy Autograd Backend

Use this path for framework learning, small deterministic examples, and tests
that should not depend on PyTorch. It is intentionally compact and CPU-only.
Large Conv2d workloads are slow because the implementation favors clarity over
throughput.

### Loss function semantics in `train-autograd`

**CrossEntropyLoss** — expects integer class labels (0 … N-1). Accuracy uses
`argmax` over the output logits. Full multiclass support.

**MSELoss** — expects integer class labels; the trainer converts them to one-hot
float targets before computing the loss. Dense float targets are **not** accepted
directly by the dataset pipeline. True regression or multilabel targets are not
supported.

**BCEWithLogitsLoss** — expects integer class labels strictly in `{0, 1}`. Labels
outside this set are rejected immediately with `ValueError`. The trainer converts
them to single-column float targets. Only a **single output channel**
(`out_features=1`, binary classification) is supported; accuracy uses
`logit >= 0` as the positive threshold. Multilabel BCE and any non-binary label
values are **not** accepted. If you have multi-class labels (e.g. 0–9 for
CIFAR-10), use `CrossEntropyLoss` instead.

### `compare` subcommand and autograd config contract

`minicnn compare` supports `autograd` as a backend option. When autograd is
selected, the config is loaded through the `train-autograd` YAML contract
(`dataset`, `model`, `optimizer`, `loss`, `train` keys), **not** the unified
dual-backend contract used by `train-dual` and `train-cuda`. The two schemas
overlap but are not identical:

- `train-autograd` reads `dataset.input_shape`, `model.layers[]`, `optimizer.type/lr`, and `train.epochs/batch_size`.
- `train-dual`/`train-cuda` reads a fixed CNN geometry under `model.conv_layers[]` and uses `minicnn.config.settings`.

Passing a `cuda_legacy`-only YAML to an autograd compare run (or vice-versa)
may produce a `KeyError` or silently use defaults. There is currently no
cross-schema validation at the CLI level.

### Global gradient norm clipping — host-side cost (CUDA legacy)

`GRAD_CLIP_GLOBAL > 0` triggers `cuda_global_grad_scale()` in the CUDA legacy
training path. This function reads every gradient buffer from device to host via
PCIe before computing the L2 norm. The round-trip cost scales with the number of
parameter tensors. For typical CIFAR-10 configurations (2–4 conv stages + FC)
this is a small but measurable throughput reduction. Disable global clip
(`grad_clip_global: 0`) in throughput-sensitive benchmarks.

## CUDA Legacy Backend

Use this path to exercise the handwritten CUDA CNN. It is intentionally narrow:
CIFAR-10, a fixed Conv/ReLU/Pool/Linear pattern, CrossEntropyLoss, and SGD-style
updates. The current roadmap is to make this backend more credible before
chasing production-scale features:

- CUDA Adam or AdamW
- CUDA BatchNorm2d, after the native kernel/state/workspace plan in
  `docs/cuda_batchnorm2d_evaluation.md`
- LayerNorm training integration or explicit experimental status
- PyTorch parity tests
- benchmark reports

### LayerNorm Status

`cpp/src/layer_norm.cu` contains native LayerNorm forward/backward kernels. The
math is covered by `tests/test_layer_norm.py`, which compares the kernel logic
against PyTorch through a NumPy mirror. The kernels are not part of the
`cuda_legacy` training graph yet, so users should treat native LayerNorm as a
tested kernel asset rather than a supported training-layer option.

## Reading Config Errors

If a config works with `engine.backend=torch` but fails with
`engine.backend=cuda_legacy`, first check whether the layer, loss, optimizer, or
dataset is listed as supported by CUDA legacy in the table above. This is often
an expected backend limitation, not a parser bug.
