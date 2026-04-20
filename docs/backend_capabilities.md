# Backend Capabilities

MiniCNN has three execution paths. Feature support should be read by backend,
not as one global project-level checklist.

## Summary Matrix

| Capability | Torch/flex | CPU/NumPy autograd | CUDA legacy |
|---|---:|---:|---:|
| YAML model config | Yes | Yes | Yes, fixed CNN pattern only |
| Custom Python components | Yes | Registry-based local components | No |
| CIFAR-10 | Yes | Yes, slow for large runs | Yes |
| MNIST | Yes | Yes, slow for large runs | No |
| Random toy data | Yes | Yes | No |
| Conv2d | Yes | Yes | Yes |
| Linear | Yes | Yes | Yes |
| MaxPool2d | Yes | Yes | Yes |
| BatchNorm2d | Yes | Yes | No; see `docs/cuda_batchnorm2d_evaluation.md` |
| LayerNorm | Via PyTorch custom config | No built-in layer | Native kernel exists and is covered by NumPy/PyTorch parity tests, but is not wired into training |
| ResidualBlock | Yes | Same-channel block | No |
| ReLU / LeakyReLU | Yes | ReLU built in | Yes |
| Sigmoid / Tanh | Yes | Yes | No |
| Dropout | Yes | Yes | No |
| CrossEntropyLoss | Yes | Yes | Yes |
| MSELoss | Yes | Yes | No |
| BCEWithLogitsLoss | Yes | Yes | No |
| SGD | Yes | Yes | Yes |
| Momentum SGD | Yes | Yes | Yes |
| Adam | Yes | Yes | No |
| AdamW | Yes | No | No |
| AMP / mixed precision | Yes on CUDA | No | No |
| Gradient accumulation | Yes | No | No |
| Per-parameter norm clipping | Via PyTorch config or optimizer code | Yes | No |
| Global gradient norm clipping | Via PyTorch utilities if configured | No | Yes |
| Elementwise gradient clipping | Via custom code | No | Yes |
| Native CUDA kernels | PyTorch-managed | No | Yes |
| cuBLAS path | PyTorch-managed | No | Yes |

## Torch/flex Backend

Use this path for the broadest component coverage and fastest iteration. It is
the right default for new model ideas, custom components, AMP experiments, and
training workflows that do not need handwritten CUDA internals.

## CPU/NumPy Autograd Backend

Use this path for framework learning, small deterministic examples, and tests
that should not depend on PyTorch. It is intentionally compact and CPU-only.
Large Conv2d workloads are slow because the implementation favors clarity over
throughput.

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
