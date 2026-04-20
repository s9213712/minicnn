# MiniCNN Comparison Completion Report

This report reviews MiniCNN against:

- [karpathy/llm.c](https://github.com/karpathy/llm.c)
- [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [BobMcDear/neural-network-cuda](https://github.com/BobMcDear/neural-network-cuda)

It builds on `docs/comparison_report.md`, but corrects the main ambiguity in that
report: MiniCNN has three different capability layers, and missing features must
be scoped to the correct layer.

## Executive Summary

MiniCNN is not mainly missing framework structure. It is missing broader and more
general handwritten CUDA backend coverage.

MiniCNN already has a stronger user-facing framework than
`neural-network-cuda`: YAML configs, CLI training, tests, docs, a Torch backend,
a CPU/NumPy autograd stack, and a CUDA legacy backend. Compared with `llm.c` and
`tiny-cuda-nn`, the gap is not that MiniCNN lacks a project shape. The gap is
that those projects specialize in production-scale or performance-first CUDA
systems, while MiniCNN is still strongest as an educational dual-backend CNN
framework.

The most valuable next work is to make `cuda_legacy` a more credible backend:
CUDA Adam/AdamW, CUDA BatchNorm2d, LayerNorm integration, parity tests, global
gradient norm clipping, and real throughput reports.

## Capability Layers

| Layer | Current status |
|---|---|
| Torch/flex backend | Supports configurable PyTorch modules, BatchNorm2d, Dropout, ResidualBlock, GlobalAvgPool2d, MSELoss, BCEWithLogitsLoss, CrossEntropyLoss, SGD, Adam, AdamW, AMP, schedulers, custom components. |
| CPU/NumPy autograd | Supports Tensor backward, Function API, Conv2d, MaxPool2d, BatchNorm2d, ResidualBlock, Sigmoid, Tanh, Dropout, CrossEntropy, SGD, Adam, weight decay, and gradient clipping. |
| CUDA legacy backend | Supports a fixed CIFAR-10 CNN pattern with Conv2d, LeakyReLU/ReLU, MaxPool2d, Flatten, Linear, CrossEntropy, SGD/momentum, weight decay, elementwise clipping, cuBLAS and handmade variants, plus a LayerNorm kernel that is not wired into the main training path. |

This split matters. Some items listed as "missing" in the original comparison
report are not missing from MiniCNN overall; they are missing specifically from
`cuda_legacy`.

## Comparison With neural-network-cuda

`neural-network-cuda` is a from-scratch CUDA/C++ neural network project intended
as a CUDA introduction. Its README describes CPU and GPU versions with similar
syntax and core pieces such as Linear, ReLU, MSE, Sequential, and simple
gradient-descent training.

MiniCNN already exceeds it in most practical dimensions:

| Area | neural-network-cuda | MiniCNN |
|---|---|---|
| CNN support | Not the main focus | Conv/Pool CUDA path and autograd Conv/Pool |
| User interface | C++/CUDA examples | CLI, YAML configs, templates, docs |
| Backend choice | CPU and GPU C++ variants | Torch, CPU/NumPy autograd, CUDA legacy |
| Testing | Basic project tests | Broad Python and CUDA smoke tests |
| Extensibility | Manual C++ composition | Config-driven model building and custom components |

What MiniCNN can still borrow from it is simplicity: a short, beginner-friendly
"CUDA from scratch" path that avoids making new users understand the whole
framework at once.

## Comparison With llm.c

`llm.c` is a pure C/CUDA LLM training project focused on GPT-2/GPT-3 style
pretraining, with PyTorch reference code, CPU and CUDA implementations,
mixed-precision testing, cuDNN Flash Attention, and multi-GPU/multi-node paths
through MPI/NCCL.

MiniCNN should not chase the full `llm.c` scope. The useful lessons are:

| Gap | Recommendation | Difficulty | Reason |
|---|---|---:|---|
| CUDA Adam/AdamW | Add | Medium | Directly improves `cuda_legacy` training usability. |
| LayerNorm training integration | Add or clearly mark experimental | Low-medium | Kernel exists; user path is missing. |
| PyTorch parity tests | Add | Medium | Critical for handwritten CUDA correctness. |
| Global gradient norm clipping | Add | Medium | More standard than elementwise clipping. |
| FP16/BF16 CUDA path | Defer | High | Requires dtype changes, loss scaling, and numerical validation. |
| Embedding/Transformer/Attention | Do not prioritize | High to very high | Pulls MiniCNN away from its CNN/framework teaching focus. |
| Multi-GPU or multi-node training | Do not prioritize now | Very high | Infrastructure-heavy and low value before single-GPU CUDA is broader. |

The best `llm.c` idea for MiniCNN is not GPT training. It is rigorous reference
testing: save debug states, compare forward/backward/update results against
PyTorch, and make CUDA regressions easy to catch.

## Comparison With tiny-cuda-nn

`tiny-cuda-nn` is a performance-first C++/CUDA framework. Its main strengths are
fully fused MLPs, multiresolution hash/grid encodings, a PyTorch extension,
JIT fusion, FP16, CUTLASS MLPs, and many losses/optimizers. It is a neural
graphics and small-MLP acceleration library, not a CNN teaching framework.

MiniCNN should borrow its component clarity and benchmarking discipline, not its
entire technical scope.

| Gap | Recommendation | Difficulty | Reason |
|---|---|---:|---|
| CUDA Adam | Add | Medium | Common optimizer expected by users. |
| More CUDA losses | Add selectively | Low-medium | MSE/BCE/L1 would broaden examples. |
| Benchmark reports | Add | Low-medium | Helps users understand backend tradeoffs. |
| FP16 CUDA path | Defer | High | Valuable but cross-cutting. |
| Fully fused MLP | Do not prioritize | Very high | Not aligned with MiniCNN's CNN/backend teaching focus. |
| HashGrid/positional encoding | Do not prioritize | High | Mostly relevant to NeRF/neural graphics. |
| JIT fusion | Do not prioritize | Very high | Major runtime compiler project. |
| CUTLASS/Tensor Core work | Do not prioritize now | Very high | cuBLAS already provides a strong baseline. |

## Corrections To The Existing Comparison Report

1. "MSE/BCE loss missing" should be scoped. Torch/flex already has MSELoss and
   BCEWithLogitsLoss. CPU/NumPy autograd and CUDA legacy are the missing layers.
2. "BatchNorm missing" should be scoped. Torch/flex and CPU/NumPy autograd
   already have BatchNorm2d. CUDA legacy is missing the native BatchNorm path.
3. "Sigmoid/Tanh from neural-network-cuda" is not a precise comparison point.
   That repo's README clearly lists Linear, ReLU, MSE, Sequential, and training
   utilities as core pieces.
4. The LayerNorm note is valid: MiniCNN has a native LayerNorm kernel, but it is
   not part of the main CUDA legacy training path.
5. Framework completeness should be phrased carefully. MiniCNN is more complete
   as a configurable educational framework, but `llm.c` is far more complete as
   an LLM training system and `tiny-cuda-nn` is far more complete as a
   performance-first CUDA MLP/encoding system.

## Recommended Work Plan

| Priority | Work item | User value | Difficulty |
|---:|---|---|---:|
| 1 | Add CUDA Adam/AdamW or at least CUDA Adam | Users can train CUDA backend with a modern optimizer. | Medium |
| 2 | Add CUDA BatchNorm2d forward/backward and YAML/runtime support | Makes VGG/ResNet-like CUDA configs realistic. | Medium |
| 3 | Wire LayerNorm into a documented CUDA path or mark it experimental | Removes an orphan native capability. | Low-medium |
| 4 | Add MSE/BCE to autograd and selected CUDA loss support | Enables regression and binary classification examples. | Low-medium |
| 5 | Add PyTorch parity tests for CUDA kernels | Raises confidence in handwritten CUDA. | Medium |
| 6 | Add benchmark reports for throughput, epoch time, and GPU memory | Gives users clear backend tradeoffs. | Low-medium |
| 7 | Add global gradient norm clipping | More standard optimizer behavior. | Medium |
| 8 | Add CUDA skip/residual add path | Unlocks more natural ResNet-like CUDA training. | Medium |
| 9 | Add CUDA Sigmoid/Tanh/Dropout | Improves completeness, but lower priority for CNN path. | Low-medium |
| 10 | Explore CUDA FP16 only after the above is stable | Performance upside, but high implementation risk. | High |

## Not Recommended For The Current Roadmap

- Multi-GPU, NCCL, MPI, or multi-node training.
- Flash Attention, Transformer blocks, or GPT-style training.
- tiny-cuda-nn style fully fused MLP kernels.
- HashGrid or neural graphics encodings.
- JIT fusion or runtime CUDA code generation.
- CUTLASS/Tensor Core rewrites before stronger baseline benchmarks exist.

These are technically interesting, but they would change MiniCNN's scope and
delay higher-value improvements to the existing user path.

## Final Positioning

MiniCNN is best positioned as a dual-backend educational CNN framework:

- Torch/flex for fast experimentation and broad PyTorch component coverage.
- CPU/NumPy autograd for framework learning and tests without PyTorch.
- CUDA legacy for owning and understanding handwritten CUDA kernels.

The next milestone should be "credible single-GPU CUDA backend for common CNN
training", not "match llm.c" or "match tiny-cuda-nn". After CUDA Adam,
BatchNorm2d, LayerNorm integration, parity tests, and benchmarks land, MiniCNN's
value proposition becomes much clearer: one config can run through PyTorch for
experimentation and through handwritten CUDA for low-level learning and backend
control.
