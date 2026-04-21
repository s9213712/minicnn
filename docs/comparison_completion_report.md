# MiniCNN Comparison Completion Report

This is the current backend-scoped comparison note for MiniCNN.

It compares MiniCNN with:

- [karpathy/llm.c](https://github.com/karpathy/llm.c)
- [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [BobMcDear/neural-network-cuda](https://github.com/BobMcDear/neural-network-cuda)

## Executive Summary

MiniCNN is already reasonably complete as a small configurable framework.

Its main gap is not "framework shape." The main gap is that its native training
path is still split between:

- a stable but narrow `cuda_legacy` backend
- an experimental `cuda_native` backend whose public CLI surface exists, but
  whose validated contract is still narrow and prototype-level

That means the next useful improvements are backend-focused, not marketing or
surface-level ones.

## Capability Layers

MiniCNN currently has four relevant layers:

| Layer | Current status |
|---|---|
| Torch/flex | Broadest practical user path; supports richer layers, losses, schedulers, augmentation, and custom dotted-path components. |
| CPU/NumPy autograd | Stable educational path; supports multiple losses, optimizers, schedulers, and a compact layer stack. |
| `cuda_legacy` | Stable handwritten CUDA path; intentionally narrow, CIFAR-10-centered, validator-driven. |
| `cuda_native` | Experimental public backend surface; CLI-visible, but still narrow, prototype-level, and not production-ready. |

Any comparison that ignores this split will overstate or understate MiniCNN's
real status.

## Comparison With `neural-network-cuda`

`neural-network-cuda` is a compact educational CUDA project. MiniCNN already
goes further in framework structure:

- YAML- and CLI-driven workflows
- multiple backend paths
- broader tests and docs
- a PyTorch path for fast iteration
- a CPU autograd path for reference behavior

What MiniCNN can still borrow from it is presentation discipline: keeping one
small, beginner-friendly path that does not require users to learn the whole
repo at once.

## Comparison With `llm.c`

`llm.c` is a production-leaning LLM training project. MiniCNN should not try to
match that scope.

The relevant lessons are narrower:

- backend parity tests matter
- real throughput reporting matters
- debug-state capture and deterministic validation matter
- optimizer/runtime completeness matters more than adding new marketing claims

Useful borrowable ideas:

- stronger forward/backward parity checks against PyTorch
- more disciplined benchmark reporting
- clearer native backend capability boundaries

Not good near-term goals:

- GPT-style training
- attention kernels
- multi-GPU / multi-node systems
- production mixed-precision infrastructure

## Comparison With `tiny-cuda-nn`

`tiny-cuda-nn` is a performance-first CUDA library. MiniCNN should not chase its
fully fused MLP or neural-graphics direction.

The useful lessons are:

- benchmark honestly
- keep capability tables explicit
- separate experimental backend work from stable public APIs

MiniCNN should borrow the discipline, not the scope.

## Current Backend-Scoped Gaps

### Higher-value gaps

- broader `cuda_legacy` training coverage
- stronger PyTorch parity tests for native kernels
- benchmark reports that explain backend tradeoffs clearly
- a cleaner promotion path from experimental `cuda_native` capabilities to a
  broader stable backend surface

### Lower-value or out-of-scope work

- Transformers or attention kernels
- multi-GPU infrastructure
- tiny-cuda-nn-style fused MLP work
- runtime CUDA codegen or JIT fusion

## Recommended Direction

The practical sequence is:

1. keep the frontend broad and honest
2. keep `cuda_legacy` narrow and validator-driven
3. document `cuda_native` as public experimental surface without pretending it is stable
4. only promote native functionality when its CLI, capability table, and tests
   all line up

## Final Positioning

MiniCNN is best understood as:

- a broad frontend plus torch path for experimentation
- a NumPy autograd path for learning and tests
- a narrow handcrafted CUDA path for low-level study
- an open workspace for prototyping a future graph-native backend

That is already a coherent position. The next gains come from tightening backend
correctness and backend generalization, not from pretending the project is a
drop-in PyTorch replacement.
