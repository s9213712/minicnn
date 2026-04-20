# MiniCNN Comparison Follow-up TODO

This file tracks follow-up work from `docs/comparison_completion_report.md`.
It keeps the roadmap scoped by backend so feature gaps are not confused across
Torch/flex, CPU/NumPy autograd, and CUDA legacy.

## In Progress

- [x] Link comparison follow-up docs from the main README so the report is not
  an orphan document.
- [x] Add CPU/NumPy autograd `mse_loss` and `bce_with_logits_loss`.
- [x] Let `minicnn train-autograd` select `CrossEntropyLoss`, `MSELoss`, or
  `BCEWithLogitsLoss` from config.
- [x] Add focused tests for the new autograd losses.
- [x] Add a backend capability matrix covering Torch/flex, CPU/NumPy autograd,
  and CUDA legacy.

## Next

- [x] Document the native LayerNorm kernel as tested but not wired into the
  CUDA legacy training path.
- [x] Add PyTorch parity tests for small autograd and CUDA kernel cases.
- [x] Add a benchmark report template for samples/sec, epoch time, and GPU
  memory.
- [x] Add benchmark fields to `minicnn compare` so users can fill the report
  template without hand-computing epoch time or samples/sec.
- [x] Fix `minicnn compare --backends ... key=value` parsing so backend lists
  and config overrides can be used in the same command.
- [x] Clean up `Tensor.__pow__` zero-base negative-exponent warnings while
  preserving the existing zero-gradient behavior.

## Later CUDA Work

- [ ] Add CUDA Adam or AdamW optimizer state and update kernels.
- [ ] Add CUDA BatchNorm2d forward/backward and config/runtime support.
- [ ] Add global gradient norm clipping for CUDA legacy.
- [ ] Add CUDA residual/skip add support after the legacy model compiler can
  represent it cleanly.

## Deferred

- [ ] CUDA FP16/AMP support.
- [ ] Multi-GPU, NCCL, MPI, or multi-node training.
- [ ] Transformer, Flash Attention, or GPT-style training.
- [ ] tiny-cuda-nn style fully fused MLPs, HashGrid encodings, or JIT fusion.
