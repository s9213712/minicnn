# CUDA Legacy BatchNorm2d Evaluation

This note records the current state of BatchNorm2d for the handcrafted
`cuda_legacy` backend.

See also:

- [../USAGE.md](../USAGE.md)
- [backend_capabilities.md](backend_capabilities.md)
- [dual_backend_guide.md](dual_backend_guide.md)

## Current Status

BatchNorm2d is supported by the Torch/flex backend and by the CPU/NumPy autograd
backend. It is not supported by CUDA legacy training yet.

The CUDA legacy backend validates a fixed CIFAR-10 training graph:

```text
Conv2d -> ReLU/LeakyReLU -> Conv2d -> ReLU/LeakyReLU -> MaxPool2d
-> Conv2d -> ReLU/LeakyReLU -> Conv2d -> ReLU/LeakyReLU -> MaxPool2d
-> Flatten -> Linear
```

If `BatchNorm2d` appears in a `cuda_legacy` config, validation now reports that
the layer is unsupported and tells users to use `engine.backend=torch` or remove
BatchNorm2d for the handcrafted CUDA path.

## What Is Missing

CUDA legacy needs more than standalone forward/backward math before BatchNorm2d
can be called supported:

| Area | Missing Work |
|---|---|
| Native kernels | BatchNorm2d train forward, eval forward, backward input, backward gamma, backward beta |
| Runtime state | Per-BN gamma, beta, running mean, running var, and velocity buffers |
| Workspace | Saved batch mean, inverse std, normalized activation, and gradient buffers per BN layer |
| Compiler/validator | A CUDA legacy layer pattern that accepts Conv-BN-Activation blocks |
| Checkpoints | Save/load BN affine parameters and running stats |
| Updates | SGD/momentum updates for BN gamma and beta, with no weight decay on beta |
| Tests | NumPy/PyTorch parity for forward, backward, running stats, and one training step |

## Recommended Integration Order

1. Add a pure NumPy BatchNorm2d reference test for train/eval forward and
   backward against PyTorch.
2. Add native CUDA kernels behind explicit C ABI symbols without wiring them
   into training.
3. Add ctypes bindings and GPU smoke tests that can skip when CUDA is missing.
4. Extend `DeviceWeights`, `VelocityBuffers`, and checkpoint format for BN
   affine and running-stat tensors.
5. Extend `BatchWorkspace` and `cuda_batch.py` for Conv-BN-Activation call
   order.
6. Extend `cuda_legacy` config validation to accept only the exact BN placement
   that the runtime supports.

## Risk

BatchNorm2d is a medium-high risk CUDA legacy change. It touches model
validation, native kernels, workspace allocation, update semantics, checkpoint
compatibility, and numerical parity. It should not be mixed into the same patch
as unrelated optimizer or benchmark changes.
