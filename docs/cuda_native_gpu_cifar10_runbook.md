# cuda_native GPU CIFAR-10 Runbook

Last updated: 2026-04-25

This runbook records the current real-data `cuda_native` strict GPU training
path for CIFAR-10.

It uses `engine.execution_mode=gpu_native`, so a successful run means the
accepted helper subset ran through native CUDA device-pointer training helpers.
It does not silently fall back to the NumPy reference path.

## Config

Use:

```bash
configs/cifar10_cuda_native_gpu_stronger.yaml
```

The model is intentionally inside the current strict `gpu_native` helper
boundary:

```text
Conv2d(valid, bias=false, 16ch)
-> ReLU
-> Conv2d(valid, bias=false, 32ch)
-> ReLU
-> MaxPool2d(2,2)
-> Flatten
-> Linear(10)
```

The optimizer is SGD with momentum and weight decay:

```text
lr=0.01
momentum=0.9
weight_decay=0.0005
scheduler=StepLR(step_size=20, gamma=0.5)
```

## Commands

Validate the config:

```bash
cd /home/s92137/NN/minicnn

PYTHONPATH=src python3 -m minicnn.cli validate-cuda-native-config \
  --config configs/cifar10_cuda_native_gpu_stronger.yaml
```

Run training:

```bash
PYTHONPATH=src timeout 7200s python3 -m minicnn.cli train-native \
  --config configs/cifar10_cuda_native_gpu_stronger.yaml
```

For a throughput experiment, increase batch size:

```bash
PYTHONPATH=src timeout 7200s python3 -m minicnn.cli train-native \
  --config configs/cifar10_cuda_native_gpu_stronger.yaml \
  train.batch_size=256
```

## Current Observed Result

Representative local run on real CIFAR-10:

```text
Loading CIFAR-10 training batches: (1, 2, 3, 4, 5)
Training samples: 50000
Test samples: 10000
Epoch 1/60:  train_loss=1.4073, train_acc=50.99%, val_acc=58.84%
Epoch 4/60:  train_loss=0.8814, train_acc=69.68%, val_acc=63.70%
Epoch 7/60:  train_loss=0.7345, train_acc=74.61%, val_acc=63.86%
Epoch 14/60: train_loss=0.5427, train_acc=80.88%, val_acc=62.68%
```

Interpretation:

- strict `gpu_native` real-data training is working for the two-Conv helper subset
- validation now uses the GPU forward path instead of a mixed NumPy eval path
- the model learns quickly, but validation accuracy plateaus around the low-to-mid 60% range
- after early epochs, train accuracy keeps rising while validation accuracy oscillates, so the current bottleneck is model capacity / regularization, not dataset loading or evaluation fallback

## Performance Notes

This path is functional but not throughput-oriented yet.

Current bottlenecks:

- per-batch helper execution still launches many small kernels
- input and parameter staging are still host-driven
- weights and optimizer state are not persistent device-resident tensors across batches
- Conv2d uses im2col + GEMM reference-style lowering rather than fused kernels
- validation still performs batch-by-batch GPU forward with host-visible logits

Recently reduced overhead:

- training loop no longer requests intermediate activations and gradients from the two-Conv helper during normal training
- `metrics.jsonl` now records `train_acc`, making train/eval diagnosis explicit
- GPU forward lowering now uses runtime batch shape instead of graph build-time batch shape

## Next Optimization Direction

The next major GPU-utilization work should be:

1. persistent device-resident parameters and optimizer state across batches
2. device-side input staging reuse for CIFAR-10 batches
3. reduced host synchronization in validation
4. fused Conv/ReLU/Pool helper variants for the current two-Conv subset
5. broader helper coverage for normalization or augmentation so higher-capacity models can stay inside strict `gpu_native`

