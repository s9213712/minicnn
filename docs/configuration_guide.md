# Configuration Guide

MiniCNN V7 is designed so that users edit configuration instead of framework internals.

## Main sections

Typical top-level sections are:
- `dataset`
- `model`
- `loss`
- `optimizer`
- `scheduler`
- `train`

## Dataset augmentation

CIFAR-10 torch flex configs may enable standard lightweight augmentation from
YAML:

```yaml
dataset:
  random_crop_padding: 4
  horizontal_flip: true
```

The legacy CUDA trainer exposes the same knobs through its compiled
experiment config and environment overrides:

```bash
MINICNN_RANDOM_CROP_PADDING=4 minicnn train-dual --config configs/dual_backend_cnn.yaml engine.backend=cuda_legacy
```

## Model section

### Flex / Torch backend

```yaml
model:
  input_shape: [3, 32, 32]
  layers:
    - type: Conv2d
      out_channels: 32
      kernel_size: 3
      padding: 1
    - type: ReLU
    - type: MaxPool2d
      kernel_size: 2
      stride: 2
    - type: Flatten
    - type: Linear
      out_features: 10
```

The torch flex builder can infer `in_channels`, `num_features`, and
`in_features` for common layers. It supports the sequential layer subset used
by `configs/alexnet_like.yaml`, plus torch-only residual blocks in
`configs/resnet_like.yaml`:

```yaml
model:
  layers:
    - type: Conv2d
      out_channels: 64
      kernel_size: 3
      padding: 1
      bias: false
    - type: BatchNorm2d
    - type: ReLU
    - type: ResidualBlock
      channels: 64
      stride: 1
    - type: GlobalAvgPool2d
    - type: Flatten
    - type: Linear
      out_features: 10
```

`ResidualBlock`, `BatchNorm2d`, and `GlobalAvgPool2d` are currently torch flex
components. The `cuda_legacy` backend still accepts only the supported CNN
subset reported by `minicnn validate-dual-config`.

### CUDA backend (`train-cuda` / `cuda_legacy`)

The handcrafted CUDA path uses `model.conv_layers` instead of `model.layers`.
Each entry specifies `out_c` (output channels) and `pool` (whether a 2×2
max-pool follows the convolution). All spatial shapes and buffer sizes are
derived automatically by `CudaNetGeometry` — no Python changes needed.

```yaml
model:
  c_in: 3
  h: 32
  w: 32
  kh: 3
  kw: 3
  fc_out: 10
  conv_layers:
    - {out_c: 32, pool: false}   # conv1
    - {out_c: 32, pool: true}    # conv2 + pool
    - {out_c: 64, pool: false}   # conv3
    - {out_c: 64, pool: true}    # conv4 + pool
```

To add a stage, append an entry. To widen a stage, increase its `out_c`. Run
`minicnn validate-dual-config --config configs/train_cuda.yaml` after editing
to confirm the config is within the supported kernel subset.

The legacy CUDA implementation is split by responsibility:

- `src/minicnn/training/train_cuda.py` handles orchestration: data loading,
  epochs, validation, checkpointing, LR reduction, early stop, and final test
  evaluation.
- `src/minicnn/training/cuda_batch.py` handles one training batch: conv
  forward, FC forward, fused loss/accuracy, FC update, and conv backward/update.
- `src/minicnn/training/loop.py` holds shared metrics, LR state, best/plateau
  state, timing, and summary formatting used by both CUDA and Torch legacy
  trainers.
- `src/minicnn/training/legacy_data.py` holds shared CIFAR-10
  load/normalization for the legacy trainers.

## Optimizer section

```yaml
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.0001
  exclude_bias_norm_weight_decay: true
```

For torch flex training, `exclude_bias_norm_weight_decay: true` keeps weight
decay off bias terms and normalization-layer parameters while preserving it for
regular weights.

## Scheduler section

```yaml
scheduler:
  type: CosineAnnealingLR
  T_max: 20
```

## Training controls

```yaml
train:
  epochs: 50
  grad_accum_steps: 4
  early_stop_patience: 8
  min_delta: 0.001

runtime:
  save_every_n_epochs: 5
```

`train-flex` flushes the final partial gradient-accumulation window at the end
of each epoch. When `early_stop_patience` is greater than zero, torch flex
training stops after that many epochs without a `val_acc` improvement larger
than `min_delta`. Periodic checkpoints are written to
`src/minicnn/training/models/` as `*_epoch_<N>.pt`; best checkpoints still use
`*_best.pt`.

## CLI overrides

```bash
minicnn train-flex --config configs/flex_cnn.yaml train.epochs=3 optimizer.lr=0.0005
```

## Design rule

If a user needs to change architecture or common hyperparameters, they should usually be able to do it from YAML.
