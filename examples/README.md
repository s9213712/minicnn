# Examples

This folder contains runnable examples, not one single progression. If you are
new to the repo, do **not** start by opening every subfolder.

## Canonical Beginner Path

Use this order:

1. `minicnn smoke`
2. `minicnn show-model --config configs/flex_cnn.yaml --format text`
3. `minicnn train-flex --config templates/mnist/lenet_like.yaml`
4. `minicnn inspect-checkpoint --path <best-checkpoint>`
5. only then branch into the example families below

Why this order:

- it keeps the first run CPU- and onboarding-friendly
- it teaches the config/frontend surface before custom Python extension points
- it avoids sending new users straight into native-library setup or niche demos

If you want ready-to-edit training configs, start with [templates/README.md](../templates/README.md)
before editing Python examples directly.

## Example Families

### 1. Extension point example

- `custom_block.py`: custom activation and custom block classes usable from YAML via dotted import paths.

Use this when you already understand the normal flex path and want to extend the
frontend with your own components.

### 2. Feature expansion demos

- `feature_expansion/`: small self-contained demos for activations, optimizers,
  schedulers, initialization, label smoothing, autograd training, flex presets,
  and augmentation.

Recommended order inside that folder:

1. `01_activations.py`
2. `02_optimizers.py`
3. `03_schedulers.py`
4. `04_initialization.py`
5. `05_label_smoothing.py`
6. `06_train_autograd_enhanced.py`
7. `07_flex_presets.py`
8. `08_augmentation.py`

### 3. Inference example

- `inference/predict_image.py`: real-image preprocessing and single-image
  prediction from a torch/flex checkpoint.

Use this only after you already have a checkpoint you understand.

### 4. Native ctypes examples

- `mnist_ctypes/`: self-contained MNIST ctypes examples for the handcrafted
  CUDA shared library
- `mnist_ctypes/check_native_library.py`: direct `.so`/`.dll` smoke test

This is **not** the beginner path. Use it when you intentionally want the
native-library route.

## Checkpoints And Reuse

Best checkpoints are written under `artifacts/models/`. Use:

```bash
minicnn inspect-checkpoint --path <file>
```

before trying to reuse one. For torch/flex checkpoints you can also run:

```bash
minicnn evaluate-checkpoint --config <yaml> --checkpoint <file.pt>
python -u examples/inference/predict_image.py --config <yaml> --checkpoint <file.pt> --image <file>
```

## Useful Config Notes

- `train.init_seed` controls torch/flex model initialization.
- CLI overrides may address layer-list entries, for example `model.layers.1.out_features=7`.
- String booleans such as `"false"` are parsed strictly for data, augmentation, AMP, and optimizer helper flags.

## cuda_native Beta Demo

Run the real-dataset beta demo against CIFAR-10:

```bash
PYTHONPATH=src python3 examples/cuda_native_amp_cifar10_beta_demo.py \
  --data-root data/cifar-10-batches-py \
  --artifacts-root /tmp/minicnn_cuda_native_beta_demo
```

This demo trains a small `cuda_native` model with `AdamW + AMP + grad_accum_steps=2`, then evaluates on the official CIFAR-10 `test_batch` split and prints a JSON summary.

Run the partial native-forward GPU demo against CIFAR-10:

```bash
PYTHONPATH=src python3 examples/cuda_native_gpu_forward_cifar10_demo.py \
  --data-root data/cifar-10-batches-py \
  --batch-size 4
```

This demo loads an official CIFAR-10 test batch, executes `Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear` through the native GPU forward executor, compares against the NumPy reference executor, and prints the native kernel execution kinds.
