# Example Expansion Plan

Last updated: 2026-04-23

This document records the canonical example-expansion path for MiniCNN.

It answers two questions:

- which examples should new users see first
- how should future examples be added without turning `examples/` into an
  unordered dump

See also:

- [master_roadmap.md](master_roadmap.md)
- [templates/README.md](../templates/README.md)
- [../examples/README.md](../examples/README.md)

## Canonical Beginner Path

New users should encounter MiniCNN in this order:

1. `minicnn smoke`
2. `minicnn show-model --config configs/flex_cnn.yaml --format text`
3. `minicnn train-flex --config templates/mnist/lenet_like.yaml`
4. `minicnn inspect-checkpoint --path <best-checkpoint>`
5. only then branch into secondary example families

This path is canonical because it:

- keeps the first run simple
- avoids immediate native build requirements
- teaches the config/frontend surface before extension points
- gives the user one successful training run before advanced topics

## Canonical Training Templates

Priority order:

1. `templates/mnist/lenet_like.yaml`
2. `templates/cifar10/vgg_mini.yaml`
3. `templates/mnist/mlp.yaml`
4. `templates/cifar10/alexnet_like.yaml`
5. `templates/cifar10/resnet_like.yaml`
6. `templates/cifar10/vgg_mini_cuda.yaml`

Interpretation:

- the first two are the main user-facing starting templates
- the middle set are secondary reference templates
- the CUDA-specific template is intentionally later because it depends on a
  narrower backend boundary

## Example Families

### Family A: Beginner-safe operational examples

These should stay closest to the main onboarding path:

- `templates/mnist/lenet_like.yaml`
- `templates/cifar10/vgg_mini.yaml`
- `examples/inference/predict_image.py`

Rule:

- keep this family small and stable

### Family B: Feature demos

These explain isolated concepts after the beginner path:

- `examples/feature_expansion/01_activations.py`
- `examples/feature_expansion/02_optimizers.py`
- `examples/feature_expansion/03_schedulers.py`
- `examples/feature_expansion/04_initialization.py`
- `examples/feature_expansion/05_label_smoothing.py`
- `examples/feature_expansion/06_train_autograd_enhanced.py`
- `examples/feature_expansion/07_flex_presets.py`
- `examples/feature_expansion/08_augmentation.py`

Rule:

- each script should teach one coherent topic
- advanced topics should come later in the numbered order

### Family C: Extension examples

These are for users who already understand the normal flex path:

- `examples/custom_block.py`

Rule:

- extension examples should assume the user already knows the standard config path

### Family D: Native-library examples

These are intentionally not the default beginner path:

- `examples/mnist_ctypes/train_mnist_so_full_cnn_frame.py`
- `examples/mnist_ctypes/check_native_library.py`
- historical `examples/mnist_ctypes/legacy/*`

Rule:

- native ctypes examples should stay clearly labeled as opt-in specialist paths

## Expansion Rules For New Examples

Add a new example only if it fits one of these buckets:

1. beginner-safe operational path
2. isolated feature demo
3. extension example
4. native-library specialist example

Every new example should define:

- intended audience
- required dependencies
- the command to run it
- whether it belongs in the canonical beginner path

## What Not To Do

Do not:

- add multiple overlapping beginner examples
- put native-library setup on the default path
- add examples that require undocumented config behavior
- leave example ordering implicit
- use `examples/` as a substitute for capability docs

## Future Expansion Direction

The next useful example-growth areas are:

- one clearer checkpoint-evaluation / inference path
- one clearer `cuda_native` example once its promoted subset is less prototype-like
- selective frontend feature demos only when they reinforce the supported docs

Not the priority:

- large numbers of architecture showcase examples
- architecture-specific examples whose backend boundaries are still unsettled

