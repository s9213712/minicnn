# Getting Started

## Recommended path

For most users, start with the configuration-driven path:

```bash
python -m pip install -e .[torch]
minicnn healthcheck
minicnn list-flex-components
minicnn train-flex --config configs/flex_cnn.yaml
```

## Common workflow

1. Copy a config file from `configs/`.
2. Edit the model, optimizer, scheduler, and dataset sections.
3. Run `minicnn train-flex --config <your-config>.yaml`.
4. Inspect the run artifacts.

## Suggested first configs

- `templates/mnist/lenet_like.yaml`: first CNN run; downloads MNIST on first use.
- `templates/mnist/mlp.yaml`: MLP baseline.
- `templates/cifar10/vgg_mini.yaml`: CIFAR-10 VGG-mini through torch/flex.
- `templates/cifar10/vgg_mini_cuda.yaml`: CIFAR-10 VGG-mini through CUDA legacy.
- `templates/cifar10/alexnet_like.yaml`: torch AlexNet-like CIFAR-10 model.
- `templates/cifar10/resnet_like.yaml`: torch ResNet-like CIFAR-10 model with residual blocks.

## Where training code lives

- `src/minicnn/training/train_cuda.py` is the CUDA legacy orchestration entrypoint.
- `src/minicnn/training/cuda_batch.py` contains CUDA batch forward/loss/backward/update steps.
- `src/minicnn/training/train_torch_baseline.py` contains Torch baseline orchestration and batch helpers.
- `src/minicnn/training/loop.py` contains shared metrics, LR, best/plateau, timing, and summary helpers.
- `src/minicnn/training/legacy_data.py` contains shared CIFAR-10 load/normalize logic for legacy trainers.

Best checkpoints are written under `src/minicnn/training/models/`.
