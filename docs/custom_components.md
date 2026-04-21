# Custom Components

MiniCNN currently supports custom dotted-path imports on the torch/flex side.

That means:

- `model.layers[].type` may point at an importable Python class or factory
- model-level `factory` entries may point at an importable Python callable
- CUDA legacy does not automatically inherit support for those components
- the examples in `src/minicnn/extensions/custom_components.py` are torch-only

## Layer Example

```yaml
model:
  layers:
    - type: minicnn.extensions.custom_components.ConvBNReLU
      out_channels: 64
      kernel_size: 3
      padding: 1
    - type: minicnn.extensions.custom_components.Swish
```

The import path is resolved through the flex builder. Local examples live in:

- `src/minicnn/extensions/custom_components.py`
- `examples/custom_block.py`

If torch is not installed, importing those example components will now fail with
a short torch-only dependency message instead of a raw import traceback.

## Model Factory Example

If the whole model is easier to build in Python, use `model.factory`:

```yaml
model:
  factory: package.module:build_model
```

The callable receives the full `model` config mapping and is expected to return
a torch `nn.Module`.

## Dataset Factory Example

The torch/flex path also accepts custom dataset factories via `dataset.type`:

```yaml
dataset:
  type: minicnn.extensions.custom_datasets:checkerboard_dataset
  input_shape: [1, 8, 8]
  num_classes: 2
```

The callable receives `(dataset_cfg, train_cfg)` and is expected to return a
dict with `train`, `val`, and optional `test` splits, where each split is
`(x, y)` as NumPy arrays.

## What This Does Not Mean

Custom dotted-path components are a torch/flex feature.

They do not automatically become valid for:

- `cuda_legacy`
- the NumPy autograd path
- the experimental `cuda_native` backend

If a component also needs to run on `cuda_legacy`, you still have to add:

- validation in `src/minicnn/unified/cuda_legacy.py`
- any required native kernels in `cpp/src/`
- ctypes bindings in `src/minicnn/core/cuda_backend.py`
- training-graph and workspace integration in `src/minicnn/training/`

## Tips

- Keep custom blocks composable and constructor arguments YAML-friendly.
- Prefer adding built-ins only after a component proves repeatedly useful.
- Keep user-facing booleans compatible with the strict config parser.
- Use dotted CLI overrides such as `model.layers.1.out_features=64` for quick experiments.
