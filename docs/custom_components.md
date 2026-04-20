# Custom Components

MiniCNN V7 supports custom dotted-path imports.

## Example config

```yaml
model:
  layers:
    - type: examples.custom_block.ConvBNReLU
      out_channels: 64
      kernel_size: 3
      padding: 1
    - type: examples.custom_block.Swish
```

## Example Python file

See `examples/custom_block.py`.

## Tips

- Keep custom blocks composable.
- Prefer constructor arguments that serialize cleanly into YAML.
- If you add a component often, consider registering it as a built-in.
- Keep user-facing options compatible with the shared config parser: booleans should parse through the strict true/false rules, and nested layer changes should work with list-index overrides such as `model.layers.1.out_features=64`.
- Custom components are torch/flex only unless you also add CUDA legacy validation, workspace allocation, ctypes bindings, and native kernels.
