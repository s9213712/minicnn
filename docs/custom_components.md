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
