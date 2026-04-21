"""Demonstration of flex block presets (conv_bn_relu, conv_bn_silu). Requires PyTorch."""
try:
    import torch
except ImportError:
    print("PyTorch not available; skipping 07_flex_presets.py")
    raise SystemExit(0)

import torch
from minicnn.flex.builder import _expand_presets, build_model


def main():
    print("=== Flex Block Presets Demo ===\n")

    # Show preset expansion
    layers_cfg = [
        {'type': 'conv_bn_relu', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
        {'type': 'conv_bn_silu', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
        {'type': 'MaxPool2d', 'kernel_size': 2},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]
    expanded = _expand_presets(layers_cfg)
    print("Expanded layer types:")
    for i, layer in enumerate(expanded):
        print(f"  [{i}] {layer['type']}")

    # Build and run
    input_shape = [16, 8, 8]
    model_cfg = {
        'layers': [
            {'type': 'conv_bn_relu', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
            {'type': 'conv_bn_silu', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
            {'type': 'MaxPool2d', 'kernel_size': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]
    }
    model = build_model(model_cfg, input_shape=input_shape)
    model.eval()
    x = torch.randn(2, *input_shape)
    with torch.no_grad():
        y = model(x)
    print(f"\nInput shape:  {list(x.shape)}")
    print(f"Output shape: {list(y.shape)}")

    print("\nChild modules:")
    for i, child in enumerate(model.children()):
        print(f"  [{i}] {child.__class__.__name__}")

    print("\nFlex presets demo ran successfully.")


if __name__ == "__main__":
    main()
