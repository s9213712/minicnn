from minicnn.flex.builder import build_model


def test_build_cnn_with_shape_inference():
    cfg = {
        'layers': [
            {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
            {'type': 'BatchNorm2d'},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]
    }
    model = build_model(cfg, input_shape=(3, 32, 32))
    assert model[0].in_channels == 3
    assert model[1].num_features == 16
    assert model[-1].in_features == 16 * 16 * 16
    assert model[-1].out_features == 10
