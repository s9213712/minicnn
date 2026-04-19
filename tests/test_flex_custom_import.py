from minicnn.flex.builder import build_model


def test_build_with_custom_activation_import():
    cfg = {
        'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 32},
            {'type': 'minicnn.extensions.custom_components.Swish'},
            {'type': 'Linear', 'out_features': 10},
        ]
    }
    model = build_model(cfg, input_shape=(1, 8, 8))
    assert model[2].__class__.__name__ == 'Swish'
    assert model[-1].in_features == 32
