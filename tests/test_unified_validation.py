from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility


def test_cuda_legacy_rejects_unsupported_layer_sequence():
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 5},
            {'type': 'ReLU'},
        ]},
        'train': {'amp': False, 'grad_accum_steps': 1},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert errors
    assert any('expects exactly' in err or 'expects layer' in err for err in errors)


def test_cuda_legacy_rejects_batchnorm2d_with_actionable_message():
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'BatchNorm2d'},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
        ]},
        'train': {'amp': False, 'grad_accum_steps': 1},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
    }

    errors = validate_cuda_legacy_compatibility(cfg)

    assert any('does not yet support BatchNorm2d' in err for err in errors)


def test_cuda_legacy_rejects_unknown_native_variant():
    cfg = {
        'runtime': {'cuda_variant': 'experimental'},
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]},
        'train': {'amp': False, 'grad_accum_steps': 1},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
    }
    errors = validate_cuda_legacy_compatibility(cfg)
    assert any('runtime.cuda_variant' in err for err in errors)


def test_cuda_legacy_reports_missing_numeric_fields_without_traceback():
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'layers': [
            {'type': 'Conv2d', 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]},
        'train': {'amp': False, 'grad_accum_steps': 1},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.01},
    }

    errors = validate_cuda_legacy_compatibility(cfg)

    assert any('Conv1.out_channels' in err for err in errors)


def test_cuda_legacy_reports_malformed_top_level_numeric_fields_without_traceback():
    cfg = {
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 'abc', 'num_samples': 'bad'},
        'model': {'layers': [
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 'bad'},
            {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
            {'type': 'LeakyReLU', 'negative_slope': 0.1},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]},
        'train': {'amp': 'false', 'grad_accum_steps': 'bad', 'epochs': 'bad'},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 'bad'},
    }

    errors = validate_cuda_legacy_compatibility(cfg)

    assert any('dataset.num_classes' in err for err in errors)
    assert any('dataset.num_samples' in err for err in errors)
    assert any('train.grad_accum_steps' in err for err in errors)
    assert any('train.epochs' in err for err in errors)
    assert any('optimizer.lr_' in err for err in errors)
    assert any('LeakyReLU.negative_slope' in err for err in errors)
