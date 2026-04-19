from minicnn.unified.config import load_unified_config
from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility, compile_to_legacy_experiment


def test_unified_config_override_backend():
    cfg = load_unified_config(None, ['engine.backend=cuda_legacy', 'train.epochs=3'])
    assert cfg['engine']['backend'] == 'cuda_legacy'
    assert cfg['train']['epochs'] == 3


def test_compile_supported_dual_config():
    cfg = {
        'project': {'name': 'x', 'run_name': 'y', 'artifacts_root': 'artifacts'},
        'engine': {'backend': 'cuda_legacy'},
        'dataset': {'type': 'cifar10', 'input_shape': [3, 32, 32], 'num_classes': 10, 'num_samples': 128, 'val_samples': 32, 'seed': 42},
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
        'train': {'epochs': 2, 'batch_size': 16, 'amp': False, 'grad_accum_steps': 1},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005},
    }
    assert validate_cuda_legacy_compatibility(cfg) == []
    exp = compile_to_legacy_experiment(cfg)
    assert exp.model.c1_out == 32
    assert exp.model.c4_out == 64
    assert exp.optim.lr_fc == 0.005
