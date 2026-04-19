from __future__ import annotations

from dataclasses import asdict
from typing import Any

from minicnn.config.schema import ExperimentConfig


CUDA_LEGACY_SUPPORTED = {
    'layer_pattern': [
        ['Conv2d'], ['ReLU', 'LeakyReLU'],
        ['Conv2d'], ['ReLU', 'LeakyReLU'],
        ['MaxPool2d'],
        ['Conv2d'], ['ReLU', 'LeakyReLU'],
        ['Conv2d'], ['ReLU', 'LeakyReLU'],
        ['MaxPool2d'],
        ['Flatten'], ['Linear'],
    ],
    'optimizer': ['SGD'],
    'loss': ['CrossEntropyLoss'],
    'dataset': ['cifar10'],
}


def _normalize_layer_name(layer: dict[str, Any]) -> str:
    return str(layer.get('type'))


def _collect_conv_blocks(model_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    layers = model_cfg.get('layers', [])
    if not isinstance(layers, list):
        raise TypeError('model.layers must be a list')
    convs: list[dict[str, Any]] = []
    activations: list[dict[str, Any]] = []
    linear: dict[str, Any] | None = None
    expected_idx = 0
    expected = CUDA_LEGACY_SUPPORTED['layer_pattern']
    for layer in layers:
        name = _normalize_layer_name(layer)
        if expected_idx >= len(expected):
            raise ValueError('Too many layers for cuda_legacy backend')
        allowed = expected[expected_idx]
        if name not in allowed:
            expected_text = allowed[0] if len(allowed) == 1 else ' or '.join(repr(x) for x in allowed)
            raise ValueError(
                f'cuda_legacy expects layer #{expected_idx + 1} to be {expected_text}, got {name!r}'
            )
        if name == 'Conv2d':
            convs.append(layer)
        elif name in {'ReLU', 'LeakyReLU'}:
            activations.append(layer)
        elif name == 'Linear':
            linear = layer
        expected_idx += 1
    if expected_idx != len(expected):
        raise ValueError(f'cuda_legacy expects exactly {len(expected)} layers, got {expected_idx}')
    if len(convs) != 4:
        raise ValueError('cuda_legacy requires exactly four Conv2d layers')
    if linear is None:
        raise ValueError('cuda_legacy requires a final Linear layer')
    return convs, activations, linear


def validate_cuda_legacy_compatibility(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    dataset = cfg.get('dataset', {})
    model = cfg.get('model', {})
    optim = cfg.get('optimizer', {})
    loss = cfg.get('loss', {})
    train = cfg.get('train', {})

    if str(dataset.get('type', 'cifar10')) not in CUDA_LEGACY_SUPPORTED['dataset']:
        errors.append('cuda_legacy only supports dataset.type=cifar10')
    if list(dataset.get('input_shape', [3, 32, 32])) != [3, 32, 32]:
        errors.append('cuda_legacy only supports dataset.input_shape=[3, 32, 32]')
    if int(dataset.get('num_classes', 10)) != 10:
        errors.append('cuda_legacy only supports 10 output classes')
    if str(optim.get('type', 'SGD')) not in CUDA_LEGACY_SUPPORTED['optimizer']:
        errors.append('cuda_legacy only supports optimizer.type=SGD')
    if str(loss.get('type', 'CrossEntropyLoss')) not in CUDA_LEGACY_SUPPORTED['loss']:
        errors.append('cuda_legacy only supports loss.type=CrossEntropyLoss')
    if int(train.get('grad_accum_steps', 1)) != 1:
        errors.append('cuda_legacy does not support grad_accum_steps != 1')
    if bool(train.get('amp', False)):
        errors.append('cuda_legacy does not support amp=true')

    try:
        convs, activations, linear = _collect_conv_blocks(model)
    except Exception as exc:
        errors.append(str(exc))
        return errors

    negative_slope = None
    for act in activations:
        act_type = str(act.get('type'))
        if act_type == 'ReLU':
            slope = 0.0
        elif act_type == 'LeakyReLU':
            slope = float(act.get('negative_slope', 0.1))
        else:
            errors.append(f'Unsupported activation for cuda_legacy: {act_type}')
            continue
        if negative_slope is None:
            negative_slope = slope
        elif abs(negative_slope - slope) > 1e-12:
            errors.append('cuda_legacy requires the same activation slope in all conv blocks')

    for idx, conv in enumerate(convs, start=1):
        if int(conv.get('kernel_size', 3)) != 3:
            errors.append(f'Conv{idx} must use kernel_size=3 for cuda_legacy')
        if int(conv.get('stride', 1)) != 1:
            errors.append(f'Conv{idx} must use stride=1 for cuda_legacy')
        if int(conv.get('padding', 0)) != 0:
            errors.append(f'Conv{idx} must use padding=0 for cuda_legacy')
        if idx > 1:
            prev_out = int(convs[idx - 2].get('out_channels'))
            in_ch = int(conv.get('in_channels', prev_out))
            if in_ch != prev_out:
                errors.append(f'Conv{idx} in_channels must equal previous out_channels ({prev_out})')
    for pool_pos in (4, 9):
        layer = model.get('layers', [])[pool_pos]
        if int(layer.get('kernel_size', 2)) != 2 or int(layer.get('stride', 2)) != 2:
            errors.append(f'Pool layer at position {pool_pos + 1} must use kernel_size=2 and stride=2')
    if int(linear.get('out_features', 10)) != 10:
        errors.append('Final Linear layer must use out_features=10 for cuda_legacy')
    return errors


def compile_to_legacy_experiment(cfg: dict[str, Any]) -> ExperimentConfig:
    errors = validate_cuda_legacy_compatibility(cfg)
    if errors:
        joined = '\n- '.join(errors)
        raise ValueError(f'Config is not compatible with cuda_legacy:\n- {joined}')

    dataset = cfg.get('dataset', {})
    model = cfg.get('model', {})
    optim = cfg.get('optimizer', {})
    train = cfg.get('train', {})
    project = cfg.get('project', {})

    convs, activations, linear = _collect_conv_blocks(model)
    negative_slope = 0.0 if activations[0].get('type') == 'ReLU' else float(activations[0].get('negative_slope', 0.1))

    exp = ExperimentConfig()
    exp.project.name = str(project.get('name', 'minicnn'))
    exp.project.run_name = str(project.get('run_name', 'dual-backend'))
    exp.project.artifacts_root = str(project.get('artifacts_root', 'artifacts'))
    exp.backend.type = 'cuda'
    exp.backend.legacy_entrypoint = True

    exp.train.batch_size = int(train.get('batch_size', exp.train.batch_size))
    exp.train.epochs = int(train.get('epochs', exp.train.epochs))
    exp.train.dataset_seed = int(dataset.get('seed', exp.train.dataset_seed))
    exp.train.n_train = int(dataset.get('num_samples', exp.train.n_train))
    exp.train.n_val = int(dataset.get('val_samples', exp.train.n_val))
    exp.train.max_steps_per_epoch = train.get('max_steps_per_epoch', exp.train.max_steps_per_epoch)

    exp.optim.lr_conv1 = float(optim.get('lr_conv1', optim.get('lr', exp.optim.lr_conv1)))
    exp.optim.lr_conv = float(optim.get('lr_conv', optim.get('lr', exp.optim.lr_conv)))
    exp.optim.lr_fc = float(optim.get('lr_fc', optim.get('lr', exp.optim.lr_fc)))
    exp.optim.momentum = float(optim.get('momentum', exp.optim.momentum))
    exp.optim.weight_decay = float(optim.get('weight_decay', exp.optim.weight_decay))
    exp.optim.leaky_alpha = negative_slope

    exp.model.c1_in = int(dataset.get('input_shape', [3, 32, 32])[0])
    exp.model.c1_out = int(convs[0]['out_channels'])
    exp.model.c2_in = int(convs[1].get('in_channels', exp.model.c1_out))
    exp.model.c2_out = int(convs[1]['out_channels'])
    exp.model.c3_in = int(convs[2].get('in_channels', exp.model.c2_out))
    exp.model.c3_out = int(convs[2]['out_channels'])
    exp.model.c4_in = int(convs[3].get('in_channels', exp.model.c3_out))
    exp.model.c4_out = int(convs[3]['out_channels'])
    exp.model.h = int(dataset.get('input_shape', [3, 32, 32])[1])
    exp.model.w = int(dataset.get('input_shape', [3, 32, 32])[2])
    exp.model.kh = 3
    exp.model.kw = 3

    return exp


def summarize_legacy_mapping(cfg: dict[str, Any]) -> dict[str, Any]:
    exp = compile_to_legacy_experiment(cfg)
    data = asdict(exp)
    return {
        'backend': 'cuda_legacy',
        'project': data['project'],
        'model': data['model'],
        'optim': {
            'lr_conv1': data['optim']['lr_conv1'],
            'lr_conv': data['optim']['lr_conv'],
            'lr_fc': data['optim']['lr_fc'],
            'momentum': data['optim']['momentum'],
            'weight_decay': data['optim']['weight_decay'],
            'leaky_alpha': data['optim']['leaky_alpha'],
        },
        'train': {
            'batch_size': data['train']['batch_size'],
            'epochs': data['train']['epochs'],
            'n_train': data['train']['n_train'],
            'n_val': data['train']['n_val'],
        },
    }
