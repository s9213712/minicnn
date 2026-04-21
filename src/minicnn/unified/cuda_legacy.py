from __future__ import annotations

from dataclasses import asdict
from typing import Any

from minicnn.config.parsing import parse_bool
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
    'optimizer': ['SGD', 'Adam'],
    'loss': ['CrossEntropyLoss', 'MSELoss'],
    'dataset': ['cifar10'],
}


def _normalize_layer_name(layer: dict[str, Any]) -> str:
    return str(layer.get('type'))


def _coerce_int(value: Any, label: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f'{label} must be an integer, got {value!r}')
        return None


def _coerce_float(value: Any, label: str, errors: list[str]) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f'{label} must be a number, got {value!r}')
        return None


def _coerce_bool(value: Any, label: str, errors: list[str]) -> bool | None:
    try:
        return parse_bool(value, label=label)
    except ValueError as exc:
        errors.append(str(exc))
        return None


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
        if name == 'BatchNorm2d':
            raise ValueError(
                'cuda_legacy does not yet support BatchNorm2d in the training graph; '
                'use engine.backend=torch or remove BatchNorm2d for cuda_legacy'
            )
        if name == 'LayerNorm':
            raise ValueError(
                'cuda_legacy does not support LayerNorm; '
                'use engine.backend=torch for LayerNorm support'
            )
        if name == 'GroupNorm':
            raise ValueError(
                'cuda_legacy does not support GroupNorm; '
                'use engine.backend=torch for GroupNorm support'
            )
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
    runtime = cfg.get('runtime', {})

    if str(dataset.get('type', 'cifar10')) not in CUDA_LEGACY_SUPPORTED['dataset']:
        errors.append('cuda_legacy only supports dataset.type=cifar10')
    if list(dataset.get('input_shape', [3, 32, 32])) != [3, 32, 32]:
        errors.append('cuda_legacy only supports dataset.input_shape=[3, 32, 32]')
    num_classes = _coerce_int(dataset.get('num_classes', 10), 'dataset.num_classes', errors)
    if num_classes is not None and num_classes != 10:
        errors.append('cuda_legacy only supports 10 output classes')
    if str(optim.get('type', 'SGD')) not in CUDA_LEGACY_SUPPORTED['optimizer']:
        errors.append('cuda_legacy only supports optimizer.type=SGD or Adam')
    loss_type = str(loss.get('type', 'CrossEntropyLoss'))
    if loss_type == 'BCEWithLogitsLoss':
        errors.append(
            'cuda_legacy backend does not support BCEWithLogitsLoss. '
            'Use loss.type=CrossEntropyLoss, or switch to engine.backend=torch.'
        )
    elif loss_type not in CUDA_LEGACY_SUPPORTED['loss']:
        supported = ', '.join(CUDA_LEGACY_SUPPORTED['loss'])
        errors.append(f'cuda_legacy only supports loss.type in [{supported}]')
    grad_accum_steps = _coerce_int(train.get('grad_accum_steps', 1), 'train.grad_accum_steps', errors)
    if grad_accum_steps is not None and grad_accum_steps != 1:
        errors.append('cuda_legacy does not support grad_accum_steps != 1')
    amp = _coerce_bool(train.get('amp', False), 'train.amp', errors)
    if amp:
        errors.append('cuda_legacy does not support amp=true')
    for label, value in (
        ('train.batch_size', train.get('batch_size', 1)),
        ('train.epochs', train.get('epochs', 1)),
        ('dataset.seed', dataset.get('seed', 42)),
        ('train.init_seed', train.get('init_seed', 42)),
        ('train.seed', train.get('seed', train.get('train_seed', 42))),
        ('dataset.num_samples', dataset.get('num_samples', 1)),
        ('dataset.val_samples', dataset.get('val_samples', 1)),
    ):
        _coerce_int(value, label, errors)
    if train.get('max_steps_per_epoch') is not None:
        _coerce_int(train.get('max_steps_per_epoch'), 'train.max_steps_per_epoch', errors)
    for label, value in (
        ('optimizer.lr_conv1', optim.get('lr_conv1', optim.get('lr', 0.0))),
        ('optimizer.lr_conv', optim.get('lr_conv', optim.get('lr', 0.0))),
        ('optimizer.lr_fc', optim.get('lr_fc', optim.get('lr', 0.0))),
        ('optimizer.momentum', optim.get('momentum', 0.0)),
        ('optimizer.weight_decay', optim.get('weight_decay', 0.0)),
        ('optimizer.grad_clip_global', optim.get('grad_clip_global', 0.0)),
    ):
        _coerce_float(value, label, errors)
    cuda_variant = runtime.get('cuda_variant')
    if cuda_variant is not None and str(cuda_variant) not in {'default', 'cublas', 'handmade', 'nocublas'}:
        errors.append('runtime.cuda_variant must be one of: default, cublas, handmade, nocublas')

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
            slope = _coerce_float(act.get('negative_slope', 0.1), 'LeakyReLU.negative_slope', errors)
            if slope is None:
                continue
        else:
            errors.append(f'Unsupported activation for cuda_legacy: {act_type}')
            continue
        if negative_slope is None:
            negative_slope = slope
        elif abs(negative_slope - slope) > 1e-12:
            errors.append('cuda_legacy requires the same activation slope in all conv blocks')

    conv_out_channels: list[int | None] = []
    for idx, conv in enumerate(convs, start=1):
        out_channels = _coerce_int(conv.get('out_channels'), f'Conv{idx}.out_channels', errors)
        conv_out_channels.append(out_channels)
        kernel_size = _coerce_int(conv.get('kernel_size', 3), f'Conv{idx}.kernel_size', errors)
        stride = _coerce_int(conv.get('stride', 1), f'Conv{idx}.stride', errors)
        padding = _coerce_int(conv.get('padding', 0), f'Conv{idx}.padding', errors)
        if kernel_size is not None and kernel_size != 3:
            errors.append(f'Conv{idx} must use kernel_size=3 for cuda_legacy')
        if stride is not None and stride != 1:
            errors.append(f'Conv{idx} must use stride=1 for cuda_legacy')
        if padding is not None and padding != 0:
            errors.append(f'Conv{idx} must use padding=0 for cuda_legacy')
        if idx > 1:
            prev_out = conv_out_channels[idx - 2]
            if prev_out is None:
                continue
            in_ch = _coerce_int(conv.get('in_channels', prev_out), f'Conv{idx}.in_channels', errors)
            if in_ch is not None and in_ch != prev_out:
                errors.append(f'Conv{idx} in_channels must equal previous out_channels ({prev_out})')
    for pool_pos in (4, 9):
        layer = model.get('layers', [])[pool_pos]
        kernel_size = _coerce_int(layer.get('kernel_size', 2), f'Pool layer at position {pool_pos + 1}.kernel_size', errors)
        stride = _coerce_int(layer.get('stride', 2), f'Pool layer at position {pool_pos + 1}.stride', errors)
        if kernel_size is not None and stride is not None and (kernel_size != 2 or stride != 2):
            errors.append(f'Pool layer at position {pool_pos + 1} must use kernel_size=2 and stride=2')
    out_features = _coerce_int(linear.get('out_features', 10), 'Final Linear out_features', errors)
    if out_features is not None and out_features != 10:
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
    loss = cfg.get('loss', {})
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
    exp.train.init_seed = int(train.get('init_seed', exp.train.init_seed))
    exp.train.train_seed = int(train.get('seed', train.get('train_seed', exp.train.train_seed)))
    exp.train.n_train = int(dataset.get('num_samples', exp.train.n_train))
    exp.train.n_val = int(dataset.get('val_samples', exp.train.n_val))
    exp.train.max_steps_per_epoch = train.get('max_steps_per_epoch', exp.train.max_steps_per_epoch)

    exp.optim.optimizer_type = str(optim.get('type', 'SGD')).lower()
    exp.optim.lr_conv1 = float(optim.get('lr_conv1', optim.get('lr', exp.optim.lr_conv1)))
    exp.optim.lr_conv = float(optim.get('lr_conv', optim.get('lr', exp.optim.lr_conv)))
    exp.optim.lr_fc = float(optim.get('lr_fc', optim.get('lr', exp.optim.lr_fc)))
    exp.optim.momentum = float(optim.get('momentum', exp.optim.momentum))
    exp.optim.weight_decay = float(optim.get('weight_decay', exp.optim.weight_decay))
    exp.optim.adam_beta1 = float(optim.get('beta1', optim.get('adam_beta1', exp.optim.adam_beta1)))
    exp.optim.adam_beta2 = float(optim.get('beta2', optim.get('adam_beta2', exp.optim.adam_beta2)))
    exp.optim.adam_eps = float(optim.get('eps', optim.get('adam_eps', exp.optim.adam_eps)))
    exp.optim.grad_clip_global = float(optim.get('grad_clip_global', exp.optim.grad_clip_global))
    exp.optim.leaky_alpha = negative_slope

    _LOSS_MAP = {
        'CrossEntropyLoss': 'cross_entropy',
        'MSELoss': 'mse',
        'BCEWithLogitsLoss': 'bce',
    }
    exp.loss.loss_type = _LOSS_MAP.get(str(loss.get('type', 'CrossEntropyLoss')), 'cross_entropy')

    input_shape = dataset.get('input_shape', [3, 32, 32])
    exp.model.c_in = int(input_shape[0])
    exp.model.h = int(input_shape[1])
    exp.model.w = int(input_shape[2])
    exp.model.kh = 3
    exp.model.kw = 3
    # Build conv_layers from the unified config's Conv2d layers.
    # Fixed cuda_legacy pattern: conv1(no-pool), conv2(pool), conv3(no-pool), conv4(pool).
    # layer_norm/batch_norm are passed through if present on the conv spec.
    _pool_flags = [False, True, False, True]
    exp.model.conv_layers = []
    for conv, pool in zip(convs, _pool_flags):
        entry: dict = {'out_c': int(conv['out_channels']), 'pool': pool}
        if conv.get('layer_norm'):
            entry['layer_norm'] = True
        if conv.get('batch_norm'):
            entry['batch_norm'] = True
        exp.model.conv_layers.append(entry)

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
            'grad_clip_global': data['optim']['grad_clip_global'],
            'leaky_alpha': data['optim']['leaky_alpha'],
        },
        'train': {
            'batch_size': data['train']['batch_size'],
            'epochs': data['train']['epochs'],
            'n_train': data['train']['n_train'],
            'n_val': data['train']['n_val'],
        },
        'runtime': {
            'cuda_variant': cfg.get('runtime', {}).get('cuda_variant'),
            'cuda_so': cfg.get('runtime', {}).get('cuda_so'),
        },
    }
