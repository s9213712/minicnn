from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest


def _write_cfg(path: Path) -> None:
    path.write_text(
        """engine:
  backend: cuda_native
dataset:
  type: random
  input_shape: [1, 8, 8]
  num_classes: 2
  num_samples: 8
  val_samples: 4
  seed: 11
model:
  layers:
    - type: Flatten
    - type: Linear
      out_features: 2
train:
  batch_size: 2
  epochs: 1
  init_seed: 11
optimizer:
  type: SGD
  lr: 0.01
loss:
  type: CrossEntropyLoss
project:
  artifacts_root: REPLACE_ARTIFACTS
""".replace('REPLACE_ARTIFACTS', str(path.parent / 'artifacts')),
        encoding='utf-8',
    )


def _run_train_native_or_skip(args: list[str]) -> int:
    from minicnn.cli import main

    try:
        return main(args)
    except RuntimeError as exc:
        if 'CUDA runtime preflight failed' in str(exc):
            pytest.skip(str(exc))
        raise


def test_validate_cuda_native_config_reports_reference_numpy_mode(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    rc = main(['validate-cuda-native-config', '--config', str(config_path), '--format', 'json'])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload['execution_mode'] == 'reference_numpy'
    assert payload['effective_execution_mode'] == 'reference_numpy'
    assert payload['tensor_execution_device'] == 'cpu'
    assert payload['tensors_ran_on'] == 'cpu'
    assert payload['execution_readiness_assessment']['selected_execution_mode'] == 'reference_numpy'
    assert payload['execution_readiness_assessment']['ready'] is True


def test_train_native_preamble_reports_reference_numpy_mode(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rc = main(['train-native', '--config', str(config_path)])
    stdout = capsys.readouterr().out
    json_text = stdout.split('Artifacts written to:')[0].strip()
    payload, end = json.JSONDecoder().raw_decode(json_text)

    assert rc == 0
    assert payload['status'] == 'beta'
    assert payload['execution_mode'] == 'reference_numpy'
    assert payload['effective_execution_mode'] == 'reference_numpy'
    assert payload['tensor_execution_device'] == 'cpu'
    assert payload['tensors_ran_on'] == 'cpu'
    assert payload['gpu_execution'] is False


def test_validate_cuda_native_config_accepts_gpu_native_linear_training_subset(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['effective_execution_mode'] == 'gpu_native'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is True
    assert payload['execution_readiness_assessment']['selected_execution_mode'] == 'gpu_native'
    assert payload['execution_readiness_assessment']['status'] == 'bootstrap_training_partial'
    assert payload['execution_readiness_assessment']['ready'] is True
    assert payload['execution_readiness_assessment']['bootstrap_subset_complete'] is True
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Flatten', 'Linear']
    assert payload['execution_readiness_assessment']['bootstrap_missing_ops'] == []
    assert payload['execution_readiness_assessment']['kernel_readiness_for_requested_ops']['Flatten']['forward_status'] == 'native_alias'
    assert payload['execution_readiness_assessment']['kernel_readiness_for_requested_ops']['Linear']['backward_status'] == 'partial_native'
    assert payload['execution_readiness_assessment']['dispatch_plan']['ready'] is True
    assert payload['execution_readiness_assessment']['dispatch_plan']['num_steps'] == 2
    assert payload['execution_readiness_assessment']['training_lowering_plan']['ready'] is True
    assert payload['execution_readiness_assessment']['training_lowering_plan']['subset_name'] == 'flatten_linear'
    assert payload['execution_readiness_assessment']['training_lowering_plan']['optimizer_steps'][0]['lowering_kind'] == 'apply_momentum_update'
    assert 'gpu_composite_block_training_pending' in payload['execution_readiness_assessment']['remaining_blockers']
    assert payload['errors'] == []


def test_train_native_runs_gpu_native_linear_training_subset(tmp_path, capsys):
    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rc = _run_train_native_or_skip(['train-native', '--config', str(config_path), 'engine.execution_mode=gpu_native'])
    assert rc == 0
    stdout = capsys.readouterr().out
    json_text = stdout.split('Artifacts written to:')[0].strip()
    payload, _ = json.JSONDecoder().raw_decode(json_text)

    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['effective_execution_mode'] == 'gpu_native'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is True
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Flatten', 'Linear']


def test_train_native_runs_gpu_native_two_linear_relu_training_subset(tmp_path, capsys):
    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)
    config_path.write_text(
        config_path.read_text(encoding='utf-8').replace(
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 4\n    - type: ReLU\n    - type: Linear\n      out_features: 2\n",
        ),
        encoding='utf-8',
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rc = _run_train_native_or_skip(['train-native', '--config', str(config_path), 'engine.execution_mode=gpu_native'])
    assert rc == 0
    stdout = capsys.readouterr().out
    json_text = stdout.split('Artifacts written to:')[0].strip()
    payload, _ = json.JSONDecoder().raw_decode(json_text)

    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['effective_execution_mode'] == 'gpu_native'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is True
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Flatten', 'Linear', 'ReLU']


def test_train_native_runs_gpu_native_pool_linear_training_subset(tmp_path, capsys):
    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)
    config_path.write_text(
        config_path.read_text(encoding='utf-8').replace(
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
            "  layers:\n    - type: MaxPool2d\n      kernel_size: 2\n      stride: 2\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
        ),
        encoding='utf-8',
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rc = _run_train_native_or_skip(['train-native', '--config', str(config_path), 'engine.execution_mode=gpu_native'])
    assert rc == 0
    stdout = capsys.readouterr().out
    json_text = stdout.split('Artifacts written to:')[0].strip()
    payload, _ = json.JSONDecoder().raw_decode(json_text)

    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['effective_execution_mode'] == 'gpu_native'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is True
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Flatten', 'Linear', 'MaxPool2d']


def test_validate_cuda_native_config_accepts_gpu_native_conv_relu_linear_training_subset(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)
    config_path.write_text(
        config_path.read_text(encoding='utf-8').replace(
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
            "  layers:\n    - type: Conv2d\n      out_channels: 2\n      kernel_size: 3\n      stride: 1\n      padding: 0\n      bias: false\n    - type: ReLU\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
        ),
        encoding='utf-8',
    )

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Conv2d', 'Flatten', 'Linear', 'ReLU']
    assert payload['execution_readiness_assessment']['dispatch_plan']['ready'] is True
    assert payload['execution_readiness_assessment']['dispatch_plan']['num_steps'] == 4
    assert payload['errors'] == []


def test_validate_cuda_native_config_accepts_gpu_native_global_avgpool_linear_training_subset(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)
    config_path.write_text(
        config_path.read_text(encoding='utf-8').replace(
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
            "  layers:\n    - type: GlobalAvgPool2d\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
        ),
        encoding='utf-8',
    )

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['execution_readiness_assessment']['training_lowering_plan']['subset_name'] == 'global_avgpool_linear'
    assert payload['execution_readiness_assessment']['training_lowering_plan']['helper'] == 'native_gpu_global_avgpool_linear_training_step'
    assert payload['errors'] == []


def test_validate_cuda_native_config_accepts_gpu_native_conv_relu_pool_linear_training_subset(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)
    config_path.write_text(
        config_path.read_text(encoding='utf-8').replace(
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
            "  layers:\n    - type: Conv2d\n      out_channels: 2\n      kernel_size: 3\n      stride: 1\n      padding: 0\n      bias: false\n    - type: ReLU\n    - type: MaxPool2d\n      kernel_size: 2\n      stride: 2\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
        ),
        encoding='utf-8',
    )

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Conv2d', 'Flatten', 'Linear', 'MaxPool2d', 'ReLU']
    assert payload['execution_readiness_assessment']['dispatch_plan']['ready'] is True
    assert payload['execution_readiness_assessment']['dispatch_plan']['num_steps'] == 5
    assert payload['errors'] == []


def test_validate_cuda_native_config_accepts_gpu_native_two_conv_relu_pool_linear_training_subset(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)
    config_path.write_text(
        config_path.read_text(encoding='utf-8').replace(
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
            "  layers:\n    - type: Conv2d\n      out_channels: 2\n      kernel_size: 2\n      stride: 1\n      padding: 0\n      bias: false\n    - type: ReLU\n    - type: Conv2d\n      out_channels: 2\n      kernel_size: 2\n      stride: 1\n      padding: 0\n      bias: false\n    - type: ReLU\n    - type: MaxPool2d\n      kernel_size: 2\n      stride: 2\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
        ),
        encoding='utf-8',
    )

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Conv2d', 'Flatten', 'Linear', 'MaxPool2d', 'ReLU']
    assert payload['execution_readiness_assessment']['dispatch_plan']['ready'] is True
    assert payload['execution_readiness_assessment']['dispatch_plan']['num_steps'] == 7
    assert payload['errors'] == []


def test_validate_cuda_native_config_reports_ops_outside_gpu_bootstrap_subset(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)
    config_path.write_text(
        config_path.read_text(encoding='utf-8').replace(
            "  layers:\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
            "  layers:\n    - type: Dropout\n      p: 0.1\n    - type: Flatten\n    - type: Linear\n      out_features: 2\n",
        ),
        encoding='utf-8',
    )

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert payload['execution_readiness_assessment']['bootstrap_supported_ops'] == ['Flatten', 'Linear']
    assert payload['execution_readiness_assessment']['bootstrap_missing_ops'] == ['Dropout']
    assert payload['execution_readiness_assessment']['kernel_readiness_for_requested_ops']['Dropout']['forward_status'] == 'outside_bootstrap'
    assert payload['execution_readiness_assessment']['dispatch_plan']['ready'] is False
    assert payload['execution_readiness_assessment']['dispatch_plan']['unsupported_ops'] == ['Dropout']
    assert payload['execution_readiness_assessment']['dispatch_plan']['steps'][0]['op_name'] == 'Dropout'
    assert payload['execution_readiness_assessment']['dispatch_plan']['steps'][0]['supported'] is False
    assert any("got ['Dropout', 'Flatten', 'Linear']" in err for err in payload['errors'])


def test_validate_cuda_native_config_accepts_gpu_native_linear_global_grad_clip(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
        'optimizer.grad_clip_global=1.0',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['execution_readiness_assessment']['training_lowering_plan']['optimizer_steps'][0]['lowering_kind'] == 'grad_l2_sumsq_scale'
    assert payload['errors'] == []


def test_validate_cuda_native_config_accepts_gpu_native_linear_grad_accum(tmp_path, capsys):
    from minicnn.cli import main

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    rc = main([
        'validate-cuda-native-config',
        '--config',
        str(config_path),
        '--format',
        'json',
        'engine.execution_mode=gpu_native',
        'train.grad_accum_steps=2',
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['execution_readiness_assessment']['training_lowering_plan']['ready'] is True
    assert payload['errors'] == []


def test_train_native_runs_gpu_native_linear_grad_accum(tmp_path, capsys):
    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rc = _run_train_native_or_skip([
            'train-native',
            '--config',
            str(config_path),
            'engine.execution_mode=gpu_native',
            'train.grad_accum_steps=2',
        ])
    assert rc == 0
    stdout = capsys.readouterr().out
    json_text = stdout.split('Artifacts written to:')[0].strip()
    payload, _ = json.JSONDecoder().raw_decode(json_text)

    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['effective_execution_mode'] == 'gpu_native'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is True


def test_execution_mode_sets_are_disjoint():
    from minicnn.cuda_native.api import _PLANNED_EXECUTION_MODES, _SUPPORTED_EXECUTION_MODES

    overlap = _SUPPORTED_EXECUTION_MODES & _PLANNED_EXECUTION_MODES
    assert overlap == set(), f'Modes in both sets: {sorted(overlap)}'
