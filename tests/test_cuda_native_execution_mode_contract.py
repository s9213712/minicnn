from __future__ import annotations

import json
import warnings
from pathlib import Path


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


def test_validate_cuda_native_config_rejects_planned_gpu_native_mode(tmp_path, capsys):
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

    assert rc == 2
    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['effective_execution_mode'] == 'unsupported'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is False
    assert any('planned but not yet implemented' in err for err in payload['errors'])


def test_train_native_preamble_reports_requested_gpu_native_mode_before_failure(tmp_path, capsys):
    from minicnn.cli import main
    import pytest

    config_path = tmp_path / 'cfg.yaml'
    _write_cfg(config_path)

    with pytest.raises(SystemExit) as excinfo:
        main(['train-native', '--config', str(config_path), 'engine.execution_mode=gpu_native'])

    assert excinfo.value.code == 2
    stdout = capsys.readouterr().out
    payload, _ = json.JSONDecoder().raw_decode(stdout.strip())

    assert payload['selected_execution_mode'] == 'gpu_native'
    assert payload['effective_execution_mode'] == 'unsupported'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is False
