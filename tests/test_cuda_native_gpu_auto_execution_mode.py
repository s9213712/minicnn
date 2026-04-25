from __future__ import annotations


def _cfg(layers: list[dict[str, object]]) -> dict[str, object]:
    return {
        'engine': {'backend': 'cuda_native', 'execution_mode': 'gpu_native_auto'},
        'dataset': {'type': 'random', 'input_shape': [1, 4, 4], 'num_classes': 2},
        'model': {'layers': layers},
        'loss': {'type': 'CrossEntropyLoss'},
        'optimizer': {'type': 'SGD'},
        'train': {'amp': False},
    }


def test_gpu_native_auto_selects_gpu_when_lowering_and_runtime_are_ready(monkeypatch):
    import minicnn.cuda_native.api as api

    monkeypatch.setattr(api, '_cuda_runtime_ready_for_gpu_native', lambda: (True, 'not_needed'))
    payload = api.resolve_cuda_native_execution_mode(_cfg([
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 2},
    ]))

    assert payload['selected_execution_mode'] == 'gpu_native_auto'
    assert payload['effective_execution_mode'] == 'gpu_native'
    assert payload['tensor_execution_device'] == 'gpu'
    assert payload['gpu_execution'] is True
    assert payload['fallback_available'] is True
    assert payload['fallback_active'] is False
    assert payload['fallback_reason'] == 'not_needed'


def test_gpu_native_auto_falls_back_to_numpy_when_lowering_is_not_ready(monkeypatch):
    import minicnn.cuda_native.api as api

    monkeypatch.setattr(api, '_cuda_runtime_ready_for_gpu_native', lambda: (True, 'not_needed'))
    cfg = _cfg([
        {'type': 'Dropout', 'p': 0.5},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 2},
    ])
    payload = api.resolve_cuda_native_execution_mode(cfg)
    readiness = api.assess_cuda_native_execution_readiness(cfg)

    assert payload['selected_execution_mode'] == 'gpu_native_auto'
    assert payload['effective_execution_mode'] == 'reference_numpy'
    assert payload['tensor_execution_device'] == 'cpu'
    assert payload['gpu_execution'] is False
    assert payload['fallback_active'] is True
    assert 'unsupported gpu_native training subset' in payload['fallback_reason']
    assert readiness['bootstrap_subset_complete'] is False
    assert readiness['bootstrap_missing_ops'] == ['Dropout']
    assert readiness['kernel_readiness_for_requested_ops']['Dropout']['forward_status'] == 'outside_bootstrap'


def test_gpu_native_auto_falls_back_to_numpy_when_runtime_is_not_ready(monkeypatch):
    import minicnn.cuda_native.api as api

    monkeypatch.setattr(api, '_cuda_runtime_ready_for_gpu_native', lambda: (False, 'cuda_runtime_not_ready'))
    payload = api.resolve_cuda_native_execution_mode(_cfg([
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 2},
    ]))

    assert payload['selected_execution_mode'] == 'gpu_native_auto'
    assert payload['effective_execution_mode'] == 'reference_numpy'
    assert payload['gpu_native_lowering_ready'] is True
    assert payload['gpu_native_runtime_ready'] is False
    assert payload['fallback_active'] is True
    assert payload['fallback_reason'] == 'cuda_runtime_not_ready'


def test_train_native_gpu_native_auto_runtime_fallback_preamble(tmp_path, capsys, monkeypatch):
    import json

    import minicnn.cuda_native.api as api
    from minicnn.cli import main

    monkeypatch.setattr(api, '_cuda_runtime_ready_for_gpu_native', lambda: (False, 'cuda_runtime_not_ready'))
    config_path = tmp_path / 'cfg.yaml'
    config_path.write_text(
        """engine:
  backend: cuda_native
  execution_mode: gpu_native_auto
dataset:
  type: random
  input_shape: [1, 4, 4]
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
""".replace('REPLACE_ARTIFACTS', str(tmp_path / 'artifacts')),
        encoding='utf-8',
    )

    rc = main(['train-native', '--config', str(config_path)])
    stdout = capsys.readouterr().out
    json_text = stdout.split('Artifacts written to:')[0].strip()
    payload, _ = json.JSONDecoder().raw_decode(json_text)

    assert rc == 0
    assert payload['selected_execution_mode'] == 'gpu_native_auto'
    assert payload['effective_execution_mode'] == 'reference_numpy'
    assert payload['tensor_execution_device'] == 'cpu'
    assert payload['gpu_execution'] is False
    assert payload['fallback_active'] is True
    assert payload['fallback_reason'] == 'cuda_runtime_not_ready'
