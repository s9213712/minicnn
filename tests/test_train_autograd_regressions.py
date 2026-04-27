from __future__ import annotations

import json
import numpy as np

import minicnn.training.train_autograd as trainer


def _base_cfg(tmp_path, *, batch_size: int, grad_accum_steps: int) -> dict[str, object]:
    return {
        'project': {'name': 'test', 'run_name': f'autograd-{batch_size}-{grad_accum_steps}', 'artifacts_root': str(tmp_path)},
        'dataset': {
            'type': 'random',
            'input_shape': [1, 4, 4],
            'num_classes': 2,
            'num_samples': 4,
            'val_samples': 2,
            'test_samples': 2,
            'seed': 7,
        },
        'model': {'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 2},
        ]},
        'optimizer': {'type': 'SGD', 'lr': 0.05},
        'loss': {'type': 'CrossEntropyLoss'},
        'train': {
            'epochs': 1,
            'batch_size': batch_size,
            'grad_accum_steps': grad_accum_steps,
            'train_seed': 11,
            'init_seed': 13,
        },
    }


def test_train_autograd_grad_accumulation_flushes_final_partial_window(tmp_path, monkeypatch):
    steps = []

    class CountingSGD(trainer.SGD):
        def step(self):
            steps.append(1)
            return super().step()

    monkeypatch.setattr(
        trainer,
        '_make_optimizer',
        lambda params, _cfg: CountingSGD(params, lr=0.01),
    )
    cfg = _base_cfg(tmp_path, batch_size=2, grad_accum_steps=4)
    cfg['dataset']['num_samples'] = 5

    trainer.train_autograd_from_config(cfg)

    assert len(steps) == 1


def test_train_autograd_grad_accumulation_matches_larger_batch_update(tmp_path):
    accum_run = trainer.train_autograd_from_config(_base_cfg(tmp_path / 'accum', batch_size=1, grad_accum_steps=2))
    batch_run = trainer.train_autograd_from_config(_base_cfg(tmp_path / 'batch', batch_size=2, grad_accum_steps=1))

    accum_state = np.load(trainer.resolve_autograd_artifacts(accum_run)[1])
    batch_state = np.load(trainer.resolve_autograd_artifacts(batch_run)[1])

    assert accum_state.files == batch_state.files
    for key in accum_state.files:
        assert np.allclose(accum_state[key], batch_state[key], atol=1e-6), key


def test_train_autograd_plateau_uses_val_loss_and_emits_correct_val_metrics(tmp_path, monkeypatch):
    scheduler_metrics: list[float] = []
    event_payloads: list[dict[str, object]] = []

    class RecordingPlateau:
        def __init__(self, optimizer, **_kwargs):
            self.optimizer = optimizer

        def step(self, metric=None):
            scheduler_metrics.append(float(metric))
            return [self.optimizer.lr]

    monkeypatch.setattr(trainer, 'ReduceLROnPlateau', RecordingPlateau)
    monkeypatch.setattr(
        trainer,
        '_eval_metrics',
        lambda *_args, **_kwargs: {'loss': 0.75, 'acc': 0.5},
    )
    monkeypatch.setattr(
        trainer,
        'emit_training_event',
        lambda event, payload, **_kwargs: event_payloads.append({'event': event, 'payload': payload}) or '',
    )

    cfg = _base_cfg(tmp_path, batch_size=2, grad_accum_steps=1)
    cfg['scheduler'] = {'enabled': True, 'type': 'plateau', 'factor': 0.5, 'patience': 1}
    run_dir = trainer.train_autograd_from_config(cfg)

    assert scheduler_metrics == [0.75]
    epoch_payload = next(item['payload'] for item in event_payloads if item['event'] == 'epoch_summary')
    assert epoch_payload['val_metrics']['loss'] == 0.75
    assert epoch_payload['val_metrics']['loss'] != epoch_payload['train_metrics']['loss']

    row = json.loads((run_dir / 'metrics.jsonl').read_text(encoding='utf-8').strip().splitlines()[-1])
    assert row['val_loss'] == 0.75
