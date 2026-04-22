"""CLI, build, checkpoint, and runtime regression tests."""

import os
import numpy as np
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'


def _run_python_without_torch(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run_python_with_sitecustomize(
        tmp_path,
        (
            'import importlib.abc\n'
            'import sys\n'
            'class _BlockTorch(importlib.abc.MetaPathFinder):\n'
            '    def find_spec(self, fullname, path=None, target=None):\n'
            "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
            "            raise ModuleNotFoundError(\"No module named 'torch'\", name='torch')\n"
            '        return None\n'
            'sys.meta_path.insert(0, _BlockTorch())\n'
        ),
        *args,
    )


def _run_python_with_broken_torch(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run_python_with_sitecustomize(
        tmp_path,
        (
            'import importlib.abc\n'
            'import sys\n'
            'class _BrokenTorch(importlib.abc.MetaPathFinder):\n'
            '    def find_spec(self, fullname, path=None, target=None):\n'
            "        if fullname == 'torch':\n"
            "            raise ImportError('broken torch install for test')\n"
            '        return None\n'
            'sys.meta_path.insert(0, _BrokenTorch())\n'
        ),
        *args,
    )


def _run_python_with_sitecustomize(
    tmp_path: Path,
    source: str,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    sitecustomize = tmp_path / 'sitecustomize.py'
    sitecustomize.write_text(source, encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(tmp_path), str(SRC_ROOT)])
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_plateau_scheduler_reduces_after_configured_patience():
    from minicnn.schedulers.plateau import ReduceLROnPlateau

    class Optimizer:
        lr = 1.0

    optimizer = Optimizer()
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=0.01)

    scheduler.step(1.0)
    scheduler.step(1.0)

    assert optimizer.lr == 0.5


def test_init_weights_does_not_consume_global_numpy_rng():
    from minicnn.models.initialization import init_weights

    np.random.seed(123)
    np.random.random()
    init_weights(1)
    actual = np.random.random()

    np.random.seed(123)
    np.random.random()
    expected = np.random.random()

    assert actual == expected


def test_torch_baseline_evaluate_does_not_force_training_mode():
    import torch

    from minicnn.training.train_torch_baseline import evaluate

    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 2), device=x.device)

    model = ToyModel()
    model.eval()
    x = np.zeros((4, 3, 32, 32), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int64)

    evaluate(model, x, y, torch.device('cpu'), batch_size=2, max_batches=1)

    assert model.training is False


def test_evaluate_helpers_return_zero_for_empty_work():
    import torch

    from minicnn.training.evaluation import evaluate as cuda_evaluate
    from minicnn.training.train_torch_baseline import evaluate as torch_evaluate

    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 2), device=x.device)

    x = np.zeros((0, 3, 32, 32), dtype=np.float32)
    y = np.zeros((0,), dtype=np.int64)

    assert cuda_evaluate(x, y, device_weights=(), max_batches=0) == 0.0
    assert torch_evaluate(ToyModel(), x, y, torch.device('cpu'), max_batches=0) == 0.0


def test_shared_scalar_parser_handles_nested_lists_for_legacy_config():
    from minicnn.config.loader import load_config

    cfg = load_config(None, ['train.train_batch_ids=[1, 2, 3]'])

    assert cfg.train.train_batch_ids == [1, 2, 3]


def test_cli_reports_missing_train_flex_config_without_traceback():
    from minicnn.cli import main

    try:
        main(['train-flex', '--config', '/tmp/minicnn-definitely-missing.yaml'])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError('expected SystemExit for missing config')


def test_cli_exposes_doctor_compare_and_backend_aliases():
    from minicnn.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    subparsers = next(action for action in parser._actions if getattr(action, 'choices', None))
    build_help = subparsers.choices['build'].format_help()

    assert 'doctor' in help_text
    assert 'smoke' in help_text
    assert 'compare' in help_text
    assert 'train-cuda' in help_text
    assert 'train-torch' in help_text
    assert 'train-autograd' in help_text
    assert 'inspect-checkpoint' in help_text
    assert 'export-torch-checkpoint' in help_text
    assert 'validate-config' in help_text
    assert 'compile' in help_text
    assert '--cuda-arch' in build_help


def test_cli_config_resolution_falls_back_to_project_root():
    from minicnn.cli import _resolve_cli_config_path

    resolved = _resolve_cli_config_path('configs/flex_cnn.yaml')

    assert resolved is not None
    assert Path(resolved).exists()
    assert resolved.endswith('configs/flex_cnn.yaml')


def test_cli_smoke_returns_structured_json(capsys):
    import json

    from minicnn.cli import main

    rc = main(['smoke'])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload['ok'] is True
    assert isinstance(payload['checks'], list)
    assert any(check['name'] == 'compiler_trace' for check in payload['checks'])
    assert all('severity' in check for check in payload['checks'])


def test_cli_healthcheck_returns_structured_json(capsys):
    import json

    from minicnn.cli import main

    rc = main(['healthcheck'])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert isinstance(payload['checks'], list)
    assert 'flex_registries' in payload


def test_cli_help_still_works_without_torch(tmp_path):
    proc = _run_python_without_torch(tmp_path, '-m', 'minicnn.cli', '--help')

    assert proc.returncode == 0
    assert 'train-flex' in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_unified_cuda_legacy_import_still_works_without_torch(tmp_path):
    proc = _run_python_without_torch(
        tmp_path,
        '-c',
        'from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility; print("ok")',
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == 'ok'
    assert 'Traceback' not in proc.stderr


def test_cli_reports_missing_torch_for_train_flex_without_traceback(tmp_path):
    proc = _run_python_without_torch(
        tmp_path,
        '-m',
        'minicnn.cli',
        'train-flex',
        '--config',
        'configs/flex_cnn.yaml',
    )

    assert proc.returncode == 2
    assert 'train-flex requires PyTorch.' in proc.stdout
    assert 'pip install -e .[torch]' in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_cli_reports_missing_torch_for_train_dual_torch_without_traceback(tmp_path):
    proc = _run_python_without_torch(
        tmp_path,
        '-m',
        'minicnn.cli',
        'train-dual',
        '--config',
        'configs/dual_backend_cnn.yaml',
        'engine.backend=torch',
    )

    assert proc.returncode == 2
    assert 'train-dual with engine.backend=torch requires PyTorch.' in proc.stdout
    assert 'pip install -e .[torch]' in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_cli_reports_broken_torch_import_without_traceback(tmp_path):
    proc = _run_python_with_broken_torch(
        tmp_path,
        '-m',
        'minicnn.cli',
        'train-flex',
        '--config',
        'configs/flex_cnn.yaml',
    )

    assert proc.returncode == 2
    assert 'could not import PyTorch from this environment' in proc.stdout
    assert 'broken torch install for test' in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_cli_reports_invalid_override_without_traceback(tmp_path):
    proc = _run_python_with_sitecustomize(
        tmp_path,
        '',
        '-m',
        'minicnn.cli',
        'validate-dual-config',
        'model.layers.foo=1',
    )

    assert proc.returncode == 2
    assert 'Invalid config override' in proc.stdout
    assert 'model.layers.foo=1' in proc.stdout
    assert 'must end with a numeric list index' in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_cli_reports_bad_yaml_without_traceback(tmp_path):
    bad_cfg = tmp_path / 'bad.yaml'
    bad_cfg.write_text('dataset: [1, 2\n', encoding='utf-8')

    proc = _run_python_with_sitecustomize(
        tmp_path,
        '',
        '-m',
        'minicnn.cli',
        'validate-dual-config',
        '--config',
        str(bad_cfg),
    )

    assert proc.returncode == 2
    assert 'Failed to parse config file' in proc.stdout
    assert str(bad_cfg) in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_cli_reports_non_mapping_top_level_config_without_traceback(tmp_path):
    bad_cfg = tmp_path / 'bad-top-level.yaml'
    bad_cfg.write_text('- just\n- a\n- list\n', encoding='utf-8')

    proc = _run_python_with_sitecustomize(
        tmp_path,
        '',
        '-m',
        'minicnn.cli',
        'validate-dual-config',
        '--config',
        str(bad_cfg),
    )

    assert proc.returncode == 2
    assert 'Failed to parse config file' in proc.stdout
    assert 'mapping at the top level' in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_cli_reports_cuda_device_unavailable_without_traceback(monkeypatch, tmp_path):
    from minicnn.cli import main
    import minicnn.flex.trainer as trainer

    config_path = tmp_path / 'flex_cpu_only.yaml'
    config_path.write_text(
        'dataset:\n'
        '  type: random\n'
        '  num_samples: 8\n'
        '  val_samples: 4\n'
        '  num_classes: 2\n'
        '  input_shape: [2]\n'
        'model:\n'
        '  layers:\n'
        '    - type: Linear\n'
        '      in_features: 2\n'
        '      out_features: 1\n'
        'loss:\n'
        '  type: BCEWithLogitsLoss\n'
        'train:\n'
        '  epochs: 1\n'
        '  batch_size: 4\n'
        '  device: cuda\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', lambda: False)

    try:
        main(['train-flex', '--config', str(config_path)])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError('expected SystemExit for unavailable CUDA runtime')


def test_choose_device_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    import minicnn.flex.trainer as trainer

    monkeypatch.setattr(trainer.torch.cuda, 'is_available', lambda: False)

    device = trainer._choose_device('auto')

    assert device.type == 'cpu'


def test_create_run_dir_is_unique_even_when_called_back_to_back(tmp_path):
    from minicnn.flex.runtime import create_run_dir

    cfg = {'project': {'artifacts_root': str(tmp_path), 'run_name': 'collision-test'}}

    first = create_run_dir(cfg)
    second = create_run_dir(cfg)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_cli_inspect_checkpoint_reports_npz_schema(capsys, tmp_path):
    import json

    from minicnn.cli import main

    ckpt = tmp_path / 'demo_autograd_best.npz'
    np.savez(
        ckpt,
        layer0_weight=np.zeros((4, 4), dtype=np.float32),
        layer0_bias=np.zeros((4,), dtype=np.float32),
    )

    rc = main(['inspect-checkpoint', '--path', str(ckpt)])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload['format'] == 'npz'
    assert payload['kind'] in {'autograd_state_dict', 'numpy_state_dict'}
    assert payload['num_keys'] == 2
    assert payload['preview']['layer0_weight']['shape'] == [4, 4]


def test_cli_inspect_checkpoint_reports_missing_torch_for_pt_without_traceback(tmp_path):
    ckpt = tmp_path / 'demo.pt'
    ckpt.write_bytes(b'not-a-real-torch-checkpoint')

    proc = _run_python_without_torch(
        tmp_path,
        '-m',
        'minicnn.cli',
        'inspect-checkpoint',
        '--path',
        str(ckpt),
    )

    assert proc.returncode == 2
    assert 'inspect-checkpoint requires PyTorch to read .pt/.pth files.' in proc.stdout
    assert 'pip install -e .[torch]' in proc.stdout
    assert 'Traceback' not in proc.stderr


def test_export_autograd_checkpoint_to_torch_pt(tmp_path):
    import json
    import torch

    from minicnn.cli import main
    from minicnn.models.builder import build_model_from_config

    config_path = tmp_path / 'autograd_export.yaml'
    config_path.write_text(
        'dataset:\n'
        '  input_shape: [3, 8, 8]\n'
        'model:\n'
        '  layers:\n'
        '    - type: Conv2d\n'
        '      out_channels: 4\n'
        '      kernel_size: 3\n'
        '      padding: 1\n'
        '    - type: BatchNorm2d\n'
        '    - type: ReLU\n'
        '    - type: Flatten\n'
        '    - type: Linear\n'
        '      out_features: 2\n',
        encoding='utf-8',
    )

    model, _ = build_model_from_config(
        {
            'layers': [
                {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'padding': 1},
                {'type': 'BatchNorm2d'},
                {'type': 'ReLU'},
                {'type': 'Flatten'},
                {'type': 'Linear', 'out_features': 2},
            ]
        },
        input_shape=(3, 8, 8),
    )
    ckpt_path = tmp_path / 'demo_autograd_best.npz'
    np.savez(ckpt_path, **model.state_dict())

    out_path = tmp_path / 'exported.pt'
    rc = main([
        'export-torch-checkpoint',
        '--path', str(ckpt_path),
        '--config', str(config_path),
        '--output', str(out_path),
    ])

    assert rc == 0
    payload = torch.load(out_path, map_location='cpu', weights_only=True)
    assert payload['source_format'] == 'autograd_state_dict'
    assert tuple(payload['model_state']['4.weight'].shape) == (2, 256)
    assert payload['model_state']['1.running_mean'].shape[0] == 4


def test_export_cuda_native_checkpoint_to_torch_pt(tmp_path):
    import torch

    from minicnn.cli import main

    config_path = tmp_path / 'native_export.yaml'
    config_path.write_text(
        'dataset:\n'
        '  input_shape: [3, 8, 8]\n'
        'model:\n'
        '  layers:\n'
        '    - type: Conv2d\n'
        '      out_channels: 4\n'
        '      kernel_size: 3\n'
        '      padding: 1\n'
        '    - type: BatchNorm2d\n'
        '    - type: ReLU\n'
        '    - type: Flatten\n'
        '    - type: Linear\n'
        '      out_features: 2\n',
        encoding='utf-8',
    )

    ckpt_path = tmp_path / 'demo_native_best.npz'
    np.savez(
        ckpt_path,
        _w_conv2d_0=np.zeros((4, 3, 3, 3), dtype=np.float32),
        _b_conv2d_0=np.zeros((4,), dtype=np.float32),
        _w_batchnorm2d_1=np.ones((4,), dtype=np.float32),
        _b_batchnorm2d_1=np.zeros((4,), dtype=np.float32),
        _running_mean_batchnorm2d_1=np.zeros((4,), dtype=np.float32),
        _running_var_batchnorm2d_1=np.ones((4,), dtype=np.float32),
        _w_linear_4=np.zeros((2, 256), dtype=np.float32),
        _b_linear_4=np.zeros((2,), dtype=np.float32),
    )

    out_path = tmp_path / 'exported_native.pt'
    rc = main([
        'export-torch-checkpoint',
        '--path', str(ckpt_path),
        '--config', str(config_path),
        '--output', str(out_path),
    ])

    assert rc == 0
    payload = torch.load(out_path, map_location='cpu', weights_only=True)
    assert payload['source_format'] == 'cuda_native_param_dict'
    assert tuple(payload['model_state']['4.weight'].shape) == (2, 256)


def test_export_cuda_legacy_checkpoint_is_rejected(capsys, tmp_path):
    from minicnn.cli import main

    config_path = tmp_path / 'legacy_export.yaml'
    config_path.write_text('dataset:\n  input_shape: [3, 32, 32]\nmodel:\n  layers: []\n', encoding='utf-8')
    ckpt_path = tmp_path / 'legacy_best.npz'
    np.savez(
        ckpt_path,
        epoch=np.int32(1),
        val_acc=np.float32(0.1),
        fc_w=np.zeros((10,), dtype=np.float32),
        fc_b=np.zeros((1,), dtype=np.float32),
    )

    try:
        main([
            'export-torch-checkpoint',
            '--path', str(ckpt_path),
            '--config', str(config_path),
            '--output', str(tmp_path / 'nope.pt'),
        ])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError('expected SystemExit for unsupported export')

    out = capsys.readouterr().out
    assert 'cuda_legacy checkpoints cannot be exported directly' in out


def test_cli_seed_overrides_keep_dataset_init_and_train_seeds_separate():
    from minicnn.cli import _common_train_overrides, build_parser

    args = build_parser().parse_args([
        'train-dual',
        '--dataset-seed', '111',
        '--init-seed', '222',
        '--train-seed', '333',
    ])

    overrides = _common_train_overrides(args)

    assert 'dataset.seed=111' in overrides
    assert 'train.init_seed=222' in overrides
    assert 'train.seed=333' in overrides
    assert 'dataset.seed=222' not in overrides


def test_cli_benchmark_fields_read_metrics_for_compare(tmp_path):
    from minicnn.cli import _benchmark_fields

    (tmp_path / 'metrics.jsonl').write_text(
        '{"epoch": 1, "epoch_time_s": 1.0}\n'
        '{"epoch": 2, "epoch_time_s": 2.0}\n',
        encoding='utf-8',
    )
    cfg = {
        'runtime': {'cuda_variant': 'cublas'},
        'dataset': {'num_samples': 128, 'val_samples': 32},
        'train': {'batch_size': 32, 'epochs': 2, 'max_steps_per_epoch': 2},
    }

    fields = _benchmark_fields('cuda_legacy', cfg, tmp_path, elapsed_s=10.0)

    assert fields['variant'] == 'cublas'
    assert fields['train_samples'] == 64
    assert fields['val_samples'] == 32
    assert fields['batch_size'] == 32
    assert fields['epochs_requested'] == 2
    assert fields['epochs_completed'] == 2
    assert fields['avg_epoch_time_s'] == 1.5
    assert fields['last_epoch_time_s'] == 2.0
    assert fields['samples_per_sec'] == 42.667

    torch_fields = _benchmark_fields('torch', cfg | {'train': {'device': 'cpu'}}, tmp_path, elapsed_s=10.0)
    assert torch_fields['variant'] == 'cpu'

    autograd_fields = _benchmark_fields('autograd', cfg | {'train': {'device': 'auto'}}, tmp_path, elapsed_s=10.0)
    assert autograd_fields['variant'] == ''


def test_cli_compare_backends_allow_key_value_overrides_after_backends():
    from minicnn.cli import _compare_backends_and_overrides, build_parser

    args = build_parser().parse_args([
        'compare',
        '--backends', 'torch', 'cuda_legacy',
        'train.epochs=1',
        'dataset.num_samples=8',
    ])

    backends, overrides = _compare_backends_and_overrides(args)

    assert backends == ['torch', 'cuda_legacy']
    assert overrides == ['train.epochs=1', 'dataset.num_samples=8']


def test_build_native_passes_cuda_arch_to_make_and_cmake(monkeypatch):
    from minicnn.core import build

    calls = []
    monkeypatch.setattr(build.subprocess, 'run', lambda cmd, check: calls.append(cmd))

    build.build_native(legacy_make=True, variant='handmade', cuda_arch='sm_75')
    assert 'CUDA_ARCH=sm_75' in calls[-1]

    calls.clear()
    build.build_native(legacy_make=False, variant='handmade', cuda_arch='75')
    assert '-DCMAKE_CUDA_ARCHITECTURES=75' in calls[0]


def test_random_crop_batch_matches_reference_crop():
    from minicnn.training.cuda_epoch import random_crop_batch

    x = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    seed = 123
    padding = 1
    actual = random_crop_batch(x, np.random.default_rng(seed), padding)

    rng = np.random.default_rng(seed)
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='reflect')
    tops = rng.integers(0, 2 * padding + 1, size=x.shape[0])
    lefts = rng.integers(0, 2 * padding + 1, size=x.shape[0])
    expected = np.stack([
        padded[i, :, top:top + x.shape[-2], left:left + x.shape[-1]]
        for i, (top, left) in enumerate(zip(tops, lefts))
    ]).astype(np.float32)

    assert np.array_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_checkpoint_reload_is_transactional(tmp_path, monkeypatch):
    import minicnn.training.checkpoints as checkpoints

    from minicnn.config.settings import get_arch
    geom = get_arch()

    freed = []
    uploaded = []
    old_weights = checkpoints.DeviceWeights(
        ['old1', 'old2', 'old3', 'old4'], 'old5', 'old6'
    )
    ckpt_path = tmp_path / 'weights.npz'
    np.savez(
        ckpt_path,
        n_conv=np.int32(4),
        w_conv1=np.array([1], dtype=np.float32),
        w_conv2=np.array([2], dtype=np.float32),
        w_conv3=np.array([3], dtype=np.float32),
        w_conv4=np.array([4], dtype=np.float32),
        fc_w=np.array([5], dtype=np.float32),
        fc_b=np.array([6], dtype=np.float32),
    )

    def failing_upload(arr):
        if len(uploaded) == 2:
            raise RuntimeError('upload failed')
        ptr = f'new{len(uploaded)}'
        uploaded.append(ptr)
        return ptr

    class Lib:
        @staticmethod
        def gpu_free(ptr):
            freed.append(ptr)

    monkeypatch.setattr(checkpoints, 'upload', failing_upload)
    monkeypatch.setattr(checkpoints, 'lib', Lib())

    try:
        checkpoints.reload_weights_from_checkpoint(ckpt_path, old_weights, geom)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError('expected upload failure')

    assert freed == ['new0', 'new1']

    freed.clear()
    uploaded.clear()
    monkeypatch.setattr(checkpoints, 'upload', lambda arr: uploaded.append(f'new{len(uploaded)}') or uploaded[-1])

    _ckpt, _fc_w, _fc_b, new_weights = checkpoints.reload_weights_from_checkpoint(ckpt_path, old_weights, geom)

    assert isinstance(new_weights, checkpoints.DeviceWeights)
    assert list(new_weights) == ['new0', 'new1', 'new2', 'new3', 'new4', 'new5']
    assert freed == list(old_weights)


def test_upload_weights_is_transactional_on_partial_upload_failure(monkeypatch):
    import minicnn.training.checkpoints as checkpoints

    uploaded = []
    freed = []

    def failing_upload(_arr):
        if len(uploaded) == 3:
            raise RuntimeError('upload failed')
        ptr = f'ptr{len(uploaded)}'
        uploaded.append(ptr)
        return ptr

    class Lib:
        @staticmethod
        def gpu_free(ptr):
            freed.append(ptr)

    monkeypatch.setattr(checkpoints, 'upload', failing_upload)
    monkeypatch.setattr(checkpoints, 'lib', Lib())

    try:
        checkpoints.upload_weights(
            [np.array([1], dtype=np.float32), np.array([2], dtype=np.float32)],
            np.array([3], dtype=np.float32),
            np.array([4], dtype=np.float32),
        )
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError('expected upload failure')

    assert freed == ['ptr0', 'ptr1', 'ptr2']


def test_free_weights_skips_none_and_accepts_none(monkeypatch):
    import minicnn.training.checkpoints as checkpoints

    freed = []

    class Lib:
        @staticmethod
        def gpu_free(ptr):
            freed.append(ptr)

    monkeypatch.setattr(checkpoints, 'lib', Lib())

    checkpoints.free_weights(None)
    checkpoints.free_weights([None, 'p1', None, 'p2'])

    assert freed == ['p1', 'p2']


def test_evaluation_uses_device_weights_container_interface(monkeypatch):
    import minicnn.training.evaluation as evaluation
    from minicnn.config.settings import get_arch
    from minicnn.training.checkpoints import DeviceWeights

    geom = get_arch()
    device_weights = DeviceWeights(
        conv_weights=[f'w_conv{i}' for i in range(geom.n_conv)],
        fc_w='fc_w',
        fc_b='fc_b',
    )

    class Workspace:
        pass

    workspace = Workspace()
    workspace.batch_size = 2
    workspace.d_x = 'x'
    workspace.d_fc_out = 'fc_out'
    workspace.geom = geom
    workspace.d_col = [f'col{i}' for i in range(geom.n_conv)]
    workspace.d_conv_raw = [f'conv_raw{i}' for i in range(geom.n_conv)]
    workspace.d_conv_nchw = [None if s.pool else f'conv_nchw{i}' for i, s in enumerate(geom.conv_stages)]
    workspace.d_pool = [f'pool{i}' if s.pool else None for i, s in enumerate(geom.conv_stages)]
    workspace.d_max_idx = [f'max_idx{i}' if s.pool else None for i, s in enumerate(geom.conv_stages)]
    workspace.d_pool_nchw = [f'pool_nchw{i}' if s.pool else None for i, s in enumerate(geom.conv_stages)]

    conv_weights_seen = []
    dense_args = []

    monkeypatch.setattr(evaluation, 'upload_to', lambda dst, x: None)
    monkeypatch.setattr(
        evaluation,
        'conv_forward_into',
        lambda prev, weight, col, out, n, in_c, h, w, out_c: conv_weights_seen.append(weight),
    )
    monkeypatch.setattr(evaluation, 'maxpool_forward_into', lambda *args: None)
    monkeypatch.setattr(evaluation, 'cnhw_to_nchw_into', lambda *args: None)

    class Lib:
        @staticmethod
        def dense_forward(fc_in, fc_w, fc_b, fc_out, n, in_f, out_f):
            dense_args.extend([fc_w, fc_b, fc_out])

    monkeypatch.setattr(evaluation, 'lib', Lib())

    x = np.zeros((2, 3, 32, 32), dtype=np.float32)

    assert evaluation._forward_logits_into(x, device_weights, workspace) == 'fc_out'
    assert conv_weights_seen == device_weights.conv_weights
    assert dense_args == ['fc_w', 'fc_b', 'fc_out']


def test_workspace_uses_int_allocators_for_index_and_label_buffers(monkeypatch):
    import minicnn.training.cuda_workspace as cuda_workspace

    float_allocs = []
    int_allocs = []
    freed = []

    monkeypatch.setattr(cuda_workspace, 'malloc_floats', lambda size: float_allocs.append(size) or ('float', size))
    monkeypatch.setattr(cuda_workspace, 'malloc_ints', lambda size: int_allocs.append(size) or ('int', size))

    class Lib:
        @staticmethod
        def gpu_free(ptr):
            freed.append(ptr)

    monkeypatch.setattr(cuda_workspace, 'lib', Lib())

    workspace = cuda_workspace.BatchWorkspace()

    # Default arch has pool at stages 1 and 3 (0-indexed)
    pool_stages = [i for i, ptr in enumerate(workspace.d_max_idx) if ptr is not None]
    assert len(pool_stages) >= 2, "Expected at least 2 pool stages in default arch"
    assert workspace.d_max_idx[pool_stages[0]][0] == 'int'
    assert workspace.d_max_idx[pool_stages[1]][0] == 'int'
    assert workspace.d_y[0] == 'int'
    assert workspace.d_correct[0] == 'int'
    assert float_allocs
    assert int_allocs
    workspace.free()
    assert len(freed) == len(float_allocs) + len(int_allocs)


def test_native_cuda_comment_tasks_are_reflected_in_source():
    cpp = REPO_ROOT / 'cpp'
    loss_layer = (cpp / 'src' / 'loss_layer.cu').read_text()
    layer_norm = (cpp / 'src' / 'layer_norm.cu').read_text()
    backward = (cpp / 'src' / 'backward.cu').read_text()
    cuda_check = (cpp / 'include' / 'cuda_check.h').read_text()
    core = (cpp / 'src' / 'core.cu').read_text()
    conv_backward = (cpp / 'src' / 'conv_backward.cu').read_text()
    leaky_relu = (cpp / 'src' / 'leaky_relu.cu').read_text()
    maxpool_nchw = (cpp / 'src' / 'maxpool_backward_nchw.cu').read_text()
    dense_layer = (cpp / 'src' / 'dense_layer.cu').read_text()
    gpu_monitor = (cpp / 'src' / 'gpu_monitor.cu').read_text()
    network = (cpp / 'src' / 'network.cu').read_text()
    network_header = (cpp / 'include' / 'network.h').read_text()

    assert 'softmax_cross_entropy' not in loss_layer
    assert 'im2col_backward' not in loss_layer
    assert 'softmax_kernel<<<N, 32>>>' in loss_layer
    assert 'softmax_xent_grad_loss_acc_kernel<<<N, 32>>>' in loss_layer
    assert '__shfl_down_sync' in loss_layer

    assert 'MINICNN_DEBUG_SYNC' in cuda_check
    assert 'cudaDeviceSynchronize()' in cuda_check
    assert 'target_compile_definitions(minimal_cuda_cnn PRIVATE $<$<CONFIG:Debug>:MINICNN_DEBUG_SYNC>)' in (
        cpp / 'CMakeLists.txt'
    ).read_text()

    assert (cpp / 'include' / 'cublas_check.h').exists()
    assert 'static void cublas_check' not in core
    assert 'static void cublas_check' not in conv_backward

    assert 'layer_norm_forward_kernel<<<N * C, tpb, tpb * sizeof(float)>>>' in layer_norm
    assert 'dy - mean_dy - x_hat * mean_dy_xhat' in layer_norm

    assert 'cudaMemset(d_grad_input, 0, n * c * h * w * sizeof(float))' in backward
    assert 'leaky_relu_forward_nchw_kernel' not in leaky_relu
    assert 'leaky_relu_backward_nchw_kernel' not in leaky_relu
    assert 'int N, int C, int in_h, int in_w, int out_h, int out_w' in maxpool_nchw
    assert 'assert(' not in maxpool_nchw
    assert 'cudaErrorInvalidValue' in maxpool_nchw
    assert 'extern "C" int maxpool_backward_nchw_status' in maxpool_nchw
    assert 'dense_backward_weights_atomic_kernel' not in dense_layer

    assert 'system(' not in gpu_monitor
    assert 'cudaMemGetInfo' in gpu_monitor
    assert 'std::unique_ptr<CudaTensor>' in network_header
    assert 'd_col_cache' in network_header
    assert 'std::make_unique<CudaTensor>' in network
    assert 'relu_forward_copy_kernel' in network
    assert 'cudaMemcpy(d_output, d_input' not in network


def test_python_comment_tasks_are_reflected_in_source():
    src = REPO_ROOT / 'src' / 'minicnn'
    sgd = (src / 'optim' / 'sgd.py').read_text()
    nn_ops = (src / 'ops' / 'nn_ops.py').read_text()
    layers = (src / 'nn' / 'layers.py').read_text()
    tensor = (src / 'nn' / 'tensor.py').read_text()
    train_autograd = (src / 'training' / 'train_autograd.py').read_text()
    cuda_epoch = (src / 'training' / 'cuda_epoch.py').read_text()
    torch_baseline = (src / 'training' / 'train_torch_baseline.py').read_text()
    flex_data = (src / 'flex' / 'data.py').read_text()
    flex_trainer = (src / 'flex' / 'trainer.py').read_text()
    evaluation = (src / 'training' / 'evaluation.py').read_text()

    assert 'self.velocities' in sgd
    assert 'self.momentum * self.velocities[i] - self.lr * grad' in sgd
    assert 'running_mean' in layers
    assert 'training=self.training' in layers
    assert 'running_mean[...]' in nn_ops
    assert 'sliding_window_view' in nn_ops
    assert 'np.einsum' in nn_ops
    assert 'np.add.at' in nn_ops
    assert 'one_hot' not in tensor
    assert 'train_rng.permutation' in train_autograd
    assert 'rng=init_rng' in train_autograd
    assert 'Always returns a new array' in cuda_epoch
    assert 'x = x.copy()' in cuda_epoch
    assert 'flip_mask = train_rng.random(len(x)) > 0.5' in torch_baseline
    assert 'self.epoch * 10_000_019' in flex_data
    assert 'def set_epoch' in flex_data
    assert 'generator=generator' in flex_data
    assert 'Modifies `optimizer_cfg` in-place' in flex_trainer
    assert 'class EvalWorkspace' in evaluation
    assert 'count_correct_batch_with_workspace' in evaluation
