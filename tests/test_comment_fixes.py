import numpy as np


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


def test_random_crop_batch_matches_reference_crop():
    from minicnn.training.train_cuda import random_crop_batch

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

    assert workspace.d_max_idx1[0] == 'int'
    assert workspace.d_max_idx2[0] == 'int'
    assert workspace.d_y[0] == 'int'
    assert workspace.d_correct[0] == 'int'
    assert float_allocs
    assert int_allocs
    workspace.free()
    assert len(freed) == len(float_allocs) + len(int_allocs)
