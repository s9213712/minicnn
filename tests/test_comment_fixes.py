import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    help_text = build_parser().format_help()

    assert 'doctor' in help_text
    assert 'compare' in help_text
    assert 'train-cuda' in help_text
    assert 'train-torch' in help_text
    assert 'train-autograd' in help_text
    assert 'validate-config' in help_text
    assert 'compile' in help_text


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

    freed = []
    uploaded = []
    old_weights = checkpoints.DeviceWeights('old1', 'old2', 'old3', 'old4', 'old5', 'old6')
    ckpt_path = tmp_path / 'weights.npz'
    np.savez(
        ckpt_path,
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
        checkpoints.reload_weights_from_checkpoint(ckpt_path, old_weights)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError('expected upload failure')

    assert freed == ['new0', 'new1']

    freed.clear()
    uploaded.clear()
    monkeypatch.setattr(checkpoints, 'upload', lambda arr: uploaded.append(f'new{len(uploaded)}') or uploaded[-1])

    _ckpt, _fc_w, _fc_b, new_weights = checkpoints.reload_weights_from_checkpoint(ckpt_path, old_weights)

    assert isinstance(new_weights, checkpoints.DeviceWeights)
    assert list(new_weights) == ['new0', 'new1', 'new2', 'new3', 'new4', 'new5']
    assert freed == list(old_weights)


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

    assert 'softmax_cross_entropy' not in loss_layer
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
    assert 'out_h * 2' not in maxpool_nchw
    assert 'dense_backward_weights_atomic_kernel' not in dense_layer
