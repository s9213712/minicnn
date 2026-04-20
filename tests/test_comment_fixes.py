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

    parser = build_parser()
    help_text = parser.format_help()
    subparsers = next(action for action in parser._actions if getattr(action, 'choices', None))
    build_help = subparsers.choices['build'].format_help()

    assert 'doctor' in help_text
    assert 'compare' in help_text
    assert 'train-cuda' in help_text
    assert 'train-torch' in help_text
    assert 'train-autograd' in help_text
    assert 'validate-config' in help_text
    assert 'compile' in help_text
    assert '--cuda-arch' in build_help


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
    assert 'softmax_kernel<<<N, 1>>>' in loss_layer
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
