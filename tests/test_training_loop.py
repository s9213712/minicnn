from pathlib import Path

from minicnn.training.loop import (
    FitState,
    LrState,
    RunningMetrics,
    format_epoch_summary,
    reduce_lr_on_plateau,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_running_metrics_accumulates_epoch_totals():
    metrics = RunningMetrics()

    metrics.update(loss_sum=3.0, correct=2, total=4)
    metrics.update(loss_sum=2.0, correct=1, total=2)

    assert metrics.loss == 5.0 / 6
    assert metrics.acc_percent == 50.0


def test_lr_state_reduces_and_clamps_to_min_lr():
    state = LrState(conv1=0.1, conv=0.02, fc=0.005)

    assert state.reduce(factor=0.5, min_lr=0.01) is True
    assert state.as_tuple() == (0.05, 0.01, 0.01)

    assert state.reduce(factor=0.5, min_lr=0.01) is True
    assert state.as_tuple() == (0.025, 0.01, 0.01)

    state.conv1 = 0.01
    assert state.reduce(factor=0.5, min_lr=0.01) is False
    assert state.as_tuple() == (0.01, 0.01, 0.01)


def test_fit_state_tracks_best_plateau_and_early_stop():
    state = FitState()

    assert state.observe(epoch=1, val_acc=20.0, min_delta=0.1) is True
    assert state.best_val_acc == 20.0
    assert state.best_epoch == 1
    assert state.epochs_no_improve == 0
    assert state.plateau_count == 0

    assert state.observe(epoch=2, val_acc=20.05, min_delta=0.1) is False
    assert state.epochs_no_improve == 1
    assert state.plateau_due(1) is True
    assert state.should_stop(2) is False

    state.reset_plateau()
    assert state.plateau_count == 0

    assert state.observe(epoch=3, val_acc=19.0, min_delta=0.1) is False
    assert state.should_stop(2) is True


def test_shared_plateau_reducer_resets_only_when_due():
    fit = FitState()
    lr_state = LrState(conv1=0.1, conv=0.1, fc=0.1)

    assert reduce_lr_on_plateau(fit, lr_state, patience=2, factor=0.5, min_lr=0.01) is False
    assert lr_state.as_tuple() == (0.1, 0.1, 0.1)

    fit.observe(epoch=1, val_acc=1.0, min_delta=0.0)
    fit.observe(epoch=2, val_acc=1.0, min_delta=0.1)
    fit.observe(epoch=3, val_acc=1.0, min_delta=0.1)

    assert reduce_lr_on_plateau(fit, lr_state, patience=2, factor=0.5, min_lr=0.01) is True
    assert lr_state.as_tuple() == (0.05, 0.05, 0.05)
    assert fit.plateau_count == 0


def test_epoch_summary_formatter_supports_backend_lr_spacing():
    metrics = RunningMetrics(loss_sum=4.0, correct=3, total=4)
    fit = FitState(best_val_acc=55.0, best_epoch=2)
    lr_state = LrState(conv1=0.1, conv=0.2, fc=0.3)

    summary = format_epoch_summary(
        2, 5, metrics, 55.0, fit, lr_state, 1.25, " [saved best]",
        lr_separator=",",
    )

    assert "Loss=1.0000" in summary
    assert "Train=75.00%" in summary
    assert "LRs=(0.100000,0.200000,0.300000)" in summary
    assert summary.endswith("Time=1.2s [saved best]")


def test_legacy_trainers_are_split_into_backend_steps():
    train_cuda = (REPO_ROOT / 'src' / 'minicnn' / 'training' / 'train_cuda.py').read_text()
    cuda_batch = (REPO_ROOT / 'src' / 'minicnn' / 'training' / 'cuda_batch.py').read_text()
    torch_baseline = (REPO_ROOT / 'src' / 'minicnn' / 'training' / 'train_torch_baseline.py').read_text()

    assert 'def train_cuda_batch' in cuda_batch
    assert 'def forward_convs' in cuda_batch
    assert 'def backward_convs_update' in cuda_batch
    assert '_d_weights' not in train_cuda
    assert 'train_cuda_batch(' in train_cuda

    assert 'def prepare_augmented_batch' in torch_baseline
    assert 'def train_torch_batch' in torch_baseline
    assert 'def run_torch_epoch' in torch_baseline
    assert 'format_epoch_summary(' in train_cuda
    assert 'format_epoch_summary(' in torch_baseline
