#!/usr/bin/env python3
"""PyTorch baseline with the same CIFAR-10 split, architecture, and initial weights."""

import numpy as np
import torch

from minicnn.training.legacy_data import load_normalized_cifar10
from minicnn.training.loop import (
    EpochTimer,
    FitState,
    LrState,
    RunningMetrics,
    format_epoch_summary,
    reduce_lr_on_plateau,
)
from minicnn.training._legacy_torch_runtime import (
    TorchRuntimeState,
    get_device,
    init_torch_runtime,
    print_training_header,
    resolve_torch_baseline_artifacts,
    run_final_evaluation,
    save_best_checkpoint,
)
from minicnn.config.settings import (
    BATCH,
    CONV_GRAD_SPATIAL_NORMALIZE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    EVAL_MAX_BATCHES,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_CONV,
    GRAD_CLIP_FC,
    H1,
    H2,
    H3,
    H4,
    LR_CONV,
    LR_CONV1,
    LR_FC,
    LR_PLATEAU_PATIENCE,
    LR_REDUCE_FACTOR,
    MIN_DELTA,
    MIN_LR,
    MOMENTUM,
    TRAIN_SEED,
    WEIGHT_DECAY,
)


def apply_momentum_update(model, velocity, lr_conv1, lr_conv, lr_fc):
    # Match the CUDA trainer: conv/fc weights get weight decay and layer-specific clipping;
    # bias only gets clipping, and optional spatial normalization keeps conv gradients comparable.
    updates = [
        (model.conv1.weight, lr_conv1, GRAD_CLIP_CONV, True, H1 * W1),
        (model.conv2.weight, lr_conv, GRAD_CLIP_CONV, True, H2 * W2),
        (model.conv3.weight, lr_conv, GRAD_CLIP_CONV, True, H3 * W3),
        (model.conv4.weight, lr_conv, GRAD_CLIP_CONV, True, H4 * W4),
        (model.fc.weight, lr_fc, GRAD_CLIP_FC, True, 1),
        (model.fc.bias, lr_fc, GRAD_CLIP_BIAS, False, 1),
    ]
    with torch.no_grad():
        for param, lr, clip_value, use_decay, grad_normalizer in updates:
            grad = param.grad
            if grad is None:
                continue
            if CONV_GRAD_SPATIAL_NORMALIZE:
                grad = grad / grad_normalizer
            if use_decay:
                grad = grad + WEIGHT_DECAY * param
            grad = torch.clamp(grad, -clip_value, clip_value)
            velocity[param].mul_(MOMENTUM).add_(grad, alpha=-lr)
            param += velocity[param]
            param.grad = None


def evaluate(model, x, y, device, batch_size=BATCH, max_batches=EVAL_MAX_BATCHES):
    model.eval()
    correct = 0
    total = 0
    nbatches = (x.shape[0] + batch_size - 1) // batch_size
    if max_batches is not None:
        nbatches = min(nbatches, max_batches)
    with torch.no_grad():
        for i in range(nbatches):
            idx_s = i * batch_size
            idx_e = min(idx_s + batch_size, x.shape[0])
            if idx_s >= idx_e:
                break
            xb = torch.from_numpy(x[idx_s:idx_e]).to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            correct += np.sum(pred == y[idx_s:idx_e])
            total += idx_e - idx_s
    if total == 0:
        return 0.0
    return correct / total * 100


def prepare_augmented_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    indices: np.ndarray,
    idx_s: int,
    idx_e: int,
    train_rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_train[indices[idx_s:idx_e]]
    y = y_train[indices[idx_s:idx_e]]

    flip_mask = train_rng.random(len(x)) > 0.5
    x = x.copy()
    if flip_mask.any():
        x[flip_mask] = x[flip_mask, :, :, ::-1]

    xb = torch.from_numpy(x).to(device)
    yb = torch.from_numpy(y.astype(np.int64, copy=False)).to(device)
    return xb, yb


def train_torch_batch(
    runtime: TorchRuntimeState,
    xb: torch.Tensor,
    yb: torch.Tensor,
    lr_state: LrState,
    metrics: RunningMetrics,
) -> float:
    logits = runtime.model(xb, clamp_pool_grad=True)
    loss = runtime.criterion(logits, yb)
    loss.backward()
    apply_momentum_update(runtime.model, runtime.velocity, lr_state.conv1, lr_state.conv, lr_state.fc)

    pred = torch.argmax(logits.detach(), dim=1)
    batch_loss = float(loss.detach().cpu().item())
    metrics.update(
        batch_loss * xb.shape[0],
        int((pred == yb).sum().detach().cpu().item()),
        xb.shape[0],
    )
    return batch_loss


def run_torch_epoch(
    runtime: TorchRuntimeState,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_rng: np.random.Generator,
    lr_state: LrState,
) -> RunningMetrics:
    runtime.model.train()
    metrics = RunningMetrics()
    indices = train_rng.permutation(x_train.shape[0])
    nbatches = (x_train.shape[0] + BATCH - 1) // BATCH

    for batch_idx in range(nbatches):
        idx_s = batch_idx * BATCH
        idx_e = min(idx_s + BATCH, x_train.shape[0])
        if idx_s >= idx_e:
            continue

        xb, yb = prepare_augmented_batch(
            x_train, y_train, indices, idx_s, idx_e, train_rng, runtime.device
        )
        batch_loss = train_torch_batch(runtime, xb, yb, lr_state, metrics)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Batch {batch_idx+1}/{nbatches}: "
                f"loss={batch_loss:.4f}, "
                f"acc={metrics.acc_percent:.1f}%"
            )

    return metrics


def reduce_lr_if_due(fit: FitState, lr_state: LrState) -> None:
    if reduce_lr_on_plateau(fit, lr_state, LR_PLATEAU_PATIENCE, LR_REDUCE_FACTOR, MIN_LR):
        print(
            f"  LR reduced -> conv1={lr_state.conv1:.6f}, "
            f"conv={lr_state.conv:.6f}, fc={lr_state.fc:.6f}"
        )


def main():
    device = get_device()
    _run_dir, best_model_path = resolve_torch_baseline_artifacts()
    x_train, y_train, x_val, y_val, x_test_final, y_test_final = load_normalized_cifar10()
    runtime = init_torch_runtime(device)
    fit = FitState()
    lr_state = LrState(LR_CONV1, LR_CONV, LR_FC)
    print_training_header(device, x_train, x_val, x_test_final)
    train_rng = np.random.default_rng(TRAIN_SEED)

    for epoch in range(EPOCHS):
        with EpochTimer() as timer:
            metrics = run_torch_epoch(runtime, x_train, y_train, train_rng, lr_state)
            val_acc = evaluate(runtime.model, x_val, y_val, runtime.device)
            improved = fit.observe(epoch + 1, val_acc, MIN_DELTA)

        if improved:
            save_best_checkpoint(best_model_path, epoch + 1, val_acc, lr_state, runtime)
            save_msg = " [saved best]"
        else:
            save_msg = ""

        reduce_lr_if_due(fit, lr_state)
        print(format_epoch_summary(
            epoch + 1, EPOCHS, metrics, val_acc, fit, lr_state, timer.elapsed, save_msg
        ))

        if fit.should_stop(EARLY_STOP_PATIENCE):
            print(
                f"Early stopping after {epoch+1} epochs; "
                f"best val {fit.best_val_acc:.2f}% at epoch {fit.best_epoch}."
            )
            break

    run_final_evaluation(best_model_path, runtime, x_test_final, y_test_final, evaluate)
    print("\nDone!")


if __name__ == "__main__":
    main()
