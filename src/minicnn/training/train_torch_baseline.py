#!/usr/bin/env python3
"""PyTorch baseline with the same CIFAR-10 split, architecture, and initial weights."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from minicnn.models.initialization import init_weights
from minicnn.training.legacy_data import load_normalized_cifar10
from minicnn.training.loop import (
    EpochTimer,
    FitState,
    LrState,
    RunningMetrics,
    format_epoch_summary,
    reduce_lr_on_plateau,
)
from minicnn.config.settings import (
    BATCH,
    C1_IN,
    C1_OUT,
    C2_IN,
    C2_OUT,
    C3_IN,
    C3_OUT,
    C4_IN,
    C4_OUT,
    CONV_GRAD_SPATIAL_NORMALIZE,
    DATASET_SEED,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    EVAL_MAX_BATCHES,
    FC_IN,
    GRAD_CLIP_BIAS,
    GRAD_CLIP_CONV,
    GRAD_CLIP_FC,
    GRAD_POOL_CLIP,
    H1,
    H2,
    H3,
    H4,
    INIT_SEED,
    KH,
    KW,
    LEAKY_ALPHA,
    LR_CONV,
    LR_CONV1,
    LR_FC,
    LR_PLATEAU_PATIENCE,
    LR_REDUCE_FACTOR,
    MIN_DELTA,
    MIN_LR,
    MOMENTUM,
    P1H,
    P1W,
    P2H,
    P2W,
    TRAIN_SEED,
    W1,
    W2,
    W3,
    W4,
    WEIGHT_DECAY,
)


from minicnn.paths import ARTIFACTS_ROOT

RUN_DIR = Path(os.environ.get('MINICNN_ARTIFACT_RUN_DIR', ARTIFACTS_ROOT / 'default'))
RUN_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = str(RUN_DIR / 'best_model_split_torch.pt')


@dataclass
class TorchRuntimeState:
    model: nn.Module
    velocity: dict
    criterion: nn.Module
    device: torch.device


def get_device():
    if os.environ.get("FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchCifarCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(C1_IN, C1_OUT, kernel_size=(KH, KW), bias=False)
        self.conv2 = nn.Conv2d(C2_IN, C2_OUT, kernel_size=(KH, KW), bias=False)
        self.conv3 = nn.Conv2d(C3_IN, C3_OUT, kernel_size=(KH, KW), bias=False)
        self.conv4 = nn.Conv2d(C4_IN, C4_OUT, kernel_size=(KH, KW), bias=False)
        self.act = nn.LeakyReLU(LEAKY_ALPHA)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(FC_IN, 10)

    def forward(self, x, clamp_pool_grad=False):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(x)
        if clamp_pool_grad and x.requires_grad:
            x.register_hook(lambda grad: grad.clamp(-GRAD_POOL_CLIP, GRAD_POOL_CLIP))
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_initial_weights(model, device):
    *conv_arrs, fc_w, fc_b = init_weights(INIT_SEED)
    w_conv1, w_conv2, w_conv3, w_conv4 = conv_arrs
    with torch.no_grad():
        model.conv1.weight.copy_(torch.from_numpy(w_conv1.reshape(C1_OUT, C1_IN, KH, KW)).to(device))
        model.conv2.weight.copy_(torch.from_numpy(w_conv2.reshape(C2_OUT, C2_IN, KH, KW)).to(device))
        model.conv3.weight.copy_(torch.from_numpy(w_conv3.reshape(C3_OUT, C3_IN, KH, KW)).to(device))
        model.conv4.weight.copy_(torch.from_numpy(w_conv4.reshape(C4_OUT, C4_IN, KH, KW)).to(device))
        model.fc.weight.copy_(torch.from_numpy(fc_w.reshape(10, FC_IN)).to(device))
        model.fc.bias.copy_(torch.from_numpy(fc_b).to(device))


def init_velocity_buffers(model):
    return {
        model.conv1.weight: torch.zeros_like(model.conv1.weight),
        model.conv2.weight: torch.zeros_like(model.conv2.weight),
        model.conv3.weight: torch.zeros_like(model.conv3.weight),
        model.conv4.weight: torch.zeros_like(model.conv4.weight),
        model.fc.weight: torch.zeros_like(model.fc.weight),
        model.fc.bias: torch.zeros_like(model.fc.bias),
    }


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


def init_torch_runtime(device: torch.device) -> TorchRuntimeState:
    torch.manual_seed(INIT_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(INIT_SEED)

    model = TorchCifarCnn().to(device)
    load_initial_weights(model, device)
    return TorchRuntimeState(
        model=model,
        velocity=init_velocity_buffers(model),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )


def print_training_header(device: torch.device, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> None:
    print(f"Device: {device}")
    print(f"Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test(official): {x_test.shape[0]}")
    print(
        "Arch: Conv1(3->32)->Conv2(32->32)->Pool1"
        f"->Conv3(32->64)->Conv4(64->64)->Pool2->FC({FC_IN}->10)"
    )
    print(
        f"Shapes: 32x32 -> {H1}x{W1} -> {H2}x{W2} -> {P1H}x{P1W}"
        f" -> {H3}x{W3} -> {H4}x{W4} -> {P2H}x{P2W}"
    )
    print(
        f"LR_conv1={LR_CONV1}, LR_conv={LR_CONV}, LR_fc={LR_FC}, "
        f"momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY}, BATCH={BATCH}, EPOCHS={EPOCHS}"
    )
    print(f"DATASET_SEED={DATASET_SEED}, INIT_SEED={INIT_SEED}, TRAIN_SEED={TRAIN_SEED}")
    print()


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


def save_best_checkpoint(epoch: int, val_acc: float, lr_state: LrState, runtime: TorchRuntimeState) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "val_acc": float(val_acc),
            "lr_conv1": float(lr_state.conv1),
            "lr_conv": float(lr_state.conv),
            "lr_fc": float(lr_state.fc),
            "model_state": runtime.model.state_dict(),
        },
        BEST_MODEL_PATH,
    )


def reduce_lr_if_due(fit: FitState, lr_state: LrState) -> None:
    if reduce_lr_on_plateau(fit, lr_state, LR_PLATEAU_PATIENCE, LR_REDUCE_FACTOR, MIN_LR):
        print(
            f"  LR reduced -> conv1={lr_state.conv1:.6f}, "
            f"conv={lr_state.conv:.6f}, fc={lr_state.fc:.6f}"
        )


def run_final_evaluation(runtime: TorchRuntimeState, x_test: np.ndarray, y_test: np.ndarray) -> None:
    print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
    if os.path.exists(BEST_MODEL_PATH):
        ckpt = torch.load(BEST_MODEL_PATH, map_location=runtime.device)
        runtime.model.load_state_dict(ckpt["model_state"])
        print(f"Reloaded best checkpoint from epoch {int(ckpt['epoch'])} with Val={float(ckpt['val_acc']):.2f}%")
    test_acc = evaluate(runtime.model, x_test, y_test, runtime.device)
    print(f"Test Accuracy: {test_acc:.2f}%")


def main():
    device = get_device()
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
            save_best_checkpoint(epoch + 1, val_acc, lr_state, runtime)
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

    run_final_evaluation(runtime, x_test_final, y_test_final)
    print("\nDone!")


if __name__ == "__main__":
    main()
