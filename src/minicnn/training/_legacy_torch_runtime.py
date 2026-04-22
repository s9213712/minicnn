from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    DATASET_SEED,
    EPOCHS,
    FC_IN,
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
from minicnn.models.initialization import init_weights
from minicnn.paths import ARTIFACTS_ROOT


def resolve_torch_baseline_artifacts() -> tuple[Path, str]:
    run_dir = Path(os.environ.get('MINICNN_ARTIFACT_RUN_DIR', ARTIFACTS_ROOT / 'default'))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, str(run_dir / 'best_model_split_torch.pt')


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
        return self.fc(torch.flatten(x, 1))


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


def save_best_checkpoint(best_model_path: str, epoch: int, val_acc: float, lr_state, runtime: TorchRuntimeState) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "val_acc": float(val_acc),
            "lr_conv1": float(lr_state.conv1),
            "lr_conv": float(lr_state.conv),
            "lr_fc": float(lr_state.fc),
            "model_state": runtime.model.state_dict(),
        },
        best_model_path,
    )


def run_final_evaluation(best_model_path: str, runtime: TorchRuntimeState, x_test: np.ndarray, y_test: np.ndarray, evaluate_fn) -> None:
    print("\n=== FINAL EVALUATION ON OFFICIAL TEST SET ===")
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=runtime.device)
        runtime.model.load_state_dict(ckpt["model_state"])
        print(f"Reloaded best checkpoint from epoch {int(ckpt['epoch'])} with Val={float(ckpt['val_acc']):.2f}%")
    test_acc = evaluate_fn(runtime.model, x_test, y_test, runtime.device)
    print(f"Test Accuracy: {test_acc:.2f}%")
