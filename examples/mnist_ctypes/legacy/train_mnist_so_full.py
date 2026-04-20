#!/usr/bin/env python3
"""MNIST CNN example using cpp/libminimal_cuda_cnn.so through ctypes

Network:
Input NCHW 1x28x28
-> Conv(1->8, 3x3) -> LeakyReLU
-> MaxPool(2x2)
-> FC(8*13*13 -> 10)

This file is intentionally self-contained and uses only NumPy plus Python's
standard library for MNIST IDX parsing. The CUDA layer code comes from the
project shared library.

Current native wrappers prefer status-returning symbols when available; the
supported package wrapper uses maxpool_backward_nchw_status for catchable
host-side errors while keeping the legacy void ABI compatible.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import json
import shutil
import struct
import urllib.request
from pathlib import Path
from ctypes import c_float, c_int, c_void_p

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIB = ROOT / "cpp" / "libminimal_cuda_cnn_handmade.so"
DEFAULT_DATA = ROOT / "data" / "mnist"
DEFAULT_RUNS = ROOT / "runs" / "mnist_so"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DEFAULT_BATCH = 64
IN_C, IN_H, IN_W = 1, 28, 28
OUT_C, KH, KW = 8, 3, 3
OUT_H, OUT_W = 26, 26
POOL_H, POOL_W = 13, 13
FC_IN = OUT_C * POOL_H * POOL_W
CLASSES = 10
LEAKY_ALPHA = 0.1


class CudaMnistLib:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self._bind()

    def _bind(self):
        lib = self.lib
        lib.gpu_malloc.argtypes = [ctypes.c_size_t]
        lib.gpu_malloc.restype = c_void_p
        lib.gpu_free.argtypes = [c_void_p]
        lib.gpu_memcpy_h2d.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        lib.gpu_memcpy_d2h.argtypes = [c_void_p, c_void_p, ctypes.c_size_t]
        lib.gpu_memset.argtypes = [c_void_p, c_int, ctypes.c_size_t]

        lib.im2col_forward.argtypes = [
            c_void_p, c_void_p,
            c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        ]
        lib.gemm_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        lib.dense_forward.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int]
        lib.dense_backward_full.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_int,
        ]
        lib.leaky_relu_forward.argtypes = [c_void_p, c_float, c_int]
        lib.leaky_relu_backward.argtypes = [c_void_p, c_void_p, c_float, c_int]
        lib.maxpool_forward_store.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        lib.maxpool_backward_use_idx.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        lib.nchw_to_cnhw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        lib.cnhw_to_nchw.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        lib.conv_backward.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        ]
        lib.apply_sgd_update.argtypes = [c_void_p, c_void_p, c_float, c_int]

    def malloc(self, nbytes: int):
        ptr = self.lib.gpu_malloc(nbytes)
        if not ptr:
            raise MemoryError(f"gpu_malloc failed for {nbytes} bytes")
        return ptr

    def free(self, ptr):
        if ptr:
            self.lib.gpu_free(ptr)

    def upload(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
        ptr = self.malloc(arr.nbytes)
        self.lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.nbytes)
        return ptr

    def zeros(self, size: int):
        ptr = self.malloc(size * 4)
        self.lib.gpu_memset(ptr, 0, size * 4)
        return ptr

    def download(self, ptr, shape):
        out = np.empty(shape, dtype=np.float32)
        self.lib.gpu_memcpy_d2h(out.ctypes.data, ptr, out.nbytes)
        return out

    def free_all(self, *ptrs):
        for ptr in ptrs:
            self.free(ptr)


def maybe_download_mnist(data_dir: Path, download: bool):
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in MNIST_FILES.values() if not (data_dir / name).exists()]
    if missing and not download:
        missing_text = "\n".join(f"  - {data_dir / name}" for name in missing)
        raise SystemExit(
            "MNIST files are missing:\n"
            f"{missing_text}\n"
            "Run again with --download, or place the .gz IDX files in that directory."
        )

    for filename in missing:
        url = f"{MNIST_URL}/{filename}"
        target = data_dir / filename
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target)


def read_idx_images(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image IDX magic in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, 1, rows, cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label IDX magic in {path}: {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels[:count].astype(np.int64)


def load_mnist(data_dir: Path, val_size: int, seed: int, download: bool):
    maybe_download_mnist(data_dir, download)
    x_train_all = read_idx_images(data_dir / MNIST_FILES["train_images"])
    y_train_all = read_idx_labels(data_dir / MNIST_FILES["train_labels"])
    x_test = read_idx_images(data_dir / MNIST_FILES["test_images"])
    y_test = read_idx_labels(data_dir / MNIST_FILES["test_labels"])

    mean = x_train_all.mean()
    std = x_train_all.std() + 1e-6
    x_train_all = (x_train_all - mean) / std
    x_test = (x_test - mean) / std

    if val_size <= 0 or val_size >= len(x_train_all):
        raise ValueError(f"val_size must be between 1 and {len(x_train_all) - 1}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train_all))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (
        x_train_all[train_idx],
        y_train_all[train_idx],
        x_train_all[val_idx],
        y_train_all[val_idx],
        x_test,
        y_test,
        float(mean),
        float(std),
    )


def he_init(rng: np.random.Generator, size: int, fan_in: int):
    return (rng.standard_normal(size).astype(np.float32) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def softmax_loss_and_grad(logits: np.ndarray, labels: np.ndarray):
    shifted = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(shifted)
    probs = expv / expv.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
    acc = np.mean(np.argmax(probs, axis=1) == labels)

    grad = probs.copy()
    grad[np.arange(len(labels)), labels] -= 1.0
    grad /= len(labels)

    return float(loss), float(acc), grad.astype(np.float32)


class MnistCnn:
    def __init__(self, cuda: CudaMnistLib, seed: int):
        self.cuda = cuda
        rng = np.random.default_rng(seed)
        self.w_conv = cuda.upload(he_init(rng, OUT_C * IN_C * KH * KW, IN_C * KH * KW))
        self.w_fc = cuda.upload(he_init(rng, CLASSES * FC_IN, FC_IN))
        self.b_fc = cuda.upload(np.zeros(CLASSES, dtype=np.float32))

    def close(self):
        self.cuda.free_all(self.w_conv, self.w_fc, self.b_fc)
        self.w_conv = None
        self.w_fc = None
        self.b_fc = None

    def state_dict(self):
        return {
            "w_conv": self.cuda.download(self.w_conv, (OUT_C, IN_C, KH, KW)),
            "w_fc": self.cuda.download(self.w_fc, (CLASSES, FC_IN)),
            "b_fc": self.cuda.download(self.b_fc, (CLASSES,)),
        }

    def load_state_dict(self, state):
        w_conv = np.ascontiguousarray(state["w_conv"], dtype=np.float32).reshape(-1)
        w_fc = np.ascontiguousarray(state["w_fc"], dtype=np.float32).reshape(-1)
        b_fc = np.ascontiguousarray(state["b_fc"], dtype=np.float32).reshape(-1)

        self.cuda.lib.gpu_memcpy_h2d(self.w_conv, w_conv.ctypes.data, w_conv.nbytes)
        self.cuda.lib.gpu_memcpy_h2d(self.w_fc, w_fc.ctypes.data, w_fc.nbytes)
        self.cuda.lib.gpu_memcpy_h2d(self.b_fc, b_fc.ctypes.data, b_fc.nbytes)

    def save_npz(self, path: Path, extra: dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        payload = {
            "w_conv": state["w_conv"],
            "w_fc": state["w_fc"],
            "b_fc": state["b_fc"],
        }
        if extra:
            for k, v in extra.items():
                payload[k] = np.array(v)
        np.savez(path, **payload)

    def forward(self, x: np.ndarray):
        lib = self.cuda.lib
        n = len(x)

        d_x = self.cuda.upload(x)
        d_col = self.cuda.malloc(IN_C * KH * KW * n * OUT_H * OUT_W * 4)
        d_conv = self.cuda.malloc(OUT_C * n * OUT_H * OUT_W * 4)

        lib.im2col_forward(d_x, d_col, n, IN_C, IN_H, IN_W, KH, KW, OUT_H, OUT_W)
        lib.gemm_forward(self.w_conv, d_col, d_conv, OUT_C, n * OUT_H * OUT_W, IN_C * KH * KW)
        lib.leaky_relu_forward(d_conv, c_float(LEAKY_ALPHA), OUT_C * n * OUT_H * OUT_W)

        d_pool = self.cuda.malloc(OUT_C * n * POOL_H * POOL_W * 4)
        d_pool_idx = self.cuda.malloc(OUT_C * n * POOL_H * POOL_W * 4)
        lib.maxpool_forward_store(d_pool, d_conv, d_pool_idx, n, OUT_C, OUT_H, OUT_W)

        d_pool_nchw = self.cuda.malloc(n * FC_IN * 4)
        lib.cnhw_to_nchw(d_pool, d_pool_nchw, n, OUT_C, POOL_H, POOL_W)

        d_logits = self.cuda.malloc(n * CLASSES * 4)
        lib.dense_forward(d_pool_nchw, self.w_fc, self.b_fc, d_logits, n, FC_IN, CLASSES)

        logits = self.cuda.download(d_logits, (n, CLASSES))
        cache = (d_x, d_col, d_conv, d_pool, d_pool_idx, d_pool_nchw, d_logits)
        return logits, cache

    def train_batch(self, x: np.ndarray, y: np.ndarray, lr_conv: float, lr_fc: float):
        lib = self.cuda.lib
        n = len(x)
        logits, cache = self.forward(x)
        d_x, d_col, d_conv, d_pool, d_pool_idx, d_pool_nchw, d_logits = cache
        loss, acc, grad_logits = softmax_loss_and_grad(logits, y)

        d_grad_logits = self.cuda.upload(grad_logits)
        d_grad_pool_nchw = self.cuda.zeros(n * FC_IN)
        d_grad_fc_w = self.cuda.zeros(CLASSES * FC_IN)
        d_grad_fc_b = self.cuda.zeros(CLASSES)

        lib.dense_backward_full(
            d_grad_logits,
            d_pool_nchw,
            self.w_fc,
            d_grad_pool_nchw,
            d_grad_fc_w,
            d_grad_fc_b,
            n,
            FC_IN,
            CLASSES,
        )

        lib.apply_sgd_update(self.w_fc, d_grad_fc_w, c_float(lr_fc), CLASSES * FC_IN)
        lib.apply_sgd_update(self.b_fc, d_grad_fc_b, c_float(lr_fc), CLASSES)

        d_grad_pool = self.cuda.malloc(OUT_C * n * POOL_H * POOL_W * 4)
        lib.nchw_to_cnhw(d_grad_pool_nchw, d_grad_pool, n, OUT_C, POOL_H, POOL_W)

        d_grad_conv = self.cuda.zeros(OUT_C * n * OUT_H * OUT_W)
        lib.maxpool_backward_use_idx(d_grad_pool, d_pool_idx, d_grad_conv, n, OUT_C, OUT_H, OUT_W)
        lib.leaky_relu_backward(d_conv, d_grad_conv, c_float(LEAKY_ALPHA), OUT_C * n * OUT_H * OUT_W)

        d_grad_conv_w = self.cuda.zeros(OUT_C * IN_C * KH * KW)
        d_grad_x = self.cuda.zeros(n * IN_C * IN_H * IN_W)

        lib.conv_backward(
            d_grad_conv,
            d_x,
            self.w_conv,
            d_grad_conv_w,
            d_grad_x,
            n,
            IN_C,
            IN_H,
            IN_W,
            KH,
            KW,
            OUT_H,
            OUT_W,
            OUT_C,
        )

        self._update_conv(d_grad_conv_w, lr_conv, n)

        self.cuda.free_all(
            *cache,
            d_grad_logits,
            d_grad_pool_nchw,
            d_grad_fc_w,
            d_grad_fc_b,
            d_grad_pool,
            d_grad_conv,
            d_grad_conv_w,
            d_grad_x,
        )
        return loss, acc

    def eval_batch(self, x: np.ndarray, y: np.ndarray):
        logits, cache = self.forward(x)
        loss, acc, _ = softmax_loss_and_grad(logits, y)
        self.cuda.free_all(*cache)
        return loss, acc

    def _update_conv(self, d_grad, lr: float, n: int):
        grad = self.cuda.download(d_grad, (OUT_C * IN_C * KH * KW,)) / float(n)
        grad = np.clip(grad, -1.0, 1.0).astype(np.float32)
        self.cuda.lib.gpu_memcpy_h2d(d_grad, grad.ctypes.data, grad.nbytes)
        self.cuda.lib.apply_sgd_update(self.w_conv, d_grad, c_float(lr), OUT_C * IN_C * KH * KW)


def batches(
    x,
    y,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
    limit: int | None,
    drop_last: bool = True,
):
    count = len(x) if limit is None else min(limit, len(x))
    indices = np.arange(count)
    if shuffle:
        rng.shuffle(indices)

    if drop_last:
        usable = (count // batch_size) * batch_size
    else:
        usable = count

    for start in range(0, usable, batch_size):
        idx = indices[start:start + batch_size]
        if len(idx) == 0:
            continue
        if drop_last and len(idx) != batch_size:
            continue
        yield np.ascontiguousarray(x[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.int64)


def run_eval(model: MnistCnn, x, y, batch_size: int, rng, limit: int | None):
    losses = []
    accs = []
    for xb, yb in batches(x, y, batch_size, False, rng, limit, drop_last=True):
        loss, acc = model.eval_batch(xb, yb)
        losses.append(loss)
        accs.append(acc)

    if not losses:
        raise RuntimeError("No evaluation batches were produced; reduce batch size or increase dataset/limit.")
    return float(np.mean(losses)), float(np.mean(accs))


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--val-size", type=int, default=5000)

    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)

    parser.add_argument("--lr-conv", type=float, default=0.01)
    parser.add_argument("--lr-fc", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS / "latest")
    parser.add_argument("--no-save-best", action="store_true")

    args = parser.parse_args()

    if args.batch_size != DEFAULT_BATCH:
        raise SystemExit(
            f"This simple example currently expects --batch-size {DEFAULT_BATCH} "
            "(library path and helper code are written around fixed-size batches)."
        )

    if not args.lib.exists():
        raise SystemExit(f"shared library not found: {args.lib}\nRun: make -C {ROOT / 'cpp'}")

    x_train, y_train, x_val, y_val, x_test, y_test, data_mean, data_std = load_mnist(
        args.data,
        args.val_size,
        args.seed,
        args.download,
    )

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"split: train={len(x_train)} val={len(x_val)} test={len(x_test)}")
    print("arch: Conv(1->8,3x3)->LeakyReLU->MaxPool->FC(1352->10)")
    print(
        f"limits: train={args.train_limit} val={args.val_limit} test={args.test_limit} "
        f"(None means full split)"
    )
    print(
        f"hyper: epochs={args.epochs} batch={args.batch_size} "
        f"lr_conv={args.lr_conv} lr_fc={args.lr_fc} patience={args.patience}"
    )
    print(f"norm: mean={data_mean:.6f} std={data_std:.6f}")
    print(f"run_dir: {run_dir}")

    save_json(
        run_dir / "config.json",
        {
            "lib": str(args.lib),
            "data": str(args.data),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_size": args.val_size,
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
            "test_limit": args.test_limit,
            "lr_conv": args.lr_conv,
            "lr_fc": args.lr_fc,
            "seed": args.seed,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "data_mean": data_mean,
            "data_std": data_std,
        },
    )

    cuda = CudaMnistLib(args.lib)
    model = MnistCnn(cuda, args.seed)
    rng = np.random.default_rng(args.seed)

    history = []
    best_val_acc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improve = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_losses = []
            train_accs = []

            for xb, yb in batches(x_train, y_train, args.batch_size, True, rng, args.train_limit, drop_last=True):
                loss, acc = model.train_batch(xb, yb, args.lr_conv, args.lr_fc)
                train_losses.append(loss)
                train_accs.append(acc)

            if not train_losses:
                raise RuntimeError("No training batches were produced; reduce batch size or increase dataset/limit.")

            train_loss = float(np.mean(train_losses))
            train_acc = float(np.mean(train_accs))
            val_loss, val_acc = run_eval(model, x_val, y_val, args.batch_size, rng, args.val_limit)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(row)

            improved = val_acc > (best_val_acc + args.min_delta)
            marker = " *" if improved else ""

            print(
                f"epoch {epoch:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}"
            )

            save_json(run_dir / "history.json", {"history": history})

            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improve = 0
                best_state = model.state_dict()

                if not args.no_save_best:
                    model.save_npz(
                        run_dir / "best_model.npz",
                        extra={
                            "best_epoch": best_epoch,
                            "best_val_acc": best_val_acc,
                        },
                    )
            else:
                epochs_without_improve += 1

            if args.patience > 0 and epochs_without_improve >= args.patience:
                print(
                    f"early_stop: no val_acc improvement for {epochs_without_improve} epoch(s), "
                    f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}"
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_acc = run_eval(model, x_test, y_test, args.batch_size, rng, args.test_limit)
        print(
            f"best_epoch={best_epoch:03d} "
            f"best_val_acc={best_val_acc:.4f} "
            f"final_test_loss={test_loss:.4f} final_test_acc={test_acc:.4f}"
        )

        summary = {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_test_loss": test_loss,
            "final_test_acc": test_acc,
            "epochs_ran": len(history),
        }
        save_json(run_dir / "summary.json", summary)

        if not args.no_save_best and (run_dir / "best_model.npz").exists():
            shutil.copy2(run_dir / "best_model.npz", run_dir / "final_model_from_best.npz")

    finally:
        model.close()


if __name__ == "__main__":
    main()
