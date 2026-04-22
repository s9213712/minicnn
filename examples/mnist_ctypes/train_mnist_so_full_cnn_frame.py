#!/usr/bin/env python3
"""Refactored MNIST CNN example using the handcrafted MiniCNN native library via ctypes.

Architecture:
    Input NCHW 1x28x28
    -> Conv(1->8, 3x3) -> LeakyReLU -> MaxPool(2x2)
    -> Conv(8->16, 3x3) -> LeakyReLU -> MaxPool(2x2)
    -> FC(16*5*5 -> 10)

Goals of this version:
- cleaner Python orchestration
- reusable ConvBlock / DenseLayer
- dataclass caches instead of tuple-position coupling
- programmatic shape inference
- keep the existing native ABI unchanged
- prefer status-returning native wrappers when available while preserving the
  legacy void ABI
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import json
import os
import shutil
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from ctypes import c_float, c_int, c_void_p

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIB = ROOT / "cpp" / (
    "minimal_cuda_cnn_handmade.dll" if os.name == "nt" else "libminimal_cuda_cnn_handmade.so"
)
DEFAULT_DATA = ROOT / "data" / "mnist"
DEFAULT_RUNS = ROOT / "runs" / "mnist_so_refactor"

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DEFAULT_BATCH = 64
LEAKY_ALPHA = 0.1
CLASSES = 10


# -----------------------------
# Utility shape helpers
# -----------------------------
def conv2d_out_size(h: int, w: int, kh: int, kw: int, stride: int = 1, pad: int = 0) -> tuple[int, int]:
    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid conv output size: in=({h},{w}) k=({kh},{kw}) -> out=({out_h},{out_w})")
    return out_h, out_w


def pool2d_out_size(h: int, w: int, kh: int = 2, kw: int = 2, stride: int = 2) -> tuple[int, int]:
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid pool output size: in=({h},{w}) k=({kh},{kw}) -> out=({out_h},{out_w})")
    return out_h, out_w


def he_init(rng: np.random.Generator, size: int, fan_in: int) -> np.ndarray:
    return (rng.standard_normal(size).astype(np.float32) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def softmax_loss_and_grad(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(shifted)
    probs = expv / expv.sum(axis=1, keepdims=True)

    loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
    acc = np.mean(np.argmax(probs, axis=1) == labels)

    grad = probs.copy()
    grad[np.arange(len(labels)), labels] -= 1.0
    grad /= len(labels)

    return float(loss), float(acc), grad.astype(np.float32)


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# CUDA shared library wrapper
# -----------------------------
class CudaLib:
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

    def malloc_bytes(self, nbytes: int):
        ptr = self.lib.gpu_malloc(nbytes)
        if not ptr:
            raise MemoryError(f"gpu_malloc failed for {nbytes} bytes")
        return ptr

    def malloc_f32(self, count: int):
        return self.malloc_bytes(count * 4)

    def malloc_i32(self, count: int):
        return self.malloc_bytes(count * 4)

    def zeros_f32(self, count: int):
        ptr = self.malloc_f32(count)
        self.lib.gpu_memset(ptr, 0, count * 4)
        return ptr

    def upload_f32(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
        ptr = self.malloc_bytes(arr.nbytes)
        self.lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.nbytes)
        return ptr

    def download_f32(self, ptr, shape):
        out = np.empty(shape, dtype=np.float32)
        self.lib.gpu_memcpy_d2h(out.ctypes.data, ptr, out.nbytes)
        return out

    def overwrite_f32(self, ptr, arr: np.ndarray):
        arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
        self.lib.gpu_memcpy_h2d(ptr, arr.ctypes.data, arr.nbytes)

    def free(self, ptr):
        if ptr:
            self.lib.gpu_free(ptr)

    def free_all(self, *ptrs):
        for ptr in ptrs:
            self.free(ptr)


# -----------------------------
# Dataset loading
# -----------------------------
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

    usable = (count // batch_size) * batch_size if drop_last else count
    for start in range(0, usable, batch_size):
        idx = indices[start:start + batch_size]
        if len(idx) == 0:
            continue
        if drop_last and len(idx) != batch_size:
            continue
        yield np.ascontiguousarray(x[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.int64)


# -----------------------------
# Shape / config dataclasses
# -----------------------------
@dataclass(frozen=True)
class TensorShape:
    c: int
    h: int
    w: int

    @property
    def flat(self) -> int:
        return self.c * self.h * self.w


@dataclass(frozen=True)
class ConvBlockSpec:
    in_shape: TensorShape
    out_channels: int
    kernel_h: int = 3
    kernel_w: int = 3
    pool_kernel: int = 2
    pool_stride: int = 2

    @property
    def conv_out_h(self) -> int:
        return conv2d_out_size(self.in_shape.h, self.in_shape.w, self.kernel_h, self.kernel_w)[0]

    @property
    def conv_out_w(self) -> int:
        return conv2d_out_size(self.in_shape.h, self.in_shape.w, self.kernel_h, self.kernel_w)[1]

    @property
    def conv_out_shape(self) -> TensorShape:
        return TensorShape(self.out_channels, self.conv_out_h, self.conv_out_w)

    @property
    def pool_out_h(self) -> int:
        return pool2d_out_size(self.conv_out_h, self.conv_out_w, self.pool_kernel, self.pool_kernel, self.pool_stride)[0]

    @property
    def pool_out_w(self) -> int:
        return pool2d_out_size(self.conv_out_h, self.conv_out_w, self.pool_kernel, self.pool_kernel, self.pool_stride)[1]

    @property
    def pool_out_shape(self) -> TensorShape:
        return TensorShape(self.out_channels, self.pool_out_h, self.pool_out_w)

    @property
    def weight_count(self) -> int:
        return self.out_channels * self.in_shape.c * self.kernel_h * self.kernel_w

    @property
    def fan_in(self) -> int:
        return self.in_shape.c * self.kernel_h * self.kernel_w

    @property
    def im2col_count(self) -> int:
        return self.in_shape.c * self.kernel_h * self.kernel_w

    @property
    def conv_output_count_per_batch(self) -> int:
        return self.out_channels * self.conv_out_h * self.conv_out_w

    @property
    def pool_output_count_per_batch(self) -> int:
        return self.out_channels * self.pool_out_h * self.pool_out_w


@dataclass(frozen=True)
class DenseSpec:
    in_features: int
    out_features: int

    @property
    def weight_count(self) -> int:
        return self.in_features * self.out_features


# -----------------------------
# Cache dataclasses
# -----------------------------
@dataclass
class ConvBlockCache:
    input_nchw: int
    im2col: int
    conv_cnhw: int
    pool_cnhw: int
    pool_idx: int
    pool_nchw: int


@dataclass
class DenseCache:
    input_nchw: int
    logits: int


@dataclass
class ModelCache:
    block1: ConvBlockCache
    block2: ConvBlockCache
    dense: DenseCache


@dataclass
class ParamGrad:
    param: int
    grad: int
    size: int
    lr: float


@dataclass
class ConvBlockBackward:
    grad_input_nchw: int
    param_grads: list[ParamGrad]
    scratch: list[int]


@dataclass
class DenseBackward:
    grad_input_nchw: int
    param_grads: list[ParamGrad]
    scratch: list[int]


class SgdOptimizer:
    """Applies parameter updates after backward has produced explicit gradients."""

    def __init__(self, cuda: CudaLib):
        self.cuda = cuda

    def step(self, param_grads: list[ParamGrad]) -> None:
        for item in param_grads:
            self.cuda.lib.apply_sgd_update(item.param, item.grad, c_float(item.lr), item.size)


# -----------------------------
# Layers
# -----------------------------
class ConvBlock:
    def __init__(self, cuda: CudaLib, spec: ConvBlockSpec, alpha: float, seed: int):
        self.cuda = cuda
        self.spec = spec
        self.alpha = alpha

        rng = np.random.default_rng(seed)
        self.weight = cuda.upload_f32(he_init(rng, spec.weight_count, spec.fan_in))

    def close(self):
        self.cuda.free(self.weight)
        self.weight = None

    def state_dict(self, key: str) -> dict[str, np.ndarray]:
        shape = (
            self.spec.out_channels,
            self.spec.in_shape.c,
            self.spec.kernel_h,
            self.spec.kernel_w,
        )
        return {key: self.cuda.download_f32(self.weight, shape)}

    def load_state_dict(self, state: dict[str, np.ndarray], key: str):
        self.cuda.overwrite_f32(self.weight, np.ascontiguousarray(state[key], dtype=np.float32).reshape(-1))

    def forward_from_nchw(self, x_nchw_ptr, n: int) -> tuple[int, ConvBlockCache]:
        lib = self.cuda.lib
        s = self.spec

        d_col = self.cuda.malloc_f32(s.im2col_count * n * s.conv_out_h * s.conv_out_w)
        d_conv = self.cuda.malloc_f32(s.conv_output_count_per_batch * n)

        lib.im2col_forward(
            x_nchw_ptr, d_col,
            n, s.in_shape.c, s.in_shape.h, s.in_shape.w,
            s.kernel_h, s.kernel_w,
            s.conv_out_h, s.conv_out_w,
        )
        lib.gemm_forward(
            self.weight, d_col, d_conv,
            s.out_channels,
            n * s.conv_out_h * s.conv_out_w,
            s.im2col_count,
        )
        lib.leaky_relu_forward(d_conv, c_float(self.alpha), s.conv_output_count_per_batch * n)

        d_pool = self.cuda.malloc_f32(s.pool_output_count_per_batch * n)
        d_pool_idx = self.cuda.malloc_i32(s.pool_output_count_per_batch * n)
        lib.maxpool_forward_store(d_pool, d_conv, d_pool_idx, n, s.out_channels, s.conv_out_h, s.conv_out_w)

        d_pool_nchw = self.cuda.malloc_f32(s.pool_output_count_per_batch * n)
        lib.cnhw_to_nchw(d_pool, d_pool_nchw, n, s.out_channels, s.pool_out_h, s.pool_out_w)

        cache = ConvBlockCache(
            input_nchw=x_nchw_ptr,
            im2col=d_col,
            conv_cnhw=d_conv,
            pool_cnhw=d_pool,
            pool_idx=d_pool_idx,
            pool_nchw=d_pool_nchw,
        )
        return d_pool_nchw, cache

    def backward_to_nchw(self, grad_out_nchw_ptr, cache: ConvBlockCache, n: int, lr: float) -> ConvBlockBackward:
        lib = self.cuda.lib
        s = self.spec

        d_grad_pool_cnhw = self.cuda.malloc_f32(s.pool_output_count_per_batch * n)
        lib.nchw_to_cnhw(grad_out_nchw_ptr, d_grad_pool_cnhw, n, s.out_channels, s.pool_out_h, s.pool_out_w)

        d_grad_conv_cnhw = self.cuda.zeros_f32(s.conv_output_count_per_batch * n)
        lib.maxpool_backward_use_idx(
            d_grad_pool_cnhw, cache.pool_idx, d_grad_conv_cnhw,
            n, s.out_channels, s.conv_out_h, s.conv_out_w,
        )
        lib.leaky_relu_backward(
            cache.conv_cnhw, d_grad_conv_cnhw,
            c_float(self.alpha),
            s.conv_output_count_per_batch * n,
        )

        d_grad_w = self.cuda.zeros_f32(s.weight_count)
        d_grad_input_nchw = self.cuda.zeros_f32(n * s.in_shape.flat)

        lib.conv_backward(
            d_grad_conv_cnhw,
            cache.input_nchw,
            self.weight,
            d_grad_w,
            d_grad_input_nchw,
            n,
            s.in_shape.c,
            s.in_shape.h,
            s.in_shape.w,
            s.kernel_h,
            s.kernel_w,
            s.conv_out_h,
            s.conv_out_w,
            s.out_channels,
        )

        grad_w = self.cuda.download_f32(d_grad_w, (s.weight_count,)) / float(n)
        grad_w = np.clip(grad_w, -1.0, 1.0).astype(np.float32)
        self.cuda.overwrite_f32(d_grad_w, grad_w)
        return ConvBlockBackward(
            grad_input_nchw=d_grad_input_nchw,
            param_grads=[ParamGrad(self.weight, d_grad_w, s.weight_count, lr)],
            scratch=[grad_out_nchw_ptr, d_grad_pool_cnhw, d_grad_conv_cnhw],
        )

    def free_cache(self, cache: ConvBlockCache, free_input: bool = False):
        ptrs = [
            cache.im2col,
            cache.conv_cnhw,
            cache.pool_cnhw,
            cache.pool_idx,
            cache.pool_nchw,
        ]
        if free_input:
            ptrs.insert(0, cache.input_nchw)
        self.cuda.free_all(*ptrs)


class DenseLayer:
    def __init__(self, cuda: CudaLib, spec: DenseSpec, seed: int):
        self.cuda = cuda
        self.spec = spec
        rng = np.random.default_rng(seed)

        self.weight = cuda.upload_f32(he_init(rng, spec.weight_count, spec.in_features))
        self.bias = cuda.upload_f32(np.zeros(spec.out_features, dtype=np.float32))

    def close(self):
        self.cuda.free_all(self.weight, self.bias)
        self.weight = None
        self.bias = None

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "w_fc": self.cuda.download_f32(self.weight, (self.spec.out_features, self.spec.in_features)),
            "b_fc": self.cuda.download_f32(self.bias, (self.spec.out_features,)),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]):
        self.cuda.overwrite_f32(self.weight, np.ascontiguousarray(state["w_fc"], dtype=np.float32).reshape(-1))
        self.cuda.overwrite_f32(self.bias, np.ascontiguousarray(state["b_fc"], dtype=np.float32).reshape(-1))

    def forward(self, x_nchw_ptr, n: int) -> tuple[np.ndarray, DenseCache]:
        d_logits = self.cuda.malloc_f32(n * self.spec.out_features)
        self.cuda.lib.dense_forward(
            x_nchw_ptr, self.weight, self.bias, d_logits,
            n, self.spec.in_features, self.spec.out_features,
        )
        logits = self.cuda.download_f32(d_logits, (n, self.spec.out_features))
        return logits, DenseCache(input_nchw=x_nchw_ptr, logits=d_logits)

    def backward(self, grad_logits: np.ndarray, cache: DenseCache, n: int, lr: float) -> DenseBackward:
        d_grad_logits = self.cuda.upload_f32(grad_logits)
        d_grad_input = self.cuda.zeros_f32(n * self.spec.in_features)
        d_grad_w = self.cuda.zeros_f32(self.spec.weight_count)
        d_grad_b = self.cuda.zeros_f32(self.spec.out_features)

        self.cuda.lib.dense_backward_full(
            d_grad_logits,
            cache.input_nchw,
            self.weight,
            d_grad_input,
            d_grad_w,
            d_grad_b,
            n,
            self.spec.in_features,
            self.spec.out_features,
        )

        return DenseBackward(
            grad_input_nchw=d_grad_input,
            param_grads=[
                ParamGrad(self.weight, d_grad_w, self.spec.weight_count, lr),
                ParamGrad(self.bias, d_grad_b, self.spec.out_features, lr),
            ],
            scratch=[d_grad_logits, cache.logits],
        )

    def free_cache(self, cache: DenseCache, free_input: bool = False):
        ptrs = [cache.logits]
        if free_input:
            ptrs.insert(0, cache.input_nchw)
        self.cuda.free_all(*ptrs)


# -----------------------------
# Model
# -----------------------------
class Mnist2ConvModel:
    def __init__(self, cuda: CudaLib, seed: int, alpha: float = LEAKY_ALPHA):
        self.cuda = cuda
        self.alpha = alpha

        in_shape = TensorShape(1, 28, 28)
        self.block1_spec = ConvBlockSpec(in_shape=in_shape, out_channels=8, kernel_h=3, kernel_w=3)
        self.block2_spec = ConvBlockSpec(in_shape=self.block1_spec.pool_out_shape, out_channels=16, kernel_h=3, kernel_w=3)
        self.dense_spec = DenseSpec(in_features=self.block2_spec.pool_out_shape.flat, out_features=CLASSES)

        self.block1 = ConvBlock(cuda, self.block1_spec, alpha, seed + 1)
        self.block2 = ConvBlock(cuda, self.block2_spec, alpha, seed + 2)
        self.fc = DenseLayer(cuda, self.dense_spec, seed + 3)
        self.optimizer = SgdOptimizer(cuda)

    def arch_string(self) -> str:
        return (
            "Conv(1->8,3x3)->LeakyReLU->MaxPool"
            "->Conv(8->16,3x3)->LeakyReLU->MaxPool"
            f"->FC({self.dense_spec.in_features}->10)"
        )

    def dims_string(self) -> str:
        b1 = self.block1_spec
        b2 = self.block2_spec
        return (
            f"{b1.in_shape.h}x{b1.in_shape.w}"
            f" -> {b1.conv_out_h}x{b1.conv_out_w}"
            f" -> {b1.pool_out_h}x{b1.pool_out_w}"
            f" -> {b2.conv_out_h}x{b2.conv_out_w}"
            f" -> {b2.pool_out_h}x{b2.pool_out_w}"
            f" -> {self.dense_spec.in_features}"
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        state.update(self.block1.state_dict("w_conv1"))
        state.update(self.block2.state_dict("w_conv2"))
        state.update(self.fc.state_dict())
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]):
        self.block1.load_state_dict(state, "w_conv1")
        self.block2.load_state_dict(state, "w_conv2")
        self.fc.load_state_dict(state)

    def save_npz(self, path: Path, extra: dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        if extra:
            for k, v in extra.items():
                payload[k] = np.array(v)
        np.savez(path, **payload)

    def close(self):
        self.block1.close()
        self.block2.close()
        self.fc.close()

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ModelCache]:
        n = len(x)
        d_x = self.cuda.upload_f32(x)

        d_b1_out, cache1 = self.block1.forward_from_nchw(d_x, n)
        d_b2_out, cache2 = self.block2.forward_from_nchw(d_b1_out, n)
        logits, dense_cache = self.fc.forward(d_b2_out, n)

        return logits, ModelCache(
            block1=cache1,
            block2=cache2,
            dense=dense_cache,
        )

    def train_batch(self, x: np.ndarray, y: np.ndarray, lr_conv1: float, lr_conv2: float, lr_fc: float):
        n = len(x)
        logits, cache = self.forward(x)
        loss, acc, grad_logits = softmax_loss_and_grad(logits, y)

        dense_bw = self.fc.backward(grad_logits, cache.dense, n, lr_fc)
        block2_bw = self.block2.backward_to_nchw(dense_bw.grad_input_nchw, cache.block2, n, lr_conv2)
        block1_bw = self.block1.backward_to_nchw(block2_bw.grad_input_nchw, cache.block1, n, lr_conv1)

        self.optimizer.step(dense_bw.param_grads + block2_bw.param_grads + block1_bw.param_grads)
        self.cuda.free_all(
            block1_bw.grad_input_nchw,
            *dense_bw.scratch,
            *block2_bw.scratch,
            *block1_bw.scratch,
            *(item.grad for item in dense_bw.param_grads + block2_bw.param_grads + block1_bw.param_grads),
        )

        self.block2.free_cache(cache.block2, free_input=False)
        self.block1.free_cache(cache.block1, free_input=True)

        return loss, acc

    def eval_batch(self, x: np.ndarray, y: np.ndarray):
        logits, cache = self.forward(x)
        loss, acc, _ = softmax_loss_and_grad(logits, y)

        self.fc.free_cache(cache.dense, free_input=False)
        self.block2.free_cache(cache.block2, free_input=False)
        self.block1.free_cache(cache.block1, free_input=True)

        return loss, acc


# -----------------------------
# Eval
# -----------------------------
def run_eval(model: Mnist2ConvModel, x, y, batch_size: int, rng, limit: int | None):
    losses = []
    accs = []
    for xb, yb in batches(x, y, batch_size, False, rng, limit, drop_last=True):
        loss, acc = model.eval_batch(xb, yb)
        losses.append(loss)
        accs.append(acc)
    if not losses:
        raise RuntimeError("No evaluation batches were produced; reduce batch size or increase dataset/limit.")
    return float(np.mean(losses)), float(np.mean(accs))


# -----------------------------
# Main
# -----------------------------
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

    parser.add_argument("--lr-conv1", type=float, default=0.01)
    parser.add_argument("--lr-conv2", type=float, default=0.01)
    parser.add_argument("--lr-fc", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS / "latest")
    parser.add_argument("--no-save-best", action="store_true")

    args = parser.parse_args()

    if args.batch_size != DEFAULT_BATCH:
        raise SystemExit(
            f"This simple example currently expects --batch-size {DEFAULT_BATCH} "
            "(the helper code and smoke assumptions are still fixed around batch=64)."
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

    cuda = CudaLib(args.lib)
    model = Mnist2ConvModel(cuda, args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"split: train={len(x_train)} val={len(x_val)} test={len(x_test)}")
    print(f"arch: {model.arch_string()}")
    print(f"dims: {model.dims_string()}")
    print(f"limits: train={args.train_limit} val={args.val_limit} test={args.test_limit}")
    print(
        f"hyper: epochs={args.epochs} batch={args.batch_size} "
        f"lr_conv1={args.lr_conv1} lr_conv2={args.lr_conv2} lr_fc={args.lr_fc}"
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
            "lr_conv1": args.lr_conv1,
            "lr_conv2": args.lr_conv2,
            "lr_fc": args.lr_fc,
            "seed": args.seed,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "data_mean": data_mean,
            "data_std": data_std,
            "arch": model.arch_string(),
            "dims": model.dims_string(),
        },
    )

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
                loss, acc = model.train_batch(
                    xb, yb,
                    lr_conv1=args.lr_conv1,
                    lr_conv2=args.lr_conv2,
                    lr_fc=args.lr_fc,
                )
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
