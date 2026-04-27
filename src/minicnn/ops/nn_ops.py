from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from minicnn.nn.tensor import Parameter, Tensor, _requires_grad
from minicnn.random import get_global_rng


def relu(x: Tensor) -> Tensor:
    return x.relu()


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    out_data = np.where(x.data > 0, x.data, negative_slope * x.data).astype(np.float32)
    out = Tensor(out_data, requires_grad=_requires_grad(x))
    out._prev = {x}
    out._op = 'leaky_relu'

    def _backward() -> None:
        if out.grad is not None:
            mask = np.where(x.data > 0, 1.0, negative_slope).astype(np.float32)
            x._add_grad(out.grad * mask)

    out._backward = _backward
    return out


def silu(x: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-x.data))
    out_data = (x.data * s).astype(np.float32)
    out = Tensor(out_data, requires_grad=_requires_grad(x))
    out._prev = {x}
    out._op = 'silu'

    def _backward() -> None:
        if out.grad is not None:
            grad = (s + x.data * s * (1.0 - s)).astype(np.float32)
            x._add_grad(out.grad * grad)

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return x
    mask = (get_global_rng().random(x.data.shape) > p).astype(np.float32) / (1.0 - p)
    out = Tensor(x.data * mask, requires_grad=_requires_grad(x))
    out._prev = {x}
    out._op = 'dropout'

    def _backward() -> None:
        if out.grad is not None:
            x._add_grad(out.grad * mask)

    out._backward = _backward
    return out


def flatten(x: Tensor) -> Tensor:
    return x.reshape((x.data.shape[0], int(np.prod(x.data.shape[1:]))))


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    y = x @ weight
    if bias is not None:
        y = y + bias
    return y


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    x_data = x.data
    w_data = weight.data
    n, c_in, h, w = x_data.shape
    c_out, w_c_in, kh, kw = w_data.shape
    if c_in != w_c_in:
        raise ValueError(f'conv2d channel mismatch: input has {c_in}, weight expects {w_c_in}')
    x_pad = np.pad(x_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    windows = sliding_window_view(x_pad, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride, :, :]
    out_data = np.einsum('ncijhw,ochw->noij', windows, w_data, optimize=True).astype(np.float32)
    if bias is not None:
        out_data += bias.data.reshape(1, c_out, 1, 1)
    out = Tensor(out_data, requires_grad=_requires_grad(x, weight, *(tuple() if bias is None else (bias,))))
    out._prev = {x, weight} | ({bias} if bias is not None else set())
    out._op = 'conv2d'

    def _backward() -> None:
        if out.grad is None:
            return
        grad = out.grad
        dx_pad = np.zeros_like(x_pad, dtype=np.float32)
        dw = np.einsum('noij,ncijhw->ochw', grad, windows, optimize=True).astype(np.float32)
        dwindows = np.einsum('noij,ochw->ncijhw', grad, w_data, optimize=True).astype(np.float32)
        for ki in range(kh):
            row_slice = slice(ki, ki + out_h * stride, stride)
            for kj in range(kw):
                col_slice = slice(kj, kj + out_w * stride, stride)
                dx_pad[:, :, row_slice, col_slice] += dwindows[:, :, :, :, ki, kj]
        if padding > 0:
            dx = dx_pad[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_pad
        x._add_grad(dx)
        weight._add_grad(dw)
        if bias is not None:
            bias._add_grad(grad.sum(axis=(0, 2, 3)))

    out._backward = _backward
    return out


def maxpool2d(x: Tensor, kernel_size: int = 2, stride: int | None = None) -> Tensor:
    stride = kernel_size if stride is None else stride
    n, c, h, w = x.data.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    windows = sliding_window_view(x.data, (kernel_size, kernel_size), axis=(2, 3))[:, :, ::stride, ::stride]
    flat = windows.reshape(n, c, out_h, out_w, -1)
    argmax = flat.argmax(axis=-1)  # (n, c, out_h, out_w)
    out_data = flat.max(axis=-1).astype(np.float32)
    out = Tensor(out_data, requires_grad=_requires_grad(x))
    out._prev = {x}
    out._op = 'maxpool2d'

    def _backward() -> None:
        if out.grad is None:
            return
        dx = np.zeros_like(x.data, dtype=np.float32)
        krows = argmax // kernel_size
        kcols = argmax % kernel_size
        i_base = (np.arange(out_h) * stride).reshape(1, 1, out_h, 1)
        j_base = (np.arange(out_w) * stride).reshape(1, 1, 1, out_w)
        src_r = (i_base + krows).ravel()
        src_c = (j_base + kcols).ravel()
        n_idx = np.broadcast_to(np.arange(n).reshape(n, 1, 1, 1), (n, c, out_h, out_w)).ravel()
        c_idx = np.broadcast_to(np.arange(c).reshape(1, c, 1, 1), (n, c, out_h, out_w)).ravel()
        np.add.at(dx, (n_idx, c_idx, src_r, src_c), out.grad.ravel())
        x._add_grad(dx)

    out._backward = _backward
    return out


def avgpool2d(x: Tensor, kernel_size: int = 2, stride: int | None = None) -> Tensor:
    stride = kernel_size if stride is None else stride
    n, c, h, w = x.data.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    windows = sliding_window_view(x.data, (kernel_size, kernel_size), axis=(2, 3))[:, :, ::stride, ::stride]
    out_data = windows.mean(axis=(-2, -1)).astype(np.float32)
    out = Tensor(out_data, requires_grad=_requires_grad(x))
    out._prev = {x}
    out._op = 'avgpool2d'
    area = float(kernel_size * kernel_size)

    def _backward() -> None:
        if out.grad is None:
            return
        dx = np.zeros_like(x.data, dtype=np.float32)
        grad_distributed = out.grad / area
        for i in range(out_h):
            for j in range(out_w):
                r0 = i * stride
                c0 = j * stride
                dx[:, :, r0:r0 + kernel_size, c0:c0 + kernel_size] += grad_distributed[:, :, i:i + 1, j:j + 1]
        x._add_grad(dx)

    out._backward = _backward
    return out


def batchnorm2d(
    x: Tensor,
    weight: Parameter,
    bias: Parameter,
    eps: float = 1e-5,
    running_mean: np.ndarray | None = None,
    running_var: np.ndarray | None = None,
    training: bool = True,
    momentum: float = 0.1,
) -> Tensor:
    axes = (0, 2, 3)
    if training:
        mean = x.data.mean(axis=axes, keepdims=True)
        var = x.data.var(axis=axes, keepdims=True)
        if running_mean is not None:
            running_mean[...] = (1.0 - momentum) * running_mean + momentum * mean.reshape(-1)
        if running_var is not None:
            running_var[...] = (1.0 - momentum) * running_var + momentum * var.reshape(-1)
    else:
        if running_mean is None or running_var is None:
            raise ValueError('batchnorm2d evaluation requires running_mean and running_var')
        mean = running_mean.reshape(1, -1, 1, 1)
        var = running_var.reshape(1, -1, 1, 1)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x.data - mean) * inv_std
    out_data = x_hat * weight.data.reshape(1, -1, 1, 1) + bias.data.reshape(1, -1, 1, 1)
    out = Tensor(out_data, requires_grad=_requires_grad(x, weight, bias))
    out._prev = {x, weight, bias}
    out._op = 'batchnorm2d'

    def _backward() -> None:
        if out.grad is None:
            return
        grad = out.grad
        gamma = weight.data.reshape(1, -1, 1, 1)
        if training:
            m = np.prod([x.data.shape[a] for a in axes])
            dx_hat = grad * gamma
            dvar = (dx_hat * (x.data - mean) * -0.5 * (var + eps) ** -1.5).sum(axis=axes, keepdims=True)
            dmean = (dx_hat * -inv_std).sum(axis=axes, keepdims=True) + dvar * (-2.0 * (x.data - mean)).sum(axis=axes, keepdims=True) / m
            dx = dx_hat * inv_std + dvar * 2.0 * (x.data - mean) / m + dmean / m
        else:
            dx = grad * gamma * inv_std
        x._add_grad(dx)
        weight._add_grad((grad * x_hat).sum(axis=axes))
        bias._add_grad(grad.sum(axis=axes))

    out._backward = _backward
    return out


def ensure_tensor(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)
