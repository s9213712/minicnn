from __future__ import annotations

from typing import Any

import numpy as np

from minicnn.nn.tensor import Parameter, Tensor, _requires_grad


def relu(x: Tensor) -> Tensor:
    return x.relu()


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
    out_data = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = x_pad[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out_data[:, :, i, j] = np.tensordot(region, w_data, axes=([1, 2, 3], [1, 2, 3]))
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
        dw = np.zeros_like(w_data, dtype=np.float32)
        db = np.zeros((c_out,), dtype=np.float32) if bias is not None else None
        for i in range(out_h):
            for j in range(out_w):
                region = x_pad[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                g = grad[:, :, i, j]
                dw += np.tensordot(g, region, axes=([0], [0]))
                dx_pad[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw] += np.tensordot(g, w_data, axes=([1], [0]))
        if padding > 0:
            dx = dx_pad[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_pad
        x._add_grad(dx)
        weight._add_grad(dw)
        if bias is not None and db is not None:
            db += grad.sum(axis=(0, 2, 3))
            bias._add_grad(db)

    out._backward = _backward
    return out


def maxpool2d(x: Tensor, kernel_size: int = 2, stride: int | None = None) -> Tensor:
    stride = kernel_size if stride is None else stride
    n, c, h, w = x.data.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    out_data = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    max_indices: dict[tuple[int, int], np.ndarray] = {}
    for i in range(out_h):
        for j in range(out_w):
            region = x.data[:, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
            flat = region.reshape(n, c, -1)
            argmax = flat.argmax(axis=2)
            max_indices[(i, j)] = argmax
            out_data[:, :, i, j] = flat.max(axis=2)
    out = Tensor(out_data, requires_grad=_requires_grad(x))
    out._prev = {x}
    out._op = 'maxpool2d'

    def _backward() -> None:
        if out.grad is None:
            return
        dx = np.zeros_like(x.data, dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                argmax = max_indices[(i, j)]
                rows = argmax // kernel_size
                cols = argmax % kernel_size
                for ni in range(n):
                    for ci in range(c):
                        dx[ni, ci, i * stride + rows[ni, ci], j * stride + cols[ni, ci]] += out.grad[ni, ci, i, j]
        x._add_grad(dx)

    out._backward = _backward
    return out


def batchnorm2d(
    x: Tensor,
    weight: Parameter,
    bias: Parameter,
    eps: float = 1e-5,
) -> Tensor:
    axes = (0, 2, 3)
    mean = x.data.mean(axis=axes, keepdims=True)
    var = x.data.var(axis=axes, keepdims=True)
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
        m = np.prod([x.data.shape[a] for a in axes])
        gamma = weight.data.reshape(1, -1, 1, 1)
        dx_hat = grad * gamma
        dvar = (dx_hat * (x.data - mean) * -0.5 * (var + eps) ** -1.5).sum(axis=axes, keepdims=True)
        dmean = (dx_hat * -inv_std).sum(axis=axes, keepdims=True) + dvar * (-2.0 * (x.data - mean)).sum(axis=axes, keepdims=True) / m
        dx = dx_hat * inv_std + dvar * 2.0 * (x.data - mean) / m + dmean / m
        x._add_grad(dx)
        weight._add_grad((grad * x_hat).sum(axis=axes))
        bias._add_grad(grad.sum(axis=axes))

    out._backward = _backward
    return out


def ensure_tensor(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)
