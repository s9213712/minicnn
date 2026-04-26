from __future__ import annotations

import ctypes

import numpy as np
import pytest

from minicnn.cuda_native.device_runtime import DeviceRuntime
from minicnn.cuda_native.gpu_training import (
    native_gpu_avgpool_linear_training_step,
    native_gpu_batchnorm_linear_training_step,
    native_gpu_conv_linear_training_step,
    native_gpu_depthwise_layernorm2d_linear_training_step,
    native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step,
    native_gpu_depthwise_layernorm2d_pointwise_linear_training_step,
    native_gpu_global_avgpool_linear_training_step,
    native_gpu_groupnorm_linear_training_step,
    native_gpu_layernorm_linear_training_step,
    native_gpu_layernorm2d_linear_training_step,
    native_gpu_linear_training_step,
    native_gpu_pool_linear_training_step,
    native_gpu_two_conv_relu_pool_linear_training_step,
    native_gpu_two_linear_relu_training_step,
)


class _RawFakeCudaLib:
    def __init__(self):
        self.next_ptr = 4096
        self.memory: dict[int, bytearray] = {}

    def _float(self, ptr):
        return np.frombuffer(self.memory[int(ptr)], dtype=np.float32)

    def _int(self, ptr):
        return np.frombuffer(self.memory[int(ptr)], dtype=np.int32)

    def gpu_malloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 16
        self.memory[ptr] = bytearray(int(nbytes))
        return ptr

    def gpu_free(self, ptr):
        self.memory.pop(int(ptr), None)

    def gpu_memcpy_h2d(self, ptr, host_ptr, nbytes):
        self.memory[int(ptr)][:int(nbytes)] = ctypes.string_at(int(host_ptr), int(nbytes))

    def gpu_memcpy_d2h(self, host_ptr, ptr, nbytes):
        ctypes.memmove(int(host_ptr), bytes(self.memory[int(ptr)][:int(nbytes)]), int(nbytes))

    def gpu_memcpy_d2d(self, dst, src, nbytes):
        self.memory[int(dst)][:int(nbytes)] = bytes(self.memory[int(src)][:int(nbytes)])

    def gpu_memset(self, ptr, value, nbytes):
        self.memory[int(ptr)][:int(nbytes)] = bytes([int(value) & 0xFF]) * int(nbytes)

    def grad_l2_sumsq(self, d_grad, d_sumsq, size):
        self._float(d_sumsq)[0] += float(np.sum(self._float(d_grad)[:int(size)] ** 2))

    def scale_inplace(self, d_values, scale, size):
        self._float(d_values)[:int(size)] *= float(scale)

    def im2col_forward(self, d_input, d_col, n, c, h, w, kh, kw, out_h, out_w):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        col = np.zeros((int(c) * int(kh) * int(kw), int(n) * int(out_h) * int(out_w)), dtype=np.float32)
        col_idx = 0
        for ni in range(int(n)):
            for oh in range(int(out_h)):
                for ow in range(int(out_w)):
                    patch_idx = 0
                    for ci in range(int(c)):
                        for r in range(int(kh)):
                            for s in range(int(kw)):
                                col[patch_idx, col_idx] = x[ni, ci, oh + r, ow + s]
                                patch_idx += 1
                    col_idx += 1
        self._float(d_col)[:] = col.reshape(-1)

    def gemm_forward(self, d_a, d_b, d_c, m, n, k):
        a = self._float(d_a).reshape(int(m), int(k))
        b = self._float(d_b).reshape(int(k), int(n))
        self._float(d_c)[:] = (a @ b).reshape(-1)

    def cnhw_to_nchw(self, d_input, d_output, n, c, h, w):
        x = self._float(d_input).reshape(int(c), int(n), int(h), int(w))
        self._float(d_output)[:] = np.transpose(x, (1, 0, 2, 3)).reshape(-1)

    def nchw_to_cnhw(self, d_input, d_output, n, c, h, w):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        self._float(d_output)[:] = np.transpose(x, (1, 0, 2, 3)).reshape(-1)

    def depthwise_conv2d_forward(
        self,
        d_input,
        d_weight,
        d_bias,
        d_output,
        n,
        c,
        h,
        w,
        out_c,
        kh,
        kw,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        has_bias,
    ):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        weight = self._float(d_weight).reshape(int(out_c), 1, int(kh), int(kw))
        bias = self._float(d_bias).reshape(int(out_c))
        out = np.zeros((int(n), int(out_c), int(out_h), int(out_w)), dtype=np.float32)
        multiplier = int(out_c) // int(c)
        for ni in range(int(n)):
            for oc in range(int(out_c)):
                ic = oc // multiplier
                for oh in range(int(out_h)):
                    for ow in range(int(out_w)):
                        total = bias[oc] if int(has_bias) else 0.0
                        for r in range(int(kh)):
                            ih = oh * int(stride_h) + r - int(pad_h)
                            if ih < 0 or ih >= int(h):
                                continue
                            for s in range(int(kw)):
                                iw = ow * int(stride_w) + s - int(pad_w)
                                if iw < 0 or iw >= int(w):
                                    continue
                                total += x[ni, ic, ih, iw] * weight[oc, 0, r, s]
                        out[ni, oc, oh, ow] = total
        self._float(d_output)[:] = out.reshape(-1)

    def depthwise_conv2d_backward(
        self,
        d_grad_output,
        d_input,
        d_weight,
        d_grad_input,
        d_grad_weight,
        d_grad_bias,
        n,
        c,
        h,
        w,
        out_c,
        kh,
        kw,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        has_bias,
    ):
        grad_out = self._float(d_grad_output).reshape(int(n), int(out_c), int(out_h), int(out_w))
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        weight = self._float(d_weight).reshape(int(out_c), 1, int(kh), int(kw))
        grad_input = np.zeros((int(n), int(c), int(h), int(w)), dtype=np.float32)
        grad_weight = np.zeros((int(out_c), 1, int(kh), int(kw)), dtype=np.float32)
        grad_bias = np.zeros((int(out_c),), dtype=np.float32)
        multiplier = int(out_c) // int(c)
        for ni in range(int(n)):
            for oc in range(int(out_c)):
                ic = oc // multiplier
                for oh in range(int(out_h)):
                    for ow in range(int(out_w)):
                        grad_val = grad_out[ni, oc, oh, ow]
                        if int(has_bias):
                            grad_bias[oc] += grad_val
                        for r in range(int(kh)):
                            ih = oh * int(stride_h) + r - int(pad_h)
                            if ih < 0 or ih >= int(h):
                                continue
                            for s in range(int(kw)):
                                iw = ow * int(stride_w) + s - int(pad_w)
                                if iw < 0 or iw >= int(w):
                                    continue
                                grad_weight[oc, 0, r, s] += x[ni, ic, ih, iw] * grad_val
                                grad_input[ni, ic, ih, iw] += weight[oc, 0, r, s] * grad_val
        self._float(d_grad_input)[:] = grad_input.reshape(-1)
        self._float(d_grad_weight)[:] = grad_weight.reshape(-1)
        self._float(d_grad_bias)[:] = grad_bias.reshape(-1)

    def conv_backward(
        self,
        d_grad_out_cnhw,
        d_input,
        d_weight,
        d_grad_weight,
        d_grad_input,
        n,
        c,
        h,
        w,
        kh,
        kw,
        out_h,
        out_w,
        out_c,
    ):
        grad_cnhw = self._float(d_grad_out_cnhw).reshape(int(out_c), int(n), int(out_h), int(out_w))
        grad_out = np.transpose(grad_cnhw, (1, 0, 2, 3))
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        weight = self._float(d_weight).reshape(int(out_c), int(c), int(kh), int(kw))
        grad_weight = np.zeros_like(weight)
        grad_input = np.zeros_like(x)
        for ni in range(int(n)):
            for oc in range(int(out_c)):
                for oh in range(int(out_h)):
                    for ow in range(int(out_w)):
                        grad_val = grad_out[ni, oc, oh, ow]
                        for ci in range(int(c)):
                            for r in range(int(kh)):
                                for s in range(int(kw)):
                                    grad_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                    grad_input[ni, ci, oh + r, ow + s] += weight[oc, ci, r, s] * grad_val
        self._float(d_grad_weight)[:] = grad_weight.reshape(-1)
        self._float(d_grad_input)[:] = grad_input.reshape(-1)

    def dense_forward(self, d_input, d_weight, d_bias, d_output, n, in_f, out_f):
        x = self._float(d_input).reshape(int(n), int(in_f))
        w = self._float(d_weight).reshape(int(out_f), int(in_f))
        b = self._float(d_bias).reshape(int(out_f))
        self._float(d_output)[:] = (x @ w.T + b).reshape(-1)

    def apply_relu(self, d_data, size):
        data = self._float(d_data)
        data[:int(size)] = np.maximum(data[:int(size)], 0.0)

    def leaky_relu_forward(self, d_data, alpha, size):
        data = self._float(d_data)
        values = data[:int(size)]
        data[:int(size)] = np.where(values > 0.0, values, float(alpha) * values)

    def sigmoid_forward(self, d_data, size):
        data = self._float(d_data)
        data[:int(size)] = 1.0 / (1.0 + np.exp(-data[:int(size)]))

    def tanh_forward(self, d_data, size):
        data = self._float(d_data)
        data[:int(size)] = np.tanh(data[:int(size)])

    def silu_forward(self, d_data, size):
        data = self._float(d_data)
        values = data[:int(size)]
        data[:int(size)] = values / (1.0 + np.exp(-values))

    def gelu_forward(self, d_data, size):
        data = self._float(d_data)
        values = data[:int(size)]
        data[:int(size)] = 0.5 * values * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3))
        )

    def apply_relu_backward(self, d_data, d_grad, size):
        data = self._float(d_data)
        grad = self._float(d_grad)
        grad[:int(size)] = np.where(data[:int(size)] > 0.0, grad[:int(size)], 0.0)

    def leaky_relu_backward(self, d_data, d_grad, alpha, size):
        data = self._float(d_data)
        grad = self._float(d_grad)
        grad[:int(size)] *= np.where(data[:int(size)] > 0.0, 1.0, float(alpha))

    def sigmoid_backward(self, d_data, d_grad, size):
        data = self._float(d_data)
        grad = self._float(d_grad)
        sig = 1.0 / (1.0 + np.exp(-data[:int(size)]))
        grad[:int(size)] *= sig * (1.0 - sig)

    def tanh_backward(self, d_data, d_grad, size):
        data = self._float(d_data)
        grad = self._float(d_grad)
        t = np.tanh(data[:int(size)])
        grad[:int(size)] *= 1.0 - t * t

    def silu_backward(self, d_data, d_grad, size):
        data = self._float(d_data)
        grad = self._float(d_grad)
        x = data[:int(size)]
        sig = 1.0 / (1.0 + np.exp(-x))
        grad[:int(size)] *= sig + x * sig * (1.0 - sig)

    def gelu_backward(self, d_data, d_grad, size):
        data = self._float(d_data)
        grad = self._float(d_grad)
        x = data[:int(size)]
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        tanh_inner = np.tanh(inner)
        sech2_inner = 1.0 - tanh_inner * tanh_inner
        inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)
        deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * inner_grad
        grad[:int(size)] *= deriv

    def apply_maxpool(self, d_input, d_output, n, c, h, w):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        out_h = int(h) // 2
        out_w = int(w) // 2
        out = np.zeros((int(n), int(c), out_h, out_w), dtype=np.float32)
        for ni in range(int(n)):
            for ci in range(int(c)):
                for oh in range(out_h):
                    for ow in range(out_w):
                        out[ni, ci, oh, ow] = np.max(x[ni, ci, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2])
        self._float(d_output)[:] = out.reshape(-1)

    def global_avgpool2d_forward(self, d_input, d_output, n, c, h, w):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        self._float(d_output)[:] = x.mean(axis=(2, 3), keepdims=True).reshape(-1)

    def global_avgpool2d_backward(self, d_grad_output, d_grad_input, n, c, h, w):
        grad = self._float(d_grad_output).reshape(int(n), int(c), 1, 1)
        out = np.broadcast_to(grad / float(int(h) * int(w)), (int(n), int(c), int(h), int(w))).copy()
        self._float(d_grad_input)[:] = out.reshape(-1)

    def avgpool2d_forward(self, d_input, d_output, n, c, h, w, out_h, out_w, kh, kw, stride_h, stride_w, pad_h, pad_w):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        out = np.zeros((int(n), int(c), int(out_h), int(out_w)), dtype=np.float32)
        for ni in range(int(n)):
            for ci in range(int(c)):
                for oh in range(int(out_h)):
                    for ow in range(int(out_w)):
                        total = 0.0
                        for r in range(int(kh)):
                            ih = oh * int(stride_h) + r - int(pad_h)
                            if ih < 0 or ih >= int(h):
                                continue
                            for s in range(int(kw)):
                                iw = ow * int(stride_w) + s - int(pad_w)
                                if iw < 0 or iw >= int(w):
                                    continue
                                total += float(x[ni, ci, ih, iw])
                        out[ni, ci, oh, ow] = total / float(int(kh) * int(kw))
        self._float(d_output)[:] = out.reshape(-1)

    def avgpool2d_backward(self, d_grad_output, d_grad_input, n, c, h, w, out_h, out_w, kh, kw, stride_h, stride_w, pad_h, pad_w):
        grad_out = self._float(d_grad_output).reshape(int(n), int(c), int(out_h), int(out_w))
        grad_input = np.zeros((int(n), int(c), int(h), int(w)), dtype=np.float32)
        for ni in range(int(n)):
            for ci in range(int(c)):
                for oh in range(int(out_h)):
                    for ow in range(int(out_w)):
                        grad = grad_out[ni, ci, oh, ow] / float(int(kh) * int(kw))
                        for r in range(int(kh)):
                            ih = oh * int(stride_h) + r - int(pad_h)
                            if ih < 0 or ih >= int(h):
                                continue
                            for s in range(int(kw)):
                                iw = ow * int(stride_w) + s - int(pad_w)
                                if iw < 0 or iw >= int(w):
                                    continue
                                grad_input[ni, ci, ih, iw] += grad
        self._float(d_grad_input)[:] = grad_input.reshape(-1)

    def maxpool_backward_nchw(self, d_grad_out, d_input, d_grad_input, n, c, in_h, in_w, out_h, out_w):
        grad_out = self._float(d_grad_out).reshape(int(n), int(c), int(out_h), int(out_w))
        x = self._float(d_input).reshape(int(n), int(c), int(in_h), int(in_w))
        grad_input = np.zeros((int(n), int(c), int(in_h), int(in_w)), dtype=np.float32)
        for ni in range(int(n)):
            for ci in range(int(c)):
                for oh in range(int(out_h)):
                    for ow in range(int(out_w)):
                        window = x[ni, ci, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                        flat_idx = int(np.argmax(window))
                        ih = oh * 2 + flat_idx // 2
                        iw = ow * 2 + flat_idx % 2
                        grad_input[ni, ci, ih, iw] += grad_out[ni, ci, oh, ow]
        self._float(d_grad_input)[:] = grad_input.reshape(-1)

    def bn_train_forward(
        self,
        d_y,
        d_x,
        d_x_hat,
        d_batch_mean,
        d_batch_inv_std,
        d_running_mean,
        d_running_var,
        d_gamma,
        d_beta,
        n,
        c,
        h,
        w,
        eps,
        momentum,
    ):
        x = self._float(d_x).reshape(int(n), int(c), int(h), int(w))
        gamma = self._float(d_gamma).reshape(int(c))
        beta = self._float(d_beta).reshape(int(c))
        mean = x.mean(axis=(0, 2, 3)).astype(np.float32)
        var = x.var(axis=(0, 2, 3)).astype(np.float32)
        inv_std = (1.0 / np.sqrt(var + float(eps))).astype(np.float32)
        x_hat = ((x - mean.reshape(1, int(c), 1, 1)) * inv_std.reshape(1, int(c), 1, 1)).astype(np.float32)
        y = x_hat * gamma.reshape(1, int(c), 1, 1) + beta.reshape(1, int(c), 1, 1)
        self._float(d_y)[:] = y.astype(np.float32).reshape(-1)
        self._float(d_x_hat)[:] = x_hat.reshape(-1)
        self._float(d_batch_mean)[:] = mean
        self._float(d_batch_inv_std)[:] = inv_std
        self._float(d_running_mean)[:] = (1.0 - float(momentum)) * self._float(d_running_mean)[:int(c)] + float(momentum) * mean
        self._float(d_running_var)[:] = (1.0 - float(momentum)) * self._float(d_running_var)[:int(c)] + float(momentum) * var

    def bn_backward(self, d_dx, d_dgamma, d_dbeta, d_dy, d_x_hat, d_gamma, d_inv_std, n, c, h, w):
        grad_out = self._float(d_dy).reshape(int(n), int(c), int(h), int(w))
        x_hat = self._float(d_x_hat).reshape(int(n), int(c), int(h), int(w))
        gamma = self._float(d_gamma).reshape(int(c))
        inv_std = self._float(d_inv_std).reshape(1, int(c), 1, 1)
        grad_gamma = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
        grad_beta = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)
        elems = float(int(n) * int(h) * int(w))
        sum_grad = grad_out.sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        sum_grad_xhat = (grad_out * x_hat).sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        grad_input = (
            (gamma.reshape(1, int(c), 1, 1) * inv_std / elems)
            * (elems * grad_out - sum_grad - x_hat * sum_grad_xhat)
        )
        self._float(d_dx)[:] = grad_input.astype(np.float32).reshape(-1)
        self._float(d_dgamma)[:] = grad_gamma
        self._float(d_dbeta)[:] = grad_beta

    def layernorm2d_forward(self, d_input, d_gamma, d_beta, d_output, n, c, h, w, eps):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        gamma = self._float(d_gamma).reshape(int(c))
        beta = self._float(d_beta).reshape(int(c))
        mean = x.mean(axis=1, keepdims=True).astype(np.float32)
        var = x.var(axis=1, keepdims=True).astype(np.float32)
        out = ((x - mean) / np.sqrt(var + float(eps))) * gamma.reshape(1, int(c), 1, 1)
        out += beta.reshape(1, int(c), 1, 1)
        self._float(d_output)[:] = out.astype(np.float32).reshape(-1)

    def layernorm2d_backward(self, d_grad_output, d_input, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta, n, c, h, w, eps):
        grad_out = self._float(d_grad_output).reshape(int(n), int(c), int(h), int(w))
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        gamma = self._float(d_gamma).reshape(1, int(c), 1, 1)
        mean = x.mean(axis=1, keepdims=True).astype(np.float32)
        var = x.var(axis=1, keepdims=True).astype(np.float32)
        inv_std = (1.0 / np.sqrt(var + float(eps))).astype(np.float32)
        x_hat = ((x - mean) * inv_std).astype(np.float32)
        dxhat = (grad_out * gamma).astype(np.float32)
        sum_dxhat = dxhat.sum(axis=1, keepdims=True).astype(np.float32)
        sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True).astype(np.float32)
        grad_input = (inv_std / float(int(c))) * (float(int(c)) * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
        self._float(d_grad_input)[:] = grad_input.astype(np.float32).reshape(-1)
        self._float(d_grad_gamma)[:] = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
        self._float(d_grad_beta)[:] = grad_out.sum(axis=(0, 2, 3)).astype(np.float32)

    def layernorm_nd_forward(self, d_input, d_gamma, d_beta, d_output, rows, features, eps):
        x = self._float(d_input).reshape(int(rows), int(features))
        gamma = self._float(d_gamma).reshape(int(features))
        beta = self._float(d_beta).reshape(int(features))
        mean = x.mean(axis=1, keepdims=True).astype(np.float32)
        var = x.var(axis=1, keepdims=True).astype(np.float32)
        x_hat = ((x - mean) / np.sqrt(var + float(eps))).astype(np.float32)
        out = x_hat * gamma.reshape(1, int(features)) + beta.reshape(1, int(features))
        self._float(d_output)[:] = out.reshape(-1)

    def layernorm_nd_backward(self, d_grad_output, d_input, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta, rows, features, eps):
        grad_out = self._float(d_grad_output).reshape(int(rows), int(features))
        x = self._float(d_input).reshape(int(rows), int(features))
        gamma = self._float(d_gamma).reshape(1, int(features))
        mean = x.mean(axis=1, keepdims=True).astype(np.float32)
        var = x.var(axis=1, keepdims=True).astype(np.float32)
        inv_std = (1.0 / np.sqrt(var + float(eps))).astype(np.float32)
        x_hat = ((x - mean) * inv_std).astype(np.float32)
        dxhat = (grad_out * gamma).astype(np.float32)
        sum_dxhat = dxhat.sum(axis=1, keepdims=True).astype(np.float32)
        sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True).astype(np.float32)
        grad_input = (inv_std / float(int(features))) * (float(int(features)) * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
        self._float(d_grad_input)[:] = grad_input.astype(np.float32).reshape(-1)
        self._float(d_grad_gamma)[:] = (grad_out * x_hat).sum(axis=0).astype(np.float32)
        self._float(d_grad_beta)[:] = grad_out.sum(axis=0).astype(np.float32)

    def groupnorm_forward(self, d_input, d_gamma, d_beta, d_output, n, c, h, w, num_groups, eps):
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        gamma = self._float(d_gamma).reshape(int(c))
        beta = self._float(d_beta).reshape(int(c))
        channels_per_group = int(c) // int(num_groups)
        x_group = x.reshape(int(n), int(num_groups), channels_per_group, int(h), int(w))
        mean = x_group.mean(axis=(2, 3, 4), keepdims=True).astype(np.float32)
        var = x_group.var(axis=(2, 3, 4), keepdims=True).astype(np.float32)
        x_hat = ((x_group - mean) / np.sqrt(var + float(eps))).reshape(int(n), int(c), int(h), int(w))
        out = x_hat * gamma.reshape(1, int(c), 1, 1) + beta.reshape(1, int(c), 1, 1)
        self._float(d_output)[:] = out.astype(np.float32).reshape(-1)

    def groupnorm_backward(self, d_grad_output, d_input, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta, n, c, h, w, num_groups, eps):
        grad_out = self._float(d_grad_output).reshape(int(n), int(c), int(h), int(w))
        x = self._float(d_input).reshape(int(n), int(c), int(h), int(w))
        gamma = self._float(d_gamma).reshape(1, int(c), 1, 1)
        channels_per_group = int(c) // int(num_groups)
        x_group = x.reshape(int(n), int(num_groups), channels_per_group, int(h), int(w)).astype(np.float32)
        grad_out_group = grad_out.reshape(int(n), int(num_groups), channels_per_group, int(h), int(w)).astype(np.float32)
        mean = x_group.mean(axis=(2, 3, 4), keepdims=True).astype(np.float32)
        var = x_group.var(axis=(2, 3, 4), keepdims=True).astype(np.float32)
        inv_std = (1.0 / np.sqrt(var + float(eps))).astype(np.float32)
        x_hat_group = ((x_group - mean) * inv_std).astype(np.float32)
        x_hat = x_hat_group.reshape(int(n), int(c), int(h), int(w)).astype(np.float32)
        dxhat = (grad_out * gamma).reshape(int(n), int(num_groups), channels_per_group, int(h), int(w)).astype(np.float32)
        norm_elems = float(channels_per_group * int(h) * int(w))
        sum_dxhat = dxhat.sum(axis=(2, 3, 4), keepdims=True).astype(np.float32)
        sum_dxhat_xhat = (dxhat * x_hat_group).sum(axis=(2, 3, 4), keepdims=True).astype(np.float32)
        grad_group = (inv_std / norm_elems) * (norm_elems * dxhat - sum_dxhat - x_hat_group * sum_dxhat_xhat)
        self._float(d_grad_input)[:] = grad_group.reshape(int(n), int(c), int(h), int(w)).astype(np.float32).reshape(-1)
        self._float(d_grad_gamma)[:] = (grad_out * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
        self._float(d_grad_beta)[:] = grad_out_group.reshape(int(n), int(c), int(h), int(w)).sum(axis=(0, 2, 3)).astype(np.float32)

    def softmax_xent_grad_loss_acc(
        self,
        d_logits,
        d_labels,
        d_probs,
        d_grad_logits,
        d_loss_sum,
        d_correct_count,
        n,
        features,
    ):
        logits = self._float(d_logits).reshape(int(n), int(features))
        labels = self._int(d_labels)[:int(n)]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        grad = probs.copy()
        grad[np.arange(int(n)), labels] -= 1.0
        grad /= float(n)
        self._float(d_probs)[:] = probs.reshape(-1)
        self._float(d_grad_logits)[:] = grad.reshape(-1)
        self._float(d_loss_sum)[0] = float(-np.log(probs[np.arange(int(n)), labels] + 1e-10).sum())
        self._int(d_correct_count)[0] = int(np.sum(np.argmax(logits, axis=1) == labels))

    def softmax_xent_smooth_grad_loss_acc(
        self,
        d_logits,
        d_labels,
        d_probs,
        d_grad_logits,
        d_loss_sum,
        d_correct_count,
        n,
        features,
        label_smoothing,
    ):
        logits = self._float(d_logits).reshape(int(n), int(features))
        labels = self._int(d_labels)[:int(n)]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        targets = np.full_like(probs, float(label_smoothing) / float(int(features)))
        targets[np.arange(int(n)), labels] += 1.0 - float(label_smoothing)
        grad = (probs - targets) / float(n)
        self._float(d_probs)[:] = probs.reshape(-1)
        self._float(d_grad_logits)[:] = grad.reshape(-1)
        self._float(d_loss_sum)[0] = float(-np.sum(targets * np.log(probs + 1e-10)))
        self._int(d_correct_count)[0] = int(np.sum(np.argmax(logits, axis=1) == labels))

    def mse_fwd_grad_loss_acc(self, d_logits, d_labels, d_grad_logits, d_loss_sum, d_correct_count, n, features):
        logits = self._float(d_logits).reshape(int(n), int(features))
        labels = self._int(d_labels)[:int(n)]
        targets = np.zeros_like(logits)
        targets[np.arange(int(n)), labels] = 1.0
        diff = logits - targets
        self._float(d_grad_logits)[:] = (2.0 * diff / float(int(n) * int(features))).reshape(-1)
        self._float(d_loss_sum)[0] = float(np.mean(diff * diff, axis=1).sum())
        self._int(d_correct_count)[0] = int(np.sum(np.argmax(logits, axis=1) == labels))

    def bce_fwd_grad_loss_acc(self, d_logits, d_labels, d_grad_logits, d_loss_sum, d_correct_count, n):
        logits = self._float(d_logits).reshape(int(n), 1)
        labels = self._int(d_labels)[:int(n)].astype(np.float32).reshape(int(n), 1)
        sig = 1.0 / (1.0 + np.exp(-logits))
        self._float(d_grad_logits)[:] = ((sig - labels) / float(int(n))).reshape(-1)
        loss = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
        self._float(d_loss_sum)[0] = float(loss.sum())
        self._int(d_correct_count)[0] = int(np.sum((logits.reshape(-1) >= 0.0).astype(np.int32) == labels.reshape(-1).astype(np.int32)))

    def dense_backward_full(self, d_dout, d_input, d_weight, d_din, d_dweight, d_dbias, n, in_f, out_f):
        grad = self._float(d_dout).reshape(int(n), int(out_f))
        x = self._float(d_input).reshape(int(n), int(in_f))
        w = self._float(d_weight).reshape(int(out_f), int(in_f))
        self._float(d_din)[:] = (grad @ w).reshape(-1)
        self._float(d_dweight)[:] = (grad.T @ x).reshape(-1)
        self._float(d_dbias)[:] = grad.sum(axis=0).reshape(-1)

    def apply_sgd_update(self, d_weight, d_grad, lr, size):
        values = self._float(d_weight)
        grads = self._float(d_grad)
        values[:int(size)] -= float(lr) * grads[:int(size)]

    def apply_momentum_update(self, d_weight, d_grad, d_velocity, lr, momentum, size):
        values = self._float(d_weight)
        grads = self._float(d_grad)
        velocity = self._float(d_velocity)
        velocity[:int(size)] = float(momentum) * velocity[:int(size)] - float(lr) * grads[:int(size)]
        values[:int(size)] += velocity[:int(size)]

    def sgd_update_fused(self, d_weight, d_grad, d_velocity, lr, momentum, weight_decay, clip_val, normalizer, size):
        values = self._float(d_weight)
        grads = self._float(d_grad)
        velocity = self._float(d_velocity)
        g = grads[:int(size)] / float(normalizer)
        if float(clip_val) > 0.0:
            g = np.clip(g, -float(clip_val), float(clip_val))
        if float(momentum) > 0.0:
            g = g + float(weight_decay) * values[:int(size)]
            velocity[:int(size)] = float(momentum) * velocity[:int(size)] - float(lr) * g
            values[:int(size)] += velocity[:int(size)]
        else:
            next_values = values[:int(size)].copy()
            if float(weight_decay) > 0.0:
                next_values *= 1.0 - float(lr) * float(weight_decay)
            values[:int(size)] = next_values - float(lr) * g

    def adam_update_fused(
        self,
        d_weight,
        d_grad,
        d_m,
        d_v,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        clip_val,
        normalizer,
        bias_corr1,
        bias_corr2,
        size,
    ):
        values = self._float(d_weight)
        grads = self._float(d_grad)
        m = self._float(d_m)
        v = self._float(d_v)
        g = grads[:int(size)] / float(normalizer)
        if float(clip_val) > 0.0:
            g = np.clip(g, -float(clip_val), float(clip_val))
        m[:int(size)] = float(beta1) * m[:int(size)] + (1.0 - float(beta1)) * g
        v[:int(size)] = float(beta2) * v[:int(size)] + (1.0 - float(beta2)) * g * g
        update = float(lr) * (m[:int(size)] / float(bias_corr1)) / (np.sqrt(v[:int(size)] / float(bias_corr2)) + float(eps))
        values[:int(size)] -= update + float(lr) * float(weight_decay) * values[:int(size)]

    def rmsprop_update_fused(
        self,
        d_weight,
        d_grad,
        d_square_avg,
        d_momentum_buffer,
        lr,
        alpha,
        eps,
        momentum,
        weight_decay,
        clip_val,
        normalizer,
        size,
    ):
        values = self._float(d_weight)
        grads = self._float(d_grad)
        square_avg = self._float(d_square_avg)
        momentum_buffer = self._float(d_momentum_buffer)
        g = grads[:int(size)] / float(normalizer) + float(weight_decay) * values[:int(size)]
        if float(clip_val) > 0.0:
            g = np.clip(g, -float(clip_val), float(clip_val))
        square_avg[:int(size)] = float(alpha) * square_avg[:int(size)] + (1.0 - float(alpha)) * g * g
        step = g / (np.sqrt(square_avg[:int(size)]) + float(eps))
        if float(momentum) > 0.0:
            momentum_buffer[:int(size)] = float(momentum) * momentum_buffer[:int(size)] + step
            step = momentum_buffer[:int(size)]
        values[:int(size)] -= float(lr) * step


class _MissingSymbolCudaLib:
    def __init__(self, missing_symbol):
        self._inner = _RawFakeCudaLib()
        self._missing_symbol = str(missing_symbol)

    def __getattr__(self, name):
        if name == self._missing_symbol:
            raise AttributeError(name)
        return getattr(self._inner, name)


def _small_linear_training_case():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    return x, labels, weight, bias


def test_native_gpu_linear_training_step_reports_missing_adam_symbol():
    x, labels, weight, bias = _small_linear_training_case()

    with pytest.raises(ValueError, match='adam_update_fused'):
        native_gpu_linear_training_step(
            x,
            labels,
            weight,
            bias,
            lr=0.1,
            optimizer_type='adam',
            bound_lib=_MissingSymbolCudaLib('adam_update_fused'),
        )


def test_native_gpu_linear_training_step_reports_missing_rmsprop_symbol():
    x, labels, weight, bias = _small_linear_training_case()

    with pytest.raises(ValueError, match='rmsprop_update_fused'):
        native_gpu_linear_training_step(
            x,
            labels,
            weight,
            bias,
            lr=0.1,
            optimizer_type='rmsprop',
            bound_lib=_MissingSymbolCudaLib('rmsprop_update_fused'),
        )


def test_native_gpu_linear_training_step_reports_missing_momentum_symbol():
    x, labels, weight, bias = _small_linear_training_case()

    with pytest.raises(ValueError, match='apply_momentum_update'):
        native_gpu_linear_training_step(
            x,
            labels,
            weight,
            bias,
            lr=0.1,
            momentum=0.9,
            bound_lib=_MissingSymbolCudaLib('apply_momentum_update'),
        )


def test_native_gpu_linear_training_step_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    lr = 0.25

    result = native_gpu_linear_training_step(x, labels, weight, bias, lr=lr, bound_lib=_RawFakeCudaLib())

    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)

    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_logits @ weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_weight, grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias, grad_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.correct_count == int(np.sum(np.argmax(logits, axis=1) == labels))
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:softmax_xent_grad_loss_acc'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_backward_full'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_sgd_update'] == 1
    assert [item['kind'] for item in result.runtime_summary['execution_trace']] == [
        'gpu_native_train:dense_forward',
        'gpu_native_train:softmax_xent_grad_loss_acc',
        'gpu_native_train:dense_backward_full',
        'gpu_native_train:apply_sgd_update',
    ]


def test_native_gpu_linear_training_step_reuses_persistent_device_state():
    x = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    weight = np.asarray([[0.2, -0.1], [0.05, 0.3]], dtype=np.float32)
    bias = np.asarray([0.01, -0.02], dtype=np.float32)
    lr = 0.05
    runtime = DeviceRuntime(
        execution_mode='gpu_native',
        tensor_execution_device='gpu',
        bound_lib=_RawFakeCudaLib(),
    )

    first = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=lr,
        device_runtime=runtime,
        persistent_device_state=True,
        persistent_cache_prefix='linear_test',
    )
    first_h2d = first.runtime_summary['host_to_device_transfer_events']
    first_allocs = first.runtime_summary['device_pointer_allocation_events']

    second = native_gpu_linear_training_step(
        x,
        labels,
        first.updated_weight,
        first.updated_bias,
        lr=lr,
        weight_velocity=first.updated_weight_velocity,
        bias_velocity=first.updated_bias_velocity,
        device_runtime=runtime,
        persistent_device_state=True,
        persistent_cache_prefix='linear_test',
    )
    reference_second = native_gpu_linear_training_step(
        x,
        labels,
        first.updated_weight,
        first.updated_bias,
        lr=lr,
        weight_velocity=first.updated_weight_velocity,
        bias_velocity=first.updated_bias_velocity,
        bound_lib=_RawFakeCudaLib(),
    )

    np.testing.assert_allclose(second.updated_weight, reference_second.updated_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(second.updated_bias, reference_second.updated_bias, rtol=1e-6, atol=1e-6)
    assert second.runtime_summary['persistent_device_cache_entries'] == 12
    assert second.runtime_summary['persistent_device_cache_misses'] == 12
    assert second.runtime_summary['persistent_device_cache_hits'] == 12
    assert second.runtime_summary['host_to_device_transfer_events'] - first_h2d == 2
    assert second.runtime_summary['device_pointer_allocation_events'] - first_allocs < 14

    runtime.clear_persistent_device_cache()
    assert runtime.summary()['persistent_device_cache_entries'] == 0


def test_native_gpu_linear_training_step_can_skip_intermediate_host_copies():
    x = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)
    weight = np.asarray([[0.2, -0.1], [0.05, 0.3]], dtype=np.float32)
    bias = np.asarray([0.01, -0.02], dtype=np.float32)

    result = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=0.05,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_weight.shape == (0,)
    assert result.grad_bias.shape == (0,)
    assert result.updated_weight.shape == weight.shape
    assert result.updated_bias.shape == bias.shape
    assert result.updated_weight_velocity is None
    assert result.updated_weight_m is None
    assert result.updated_weight_rmsprop_v is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 4


def test_native_gpu_linear_training_step_grad_accum_mega_batch_matches_reference_math():
    x = np.asarray(
        [
            [1.0, 2.0, -1.0],
            [0.5, -1.5, 2.0],
            [-0.25, 0.75, 1.5],
            [2.0, 0.25, -0.5],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([2, 0, 1, 2], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    lr = 0.1

    result = native_gpu_linear_training_step(x, labels, weight, bias, lr=lr, bound_lib=_RawFakeCudaLib())

    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)

    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_weight, grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias, grad_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_trace'][0]['kind'] == 'gpu_native_train:dense_forward'
    assert result.runtime_summary['execution_trace'][-1]['kind'] == 'gpu_native_train:apply_sgd_update'


def test_native_gpu_linear_training_step_sgd_weight_decay_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    lr = 0.25
    weight_decay = 0.1

    result = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=lr,
        weight_decay=weight_decay,
        bound_lib=_RawFakeCudaLib(),
    )

    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)

    np.testing.assert_allclose(result.updated_weight, weight * (1.0 - lr * weight_decay) - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias * (1.0 - lr * weight_decay) - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:sgd_update_fused'] == 1


def test_native_gpu_linear_training_step_label_smoothing_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    lr = 0.05
    smoothing = 0.2

    result = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=lr,
        label_smoothing=smoothing,
        bound_lib=_RawFakeCudaLib(),
    )

    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    targets = np.full_like(probs, smoothing / float(probs.shape[1]))
    targets[np.arange(labels.shape[0]), labels] += 1.0 - smoothing
    grad_logits = (probs - targets) / float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)
    loss_sum = float(-np.sum(targets * np.log(probs + 1e-10)))

    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.loss_sum, loss_sum, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:softmax_xent_smooth_grad_loss_acc'] == 1


def test_native_gpu_linear_training_step_global_grad_clip_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    lr = 0.25
    max_norm = 0.25

    result = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=lr,
        grad_clip_value=max_norm,
        bound_lib=_RawFakeCudaLib(),
    )

    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)
    grad_norm = float(np.sqrt(np.sum(grad_weight ** 2) + np.sum(grad_bias ** 2)))
    clip_scale = max_norm / (grad_norm + 1e-12)
    clipped_weight_grad = grad_weight * clip_scale
    clipped_bias_grad = grad_bias * clip_scale

    np.testing.assert_allclose(result.grad_weight, clipped_weight_grad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias, clipped_bias_grad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * clipped_weight_grad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * clipped_bias_grad, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:grad_clip_global'] == 1


def test_native_gpu_linear_training_step_mse_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    lr = 0.05

    result = native_gpu_linear_training_step(x, labels, weight, bias, lr=lr, loss_type='mse', bound_lib=_RawFakeCudaLib())

    logits = x @ weight.T + bias
    targets = np.zeros_like(logits)
    targets[np.arange(labels.shape[0]), labels] = 1.0
    grad_logits = 2.0 * (logits - targets) / float(labels.shape[0] * logits.shape[1])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)

    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.loss_mean, float(np.mean((logits - targets) ** 2, axis=1).mean()), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:mse_fwd_grad_loss_acc'] == 1


def test_native_gpu_linear_training_step_bce_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([1, 0], dtype=np.int32)
    weight = np.asarray([[0.2, -0.1, 0.05]], dtype=np.float32)
    bias = np.asarray([0.01], dtype=np.float32)
    lr = 0.05

    result = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=lr,
        loss_type='bce_with_logits',
        bound_lib=_RawFakeCudaLib(),
    )

    logits = x @ weight.T + bias
    targets = labels.astype(np.float32).reshape(-1, 1)
    sig = 1.0 / (1.0 + np.exp(-logits))
    grad_logits = (sig - targets) / float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))

    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.loss_mean, float(loss.mean()), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:bce_fwd_grad_loss_acc'] == 1


def test_native_gpu_linear_training_step_adamw_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.99
    eps = 1e-8
    weight_decay = 0.1

    result = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=lr,
        optimizer_type='adamw',
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        step_index=1,
        bound_lib=_RawFakeCudaLib(),
    )

    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)
    weight_m = (1.0 - beta1) * grad_weight
    weight_v = (1.0 - beta2) * grad_weight * grad_weight
    bias_m = (1.0 - beta1) * grad_bias
    bias_v = (1.0 - beta2) * grad_bias * grad_bias
    weight_update = lr * (weight_m / (1.0 - beta1)) / (np.sqrt(weight_v / (1.0 - beta2)) + eps)
    bias_update = lr * (bias_m / (1.0 - beta1)) / (np.sqrt(bias_v / (1.0 - beta2)) + eps)

    np.testing.assert_allclose(result.updated_weight_m, weight_m, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight_v, weight_v, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias_m, bias_m, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias_v, bias_v, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - weight_update - lr * weight_decay * weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - bias_update - lr * weight_decay * bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:adam_update_fused'] == 1


def test_native_gpu_linear_training_step_rmsprop_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([2, 0], dtype=np.int32)
    weight = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    weight_rmsprop_v = np.full_like(weight, 0.02)
    weight_rmsprop_buf = np.full_like(weight, -0.03)
    bias_rmsprop_v = np.full_like(bias, 0.01)
    bias_rmsprop_buf = np.full_like(bias, 0.04)
    lr = 0.05
    alpha = 0.8
    eps = 1e-6
    momentum = 0.5
    weight_decay = 0.1

    result = native_gpu_linear_training_step(
        x,
        labels,
        weight,
        bias,
        lr=lr,
        optimizer_type='rmsprop',
        weight_decay=weight_decay,
        momentum=momentum,
        eps=eps,
        rmsprop_alpha=alpha,
        weight_rmsprop_v=weight_rmsprop_v,
        weight_rmsprop_buf=weight_rmsprop_buf,
        bias_rmsprop_v=bias_rmsprop_v,
        bias_rmsprop_buf=bias_rmsprop_buf,
        bound_lib=_RawFakeCudaLib(),
    )

    logits = x @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ x
    grad_bias = grad_logits.sum(axis=0)
    weight_rmsprop_grad = grad_weight + weight_decay * weight
    bias_rmsprop_grad = grad_bias + weight_decay * bias
    next_weight_v = alpha * weight_rmsprop_v + (1.0 - alpha) * weight_rmsprop_grad * weight_rmsprop_grad
    next_bias_v = alpha * bias_rmsprop_v + (1.0 - alpha) * bias_rmsprop_grad * bias_rmsprop_grad
    weight_step = weight_rmsprop_grad / (np.sqrt(next_weight_v) + eps)
    bias_step = bias_rmsprop_grad / (np.sqrt(next_bias_v) + eps)
    next_weight_buf = momentum * weight_rmsprop_buf + weight_step
    next_bias_buf = momentum * bias_rmsprop_buf + bias_step

    np.testing.assert_allclose(result.updated_weight_rmsprop_v, next_weight_v, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight_rmsprop_buf, next_weight_buf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias_rmsprop_v, next_bias_v, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias_rmsprop_buf, next_bias_buf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * next_weight_buf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * next_bias_buf, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:rmsprop_update_fused'] == 1


def test_native_gpu_two_linear_relu_training_step_matches_reference_math():
    x = np.asarray([[1.0, 2.0, -1.0], [0.5, -1.5, 2.0]], dtype=np.float32)
    labels = np.asarray([1, 0], dtype=np.int32)
    w1 = np.asarray(
        [
            [0.2, -0.1, 0.05],
            [-0.3, 0.4, 0.1],
            [0.15, -0.2, 0.3],
            [0.05, 0.25, -0.15],
        ],
        dtype=np.float32,
    )
    b1 = np.asarray([0.01, -0.02, 0.03, 0.04], dtype=np.float32)
    w2 = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05],
            [-0.05, 0.25, -0.15, 0.2],
        ],
        dtype=np.float32,
    )
    b2 = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.1

    result = native_gpu_two_linear_relu_training_step(
        x,
        labels,
        w1,
        b1,
        w2,
        b2,
        lr=lr,
        bound_lib=_RawFakeCudaLib(),
    )

    hidden_pre = x @ w1.T + b1
    hidden = np.maximum(hidden_pre, 0.0)
    logits = hidden @ w2.T + b2
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_w2 = grad_logits.T @ hidden
    grad_b2 = grad_logits.sum(axis=0)
    grad_hidden = grad_logits @ w2
    grad_hidden = np.where(hidden > 0.0, grad_hidden, 0.0)
    grad_w1 = grad_hidden.T @ x
    grad_b1 = grad_hidden.sum(axis=0)

    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_hidden, grad_hidden, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_hidden @ w1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_weight1, grad_w1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias1, grad_b1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_weight2, grad_w2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias2, grad_b2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight1, w1 - lr * grad_w1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias1, b1 - lr * grad_b1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight2, w2 - lr * grad_w2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias2, b2 - lr * grad_b2, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_forward_1'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_relu'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_forward_2'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_relu_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_backward_full_1'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_backward_full_2'] == 1


def test_native_gpu_two_linear_modern_activation_training_step_matches_reference_math():
    x = np.asarray([[0.2, -0.4, 0.6], [-0.1, 0.3, -0.5]], dtype=np.float32)
    labels = np.asarray([1, 0], dtype=np.int32)
    w1 = np.asarray([[0.2, -0.1, 0.05], [-0.3, 0.25, 0.15]], dtype=np.float32)
    b1 = np.asarray([0.01, -0.02], dtype=np.float32)
    w2 = np.asarray([[0.1, -0.2], [-0.05, 0.3]], dtype=np.float32)
    b2 = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.03

    def activate(values: np.ndarray, activation: str) -> tuple[np.ndarray, np.ndarray]:
        if activation == 'LeakyReLU':
            alpha = 0.2
            y = np.where(values > 0.0, values, alpha * values)
            return y.astype(np.float32), np.where(values > 0.0, 1.0, alpha).astype(np.float32)
        if activation == 'Sigmoid':
            y = 1.0 / (1.0 + np.exp(-values))
            return y.astype(np.float32), (y * (1.0 - y)).astype(np.float32)
        if activation == 'Tanh':
            y = np.tanh(values)
            return y.astype(np.float32), (1.0 - y * y).astype(np.float32)
        if activation == 'SiLU':
            sig = 1.0 / (1.0 + np.exp(-values))
            y = values * sig
            return y.astype(np.float32), (sig + values * sig * (1.0 - sig)).astype(np.float32)
        if activation == 'GELU':
            inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
            tanh_inner = np.tanh(inner)
            y = 0.5 * values * (1.0 + tanh_inner)
            sech2_inner = 1.0 - tanh_inner * tanh_inner
            inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * values ** 2)
            deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * values * sech2_inner * inner_grad
            return y.astype(np.float32), deriv.astype(np.float32)
        raise AssertionError(activation)

    for activation in ('LeakyReLU', 'Sigmoid', 'Tanh', 'SiLU', 'GELU'):
        result = native_gpu_two_linear_relu_training_step(
            x,
            labels,
            w1,
            b1,
            w2,
            b2,
            lr=lr,
            activation=activation,
            activation_alpha=0.2,
            bound_lib=_RawFakeCudaLib(),
        )

        hidden_pre = x @ w1.T + b1
        hidden, activation_grad = activate(hidden_pre, activation)
        logits = hidden @ w2.T + b2
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        grad_logits = probs.copy()
        grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
        grad_logits /= float(labels.shape[0])
        grad_w2 = grad_logits.T @ hidden
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = (grad_logits @ w2) * activation_grad
        grad_w1 = grad_hidden.T @ x
        grad_b1 = grad_hidden.sum(axis=0)

        np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.grad_hidden, grad_hidden, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.grad_input, grad_hidden @ w1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.grad_weight1, grad_w1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.grad_bias1, grad_b1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.grad_weight2, grad_w2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.grad_bias2, grad_b2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.updated_weight1, w1 - lr * grad_w1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.updated_bias1, b1 - lr * grad_b1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.updated_weight2, w2 - lr * grad_w2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.updated_bias2, b2 - lr * grad_b2, rtol=1e-6, atol=1e-6)
        if activation == 'LeakyReLU':
            assert result.runtime_summary['execution_kinds']['gpu_native_train:leaky_relu_forward'] == 1
            assert result.runtime_summary['execution_kinds']['gpu_native_train:leaky_relu_backward'] == 1
        runtime_key = 'leaky_relu' if activation == 'LeakyReLU' else activation.lower()
        assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{runtime_key}_forward'] == 1
        assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{runtime_key}_backward'] == 1


def test_native_gpu_two_linear_relu_training_step_can_skip_intermediate_host_copies():
    x = np.asarray([[0.5, -1.0], [1.5, 0.25]], dtype=np.float32)
    labels = np.asarray([1, 0], dtype=np.int32)
    w1 = np.asarray([[0.2, -0.1], [0.05, 0.3], [-0.25, 0.15]], dtype=np.float32)
    b1 = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
    w2 = np.asarray([[0.1, -0.2, 0.05], [-0.05, 0.25, -0.15]], dtype=np.float32)
    b2 = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_two_linear_relu_training_step(
        x,
        labels,
        w1,
        b1,
        w2,
        b2,
        lr=0.03,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_hidden.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_weight1.shape == (0,)
    assert result.grad_bias1.shape == (0,)
    assert result.grad_weight2.shape == (0,)
    assert result.grad_bias2.shape == (0,)
    assert result.updated_weight1.shape == w1.shape
    assert result.updated_bias1.shape == b1.shape
    assert result.updated_weight2.shape == w2.shape
    assert result.updated_bias2.shape == b2.shape
    assert result.updated_weight1_velocity is None
    assert result.updated_weight2_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 6


def test_native_gpu_pool_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0, -1.0, 0.0], [3.0, 4.0, 2.0, 1.0], [0.5, -0.5, 1.5, 2.5], [1.0, 0.0, 3.0, 2.0]]
            ],
            [
                [[-1.0, 1.0, 0.5, 2.0], [2.0, 0.0, 1.0, 3.0], [1.5, 2.5, -0.5, 0.5], [0.0, 1.0, 2.0, 4.0]]
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    weight = np.asarray([[0.1, -0.2, 0.3, 0.05], [-0.05, 0.25, -0.15, 0.2]], dtype=np.float32)
    bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.05

    result = native_gpu_pool_linear_training_step(x, labels, weight, bias, lr=lr, bound_lib=_RawFakeCudaLib())

    pooled = np.asarray(
        [
            [[[4.0, 2.0], [1.0, 3.0]]],
            [[[2.0, 3.0], [2.5, 4.0]]],
        ],
        dtype=np.float32,
    )
    flat = pooled.reshape(2, -1)
    logits = flat @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ flat
    grad_bias = grad_logits.sum(axis=0)
    grad_pooled = (grad_logits @ weight).reshape(pooled.shape)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oh in range(2):
            for ow in range(2):
                window = x[ni, 0, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                flat_idx = int(np.argmax(window))
                ih = oh * 2 + flat_idx // 2
                iw = ow * 2 + flat_idx % 2
                grad_input[ni, 0, ih, iw] += grad_pooled[ni, 0, oh, ow]

    np.testing.assert_allclose(result.pooled, pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pooled, grad_pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_weight, grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias, grad_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_maxpool'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:maxpool_backward_nchw'] == 1


def test_native_gpu_global_avgpool_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-1.0, 0.0], [1.0, 2.0]],
            ],
            [
                [[0.5, 1.5], [2.5, 3.5]],
                [[2.0, 1.0], [0.0, -1.0]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    weight = np.asarray([[0.1, -0.2], [-0.05, 0.25]], dtype=np.float32)
    bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.05

    result = native_gpu_global_avgpool_linear_training_step(x, labels, weight, bias, lr=lr, bound_lib=_RawFakeCudaLib())

    pooled = x.mean(axis=(2, 3), keepdims=True).astype(np.float32)
    flat = pooled.reshape(2, -1)
    logits = flat @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ flat
    grad_bias = grad_logits.sum(axis=0)
    grad_pooled = (grad_logits @ weight).reshape(pooled.shape)
    grad_input = np.broadcast_to(grad_pooled / float(x.shape[2] * x.shape[3]), x.shape).copy()

    np.testing.assert_allclose(result.pooled, pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pooled, grad_pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_weight, grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias, grad_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:global_avgpool2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:global_avgpool2d_backward'] == 1


def test_native_gpu_avgpool_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0, -1.0, 0.0], [3.0, 4.0, 2.0, 1.0], [0.5, -0.5, 1.5, 2.5], [1.0, 0.0, 3.0, 2.0]]
            ],
            [
                [[-1.0, 1.0, 0.5, 2.0], [2.0, 0.0, 1.0, 3.0], [1.5, 2.5, -0.5, 0.5], [0.0, 1.0, 2.0, 4.0]]
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    weight = np.asarray([[0.1, -0.2, 0.3, 0.05], [-0.05, 0.25, -0.15, 0.2]], dtype=np.float32)
    bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.05

    result = native_gpu_avgpool_linear_training_step(x, labels, weight, bias, lr=lr, bound_lib=_RawFakeCudaLib())

    pooled = np.asarray(
        [
            [[[2.5, 0.5], [0.25, 2.25]]],
            [[[0.5, 1.625], [1.25, 1.5]]],
        ],
        dtype=np.float32,
    )
    flat = pooled.reshape(2, -1)
    logits = flat @ weight.T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_weight = grad_logits.T @ flat
    grad_bias = grad_logits.sum(axis=0)
    grad_pooled = (grad_logits @ weight).reshape(pooled.shape)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oh in range(2):
            for ow in range(2):
                grad_input[ni, 0, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2] += grad_pooled[ni, 0, oh, ow] / 4.0

    np.testing.assert_allclose(result.pooled, pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pooled, grad_pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_weight, grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bias, grad_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_weight, weight - lr * grad_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bias, bias - lr * grad_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:avgpool2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:avgpool2d_backward'] == 1


@pytest.mark.parametrize(
    ('step_fn', 'weight'),
    [
        (
            native_gpu_pool_linear_training_step,
            np.asarray([[0.1, -0.2, 0.3, 0.05], [-0.05, 0.25, -0.15, 0.2]], dtype=np.float32),
        ),
        (
            native_gpu_avgpool_linear_training_step,
            np.asarray([[0.1, -0.2, 0.3, 0.05], [-0.05, 0.25, -0.15, 0.2]], dtype=np.float32),
        ),
        (
            native_gpu_global_avgpool_linear_training_step,
            np.asarray([[0.1], [-0.05]], dtype=np.float32),
        ),
    ],
)
def test_native_gpu_pool_family_training_steps_can_skip_intermediate_host_copies(step_fn, weight):
    x = np.asarray(
        [
            [[[1.0, 2.0, -1.0, 0.0], [3.0, 4.0, 2.0, 1.0], [0.5, -0.5, 1.5, 2.5], [1.0, 0.0, 3.0, 2.0]]],
            [[[-1.0, 1.0, 0.5, 2.0], [2.0, 0.0, 1.0, 3.0], [1.5, 2.5, -0.5, 0.5], [0.0, 1.0, 2.0, 4.0]]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = step_fn(
        x,
        labels,
        weight,
        bias,
        lr=0.05,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.pooled.shape == (0,)
    assert result.grad_pooled.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_weight.shape == (0,)
    assert result.grad_bias.shape == (0,)
    assert result.updated_weight.shape == weight.shape
    assert result.updated_bias.shape == bias.shape
    assert result.updated_weight_velocity is None
    assert result.updated_bias_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 4


def test_native_gpu_batchnorm_linear_training_step_matches_reference_math():
    x = (np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) - 6.0) / 5.0
    labels = np.asarray([1, 0], dtype=np.int32)
    bn_weight = np.asarray([1.0, 1.5], dtype=np.float32)
    bn_bias = np.asarray([-0.25, 0.5], dtype=np.float32)
    running_mean = np.asarray([0.1, -0.2], dtype=np.float32)
    running_var = np.asarray([1.2, 0.8], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.15, -0.05],
            [-0.05, 0.25, -0.15, 0.2, 0.12, -0.18, 0.08, 0.04],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.05
    eps = 1e-5
    bn_momentum = 0.2

    result = native_gpu_batchnorm_linear_training_step(
        x,
        labels,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        linear_weight,
        linear_bias,
        lr=lr,
        bn_eps=eps,
        bn_momentum=bn_momentum,
        bound_lib=_RawFakeCudaLib(),
    )

    mean = x.mean(axis=(0, 2, 3)).astype(np.float32)
    var = x.var(axis=(0, 2, 3)).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean.reshape(1, 2, 1, 1)) * inv_std.reshape(1, 2, 1, 1)).astype(np.float32)
    bn_output = x_hat * bn_weight.reshape(1, 2, 1, 1) + bn_bias.reshape(1, 2, 1, 1)
    flat = bn_output.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_bn_output = (grad_logits @ linear_weight).reshape(bn_output.shape)
    grad_bn_weight = (grad_bn_output * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_bn_bias = grad_bn_output.sum(axis=(0, 2, 3)).astype(np.float32)
    elems = float(x.shape[0] * x.shape[2] * x.shape[3])
    sum_grad = grad_bn_output.sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    sum_grad_xhat = (grad_bn_output * x_hat).sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    grad_input = (
        (bn_weight.reshape(1, 2, 1, 1) * inv_std.reshape(1, 2, 1, 1) / elems)
        * (elems * grad_bn_output - sum_grad - x_hat * sum_grad_xhat)
    )
    next_running_mean = (1.0 - bn_momentum) * running_mean + bn_momentum * mean
    next_running_var = (1.0 - bn_momentum) * running_var + bn_momentum * var

    np.testing.assert_allclose(result.bn_output, bn_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.x_hat, x_hat, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.batch_mean, mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.batch_inv_std, inv_std, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bn_output, grad_bn_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bn_weight, grad_bn_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_bn_bias, grad_bn_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bn_weight, bn_weight - lr * grad_bn_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_bn_bias, bn_bias - lr * grad_bn_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_running_mean, next_running_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_running_var, next_running_var, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:bn_train_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:bn_backward'] == 1


def test_native_gpu_batchnorm_linear_training_step_can_skip_intermediate_host_copies():
    x = (np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) - 6.0) / 5.0
    labels = np.asarray([1, 0], dtype=np.int32)
    bn_weight = np.asarray([1.0, 1.5], dtype=np.float32)
    bn_bias = np.asarray([-0.25, 0.5], dtype=np.float32)
    running_mean = np.asarray([0.1, -0.2], dtype=np.float32)
    running_var = np.asarray([1.2, 0.8], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.15, -0.05],
            [-0.05, 0.25, -0.15, 0.2, 0.12, -0.18, 0.08, 0.04],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_batchnorm_linear_training_step(
        x,
        labels,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        linear_weight,
        linear_bias,
        lr=0.05,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.bn_output.shape == (0,)
    assert result.x_hat.shape == (0,)
    assert result.batch_mean.shape == (0,)
    assert result.batch_inv_std.shape == (0,)
    assert result.grad_bn_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_bn_weight.shape == (0,)
    assert result.grad_bn_bias.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.updated_bn_weight.shape == bn_weight.shape
    assert result.updated_bn_bias.shape == bn_bias.shape
    assert result.updated_running_mean.shape == running_mean.shape
    assert result.updated_running_var.shape == running_var.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_bn_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 8


def test_native_gpu_layernorm2d_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ],
            [
                [[-1.0, 0.5], [2.0, -0.5]],
                [[1.5, -1.0], [0.0, 2.5]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    norm_weight = np.asarray([1.0, 1.5], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.25], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    eps = 1e-5
    lr = 0.04

    result = native_gpu_layernorm2d_linear_training_step(
        x,
        labels,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=lr,
        norm_eps=eps,
        bound_lib=_RawFakeCudaLib(),
    )

    mean = x.mean(axis=1, keepdims=True).astype(np.float32)
    var = x.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((x - mean) * inv_std).astype(np.float32)
    norm = x_hat * norm_weight.reshape(1, 2, 1, 1) + norm_bias.reshape(1, 2, 1, 1)
    flat = norm.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_norm = (grad_logits @ linear_weight).reshape(norm.shape)
    grad_norm_weight = (grad_norm * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = grad_norm * norm_weight.reshape(1, 2, 1, 1)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_input = (inv_std / 2.0) * (2.0 * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)

    np.testing.assert_allclose(result.norm_output, norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_backward'] == 1


def test_native_gpu_layernorm2d_linear_training_step_can_skip_intermediate_host_copies():
    x = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ],
            [
                [[-1.0, 0.5], [2.0, -0.5]],
                [[1.5, -1.0], [0.0, 2.5]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    norm_weight = np.asarray([1.0, 1.5], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.25], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_layernorm2d_linear_training_step(
        x,
        labels,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=0.04,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.norm_output.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_norm_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_norm_weight.shape == (0,)
    assert result.grad_norm_bias.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.updated_norm_weight.shape == norm_weight.shape
    assert result.updated_norm_bias.shape == norm_bias.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_norm_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 6


def test_native_gpu_layernorm_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ],
            [
                [[-1.0, 0.5], [2.0, -0.5]],
                [[1.5, -1.0], [0.0, 2.5]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    norm_weight = np.asarray([1.0, 1.5, 0.5, 1.25, 0.8, 1.1, 0.9, 1.3], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.25, 0.1, -0.05, 0.2, -0.15, 0.05, -0.1], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    eps = 1e-5
    lr = 0.04

    result = native_gpu_layernorm_linear_training_step(
        x,
        labels,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=lr,
        normalized_shape=8,
        norm_eps=eps,
        bound_lib=_RawFakeCudaLib(),
    )

    flat = x.reshape(2, -1)
    mean = flat.mean(axis=1, keepdims=True).astype(np.float32)
    var = flat.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((flat - mean) * inv_std).astype(np.float32)
    norm = x_hat * norm_weight.reshape(1, 8) + norm_bias.reshape(1, 8)
    logits = norm @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ norm
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_norm = grad_logits @ linear_weight
    grad_norm_weight = (grad_norm * x_hat).sum(axis=0).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=0).astype(np.float32)
    dxhat = grad_norm * norm_weight.reshape(1, 8)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_input = (inv_std / 8.0) * (8.0 * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)

    np.testing.assert_allclose(result.norm_output, norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input.reshape(x.shape), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm_backward'] == 1


def test_native_gpu_layernorm_linear_training_step_can_skip_intermediate_host_copies():
    x = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ],
            [
                [[-1.0, 0.5], [2.0, -0.5]],
                [[1.5, -1.0], [0.0, 2.5]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    norm_weight = np.asarray([1.0, 1.5, 0.5, 1.25, 0.8, 1.1, 0.9, 1.3], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.25, 0.1, -0.05, 0.2, -0.15, 0.05, -0.1], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_layernorm_linear_training_step(
        x,
        labels,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=0.04,
        activation='SiLU',
        normalized_shape=8,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.norm_output.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_norm_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_norm_weight.shape == (0,)
    assert result.grad_norm_bias.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.updated_norm_weight.shape == norm_weight.shape
    assert result.updated_norm_bias.shape == norm_bias.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_norm_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 6


def test_native_gpu_layernorm_silu_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ],
            [
                [[-1.0, 0.5], [2.0, -0.5]],
                [[1.5, -1.0], [0.0, 2.5]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    norm_weight = np.asarray([1.0, 1.5, 0.5, 1.25, 0.8, 1.1, 0.9, 1.3], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.25, 0.1, -0.05, 0.2, -0.15, 0.05, -0.1], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    eps = 1e-5
    lr = 0.04

    result = native_gpu_layernorm_linear_training_step(
        x,
        labels,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=lr,
        activation='SiLU',
        normalized_shape=8,
        norm_eps=eps,
        bound_lib=_RawFakeCudaLib(),
    )

    flat = x.reshape(2, -1)
    mean = flat.mean(axis=1, keepdims=True).astype(np.float32)
    var = flat.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((flat - mean) * inv_std).astype(np.float32)
    norm = x_hat * norm_weight.reshape(1, 8) + norm_bias.reshape(1, 8)
    sig = 1.0 / (1.0 + np.exp(-norm))
    activated = norm * sig
    logits = activated @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ activated
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_activation = grad_logits @ linear_weight
    silu_grad = sig + norm * sig * (1.0 - sig)
    grad_norm = grad_activation * silu_grad
    grad_norm_weight = (grad_norm * x_hat).sum(axis=0).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=0).astype(np.float32)
    dxhat = grad_norm * norm_weight.reshape(1, 8)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_input = (inv_std / 8.0) * (8.0 * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)

    np.testing.assert_allclose(result.norm_output, activated, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input.reshape(x.shape), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:silu_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:silu_backward'] == 1


def test_native_gpu_groupnorm_linear_training_step_matches_reference_math():
    x = (np.arange(64, dtype=np.float32).reshape(2, 4, 2, 4) - 20.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    norm_weight = np.asarray([1.0, 1.5, 0.75, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.25, 0.1, -0.15], dtype=np.float32)
    linear_weight = np.linspace(-0.2, 0.25, num=64, dtype=np.float32).reshape(2, 32)
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    num_groups = 2
    eps = 1e-5
    lr = 0.03

    result = native_gpu_groupnorm_linear_training_step(
        x,
        labels,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=lr,
        num_groups=num_groups,
        norm_eps=eps,
        bound_lib=_RawFakeCudaLib(),
    )

    n, c, h, w = x.shape
    channels_per_group = c // num_groups
    x_group = x.reshape(n, num_groups, channels_per_group, h, w)
    mean = x_group.mean(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    var = x_group.var(axis=(2, 3, 4), keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat_group = ((x_group - mean) * inv_std).astype(np.float32)
    x_hat = x_hat_group.reshape(x.shape)
    norm = x_hat * norm_weight.reshape(1, c, 1, 1) + norm_bias.reshape(1, c, 1, 1)
    flat = norm.reshape(n, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_norm = (grad_logits @ linear_weight).reshape(norm.shape)
    grad_norm_weight = (grad_norm * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = (grad_norm * norm_weight.reshape(1, c, 1, 1)).reshape(n, num_groups, channels_per_group, h, w)
    norm_elems = float(channels_per_group * h * w)
    sum_dxhat = dxhat.sum(axis=(2, 3, 4), keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat_group).sum(axis=(2, 3, 4), keepdims=True)
    grad_input = (inv_std / norm_elems) * (norm_elems * dxhat - sum_dxhat - x_hat_group * sum_dxhat_xhat)
    grad_input = grad_input.reshape(x.shape)

    np.testing.assert_allclose(result.norm_output, norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:groupnorm_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:groupnorm_backward'] == 1


def test_native_gpu_groupnorm_linear_training_step_can_skip_intermediate_host_copies():
    x = (np.arange(64, dtype=np.float32).reshape(2, 4, 2, 4) - 20.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    norm_weight = np.asarray([1.0, 1.5, 0.75, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.25, 0.1, -0.15], dtype=np.float32)
    linear_weight = np.linspace(-0.2, 0.25, num=64, dtype=np.float32).reshape(2, 32)
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_groupnorm_linear_training_step(
        x,
        labels,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=0.03,
        num_groups=2,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.norm_output.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_norm_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_norm_weight.shape == (0,)
    assert result.grad_norm_bias.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.updated_norm_weight.shape == norm_weight.shape
    assert result.updated_norm_bias.shape == norm_bias.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_norm_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 6


def test_native_gpu_conv_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [[[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]]],
            [[[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    conv_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.07

    result = native_gpu_conv_linear_training_step(
        x,
        labels,
        conv_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        bound_lib=_RawFakeCudaLib(),
    )

    conv = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    conv[ni, oc, oh, ow] = np.sum(x[ni, :, oh:oh + 2, ow:ow + 2] * conv_weight[oc])
    flat = conv.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_conv_output = (grad_logits @ linear_weight).reshape(conv.shape)
    grad_conv_weight = np.zeros_like(conv_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_conv_output[ni, oc, oh, ow]
                    for ci in range(x.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                grad_input[ni, ci, oh + r, ow + s] += conv_weight[oc, ci, r, s] * grad_val

    np.testing.assert_allclose(result.conv_output, conv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_output, grad_conv_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_weight, grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_conv_weight, conv_weight - lr * grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_backward_full'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv_backward'] == 1


def test_native_gpu_depthwise_conv_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]],
                [[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]],
            ],
            [
                [[0.5, -1.0, 1.0], [2.0, 0.0, -0.5], [1.5, 2.5, -1.5]],
                [[1.0, -0.5, 0.25], [-1.25, 1.5, 2.0], [0.75, -2.0, 1.25]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    conv_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.06

    result = native_gpu_conv_linear_training_step(
        x,
        labels,
        conv_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        conv_kind='depthwise',
        bound_lib=_RawFakeCudaLib(),
    )

    conv = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    conv[ni, oc, oh, ow] = np.sum(x[ni, ic, oh:oh + 2, ow:ow + 2] * conv_weight[oc, 0])
    flat = conv.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_conv_output = (grad_logits @ linear_weight).reshape(conv.shape)
    grad_conv_weight = np.zeros_like(conv_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_conv_output[ni, oc, oh, ow]
                    for r in range(2):
                        for s in range(2):
                            grad_conv_weight[oc, 0, r, s] += x[ni, ic, oh + r, ow + s] * grad_val
                            grad_input[ni, ic, oh + r, ow + s] += conv_weight[oc, 0, r, s] * grad_val

    np.testing.assert_allclose(result.conv_output, conv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_output, grad_conv_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_weight, grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_conv_weight, conv_weight - lr * grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:dense_backward_full'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_backward'] == 1


@pytest.mark.parametrize('conv_kind', ['conv2d', 'depthwise'])
def test_native_gpu_conv_linear_training_step_can_skip_intermediate_host_copies(conv_kind):
    if conv_kind == 'conv2d':
        x = np.asarray(
            [
                [[[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]]],
                [[[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]]],
            ],
            dtype=np.float32,
        )
        conv_weight = np.asarray(
            [
                [[[0.2, -0.1], [0.05, 0.3]]],
                [[[-0.2, 0.1], [0.25, -0.05]]],
            ],
            dtype=np.float32,
        )
    else:
        x = np.asarray(
            [
                [
                    [[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]],
                    [[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]],
                ],
                [
                    [[0.5, -1.0, 1.0], [2.0, 0.0, -0.5], [1.5, 2.5, -1.5]],
                    [[1.0, -0.5, 0.25], [-1.25, 1.5, 2.0], [0.75, -2.0, 1.25]],
                ],
            ],
            dtype=np.float32,
        )
        conv_weight = np.asarray(
            [
                [[[0.2, -0.1], [0.05, 0.3]]],
                [[[-0.2, 0.1], [0.25, -0.05]]],
            ],
            dtype=np.float32,
        )
    labels = np.asarray([1, 0], dtype=np.int32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_conv_linear_training_step(
        x,
        labels,
        conv_weight,
        linear_weight,
        linear_bias,
        lr=0.05,
        conv_kind=conv_kind,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.conv_output.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_conv_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_conv_weight.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.pooled_output is None
    assert result.grad_pooled is None
    assert result.updated_conv_weight.shape == conv_weight.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_conv_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 5


def test_native_gpu_depthwise_layernorm2d_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [
                [[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]],
                [[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]],
            ],
            [
                [[0.5, -1.0, 1.0], [2.0, 0.0, -0.5], [1.5, 2.5, -1.5]],
                [[1.0, -0.5, 0.25], [-1.25, 1.5, 2.0], [0.75, -2.0, 1.25]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    conv_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    norm_weight = np.asarray([1.0, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.1], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    eps = 1e-5
    lr = 0.05

    result = native_gpu_depthwise_layernorm2d_linear_training_step(
        x,
        labels,
        conv_weight,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=lr,
        norm_eps=eps,
        bound_lib=_RawFakeCudaLib(),
    )

    conv = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    conv[ni, oc, oh, ow] = np.sum(x[ni, ic, oh:oh + 2, ow:ow + 2] * conv_weight[oc, 0])
    mean = conv.mean(axis=1, keepdims=True).astype(np.float32)
    var = conv.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((conv - mean) * inv_std).astype(np.float32)
    norm = x_hat * norm_weight.reshape(1, 2, 1, 1) + norm_bias.reshape(1, 2, 1, 1)
    flat = norm.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_norm = (grad_logits @ linear_weight).reshape(norm.shape)
    grad_norm_weight = (grad_norm * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = grad_norm * norm_weight.reshape(1, 2, 1, 1)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_conv = (inv_std / 2.0) * (2.0 * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    grad_conv_weight = np.zeros_like(conv_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_conv[ni, oc, oh, ow]
                    for r in range(2):
                        for s in range(2):
                            grad_conv_weight[oc, 0, r, s] += x[ni, ic, oh + r, ow + s] * grad_val
                            grad_input[ni, ic, oh + r, ow + s] += conv_weight[oc, 0, r, s] * grad_val

    np.testing.assert_allclose(result.conv_output, conv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.norm_output, norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_output, grad_conv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_weight, grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_conv_weight, conv_weight - lr * grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_backward'] == 1


def test_native_gpu_depthwise_layernorm2d_linear_training_step_can_skip_intermediate_host_copies():
    x = np.asarray(
        [
            [
                [[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]],
                [[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]],
            ],
            [
                [[0.5, -1.0, 1.0], [2.0, 0.0, -0.5], [1.5, 2.5, -1.5]],
                [[1.0, -0.5, 0.25], [-1.25, 1.5, 2.0], [0.75, -2.0, 1.25]],
            ],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    conv_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    norm_weight = np.asarray([1.0, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.1], dtype=np.float32)
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_depthwise_layernorm2d_linear_training_step(
        x,
        labels,
        conv_weight,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        lr=0.05,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.conv_output.shape == (0,)
    assert result.norm_output.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_norm_output.shape == (0,)
    assert result.grad_conv_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_conv_weight.shape == (0,)
    assert result.grad_norm_weight.shape == (0,)
    assert result.grad_norm_bias.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.updated_conv_weight.shape == conv_weight.shape
    assert result.updated_norm_weight.shape == norm_weight.shape
    assert result.updated_norm_bias.shape == norm_bias.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_conv_weight_velocity is None
    assert result.updated_norm_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 7


def test_native_gpu_depthwise_layernorm2d_pointwise_linear_training_step_matches_reference_math():
    x = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) - 12.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    depthwise_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    norm_weight = np.asarray([1.0, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.1], dtype=np.float32)
    pointwise_weight = np.asarray(
        [
            [[[0.1]], [[-0.2]]],
            [[[0.25]], [[0.05]]],
            [[[-0.15]], [[0.2]]],
        ],
        dtype=np.float32,
    )
    linear_weight = (np.arange(24, dtype=np.float32).reshape(2, 12) - 8.0) / 50.0
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    eps = 1e-5
    lr = 0.04

    result = native_gpu_depthwise_layernorm2d_pointwise_linear_training_step(
        x,
        labels,
        depthwise_weight,
        norm_weight,
        norm_bias,
        pointwise_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        norm_eps=eps,
        bound_lib=_RawFakeCudaLib(),
    )

    depthwise = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(depthwise_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    depthwise[ni, oc, oh, ow] = np.sum(
                        x[ni, ic, oh:oh + 2, ow:ow + 2] * depthwise_weight[oc, 0]
                    )
    mean = depthwise.mean(axis=1, keepdims=True).astype(np.float32)
    var = depthwise.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((depthwise - mean) * inv_std).astype(np.float32)
    norm = x_hat * norm_weight.reshape(1, 2, 1, 1) + norm_bias.reshape(1, 2, 1, 1)
    pointwise = np.zeros((2, 3, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(pointwise_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    pointwise[ni, oc, oh, ow] = np.sum(norm[ni, :, oh, ow] * pointwise_weight[oc, :, 0, 0])
    flat = pointwise.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pointwise = (grad_logits @ linear_weight).reshape(pointwise.shape)
    grad_pointwise_weight = np.zeros_like(pointwise_weight)
    grad_norm = np.zeros_like(norm)
    for ni in range(x.shape[0]):
        for oc in range(pointwise_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_pointwise[ni, oc, oh, ow]
                    for ci in range(norm.shape[1]):
                        grad_pointwise_weight[oc, ci, 0, 0] += norm[ni, ci, oh, ow] * grad_val
                        grad_norm[ni, ci, oh, ow] += pointwise_weight[oc, ci, 0, 0] * grad_val
    grad_norm_weight = (grad_norm * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = grad_norm * norm_weight.reshape(1, 2, 1, 1)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_depthwise = (inv_std / 2.0) * (2.0 * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    grad_depthwise_weight = np.zeros_like(depthwise_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(depthwise_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_depthwise[ni, oc, oh, ow]
                    for r in range(2):
                        for s in range(2):
                            grad_depthwise_weight[oc, 0, r, s] += x[ni, ic, oh + r, ow + s] * grad_val
                            grad_input[ni, ic, oh + r, ow + s] += depthwise_weight[oc, 0, r, s] * grad_val

    np.testing.assert_allclose(result.depthwise_output, depthwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.norm_output, norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pointwise_output, pointwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise_output, grad_pointwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_depthwise_output, grad_depthwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_depthwise_weight, grad_depthwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise_weight, grad_pointwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_depthwise_weight, depthwise_weight - lr * grad_depthwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_pointwise_weight, pointwise_weight - lr * grad_pointwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise_conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise_conv2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_backward'] == 1


def test_native_gpu_depthwise_layernorm2d_pointwise_linear_training_step_can_skip_intermediate_host_copies():
    x = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) - 12.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    depthwise_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    norm_weight = np.asarray([1.0, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.1], dtype=np.float32)
    pointwise_weight = np.asarray(
        [
            [[[0.1]], [[-0.2]]],
            [[[0.25]], [[0.05]]],
            [[[-0.15]], [[0.2]]],
        ],
        dtype=np.float32,
    )
    linear_weight = (np.arange(24, dtype=np.float32).reshape(2, 12) - 8.0) / 50.0
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_depthwise_layernorm2d_pointwise_linear_training_step(
        x,
        labels,
        depthwise_weight,
        norm_weight,
        norm_bias,
        pointwise_weight,
        linear_weight,
        linear_bias,
        lr=0.04,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.depthwise_output.shape == (0,)
    assert result.norm_output.shape == (0,)
    assert result.pointwise_output.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_pointwise_output.shape == (0,)
    assert result.grad_norm_output.shape == (0,)
    assert result.grad_depthwise_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_depthwise_weight.shape == (0,)
    assert result.grad_norm_weight.shape == (0,)
    assert result.grad_norm_bias.shape == (0,)
    assert result.grad_pointwise_weight.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.updated_depthwise_weight.shape == depthwise_weight.shape
    assert result.updated_norm_weight.shape == norm_weight.shape
    assert result.updated_norm_bias.shape == norm_bias.shape
    assert result.updated_pointwise_weight.shape == pointwise_weight.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_depthwise_weight_velocity is None
    assert result.updated_norm_weight_velocity is None
    assert result.updated_pointwise_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 8


def test_native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step_matches_reference_math():
    x = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) - 12.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    depthwise_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    norm_weight = np.asarray([1.0, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.1], dtype=np.float32)
    pointwise1_weight = np.asarray(
        [
            [[[0.1]], [[-0.2]]],
            [[[0.25]], [[0.05]]],
            [[[-0.15]], [[0.2]]],
        ],
        dtype=np.float32,
    )
    pointwise2_weight = np.asarray(
        [
            [[[0.12]], [[-0.18]], [[0.07]]],
            [[[-0.05]], [[0.22]], [[-0.11]]],
        ],
        dtype=np.float32,
    )
    linear_weight = (np.arange(16, dtype=np.float32).reshape(2, 8) - 5.0) / 40.0
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    eps = 1e-5
    lr = 0.035

    result = native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
        x,
        labels,
        depthwise_weight,
        norm_weight,
        norm_bias,
        pointwise1_weight,
        pointwise2_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        norm_eps=eps,
        bound_lib=_RawFakeCudaLib(),
    )

    def gelu(values):
        inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
        return 0.5 * values * (1.0 + np.tanh(inner))

    def gelu_grad(values):
        inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
        tanh_inner = np.tanh(inner)
        left = 0.5 * (1.0 + tanh_inner)
        right = 0.5 * values * (1.0 - tanh_inner ** 2) * np.sqrt(2.0 / np.pi) * (
            1.0 + 3.0 * 0.044715 * values ** 2
        )
        return left + right

    depthwise = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(depthwise_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    depthwise[ni, oc, oh, ow] = np.sum(
                        x[ni, ic, oh:oh + 2, ow:ow + 2] * depthwise_weight[oc, 0]
                    )
    mean = depthwise.mean(axis=1, keepdims=True).astype(np.float32)
    var = depthwise.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((depthwise - mean) * inv_std).astype(np.float32)
    norm = x_hat * norm_weight.reshape(1, 2, 1, 1) + norm_bias.reshape(1, 2, 1, 1)
    pointwise1 = np.zeros((2, 3, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(pointwise1_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    pointwise1[ni, oc, oh, ow] = np.sum(norm[ni, :, oh, ow] * pointwise1_weight[oc, :, 0, 0])
    activation = gelu(pointwise1).astype(np.float32)
    pointwise2 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(pointwise2_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    pointwise2[ni, oc, oh, ow] = np.sum(activation[ni, :, oh, ow] * pointwise2_weight[oc, :, 0, 0])
    flat = pointwise2.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pointwise2 = (grad_logits @ linear_weight).reshape(pointwise2.shape)
    grad_pointwise2_weight = np.zeros_like(pointwise2_weight)
    grad_activation = np.zeros_like(activation)
    for ni in range(x.shape[0]):
        for oc in range(pointwise2_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_pointwise2[ni, oc, oh, ow]
                    for ci in range(activation.shape[1]):
                        grad_pointwise2_weight[oc, ci, 0, 0] += activation[ni, ci, oh, ow] * grad_val
                        grad_activation[ni, ci, oh, ow] += pointwise2_weight[oc, ci, 0, 0] * grad_val
    grad_pointwise1 = grad_activation * gelu_grad(pointwise1).astype(np.float32)
    grad_pointwise1_weight = np.zeros_like(pointwise1_weight)
    grad_norm = np.zeros_like(norm)
    for ni in range(x.shape[0]):
        for oc in range(pointwise1_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_pointwise1[ni, oc, oh, ow]
                    for ci in range(norm.shape[1]):
                        grad_pointwise1_weight[oc, ci, 0, 0] += norm[ni, ci, oh, ow] * grad_val
                        grad_norm[ni, ci, oh, ow] += pointwise1_weight[oc, ci, 0, 0] * grad_val
    grad_norm_weight = (grad_norm * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = grad_norm * norm_weight.reshape(1, 2, 1, 1)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_depthwise = (inv_std / 2.0) * (2.0 * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    grad_depthwise_weight = np.zeros_like(depthwise_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(depthwise_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_depthwise[ni, oc, oh, ow]
                    for r in range(2):
                        for s in range(2):
                            grad_depthwise_weight[oc, 0, r, s] += x[ni, ic, oh + r, ow + s] * grad_val
                            grad_input[ni, ic, oh + r, ow + s] += depthwise_weight[oc, 0, r, s] * grad_val

    np.testing.assert_allclose(result.depthwise_output, depthwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.norm_output, norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pointwise1_output, pointwise1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.activation_output, activation, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pointwise2_output, pointwise2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise2_output, grad_pointwise2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_activation_output, grad_activation, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise1_output, grad_pointwise1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_depthwise_output, grad_depthwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_depthwise_weight, grad_depthwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise1_weight, grad_pointwise1_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise2_weight, grad_pointwise2_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_depthwise_weight, depthwise_weight - lr * grad_depthwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_pointwise1_weight, pointwise1_weight - lr * grad_pointwise1_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_pointwise2_weight, pointwise2_weight - lr * grad_pointwise2_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise1_conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:gelu_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise2_conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise2_conv2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:gelu_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise1_conv2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_backward'] == 1


def test_native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step_can_skip_intermediate_host_copies():
    x = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) - 12.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    depthwise_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    norm_weight = np.asarray([1.0, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.1], dtype=np.float32)
    pointwise1_weight = np.asarray(
        [
            [[[0.1]], [[-0.2]]],
            [[[0.25]], [[0.05]]],
            [[[-0.15]], [[0.2]]],
        ],
        dtype=np.float32,
    )
    pointwise2_weight = np.asarray(
        [
            [[[0.12]], [[-0.18]], [[0.07]]],
            [[[-0.05]], [[0.22]], [[-0.11]]],
        ],
        dtype=np.float32,
    )
    linear_weight = (np.arange(16, dtype=np.float32).reshape(2, 8) - 5.0) / 40.0
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
        x,
        labels,
        depthwise_weight,
        norm_weight,
        norm_bias,
        pointwise1_weight,
        pointwise2_weight,
        linear_weight,
        linear_bias,
        lr=0.035,
        norm_eps=1e-5,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.probabilities.shape == (0,)
    assert result.depthwise_output.shape == (0,)
    assert result.norm_output.shape == (0,)
    assert result.pointwise1_output.shape == (0,)
    assert result.activation_output.shape == (0,)
    assert result.pointwise2_output.shape == (0,)
    assert result.grad_logits.shape == (0,)
    assert result.grad_pointwise2_output.shape == (0,)
    assert result.grad_activation_output.shape == (0,)
    assert result.grad_pointwise1_output.shape == (0,)
    assert result.grad_norm_output.shape == (0,)
    assert result.grad_depthwise_output.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.grad_depthwise_weight.shape == (0,)
    assert result.grad_norm_weight.shape == (0,)
    assert result.grad_norm_bias.shape == (0,)
    assert result.grad_pointwise1_weight.shape == (0,)
    assert result.grad_pointwise2_weight.shape == (0,)
    assert result.grad_linear_weight.shape == (0,)
    assert result.grad_linear_bias.shape == (0,)
    assert result.updated_depthwise_weight.shape == depthwise_weight.shape
    assert result.updated_norm_weight.shape == norm_weight.shape
    assert result.updated_norm_bias.shape == norm_bias.shape
    assert result.updated_pointwise1_weight.shape == pointwise1_weight.shape
    assert result.updated_pointwise2_weight.shape == pointwise2_weight.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.updated_linear_bias.shape == linear_bias.shape
    assert result.updated_depthwise_weight_velocity is None
    assert result.updated_norm_weight_velocity is None
    assert result.updated_norm_bias_velocity is None
    assert result.updated_pointwise1_weight_velocity is None
    assert result.updated_pointwise2_weight_velocity is None
    assert result.updated_linear_weight_velocity is None
    assert result.updated_linear_bias_velocity is None
    assert result.runtime_summary['device_to_host_transfer_events'] == 9


def test_native_gpu_depthwise_layernorm2d_pointwise_silu_pointwise_linear_training_step_matches_reference_math():
    x = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) - 12.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    depthwise_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    norm_weight = np.asarray([1.0, 1.25], dtype=np.float32)
    norm_bias = np.asarray([0.0, -0.1], dtype=np.float32)
    pointwise1_weight = np.asarray(
        [
            [[[0.1]], [[-0.2]]],
            [[[0.25]], [[0.05]]],
            [[[-0.15]], [[0.2]]],
        ],
        dtype=np.float32,
    )
    pointwise2_weight = np.asarray(
        [
            [[[0.12]], [[-0.18]], [[0.07]]],
            [[[-0.05]], [[0.22]], [[-0.11]]],
        ],
        dtype=np.float32,
    )
    linear_weight = (np.arange(16, dtype=np.float32).reshape(2, 8) - 5.0) / 40.0
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    eps = 1e-5
    lr = 0.035

    result = native_gpu_depthwise_layernorm2d_pointwise_gelu_pointwise_linear_training_step(
        x,
        labels,
        depthwise_weight,
        norm_weight,
        norm_bias,
        pointwise1_weight,
        pointwise2_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        norm_eps=eps,
        activation_kind='SiLU',
        bound_lib=_RawFakeCudaLib(),
    )

    def silu(values):
        sigmoid = 1.0 / (1.0 + np.exp(-values))
        return values * sigmoid

    def silu_grad(values):
        sigmoid = 1.0 / (1.0 + np.exp(-values))
        return sigmoid + values * sigmoid * (1.0 - sigmoid)

    depthwise = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(depthwise_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    depthwise[ni, oc, oh, ow] = np.sum(
                        x[ni, ic, oh:oh + 2, ow:ow + 2] * depthwise_weight[oc, 0]
                    )
    mean = depthwise.mean(axis=1, keepdims=True).astype(np.float32)
    var = depthwise.var(axis=1, keepdims=True).astype(np.float32)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = ((depthwise - mean) * inv_std).astype(np.float32)
    norm = x_hat * norm_weight.reshape(1, 2, 1, 1) + norm_bias.reshape(1, 2, 1, 1)
    pointwise1 = np.zeros((2, 3, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(pointwise1_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    pointwise1[ni, oc, oh, ow] = np.sum(norm[ni, :, oh, ow] * pointwise1_weight[oc, :, 0, 0])
    activation = silu(pointwise1).astype(np.float32)
    pointwise2 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(pointwise2_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    pointwise2[ni, oc, oh, ow] = np.sum(activation[ni, :, oh, ow] * pointwise2_weight[oc, :, 0, 0])
    flat = pointwise2.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pointwise2 = (grad_logits @ linear_weight).reshape(pointwise2.shape)
    grad_pointwise2_weight = np.zeros_like(pointwise2_weight)
    grad_activation = np.zeros_like(activation)
    for ni in range(x.shape[0]):
        for oc in range(pointwise2_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_pointwise2[ni, oc, oh, ow]
                    for ci in range(activation.shape[1]):
                        grad_pointwise2_weight[oc, ci, 0, 0] += activation[ni, ci, oh, ow] * grad_val
                        grad_activation[ni, ci, oh, ow] += pointwise2_weight[oc, ci, 0, 0] * grad_val
    grad_pointwise1 = grad_activation * silu_grad(pointwise1).astype(np.float32)
    grad_pointwise1_weight = np.zeros_like(pointwise1_weight)
    grad_norm = np.zeros_like(norm)
    for ni in range(x.shape[0]):
        for oc in range(pointwise1_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_pointwise1[ni, oc, oh, ow]
                    for ci in range(norm.shape[1]):
                        grad_pointwise1_weight[oc, ci, 0, 0] += norm[ni, ci, oh, ow] * grad_val
                        grad_norm[ni, ci, oh, ow] += pointwise1_weight[oc, ci, 0, 0] * grad_val
    grad_norm_weight = (grad_norm * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
    grad_norm_bias = grad_norm.sum(axis=(0, 2, 3)).astype(np.float32)
    dxhat = grad_norm * norm_weight.reshape(1, 2, 1, 1)
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = (dxhat * x_hat).sum(axis=1, keepdims=True)
    grad_depthwise = (inv_std / 2.0) * (2.0 * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
    grad_depthwise_weight = np.zeros_like(depthwise_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(depthwise_weight.shape[0]):
            ic = oc
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_depthwise[ni, oc, oh, ow]
                    for r in range(2):
                        for s in range(2):
                            grad_depthwise_weight[oc, 0, r, s] += x[ni, ic, oh + r, ow + s] * grad_val
                            grad_input[ni, ic, oh + r, ow + s] += depthwise_weight[oc, 0, r, s] * grad_val

    np.testing.assert_allclose(result.depthwise_output, depthwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.norm_output, norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pointwise1_output, pointwise1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.activation_output, activation, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pointwise2_output, pointwise2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise2_output, grad_pointwise2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_activation_output, grad_activation, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise1_output, grad_pointwise1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_output, grad_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_depthwise_output, grad_depthwise, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_depthwise_weight, grad_depthwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_weight, grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_norm_bias, grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise1_weight, grad_pointwise1_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pointwise2_weight, grad_pointwise2_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_depthwise_weight, depthwise_weight - lr * grad_depthwise_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_weight, norm_weight - lr * grad_norm_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_norm_bias, norm_bias - lr * grad_norm_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_pointwise1_weight, pointwise1_weight - lr * grad_pointwise1_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_pointwise2_weight, pointwise2_weight - lr * grad_pointwise2_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise1_conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:silu_forward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise2_conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise2_conv2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:silu_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:pointwise1_conv2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:layernorm2d_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:depthwise_conv2d_backward'] == 1


def test_native_gpu_conv_relu_linear_training_step_matches_reference_math():
    x = np.asarray(
        [
            [[[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]]],
            [[[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    conv_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.04

    result = native_gpu_conv_linear_training_step(
        x,
        labels,
        conv_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        apply_relu_activation=True,
        bound_lib=_RawFakeCudaLib(),
    )

    conv_pre = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    conv_pre[ni, oc, oh, ow] = np.sum(x[ni, :, oh:oh + 2, ow:ow + 2] * conv_weight[oc])
    conv = np.maximum(conv_pre, 0.0)
    flat = conv.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_conv_output = (grad_logits @ linear_weight).reshape(conv.shape)
    grad_conv_output = np.where(conv > 0.0, grad_conv_output, 0.0)
    grad_conv_weight = np.zeros_like(conv_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_conv_output[ni, oc, oh, ow]
                    for ci in range(x.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                grad_input[ni, ci, oh + r, ow + s] += conv_weight[oc, ci, r, s] * grad_val

    np.testing.assert_allclose(result.conv_output, conv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_output, grad_conv_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_weight, grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_conv_weight, conv_weight - lr * grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_relu'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_relu_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv_backward'] == 1


@pytest.mark.parametrize(
    ('activation', 'forward_key', 'backward_key', 'alpha'),
    (
        ('LeakyReLU', 'leaky_relu_forward', 'leaky_relu_backward', 0.2),
        ('GELU', 'gelu_forward', 'gelu_backward', 0.01),
        ('SiLU', 'silu_forward', 'silu_backward', 0.01),
        ('Sigmoid', 'sigmoid_forward', 'sigmoid_backward', 0.01),
        ('Tanh', 'tanh_forward', 'tanh_backward', 0.01),
    ),
)
def test_native_gpu_conv_modern_activation_linear_training_step_matches_reference_math(
    activation,
    forward_key,
    backward_key,
    alpha,
):
    x = np.asarray(
        [
            [[[1.0, 2.0, -1.0], [0.0, 1.5, 2.5], [3.0, -0.5, 1.0]]],
            [[[-1.0, 0.5, 2.0], [1.0, -1.5, 0.0], [2.5, 1.5, -0.5]]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0], dtype=np.int32)
    conv_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.04

    result = native_gpu_conv_linear_training_step(
        x,
        labels,
        conv_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        activation_kind=activation,
        activation_alpha=alpha,
        bound_lib=_RawFakeCudaLib(),
    )

    conv_pre = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    conv_pre[ni, oc, oh, ow] = np.sum(x[ni, :, oh:oh + 2, ow:ow + 2] * conv_weight[oc])
    if activation == 'LeakyReLU':
        conv = np.where(conv_pre > 0.0, conv_pre, alpha * conv_pre)
        activation_grad = np.where(conv_pre > 0.0, 1.0, alpha)
    elif activation == 'GELU':
        conv = 0.5 * conv_pre * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (conv_pre + 0.044715 * conv_pre ** 3))
        )
        inner = np.sqrt(2.0 / np.pi) * (conv_pre + 0.044715 * conv_pre ** 3)
        tanh_inner = np.tanh(inner)
        sech2_inner = 1.0 - tanh_inner * tanh_inner
        inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * conv_pre ** 2)
        activation_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * conv_pre * sech2_inner * inner_grad
    elif activation == 'SiLU':
        sig = 1.0 / (1.0 + np.exp(-conv_pre))
        conv = conv_pre * sig
        activation_grad = sig + conv_pre * sig * (1.0 - sig)
    elif activation == 'Sigmoid':
        conv = 1.0 / (1.0 + np.exp(-conv_pre))
        activation_grad = conv * (1.0 - conv)
    else:
        conv = np.tanh(conv_pre)
        activation_grad = 1.0 - conv * conv
    flat = conv.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_conv_output = (grad_logits @ linear_weight).reshape(conv.shape) * activation_grad
    grad_conv_weight = np.zeros_like(conv_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(2):
                for ow in range(2):
                    grad_val = grad_conv_output[ni, oc, oh, ow]
                    for ci in range(x.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                grad_input[ni, ci, oh + r, ow + s] += conv_weight[oc, ci, r, s] * grad_val

    np.testing.assert_allclose(result.conv_output, conv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_output, grad_conv_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_weight, grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{forward_key}'] == 1
    assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{backward_key}'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv_backward'] == 1


def test_native_gpu_conv_relu_pool_linear_training_step_matches_reference_math():
    x = (np.arange(50, dtype=np.float32).reshape(2, 1, 5, 5) - 20.0) / 10.0
    labels = np.asarray([1, 0], dtype=np.int32)
    conv_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.03

    result = native_gpu_conv_linear_training_step(
        x,
        labels,
        conv_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        apply_relu_activation=True,
        apply_maxpool=True,
        bound_lib=_RawFakeCudaLib(),
    )

    conv_pre = np.zeros((2, 2, 4, 4), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(4):
                for ow in range(4):
                    conv_pre[ni, oc, oh, ow] = np.sum(x[ni, :, oh:oh + 2, ow:ow + 2] * conv_weight[oc])
    conv = np.maximum(conv_pre, 0.0)
    pooled = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv.shape[1]):
            for oh in range(2):
                for ow in range(2):
                    pooled[ni, oc, oh, ow] = np.max(conv[ni, oc, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2])
    flat = pooled.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pooled = (grad_logits @ linear_weight).reshape(pooled.shape)
    grad_conv_output = np.zeros_like(conv)
    for ni in range(x.shape[0]):
        for oc in range(conv.shape[1]):
            for oh in range(2):
                for ow in range(2):
                    window = conv[ni, oc, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                    flat_idx = int(np.argmax(window))
                    ih = oh * 2 + flat_idx // 2
                    iw = ow * 2 + flat_idx % 2
                    grad_conv_output[ni, oc, ih, iw] += grad_pooled[ni, oc, oh, ow]
    grad_conv_output = np.where(conv > 0.0, grad_conv_output, 0.0)
    grad_conv_weight = np.zeros_like(conv_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv_weight.shape[0]):
            for oh in range(4):
                for ow in range(4):
                    grad_val = grad_conv_output[ni, oc, oh, ow]
                    for ci in range(x.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                grad_input[ni, ci, oh + r, ow + s] += conv_weight[oc, ci, r, s] * grad_val

    np.testing.assert_allclose(result.conv_output, conv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pooled_output, pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pooled, grad_pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_output, grad_conv_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv_weight, grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_conv_weight, conv_weight - lr * grad_conv_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv2d_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_relu'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_maxpool'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:maxpool_backward_nchw'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:apply_relu_backward'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv_backward'] == 1


def test_native_gpu_training_subset_parity_matrix_covers_current_surface():
    from minicnn.cuda_native.capabilities import GPU_NATIVE_TRAINING_SUBSETS

    matrix = {tuple(item['ops']): item for item in GPU_NATIVE_TRAINING_SUBSETS}
    expected_ops = {
        ('Linear',),
        ('Flatten', 'Linear'),
        ('Linear', 'ReLU', 'Linear'),
        ('Flatten', 'Linear', 'ReLU', 'Linear'),
        ('Linear', 'LeakyReLU', 'Linear'),
        ('Flatten', 'Linear', 'LeakyReLU', 'Linear'),
        ('Linear', 'GELU', 'Linear'),
        ('Flatten', 'Linear', 'GELU', 'Linear'),
        ('Linear', 'SiLU', 'Linear'),
        ('Flatten', 'Linear', 'SiLU', 'Linear'),
        ('Linear', 'Sigmoid', 'Linear'),
        ('Flatten', 'Linear', 'Sigmoid', 'Linear'),
        ('Linear', 'Tanh', 'Linear'),
        ('Flatten', 'Linear', 'Tanh', 'Linear'),
        ('MaxPool2d', 'Flatten', 'Linear'),
        ('AvgPool2d', 'Flatten', 'Linear'),
        ('BatchNorm2d', 'Flatten', 'Linear'),
        ('Flatten', 'LayerNorm', 'Linear'),
        ('Flatten', 'LayerNorm', 'ReLU', 'Linear'),
        ('Flatten', 'LayerNorm', 'LeakyReLU', 'Linear'),
        ('Flatten', 'LayerNorm', 'GELU', 'Linear'),
        ('Flatten', 'LayerNorm', 'SiLU', 'Linear'),
        ('Flatten', 'LayerNorm', 'Sigmoid', 'Linear'),
        ('Flatten', 'LayerNorm', 'Tanh', 'Linear'),
        ('LayerNorm2d', 'Flatten', 'Linear'),
        ('GroupNorm', 'Flatten', 'Linear'),
        ('GlobalAvgPool2d', 'Flatten', 'Linear'),
        ('AdaptiveAvgPool2d', 'Flatten', 'Linear'),
        ('Conv2d', 'Flatten', 'Linear'),
        ('Conv2d', 'ReLU', 'Flatten', 'Linear'),
        ('PointwiseConv2d', 'Flatten', 'Linear'),
        ('PointwiseConv2d', 'ReLU', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'ReLU', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', 'GELU', 'PointwiseConv2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'MaxPool2d', 'Flatten', 'Linear'),
        ('DepthwiseConv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'),
        ('Conv2d', 'MaxPool2d', 'Flatten', 'Linear'),
        ('Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'),
        ('Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear'),
    }
    for conv_op in ('Conv2d', 'PointwiseConv2d', 'DepthwiseConv2d'):
        for activation in ('LeakyReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh'):
            expected_ops.add((conv_op, activation, 'Flatten', 'Linear'))
            if conv_op != 'PointwiseConv2d':
                expected_ops.add((conv_op, activation, 'MaxPool2d', 'Flatten', 'Linear'))
    for activation in ('ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh'):
        expected_ops.add(('Flatten', 'LayerNorm', activation, 'Linear'))
    for activation in ('ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh'):
        expected_ops.add(('Conv2d', activation, 'Conv2d', activation, 'MaxPool2d', 'Flatten', 'Linear'))
        expected_ops.add(('DepthwiseConv2d', 'LayerNorm2d', 'PointwiseConv2d', activation, 'PointwiseConv2d', 'Flatten', 'Linear'))

    assert set(matrix) == expected_ops
    assert matrix[('Linear',)]['losses'] == ['CrossEntropyLoss', 'MSELoss', 'BCEWithLogitsLoss']


def test_native_gpu_training_subset_capabilities_declare_loss_and_optimizer_contracts():
    from minicnn.cuda_native.capabilities import GPU_NATIVE_TRAINING_SUBSETS

    matrix = {tuple(item['ops']): item for item in GPU_NATIVE_TRAINING_SUBSETS}

    for subset in GPU_NATIVE_TRAINING_SUBSETS:
        assert subset['losses']
        assert subset['optimizers']

    full_optimizer_subsets = {'linear', 'flatten_linear'}
    for subset in GPU_NATIVE_TRAINING_SUBSETS:
        if subset['name'] in full_optimizer_subsets:
            assert subset['optimizers'] == ['SGD', 'Adam', 'AdamW', 'RMSprop']
        else:
            assert subset['losses'] == ['CrossEntropyLoss']
            assert subset['optimizers'] == ['SGD']
    assert matrix[('Flatten', 'Linear')]['losses'] == ['CrossEntropyLoss', 'MSELoss', 'BCEWithLogitsLoss']
    assert matrix[('Linear',)]['optimizers'] == ['SGD', 'Adam', 'AdamW', 'RMSprop']
    assert matrix[('Flatten', 'Linear')]['optimizers'] == ['SGD', 'Adam', 'AdamW', 'RMSprop']
    for item in matrix.values():
        assert item['parity'] == 'hermetic_reference_math'
        assert item['helper'].startswith('native_gpu_')


def test_native_gpu_two_conv_relu_pool_linear_training_step_matches_reference_math():
    x = (np.arange(72, dtype=np.float32).reshape(2, 1, 6, 6) - 30.0) / 20.0
    labels = np.asarray([1, 0], dtype=np.int32)
    conv1_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    conv2_weight = np.asarray(
        [
            [[[0.1, -0.2], [0.05, 0.15]], [[-0.05, 0.12], [0.2, -0.08]]],
            [[[-0.15, 0.05], [0.18, -0.1]], [[0.22, -0.04], [-0.06, 0.11]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.02

    result = native_gpu_two_conv_relu_pool_linear_training_step(
        x,
        labels,
        conv1_weight,
        conv2_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        bound_lib=_RawFakeCudaLib(),
    )

    conv1_pre = np.zeros((2, 2, 5, 5), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv1_weight.shape[0]):
            for oh in range(5):
                for ow in range(5):
                    conv1_pre[ni, oc, oh, ow] = np.sum(x[ni, :, oh:oh + 2, ow:ow + 2] * conv1_weight[oc])
    conv1 = np.maximum(conv1_pre, 0.0)
    conv2_pre = np.zeros((2, 2, 4, 4), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv2_weight.shape[0]):
            for oh in range(4):
                for ow in range(4):
                    conv2_pre[ni, oc, oh, ow] = np.sum(conv1[ni, :, oh:oh + 2, ow:ow + 2] * conv2_weight[oc])
    conv2 = np.maximum(conv2_pre, 0.0)
    pooled = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv2.shape[1]):
            for oh in range(2):
                for ow in range(2):
                    pooled[ni, oc, oh, ow] = np.max(conv2[ni, oc, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2])
    flat = pooled.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pooled = (grad_logits @ linear_weight).reshape(pooled.shape)
    grad_conv2 = np.zeros_like(conv2)
    for ni in range(x.shape[0]):
        for oc in range(conv2.shape[1]):
            for oh in range(2):
                for ow in range(2):
                    window = conv2[ni, oc, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                    flat_idx = int(np.argmax(window))
                    ih = oh * 2 + flat_idx // 2
                    iw = ow * 2 + flat_idx % 2
                    grad_conv2[ni, oc, ih, iw] += grad_pooled[ni, oc, oh, ow]
    grad_conv2 = np.where(conv2 > 0.0, grad_conv2, 0.0)
    grad_conv2_weight = np.zeros_like(conv2_weight)
    grad_conv1 = np.zeros_like(conv1)
    for ni in range(x.shape[0]):
        for oc in range(conv2_weight.shape[0]):
            for oh in range(4):
                for ow in range(4):
                    grad_val = grad_conv2[ni, oc, oh, ow]
                    for ci in range(conv2_weight.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv2_weight[oc, ci, r, s] += conv1[ni, ci, oh + r, ow + s] * grad_val
                                grad_conv1[ni, ci, oh + r, ow + s] += conv2_weight[oc, ci, r, s] * grad_val
    grad_conv1 = np.where(conv1 > 0.0, grad_conv1, 0.0)
    grad_conv1_weight = np.zeros_like(conv1_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv1_weight.shape[0]):
            for oh in range(5):
                for ow in range(5):
                    grad_val = grad_conv1[ni, oc, oh, ow]
                    for ci in range(x.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv1_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                grad_input[ni, ci, oh + r, ow + s] += conv1_weight[oc, ci, r, s] * grad_val

    np.testing.assert_allclose(result.conv1_output, conv1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.conv2_output, conv2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pooled_output, pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pooled, grad_pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv2_output, grad_conv2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv1_output, grad_conv1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv1_weight, grad_conv1_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv2_weight, grad_conv2_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_conv1_weight, conv1_weight - lr * grad_conv1_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_conv2_weight, conv2_weight - lr * grad_conv2_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_weight, linear_weight - lr * grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.updated_linear_bias, linear_bias - lr * grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv2d_1_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv2d_2_im2col_gemm'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv_backward_1'] == 1
    assert result.runtime_summary['execution_kinds']['gpu_native_train:conv_backward_2'] == 1


@pytest.mark.parametrize(
    ('activation', 'forward_key', 'backward_key', 'alpha'),
    (
        ('LeakyReLU', 'leaky_relu_forward', 'leaky_relu_backward', 0.2),
        ('GELU', 'gelu_forward', 'gelu_backward', 0.01),
        ('SiLU', 'silu_forward', 'silu_backward', 0.01),
        ('Sigmoid', 'sigmoid_forward', 'sigmoid_backward', 0.01),
        ('Tanh', 'tanh_forward', 'tanh_backward', 0.01),
    ),
)
def test_native_gpu_two_conv_modern_activation_pool_linear_training_step_matches_reference_math(
    activation,
    forward_key,
    backward_key,
    alpha,
):
    x = (np.arange(72, dtype=np.float32).reshape(2, 1, 6, 6) - 30.0) / 20.0
    labels = np.asarray([1, 0], dtype=np.int32)
    conv1_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    conv2_weight = np.asarray(
        [
            [[[0.1, -0.2], [0.05, 0.15]], [[-0.05, 0.12], [0.2, -0.08]]],
            [[[-0.15, 0.05], [0.18, -0.1]], [[0.22, -0.04], [-0.06, 0.11]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)
    lr = 0.02

    result = native_gpu_two_conv_relu_pool_linear_training_step(
        x,
        labels,
        conv1_weight,
        conv2_weight,
        linear_weight,
        linear_bias,
        lr=lr,
        activation_kind=activation,
        activation_alpha=alpha,
        bound_lib=_RawFakeCudaLib(),
    )

    def _activate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if activation == 'LeakyReLU':
            out = np.where(values > 0.0, values, alpha * values)
            grad = np.where(values > 0.0, 1.0, alpha)
            return out.astype(np.float32), grad.astype(np.float32)
        if activation == 'GELU':
            out = 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)))
            inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
            tanh_inner = np.tanh(inner)
            sech2_inner = 1.0 - tanh_inner * tanh_inner
            inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * values ** 2)
            grad = 0.5 * (1.0 + tanh_inner) + 0.5 * values * sech2_inner * inner_grad
            return out.astype(np.float32), grad.astype(np.float32)
        if activation == 'SiLU':
            sig = 1.0 / (1.0 + np.exp(-values))
            out = values * sig
            grad = sig + values * sig * (1.0 - sig)
            return out.astype(np.float32), grad.astype(np.float32)
        if activation == 'Sigmoid':
            out = 1.0 / (1.0 + np.exp(-values))
            grad = out * (1.0 - out)
            return out.astype(np.float32), grad.astype(np.float32)
        out = np.tanh(values)
        grad = 1.0 - out * out
        return out.astype(np.float32), grad.astype(np.float32)

    conv1_pre = np.zeros((2, 2, 5, 5), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv1_weight.shape[0]):
            for oh in range(5):
                for ow in range(5):
                    conv1_pre[ni, oc, oh, ow] = np.sum(x[ni, :, oh:oh + 2, ow:ow + 2] * conv1_weight[oc])
    conv1, conv1_grad = _activate(conv1_pre)
    conv2_pre = np.zeros((2, 2, 4, 4), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv2_weight.shape[0]):
            for oh in range(4):
                for ow in range(4):
                    conv2_pre[ni, oc, oh, ow] = np.sum(conv1[ni, :, oh:oh + 2, ow:ow + 2] * conv2_weight[oc])
    conv2, conv2_grad = _activate(conv2_pre)
    pooled = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ni in range(x.shape[0]):
        for oc in range(conv2.shape[1]):
            for oh in range(2):
                for ow in range(2):
                    pooled[ni, oc, oh, ow] = np.max(conv2[ni, oc, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2])
    flat = pooled.reshape(2, -1)
    logits = flat @ linear_weight.T + linear_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    grad_logits = probs.copy()
    grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
    grad_logits /= float(labels.shape[0])
    grad_linear_weight = grad_logits.T @ flat
    grad_linear_bias = grad_logits.sum(axis=0)
    grad_pooled = (grad_logits @ linear_weight).reshape(pooled.shape)
    grad_conv2 = np.zeros_like(conv2)
    for ni in range(x.shape[0]):
        for oc in range(conv2.shape[1]):
            for oh in range(2):
                for ow in range(2):
                    window = conv2[ni, oc, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                    flat_idx = int(np.argmax(window))
                    ih = oh * 2 + flat_idx // 2
                    iw = ow * 2 + flat_idx % 2
                    grad_conv2[ni, oc, ih, iw] += grad_pooled[ni, oc, oh, ow]
    grad_conv2 *= conv2_grad
    grad_conv2_weight = np.zeros_like(conv2_weight)
    grad_conv1 = np.zeros_like(conv1)
    for ni in range(x.shape[0]):
        for oc in range(conv2_weight.shape[0]):
            for oh in range(4):
                for ow in range(4):
                    grad_val = grad_conv2[ni, oc, oh, ow]
                    for ci in range(conv2_weight.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv2_weight[oc, ci, r, s] += conv1[ni, ci, oh + r, ow + s] * grad_val
                                grad_conv1[ni, ci, oh + r, ow + s] += conv2_weight[oc, ci, r, s] * grad_val
    grad_conv1 *= conv1_grad
    grad_conv1_weight = np.zeros_like(conv1_weight)
    grad_input = np.zeros_like(x)
    for ni in range(x.shape[0]):
        for oc in range(conv1_weight.shape[0]):
            for oh in range(5):
                for ow in range(5):
                    grad_val = grad_conv1[ni, oc, oh, ow]
                    for ci in range(x.shape[1]):
                        for r in range(2):
                            for s in range(2):
                                grad_conv1_weight[oc, ci, r, s] += x[ni, ci, oh + r, ow + s] * grad_val
                                grad_input[ni, ci, oh + r, ow + s] += conv1_weight[oc, ci, r, s] * grad_val

    np.testing.assert_allclose(result.conv1_output, conv1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.conv2_output, conv2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pooled_output, pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.logits, logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.probabilities, probs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_logits, grad_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_pooled, grad_pooled, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv2_output, grad_conv2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv1_output, grad_conv1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_input, grad_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv1_weight, grad_conv1_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_conv2_weight, grad_conv2_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_weight, grad_linear_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.grad_linear_bias, grad_linear_bias, rtol=1e-6, atol=1e-6)
    assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{forward_key}_1'] == 1
    assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{forward_key}_2'] == 1
    assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{backward_key}_2'] == 1
    assert result.runtime_summary['execution_kinds'][f'gpu_native_train:{backward_key}_1'] == 1


def test_native_gpu_two_conv_relu_pool_linear_training_step_can_skip_intermediate_host_copies():
    x = (np.arange(72, dtype=np.float32).reshape(2, 1, 6, 6) - 30.0) / 20.0
    labels = np.asarray([1, 0], dtype=np.int32)
    conv1_weight = np.asarray(
        [
            [[[0.2, -0.1], [0.05, 0.3]]],
            [[[-0.2, 0.1], [0.25, -0.05]]],
        ],
        dtype=np.float32,
    )
    conv2_weight = np.asarray(
        [
            [[[0.1, -0.2], [0.05, 0.15]], [[-0.05, 0.12], [0.2, -0.08]]],
            [[[-0.15, 0.05], [0.18, -0.1]], [[0.22, -0.04], [-0.06, 0.11]]],
        ],
        dtype=np.float32,
    )
    linear_weight = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.05, 0.15],
            [-0.05, 0.25, -0.15, 0.2, 0.05, -0.1, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    linear_bias = np.asarray([0.02, -0.01], dtype=np.float32)

    result = native_gpu_two_conv_relu_pool_linear_training_step(
        x,
        labels,
        conv1_weight,
        conv2_weight,
        linear_weight,
        linear_bias,
        lr=0.02,
        bound_lib=_RawFakeCudaLib(),
        return_intermediates=False,
    )

    assert result.logits.shape == (0,)
    assert result.grad_input.shape == (0,)
    assert result.updated_conv1_weight.shape == conv1_weight.shape
    assert result.updated_conv2_weight.shape == conv2_weight.shape
    assert result.updated_linear_weight.shape == linear_weight.shape
    assert result.runtime_summary['device_to_host_transfer_events'] == 10
