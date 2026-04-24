from __future__ import annotations

import ctypes

import numpy as np

from minicnn.cuda_native.gpu_training import (
    native_gpu_conv_linear_training_step,
    native_gpu_linear_training_step,
    native_gpu_pool_linear_training_step,
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

    def gpu_memset(self, ptr, value, nbytes):
        self.memory[int(ptr)][:int(nbytes)] = bytes([int(value) & 0xFF]) * int(nbytes)

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

    def apply_relu_backward(self, d_data, d_grad, size):
        data = self._float(d_data)
        grad = self._float(d_grad)
        grad[:int(size)] = np.where(data[:int(size)] > 0.0, grad[:int(size)], 0.0)

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
