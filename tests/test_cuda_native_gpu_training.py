from __future__ import annotations

import ctypes

import numpy as np

from minicnn.cuda_native.gpu_training import native_gpu_linear_training_step


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

    def dense_forward(self, d_input, d_weight, d_bias, d_output, n, in_f, out_f):
        x = self._float(d_input).reshape(int(n), int(in_f))
        w = self._float(d_weight).reshape(int(out_f), int(in_f))
        b = self._float(d_bias).reshape(int(out_f))
        self._float(d_output)[:] = (x @ w.T + b).reshape(-1)

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
