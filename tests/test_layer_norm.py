"""Numerical gradient check: MiniCNN LayerNorm forward/backward vs PyTorch.

Tests run on CPU via pure NumPy so they do not require a built .so or GPU.
The reference implementation mirrors the CUDA kernel logic exactly:
  forward : y = (x - mean) / sqrt(var + eps) * gamma + beta
  backward: dx = gamma * inv_std * (dy - mean(dy) - x_hat * mean(dy * x_hat))
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pure-NumPy reference (mirrors cpp/src/layer_norm.cu logic)
# ---------------------------------------------------------------------------

def _ln_forward(x, gamma, beta, eps=1e-5):
    """x: (N, C, H, W); normalize over H*W per (N,C) slice."""
    mean = x.mean(axis=(-2, -1), keepdims=True)
    var  = x.var(axis=(-2, -1), keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * gamma[:, np.newaxis, np.newaxis] + beta[:, np.newaxis, np.newaxis], x_hat, mean, var


def _ln_backward(dy, x_hat, gamma, eps_inv_std):
    """
    dx = gamma * inv_std * (dy - mean(dy) - x_hat * mean(dy * x_hat))
    Matches layer_norm_backward_kernel in cpp/src/layer_norm.cu exactly.
    """
    mean_dy      = dy.mean(axis=(-2, -1), keepdims=True)
    mean_dy_xhat = (dy * x_hat).mean(axis=(-2, -1), keepdims=True)
    dx = gamma[:, np.newaxis, np.newaxis] * eps_inv_std * (dy - mean_dy - x_hat * mean_dy_xhat)
    dgamma = (dy * x_hat).sum(axis=(0, 2, 3))
    dbeta  = dy.sum(axis=(0, 2, 3))
    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_ln_reference(x_np, gamma_np, beta_np, eps=1e-5):
    """Run PyTorch LayerNorm over (H,W) and return output + input grad."""
    torch = pytest.importorskip("torch")
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(gamma_np, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(beta_np,  dtype=torch.float64, requires_grad=True)
    # PyTorch LayerNorm normalizes over normalized_shape = [H, W]
    N, C, H, W = x.shape
    ln = torch.nn.LayerNorm([H, W], eps=eps, elementwise_affine=False, dtype=torch.float64)
    y = ln(x) * g[None, :, None, None] + b[None, :, None, None]
    loss = y.sum()
    loss.backward()
    return y.detach().numpy(), x.grad.numpy(), g.grad.numpy(), b.grad.numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLayerNormForward:
    def test_output_shape(self):
        N, C, H, W = 2, 3, 4, 4
        x = np.random.randn(N, C, H, W).astype(np.float32)
        gamma = np.ones(C, dtype=np.float32)
        beta  = np.zeros(C, dtype=np.float32)
        y, _, _, _ = _ln_forward(x, gamma, beta)
        assert y.shape == (N, C, H, W)

    def test_zero_mean_unit_var_per_slice(self):
        N, C, H, W = 2, 3, 5, 5
        rng = np.random.default_rng(0)
        x = rng.standard_normal((N, C, H, W)).astype(np.float32)
        gamma = np.ones(C, dtype=np.float32)
        beta  = np.zeros(C, dtype=np.float32)
        y, _, _, _ = _ln_forward(x, gamma, beta)
        mean = y.mean(axis=(-2, -1))
        std  = y.std(axis=(-2, -1))
        np.testing.assert_allclose(mean, 0.0, atol=1e-5)
        np.testing.assert_allclose(std,  1.0, atol=1e-5)

    def test_matches_pytorch_forward(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(42)
        N, C, H, W = 2, 4, 6, 6
        x     = rng.standard_normal((N, C, H, W))
        gamma = rng.standard_normal(C)
        beta  = rng.standard_normal(C)
        y_ref, _, _, _ = _torch_ln_reference(x, gamma, beta)
        y_our, _, _, _ = _ln_forward(x, gamma, beta)
        np.testing.assert_allclose(y_our, y_ref, atol=1e-5,
                                   err_msg="forward output differs from PyTorch")


class TestLayerNormBackward:
    def _run(self, N=2, C=3, H=4, W=4, seed=7):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(seed)
        x     = rng.standard_normal((N, C, H, W))
        gamma = rng.standard_normal(C) * 0.5 + 1.0
        beta  = rng.standard_normal(C) * 0.1
        eps   = 1e-5

        y_our, x_hat, mean, var = _ln_forward(x, gamma, beta, eps)
        inv_std = 1.0 / np.sqrt(var + eps)                 # shape (N,C,1,1)
        dy = np.ones_like(y_our)

        dx_our, dg_our, db_our = _ln_backward(dy, x_hat, gamma, inv_std)

        _, dx_ref, dg_ref, db_ref = _torch_ln_reference(x, gamma, beta, eps)
        return (dx_our, dg_our, db_our), (dx_ref, dg_ref, db_ref)

    def test_dx_matches_pytorch(self):
        (dx_our, _, _), (dx_ref, _, _) = self._run()
        max_err = np.abs(dx_our - dx_ref).max()
        assert max_err < 1e-4, f"dx max_err={max_err:.2e} exceeds 1e-4"

    def test_dgamma_matches_pytorch(self):
        (_, dg_our, _), (_, dg_ref, _) = self._run()
        max_err = np.abs(dg_our - dg_ref).max()
        assert max_err < 1e-4, f"dgamma max_err={max_err:.2e} exceeds 1e-4"

    def test_dbeta_matches_pytorch(self):
        (_, _, db_our), (_, _, db_ref) = self._run()
        max_err = np.abs(db_our - db_ref).max()
        assert max_err < 1e-4, f"dbeta max_err={max_err:.2e} exceeds 1e-4"

    def test_larger_spatial(self):
        """Stress with larger H×W to catch reduction edge cases."""
        (dx_our, dg_our, db_our), (dx_ref, dg_ref, db_ref) = self._run(N=3, C=8, H=16, W=16, seed=99)
        assert np.abs(dx_our - dx_ref).max() < 1e-4
        assert np.abs(dg_our - dg_ref).max() < 1e-4
        assert np.abs(db_our - db_ref).max() < 1e-4

    def test_finite_difference_consistency(self):
        """FD check on dx: perturb x[0,0,0,0] and verify dx matches slope."""
        rng = np.random.default_rng(1)
        N, C, H, W = 1, 2, 4, 4
        x     = rng.standard_normal((N, C, H, W))
        gamma = np.ones(C)
        beta  = np.zeros(C)
        eps_ln = 1e-5
        eps_fd = 1e-4

        def fwd(xp):
            y, _, _, _ = _ln_forward(xp, gamma, beta, eps_ln)
            return y.sum()

        y0, x_hat, _, var = _ln_forward(x, gamma, beta, eps_ln)
        inv_std = 1.0 / np.sqrt(var + eps_ln)
        dy = np.ones_like(y0)
        dx_our, _, _ = _ln_backward(dy, x_hat, gamma, inv_std)

        # FD for each element
        dx_fd = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            xp = x.copy(); xp[idx] += eps_fd
            xm = x.copy(); xm[idx] -= eps_fd
            dx_fd[idx] = (fwd(xp) - fwd(xm)) / (2 * eps_fd)

        np.testing.assert_allclose(dx_our, dx_fd, atol=1e-3,
                                   err_msg="dx differs from finite difference")
