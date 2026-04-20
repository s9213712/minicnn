import numpy as np
import pytest

from minicnn.nn import BatchNorm2d, Linear, Tensor, bce_with_logits_loss, mse_loss


torch = pytest.importorskip("torch")


def test_linear_forward_backward_matches_torch():
    rng = np.random.default_rng(123)
    x_np = rng.standard_normal((3, 4)).astype(np.float32)
    w_np = rng.standard_normal((4, 2)).astype(np.float32)
    b_np = rng.standard_normal(2).astype(np.float32)

    layer = Linear(4, 2)
    layer.weight.data[...] = w_np
    layer.bias.data[...] = b_np
    x = Tensor(x_np, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    x_ref = torch.tensor(x_np, requires_grad=True)
    w_ref = torch.tensor(w_np.T, requires_grad=True)
    b_ref = torch.tensor(b_np, requires_grad=True)
    y_ref = torch.nn.functional.linear(x_ref, w_ref, b_ref)
    y_ref.sum().backward()

    np.testing.assert_allclose(y.data, y_ref.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(x.grad, x_ref.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(layer.weight.grad, w_ref.grad.numpy().T, atol=1e-6)
    np.testing.assert_allclose(layer.bias.grad, b_ref.grad.numpy(), atol=1e-6)


def test_mse_loss_backward_matches_torch():
    predictions_np = np.array([[0.5, -1.0], [2.0, 3.0]], dtype=np.float32)
    targets_np = np.array([[1.5, 0.0], [1.0, 2.5]], dtype=np.float32)

    predictions = Tensor(predictions_np, requires_grad=True)
    loss = mse_loss(predictions, targets_np)
    loss.backward()

    predictions_ref = torch.tensor(predictions_np, requires_grad=True)
    targets_ref = torch.tensor(targets_np)
    loss_ref = torch.nn.functional.mse_loss(predictions_ref, targets_ref)
    loss_ref.backward()

    np.testing.assert_allclose(loss.data, loss_ref.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(predictions.grad, predictions_ref.grad.numpy(), atol=1e-6)


def test_bce_with_logits_loss_backward_matches_torch():
    logits_np = np.array([[0.0, 2.0], [-1.0, 4.0]], dtype=np.float32)
    targets_np = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    logits = Tensor(logits_np, requires_grad=True)
    loss = bce_with_logits_loss(logits, targets_np)
    loss.backward()

    logits_ref = torch.tensor(logits_np, requires_grad=True)
    targets_ref = torch.tensor(targets_np)
    loss_ref = torch.nn.functional.binary_cross_entropy_with_logits(logits_ref, targets_ref)
    loss_ref.backward()

    np.testing.assert_allclose(loss.data, loss_ref.detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(logits.grad, logits_ref.grad.numpy(), atol=1e-6)


def test_batchnorm2d_forward_backward_matches_torch_train_mode():
    rng = np.random.default_rng(456)
    x_np = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    gamma_np = rng.standard_normal(3).astype(np.float32)
    beta_np = rng.standard_normal(3).astype(np.float32)

    bn = BatchNorm2d(3, eps=1e-5, momentum=0.1)
    bn.weight.data[...] = gamma_np
    bn.bias.data[...] = beta_np
    x = Tensor(x_np, requires_grad=True)
    y = bn(x)
    y.sum().backward()

    bn_ref = torch.nn.BatchNorm2d(3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    with torch.no_grad():
        bn_ref.weight.copy_(torch.tensor(gamma_np))
        bn_ref.bias.copy_(torch.tensor(beta_np))
        bn_ref.running_mean.zero_()
        bn_ref.running_var.fill_(1.0)
    bn_ref.train()
    x_ref = torch.tensor(x_np, requires_grad=True)
    y_ref = bn_ref(x_ref)
    y_ref.sum().backward()

    np.testing.assert_allclose(y.data, y_ref.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x.grad, x_ref.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(bn.weight.grad, bn_ref.weight.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(bn.bias.grad, bn_ref.bias.grad.numpy(), atol=1e-5)
