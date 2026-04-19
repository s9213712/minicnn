import numpy as np

from minicnn.nn import Parameter, Tensor, cross_entropy
from minicnn.optim.sgd import SGD


def test_scalar_autograd_chain_rule():
    x = Tensor(2.0, requires_grad=True)
    y = (x * x + 3.0 * x).mean()

    y.backward()

    assert np.allclose(y.data, 10.0)
    assert np.allclose(x.grad, 7.0)


def test_matmul_mean_backward():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)

    y = (x @ w).mean()
    y.backward()

    assert np.allclose(y.data, 11.0)
    assert np.allclose(x.grad, [[3.0, 4.0]])
    assert np.allclose(w.grad, [[1.0], [2.0]])


def test_broadcast_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)

    y = (x + b).sum()
    y.backward()

    assert np.allclose(x.grad, np.ones((2, 2), dtype=np.float32))
    assert np.allclose(b.grad, [2.0, 2.0])


def test_cross_entropy_backward_matches_softmax_gradient():
    logits = Tensor([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]], requires_grad=True)
    targets = np.array([0, 1])

    loss = cross_entropy(logits, targets)
    loss.backward()

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    expected = probs
    expected[np.arange(2), targets] -= 1.0
    expected /= 2.0

    assert loss.data.shape == ()
    assert np.allclose(logits.grad, expected, atol=1e-6)


def test_parameter_and_sgd_step_without_torch():
    w = Parameter([1.0, -1.0], name='w')
    x = Tensor([2.0, 3.0])
    loss = ((w * x).sum() - 1.0) ** 2

    loss.backward()
    SGD([w], lr=0.1).step()

    assert np.allclose(w.grad, [-8.0, -12.0])
    assert np.allclose(w.data, [1.8, 0.2])
