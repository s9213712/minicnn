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


def test_sgd_momentum_accumulates_velocity():
    w = Parameter([0.0], name='w')
    opt = SGD([w], lr=0.1, momentum=0.9)

    w.grad = np.array([1.0], dtype=np.float32)
    opt.step()
    assert np.allclose(w.data, [-0.1], atol=1e-6)

    w.grad = np.array([1.0], dtype=np.float32)
    opt.step()
    assert np.allclose(w.data, [-0.29], atol=1e-6)


def test_sgd_without_momentum_is_vanilla_gradient_descent():
    w = Parameter([0.0], name='w')
    opt = SGD([w], lr=0.1, momentum=0.0)

    w.grad = np.array([1.0], dtype=np.float32)
    opt.step()
    w.grad = np.array([1.0], dtype=np.float32)
    opt.step()

    assert np.allclose(w.data, [-0.2], atol=1e-6)


def test_function_apply_wires_backward():
    """Function.apply() should propagate gradients through custom ops."""
    from minicnn.autograd.function import Function
    from minicnn.nn.tensor import Tensor
    import numpy as np

    class Square(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return Tensor(x.data ** 2, requires_grad=x.requires_grad)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * 2.0 * x.data

    x = Tensor([3.0, -2.0], requires_grad=True)
    y = Square.apply(x)
    y.sum().backward()

    assert np.allclose(y.data, [9.0, 4.0])
    assert np.allclose(x.grad, [6.0, -4.0])


def test_sgd_grad_clip():
    from minicnn.optim.sgd import SGD
    from minicnn.nn.tensor import Parameter
    import numpy as np

    w = Parameter([0.0, 0.0])
    opt = SGD([w], lr=1.0, grad_clip=1.0)
    w.grad = np.array([3.0, 4.0], dtype=np.float32)  # norm=5, should be clipped to norm=1
    opt.step()
    assert np.allclose(np.linalg.norm(w.data), 1.0, atol=1e-5)


def test_adam_grad_clip():
    from minicnn.optim.adam import Adam
    from minicnn.nn.tensor import Parameter
    import numpy as np

    w = Parameter([0.0, 0.0])
    opt = Adam([w], lr=0.1, grad_clip=0.5)
    w.grad = np.array([10.0, 0.0], dtype=np.float32)
    opt.step()
    # After clipping, gradient norm is 0.5; Adam still updates but in clipped direction
    assert w.data[0] < 0.0  # should have decreased
    assert w.data[1] == 0.0  # no gradient, no update
