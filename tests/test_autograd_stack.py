import numpy as np

from minicnn.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, ReLU, ResidualBlock, Sequential, Tensor, no_grad
from minicnn.optim import Adam


def test_linear_relu_flatten_backward():
    model = Sequential(Flatten(), Linear(4, 3), ReLU(), Linear(3, 2))
    x = Tensor(np.ones((2, 1, 2, 2), dtype=np.float32))
    y = model(x).sum()

    y.backward()

    assert model[1].weight.grad.shape == model[1].weight.data.shape
    assert model[3].weight.grad.shape == model[3].weight.data.shape


def test_conv_pool_batchnorm_residual_backward_shapes():
    conv = Conv2d(1, 2, kernel_size=3, padding=1)
    pool = MaxPool2d(2, 2)
    bn = BatchNorm2d(2)
    x = Tensor(np.random.randn(2, 1, 4, 4).astype(np.float32), requires_grad=True)

    y = bn(pool(conv(x))).sum()
    y.backward()

    assert x.grad.shape == x.data.shape
    assert conv.weight.grad.shape == conv.weight.data.shape
    assert bn.weight.grad.shape == bn.weight.data.shape

    block = ResidualBlock(2)
    z = block(Tensor(np.random.randn(2, 2, 4, 4).astype(np.float32), requires_grad=True)).sum()
    z.backward()
    assert all(p.grad is not None for p in block.parameters())


def test_no_grad_and_adam_step():
    w = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=True)

    with no_grad():
        y = w(x)

    assert y.requires_grad is False

    loss = w(x).sum()
    loss.backward()
    before = w.weight.data.copy()
    Adam(w.parameters(), lr=0.01).step()

    assert not np.allclose(before, w.weight.data)
