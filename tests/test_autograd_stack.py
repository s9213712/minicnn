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


def test_residual_block_supports_downsample_shortcut():
    block = ResidualBlock(in_channels=2, out_channels=4, stride=2)
    x = Tensor(np.random.randn(2, 2, 8, 8).astype(np.float32), requires_grad=True)
    y = block(x)
    assert y.data.shape == (2, 4, 4, 4)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.data.shape
    assert all(p.grad is not None for p in block.parameters())


def test_autograd_builder_supports_residual_block_stride_and_channel_change():
    from minicnn.models.builder import build_model_from_config

    cfg = {
        'input_shape': [2, 8, 8],
        'layers': [
            {'type': 'ResidualBlock', 'channels': 4, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 3},
        ],
    }
    model, final_shape = build_model_from_config(cfg)
    assert model.inferred_shapes[1] == (4, 4, 4)
    assert final_shape == (3,)


def test_batchnorm2d_eval_uses_running_statistics():
    bn = BatchNorm2d(2)
    train_x = Tensor(np.array([[[[1.0, 3.0]], [[2.0, 6.0]]]], dtype=np.float32))

    _ = bn(train_x)
    running_mean = bn.running_mean.copy()
    running_var = bn.running_var.copy()
    bn.eval()

    eval_x = Tensor(np.array([[[[100.0, 120.0]], [[-50.0, -10.0]]]], dtype=np.float32))
    y = bn(eval_x)
    expected = (eval_x.data - running_mean.reshape(1, -1, 1, 1)) / np.sqrt(
        running_var.reshape(1, -1, 1, 1) + bn.eps
    )

    assert np.allclose(y.data, expected, atol=1e-6)
    assert np.array_equal(bn.running_mean, running_mean)
    assert np.array_equal(bn.running_var, running_var)


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
