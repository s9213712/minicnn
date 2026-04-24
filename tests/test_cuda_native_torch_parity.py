from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip('torch')


def test_add_forward_matches_torch():
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'sum'},
        ],
        (2, 3, 4),
    )
    x = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)

    native = ForwardExecutor().run_inference(graph, x)
    expected = (torch.from_numpy(x) + torch.from_numpy(x)).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-6)


def test_concat_forward_matches_torch():
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Concat', 'inputs': ['left', 'right'], 'axis': 1, 'output': 'cat'},
        ],
        (2, 3, 4),
    )
    x = np.random.default_rng(1).standard_normal((2, 3, 4)).astype(np.float32)

    native = ForwardExecutor().run_inference(graph, x)
    expected = torch.cat([torch.from_numpy(x), torch.from_numpy(x)], dim=1).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-6)


def test_add_backward_matches_torch():
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Add', 'inputs': ['left', 'right'], 'output': 'sum'},
        ],
        (2, 3, 4),
    )
    x = np.random.default_rng(11).standard_normal((2, 3, 4)).astype(np.float32)
    grad_out = np.random.default_rng(12).standard_normal((2, 3, 4)).astype(np.float32)

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tout = tx + tx
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert native_param_grads == {}
    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-6)


def test_concat_backward_matches_torch():
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [
            {'type': 'Identity', 'output': 'stem'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'left'},
            {'type': 'Identity', 'inputs': ['stem'], 'output': 'right'},
            {'type': 'Concat', 'inputs': ['left', 'right'], 'axis': 1, 'output': 'cat'},
        ],
        (2, 3, 4),
    )
    x = np.random.default_rng(13).standard_normal((2, 3, 4)).astype(np.float32)
    grad_out = np.random.default_rng(14).standard_normal((2, 6, 4)).astype(np.float32)

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tout = torch.cat([tx, tx], dim=1)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert native_param_grads == {}
    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-6)


def test_linear_forward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'Linear', 'out_features': 3}],
        (2, 4),
    )
    x = np.random.default_rng(29).standard_normal((2, 4)).astype(np.float32)
    weight = np.random.default_rng(30).standard_normal((3, 4)).astype(np.float32)
    bias = np.random.default_rng(31).standard_normal((3,)).astype(np.float32)
    params = {
        '_w_linear_0': weight,
        '_b_linear_0': bias,
    }

    native = ForwardExecutor().run_inference(graph, x, params=params)
    expected = F.linear(
        torch.from_numpy(x),
        weight=torch.from_numpy(weight),
        bias=torch.from_numpy(bias),
    ).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-6, rtol=1e-6)


def test_linear_backward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'Linear', 'out_features': 3}],
        (2, 4),
    )
    x = np.random.default_rng(32).standard_normal((2, 4)).astype(np.float32)
    grad_out = np.random.default_rng(33).standard_normal((2, 3)).astype(np.float32)
    weight = np.random.default_rng(34).standard_normal((3, 4)).astype(np.float32)
    bias = np.random.default_rng(35).standard_normal((3,)).astype(np.float32)
    params = {
        '_w_linear_0': weight,
        '_b_linear_0': bias,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float32, requires_grad=True)
    tout = F.linear(tx, weight=tw, bias=tb)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)
    assert np.allclose(native_param_grads['_w_linear_0'], tw.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)
    assert np.allclose(native_param_grads['_b_linear_0'], tb.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)


def test_conv2d_forward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3, 'padding': 1}],
        (2, 3, 5, 5),
    )
    x = np.random.default_rng(36).standard_normal((2, 3, 5, 5)).astype(np.float32)
    weight = np.random.default_rng(37).standard_normal((2, 3, 3, 3)).astype(np.float32)
    bias = np.random.default_rng(38).standard_normal((2,)).astype(np.float32)
    params = {
        '_w_conv2d_0': weight,
        '_b_conv2d_0': bias,
    }

    native = ForwardExecutor().run_inference(graph, x, params=params)
    expected = F.conv2d(
        torch.from_numpy(x),
        weight=torch.from_numpy(weight),
        bias=torch.from_numpy(bias),
        stride=1,
        padding=1,
    ).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_conv2d_backward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3, 'padding': 1}],
        (2, 3, 5, 5),
    )
    x = np.random.default_rng(39).standard_normal((2, 3, 5, 5)).astype(np.float32)
    grad_out = np.random.default_rng(40).standard_normal((2, 2, 5, 5)).astype(np.float32)
    weight = np.random.default_rng(41).standard_normal((2, 3, 3, 3)).astype(np.float32)
    bias = np.random.default_rng(42).standard_normal((2,)).astype(np.float32)
    params = {
        '_w_conv2d_0': weight,
        '_b_conv2d_0': bias,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float32, requires_grad=True)
    tout = F.conv2d(tx, weight=tw, bias=tb, stride=1, padding=1)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_w_conv2d_0'], tw.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_b_conv2d_0'], tb.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_layernorm_forward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'LayerNorm', 'normalized_shape': [4, 5], 'eps': 1e-5}],
        (2, 3, 4, 5),
    )
    x = np.random.default_rng(2).standard_normal((2, 3, 4, 5)).astype(np.float32)
    gamma = np.random.default_rng(3).standard_normal((4, 5)).astype(np.float32)
    beta = np.random.default_rng(4).standard_normal((4, 5)).astype(np.float32)
    params = {
        '_w_layernorm_0': gamma,
        '_b_layernorm_0': beta,
    }

    native = ForwardExecutor().run_inference(graph, x, params=params)
    expected = F.layer_norm(
        torch.from_numpy(x),
        normalized_shape=(4, 5),
        weight=torch.from_numpy(gamma),
        bias=torch.from_numpy(beta),
        eps=1e-5,
    ).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_groupnorm_forward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'GroupNorm', 'num_groups': 2, 'eps': 1e-5}],
        (2, 4, 3, 3),
    )
    x = np.random.default_rng(15).standard_normal((2, 4, 3, 3)).astype(np.float32)
    gamma = np.random.default_rng(16).standard_normal((4,)).astype(np.float32)
    beta = np.random.default_rng(17).standard_normal((4,)).astype(np.float32)
    params = {
        '_w_groupnorm_0': gamma,
        '_b_groupnorm_0': beta,
    }

    native = ForwardExecutor().run_inference(graph, x, params=params)
    expected = F.group_norm(
        torch.from_numpy(x),
        num_groups=2,
        weight=torch.from_numpy(gamma),
        bias=torch.from_numpy(beta),
        eps=1e-5,
    ).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_batchnorm2d_eval_forward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'BatchNorm2d', 'eps': 1e-5}],
        (2, 3, 4, 4),
    )
    x = np.random.default_rng(43).standard_normal((2, 3, 4, 4)).astype(np.float32)
    gamma = np.random.default_rng(44).standard_normal((3,)).astype(np.float32)
    beta = np.random.default_rng(45).standard_normal((3,)).astype(np.float32)
    running_mean = np.random.default_rng(46).standard_normal((3,)).astype(np.float32)
    running_var = np.abs(np.random.default_rng(47).standard_normal((3,)).astype(np.float32)) + 0.5
    params = {
        '_w_batchnorm2d_0': gamma,
        '_b_batchnorm2d_0': beta,
        '_running_mean_batchnorm2d_0': running_mean.copy(),
        '_running_var_batchnorm2d_0': running_var.copy(),
    }

    native = ForwardExecutor().run_inference(graph, x, params=params, mode='eval')
    expected = F.batch_norm(
        torch.from_numpy(x),
        running_mean=torch.from_numpy(running_mean),
        running_var=torch.from_numpy(running_var),
        weight=torch.from_numpy(gamma),
        bias=torch.from_numpy(beta),
        training=False,
        momentum=0.1,
        eps=1e-5,
    ).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_batchnorm2d_eval_backward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'BatchNorm2d', 'eps': 1e-5}],
        (2, 3, 4, 4),
    )
    x = np.random.default_rng(48).standard_normal((2, 3, 4, 4)).astype(np.float32)
    grad_out = np.random.default_rng(49).standard_normal((2, 3, 4, 4)).astype(np.float32)
    gamma = np.random.default_rng(50).standard_normal((3,)).astype(np.float32)
    beta = np.random.default_rng(51).standard_normal((3,)).astype(np.float32)
    running_mean = np.random.default_rng(52).standard_normal((3,)).astype(np.float32)
    running_var = np.abs(np.random.default_rng(53).standard_normal((3,)).astype(np.float32)) + 0.5
    params = {
        '_w_batchnorm2d_0': gamma,
        '_b_batchnorm2d_0': beta,
        '_running_mean_batchnorm2d_0': running_mean.copy(),
        '_running_var_batchnorm2d_0': running_var.copy(),
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='eval')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw = torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(beta, dtype=torch.float32, requires_grad=True)
    tout = F.batch_norm(
        tx,
        running_mean=torch.tensor(running_mean, dtype=torch.float32),
        running_var=torch.tensor(running_var, dtype=torch.float32),
        weight=tw,
        bias=tb,
        training=False,
        momentum=0.1,
        eps=1e-5,
    )
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_w_batchnorm2d_0'], tw.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_b_batchnorm2d_0'], tb.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_batchnorm2d_train_forward_matches_torch_and_updates_running_stats():
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'BatchNorm2d', 'eps': 1e-5, 'momentum': 0.25}],
        (2, 3, 4, 4),
    )
    x = np.random.default_rng(54).standard_normal((2, 3, 4, 4)).astype(np.float32)
    gamma = np.random.default_rng(55).standard_normal((3,)).astype(np.float32)
    beta = np.random.default_rng(56).standard_normal((3,)).astype(np.float32)
    running_mean = np.random.default_rng(57).standard_normal((3,)).astype(np.float32)
    running_var = np.abs(np.random.default_rng(58).standard_normal((3,)).astype(np.float32)) + 0.5
    params = {
        '_w_batchnorm2d_0': gamma,
        '_b_batchnorm2d_0': beta,
        '_running_mean_batchnorm2d_0': running_mean.copy(),
        '_running_var_batchnorm2d_0': running_var.copy(),
    }

    native = ForwardExecutor().run_inference(graph, x, params=params, mode='train')

    bn = torch.nn.BatchNorm2d(
        num_features=3,
        eps=1e-5,
        momentum=0.25,
        affine=True,
        track_running_stats=True,
    )
    bn.train()
    with torch.no_grad():
        bn.weight.copy_(torch.from_numpy(gamma))
        bn.bias.copy_(torch.from_numpy(beta))
        bn.running_mean.copy_(torch.from_numpy(running_mean))
        bn.running_var.copy_(torch.from_numpy(running_var))
    expected = bn(torch.from_numpy(x)).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-5, rtol=1e-5)
    assert np.allclose(params['_running_mean_batchnorm2d_0'], bn.running_mean.detach().cpu().numpy(), atol=1e-6)
    assert np.allclose(params['_running_var_batchnorm2d_0'], bn.running_var.detach().cpu().numpy(), atol=1e-6)


def test_batchnorm2d_train_backward_matches_torch():
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'BatchNorm2d', 'eps': 1e-5, 'momentum': 0.25}],
        (2, 3, 4, 4),
    )
    x = np.random.default_rng(59).standard_normal((2, 3, 4, 4)).astype(np.float32)
    grad_out = np.random.default_rng(60).standard_normal((2, 3, 4, 4)).astype(np.float32)
    gamma = np.random.default_rng(61).standard_normal((3,)).astype(np.float32)
    beta = np.random.default_rng(62).standard_normal((3,)).astype(np.float32)
    running_mean = np.random.default_rng(63).standard_normal((3,)).astype(np.float32)
    running_var = np.abs(np.random.default_rng(64).standard_normal((3,)).astype(np.float32)) + 0.5
    params = {
        '_w_batchnorm2d_0': gamma,
        '_b_batchnorm2d_0': beta,
        '_running_mean_batchnorm2d_0': running_mean.copy(),
        '_running_var_batchnorm2d_0': running_var.copy(),
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    bn = torch.nn.BatchNorm2d(
        num_features=3,
        eps=1e-5,
        momentum=0.25,
        affine=True,
        track_running_stats=True,
    )
    bn.train()
    with torch.no_grad():
        bn.weight.copy_(torch.from_numpy(gamma))
        bn.bias.copy_(torch.from_numpy(beta))
        bn.running_mean.copy_(torch.from_numpy(running_mean))
        bn.running_var.copy_(torch.from_numpy(running_var))
    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tout = bn(tx)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_w_batchnorm2d_0'], bn.weight.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_b_batchnorm2d_0'], bn.bias.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_layernorm2d_forward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'LayerNorm2d', 'eps': 1e-6}],
        (2, 4, 3, 3),
    )
    x = np.random.default_rng(22).standard_normal((2, 4, 3, 3)).astype(np.float32)
    gamma = np.random.default_rng(23).standard_normal((4,)).astype(np.float32)
    beta = np.random.default_rng(24).standard_normal((4,)).astype(np.float32)
    params = {
        '_w_layernorm2d_0': gamma,
        '_b_layernorm2d_0': beta,
    }

    native = ForwardExecutor().run_inference(graph, x, params=params)
    tx = torch.from_numpy(x).permute(0, 2, 3, 1)
    expected = F.layer_norm(
        tx,
        normalized_shape=(4,),
        weight=torch.from_numpy(gamma),
        bias=torch.from_numpy(beta),
        eps=1e-6,
    ).permute(0, 3, 1, 2).contiguous().detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_layernorm_backward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'LayerNorm', 'normalized_shape': [4, 5], 'eps': 1e-5}],
        (2, 3, 4, 5),
    )
    x = np.random.default_rng(5).standard_normal((2, 3, 4, 5)).astype(np.float32)
    grad_out = np.random.default_rng(6).standard_normal((2, 3, 4, 5)).astype(np.float32)
    gamma = np.random.default_rng(7).standard_normal((4, 5)).astype(np.float32)
    beta = np.random.default_rng(8).standard_normal((4, 5)).astype(np.float32)
    params = {
        '_w_layernorm_0': gamma,
        '_b_layernorm_0': beta,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw = torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(beta, dtype=torch.float32, requires_grad=True)
    tout = F.layer_norm(tx, normalized_shape=(4, 5), weight=tw, bias=tb, eps=1e-5)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_w_layernorm_0'], tw.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_b_layernorm_0'], tb.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)


def test_groupnorm_backward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'GroupNorm', 'num_groups': 2, 'eps': 1e-5}],
        (2, 4, 3, 3),
    )
    x = np.random.default_rng(18).standard_normal((2, 4, 3, 3)).astype(np.float32)
    grad_out = np.random.default_rng(19).standard_normal((2, 4, 3, 3)).astype(np.float32)
    gamma = np.random.default_rng(20).standard_normal((4,)).astype(np.float32)
    beta = np.random.default_rng(21).standard_normal((4,)).astype(np.float32)
    params = {
        '_w_groupnorm_0': gamma,
        '_b_groupnorm_0': beta,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw = torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(beta, dtype=torch.float32, requires_grad=True)
    tout = F.group_norm(tx, num_groups=2, weight=tw, bias=tb, eps=1e-5)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_w_groupnorm_0'], tw.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_b_groupnorm_0'], tb.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)


def test_layernorm2d_backward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'LayerNorm2d', 'eps': 1e-6}],
        (2, 4, 3, 3),
    )
    x = np.random.default_rng(25).standard_normal((2, 4, 3, 3)).astype(np.float32)
    grad_out = np.random.default_rng(26).standard_normal((2, 4, 3, 3)).astype(np.float32)
    gamma = np.random.default_rng(27).standard_normal((4,)).astype(np.float32)
    beta = np.random.default_rng(28).standard_normal((4,)).astype(np.float32)
    params = {
        '_w_layernorm2d_0': gamma,
        '_b_layernorm2d_0': beta,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw = torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(beta, dtype=torch.float32, requires_grad=True)
    tout = F.layer_norm(
        tx.permute(0, 2, 3, 1),
        normalized_shape=(4,),
        weight=tw,
        bias=tb,
        eps=1e-6,
    ).permute(0, 3, 1, 2).contiguous()
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_w_layernorm2d_0'], tw.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_b_layernorm2d_0'], tb.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)
