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


def test_grouped_conv2d_forward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'padding': 1, 'groups': 4}],
        (2, 4, 5, 5),
    )
    x = np.random.default_rng(85).standard_normal((2, 4, 5, 5)).astype(np.float32)
    weight = np.random.default_rng(86).standard_normal((4, 1, 3, 3)).astype(np.float32)
    bias = np.random.default_rng(87).standard_normal((4,)).astype(np.float32)
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
        groups=4,
    ).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-5, rtol=1e-5)


def test_grouped_conv2d_backward_matches_torch():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'padding': 1, 'groups': 4}],
        (2, 4, 5, 5),
    )
    x = np.random.default_rng(88).standard_normal((2, 4, 5, 5)).astype(np.float32)
    grad_out = np.random.default_rng(89).standard_normal((2, 4, 5, 5)).astype(np.float32)
    weight = np.random.default_rng(90).standard_normal((4, 1, 3, 3)).astype(np.float32)
    bias = np.random.default_rng(91).standard_normal((4,)).astype(np.float32)
    params = {
        '_w_conv2d_0': weight,
        '_b_conv2d_0': bias,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(bias, dtype=torch.float32, requires_grad=True)
    tout = F.conv2d(tx, weight=tw, bias=tb, stride=1, padding=1, groups=4)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_w_conv2d_0'], tw.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads['_b_conv2d_0'], tb.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_residualblock_eval_forward_matches_torch_reference():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ResidualBlock', 'channels': 4, 'stride': 1, 'bias': False}],
        (2, 4, 5, 5),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(65).standard_normal((2, 4, 5, 5)).astype(np.float32)
    w1 = np.random.default_rng(66).standard_normal((4, 4, 3, 3)).astype(np.float32)
    w2 = np.random.default_rng(67).standard_normal((4, 4, 3, 3)).astype(np.float32)
    bn1_w = np.random.default_rng(68).standard_normal((4,)).astype(np.float32)
    bn1_b = np.random.default_rng(69).standard_normal((4,)).astype(np.float32)
    bn2_w = np.random.default_rng(70).standard_normal((4,)).astype(np.float32)
    bn2_b = np.random.default_rng(71).standard_normal((4,)).astype(np.float32)
    rm1 = np.random.default_rng(72).standard_normal((4,)).astype(np.float32)
    rv1 = np.abs(np.random.default_rng(73).standard_normal((4,)).astype(np.float32)) + 0.5
    rm2 = np.random.default_rng(74).standard_normal((4,)).astype(np.float32)
    rv2 = np.abs(np.random.default_rng(75).standard_normal((4,)).astype(np.float32)) + 0.5
    params = {
        f'_w_conv1_{node_name}': w1,
        f'_w_conv2_{node_name}': w2,
        f'_w_bn1_{node_name}': bn1_w,
        f'_b_bn1_{node_name}': bn1_b,
        f'_running_mean_bn1_{node_name}': rm1.copy(),
        f'_running_var_bn1_{node_name}': rv1.copy(),
        f'_w_bn2_{node_name}': bn2_w,
        f'_b_bn2_{node_name}': bn2_b,
        f'_running_mean_bn2_{node_name}': rm2.copy(),
        f'_running_var_bn2_{node_name}': rv2.copy(),
    }

    native = ForwardExecutor().run_inference(graph, x, params=params, mode='eval')

    tx = torch.from_numpy(x)
    t = F.conv2d(tx, weight=torch.from_numpy(w1), bias=None, stride=1, padding=1)
    t = F.batch_norm(
        t,
        running_mean=torch.from_numpy(rm1),
        running_var=torch.from_numpy(rv1),
        weight=torch.from_numpy(bn1_w),
        bias=torch.from_numpy(bn1_b),
        training=False,
        momentum=0.1,
        eps=1e-5,
    )
    t = F.relu(t)
    t = F.conv2d(t, weight=torch.from_numpy(w2), bias=None, stride=1, padding=1)
    t = F.batch_norm(
        t,
        running_mean=torch.from_numpy(rm2),
        running_var=torch.from_numpy(rv2),
        weight=torch.from_numpy(bn2_w),
        bias=torch.from_numpy(bn2_b),
        training=False,
        momentum=0.1,
        eps=1e-5,
    )
    expected = F.relu(t + tx).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-4, rtol=1e-4)


def test_residualblock_eval_backward_matches_torch_reference():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ResidualBlock', 'channels': 4, 'stride': 1, 'bias': False}],
        (2, 4, 5, 5),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(92).standard_normal((2, 4, 5, 5)).astype(np.float32)
    grad_out = np.random.default_rng(93).standard_normal((2, 4, 5, 5)).astype(np.float32)
    w1 = np.random.default_rng(94).standard_normal((4, 4, 3, 3)).astype(np.float32)
    w2 = np.random.default_rng(95).standard_normal((4, 4, 3, 3)).astype(np.float32)
    bn1_w = np.random.default_rng(96).standard_normal((4,)).astype(np.float32)
    bn1_b = np.random.default_rng(97).standard_normal((4,)).astype(np.float32)
    bn2_w = np.random.default_rng(98).standard_normal((4,)).astype(np.float32)
    bn2_b = np.random.default_rng(99).standard_normal((4,)).astype(np.float32)
    rm1 = np.random.default_rng(100).standard_normal((4,)).astype(np.float32)
    rv1 = np.abs(np.random.default_rng(101).standard_normal((4,)).astype(np.float32)) + 0.5
    rm2 = np.random.default_rng(102).standard_normal((4,)).astype(np.float32)
    rv2 = np.abs(np.random.default_rng(103).standard_normal((4,)).astype(np.float32)) + 0.5
    params = {
        f'_w_conv1_{node_name}': w1,
        f'_w_conv2_{node_name}': w2,
        f'_w_bn1_{node_name}': bn1_w,
        f'_b_bn1_{node_name}': bn1_b,
        f'_running_mean_bn1_{node_name}': rm1.copy(),
        f'_running_var_bn1_{node_name}': rv1.copy(),
        f'_w_bn2_{node_name}': bn2_w,
        f'_b_bn2_{node_name}': bn2_b,
        f'_running_mean_bn2_{node_name}': rm2.copy(),
        f'_running_var_bn2_{node_name}': rv2.copy(),
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='eval')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
    tw2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)
    tbn1_w = torch.tensor(bn1_w, dtype=torch.float32, requires_grad=True)
    tbn1_b = torch.tensor(bn1_b, dtype=torch.float32, requires_grad=True)
    tbn2_w = torch.tensor(bn2_w, dtype=torch.float32, requires_grad=True)
    tbn2_b = torch.tensor(bn2_b, dtype=torch.float32, requires_grad=True)
    t = F.conv2d(tx, weight=tw1, bias=None, stride=1, padding=1)
    t = F.batch_norm(
        t,
        running_mean=torch.tensor(rm1, dtype=torch.float32),
        running_var=torch.tensor(rv1, dtype=torch.float32),
        weight=tbn1_w,
        bias=tbn1_b,
        training=False,
        momentum=0.1,
        eps=1e-5,
    )
    t = F.relu(t)
    t = F.conv2d(t, weight=tw2, bias=None, stride=1, padding=1)
    t = F.batch_norm(
        t,
        running_mean=torch.tensor(rm2, dtype=torch.float32),
        running_var=torch.tensor(rv2, dtype=torch.float32),
        weight=tbn2_w,
        bias=tbn2_b,
        training=False,
        momentum=0.1,
        eps=1e-5,
    )
    tout = F.relu(t + tx)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads[f'_w_conv1_{node_name}'], tw1.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads[f'_w_conv2_{node_name}'], tw2.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads[f'_w_bn1_{node_name}'], tbn1_w.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads[f'_b_bn1_{node_name}'], tbn1_b.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads[f'_w_bn2_{node_name}'], tbn2_w.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)
    assert np.allclose(native_param_grads[f'_b_bn2_{node_name}'], tbn2_b.grad.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_residualblock_train_forward_with_shortcut_matches_torch_and_updates_running_stats():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ResidualBlock', 'channels': 8, 'stride': 2, 'bias': False, 'momentum': 0.2}],
        (2, 4, 6, 6),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(114).standard_normal((2, 4, 6, 6)).astype(np.float32)
    w1 = np.random.default_rng(115).standard_normal((8, 4, 3, 3)).astype(np.float32)
    w2 = np.random.default_rng(116).standard_normal((8, 8, 3, 3)).astype(np.float32)
    w_short = np.random.default_rng(117).standard_normal((8, 4, 1, 1)).astype(np.float32)
    bn1_w = np.random.default_rng(118).standard_normal((8,)).astype(np.float32)
    bn1_b = np.random.default_rng(119).standard_normal((8,)).astype(np.float32)
    bn2_w = np.random.default_rng(120).standard_normal((8,)).astype(np.float32)
    bn2_b = np.random.default_rng(121).standard_normal((8,)).astype(np.float32)
    short_bn_w = np.random.default_rng(122).standard_normal((8,)).astype(np.float32)
    short_bn_b = np.random.default_rng(123).standard_normal((8,)).astype(np.float32)
    rm1 = np.random.default_rng(124).standard_normal((8,)).astype(np.float32)
    rv1 = np.abs(np.random.default_rng(125).standard_normal((8,)).astype(np.float32)) + 0.5
    rm2 = np.random.default_rng(126).standard_normal((8,)).astype(np.float32)
    rv2 = np.abs(np.random.default_rng(127).standard_normal((8,)).astype(np.float32)) + 0.5
    rm_short = np.random.default_rng(128).standard_normal((8,)).astype(np.float32)
    rv_short = np.abs(np.random.default_rng(129).standard_normal((8,)).astype(np.float32)) + 0.5
    params = {
        f'_w_conv1_{node_name}': w1,
        f'_w_conv2_{node_name}': w2,
        f'_w_bn1_{node_name}': bn1_w,
        f'_b_bn1_{node_name}': bn1_b,
        f'_running_mean_bn1_{node_name}': rm1.copy(),
        f'_running_var_bn1_{node_name}': rv1.copy(),
        f'_w_bn2_{node_name}': bn2_w,
        f'_b_bn2_{node_name}': bn2_b,
        f'_running_mean_bn2_{node_name}': rm2.copy(),
        f'_running_var_bn2_{node_name}': rv2.copy(),
        f'_w_shortcut_conv_{node_name}': w_short,
        f'_w_shortcut_bn_{node_name}': short_bn_w,
        f'_b_shortcut_bn_{node_name}': short_bn_b,
        f'_running_mean_shortcut_bn_{node_name}': rm_short.copy(),
        f'_running_var_shortcut_bn_{node_name}': rv_short.copy(),
    }

    native = ForwardExecutor().run_inference(graph, x, params=params, mode='train')

    tx = torch.from_numpy(x)
    bn1 = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=0.2, affine=True, track_running_stats=True)
    bn2 = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=0.2, affine=True, track_running_stats=True)
    bn_short = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=0.2, affine=True, track_running_stats=True)
    bn1.train()
    bn2.train()
    bn_short.train()
    with torch.no_grad():
        bn1.weight.copy_(torch.from_numpy(bn1_w))
        bn1.bias.copy_(torch.from_numpy(bn1_b))
        bn1.running_mean.copy_(torch.from_numpy(rm1))
        bn1.running_var.copy_(torch.from_numpy(rv1))
        bn2.weight.copy_(torch.from_numpy(bn2_w))
        bn2.bias.copy_(torch.from_numpy(bn2_b))
        bn2.running_mean.copy_(torch.from_numpy(rm2))
        bn2.running_var.copy_(torch.from_numpy(rv2))
        bn_short.weight.copy_(torch.from_numpy(short_bn_w))
        bn_short.bias.copy_(torch.from_numpy(short_bn_b))
        bn_short.running_mean.copy_(torch.from_numpy(rm_short))
        bn_short.running_var.copy_(torch.from_numpy(rv_short))

    t = F.conv2d(tx, weight=torch.from_numpy(w1), bias=None, stride=2, padding=1)
    t = bn1(t)
    t = F.relu(t)
    t = F.conv2d(t, weight=torch.from_numpy(w2), bias=None, stride=1, padding=1)
    t = bn2(t)
    shortcut = F.conv2d(tx, weight=torch.from_numpy(w_short), bias=None, stride=2, padding=0)
    shortcut = bn_short(shortcut)
    expected = F.relu(t + shortcut).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=2e-4, rtol=2e-4)
    assert np.allclose(params[f'_running_mean_bn1_{node_name}'], bn1.running_mean.detach().cpu().numpy(), atol=1e-6)
    assert np.allclose(params[f'_running_var_bn1_{node_name}'], bn1.running_var.detach().cpu().numpy(), atol=1e-6)
    assert np.allclose(params[f'_running_mean_bn2_{node_name}'], bn2.running_mean.detach().cpu().numpy(), atol=1e-6)
    assert np.allclose(params[f'_running_var_bn2_{node_name}'], bn2.running_var.detach().cpu().numpy(), atol=1e-6)
    assert np.allclose(params[f'_running_mean_shortcut_bn_{node_name}'], bn_short.running_mean.detach().cpu().numpy(), atol=1e-6)
    assert np.allclose(params[f'_running_var_shortcut_bn_{node_name}'], bn_short.running_var.detach().cpu().numpy(), atol=1e-6)


def test_residualblock_train_backward_with_shortcut_matches_torch_reference():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ResidualBlock', 'channels': 8, 'stride': 2, 'bias': False, 'momentum': 0.2}],
        (2, 4, 6, 6),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(130).standard_normal((2, 4, 6, 6)).astype(np.float32)
    grad_out = np.random.default_rng(131).standard_normal((2, 8, 3, 3)).astype(np.float32)
    w1 = np.random.default_rng(132).standard_normal((8, 4, 3, 3)).astype(np.float32)
    w2 = np.random.default_rng(133).standard_normal((8, 8, 3, 3)).astype(np.float32)
    w_short = np.random.default_rng(134).standard_normal((8, 4, 1, 1)).astype(np.float32)
    bn1_w = np.random.default_rng(135).standard_normal((8,)).astype(np.float32)
    bn1_b = np.random.default_rng(136).standard_normal((8,)).astype(np.float32)
    bn2_w = np.random.default_rng(137).standard_normal((8,)).astype(np.float32)
    bn2_b = np.random.default_rng(138).standard_normal((8,)).astype(np.float32)
    short_bn_w = np.random.default_rng(139).standard_normal((8,)).astype(np.float32)
    short_bn_b = np.random.default_rng(140).standard_normal((8,)).astype(np.float32)
    rm1 = np.random.default_rng(141).standard_normal((8,)).astype(np.float32)
    rv1 = np.abs(np.random.default_rng(142).standard_normal((8,)).astype(np.float32)) + 0.5
    rm2 = np.random.default_rng(143).standard_normal((8,)).astype(np.float32)
    rv2 = np.abs(np.random.default_rng(144).standard_normal((8,)).astype(np.float32)) + 0.5
    rm_short = np.random.default_rng(145).standard_normal((8,)).astype(np.float32)
    rv_short = np.abs(np.random.default_rng(146).standard_normal((8,)).astype(np.float32)) + 0.5
    params = {
        f'_w_conv1_{node_name}': w1,
        f'_w_conv2_{node_name}': w2,
        f'_w_bn1_{node_name}': bn1_w,
        f'_b_bn1_{node_name}': bn1_b,
        f'_running_mean_bn1_{node_name}': rm1.copy(),
        f'_running_var_bn1_{node_name}': rv1.copy(),
        f'_w_bn2_{node_name}': bn2_w,
        f'_b_bn2_{node_name}': bn2_b,
        f'_running_mean_bn2_{node_name}': rm2.copy(),
        f'_running_var_bn2_{node_name}': rv2.copy(),
        f'_w_shortcut_conv_{node_name}': w_short,
        f'_w_shortcut_bn_{node_name}': short_bn_w,
        f'_b_shortcut_bn_{node_name}': short_bn_b,
        f'_running_mean_shortcut_bn_{node_name}': rm_short.copy(),
        f'_running_var_shortcut_bn_{node_name}': rv_short.copy(),
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='train')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
    tw2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)
    tw_short = torch.tensor(w_short, dtype=torch.float32, requires_grad=True)
    bn1 = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=0.2, affine=True, track_running_stats=True)
    bn2 = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=0.2, affine=True, track_running_stats=True)
    bn_short = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=0.2, affine=True, track_running_stats=True)
    bn1.train()
    bn2.train()
    bn_short.train()
    with torch.no_grad():
        bn1.weight.copy_(torch.from_numpy(bn1_w))
        bn1.bias.copy_(torch.from_numpy(bn1_b))
        bn1.running_mean.copy_(torch.from_numpy(rm1))
        bn1.running_var.copy_(torch.from_numpy(rv1))
        bn2.weight.copy_(torch.from_numpy(bn2_w))
        bn2.bias.copy_(torch.from_numpy(bn2_b))
        bn2.running_mean.copy_(torch.from_numpy(rm2))
        bn2.running_var.copy_(torch.from_numpy(rv2))
        bn_short.weight.copy_(torch.from_numpy(short_bn_w))
        bn_short.bias.copy_(torch.from_numpy(short_bn_b))
        bn_short.running_mean.copy_(torch.from_numpy(rm_short))
        bn_short.running_var.copy_(torch.from_numpy(rv_short))

    t = F.conv2d(tx, weight=tw1, bias=None, stride=2, padding=1)
    t = bn1(t)
    t = F.relu(t)
    t = F.conv2d(t, weight=tw2, bias=None, stride=1, padding=1)
    t = bn2(t)
    shortcut = F.conv2d(tx, weight=tw_short, bias=None, stride=2, padding=0)
    shortcut = bn_short(shortcut)
    tout = F.relu(t + shortcut)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_conv1_{node_name}'], tw1.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_conv2_{node_name}'], tw2.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_shortcut_conv_{node_name}'], tw_short.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_bn1_{node_name}'], bn1.weight.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_b_bn1_{node_name}'], bn1.bias.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_bn2_{node_name}'], bn2.weight.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_b_bn2_{node_name}'], bn2.bias.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_shortcut_bn_{node_name}'], bn_short.weight.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_b_shortcut_bn_{node_name}'], bn_short.bias.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)


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


def test_convnextblock_forward_matches_torch_reference():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ConvNeXtBlock', 'kernel_size': 7, 'bias': True}],
        (2, 4, 5, 5),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(76).standard_normal((2, 4, 5, 5)).astype(np.float32)
    w_dw = np.random.default_rng(77).standard_normal((4, 1, 7, 7)).astype(np.float32)
    b_dw = np.random.default_rng(78).standard_normal((4,)).astype(np.float32)
    ln_w = np.random.default_rng(79).standard_normal((4,)).astype(np.float32)
    ln_b = np.random.default_rng(80).standard_normal((4,)).astype(np.float32)
    w_pw1 = np.random.default_rng(81).standard_normal((16, 4, 1, 1)).astype(np.float32)
    b_pw1 = np.random.default_rng(82).standard_normal((16,)).astype(np.float32)
    w_pw2 = np.random.default_rng(83).standard_normal((4, 16, 1, 1)).astype(np.float32)
    b_pw2 = np.random.default_rng(84).standard_normal((4,)).astype(np.float32)
    params = {
        f'_w_depthwise_{node_name}': w_dw,
        f'_b_depthwise_{node_name}': b_dw,
        f'_w_ln_{node_name}': ln_w,
        f'_b_ln_{node_name}': ln_b,
        f'_w_pw1_{node_name}': w_pw1,
        f'_b_pw1_{node_name}': b_pw1,
        f'_w_pw2_{node_name}': w_pw2,
        f'_b_pw2_{node_name}': b_pw2,
    }

    native = ForwardExecutor().run_inference(graph, x, params=params, mode='eval')

    tx = torch.from_numpy(x)
    t = F.conv2d(
        tx,
        weight=torch.from_numpy(w_dw),
        bias=torch.from_numpy(b_dw),
        stride=1,
        padding=3,
        groups=4,
    )
    t = F.layer_norm(
        t.permute(0, 2, 3, 1),
        normalized_shape=(4,),
        weight=torch.from_numpy(ln_w),
        bias=torch.from_numpy(ln_b),
        eps=1e-6,
    ).permute(0, 3, 1, 2).contiguous()
    t = F.conv2d(t, weight=torch.from_numpy(w_pw1), bias=torch.from_numpy(b_pw1), stride=1, padding=0)
    t = F.gelu(t, approximate='tanh')
    t = F.conv2d(t, weight=torch.from_numpy(w_pw2), bias=torch.from_numpy(b_pw2), stride=1, padding=0)
    expected = (tx + t).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=1e-4, rtol=1e-4)


def test_convnextblock_backward_matches_torch_reference():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ConvNeXtBlock', 'kernel_size': 7, 'bias': True}],
        (2, 4, 5, 5),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(104).standard_normal((2, 4, 5, 5)).astype(np.float32)
    grad_out = np.random.default_rng(105).standard_normal((2, 4, 5, 5)).astype(np.float32)
    w_dw = np.random.default_rng(106).standard_normal((4, 1, 7, 7)).astype(np.float32)
    b_dw = np.random.default_rng(107).standard_normal((4,)).astype(np.float32)
    ln_w = np.random.default_rng(108).standard_normal((4,)).astype(np.float32)
    ln_b = np.random.default_rng(109).standard_normal((4,)).astype(np.float32)
    w_pw1 = np.random.default_rng(110).standard_normal((16, 4, 1, 1)).astype(np.float32)
    b_pw1 = np.random.default_rng(111).standard_normal((16,)).astype(np.float32)
    w_pw2 = np.random.default_rng(112).standard_normal((4, 16, 1, 1)).astype(np.float32)
    b_pw2 = np.random.default_rng(113).standard_normal((4,)).astype(np.float32)
    params = {
        f'_w_depthwise_{node_name}': w_dw,
        f'_b_depthwise_{node_name}': b_dw,
        f'_w_ln_{node_name}': ln_w,
        f'_b_ln_{node_name}': ln_b,
        f'_w_pw1_{node_name}': w_pw1,
        f'_b_pw1_{node_name}': b_pw1,
        f'_w_pw2_{node_name}': w_pw2,
        f'_b_pw2_{node_name}': b_pw2,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='eval')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw_dw = torch.tensor(w_dw, dtype=torch.float32, requires_grad=True)
    tb_dw = torch.tensor(b_dw, dtype=torch.float32, requires_grad=True)
    tln_w = torch.tensor(ln_w, dtype=torch.float32, requires_grad=True)
    tln_b = torch.tensor(ln_b, dtype=torch.float32, requires_grad=True)
    tw_pw1 = torch.tensor(w_pw1, dtype=torch.float32, requires_grad=True)
    tb_pw1 = torch.tensor(b_pw1, dtype=torch.float32, requires_grad=True)
    tw_pw2 = torch.tensor(w_pw2, dtype=torch.float32, requires_grad=True)
    tb_pw2 = torch.tensor(b_pw2, dtype=torch.float32, requires_grad=True)
    t = F.conv2d(tx, weight=tw_dw, bias=tb_dw, stride=1, padding=3, groups=4)
    t = F.layer_norm(
        t.permute(0, 2, 3, 1),
        normalized_shape=(4,),
        weight=tln_w,
        bias=tln_b,
        eps=1e-6,
    ).permute(0, 3, 1, 2).contiguous()
    t = F.conv2d(t, weight=tw_pw1, bias=tb_pw1, stride=1, padding=0)
    t = F.gelu(t, approximate='tanh')
    t = F.conv2d(t, weight=tw_pw2, bias=tb_pw2, stride=1, padding=0)
    tout = tx + t
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_w_depthwise_{node_name}'], tw_dw.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_b_depthwise_{node_name}'], tb_dw.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_w_ln_{node_name}'], tln_w.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_b_ln_{node_name}'], tln_b.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_w_pw1_{node_name}'], tw_pw1.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_b_pw1_{node_name}'], tb_pw1.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_w_pw2_{node_name}'], tw_pw2.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)
    assert np.allclose(native_param_grads[f'_b_pw2_{node_name}'], tb_pw2.grad.detach().cpu().numpy(), atol=2e-4, rtol=2e-4)


def test_convnextblock_forward_with_layer_scale_matches_torch_reference():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ConvNeXtBlock', 'kernel_size': 7, 'bias': True, 'layer_scale_init_value': 0.25}],
        (2, 4, 5, 5),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(147).standard_normal((2, 4, 5, 5)).astype(np.float32)
    w_dw = np.random.default_rng(148).standard_normal((4, 1, 7, 7)).astype(np.float32)
    b_dw = np.random.default_rng(149).standard_normal((4,)).astype(np.float32)
    ln_w = np.random.default_rng(150).standard_normal((4,)).astype(np.float32)
    ln_b = np.random.default_rng(151).standard_normal((4,)).astype(np.float32)
    w_pw1 = np.random.default_rng(152).standard_normal((16, 4, 1, 1)).astype(np.float32)
    b_pw1 = np.random.default_rng(153).standard_normal((16,)).astype(np.float32)
    w_pw2 = np.random.default_rng(154).standard_normal((4, 16, 1, 1)).astype(np.float32)
    b_pw2 = np.random.default_rng(155).standard_normal((4,)).astype(np.float32)
    layer_scale = np.random.default_rng(156).standard_normal((4,)).astype(np.float32)
    params = {
        f'_w_depthwise_{node_name}': w_dw,
        f'_b_depthwise_{node_name}': b_dw,
        f'_w_ln_{node_name}': ln_w,
        f'_b_ln_{node_name}': ln_b,
        f'_w_pw1_{node_name}': w_pw1,
        f'_b_pw1_{node_name}': b_pw1,
        f'_w_pw2_{node_name}': w_pw2,
        f'_b_pw2_{node_name}': b_pw2,
        f'_layer_scale_{node_name}': layer_scale,
    }

    native = ForwardExecutor().run_inference(graph, x, params=params, mode='eval')

    tx = torch.from_numpy(x)
    t = F.conv2d(tx, weight=torch.from_numpy(w_dw), bias=torch.from_numpy(b_dw), stride=1, padding=3, groups=4)
    t = F.layer_norm(
        t.permute(0, 2, 3, 1),
        normalized_shape=(4,),
        weight=torch.from_numpy(ln_w),
        bias=torch.from_numpy(ln_b),
        eps=1e-6,
    ).permute(0, 3, 1, 2).contiguous()
    t = F.conv2d(t, weight=torch.from_numpy(w_pw1), bias=torch.from_numpy(b_pw1), stride=1, padding=0)
    t = F.gelu(t, approximate='tanh')
    t = F.conv2d(t, weight=torch.from_numpy(w_pw2), bias=torch.from_numpy(b_pw2), stride=1, padding=0)
    expected = (tx + t * torch.from_numpy(layer_scale).view(1, -1, 1, 1)).detach().cpu().numpy()

    assert np.allclose(native, expected, atol=2e-4, rtol=2e-4)


def test_convnextblock_backward_with_layer_scale_matches_torch_reference():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.graph import build_graph

    graph = build_graph(
        [{'type': 'ConvNeXtBlock', 'kernel_size': 7, 'bias': True, 'layer_scale_init_value': 0.25}],
        (2, 4, 5, 5),
    )
    node_name = graph.nodes[0].name
    x = np.random.default_rng(157).standard_normal((2, 4, 5, 5)).astype(np.float32)
    grad_out = np.random.default_rng(158).standard_normal((2, 4, 5, 5)).astype(np.float32)
    w_dw = np.random.default_rng(159).standard_normal((4, 1, 7, 7)).astype(np.float32)
    b_dw = np.random.default_rng(160).standard_normal((4,)).astype(np.float32)
    ln_w = np.random.default_rng(161).standard_normal((4,)).astype(np.float32)
    ln_b = np.random.default_rng(162).standard_normal((4,)).astype(np.float32)
    w_pw1 = np.random.default_rng(163).standard_normal((16, 4, 1, 1)).astype(np.float32)
    b_pw1 = np.random.default_rng(164).standard_normal((16,)).astype(np.float32)
    w_pw2 = np.random.default_rng(165).standard_normal((4, 16, 1, 1)).astype(np.float32)
    b_pw2 = np.random.default_rng(166).standard_normal((4,)).astype(np.float32)
    layer_scale = np.random.default_rng(167).standard_normal((4,)).astype(np.float32)
    params = {
        f'_w_depthwise_{node_name}': w_dw,
        f'_b_depthwise_{node_name}': b_dw,
        f'_w_ln_{node_name}': ln_w,
        f'_b_ln_{node_name}': ln_b,
        f'_w_pw1_{node_name}': w_pw1,
        f'_b_pw1_{node_name}': b_pw1,
        f'_w_pw2_{node_name}': w_pw2,
        f'_b_pw2_{node_name}': b_pw2,
        f'_layer_scale_{node_name}': layer_scale,
    }

    _ctx, cache = ForwardExecutor().run_with_cache(graph, {'input': x}, params=params, mode='eval')
    native_grad_input, native_param_grads = BackwardExecutor().run(graph, grad_out, cache)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw_dw = torch.tensor(w_dw, dtype=torch.float32, requires_grad=True)
    tb_dw = torch.tensor(b_dw, dtype=torch.float32, requires_grad=True)
    tln_w = torch.tensor(ln_w, dtype=torch.float32, requires_grad=True)
    tln_b = torch.tensor(ln_b, dtype=torch.float32, requires_grad=True)
    tw_pw1 = torch.tensor(w_pw1, dtype=torch.float32, requires_grad=True)
    tb_pw1 = torch.tensor(b_pw1, dtype=torch.float32, requires_grad=True)
    tw_pw2 = torch.tensor(w_pw2, dtype=torch.float32, requires_grad=True)
    tb_pw2 = torch.tensor(b_pw2, dtype=torch.float32, requires_grad=True)
    tls = torch.tensor(layer_scale, dtype=torch.float32, requires_grad=True)
    t = F.conv2d(tx, weight=tw_dw, bias=tb_dw, stride=1, padding=3, groups=4)
    t = F.layer_norm(
        t.permute(0, 2, 3, 1),
        normalized_shape=(4,),
        weight=tln_w,
        bias=tln_b,
        eps=1e-6,
    ).permute(0, 3, 1, 2).contiguous()
    t = F.conv2d(t, weight=tw_pw1, bias=tb_pw1, stride=1, padding=0)
    t = F.gelu(t, approximate='tanh')
    t = F.conv2d(t, weight=tw_pw2, bias=tb_pw2, stride=1, padding=0)
    tout = tx + t * tls.view(1, -1, 1, 1)
    tout.backward(torch.tensor(grad_out, dtype=torch.float32))

    assert np.allclose(native_grad_input, tx.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_depthwise_{node_name}'], tw_dw.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_b_depthwise_{node_name}'], tb_dw.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_ln_{node_name}'], tln_w.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_b_ln_{node_name}'], tln_b.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_pw1_{node_name}'], tw_pw1.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_b_pw1_{node_name}'], tb_pw1.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_w_pw2_{node_name}'], tw_pw2.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_b_pw2_{node_name}'], tb_pw2.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)
    assert np.allclose(native_param_grads[f'_layer_scale_{node_name}'], tls.grad.detach().cpu().numpy(), atol=3e-4, rtol=3e-4)


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
