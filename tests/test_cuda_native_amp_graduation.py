from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def _build_linear_relu_linear_graph():
    from minicnn.cuda_native.graph import build_graph

    return build_graph(
        [
            {"type": "Linear", "out_features": 6},
            {"type": "ReLU"},
            {"type": "Linear", "out_features": 3},
        ],
        (4, 5),
    )


def _make_linear_relu_linear_params():
    rng = np.random.default_rng(7)
    return {
        "_w_linear_0": (rng.standard_normal((6, 5)) * 0.1).astype(np.float32),
        "_b_linear_0": (rng.standard_normal((6,)) * 0.05).astype(np.float32),
        "_w_linear_2": (rng.standard_normal((3, 6)) * 0.1).astype(np.float32),
        "_b_linear_2": (rng.standard_normal((3,)) * 0.05).astype(np.float32),
    }


def _native_amp_param_grads(graph, x, y, params):
    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.loss import cross_entropy_loss

    fwd = ForwardExecutor()
    bwd = BackwardExecutor()
    amp_params = {
        key: value.astype(np.float16) if np.issubdtype(value.dtype, np.floating) else value
        for key, value in params.items()
    }
    ctx, cache = fwd.run_with_cache(
        graph,
        {"input": x.astype(np.float16)},
        params=amp_params,
        mode="train",
    )
    logits = ctx[graph.output_spec.name]
    _loss, grad_logits = cross_entropy_loss(logits, y)
    grad_logits = (grad_logits * 128.0).astype(np.float32)
    _grad_input, param_grads = bwd.run(graph, grad_logits, cache)
    return {
        key: (grad / 128.0).astype(np.float32)
        for key, grad in param_grads.items()
    }


def test_cuda_native_amp_forward_matches_fp32_and_torch_autocast_baseline():
    import torch.nn.functional as F

    from minicnn.cuda_native.executor import ForwardExecutor

    graph = _build_linear_relu_linear_graph()
    params = _make_linear_relu_linear_params()
    x = np.random.default_rng(11).standard_normal((4, 5)).astype(np.float32)

    native_fp32 = ForwardExecutor().run_inference(graph, x, params=params)
    native_amp = ForwardExecutor().run_inference(
        graph,
        x.astype(np.float16),
        params={key: value.astype(np.float16) for key, value in params.items()},
    ).astype(np.float32)

    tx = torch.from_numpy(x)
    tw1 = torch.from_numpy(params["_w_linear_0"])
    tb1 = torch.from_numpy(params["_b_linear_0"])
    tw2 = torch.from_numpy(params["_w_linear_2"])
    tb2 = torch.from_numpy(params["_b_linear_2"])
    torch_fp32 = F.linear(F.relu(F.linear(tx, tw1, tb1)), tw2, tb2).detach().cpu().numpy()
    with torch.amp.autocast("cpu"):
        torch_amp = F.linear(F.relu(F.linear(tx, tw1, tb1)), tw2, tb2)
    torch_amp = torch_amp.detach().cpu().float().numpy()

    assert np.allclose(native_fp32, torch_fp32, atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(native_amp - native_fp32)) <= 5e-2
    assert np.max(np.abs(torch_amp - torch_fp32)) <= 5e-2
    assert np.max(np.abs(native_amp - torch_amp)) <= 8e-2


def test_cuda_native_amp_backward_gradients_stay_within_tolerance():
    import torch.nn.functional as F

    from minicnn.cuda_native.backward import BackwardExecutor
    from minicnn.cuda_native.executor import ForwardExecutor
    from minicnn.cuda_native.loss import cross_entropy_loss

    graph = _build_linear_relu_linear_graph()
    params = _make_linear_relu_linear_params()
    x = np.random.default_rng(12).standard_normal((4, 5)).astype(np.float32)
    y = np.array([0, 1, 2, 1], dtype=np.int64)

    ctx, cache = ForwardExecutor().run_with_cache(graph, {"input": x}, params=params, mode="train")
    logits = ctx[graph.output_spec.name]
    _loss, grad_logits = cross_entropy_loss(logits, y)
    _grad_input, fp32_param_grads = BackwardExecutor().run(graph, grad_logits, cache)
    amp_param_grads = _native_amp_param_grads(graph, x, y, params)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tw1 = torch.tensor(params["_w_linear_0"], dtype=torch.float32, requires_grad=True)
    tb1 = torch.tensor(params["_b_linear_0"], dtype=torch.float32, requires_grad=True)
    tw2 = torch.tensor(params["_w_linear_2"], dtype=torch.float32, requires_grad=True)
    tb2 = torch.tensor(params["_b_linear_2"], dtype=torch.float32, requires_grad=True)
    tout = F.linear(F.relu(F.linear(tx, tw1, tb1)), tw2, tb2)
    F.cross_entropy(tout, torch.tensor(y, dtype=torch.int64)).backward()
    torch_fp32_grads = {
        "_w_linear_0": tw1.grad.detach().cpu().numpy(),
        "_b_linear_0": tb1.grad.detach().cpu().numpy(),
        "_w_linear_2": tw2.grad.detach().cpu().numpy(),
        "_b_linear_2": tb2.grad.detach().cpu().numpy(),
    }

    for key, fp32_grad in fp32_param_grads.items():
        assert np.allclose(fp32_grad, torch_fp32_grads[key], atol=1e-5, rtol=1e-5)
        assert np.max(np.abs(amp_param_grads[key] - fp32_grad)) <= 5e-2
        assert np.max(np.abs(amp_param_grads[key] - torch_fp32_grads[key])) <= 5e-2


def test_cuda_native_amp_loss_scale_growth_matches_torch_gradscaler_policy():
    import torch.nn.functional as F

    from minicnn.cuda_native.training import train_step

    graph = _build_linear_relu_linear_graph()
    params = _make_linear_relu_linear_params()
    native_state: dict[str, object] = {}
    x = np.random.default_rng(21).standard_normal((4, 5)).astype(np.float32)
    y = np.array([0, 1, 2, 1], dtype=np.int64)

    for _ in range(2):
        _loss, params = train_step(
            graph,
            x,
            y,
            params,
            lr=0.01,
            optimizer_type="adamw",
            amp_enabled=True,
            amp_loss_scale=128.0,
            amp_dynamic_scale=True,
            amp_scale_growth=2.0,
            amp_scale_backoff=0.5,
            amp_scale_window=2,
            optimizer_state=native_state,
        )

    torch.manual_seed(21)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 3),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_w_linear_0"]))
        model[0].bias.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_b_linear_0"]))
        model[2].weight.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_w_linear_2"]))
        model[2].bias.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_b_linear_2"]))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=128.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
    )
    tx = torch.from_numpy(x)
    ty = torch.tensor(y, dtype=torch.int64)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cpu"):
            loss = F.cross_entropy(model(tx), ty)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    assert native_state["amp"]["loss_scale"] == scaler.get_scale() == 256.0
    assert native_state["amp"]["overflow_steps"] == 0
    assert native_state["amp"]["skipped_steps"] == 0


def test_cuda_native_amp_overflow_backoff_matches_torch_gradscaler_policy():
    import torch.nn.functional as F

    from minicnn.cuda_native.training import train_step

    graph = _build_linear_relu_linear_graph()
    params = _make_linear_relu_linear_params()
    native_state: dict[str, object] = {}
    x = np.full((4, 5), 1e10, dtype=np.float32)
    y = np.array([0, 1, 2, 1], dtype=np.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _loss, new_params = train_step(
            graph,
            x,
            y,
            params,
            lr=0.01,
            optimizer_type="adamw",
            amp_enabled=True,
            amp_loss_scale=128.0,
            amp_dynamic_scale=True,
            amp_scale_growth=2.0,
            amp_scale_backoff=0.5,
            amp_scale_window=2,
            optimizer_state=native_state,
        )

    torch.manual_seed(22)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 3),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_w_linear_0"]))
        model[0].bias.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_b_linear_0"]))
        model[2].weight.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_w_linear_2"]))
        model[2].bias.copy_(torch.from_numpy(_make_linear_relu_linear_params()["_b_linear_2"]))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=128.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
    )
    tx = torch.from_numpy(x)
    ty = torch.tensor(y, dtype=torch.int64)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cpu"):
        loss = F.cross_entropy(model(tx), ty) * torch.tensor(float("inf"))
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    for key in params:
        np.testing.assert_array_equal(new_params[key], params[key])
    assert native_state["amp"]["loss_scale"] == scaler.get_scale() == 64.0
    assert native_state["amp"]["overflow_steps"] == 1
    assert native_state["amp"]["skipped_steps"] == 1

    for _ in range(8):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _loss, _ = train_step(
                graph,
                x,
                y,
                params,
                lr=0.01,
                optimizer_type="adamw",
                amp_enabled=True,
                amp_loss_scale=128.0,
                amp_dynamic_scale=True,
                amp_scale_growth=2.0,
                amp_scale_backoff=0.5,
                amp_scale_window=2,
                optimizer_state=native_state,
            )
    assert native_state["amp"]["loss_scale"] >= 1.0


def test_cuda_native_amp_training_remains_convergent_on_tiny_problem():
    from minicnn.cuda_native.training import train_step

    graph = _build_linear_relu_linear_graph()
    x = np.array(
        [
            [2.0, 1.5, 0.5, 0.0, 0.0],
            [1.5, 2.0, 0.0, 0.5, 0.0],
            [0.0, 0.5, 2.0, 1.5, 0.0],
            [0.0, 0.0, 1.5, 2.0, 0.5],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    fp32_params = _make_linear_relu_linear_params()
    amp_params = copy.deepcopy(fp32_params)
    fp32_first_loss, _ = train_step(graph, x, y, fp32_params, lr=0.05, optimizer_type="adamw", amp_enabled=False)
    amp_state: dict[str, object] = {}
    amp_first_loss, _ = train_step(
        graph,
        x,
        y,
        amp_params,
        lr=0.05,
        optimizer_type="adamw",
        amp_enabled=True,
        amp_loss_scale=128.0,
        amp_scale_window=4,
        optimizer_state=amp_state,
    )

    for _ in range(24):
        _fp32_loss, fp32_params = train_step(
            graph, x, y, fp32_params, lr=0.05, optimizer_type="adamw", amp_enabled=False
        )
        _amp_loss, amp_params = train_step(
            graph,
            x,
            y,
            amp_params,
            lr=0.05,
            optimizer_type="adamw",
            amp_enabled=True,
            amp_loss_scale=128.0,
            amp_scale_window=4,
            optimizer_state=amp_state,
        )

    fp32_last_loss, _ = train_step(graph, x, y, fp32_params, lr=0.05, optimizer_type="adamw", amp_enabled=False)
    amp_last_loss, _ = train_step(
        graph,
        x,
        y,
        amp_params,
        lr=0.05,
        optimizer_type="adamw",
        amp_enabled=True,
        amp_loss_scale=128.0,
        amp_scale_window=4,
        optimizer_state=amp_state,
    )

    assert fp32_last_loss < fp32_first_loss
    assert amp_last_loss < amp_first_loss
    assert abs(fp32_last_loss - amp_last_loss) <= 0.2
