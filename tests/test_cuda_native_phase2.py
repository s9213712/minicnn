"""Phase 2 tests: planner, pool shape inference, pool kernels, end-to-end."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pool shape inference
# ---------------------------------------------------------------------------

def test_maxpool_shape_default_stride():
    from minicnn.cuda_native.shapes import infer_pool2d
    # default stride == kernel_size
    assert infer_pool2d((1, 16, 8, 8), kernel_size=2) == (1, 16, 4, 4)


def test_maxpool_shape_explicit_stride():
    from minicnn.cuda_native.shapes import infer_pool2d
    assert infer_pool2d((1, 16, 8, 8), kernel_size=2, stride=2) == (1, 16, 4, 4)


def test_avgpool_shape():
    from minicnn.cuda_native.shapes import infer_pool2d
    assert infer_pool2d((2, 32, 16, 16), kernel_size=4, stride=4) == (2, 32, 4, 4)


def test_pool_invalid_shape_raises():
    from minicnn.cuda_native.shapes import infer_pool2d
    with pytest.raises(ValueError, match='Invalid Pool2d output shape'):
        infer_pool2d((1, 16, 2, 2), kernel_size=5, stride=1)


def test_pool_requires_4d_input():
    from minicnn.cuda_native.shapes import infer_pool2d
    with pytest.raises(ValueError, match='4-D input'):
        infer_pool2d((16, 8, 8), kernel_size=2)


def test_infer_shape_dispatch_maxpool():
    from minicnn.cuda_native.shapes import infer_shape
    out = infer_shape('MaxPool2d', (1, 16, 8, 8), {'kernel_size': 2, 'stride': 2})
    assert out == (1, 16, 4, 4)


def test_infer_shape_dispatch_avgpool():
    from minicnn.cuda_native.shapes import infer_shape
    out = infer_shape('AvgPool2d', (2, 8, 4, 4), {'kernel_size': 2})
    assert out == (2, 8, 2, 2)


# ---------------------------------------------------------------------------
# Pool validators
# ---------------------------------------------------------------------------

def test_validator_accepts_maxpool():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([{'type': 'MaxPool2d', 'kernel_size': 2}])
    assert errors == []


def test_validator_accepts_avgpool():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([{'type': 'AvgPool2d', 'kernel_size': 2}])
    assert errors == []


# ---------------------------------------------------------------------------
# Pool kernels (reference numpy)
# ---------------------------------------------------------------------------

def test_maxpool2d_kernel():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    layers = [{'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2}]
    g = build_graph(layers, (1, 1, 4, 4))
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    out = ForwardExecutor().run_inference(g, x)
    assert out.shape == (1, 1, 2, 2)
    # top-left 2x2 patch max = 5
    assert out[0, 0, 0, 0] == 5.0


def test_avgpool2d_kernel():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    layers = [{'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 2}]
    g = build_graph(layers, (1, 1, 4, 4))
    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    out = ForwardExecutor().run_inference(g, x)
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(out, np.ones((1, 1, 2, 2), dtype=np.float32))


def test_avgpool2d_forward():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    layers = [{'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 2}]
    g = build_graph(layers, (1, 1, 4, 4))
    x = np.array(
        [[[[1.0, 3.0, 2.0, 4.0],
           [5.0, 7.0, 6.0, 8.0],
           [0.0, 2.0, 1.0, 3.0],
           [4.0, 6.0, 5.0, 7.0]]]],
        dtype=np.float32,
    )
    out = ForwardExecutor().run_inference(g, x)
    expected = np.array([[[[4.0, 5.0], [3.0, 4.0]]]], dtype=np.float32)
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_maxpool2d_kernel_supports_padding():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    layers = [{'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2, 'padding': 1}]
    g = build_graph(layers, (1, 1, 2, 2))
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)

    out = ForwardExecutor().run_inference(g, x)

    expected = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_avgpool2d_kernel_rejects_non_positive_stride():
    from minicnn.cuda_native.graph import build_graph

    with pytest.raises(ValueError, match='stride must be positive'):
        build_graph([{'type': 'AvgPool2d', 'kernel_size': 2, 'stride': 0}], (1, 1, 4, 4))


def test_maxpool_avgpool_different_outputs():
    """MaxPool and AvgPool must produce distinct results on non-uniform input."""
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    g_max = build_graph([{'type': 'MaxPool2d', 'kernel_size': 2}], (1, 1, 2, 2))
    g_avg = build_graph([{'type': 'AvgPool2d', 'kernel_size': 2}], (1, 1, 2, 2))
    out_max = ForwardExecutor().run_inference(g_max, x)
    out_avg = ForwardExecutor().run_inference(g_avg, x)
    assert out_max[0, 0, 0, 0] == 4.0
    assert out_avg[0, 0, 0, 0] == 2.5


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def test_planner_produces_plan_for_linear_chain():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.planner import make_naive_plan
    g = build_graph([
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ], (4, 3, 8, 8))
    plan = make_naive_plan(g)
    assert len(plan.steps) == 2
    assert plan.steps[0].op_type == 'Flatten'
    assert plan.steps[1].op_type == 'Linear'


def test_planner_step_order_matches_graph():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.planner import make_naive_plan
    layers = [
        {'type': 'Conv2d', 'out_channels': 8, 'kernel_size': 3},
        {'type': 'ReLU'},
        {'type': 'MaxPool2d', 'kernel_size': 2},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 5},
    ]
    g = build_graph(layers, (1, 3, 8, 8))
    plan = make_naive_plan(g)
    assert [s.op_type for s in plan.steps] == ['Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear']


def test_planner_buffer_count():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.planner import make_naive_plan
    g = build_graph([
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 3},
    ], (2, 4))
    plan = make_naive_plan(g)
    # 1 input buffer + 1 per node output = 3 total
    assert plan.buffer_plan.num_buffers == 3


def test_planner_reports_total_bytes():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.planner import make_naive_plan
    g = build_graph([{'type': 'Linear', 'out_features': 10}], (1, 4))
    plan = make_naive_plan(g)
    # input: (1,4)*4=16 bytes; output: (1,10)*4=40 bytes; total=56
    assert plan.buffer_plan.total_nbytes == 56


def test_planner_summary_has_required_keys():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.planner import make_naive_plan
    g = build_graph([{'type': 'ReLU'}], (1, 4))
    plan = make_naive_plan(g)
    s = plan.summary()
    assert 'steps' in s
    assert 'buffer_plan' in s
    assert 'total_bytes' in s['buffer_plan']
    assert 'num_buffers' in s['buffer_plan']


def test_planner_empty_graph():
    from minicnn.cuda_native.graph import NativeGraph
    from minicnn.cuda_native.planner import make_naive_plan
    g = NativeGraph()
    plan = make_naive_plan(g)
    assert len(plan.steps) == 0
    assert plan.buffer_plan.total_nbytes == 0


# ---------------------------------------------------------------------------
# End-to-end with pool
# ---------------------------------------------------------------------------

def test_conv_pool_flatten_linear_forward():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    layers = [
        {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3, 'padding': 1},
        {'type': 'ReLU'},
        {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]
    # (1,3,8,8) -> conv(pad=1) -> (1,4,8,8) -> pool(2,2) -> (1,4,4,4) -> flatten -> (1,64) -> linear -> (1,10)
    g = build_graph(layers, (1, 3, 8, 8))
    assert g.output_spec.shape == (1, 10)
    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    w_conv = np.random.randn(4, 3, 3, 3).astype(np.float32)
    w_fc = np.random.randn(10, 64).astype(np.float32)
    params = {'_w_conv2d_0': w_conv, '_w_linear_4': w_fc}
    out = ForwardExecutor().run_inference(g, x, params=params)
    assert out.shape == (1, 10)


def test_capabilities_include_pool_ops():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    caps = get_cuda_native_capabilities()
    assert 'MaxPool2d' in caps['supported_ops']
    assert 'AvgPool2d' in caps['supported_ops']
