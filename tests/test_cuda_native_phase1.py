"""Phase 1 tests: IR nodes, shape inference, validator attrs, graph build, executor."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

def test_tensor_spec_numel():
    from minicnn.cuda_native.nodes import TensorSpec
    t = TensorSpec(name='x', shape=(2, 3, 4, 4))
    assert t.numel() == 2 * 3 * 4 * 4


def test_tensor_spec_nbytes():
    from minicnn.cuda_native.nodes import TensorSpec
    t = TensorSpec(name='x', shape=(1, 3, 32, 32), dtype='float32')
    assert t.nbytes() == 1 * 3 * 32 * 32 * 4


def test_node_repr():
    from minicnn.cuda_native.nodes import Node, TensorSpec
    n = Node(
        name='relu_0', op_type='ReLU',
        input_specs=[TensorSpec('t0', (1, 16, 8, 8))],
        output_specs=[TensorSpec('t1', (1, 16, 8, 8))],
    )
    r = repr(n)
    assert 'ReLU' in r
    assert 'relu_0' in r


# ---------------------------------------------------------------------------
# shape inference
# ---------------------------------------------------------------------------

def test_conv2d_shape_basic():
    from minicnn.cuda_native.shapes import infer_conv2d
    assert infer_conv2d((1, 3, 32, 32), 16, kernel_size=3, stride=1, padding=0) == (1, 16, 30, 30)


def test_conv2d_shape_with_padding():
    from minicnn.cuda_native.shapes import infer_conv2d
    assert infer_conv2d((1, 3, 32, 32), 16, kernel_size=3, stride=1, padding=1) == (1, 16, 32, 32)


def test_conv2d_shape_invalid_raises():
    from minicnn.cuda_native.shapes import infer_conv2d
    with pytest.raises(ValueError, match='Invalid Conv2d output shape'):
        infer_conv2d((1, 3, 2, 2), 16, kernel_size=5, stride=1, padding=0)


def test_flatten_shape():
    from minicnn.cuda_native.shapes import infer_flatten
    assert infer_flatten((2, 16, 4, 4)) == (2, 256)


def test_flatten_1d_raises():
    from minicnn.cuda_native.shapes import infer_flatten
    with pytest.raises(ValueError):
        infer_flatten((4,))


def test_linear_shape():
    from minicnn.cuda_native.shapes import infer_linear
    assert infer_linear((4, 256), 10) == (4, 10)


def test_linear_requires_2d():
    from minicnn.cuda_native.shapes import infer_linear
    with pytest.raises(ValueError, match='Flatten'):
        infer_linear((4, 16, 8, 8), 10)


def test_activation_passthrough():
    from minicnn.cuda_native.shapes import infer_activation
    s = (2, 16, 8, 8)
    assert infer_activation(s) == s


def test_infer_shape_dispatch_conv2d():
    from minicnn.cuda_native.shapes import infer_shape
    out = infer_shape('Conv2d', (1, 3, 32, 32), {'out_channels': 16, 'kernel_size': 3})
    assert out == (1, 16, 30, 30)


def test_infer_shape_missing_out_channels():
    from minicnn.cuda_native.shapes import infer_shape
    with pytest.raises(ValueError, match='out_channels'):
        infer_shape('Conv2d', (1, 3, 32, 32), {}, node_name='conv_0')


def test_infer_shape_missing_out_features():
    from minicnn.cuda_native.shapes import infer_shape
    with pytest.raises(ValueError, match='out_features'):
        infer_shape('Linear', (4, 256), {}, node_name='fc_0')


def test_infer_shape_unknown_op():
    from minicnn.cuda_native.shapes import infer_shape
    with pytest.raises(ValueError, match='No shape inference rule'):
        infer_shape('GroupNorm', (1, 16, 8, 8), {})


def test_infer_shape_batchnorm2d_passthrough():
    from minicnn.cuda_native.shapes import infer_shape
    out = infer_shape('BatchNorm2d', (1, 16, 8, 8), {})
    assert out == (1, 16, 8, 8)


# ---------------------------------------------------------------------------
# validators — attr checks
# ---------------------------------------------------------------------------

def test_validator_rejects_conv2d_missing_out_channels():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([{'type': 'Conv2d', 'kernel_size': 3}])
    assert any('out_channels' in e for e in errors)


def test_validator_rejects_linear_missing_out_features():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([{'type': 'Linear'}])
    assert any('out_features' in e for e in errors)


def test_validator_accepts_full_conv2d():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([{'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3}])
    assert errors == []


# ---------------------------------------------------------------------------
# graph build
# ---------------------------------------------------------------------------

def test_build_graph_sequential():
    from minicnn.cuda_native.graph import build_graph
    layers = [
        {'type': 'Conv2d', 'out_channels': 16, 'kernel_size': 3},
        {'type': 'ReLU'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]
    g = build_graph(layers, input_shape=(1, 3, 32, 32))
    assert len(g.nodes) == 4
    assert g.input_spec.shape == (1, 3, 32, 32)
    assert g.output_spec.shape == (1, 10)


def test_build_graph_shape_propagation():
    from minicnn.cuda_native.graph import build_graph
    layers = [
        {'type': 'Conv2d', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
        {'type': 'LeakyReLU', 'negative_slope': 0.1},
        {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
        {'type': 'ReLU'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]
    g = build_graph(layers, input_shape=(2, 3, 8, 8))
    assert g.nodes[0].output_specs[0].shape == (2, 32, 8, 8)
    assert g.nodes[2].output_specs[0].shape == (2, 64, 8, 8)
    assert g.output_spec.shape == (2, 10)


def test_build_graph_rejects_unsupported_op():
    from minicnn.cuda_native.graph import build_graph
    with pytest.raises(ValueError, match='cuda_native validation failed'):
        build_graph([{'type': 'GroupNorm'}], (1, 3, 32, 32))


def test_build_graph_topological_order():
    from minicnn.cuda_native.graph import build_graph
    layers = [
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 5},
    ]
    g = build_graph(layers, (4, 16))
    order = g.topological_order()
    assert [n.op_type for n in order] == ['Flatten', 'Linear']


def test_build_graph_summary():
    from minicnn.cuda_native.graph import build_graph
    g = build_graph(
        [{'type': 'Flatten'}, {'type': 'Linear', 'out_features': 3}],
        (2, 8),
    )
    s = g.summary()
    assert s['num_nodes'] == 2
    assert s['ops'] == ['Flatten', 'Linear']


# ---------------------------------------------------------------------------
# kernel registry
# ---------------------------------------------------------------------------

def test_kernel_registry_get_missing_raises():
    from minicnn.cuda_native.kernels import KernelRegistry
    reg = KernelRegistry()
    with pytest.raises(KeyError, match='No cuda_native kernel'):
        reg.get('BatchNorm2d')


def test_default_registry_has_minimum_ops():
    from minicnn.cuda_native.kernels import make_default_registry
    reg = make_default_registry()
    for op in (
        'BatchNorm2d',
        'Conv2d',
        'ReLU',
        'LeakyReLU',
        'Sigmoid',
        'Tanh',
        'SiLU',
        'Flatten',
        'Linear',
    ):
        assert reg.has(op), f'missing kernel for {op}'


def test_default_registry_registered_ops_are_stable():
    from minicnn.cuda_native.kernels import DEFAULT_KERNEL_SPECS, make_default_registry

    reg = make_default_registry()

    assert reg.registered_ops() == sorted(spec.op_name for spec in DEFAULT_KERNEL_SPECS)


def test_default_registry_keeps_activation_kernel_surface_stable():
    from minicnn.cuda_native.kernels import DEFAULT_KERNEL_SPECS

    activation_ops = [
        spec.op_name
        for spec in DEFAULT_KERNEL_SPECS
        if spec.category == 'activation'
    ]

    assert activation_ops == ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'SiLU']


def test_default_registry_exposes_kernel_metadata():
    from minicnn.cuda_native.kernels import make_default_registry

    reg = make_default_registry()

    assert reg.describe('Conv2d') == {
        'op_name': 'Conv2d',
        'category': 'convolution',
    }
    assert [(spec.op_name, spec.category) for spec in reg.registered_specs()] == [
        ('AvgPool2d', 'pool'),
        ('BatchNorm2d', 'normalization'),
        ('Conv2d', 'convolution'),
        ('Flatten', 'shape'),
        ('LeakyReLU', 'activation'),
        ('Linear', 'linear'),
        ('MaxPool2d', 'pool'),
        ('ReLU', 'activation'),
        ('SiLU', 'activation'),
        ('Sigmoid', 'activation'),
        ('Tanh', 'activation'),
    ]


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------

def test_executor_relu_forward():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    g = build_graph([{'type': 'ReLU'}], (2, 4))
    x = np.array([[-1.0, 2.0, -3.0, 4.0], [0.5, -0.5, 1.0, -1.0]], dtype=np.float32)
    ex = ForwardExecutor()
    out = ex.run_inference(g, x)
    expected = np.maximum(x, 0)
    np.testing.assert_array_equal(out, expected)


def test_executor_leaky_relu_forward():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    g = build_graph([{'type': 'LeakyReLU', 'negative_slope': 0.1}], (1, 4))
    x = np.array([[-2.0, 1.0, -0.5, 3.0]], dtype=np.float32)
    out = ForwardExecutor().run_inference(g, x)
    expected = np.where(x >= 0, x, 0.1 * x).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


@pytest.mark.parametrize(
    ('op_type', 'expected'),
    [
        ('Sigmoid', np.array([[0.26894143, 0.5, 0.7310586]], dtype=np.float32)),
        ('Tanh', np.tanh(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)).astype(np.float32)),
        ('SiLU', (np.array([[-1.0, 0.0, 1.0]], dtype=np.float32) / (1.0 + np.exp(-np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)))).astype(np.float32)),
    ],
)
def test_executor_extra_activations_forward(op_type, expected):
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    g = build_graph([{'type': op_type}], (1, 3))
    x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    out = ForwardExecutor().run_inference(g, x)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_executor_flatten_forward():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    g = build_graph([{'type': 'Flatten'}], (2, 3, 4, 4))
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    out = ForwardExecutor().run_inference(g, x)
    assert out.shape == (2, 48)
    np.testing.assert_array_equal(out, x.reshape(2, -1))


def test_executor_linear_forward():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    g = build_graph([{'type': 'Linear', 'out_features': 3}], (2, 4))
    x = np.ones((2, 4), dtype=np.float32)
    w = np.eye(3, 4, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    params = {'_w_linear_0': w, '_b_linear_0': b}
    out = ForwardExecutor().run_inference(g, x, params=params)
    assert out.shape == (2, 3)


def test_executor_conv_relu_flatten_linear():
    """End-to-end forward pass through a minimal CNN stack."""
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    layers = [
        {'type': 'Conv2d', 'out_channels': 4, 'kernel_size': 3},
        {'type': 'ReLU'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 5},
    ]
    g = build_graph(layers, input_shape=(1, 1, 5, 5))
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)
    w_conv = np.random.randn(4, 1, 3, 3).astype(np.float32)
    # After conv(1,1,5,5, k=3) -> (1,4,3,3) -> flatten -> (1,36) -> linear(36->5)
    w_fc = np.random.randn(5, 36).astype(np.float32)
    b_fc = np.zeros(5, dtype=np.float32)
    params = {
        '_w_conv2d_0': w_conv,
        '_w_linear_3': w_fc,
        '_b_linear_3': b_fc,
    }
    out = ForwardExecutor().run_inference(g, x, params=params)
    assert out.shape == (1, 5)


def test_executor_conv2d_padding_preserves_spatial_shape():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    g = build_graph([{'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3, 'padding': 1}], input_shape=(1, 1, 4, 4))
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    w = np.ones((2, 1, 3, 3), dtype=np.float32)

    out = ForwardExecutor().run_inference(g, x, params={'_w_conv2d_0': w})

    assert out.shape == (1, 2, 4, 4)


def test_executor_conv2d_rejects_weight_channel_mismatch():
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor

    g = build_graph([{'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3}], input_shape=(1, 1, 5, 5))
    x = np.ones((1, 1, 5, 5), dtype=np.float32)
    bad_w = np.ones((2, 3, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match='weight expects 3 input channels, got 1'):
        ForwardExecutor().run_inference(g, x, params={'_w_conv2d_0': bad_w})


def test_executor_debug_mode_does_not_crash(capsys):
    from minicnn.cuda_native.graph import build_graph
    from minicnn.cuda_native.executor import ForwardExecutor
    g = build_graph([{'type': 'ReLU'}], (1, 4))
    x = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)
    ForwardExecutor(debug=True).run_inference(g, x)
    captured = capsys.readouterr()
    assert '[cuda_native]' in captured.out


def test_executor_missing_kernel_raises():
    from minicnn.cuda_native.graph import NativeGraph
    from minicnn.cuda_native.nodes import Node, TensorSpec
    from minicnn.cuda_native.kernels import KernelRegistry
    from minicnn.cuda_native.executor import ForwardExecutor
    node = Node(
        name='bn_0', op_type='BatchNorm2d',
        inputs=['t_0'], outputs=['t_1'],
        input_specs=[TensorSpec('t_0', (1, 16, 8, 8))],
        output_specs=[TensorSpec('t_1', (1, 16, 8, 8))],
    )
    g = NativeGraph(nodes=[node])
    g.input_spec = TensorSpec('input', (1, 16, 8, 8))
    g.output_spec = TensorSpec('output', (1, 16, 8, 8))
    ex = ForwardExecutor(registry=KernelRegistry())
    x = np.zeros((1, 16, 8, 8), dtype=np.float32)
    with pytest.raises(KeyError, match='No cuda_native kernel'):
        ex.run(g, {'input': x})


def test_node_trainable_state_defaults_to_empty_dict():
    from minicnn.cuda_native.nodes import Node
    node = Node(name='relu_0', op_type='ReLU')
    assert node.trainable_state == {}


def test_planner_buffer_type_enum_exposes_expected_members():
    from minicnn.cuda_native.planner import BufferType
    assert BufferType.ACTIVATION.value == 'activation'
    assert BufferType.PARAMETER.value == 'parameter'
    assert BufferType.STATISTIC.value == 'statistic'
    assert BufferType.GRADIENT.value == 'gradient'


# ---------------------------------------------------------------------------
# api integration
# ---------------------------------------------------------------------------

def test_api_build_cuda_native_graph():
    from minicnn.cuda_native.api import build_cuda_native_graph
    g = build_cuda_native_graph(
        {'layers': [
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 10},
        ]},
        input_shape=(4, 3, 8, 8),
    )
    assert g.output_spec.shape == (4, 10)
