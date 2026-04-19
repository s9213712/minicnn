import numpy as np
import pytest

from minicnn.compiler import optimize, trace_model_config
from minicnn.compiler.passes import detect_conv_bn_relu
from minicnn.core.fused_ops import fused_conv_bn_relu
from minicnn.models import build_model_from_config
from minicnn.runtime.memory import MemoryPool
from minicnn.runtime.profiler import Profiler


def test_minicnn_builder_and_compiler_pattern_detection():
    cfg = {
        'input_shape': [1, 4, 4],
        'layers': [
            {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3, 'padding': 1},
            {'type': 'BatchNorm2d'},
            {'type': 'ReLU'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'out_features': 3},
        ],
    }

    model, final_shape = build_model_from_config(cfg)
    graph = optimize(trace_model_config(cfg))

    assert final_shape == (3,)
    assert model.inferred_shapes[-1] == (3,)
    assert detect_conv_bn_relu(graph) == [('conv2d_0', 'batchnorm2d_1', 'relu_2')]
    assert graph.summary()['num_nodes'] == 5


def test_fused_conv_bn_relu_matches_unfused_numpy_reference():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 1, 5, 5)).astype(np.float32)
    w = rng.normal(size=(3, 1, 3, 3)).astype(np.float32)
    bias = rng.normal(size=(3,)).astype(np.float32)
    mean = rng.normal(size=(3,)).astype(np.float32)
    var = np.abs(rng.normal(size=(3,)).astype(np.float32)) + 0.5
    gamma = rng.normal(size=(3,)).astype(np.float32)
    beta = rng.normal(size=(3,)).astype(np.float32)

    fused, meta = fused_conv_bn_relu(x, w, bias, mean, var, gamma, beta, padding=1)

    conv = np.zeros_like(fused)
    x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    for i in range(5):
        for j in range(5):
            region = x_pad[:, :, i:i + 3, j:j + 3]
            conv[:, :, i, j] = np.tensordot(region, w, axes=([1, 2, 3], [1, 2, 3])) + bias
    expected = np.maximum(conv * (gamma / np.sqrt(var + 1e-5)).reshape(1, -1, 1, 1) + (beta - mean * gamma / np.sqrt(var + 1e-5)).reshape(1, -1, 1, 1), 0.0)

    assert meta['fused'] is True
    assert np.allclose(fused, expected, atol=1e-4, rtol=1e-4)


def test_builder_and_fused_ops_raise_clear_validation_errors():
    with pytest.raises(ValueError, match='add Flatten first'):
        build_model_from_config({'input_shape': [1, 4, 4], 'layers': [{'type': 'Linear', 'out_features': 2}]})

    x = np.zeros((1, 2, 5, 5), dtype=np.float32)
    w = np.zeros((3, 1, 3, 3), dtype=np.float32)
    c = np.zeros((3,), dtype=np.float32)
    with pytest.raises(ValueError, match='Input channels'):
        fused_conv_bn_relu(x, w, None, c, c + 1.0, c, c)


def test_memory_pool_and_profiler():
    pool = MemoryPool()
    arr = pool.alloc((2, 2))
    pool.release(arr)
    assert pool.alloc((2, 2)) is arr

    profiler = Profiler()
    with profiler.record('unit'):
        pass
    assert profiler.summary()['events'][0]['name'] == 'unit'
