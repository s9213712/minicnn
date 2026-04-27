import numpy as np

from minicnn.nn import BatchNorm2d, Linear, ReLU, Sequential


def test_state_dict_returns_snapshot_copies():
    model = Sequential(Linear(2, 2, rng=np.random.default_rng(0)))

    state = model.state_dict()
    weight_key = '0.weight'
    original = model.get_parameter(weight_key).data.copy()

    assert state[weight_key] is not model.get_parameter(weight_key).data
    state[weight_key][0, 0] += 123.0

    assert np.allclose(model.get_parameter(weight_key).data, original)


def test_module_api_helpers_cover_nested_sequential_models():
    inner = Sequential(ReLU(), Linear(2, 1, rng=np.random.default_rng(1)))
    model = Sequential(Linear(2, 2, rng=np.random.default_rng(0)), inner)

    seen = []
    model.apply(lambda module: seen.append(module.__class__.__name__))

    assert len(model) == 2
    assert model[0] in model
    assert [name for name, _ in model.named_modules()] == ['', '0', '1', '1.0', '1.1']
    assert seen.count('Linear') == 2
    assert seen.count('ReLU') == 1
    assert model.get_parameter('1.1.weight') is inner[1].weight


def test_state_dict_includes_batchnorm_buffers_and_load_restores_them():
    model = Sequential(BatchNorm2d(2))
    bn = model[0]
    bn.running_mean[...] = np.array([1.0, -2.0], dtype=np.float32)
    bn.running_var[...] = np.array([3.0, 4.0], dtype=np.float32)

    state = model.state_dict()

    assert np.array_equal(state['0.running_mean'], [1.0, -2.0])
    assert np.array_equal(state['0.running_var'], [3.0, 4.0])

    bn.running_mean[...] = 0.0
    bn.running_var[...] = 1.0
    model.load_state_dict(state)

    assert np.array_equal(model.get_buffer('0.running_mean'), [1.0, -2.0])
    assert np.array_equal(model.get_buffer('0.running_var'), [3.0, 4.0])
