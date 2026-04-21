from __future__ import annotations

import math
import numpy as np
import pytest
from minicnn.models.initialization import (
    kaiming_uniform,
    kaiming_normal,
    xavier_uniform,
    xavier_normal,
    normal_init,
    zeros_init,
    get_initializer,
)


SHAPES_2D = [(64, 32), (128, 64)]
SHAPES_4D = [(16, 8, 3, 3), (32, 16, 3, 3)]


@pytest.mark.parametrize('shape', SHAPES_2D + SHAPES_4D)
def test_kaiming_uniform_shape(shape):
    arr = kaiming_uniform(shape)
    assert arr.shape == tuple(shape)
    assert arr.dtype == np.float32


@pytest.mark.parametrize('shape', SHAPES_2D + SHAPES_4D)
def test_kaiming_normal_shape(shape):
    arr = kaiming_normal(shape)
    assert arr.shape == tuple(shape)
    assert arr.dtype == np.float32


@pytest.mark.parametrize('shape', SHAPES_2D + SHAPES_4D)
def test_xavier_uniform_shape(shape):
    arr = xavier_uniform(shape)
    assert arr.shape == tuple(shape)
    assert arr.dtype == np.float32


@pytest.mark.parametrize('shape', SHAPES_2D + SHAPES_4D)
def test_xavier_normal_shape(shape):
    arr = xavier_normal(shape)
    assert arr.shape == tuple(shape)
    assert arr.dtype == np.float32


def test_normal_init_shape():
    arr = normal_init((10, 5))
    assert arr.shape == (10, 5)
    assert arr.dtype == np.float32


def test_zeros_init_shape():
    arr = zeros_init((8, 4))
    assert arr.shape == (8, 4)
    assert arr.dtype == np.float32
    assert np.all(arr == 0.0)


def test_kaiming_normal_std_sanity():
    rng = np.random.default_rng(42)
    shape = (1000, 500)
    arr = kaiming_normal(shape, a=0, mode='fan_in', rng=rng)
    fan_in = 500
    expected_std = math.sqrt(2.0 / fan_in)
    assert abs(arr.std() - expected_std) < 0.01 * expected_std + 0.001


def test_xavier_normal_std_sanity():
    rng = np.random.default_rng(42)
    shape = (500, 500)
    arr = xavier_normal(shape, gain=1.0, rng=rng)
    fan_in, fan_out = 500, 500
    expected_std = math.sqrt(2.0 / (fan_in + fan_out))
    assert abs(arr.std() - expected_std) < 0.01 * expected_std + 0.001


def test_get_initializer_kaiming_uniform():
    fn = get_initializer('kaiming_uniform')
    assert fn is kaiming_uniform


def test_get_initializer_kaiming_normal():
    fn = get_initializer('kaiming_normal')
    assert fn is kaiming_normal


def test_get_initializer_xavier_uniform():
    fn = get_initializer('xavier_uniform')
    assert fn is xavier_uniform


def test_get_initializer_xavier_normal():
    fn = get_initializer('xavier_normal')
    assert fn is xavier_normal


def test_get_initializer_normal():
    fn = get_initializer('normal')
    assert fn is normal_init


def test_get_initializer_zeros():
    fn = get_initializer('zeros')
    assert fn is zeros_init


def test_get_initializer_he_alias():
    fn = get_initializer('he')
    assert fn is kaiming_uniform


def test_get_initializer_unknown():
    with pytest.raises(KeyError):
        get_initializer('bogus')
