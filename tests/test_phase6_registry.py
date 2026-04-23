from __future__ import annotations

import pytest


def test_get_supported_components_returns_dict():
    from minicnn.flex.registry import describe_registries
    result = describe_registries()
    assert isinstance(result, dict)


def test_optimizers_registered():
    from minicnn.flex.registry import describe_registries
    result = describe_registries()
    assert 'optimizers' in result
    opts = result['optimizers']
    for name in ('SGD', 'Adam', 'AdamW'):
        assert name in opts, f"optimizer '{name}' not registered"


def test_schedulers_registered():
    from minicnn.flex.registry import describe_registries
    result = describe_registries()
    assert 'schedulers' in result
    scheds = result['schedulers']
    for name in ('StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'):
        assert name in scheds, f"scheduler '{name}' not registered"


def test_activations_registered():
    from minicnn.flex.registry import describe_registries
    result = describe_registries()
    assert 'activations' in result
    acts = result['activations']
    for name in ('ReLU', 'LeakyReLU', 'SiLU', 'Tanh', 'Sigmoid'):
        assert name in acts, f"activation '{name}' not registered"


def test_layers_registered():
    from minicnn.flex.registry import describe_registries
    result = describe_registries()
    assert 'layers' in result
    layers = result['layers']
    for name in ('Conv2d', 'Linear', 'BatchNorm2d', 'ResidualBlock'):
        assert name in layers


def test_registry_factories_are_real_callables():
    from minicnn.flex.registry import REGISTRY, describe_registries

    describe_registries()

    assert callable(REGISTRY.get('optimizers', 'AdamW'))
    assert callable(REGISTRY.get('activations', 'Sigmoid'))


def test_registry_rejects_duplicate_registration_by_default():
    from minicnn.flex.registry import Registry

    registry = Registry()
    registry.register('layers', 'Linear', object)

    with pytest.raises(ValueError, match='replace=True'):
        registry.register('layers', 'Linear', dict)


def test_registry_allows_explicit_replace():
    from minicnn.flex.registry import Registry

    registry = Registry()
    registry.register('layers', 'Linear', object)
    registry.register('layers', 'Linear', dict, replace=True)

    assert registry.get('layers', 'Linear') is dict
