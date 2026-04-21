from __future__ import annotations

import pytest


def test_get_supported_components_returns_dict():
    from minicnn.framework.components import get_supported_components
    result = get_supported_components()
    assert isinstance(result, dict)


def test_optimizers_registered():
    from minicnn.framework.components import get_supported_components
    result = get_supported_components()
    assert 'optimizer' in result
    opts = result['optimizer']
    for name in ('sgd', 'adam', 'adamw', 'rmsprop'):
        assert name in opts, f"optimizer '{name}' not registered"


def test_schedulers_registered():
    from minicnn.framework.components import get_supported_components
    result = get_supported_components()
    assert 'scheduler' in result
    scheds = result['scheduler']
    for name in ('plateau', 'step', 'cosine'):
        assert name in scheds, f"scheduler '{name}' not registered"


def test_activations_registered():
    from minicnn.framework.components import get_supported_components
    result = get_supported_components()
    assert 'activation' in result
    acts = result['activation']
    for name in ('relu', 'leaky_relu', 'silu', 'tanh', 'sigmoid'):
        assert name in acts, f"activation '{name}' not registered"


def test_backends_registered():
    from minicnn.framework.components import get_supported_components
    result = get_supported_components()
    assert 'backend' in result
    backends = result['backend']
    assert 'cuda' in backends
    assert 'torch' in backends


def test_registry_class_references():
    from minicnn.framework.components import register_builtin_components
    from minicnn.framework.registry import GLOBAL_REGISTRY
    from minicnn.optim.adamw import AdamW
    from minicnn.optim.rmsprop import RMSprop
    register_builtin_components()
    adamw_spec = GLOBAL_REGISTRY.get('optimizer', 'adamw')
    assert adamw_spec.factory is AdamW
    rmsprop_spec = GLOBAL_REGISTRY.get('optimizer', 'rmsprop')
    assert rmsprop_spec.factory is RMSprop
