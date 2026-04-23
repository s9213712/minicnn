from __future__ import annotations


def test_gelu_is_available_in_registry():
    from minicnn.flex.registry import describe_registries

    summary = describe_registries()

    assert 'GELU' in summary['activations']
