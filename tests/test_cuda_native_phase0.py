"""Phase 0 smoke tests: import, capability surface, validator boundaries."""
from __future__ import annotations

import pytest


def test_module_imports():
    import minicnn.cuda_native  # noqa: F401


def test_capabilities_module_imports():
    from minicnn.cuda_native.capabilities import (
        CUDA_NATIVE_CAPABILITIES,
        get_cuda_native_capabilities,
    )
    assert isinstance(CUDA_NATIVE_CAPABILITIES, dict)
    caps = get_cuda_native_capabilities()
    assert caps is not CUDA_NATIVE_CAPABILITIES


def test_capability_required_fields():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    caps = get_cuda_native_capabilities()
    for key in ('experimental', 'sequential_only', 'forward_only', 'training',
                'backward', 'supported_ops'):
        assert key in caps, f'missing capability key: {key}'


def test_capability_experimental_is_true():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    assert get_cuda_native_capabilities()['experimental'] is True


def test_capability_training_is_false():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    assert get_cuda_native_capabilities()['training'] is False


def test_capability_supported_ops_not_empty():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    ops = get_cuda_native_capabilities()['supported_ops']
    assert len(ops) > 0
    assert 'BatchNorm2d' in ops
    assert 'Conv2d' in ops
    assert 'Linear' in ops


def test_validator_accepts_supported_ops():
    from minicnn.cuda_native.validators import validate_op_type
    for op in ('BatchNorm2d', 'Conv2d', 'ReLU', 'LeakyReLU', 'Flatten', 'Linear'):
        assert validate_op_type(op) == [], f'{op} should be accepted'


def test_validator_rejects_groupnorm():
    from minicnn.cuda_native.validators import validate_op_type
    errors = validate_op_type('GroupNorm', node_name='gn1')
    assert len(errors) == 1
    assert 'GroupNorm' in errors[0]
    assert 'gn1' in errors[0]


def test_validator_rejects_unknown_op():
    from minicnn.cuda_native.validators import validate_op_type
    errors = validate_op_type('MyCustomOp', node_name='node0')
    assert errors
    assert 'MyCustomOp' in errors[0]


def test_validate_layer_list_accepts_valid():
    from minicnn.cuda_native.validators import validate_layer_list
    layers = [
        {'type': 'Conv2d', 'out_channels': 32},
        {'type': 'ReLU'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]
    assert validate_layer_list(layers) == []


def test_validate_layer_list_rejects_unsupported():
    from minicnn.cuda_native.validators import validate_layer_list
    layers = [
        {'type': 'Conv2d', 'out_channels': 32},
        {'type': 'GroupNorm'},
        {'type': 'ReLU'},
    ]
    errors = validate_layer_list(layers)
    assert any('GroupNorm' in e for e in errors)


def test_validate_layer_list_missing_type():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([{'out_channels': 32}])
    assert errors


def test_api_validate_config_accepts_supported():
    from minicnn.cuda_native.api import validate_cuda_native_config
    cfg = {'model': {'layers': [
        {'type': 'Conv2d', 'out_channels': 16},
        {'type': 'ReLU'},
        {'type': 'Flatten'},
        {'type': 'Linear', 'out_features': 10},
    ]}}
    assert validate_cuda_native_config(cfg) == []


def test_api_validate_config_rejects_groupnorm():
    from minicnn.cuda_native.api import validate_cuda_native_config
    cfg = {'model': {'layers': [
        {'type': 'Conv2d', 'out_channels': 16},
        {'type': 'GroupNorm'},
    ]}}
    errors = validate_cuda_native_config(cfg)
    assert any('GroupNorm' in e for e in errors)


def test_api_build_graph_returns_graph():
    from minicnn.cuda_native.api import build_cuda_native_graph
    g = build_cuda_native_graph(
        {'layers': [{'type': 'Linear', 'out_features': 10}]},
        (1, 4),
    )
    assert g is not None


def test_api_build_graph_raises_validation_error_before_not_implemented():
    from minicnn.cuda_native.api import build_cuda_native_graph
    with pytest.raises(ValueError, match='cuda_native validation failed'):
        build_cuda_native_graph(
            {'layers': [{'type': 'GroupNorm'}]},
            (1, 3, 32, 32),
        )


def test_unified_bridge_imports():
    from minicnn.unified import cuda_native as bridge  # noqa: F401


def test_unified_bridge_check_config():
    from minicnn.unified.cuda_native import check_config
    errors = check_config({'model': {'layers': [{'type': 'Linear', 'out_features': 10}]}})
    assert errors == []


def test_unified_bridge_get_summary():
    from minicnn.unified.cuda_native import get_summary
    s = get_summary()
    assert 'experimental' in s
    assert s['experimental'] is True
