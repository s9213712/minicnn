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
    for key in (
        'schema_version',
        'backend',
        'status',
        'summary_status',
        'experimental',
        'sequential_only',
        'forward_only',
        'training',
        'backward',
        'supported_ops',
        'supported_op_categories',
        'kernel_registry_surface',
    ):
        assert key in caps, f'missing capability key: {key}'


def test_capability_surface_is_versioned_and_sorted():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities

    caps = get_cuda_native_capabilities()

    assert caps['schema_version'] == 1
    assert caps['backend'] == 'cuda_native'
    assert caps['status'] == 'ok'
    assert caps['summary_status'] == 'beta'
    assert caps['sequential_only'] is False
    assert caps['branching_graph'] is True
    assert caps['supported_datasets'] == sorted(caps['supported_datasets'])
    assert caps['supported_schedulers'] == sorted(caps['supported_schedulers'])


def test_capability_surface_exposes_kernel_categories():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities

    caps = get_cuda_native_capabilities()

    assert caps['supported_op_categories'] == [
        'activation',
        'composite',
        'convolution',
        'elementwise',
        'linear',
        'normalization',
        'pool',
        'regularization',
        'shape',
    ]
    assert {'op_name': 'Add', 'category': 'elementwise'} in caps['kernel_registry_surface']
    assert caps['kernel_registry_surface'][0] == {
        'op_name': 'AdaptiveAvgPool2d',
        'category': 'pool',
    }
    assert {'op_name': 'AvgPool2d', 'category': 'pool'} in caps['kernel_registry_surface']


def test_capability_experimental_is_false():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    assert get_cuda_native_capabilities()['experimental'] is False


def test_capability_training_is_beta_ready():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    caps = get_cuda_native_capabilities()
    assert caps['training'] is True
    assert caps['training_stable'] is True


def test_capability_backward_is_beta_ready():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    caps = get_cuda_native_capabilities()
    assert caps['backward'] is True
    assert caps['backward_stable'] is True


def test_capability_supported_ops_not_empty():
    from minicnn.cuda_native.capabilities import get_cuda_native_capabilities
    ops = get_cuda_native_capabilities()['supported_ops']
    assert len(ops) > 0
    assert 'BatchNorm2d' in ops
    assert 'Conv2d' in ops
    assert 'Linear' in ops


def test_validator_accepts_supported_ops():
    from minicnn.cuda_native.validators import validate_op_type
    for op in (
        'BatchNorm2d',
        'Conv2d',
        'ReLU',
        'LeakyReLU',
        'Flatten',
        'Linear',
        'Dropout',
        'Add',
        'Concat',
        'GroupNorm',
        'LayerNorm',
        'DropPath',
        'ResidualBlock',
        'ConvNeXtBlock',
    ):
        assert validate_op_type(op) == [], f'{op} should be accepted'


def test_validator_accepts_groupnorm():
    from minicnn.cuda_native.validators import validate_op_type
    assert validate_op_type('GroupNorm', node_name='gn1') == []


def test_validator_accepts_convnext_block():
    from minicnn.cuda_native.validators import validate_op_type
    assert validate_op_type('ConvNeXtBlock', node_name='cnx1') == []


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
        {'type': 'CustomNorm'},
        {'type': 'ReLU'},
    ]
    errors = validate_layer_list(layers)
    assert any('CustomNorm' in e for e in errors)


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
    ]}, 'optimizer': {'type': 'SGD', 'momentum': 0.0}, 'scheduler': {'enabled': False}}
    assert validate_cuda_native_config(cfg) == []


def test_api_validate_config_accepts_groupnorm():
    from minicnn.cuda_native.api import validate_cuda_native_config
    cfg = {'model': {'layers': [
        {'type': 'Conv2d', 'out_channels': 16},
        {'type': 'GroupNorm', 'num_groups': 4},
    ]}, 'optimizer': {'type': 'SGD', 'momentum': 0.0}, 'scheduler': {'enabled': False}}
    assert validate_cuda_native_config(cfg) == []


def test_api_validate_config_accepts_layernorm():
    from minicnn.cuda_native.api import validate_cuda_native_config
    cfg = {'model': {'layers': [
        {'type': 'Flatten'},
        {'type': 'LayerNorm', 'normalized_shape': 48},
        {'type': 'Linear', 'out_features': 10},
    ]}, 'dataset': {'type': 'random', 'input_shape': [3, 4, 4], 'num_classes': 10, 'num_samples': 4, 'val_samples': 2}, 'optimizer': {'type': 'SGD', 'momentum': 0.0}, 'scheduler': {'enabled': False}}
    assert validate_cuda_native_config(cfg) == []


def test_api_validate_config_accepts_convnext_block():
    from minicnn.cuda_native.api import validate_cuda_native_config
    cfg = {'model': {'layers': [
        {'type': 'Conv2d', 'out_channels': 16},
        {'type': 'ConvNeXtBlock'},
    ]}, 'optimizer': {'type': 'SGD', 'momentum': 0.0}, 'scheduler': {'enabled': False}}
    assert validate_cuda_native_config(cfg) == []


def test_validate_layer_list_collects_missing_type_and_later_attr_error():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([
        {'out_channels': 16},
        {'type': 'Linear'},
    ])
    assert any('missing "type"' in e for e in errors)
    assert any('missing required attr "out_features"' in e for e in errors)


def test_validate_layer_list_collects_unsupported_and_later_conv_attrs():
    from minicnn.cuda_native.validators import validate_layer_list
    errors = validate_layer_list([
        {'type': 'CustomNorm'},
        {'type': 'Conv2d', 'kernel_size': 'bad'},
    ])
    assert any('CustomNorm' in e for e in errors)
    assert any('kernel_size' in e for e in errors)


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
            {'layers': [{'type': 'LayerNorm'}]},
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
    assert s['experimental'] is False
