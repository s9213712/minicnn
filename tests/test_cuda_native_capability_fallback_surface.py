from __future__ import annotations


def test_gpu_native_training_capability_surface_declares_numpy_fallback():
    from minicnn.cuda_native.capabilities import GPU_NATIVE_TRAINING_SUBSETS

    for subset in GPU_NATIVE_TRAINING_SUBSETS:
        assert subset['execution_mode'] == 'gpu_native'
        assert subset['fallback_execution_mode'] == 'reference_numpy'
        assert subset['fallback_policy'] == {
            'fallback_execution_mode': 'reference_numpy',
            'fallback_available': True,
            'fallback_active_when': 'gpu_native_lowering_not_ready',
        }


def test_gpu_native_training_capability_surface_declares_helper_constraints():
    from minicnn.cuda_native.capabilities import GPU_NATIVE_TRAINING_SUBSETS

    by_name = {subset['name']: subset for subset in GPU_NATIVE_TRAINING_SUBSETS}

    assert by_name['flatten_linear']['constraints'] == []
    assert by_name['conv_linear']['constraints'] == [
        'Conv-family training helpers require bias=false, stride=1, padding=0, dilation=1',
    ]
    assert by_name['depthwise_layernorm2d_pointwise_gelu_pointwise_linear']['constraints'] == [
        'Conv-family training helpers require bias=false, stride=1, padding=0, dilation=1',
        'PointwiseConv2d training helpers require kernel_size=1',
    ]
    assert by_name['adaptive_avgpool_linear']['constraints'] == [
        'AdaptiveAvgPool2d training helper requires output_size=1',
    ]
