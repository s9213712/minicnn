from minicnn.flex.registry import describe_registries


def test_registry_has_common_components():
    summary = describe_registries()
    assert 'layers' in summary
    assert 'Conv2d' in summary['layers']
    assert 'CrossEntropyLoss' in summary['losses']
    assert 'AdamW' in summary['optimizers']
