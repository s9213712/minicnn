from minicnn.flex.registry import describe_registries


def test_registry_has_common_components():
    summary = describe_registries()
    assert 'layers' in summary
    assert 'Conv2d' in summary['layers']
    assert 'CrossEntropyLoss' in summary['losses']
    assert 'AdamW' in summary['optimizers']


def test_registry_summary_is_stably_sorted():
    summary = describe_registries()

    assert list(summary) == sorted(summary)
    for names in summary.values():
        assert names == sorted(names)
