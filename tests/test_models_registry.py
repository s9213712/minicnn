from __future__ import annotations

import pytest


def test_list_model_components_is_stably_sorted():
    from minicnn.models import list_model_components

    names = list_model_components()

    assert names == sorted(names)
    assert 'Conv2d' in names
    assert 'Linear' in names


def test_get_model_component_error_lists_sorted_choices():
    from minicnn.models.registry import get_model_component, list_model_components

    with pytest.raises(KeyError) as excinfo:
        get_model_component('NopeLayer')

    message = str(excinfo.value)
    expected_choices = ', '.join(list_model_components())
    assert expected_choices in message
