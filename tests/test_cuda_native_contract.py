from __future__ import annotations

import warnings


def test_experimental_warning_is_suppressed_when_stdout_is_not_a_tty(monkeypatch):
    import minicnn.cuda_native.contract as contract

    class _FakeStdout:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(contract.sys, 'stdout', _FakeStdout())
    monkeypatch.delenv(contract.SUPPRESS_EXPERIMENTAL_WARN_ENV, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        contract.emit_experimental_warning('experimental path', stacklevel=1)

    assert caught == []


def test_experimental_warning_can_be_suppressed_by_env(monkeypatch):
    import minicnn.cuda_native.contract as contract

    class _FakeStdout:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(contract.sys, 'stdout', _FakeStdout())
    monkeypatch.setenv(contract.SUPPRESS_EXPERIMENTAL_WARN_ENV, '1')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        contract.emit_experimental_warning('experimental path', stacklevel=1)

    assert caught == []


def test_experimental_warning_emits_in_interactive_mode(monkeypatch):
    import minicnn.cuda_native.contract as contract

    class _FakeStdout:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(contract.sys, 'stdout', _FakeStdout())
    monkeypatch.delenv(contract.SUPPRESS_EXPERIMENTAL_WARN_ENV, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        contract.emit_experimental_warning('interactive experimental path', stacklevel=1)

    assert len(caught) == 1
    assert 'interactive experimental path' in str(caught[0].message)
