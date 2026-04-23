from __future__ import annotations


def test_train_flex_reports_structured_dataset_split_error_for_mnist(capsys):
    import pytest

    from minicnn.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main([
            'train-flex',
            '--config',
            'templates/mnist/lenet_like.yaml',
            'dataset.num_samples=55001',
            'dataset.val_samples=5000',
            'train.device=cpu',
        ])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert captured.out == ''
    assert '[ERROR] Dataset split invalid' in captured.err
    assert 'Cause: num_samples + val_samples exceeds available training samples for mnist' in captured.err
    assert 'Fix: Reduce num_samples or val_samples so the train/validation split fits inside the training pool.' in captured.err
    assert 'num_samples=55000' in captured.err
    assert 'val_samples=5000' in captured.err


def test_train_flex_reports_structured_dataset_split_error_for_cifar10(capsys):
    import pytest

    from minicnn.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main([
            'train-flex',
            '--config',
            'templates/cifar10/vgg_mini.yaml',
            'dataset.num_samples=45001',
            'dataset.val_samples=5000',
            'train.device=cpu',
        ])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert captured.out == ''
    assert '[ERROR] Dataset split invalid' in captured.err
    assert 'Cause: num_samples + val_samples exceeds available training samples for cifar10' in captured.err
    assert 'Fix: Reduce num_samples or val_samples so the train/validation split fits inside the training pool.' in captured.err
    assert 'num_samples=45000' in captured.err
    assert 'val_samples=5000' in captured.err
