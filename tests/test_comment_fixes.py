import numpy as np


def test_plateau_scheduler_reduces_after_configured_patience():
    from minicnn.schedulers.plateau import ReduceLROnPlateau

    class Optimizer:
        lr = 1.0

    optimizer = Optimizer()
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=0.01)

    scheduler.step(1.0)
    scheduler.step(1.0)

    assert optimizer.lr == 0.5


def test_init_weights_does_not_consume_global_numpy_rng():
    from minicnn.models.initialization import init_weights

    np.random.seed(123)
    np.random.random()
    init_weights(1)
    actual = np.random.random()

    np.random.seed(123)
    np.random.random()
    expected = np.random.random()

    assert actual == expected


def test_torch_baseline_evaluate_does_not_force_training_mode():
    import torch

    from minicnn.training.train_torch_baseline import evaluate

    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 2), device=x.device)

    model = ToyModel()
    model.eval()
    x = np.zeros((4, 3, 32, 32), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int64)

    evaluate(model, x, y, torch.device('cpu'), batch_size=2, max_batches=1)

    assert model.training is False
