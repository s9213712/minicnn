from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from minicnn.flex._datasets import load_test_arrays
from minicnn.flex._training_steps import adapt_targets as _adapt_targets_impl
from minicnn.flex._training_steps import pred_accuracy as _pred_accuracy_impl
from minicnn.flex.builder import build_loss, build_model
from minicnn.torch_runtime import require_torch, resolve_torch_device

CIFAR10_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

MNIST_CLASS_NAMES = [str(i) for i in range(10)]


def _load_pillow():
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError as exc:  # pragma: no cover
        if exc.name == 'PIL':
            raise RuntimeError(
                'Image inference requires Pillow.\n'
                'Install it with:\n'
                '  pip install Pillow'
            ) from exc
        raise
    return Image, ImageOps


def load_best_model_path_from_summary(path: str | Path) -> str | None:
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary file not found: {summary_path}')
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    best_model_path = payload.get('best_model_path')
    if not best_model_path:
        raise ValueError(f'Summary file does not contain best_model_path: {summary_path}')
    return str(best_model_path)


def resolve_checkpoint_path(*, checkpoint_path: str | None, summary_path: str | None) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path)
    elif summary_path:
        summary_file = Path(summary_path)
        path = Path(load_best_model_path_from_summary(summary_file))
        if not path.is_absolute():
            path = summary_file.parent / path
    else:
        raise ValueError('You must provide --checkpoint or --summary')
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def resolve_input_shape(cfg: dict[str, Any]) -> tuple[int, int, int]:
    dataset_cfg = cfg.get('dataset', {})
    input_shape = dataset_cfg.get('input_shape') or dataset_cfg.get('input_size')
    if not input_shape:
        raise ValueError('Config missing dataset.input_shape or dataset.input_size')
    if len(input_shape) != 3:
        raise ValueError(f'Expected image input shape [C,H,W], got {input_shape!r}')
    return tuple(int(dim) for dim in input_shape)


def resolve_class_names(cfg: dict[str, Any]) -> list[str]:
    dataset_cfg = cfg.get('dataset', {})
    explicit = dataset_cfg.get('class_names') or dataset_cfg.get('classes')
    if isinstance(explicit, (list, tuple)) and explicit:
        return [str(item) for item in explicit]

    dataset_type = str(dataset_cfg.get('type', 'cifar10'))
    if dataset_type == 'cifar10':
        return list(CIFAR10_CLASS_NAMES)
    if dataset_type == 'mnist':
        return list(MNIST_CLASS_NAMES)

    num_classes = int(dataset_cfg.get('num_classes', 0) or 0)
    return [f'class_{idx}' for idx in range(num_classes)]


def _normalize_batch_for_dataset(x: np.ndarray, dataset_type: str) -> np.ndarray:
    if dataset_type == 'cifar10':
        from minicnn.data.cifar10 import normalize_cifar

        x_f32 = x.astype(np.float32)
        if np.issubdtype(x.dtype, np.integer) or float(np.max(x_f32)) > 1.5:
            x_f32 = x_f32 / 255.0
        return normalize_cifar(x_f32)
    if dataset_type == 'mnist':
        from minicnn.data.mnist import normalize_mnist

        return normalize_mnist(x.astype(np.float32))
    return x.astype(np.float32)


def _coerce_image_batch_layout(x: np.ndarray, input_shape: tuple[int, int, int]) -> np.ndarray:
    expected_channels, expected_height, expected_width = input_shape
    if x.ndim == 3:
        if x.shape == input_shape:
            return x[None, ...]
        if x.shape == (expected_height, expected_width, expected_channels):
            return np.transpose(x, (2, 0, 1))[None, ...]
    if x.ndim == 4:
        if tuple(x.shape[1:]) == input_shape:
            return x
        if tuple(x.shape[1:]) == (expected_height, expected_width, expected_channels):
            return np.transpose(x, (0, 3, 1, 2))
    raise ValueError(
        f'Test data x shape {tuple(x.shape)} is incompatible with input_shape {input_shape}. '
        'Expected NCHW/CHW or NHWC/HWC.'
    )


def load_test_arrays_for_inference(
    cfg: dict[str, Any],
    *,
    test_data_path: str | None = None,
    test_data_normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    dataset_cfg = cfg.get('dataset', {})
    train_cfg = cfg.get('train', {})
    dataset_type = str(dataset_cfg.get('type', 'cifar10'))

    if test_data_path:
        path = Path(test_data_path)
        if not path.exists():
            raise FileNotFoundError(f'Test data not found: {path}')
        with np.load(path, allow_pickle=False) as payload:
            if 'x' in payload and 'y' in payload:
                x = payload['x']
                y = payload['y']
            elif 'images' in payload and 'labels' in payload:
                x = payload['images']
                y = payload['labels']
            elif 'x_test' in payload and 'y_test' in payload:
                x = payload['x_test']
                y = payload['y_test']
            else:
                raise ValueError(
                    f'Test data {path} must contain x/y, images/labels, or x_test/y_test arrays'
                )
        x = _coerce_image_batch_layout(np.asarray(x), resolve_input_shape(cfg))
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f'Test data batch mismatch: x has {x.shape[0]} samples but y has {y.shape[0]} labels'
            )
        if not test_data_normalized:
            x = _normalize_batch_for_dataset(x, dataset_type)
        else:
            x = x.astype(np.float32)
        return x, y, str(path)

    x_test, y_test = load_test_arrays(dataset_cfg, train_cfg)
    if x_test is None or y_test is None:
        raise ValueError(
            f'dataset.type={dataset_type!r} does not expose an official test split. '
            'Pass --test-data <file.npz>.'
        )
    return np.asarray(x_test, dtype=np.float32), np.asarray(y_test, dtype=np.int64), f'official:{dataset_type}'


def build_torch_model_from_config(
    cfg: dict[str, Any],
    *,
    device_name: str = 'auto',
    checkpoint_path: str | Path | None = None,
    torch_module=None,
):
    torch = torch_module or require_torch('inference', action='run model inference')
    device = resolve_torch_device(device_name, torch)
    input_shape = resolve_input_shape(cfg)
    model = build_model(cfg.get('model', {}), input_shape=input_shape).to(device)
    if checkpoint_path is not None:
        load_torch_checkpoint(model, checkpoint_path, device=device, torch_module=torch)
    model.eval()
    return model, device, torch


def load_torch_checkpoint(model, checkpoint_path: str | Path, *, device, torch_module=None) -> None:
    torch = torch_module or require_torch('inference', action='load a checkpoint')
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f'Unsupported checkpoint format: {type(checkpoint)}')
    model.load_state_dict(state_dict)


def evaluate_checkpoint(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    device_name: str = 'auto',
    batch_size: int | None = None,
    test_data_path: str | None = None,
    test_data_normalized: bool = False,
) -> dict[str, Any]:
    model, device, torch = build_torch_model_from_config(
        cfg,
        device_name=device_name,
        checkpoint_path=checkpoint_path,
    )
    x_test, y_test, dataset_source = load_test_arrays_for_inference(
        cfg,
        test_data_path=test_data_path,
        test_data_normalized=test_data_normalized,
    )
    effective_batch_size = max(1, int(batch_size or cfg.get('train', {}).get('batch_size', 64)))
    loss_cfg = dict(cfg.get('loss', {'type': 'CrossEntropyLoss'}))
    loss_type = str(loss_cfg.get('type', 'CrossEntropyLoss'))
    criterion = build_loss(loss_cfg)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    with torch.no_grad():
        for start in range(0, len(x_test), effective_batch_size):
            end = min(start + effective_batch_size, len(x_test))
            xb = torch.tensor(x_test[start:end], dtype=torch.float32, device=device)
            yb = torch.tensor(y_test[start:end], dtype=torch.long, device=device)
            logits = model(xb)
            adapted = _adapt_targets_impl(torch, yb, logits, loss_type)
            loss = criterion(logits, adapted)
            batch_size_now = int(xb.shape[0])
            total_loss += float(loss.item()) * batch_size_now
            total_acc += _pred_accuracy_impl(torch, logits, yb, loss_type) * batch_size_now
            total_count += batch_size_now

    return {
        'status': 'ok',
        'checkpoint_path': str(Path(checkpoint_path)),
        'dataset_source': dataset_source,
        'device': str(device),
        'num_samples': total_count,
        'batch_size': effective_batch_size,
        'loss_type': loss_type,
        'accuracy': total_acc / max(total_count, 1),
        'loss': total_loss / max(total_count, 1),
    }


def preprocess_image_for_model(
    cfg: dict[str, Any],
    image_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    Image, ImageOps = _load_pillow()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f'Image not found: {path}')

    dataset_cfg = cfg.get('dataset', {})
    dataset_type = str(dataset_cfg.get('type', 'cifar10'))
    channels, height, width = resolve_input_shape(cfg)
    source = Image.open(path)
    source = ImageOps.exif_transpose(source)
    original_size = source.size
    if channels == 1:
        source = source.convert('L')
    elif channels == 3:
        source = source.convert('RGB')
    else:
        raise ValueError(f'Only 1-channel or 3-channel image inputs are supported, got C={channels}')

    resized = ImageOps.fit(source, (width, height), method=Image.BILINEAR, centering=(0.5, 0.5))
    arr = np.asarray(resized)
    if channels == 1:
        arr = arr[np.newaxis, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    batch = arr[np.newaxis, ...]
    batch = _normalize_batch_for_dataset(batch, dataset_type)
    return batch, {
        'original_size': list(original_size),
        'target_size': [width, height],
        'channels': channels,
        'dataset_type': dataset_type,
    }


def predict_image(
    cfg: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    image_path: str | Path,
    device_name: str = 'auto',
    topk: int = 5,
) -> dict[str, Any]:
    model, device, torch = build_torch_model_from_config(
        cfg,
        device_name=device_name,
        checkpoint_path=checkpoint_path,
    )
    x, preprocessing = preprocess_image_for_model(cfg, image_path)
    xb = torch.tensor(x, dtype=torch.float32, device=device)
    class_names = resolve_class_names(cfg)

    with torch.no_grad():
        logits = model(xb)

    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if logits.shape[1] == 1:
        prob = float(torch.sigmoid(logits)[0, 0].item())
        labels = class_names[:2] if len(class_names) >= 2 else ['negative', 'positive']
        predicted_index = 1 if prob >= 0.5 else 0
        predictions = [
            {'rank': 1, 'index': predicted_index, 'label': labels[predicted_index], 'probability': prob if predicted_index == 1 else 1.0 - prob},
            {'rank': 2, 'index': 1 - predicted_index, 'label': labels[1 - predicted_index], 'probability': 1.0 - prob if predicted_index == 1 else prob},
        ]
    else:
        probabilities = torch.softmax(logits, dim=1)
        k = min(max(1, int(topk)), int(probabilities.shape[1]))
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)
        predictions = []
        for rank in range(k):
            index = int(top_indices[0, rank].item())
            label = class_names[index] if index < len(class_names) else f'class_{index}'
            predictions.append({
                'rank': rank + 1,
                'index': index,
                'label': label,
                'probability': float(top_probs[0, rank].item()),
            })

    return {
        'status': 'ok',
        'checkpoint_path': str(Path(checkpoint_path)),
        'image_path': str(Path(image_path)),
        'device': str(device),
        'preprocessing': preprocessing,
        'predictions': predictions,
    }
