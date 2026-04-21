"""Phase 12: Capability docs + example config smoke tests."""
from __future__ import annotations
from pathlib import Path
import pytest


CONFIGS_DIR = Path(__file__).parent.parent / 'configs'


def test_flex_broad_config_parses():
    from minicnn.flex.config import load_flex_config
    cfg = load_flex_config(CONFIGS_DIR / 'flex_broad.yaml')
    assert cfg['optimizer']['type'] == 'AdamW'
    assert cfg['scheduler']['type'] == 'CosineAnnealingLR'
    aug = cfg.get('augmentation', {})
    assert aug.get('horizontal_flip') is True


def test_autograd_enhanced_config_parses():
    import yaml
    raw = yaml.safe_load((CONFIGS_DIR / 'autograd_enhanced.yaml').read_text())
    assert raw['optimizer']['type'] == 'AdamW'
    assert raw['scheduler']['type'] == 'cosine'
    assert raw['loss'].get('label_smoothing', 0) > 0


def test_cuda_legacy_strict_config_validates():
    from minicnn.unified.cuda_legacy import validate_cuda_legacy_compatibility
    import yaml
    raw = yaml.safe_load((CONFIGS_DIR / 'cuda_legacy_strict.yaml').read_text())
    errors = validate_cuda_legacy_compatibility(raw)
    assert errors == [], f'Unexpected validation errors: {errors}'


def test_backend_capabilities_doc_exists():
    doc = Path(__file__).parent.parent / 'docs' / 'backend_capabilities.md'
    assert doc.exists()
    text = doc.read_text()
    assert 'Flex' in text
    assert 'autograd' in text.lower()
    assert 'CUDA legacy' in text or 'cuda_legacy' in text.lower()


def test_usage_md_links_capabilities():
    doc = Path(__file__).parent.parent / 'USAGE.md'
    text = doc.read_text()
    assert 'backend_capabilities.md' in text
