from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'cifar-10-batches-py'
CPP_ROOT = PROJECT_ROOT / 'cpp'
DOCS_ROOT = PROJECT_ROOT / 'docs'
ARTIFACTS_ROOT = PROJECT_ROOT / 'artifacts'
BEST_MODELS_ROOT = ARTIFACTS_ROOT / 'models'
DEFAULT_RUN_DIR = Path(os.environ.get('MINICNN_ARTIFACT_RUN_DIR', ARTIFACTS_ROOT / 'default'))
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / 'best_model_split.npz'
