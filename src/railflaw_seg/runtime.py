import os
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_runtime():
    temporary = PROJECT_ROOT / '.cache/tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = str(temporary)
    os.environ['TMP'] = str(temporary)
    tempfile.tempdir = str(temporary)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / '.cache/ultralytics'))
    os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT / '.cache/torch'))
    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache/matplotlib'))
    os.environ.setdefault('WANDB_MODE', 'disabled')
    from ultralytics import settings
    settings.update({'sync': False})


def resolve_device(value):
    if value != 'auto':
        return value
    import torch
    return 0 if torch.cuda.is_available() else 'cpu'
