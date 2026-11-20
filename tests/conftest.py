import os
from pathlib import Path


def pytest_configure(config):
    directory = Path(__file__).resolve().parents[1] / '.cache/pytest_tmp'
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['PYTEST_DEBUG_TEMPROOT'] = str(directory)
