from pathlib import Path
import sys


def bootstrap():
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
