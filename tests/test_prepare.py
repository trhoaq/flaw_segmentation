import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from railflaw_seg.catalog import Sample
from railflaw_seg import prepare as module


def test_preparation_writes_portable_label_bytes_and_preserves_sources(tmp_path, monkeypatch):
    image, mask = tmp_path / 'photo.png', tmp_path / 'mask.png'
    Image.fromarray(np.full((12, 14, 3), 120, np.uint8)).save(image)
    binary = np.zeros((12, 14), np.uint8); binary[2:9, 3:11] = 255
    Image.fromarray(binary).save(mask)
    originals = (image.read_bytes(), mask.read_bytes())
    sample = Sample('test', 'photo', image, mask, 'binary_255', 'test:photo')
    monkeypatch.setattr(module, 'discover', lambda *args: ([sample], []))
    cfg = {'source_root': str(tmp_path), 'output_root': str(tmp_path / 'yolo'), 'seed': 42,
           'train_ratio': .8, 'val_ratio': .1, 'workers': 1, 'class_name': 'rail_surface_defect'}
    config = tmp_path / 'prepare.yaml'; config.write_text(yaml.safe_dump(cfg), 'utf-8')
    module.prepare(config)
    with (tmp_path / 'yolo/manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        row = next(csv.DictReader(f))
    data = Path(row['label']).read_bytes()
    assert b'\r' not in data and data.endswith(b'\n')
    assert hashlib.sha256(data).hexdigest() == row['label_sha256']
    assert (image.read_bytes(), mask.read_bytes()) == originals
    assert Path(row['image']).read_bytes() == originals[0]
