from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .masks import parse_labels


def validate(root: Path):
    with (root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    problems = []
    group_splits, digest_splits = defaultdict(set), defaultdict(set)
    for index, row in enumerate(rows, 1):
        image, label = Path(row['image']), Path(row['label'])
        if not image.is_file() or not label.is_file():
            problems.append(f'Missing pair: {image}')
            continue
        data = label.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['label_sha256']:
            problems.append(f'Label hash mismatch: {label}')
        try:
            polygons = parse_labels(data.decode('utf-8'))
            if len(polygons) != int(row['polygons']):
                problems.append(f'Polygon count mismatch: {label}')
        except ValueError as e:
            problems.append(f'{label}: {e}')
        group_splits[row['group']].add(row['split'])
        digest_splits[row['image_sha256']].add(row['split'])
        if index % 2000 == 0:
            print(f'Validated {index}/{len(rows)} labels', flush=True)
    for group, splits in group_splits.items():
        if len(splits) > 1:
            problems.append(f'Source-frame group crosses splits: {group}')
    for digest, splits in digest_splits.items():
        if len(splits) > 1:
            problems.append(f'Duplicate image crosses splits: {digest}')
    report = {'pairs': len(rows), 'problems': problems, 'valid': not problems}
    (root / 'validation_report.json').write_text(json.dumps(report, indent=2), 'utf-8')
    print(json.dumps({'pairs': len(rows), 'valid': not problems, 'problem_count': len(problems), 'first_problems': problems[:10]}, indent=2))
    if problems:
        raise ValueError(f'{len(problems)} validation failures')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('D:/dataset/rail/yolo'))
    validate(p.parse_args().data_root)
