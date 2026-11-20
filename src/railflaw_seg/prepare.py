from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import random

import cv2
from PIL import Image
import yaml

from .catalog import discover
from .masks import load_binary_mask, mask_iou, mask_to_polygons, parse_labels, rasterize, serialize_polygons


def known_hashes(root):
    records = {}
    manifests = [root / 'file_manifest.csv'] + [root / n / 'data_manifest.csv' for n in ['NRSD-MN', 'NEU_RSDDS_113', 'CGAN_Rail_Surface_Defects']]
    for file in manifests:
        if file.exists():
            with file.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    records[row['path'].replace('\\', '/').casefold()] = row['sha256']
    return records


def assign_splits(samples, hashes, root, seed, train_ratio, val_ratio):
    parents = {s.group: s.group for s in samples}

    def find(key):
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    by_digest = {}
    for s in samples:
        digest = hashes.get(s.image.relative_to(root).as_posix().casefold())
        if digest:
            if digest in by_digest:
                a, b = find(s.group), find(by_digest[digest])
                if a != b:
                    parents[max(a, b)] = min(a, b)
            else:
                by_digest[digest] = s.group
    families = defaultdict(set)
    for group in parents:
        canonical = find(group)
        families[canonical.split(':', 1)[0]].add(canonical)
    assignments = {}
    for family, groups in families.items():
        ordered = sorted(groups)
        random.Random(f'{seed}:{family}').shuffle(ordered)
        count = len(ordered)
        if count >= 3:
            nval = max(1, round(count * val_ratio))
            ntest = max(1, round(count * (1 - train_ratio - val_ratio)))
            ntrain = max(1, count - nval - ntest)
        else:
            ntrain, nval = 1, int(count == 2)
        for index, group in enumerate(ordered):
            assignments[group] = 'train' if index < ntrain else 'val' if index < ntrain + nval else 'test'
    return {group: (find(group), assignments[find(group)]) for group in parents}


def prepare(config_path: Path, limit: int | None = None):
    cfg = yaml.safe_load(config_path.read_text('utf-8'))
    root, output = Path(cfg['source_root']).resolve(), Path(cfg['output_root']).resolve()
    if output == root or not output.is_relative_to(root) or output.name.casefold() != 'yolo':
        raise ValueError('output_root must be the yolo subdirectory; source directories are read-only')
    cv2.setNumThreads(1)
    samples, omitted = discover(root, cfg.get('convert_nrsd_original', True), cfg.get('include_coarse_rail5k', False))
    digests = known_hashes(root)
    assignments = assign_splits(samples, digests, root, cfg['seed'], cfg['train_ratio'], cfg['val_ratio'])
    if limit:
        samples = samples[:limit]
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    config_record = output / 'prepare_config.json'
    if config_record.exists() and json.loads(config_record.read_text('utf-8'))['fingerprint'] != fingerprint:
        raise ValueError('Output was prepared with different settings; choose a fresh yolo output or archive it first')
    config_record.write_text(json.dumps({'fingerprint': fingerprint, 'config': cfg}, indent=2), 'utf-8')
    progress = output / 'prepare_progress.json'
    rows, errors = [], []

    def one(sample):
        canonical, split = assignments[sample.group]
        image_bytes = sample.image.read_bytes()
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        expected = digests.get(sample.image.relative_to(root).as_posix().casefold())
        if expected and expected != image_sha:
            raise ValueError(f'Original image changed since verified download: {sample.image}')
        with Image.open(io.BytesIO(image_bytes)) as im:
            width, height = im.size
        mask = load_binary_mask(sample.mask, sample.encoding)
        if mask.shape != (height, width):
            raise ValueError(f'Image/mask dimension mismatch: {sample.image}')
        polygons = mask_to_polygons(mask)
        label_text = serialize_polygons(polygons)
        decoded = parse_labels(label_text)
        iou = mask_iou(mask, rasterize(decoded, mask.shape))
        if iou < cfg.get('minimum_roundtrip_iou', .995):
            raise ValueError(f'Mask/polygon roundtrip IoU {iou:.6f}: {sample.mask}')
        base = output / sample.dataset
        dest = base / 'images' / split / (sample.key + sample.image.suffix.lower())
        label = base / 'labels' / split / (sample.key + '.txt')
        dest.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != image_sha:
                raise ValueError(f'Output image differs; refusing to overwrite: {dest}')
        else:
            dest.write_bytes(image_bytes)
        label.write_bytes(label_text.encode('utf-8'))
        return {'dataset': sample.dataset, 'id': sample.key, 'split': split, 'group': canonical,
                'source_image': str(sample.image), 'source_mask': str(sample.mask), 'encoding': sample.encoding,
                'image': dest.as_posix(), 'label': label.as_posix(), 'width': width, 'height': height,
                'image_sha256': image_sha, 'label_sha256': hashlib.sha256(label_text.encode()).hexdigest(),
                'mask_sha256': hashlib.sha256(sample.mask.read_bytes()).hexdigest(),
                'polygons': len(polygons), 'foreground_pixels': int(mask.sum()), 'roundtrip_iou': iou,
                'synthetic': sample.synthetic, 'combined': sample.combined}

    print(f'Converting {len(samples)} image/mask pairs to {output}', flush=True)
    with ThreadPoolExecutor(max_workers=cfg.get('workers', 4)) as pool:
        futures = {pool.submit(one, sample): sample for sample in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                rows.append(future.result())
            except Exception as e:
                errors.append({'dataset': sample.dataset, 'id': sample.key, 'error': str(e)})
            done = len(rows) + len(errors)
            if done % 100 == 0 or done == len(samples):
                print(f'Prepared {done}/{len(samples)}; failures={len(errors)}', flush=True)
                progress.write_text(json.dumps({'completed': done, 'total': len(samples), 'errors': len(errors)}), 'utf-8')
    rows.sort(key=lambda r: (r['dataset'], r['id']))
    (output / 'conversion_errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), 'utf-8')
    (output / 'omitted_sources.json').write_text(json.dumps(omitted, ensure_ascii=False, indent=2), 'utf-8')
    if errors:
        raise RuntimeError(f'{len(errors)} conversions failed. See {output / "conversion_errors.json"}')
    with (output / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = {0: cfg['class_name']}

    def write_yaml(path, base, lists):
        path.write_text(yaml.safe_dump({'path': base.as_posix(), **lists, 'names': names}, sort_keys=False), 'utf-8')

    datasets = sorted({r['dataset'] for r in rows})
    for dataset in datasets:
        base = output / dataset
        for split in ['train', 'val', 'test']:
            (base / 'images' / split).mkdir(parents=True, exist_ok=True)
            (base / 'labels' / split).mkdir(parents=True, exist_ok=True)
        write_yaml(base / 'data.yaml', base, {split: f'images/{split}' for split in ['train', 'val', 'test']})
    for real_only, filename in [(False, 'data.yaml'), (True, 'data_real.yaml')]:
        prefix = 'real_' if real_only else ''
        for split in ['train', 'val', 'test']:
            selected = [r for r in rows if r['combined'] and r['split'] == split and
                        (not r['synthetic'] or (split == 'train' and not real_only))]
            (output / f'{prefix}{split}.txt').write_text(''.join(r['image'] + '\n' for r in selected), 'utf-8')
        write_yaml(output / filename, output, {split: f'{prefix}{split}.txt' for split in ['train', 'val', 'test']})
    statistics = {
        'converted_pairs': len(rows), 'datasets': {d: dict(Counter(r['split'] for r in rows if r['dataset'] == d)) for d in datasets},
        'background_images': sum(r['polygons'] == 0 for r in rows), 'total_polygons': sum(r['polygons'] for r in rows),
        'minimum_roundtrip_iou': min(r['roundtrip_iou'] for r in rows),
        'mean_roundtrip_iou': sum(r['roundtrip_iou'] for r in rows) / len(rows),
        'default_split_counts': {s: len((output / f'{s}.txt').read_text().splitlines()) for s in ['train', 'val', 'test']},
        'class_names': names, 'original_files_modified': False,
        'rail5k_coarse_masks_in_combined': cfg.get('include_coarse_rail5k', False),
    }
    (output / 'conversion_summary.json').write_text(json.dumps(statistics, indent=2), 'utf-8')
    print(json.dumps(statistics, indent=2), flush=True)
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/prepare.yaml'))
    parser.add_argument('--limit', type=int, help='Development-only subset; do not mix with a full output directory')
    args = parser.parse_args()
    prepare(args.config, args.limit)
