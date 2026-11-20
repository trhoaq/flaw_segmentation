import argparse
import csv
import json
from pathlib import Path

import yaml

from .runtime import PROJECT_ROOT, configure_runtime, resolve_device


def smoke_data(source_yaml: Path):
    cfg = yaml.safe_load(source_yaml.read_text('utf-8'))
    root = Path(cfg['path'])
    with (root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    destination = PROJECT_ROOT / 'reports/smoke'
    destination.mkdir(parents=True, exist_ok=True)
    for split, count in [('train', 8), ('val', 4)]:
        candidates = [r for r in rows if r['split'] == split and r['combined'] == 'True' and r['synthetic'] == 'False'
                      and 0 < int(r['polygons']) <= 20 and int(r['foreground_pixels']) >= 64]
        chosen = sorted(candidates, key=lambda r: (int(r['polygons']), int(r['width']) * int(r['height']), r['id']))[:count]
        if len(chosen) < count:
            raise ValueError(f'Not enough smoke-test {split} samples')
        (destination / f'{split}.txt').write_text(''.join(r['image'] + '\n' for r in chosen), 'utf-8')
    result = destination / 'data.yaml'
    result.write_text(yaml.safe_dump({'path': destination.as_posix(), 'train': 'train.txt', 'val': 'val.txt', 'names': cfg['names']}, sort_keys=False), 'utf-8')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/train.yaml')
    p.add_argument('--data', type=Path)
    p.add_argument('--model')
    p.add_argument('--epochs', type=int)
    p.add_argument('--imgsz', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--device')
    p.add_argument('--smoke', action='store_true', help='1 CPU epoch on 8 train / 4 val images; not a trained benchmark model')
    p.add_argument('--resume', type=Path, help='Resume a checkpoint from the project runs directory')
    args = p.parse_args()
    configure_runtime()
    import torch
    from ultralytics import YOLO
    cfg = yaml.safe_load(args.config.read_text('utf-8'))
    for key in ['data', 'model', 'epochs', 'imgsz', 'batch', 'device']:
        if (value := getattr(args, key)) is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    model_name = cfg.pop('model')
    models = PROJECT_ROOT / 'models'
    models.mkdir(exist_ok=True)
    model_path = Path(model_name)
    if model_path.name == model_name:
        model_path = models / model_name
    if args.resume:
        model = YOLO(str(args.resume), task='segment')
        model.train(resume=True)
        return
    if not model_path.name.startswith('yolo26') or not model_path.name.endswith('-seg.pt'):
        raise ValueError('Use an official YOLO26 -seg.pt model for initial training')
    cfg['data'] = str(Path(cfg['data']).resolve())
    cfg['device'] = resolve_device(cfg.get('device', 'auto'))
    cfg['project'] = str((PROJECT_ROOT / cfg.get('project', 'runs')).resolve())
    if args.smoke:
        torch.set_num_threads(min(4, torch.get_num_threads()))
        cfg.update(data=str(smoke_data(Path(cfg['data']))), epochs=1, imgsz=256, batch=2, device='cpu',
                   workers=0, name='smoke_yolo26n_seg', plots=False, amp=False, mosaic=0.0, close_mosaic=0, patience=0)
    model = YOLO(str(model_path), task='segment')
    if model.task != 'segment':
        raise ValueError('Loaded checkpoint is not an instance segmentation model')
    result = model.train(**cfg)
    report = {'smoke_test': args.smoke, 'device': cfg['device'], 'save_dir': str(model.trainer.save_dir),
              'best': str(model.trainer.best), 'last': str(model.trainer.last), 'metrics': result.results_dict if result else {}}
    (PROJECT_ROOT / 'reports').mkdir(exist_ok=True)
    target = PROJECT_ROOT / 'reports' / ('smoke_training.json' if args.smoke else 'training.json')
    target.write_text(json.dumps(report, default=str, indent=2), 'utf-8')
    print(json.dumps(report, default=str, indent=2))
