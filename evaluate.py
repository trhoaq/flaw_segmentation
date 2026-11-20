import argparse
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from railflaw_seg.runtime import PROJECT_ROOT, configure_runtime, resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', default='D:/dataset/rail/yolo/data.yaml')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    configure_runtime()
    from ultralytics import YOLO
    model = YOLO(str(args.weights), task='segment')
    if model.task != 'segment':
        raise ValueError('Expected segmentation checkpoint')
    model.val(data=args.data, split=args.split, device=resolve_device(args.device), workers=0,
              project=str(PROJECT_ROOT / 'runs/evaluate'))


if __name__ == '__main__':
    main()
