import argparse
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from railflaw_seg.runtime import PROJECT_ROOT, configure_runtime, resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--conf', type=float, default=.25)
    args = p.parse_args()
    configure_runtime()
    from ultralytics import YOLO
    model = YOLO(str(args.weights), task='segment')
    if model.task != 'segment':
        raise ValueError('Expected segmentation checkpoint')
    model.predict(source=args.source, device=resolve_device(args.device), conf=args.conf,
                  retina_masks=True, save=True, project=str(PROJECT_ROOT / 'runs/predict'))


if __name__ == '__main__':
    main()
