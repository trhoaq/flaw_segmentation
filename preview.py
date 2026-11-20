import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from _bootstrap import bootstrap
bootstrap()
from railflaw_seg.masks import parse_labels, rasterize
from railflaw_seg.runtime import PROJECT_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('D:/dataset/rail/yolo'))
    p.add_argument('--output', type=Path, default=PROJECT_ROOT / 'reports/dataset_preview.jpg')
    args = p.parse_args()
    with (args.data_root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    names = ['RSDDs_Type_1', 'RSDDs_Type_2', 'NRSD_relabel', 'NEU_RSDDS_113', 'CGAN', 'Rail5k_sample']
    canvas = Image.new('RGB', (1100, 1080), '#141b26')
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        candidates = [r for r in rows if r['dataset'] == name and 0 < int(r['polygons']) <= 10]
        row = min(candidates, key=lambda r: abs(int(r['foreground_pixels']) / (int(r['width']) * int(r['height'])) - .15))
        image = np.asarray(Image.open(row['image']).convert('RGB')).copy()
        mask = rasterize(parse_labels(Path(row['label']).read_text('utf-8')), (int(row['height']), int(row['width'])))
        foreground = mask.astype(bool)
        image[foreground] = (image[foreground] * .55 + np.array([30, 245, 140]) * .45).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 210, 50), 1)
        if image.shape[0] > image.shape[1] * 3:
            ys, xs = np.nonzero(mask)
            center = int(np.median(ys))
            half = min(image.shape[0] // 2, max(image.shape[1], 100))
            start = max(0, min(center - half, image.shape[0] - 2 * half))
            image = image[start:start + 2 * half]
        tile = ImageOps.contain(Image.fromarray(image), (520, 295))
        x, y = (index % 2) * 550, (index // 2) * 360
        canvas.paste(tile, (x + (550 - tile.width) // 2, y + 52 + (295 - tile.height) // 2))
        draw.text((x + 18, y + 12), name, fill='#ffffff')
        draw.text((x + 18, y + 29), f"{row['id']} | green=defect; yellow=boundary", fill='#b9c6d8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(args.output)


if __name__ == '__main__':
    main()
