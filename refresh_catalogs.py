"""Change merged training selection without re-converting masks or touching sources."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('D:/dataset/rail/yolo'))
    p.add_argument('--include-coarse-rail5k', action='store_true')
    args = p.parse_args()
    root = args.data_root.resolve()
    with (root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    for row in rows:
        if row['dataset'] == 'Rail5k_sample':
            row['combined'] = str(args.include_coarse_rail5k)
    with (root / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for real_only in [False, True]:
        prefix = 'real_' if real_only else ''
        for split in ['train', 'val', 'test']:
            chosen = [r for r in rows if r['combined']=='True' and r['split']==split and
                      (r['synthetic']=='False' or (split=='train' and not real_only))]
            (root / f'{prefix}{split}.txt').write_text(''.join(r['image']+'\n' for r in chosen), 'utf-8')
    summary_path = root / 'conversion_summary.json'
    summary = json.loads(summary_path.read_text('utf-8'))
    summary['default_split_counts'] = {s:len((root/f'{s}.txt').read_text().splitlines()) for s in ['train','val','test']}
    summary['rail5k_coarse_masks_in_combined'] = args.include_coarse_rail5k
    summary_path.write_text(json.dumps(summary,indent=2),'utf-8')
    record_path = root/'prepare_config.json'
    record=json.loads(record_path.read_text('utf-8'));record['config']['include_coarse_rail5k']=args.include_coarse_rail5k
    record['fingerprint']=hashlib.sha256(json.dumps(record['config'],sort_keys=True).encode()).hexdigest()
    record_path.write_text(json.dumps(record,indent=2),'utf-8')
    print(json.dumps({'rail5k_in_combined':args.include_coarse_rail5k,'splits':summary['default_split_counts']},indent=2))


if __name__ == '__main__':
    main()
