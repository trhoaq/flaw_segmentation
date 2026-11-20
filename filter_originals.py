"""Move non-segmentation datasets outside the active collection; never delete files."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path('D:/dataset/rail').resolve()
    excluded = Path('D:/dataset/rail_excluded_no_segmentation').resolve()
    assert str(root).lower() == 'd:\\dataset\\rail'
    assert str(excluded).lower() == 'd:\\dataset\\rail_excluded_no_segmentation'
    names = ['RSD-YOLO', 'RailDefectDetection_DATEM', 'Railway_Track_Surface_Faults', 'RailEval']
    plan = []
    for name in names:
        source, target = (root / name).resolve(), (excluded / name).resolve()
        assert source.is_relative_to(root) and target.is_relative_to(excluded)
        assert source != root and target != excluded
        if source.exists() and target.exists():
            raise FileExistsError(f'Both source and destination exist; refusing to overwrite {target}')
        plan.append({'dataset': name, 'from': str(source), 'to': str(target), 'already_moved': not source.exists() and target.exists()})
    if args.apply:
        excluded.mkdir(exist_ok=True)
        for item in plan:
            source, target = Path(item['from']), Path(item['to'])
            if source.exists():
                source.rename(target)
            assert target.is_dir() and not source.exists()
        result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'original_file_contents_modified': False,
                  'retained_segmentation_datasets': ['RSDDs', 'NRSD-MN', 'NEU_RSDDS_113', 'CGAN_Rail_Surface_Defects', 'Rail-5k'],
                  'excluded': plan, 'note': 'Rail-5k originals include unmatched samples; only its 10 matched image/mask pairs enter yolo. Historical download reports retain their original download-time paths.'}
        (root / 'segmentation_selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
        (excluded / 'RESTORE_INFO.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({'applied': args.apply, 'moves': plan}, indent=2))


if __name__ == '__main__':
    main()
