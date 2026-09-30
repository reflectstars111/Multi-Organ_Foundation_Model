"""Merge available AUL liver/mass annotations into indexed, CAMUS-colored PNGs."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess_pointed_data import CAMUS_PALETTE


def main():
    workspace = Path(__file__).resolve().parents[1]
    datasets = next(p for p in workspace.iterdir() if '(Datasets)' in p.name)
    dataset = next(p for p in (datasets / 'pointed_data').iterdir() if '(AUL)' in p.name)
    root = dataset / 'processed_png'
    output = root / 'masks' / 'liver_mass'
    manifest = root / 'manifest_liver_mass.csv'
    report_path = root / 'report_liver_mass.json'
    palette = [channel for color in CAMUS_PALETTE for channel in color]
    palette += [0] * (768 - len(palette))
    rows, pending = [], []
    counts = Counter()
    outside_pixels = 0
    # Validate all inputs before creating outputs. Missing annotation is not background.
    for source in sorted((root / 'images').rglob('*.png')):
        relative = source.relative_to(root / 'images')
        liver_path = root / 'masks' / 'liver' / relative
        mass_path = root / 'masks' / 'mass' / relative
        missing = [name for name, path in [('liver', liver_path), ('mass', mass_path)] if not path.exists()]
        row = dict(image=source.relative_to(root).as_posix(), mask='', status='missing_' + '_'.join(missing) if missing else 'paired', mass_outside_liver_pixels=0)
        if len(missing) == 2:
            raise ValueError(f'Both annotations missing: {source}')
        if len(missing) < 2:
            with Image.open(source) as image:
                size = image.size
            arrays = []
            for path in (liver_path, mass_path):
                if not path.exists():
                    arrays.append(np.zeros((size[1], size[0]), dtype=bool))
                    continue
                with Image.open(path) as image:
                    array = np.array(image)
                if array.ndim != 2 or array.shape != (size[1], size[0]) or not set(np.unique(array)).issubset({0, 255}):
                    raise ValueError(f'Invalid binary mask: {path}')
                arrays.append(array > 0)
            liver, mass = arrays
            merged = np.zeros(liver.shape, dtype=np.uint8)
            merged[liver] = 1
            merged[mass] = 2
            outside = int(np.count_nonzero(mass & ~liver)) if not missing else 0
            outside_pixels += outside
            row['mass_outside_liver_pixels'] = outside
            target = output / relative
            row['mask'] = target.relative_to(root).as_posix()
            if target.exists():
                with Image.open(target) as existing:
                    if existing.mode != 'P' or not np.array_equal(np.array(existing), merged) or existing.getpalette()[:9] != palette[:9]:
                        raise ValueError(f'Existing output differs; refusing overwrite: {target}')
            pending.append((target, merged))
            counts[relative.parts[0]] += 1
        rows.append(row)
    for target, merged in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            image = Image.fromarray(merged).convert('P')
            image.putpalette(palette)
            image.save(target)
        with Image.open(target) as check:
            assert check.mode == 'P'
            assert np.array_equal(np.array(check), merged)
            assert check.getpalette()[:9] == palette[:9]
    with manifest.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(merged=len(pending), by_group=dict(counts), skipped=[], partial_annotations=[r for r in rows if r['status'] != 'paired'], status_counts=dict(Counter(r['status'] for r in rows)),
                  classes={'0': 'background', '1': 'available liver excluding annotated mass', '2': 'mass'},
                  palette_rgb={str(i): list(CAMUS_PALETTE[i]) for i in range(3)},
                  overlap_priority='mass', missing_policy='merge available label; zero is unannotated remainder for partial samples, not confirmed background',
                  mass_outside_liver_pixels=outside_pixels, verification='all saved PNG indices and palettes checked',
                  original_files_modified=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('skipped', 'partial_annotations')}, ensure_ascii=False, indent=2))
    print(f'Skipped: {len(report["skipped"])}')


if __name__ == '__main__':
    main()
