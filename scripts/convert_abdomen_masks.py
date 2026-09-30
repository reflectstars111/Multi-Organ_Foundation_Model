"""Generate separate indexed PNG masks and manifest; preserve all originals."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unet_moe.abdomen import COLORS, NAMES, rgb_to_ids


def main():
    workspace = Path(__file__).resolve().parents[1]
    datasets = next(p for p in workspace.iterdir() if '(Datasets)' in p.name)
    dataset = next(p for p in (datasets / 'pointed_data').iterdir() if p.is_dir() and 'AbdomenUS' in p.name)
    root = dataset / 'processed_png'
    output = root / 'masks_indexed'
    manifest = root / 'manifest_indexed.csv'
    if output.exists() or manifest.exists():
        raise FileExistsError('Indexed outputs already exist; refusing overwrite')
    palette = [v for color in COLORS for v in color] + [0] * (768 - len(COLORS)*3)
    palette[765:768] = [10, 10, 10]
    rows, pending = [], []
    counts = np.zeros(256, dtype=np.int64)
    for path in sorted((root / 'images/AUS').rglob('*.png')):
        relative = path.relative_to(root / 'images')
        source = root / 'masks' / relative
        with Image.open(source) as mask:
            rgb = np.array(mask.convert('RGB'))
        ids = rgb_to_ids(rgb)
        with Image.open(path) as image:
            if image.size != (ids.shape[1], ids.shape[0]):
                raise ValueError(f'Size mismatch: {source}')
        counts += np.bincount(ids.ravel(), minlength=256)
        target = output / relative
        pending.append((target, ids))
        rows.append(dict(dataset='AbdomenUS', sample_id=path.stem, split=relative.parts[1],
                         image=path.relative_to(root).as_posix(), mask=target.relative_to(root).as_posix(),
                         pairing_status='paired', label='multiclass'))
    for target, ids in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(ids).convert('P')
        image.putpalette(palette)
        image.save(target)
        with Image.open(target) as saved:
            if saved.mode != 'P' or not np.array_equal(np.array(saved), ids):
                raise ValueError(f'Roundtrip failed: {target}')
    with manifest.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(converted=len(rows), classes={i:dict(name=name,rgb=COLORS[i],pixels=int(counts[i])) for i,name in enumerate(NAMES)},
                  ignore=dict(value=255,rgb=[10,10,10],pixels=int(counts[255]),reason='unspecified semantics'),
                  unknown_colors=0, exact_mapping=True, roundtrip_verified=True)
    (root/'report_indexed.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
