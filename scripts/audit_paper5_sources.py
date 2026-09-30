"""Read-only positive-image coverage by split and source for the frozen paper-five run."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding='utf-8'))['records']
    groups = defaultdict(list)
    for row in records:
        groups[row['image']].append(row)
    totals = defaultdict(lambda: dict(images=0, positive=defaultdict(int)))
    for rows in groups.values():
        row = rows[0]
        key = (row['dataset'], row['split'], row['group'] if row['dataset'] == 'AbdomenUS' else 'all')
        totals[key]['images'] += 1
        if row['dataset'] != 'AbdomenUS':
            continue
        with Image.open(row['mask']) as mask:
            present = set(np.unique(np.asarray(mask)))
        for name, class_id in [('liver', 1), ('kidney', 2), ('vessels', 4),
                               ('gallbladder', 6), ('spleen', 8)]:
            totals[key]['positive'][name] += class_id in present
    for key, result in sorted(totals.items()):
        print(*key, 'images=' + str(result['images']),
              'positive=' + json.dumps(result['positive'], sort_keys=True))


if __name__ == '__main__':
    main()
