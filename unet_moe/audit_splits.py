"""Export source-aware splits and unique-image/group counts without training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .data import SPLIT_VERSION, build_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    records, skipped = build_records(args.data_root, args.seed)
    buckets = defaultdict(list)
    for row in records:
        buckets[(row['dataset'], row['split'])].append(row)
    counts = {f'{dataset}/{split}': dict(records=len(rows),
              images=len({r['image'] for r in rows}), groups=len({r['group'] for r in rows}),
              protocols=sorted({r['split_protocol'] for r in rows}))
              for (dataset, split), rows in sorted(buckets.items())}
    report = dict(version=SPLIT_VERSION, seed=args.seed, counts=counts, skipped=skipped,
                  group_overlap=False, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
