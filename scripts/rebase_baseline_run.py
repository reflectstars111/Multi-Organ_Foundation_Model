"""Rebase a copied Windows baseline run to Linux paths after exact record audit."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from unet_moe.baseline_suite import frozen_records


def semantic_key(row):
    def relative(value):
        path = value.replace('\\', '/')
        marker = '/processed_png/'
        if marker not in path:
            raise ValueError(f'Missing processed_png in {path}')
        return path.split(marker, 1)[1]
    return (row['dataset'], row['task'], row['class_id'], row['split'], row['group'],
            row['split_protocol'], row['label_version'], relative(row['image']),
            relative(row['mask']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    if (args.run/'migration.json').exists():
        raise FileExistsError('Run already rebased')
    record_path = args.run/'records.json'
    old = json.loads(record_path.read_text(encoding='utf-8'))
    protocol_path = args.run/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    seed = int(protocol['args']['seed'])
    fresh, skipped, digest = frozen_records(args.data_root, seed)
    if Counter(map(semantic_key, old['records'])) != Counter(map(semantic_key, fresh)):
        raise ValueError('Linux records differ from the archived experiment beyond file paths')
    previous = old['sha256']
    old.update(records=fresh, skipped=skipped, sha256=digest)
    record_path.write_text(json.dumps(old, ensure_ascii=False), encoding='utf-8')
    protocol['record_sha256'] = digest
    protocol['args']['data_root'] = str(args.data_root.resolve())
    protocol['args']['output'] = str(args.run.resolve())
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    changed = []
    for folder in args.run.iterdir():
        if not folder.is_dir() or not (folder/'config.json').exists():
            continue
        config_path = folder/'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if config['record_sha256'] != previous:
            raise ValueError(f'Checkpoint metadata mismatch: {folder}')
        config['record_sha256'] = digest
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        for filename in ('best.pt', 'last.pt'):
            path = folder/filename
            if path.exists():
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                checkpoint['config']['record_sha256'] = digest
                torch.save(checkpoint, path)
                changed.append(str(path))
        test_path = folder/'test_metrics.json'
        if test_path.exists():
            test = json.loads(test_path.read_text(encoding='utf-8'))
            test['record_sha256'] = digest
            test_path.write_text(json.dumps(test, indent=2), encoding='utf-8')
    comparison = args.run/'comparison.json'
    if comparison.exists():
        data = json.loads(comparison.read_text(encoding='utf-8'))
        data['record_sha256'] = digest
        comparison.write_text(json.dumps(data,indent=2),encoding='utf-8')
    report = dict(old_record_sha256=previous, new_record_sha256=digest,
                  equal_semantic_records=len(fresh), checkpoint_files=changed)
    (args.run/'migration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(equal_semantic_records=len(fresh),checkpoint_count=len(changed),
                          new_record_sha256=digest)),flush=True)


if __name__ == '__main__':
    main()
