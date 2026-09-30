"""Summarize nine independent U-Nets trained against one frozen protocol."""
import argparse
import json
from pathlib import Path

import numpy as np

from unet_moe.data import TASKS
from unet_moe.region import DOMAINS, PAPER_ALIGNED_TASK_IDS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    expected = {TASKS[task] for task in PAPER_ALIGNED_TASK_IDS}
    digest = read(args.run / 'records.json')['sha256']
    structures = {}
    parameters = {}
    epochs = {}
    for domain in DOMAINS:
        folder = args.run / domain
        metrics = read(folder / 'test_metrics.json')
        config = read(folder / 'config.json')
        if metrics['record_sha256'] != digest or config['record_sha256'] != digest:
            raise ValueError(f'Frozen record mismatch: {domain}')
        overlap = structures.keys() & metrics['per_structure'].keys()
        if overlap:
            raise ValueError(f'Duplicate structure labels: {overlap}')
        structures.update(metrics['per_structure'])
        parameters[domain] = config['parameters']
        epochs[domain] = metrics['checkpoint_epoch']
    if set(structures) != expected:
        raise ValueError(f'Expected 16 structures, got {sorted(structures)}')
    values = [v['positive_dice'] for v in structures.values()]
    if any(value is None for value in values):
        raise ValueError('A structure has no positive test sample')
    summary = dict(protocol='paper5_ct7_2_v1', record_sha256=digest,
                   macro_positive_dice=float(np.mean(values)),
                   parameters_total=sum(parameters.values()),
                   parameters_by_domain=parameters,
                   best_epoch_by_domain=epochs,
                   per_structure=structures)
    output = args.run / 'independent_comparison.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output),
                          macro_positive_dice=summary['macro_positive_dice']), ensure_ascii=False))


if __name__ == '__main__':
    main()
