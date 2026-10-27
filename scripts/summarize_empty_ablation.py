"""Summarize a prespecified weight-0 versus weight-0.1 validation-only pilot.

No test files or checkpoints are opened. Each arm uses the first epoch attaining
its best overall validation macro positive Dice, as fair_suite training does.
The five acceptance rules below were specified before observing this pilot.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


ABDOMEN_STRUCTURES = ('abdomen_liver', 'abdomen_kidney', 'abdomen_vessels',
                     'abdomen_gallbladder', 'abdomen_spleen')
FIELDS = ('positive_dice', 'empty_false_positive_rate',
          'empty_mean_false_positive_area_fraction', 'positive_zero_overlap_rate',
          'empty_mean_false_positive_pixels')
LIMITS = dict(abdomen_empty_fp_rate_increase=.01, abdomen_positive_dice_decrease=.01,
              overall_positive_dice_decrease=.005, abdomen_zero_overlap_increase=.01)
SELECTION = 'best_validation_macro_positive_dice'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _number(value, label, bounded=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or bounded and value > 1):
        raise ValueError(f'Invalid finite metric: {label}')
    return value


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Invalid {label} hash')
    return value


def _read_arm(folder, expected_weight):
    protocol_path = folder / 'protocol.json'
    protocol = _load(protocol_path)
    config, budget = protocol['config'], protocol['budget']
    if protocol.get('config_sha256') != _digest(config):
        raise ValueError(f'Config hash is invalid: {folder}')
    _hash(protocol.get('record_sha256'), 'record')
    _hash(protocol.get('source_sha256'), 'source')
    source_files = protocol.get('source_files')
    if not source_files or protocol['source_sha256'] != _digest(source_files):
        raise ValueError(f'Source hash does not match source files: {folder}')
    if config.get('empty_weight') != expected_weight or config.get('empty_fraction') != .01:
        raise ValueError('Config must specify the prespecified weights 0 / 0.1 and fraction 0.01')
    if config.get('checkpoint_selection') != SELECTION or config.get('threshold') != .5:
        raise ValueError('Config must use fixed threshold 0.5 and validation macro positive Dice selection')
    if 'region' not in config.get('models', []) or 42 not in config.get('seeds', []):
        raise ValueError('Config must include the region model and training seed 42')
    epochs = budget.get('planned_epochs')
    count = budget.get('images_per_epoch')
    if (type(epochs) is not int or epochs < 1 or epochs != config.get('epochs') or
            type(count) is not int or count < 1 or budget.get('planned_images_per_run') != epochs * count):
        raise ValueError('Inconsistent planned epoch or image budget')
    metric_path = folder / 'region/seed_42/metrics.jsonl'
    rows = [json.loads(line) for line in metric_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row.get('epoch') for row in rows] != list(range(1, epochs + 1)):
        raise ValueError('Require exactly one ordered metric row per epoch and the complete planned budget')
    for row in rows:
        if row.get('kind') != 'region' or row.get('seed') != 42 or row.get('threshold') != .5:
            raise ValueError('Metric model, seed or threshold differs from the frozen comparison')
        if row.get('sample_count') != count:
            raise ValueError('Epoch sample_count differs from the frozen image budget')
        _hash(row.get('sample_sha256'), 'sample draw')
        _number(row.get('macro_positive_dice'), 'macro_positive_dice')
        structures = row.get('per_structure', {})
        if set(name for name in structures if name.startswith('abdomen_')) != set(ABDOMEN_STRUCTURES):
            raise ValueError('Require all five and only five abdominal structures')
        for name, metrics in structures.items():
            samples, positive, empty = (metrics.get(key) for key in
                                        ('samples', 'positive_samples', 'empty_samples'))
            if (any(type(n) is not int or n < 0 for n in (samples, positive, empty)) or
                    samples != positive + empty):
                raise ValueError(f'Invalid samples/positive/empty denominators: {name}')
            if name in ABDOMEN_STRUCTURES and (not positive or not empty):
                raise ValueError(f'All abdominal positive and empty denominators must be nonzero: {name}')
            for field in FIELDS:
                denominator = positive if field in ('positive_dice', 'positive_zero_overlap_rate') else empty
                if not denominator:
                    if metrics.get(field) is not None:
                        raise ValueError(f'Metric with zero denominator must be null: {name}/{field}')
                else:
                    _number(metrics.get(field), f'{name}/{field}', bounded=field != 'empty_mean_false_positive_pixels')
        computed_macro = _macro(structures)['positive_dice']
        if computed_macro is None or not math.isclose(computed_macro, row['macro_positive_dice'], abs_tol=1e-10):
            raise ValueError('Stored overall macro positive Dice differs from the per-structure average')
    return dict(folder=str(folder), protocol=protocol, rows=rows,
                input_sha256={str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (protocol_path, metric_path)})


def _macro(structures):
    result = {}
    for field in FIELDS:
        values = [metrics[field] for metrics in structures.values() if metrics.get(field) is not None]
        result[field] = sum(values) / len(values) if values else None
    return result


def _validate_pair(baseline, candidate):
    left, right = baseline['protocol'], candidate['protocol']
    for key in ('record_sha256', 'source_sha256', 'source_files', 'budget'):
        if left[key] != right[key]:
            raise ValueError(f'{key} differs between comparison arms')
    configs = [{key: value for key, value in arm['config'].items() if key != 'empty_weight'}
               for arm in (left, right)]
    if configs[0] != configs[1]:
        raise ValueError('Config differs beyond the permitted empty_weight change')
    reference_counts = None
    for a, b in zip(baseline['rows'], candidate['rows']):
        for field in ('epoch', 'sample_sha256', 'sample_count', 'annotated_channel_queries',
                      'model_input_images', 'parameter_count', 'lr'):
            if field not in a or field not in b or a[field] != b[field]:
                raise ValueError(f'Paired epoch draw/budget metadata differs: {field}')
        for row in (a, b):
            counts = {name: [metrics[key] for key in ('samples', 'positive_samples', 'empty_samples')]
                      for name, metrics in row['per_structure'].items()}
            if reference_counts is None:
                reference_counts = counts
            if counts != reference_counts:
                raise ValueError('Validation structures or sample denominators differ across arms/epochs')


def _selected(arm):
    # Python max retains the first element on ties, matching training's strict >.
    row = max(arm['rows'], key=lambda metric: metric['macro_positive_dice'])
    structures = row['per_structure']
    return dict(folder=arm['folder'], empty_weight=arm['protocol']['config']['empty_weight'],
                selected_epoch=row['epoch'], overall=_macro(structures),
                abdomen=_macro({name: structures[name] for name in ABDOMEN_STRUCTURES}),
                per_structure={name: {key: metrics[key] for key in
                    ('samples', 'positive_samples', 'empty_samples', *FIELDS)}
                               for name, metrics in structures.items()},
                input_sha256=arm['input_sha256'])


def _differences(baseline, candidate):
    return {field: candidate[field] - baseline[field]
            if candidate[field] is not None and baseline[field] is not None else None for field in FIELDS}


def _decision(differences):
    abdomen, overall = differences['abdomen'], differences['overall']
    rules = {
        'abdomen_empty_area_reduced': (abdomen['empty_mean_false_positive_area_fraction'], '<', 0.),
        'abdomen_empty_fp_rate_preserved': (abdomen['empty_false_positive_rate'], '<=', LIMITS['abdomen_empty_fp_rate_increase']),
        'abdomen_positive_dice_preserved': (abdomen['positive_dice'], '>=', -LIMITS['abdomen_positive_dice_decrease']),
        'overall_positive_dice_preserved': (overall['positive_dice'], '>=', -LIMITS['overall_positive_dice_decrease']),
        'abdomen_zero_overlap_preserved': (abdomen['positive_zero_overlap_rate'], '<=', LIMITS['abdomen_zero_overlap_increase']),
    }
    checks = {}
    for name, (value, operation, bound) in rules.items():
        if value is None:
            raise ValueError(f'Missing metric denominator for prespecified rule: {name}')
        passed = value < bound if operation == '<' else (
            value <= bound + 1e-12 if operation == '<=' else value >= bound - 1e-12)
        checks[name] = dict(candidate_minus_baseline=value, operator=operation, bound=bound, passed=passed)
    promising = all(check['passed'] for check in checks.values())
    return dict(status='pilot_promising' if promising else 'not_selected', checks=checks,
                next_step='Further independent training/validation replication is warranted.' if promising
                else 'Retain disabled empty loss; this candidate fails the prespecified safeguards.',
                interpretation='Single-seed validation pilot only; no proven improvement or test claim.')


def summarize(root):
    root = Path(root)
    output = root / 'comparison.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    baseline = _read_arm(root / 'weight_0', 0.)
    candidate = _read_arm(root / 'weight_0.1', .1)
    _validate_pair(baseline, candidate)
    selected_baseline, selected_candidate = _selected(baseline), _selected(candidate)
    differences = {scope: _differences(selected_baseline[scope], selected_candidate[scope])
                   for scope in ('overall', 'abdomen')}
    differences['per_structure'] = {name: _differences(selected_baseline['per_structure'][name],
                                                      selected_candidate['per_structure'][name])
                                    for name in selected_baseline['per_structure']}
    protocol = baseline['protocol']
    result = dict(protocol='prespecified_empty_ablation_pilot_v1', record_sha256=protocol['record_sha256'],
                  source_sha256=protocol['source_sha256'], full_budget=protocol['budget'],
                  selection=dict(method=SELECTION, tie_break='first epoch', test_data_used=False,
                                 aggregation='Unweighted mean over structures with applicable denominators; abdomen uses exactly five structures.',
                                 prespecified_limits=LIMITS, area_improvement='strict decrease',
                                 inclusive_bound_numeric_tolerance=1e-12),
                  paired_epoch_draws=[{key: row[key] for key in
                      ('epoch', 'sample_sha256', 'sample_count', 'annotated_channel_queries')}
                                      for row in baseline['rows']],
                  baseline=selected_baseline, candidate=selected_candidate, differences=differences,
                  decision=_decision(differences), default_empty_weight=0.)
    # Exclusive creation also protects against a concurrent summary invocation.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs_root)
    print(json.dumps(dict(output=str(args.runs_root / 'comparison.json'),
                          status=result['decision']['status'],
                          baseline_epoch=result['baseline']['selected_epoch'],
                          candidate_epoch=result['candidate']['selected_epoch'],
                          default_empty_weight=result['default_empty_weight'])))


if __name__ == '__main__':
    main()
