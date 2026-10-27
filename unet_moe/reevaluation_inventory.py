"""Read-only inventory for retrospective medical segmentation evaluation.

This module never opens an image, a mask or a torch checkpoint.  It describes
available artifacts and their original evaluation permissions; the evaluator
must additionally validate the checkpoint and freeze its bytes before use.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNS = Path('/ssd1/data/Multi-Organ_Foundation_Model/runs')
SELECTIONS = ('best', 'last', 'guardrail')
STATUSES = ('ready', 'deferred', 'excluded', 'unavailable', 'not_selected')
_RECORD_CACHE: dict[tuple, tuple] = {}
_HISTORY_CACHE: dict[tuple, list] = {}


def _read(path: Path, default=None):
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _records_info(path: Path):
    stat = path.stat() if path.is_file() else None
    key = (str(path), stat.st_mtime_ns, stat.st_size) if stat else (str(path), None)
    if key in _RECORD_CACHE:
        return _RECORD_CACHE[key]
    value = _read(path, [])
    rows = value.get('records', []) if isinstance(value, dict) else value
    result = Counter(row.get('split') for row in rows), _digest(value) if path.is_file() else None
    _RECORD_CACHE[key] = result
    return result


def _history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _HISTORY_CACHE:
        return _HISTORY_CACHE[key]
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live trainer may be in the middle of writing its final line.
            if index != len(lines) - 1:
                raise
    _HISTORY_CACHE[key] = rows
    return rows


def _summary_for(protocol_root: Path, run_dir: Path, kind, seed):
    for name in ('run_summary.json', 'result.json'):
        if (run_dir / name).is_file():
            return _read(run_dir / name)
    for name in ('summary.json', 'training_summary.json'):
        summary = _read(protocol_root / name)
        for row in summary.get('runs', []) if isinstance(summary, dict) else []:
            directory = row.get('checkpoint_directory')
            if directory and Path(directory).resolve() == run_dir.resolve():
                return row
            if not directory and row.get('kind', kind) == kind and row.get('seed') == seed:
                return row
    return {}


def _family(experiment: Path, protocol: dict) -> str:
    version = str(protocol.get('protocol_version', protocol.get('config', {}).get('protocol_version', '')))
    if 'region_normalization' in version or 'region_domain_norm' in experiment.name:
        return 'region_normalization'
    if 'organ_weighted' in version or experiment.name.startswith('q01_'):
        return 'organ_weighted'
    if 'single_channel' in version:
        return 'single_union'
    if 'fair_image' in version:
        return 'fair'
    if 'args' in protocol:
        return 'legacy_baseline'
    if experiment.name.startswith('task_id_'):
        return 'legacy_task_id'
    if experiment.name.startswith('region_'):
        return 'legacy_region'
    return 'unknown'


def _source_root(experiment: Path, protocol_root: Path, project_root: Path):
    for parent in (protocol_root, experiment):
        candidate = parent / 'source_snapshot'
        if candidate.is_dir():
            return candidate, True
    return project_root, False


def _run_specs(experiment: Path, protocol_root: Path, protocol: dict, family: str):
    config = protocol.get('config', protocol.get('args', {}))
    if family in ('organ_weighted', 'region_normalization'):
        for arm in config['arms']:
            for kind in config['models']:
                for seed in config['seeds']:
                    yield protocol_root / arm / kind / f'seed_{seed}', kind, arm, seed
    elif family == 'single_union':
        for seed in config['seeds']:
            yield protocol_root / f'seed_{seed}', 'single_union', None, seed
    elif family == 'fair':
        for kind in config['models']:
            for seed in config['seeds']:
                arm = protocol_root.name if protocol_root != experiment and protocol_root.name.startswith('weight_') else None
                yield protocol_root / kind / f'seed_{seed}', kind, arm, seed
    else:
        parents = sorted({p.parent for p in experiment.rglob('*.pt')
                          if 'source_snapshot' not in p.parts})
        for folder in parents:
            local = _read(folder / 'config.json') or _read(folder / 'training_config.json')
            kind = ('task_id' if family == 'legacy_task_id' else 'region' if family == 'legacy_region'
                    else 'shared' if folder.name == 'shared' else 'independent')
            yield folder, kind, local.get('domain'), local.get('seed', config.get('seed'))


def _jobs_for_protocol(experiment, protocol_root, protocol, project_root):
    family = _family(experiment, protocol)
    config = protocol.get('config', protocol.get('args', {}))
    protocol_path = protocol_root / 'protocol.json'
    records_path = protocol_root / 'records.json'
    source_root, source_frozen = _source_root(experiment, protocol_root, project_root)
    specs = list(_run_specs(experiment, protocol_root, protocol, family))
    guarded = family in ('organ_weighted', 'region_normalization')
    summaries = {(kind, arm, seed): _summary_for(protocol_root, folder, kind, seed)
                 for folder, kind, arm, seed in specs}
    all_training_complete = bool(specs) and all(
        row.get('epochs_completed', -1) >= config.get('epochs', 1) for row in summaries.values())
    # The original controller may have stale status after a priority handoff.
    # Actual run summaries and an actual opened test are the authority.
    test_opened = any(row.get('test_evaluated') for row in summaries.values())
    overall = _read(experiment / 'summary.json')
    test_opened = test_opened or bool(isinstance(overall, dict) and overall.get('complete') and overall.get('test_evaluated'))
    splits, records_digest = _records_info(records_path)
    for folder, kind, arm, seed in specs:
        run_config = dict(config)
        if family.startswith('legacy_'):
            run_config.update(_read(folder / 'config.json') or _read(folder / 'training_config.json'))
        result = summaries[(kind, arm, seed)]
        history = [] if 'epochs_completed' in result else _history(folder / 'metrics.jsonl')
        completed = result.get('epochs_completed', max((r.get('epoch', 0) for r in history), default=0))
        budget = run_config.get('epochs')
        finished = budget is not None and completed >= budget
        archived_partial = experiment.name == 'empty_ablation_20260930' and 0 < completed < (budget or 0)
        selections = SELECTIONS if guarded else ('best', 'last')
        for selection in selections:
            checkpoint = folder / f'{selection}.pt'
            expected_epoch = result.get('best_epoch') if selection == 'best' else (
                result.get('guardrail_epoch') if selection == 'guardrail' else completed if archived_partial else budget)
            for split in ('val', 'test'):
                status, reason = 'ready', 'completed checkpoint; original split records available'
                if 'smoke' in experiment.name.lower():
                    status, reason = 'excluded', 'engineering smoke run; not a scientific experiment result'
                elif family == 'unknown':
                    status, reason = 'unavailable', 'checkpoint model/data protocol is not recognized'
                elif not finished and experiment.name == 'fair_80_base24_batch16_512_20260930':
                    status, reason = 'excluded', 'aborted 80-epoch launch; superseded by 20261001_fix and produced no trained checkpoint'
                elif not finished and not archived_partial:
                    status, reason = 'deferred', 'training budget not complete; checkpoint may still change'
                elif selection == 'guardrail' and result.get('guardrail_epoch') is None and checkpoint.is_file():
                    status, reason = 'unavailable', 'guardrail file exists although the run reports no selected guardrail epoch'
                elif selection == 'guardrail' and result.get('guardrail_epoch') is None:
                    status, reason = 'not_selected', 'no epoch passed the prespecified guardrail; no checkpoint substitution'
                elif not checkpoint.is_file():
                    status, reason = 'unavailable', 'completed run is missing the requested checkpoint'
                elif not records_path.is_file():
                    status, reason = 'unavailable', 'original frozen split records are missing'
                elif family in ('fair', 'single_union', 'organ_weighted', 'region_normalization') and protocol.get('record_sha256') not in (None, records_digest):
                    status, reason = 'unavailable', 'frozen records differ from the original protocol digest'
                elif not splits.get(split):
                    status, reason = 'excluded', f'original records contain no {split} samples'
                elif split == 'test' and guarded and not (all_training_complete and test_opened):
                    status, reason = 'deferred', 'original all-training/test-opening barrier has not completed'
                elif split == 'test' and experiment.name.startswith('empty_ablation') and not result.get('test_evaluated'):
                    status, reason = 'excluded', 'original ablation is validation-only; no original held-out test result'
                elif split == 'test' and family in ('fair', 'single_union') and not result.get('test_evaluated'):
                    status, reason = 'deferred', 'original fair/single-union test-opening barrier has not completed'
                elif archived_partial:
                    reason = 'archived partial training checkpoint; descriptive validation only, not a completed-budget comparison'
                run_identity = str(folder.resolve())
                identity = dict(run_identity=run_identity, selection=selection, split=split)
                stat = checkpoint.stat() if checkpoint.is_file() else None
                yield dict(job_id=_digest(identity), run_identity=run_identity, experiment=experiment.name,
                           experiment_root=str(experiment), run_dir=str(folder), checkpoint=str(checkpoint),
                           selection=selection, split=split, family=family, kind=kind, arm=arm, seed=seed,
                           domain=run_config.get('domain'),
                           protocol_root=str(protocol_root), protocol_path=str(protocol_path) if protocol_path.is_file() else None,
                           records_path=str(records_path), source_root=str(source_root), source_frozen=source_frozen,
                           source_files=protocol.get('source_files', {}), config=run_config,
                           checkpoint_selection_policy=run_config.get('checkpoint_selection', protocol.get('selection')),
                           task_ids=run_config.get('task_ids'), size=run_config.get('size'),
                           protocol_sha256=protocol.get('protocol_sha256', _digest(protocol)),
                           records_sha256=records_digest, epochs_completed=completed, training_epochs=budget,
                           archived_partial=archived_partial, training_status='archived_partial' if archived_partial else 'complete' if finished else 'incomplete',
                           expected_epoch=expected_epoch, all_training_complete=all_training_complete,
                           original_test_opened=test_opened, status=status, reason=reason,
                           checkpoint_size=stat.st_size if stat else None,
                           checkpoint_mtime_ns=stat.st_mtime_ns if stat else None,
                           comparability_group=_digest(dict(records=records_digest, size=run_config.get('size'),
                                                           split_policy=run_config.get('split_policy'),
                                                           abdomen_split=run_config.get('abdomen_split'),
                                                           epochs=completed if archived_partial else budget, family=family,
                                                           training_status='archived_partial' if archived_partial else 'complete',
                                                           task_ids=run_config.get('task_ids'))))


def _recovery_jobs(experiment, project_root):
    plan_path = experiment / 'plan.json'
    plan = _read(plan_path)
    reference = Path(plan['fair_root'])
    reference_root = reference / 'shards/gpu1'
    protocol = _read(reference_root / 'protocol.json')
    config = {**protocol.get('config', {}), **plan['design']}
    records_path = reference_root / 'records.json'
    splits, records_digest = _records_info(records_path)
    source_root, source_frozen = _source_root(experiment, experiment, project_root)
    if not source_frozen:
        source_root, source_frozen = _source_root(reference, reference_root, project_root)
    for arm in plan['arms']:
        arm_name = arm['name'] if isinstance(arm, dict) else arm
        for seed in plan['seeds']:
            folder = experiment / arm_name / f'seed_{seed}'
            result = _read(folder / 'result.json')
            finished = result.get('epochs_completed', 0) >= config['epochs']
            for selection in ('best', 'last'):
                checkpoint = folder / f'{selection}.pt'
                for split in ('val', 'test'):
                    status, reason = 'ready', 'completed validation-only recovery experiment'
                    if split == 'test':
                        status, reason = 'excluded', 'original recovery protocol explicitly prohibits opening test data'
                    elif not finished:
                        status, reason = 'deferred', 'recovery training is not complete'
                    elif not checkpoint.is_file() or not splits.get(split):
                        status, reason = 'unavailable', 'recovery checkpoint or original validation records missing'
                    identity = dict(run_identity=str(folder), selection=selection, split=split)
                    yield dict(**identity, job_id=_digest(identity), experiment=experiment.name,
                               experiment_root=str(experiment), run_dir=str(folder), checkpoint=str(checkpoint),
                               family='abdomen_recovery', kind='region', arm=arm_name, seed=seed,
                               protocol_root=str(experiment), protocol_path=str(plan_path),
                               recovery_plan_path=str(plan_path), records_path=str(records_path),
                               model_protocol_path=str(reference_root / 'protocol.json'),
                               source_root=str(source_root), source_frozen=source_frozen, config=config,
                               source_files=plan.get('source_files', {}), task_ids=config.get('task_ids'), size=config['size'],
                               records_sha256=records_digest, protocol_sha256=plan.get('plan_sha256'),
                               expected_epoch=result.get('selected_epoch') if selection == 'best' else config['epochs'],
                               epochs_completed=result.get('epochs_completed', 0), training_epochs=config['epochs'],
                               comparability_group='validation-only-abdomen-recovery-' + str(config['size']),
                               status=status, reason=reason)


def _screen_job(experiment, project_root):
    screen_path = experiment / 'screen.json'
    screen = _read(screen_path)
    protocol = screen['protocol']
    checkpoint = Path(protocol['checkpoint'])
    protocol_root = checkpoint.parent.parent.parent
    original = _read(protocol_root / 'protocol.json')
    config = original.get('config', {})
    records_path = protocol_root / 'records.json'
    source_root, frozen = _source_root(protocol_root.parent.parent, protocol_root, project_root)
    identity = dict(run_identity=str(experiment), selection='screen80', split='val')
    superseded = (experiment / 'superseded.json').exists()
    return dict(**identity, job_id=_digest(identity), experiment=experiment.name, experiment_root=str(experiment),
                run_dir=str(experiment), checkpoint=str(checkpoint), checkpoint_sha256=protocol['checkpoint_sha256'],
                expected_epoch=protocol['checkpoint_epoch'], family='postprocess_screen', kind='region', arm=None,
                seed=protocol.get('seed', 42), protocol_root=str(experiment), protocol_path=str(screen_path),
                model_protocol_path=str(protocol_root / 'protocol.json'),
                records_path=str(records_path), config=config, task_ids=config.get('task_ids'), size=protocol['input_size'],
                source_root=str(source_root), source_frozen=frozen, source_files=original.get('source_files', {}),
                records_sha256=_records_info(records_path)[1], protocol_sha256=_digest(protocol),
                candidate_grid=protocol.get('candidate_grid', []), comparability_group='validation-exploratory-postprocess',
                status='excluded' if superseded else 'ready',
                reason='superseded invalid screening; not a valid result' if superseded else
                       'validation-only exploratory screening; replay each fixed candidate without tuning')


def build_inventory(runs_root=DEFAULT_RUNS, project_root=None, *, hash_checkpoints=False) -> dict:
    """Discover all medical experiment artifacts, including explicitly excluded ones.

    Refreshing this function is safe while training runs.  ``deferred`` jobs
    become ready only after their original training/evaluation barriers pass.
    Hashing all checkpoint bytes is optional because repeated background polling
    should not read tens of gigabytes.  Workers must hash a ready checkpoint when
    admitting it and reject a changed size/mtime/hash before evaluating.
    """
    root = Path(runs_root).resolve()
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    jobs, experiments = [], []
    for experiment in sorted(p for p in root.iterdir() if p.is_dir()):
        if experiment.name == 'radar_vla':
            continue
        entry = dict(experiment=experiment.name, path=str(experiment))
        start = len(jobs)
        try:
            plan = _read(experiment / 'plan.json')
            if plan.get('schema') == 'abdomen_recovery_v1':
                jobs.extend(_recovery_jobs(experiment, project))
            elif (experiment / 'screen.json').is_file():
                jobs.append(_screen_job(experiment, project))
                entry['category'] = 'derived_screening'
                entry['candidate_count'] = len(jobs[-1]['candidate_grid'])
            else:
                protocols = sorted(p for p in experiment.rglob('protocol.json')
                                   if 'source_snapshot' not in p.parts)
                train_protocols = [p for p in protocols if any(k in _read(p) for k in ('config', 'args'))]
                if train_protocols:
                    for protocol_path in train_protocols:
                        jobs.extend(_jobs_for_protocol(experiment, protocol_path.parent, _read(protocol_path), project))
                elif any(experiment.glob('*.pt')):
                    jobs.extend(_jobs_for_protocol(experiment, experiment, {}, project))
            if len(jobs) == start:
                if experiment.name.endswith(('-logs', '_logs')) or experiment.name == 'medical_priority_q03_20261011':
                    entry.update(category='operational_artifacts', status='excluded', reason='logs/scheduler artifacts; no trained model')
                elif 'union_eval' in experiment.name or 'audit' in experiment.name:
                    entry.update(category='derived_evaluation', status='excluded', reason='derived existing checkpoint evaluation; source models inventoried separately')
                elif 'preflight' in experiment.name or 'smoke' in experiment.name:
                    entry.update(category='engineering', status='excluded', reason='engineering checks; no scientific result checkpoint')
                else:
                    entry.update(category='no_checkpoint', status='unavailable', reason='no model checkpoint or supported training protocol found')
            else:
                entry.setdefault('category', 'training')
                entry['job_count'] = len(jobs) - start
                entry['job_status_counts'] = dict(Counter(j['status'] for j in jobs[start:]))
                if not any(Path(j['checkpoint']).is_file() for j in jobs[start:]):
                    entry.update(status='unavailable', reason='protocol exists but no model checkpoint was produced')
        except (KeyError, TypeError, ValueError, OSError) as error:
            # Report malformed/unrecognized artifacts explicitly without losing
            # the rest of the project inventory.
            del jobs[start:]
            entry.update(category='invalid_artifacts', status='unavailable', reason=f'{type(error).__name__}: {error}')
        experiments.append(entry)
    cache = {}
    if hash_checkpoints:
        for job in jobs:
            if job['status'] == 'ready':
                path = job['checkpoint']
                cache.setdefault(path, None)
                if cache[path] is None:
                    cache[path] = _file_sha256(Path(path))
                expected = job.get('checkpoint_sha256')
                if expected and cache[path] != expected:
                    job.update(status='unavailable', reason='checkpoint bytes differ from original recorded SHA-256')
                else:
                    job['checkpoint_sha256'] = cache[path]
    return dict(schema='medical_metric_inventory_v1', generated_at=datetime.now(timezone.utc).isoformat(),
                runs_root=str(root), project_root=str(project), jobs=jobs, experiments=experiments,
                counts=dict(Counter(job['status'] for job in jobs)),
                checkpoint_count=len({job['checkpoint'] for job in jobs if Path(job['checkpoint']).is_file()}),
                rules=dict(metrics=['DSC (%)', 'HD95 (pixel)', 'ASSD (pixel)'],
                           split_policy='preserve original records and test barriers; no resplitting',
                           pixel_grid='original experiment input resolution; different grids are not pooled',
                           selection_policy='evaluate existing selections; never reselect on new metrics'))


def discover_jobs(runs_root=DEFAULT_RUNS, project_root=None, *, hash_checkpoints=False) -> list[dict]:
    return build_inventory(runs_root, project_root, hash_checkpoints=hash_checkpoints)['jobs']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hash-checkpoints', action='store_true')
    args = parser.parse_args()
    result = build_inventory(args.runs_root, args.project_root, hash_checkpoints=args.hash_checkpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), counts=result['counts'], checkpoints=result['checkpoint_count'])))


if __name__ == '__main__':
    main()
