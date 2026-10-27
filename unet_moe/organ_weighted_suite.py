"""Frozen Q01 organ-weighted paired study; independent of historical trainers.

Prepare once, preflight all architectures, then run through the GPU controller.
Test mode requires all 27 training/validation runs to be complete.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from . import fair_suite as fair
from .organ_weighting import ARMS, collect_statistics, weighted_loss
from .organ_diagnostics import GUARDRAIL_CONFIG, diagnose_gradients, evaluate, guardrail_passes
from .sampling import domain_draws


VERSION = 'q01_organ_weighted_v1'
RECORD_SHA = 'f621896f10234777d3067a019d96a7cb1dae3b6add305bca32815298744315e0'
RUNS_ROOT = Path('/ssd1/data/Multi-Organ_Foundation_Model/runs')
PREREQUISITES = (
    'fair_80_base24_batch16_512_20261001_fix',
    'single_channel_union_80_base24_batch16_512_20261001_fix',
    'fair_union_eval_80_base24_batch16_512_20261001_fix',
    'fair_union_eval_512_20260930',
)
CONFIG = dict(size=512, base=24, epochs=80, batch_size=16, micro_batch_size=4,
              quota_per_domain=500, workers=0, lr=1e-4, weight_decay=1e-4,
              experts=9, top_k=1, seeds=[42, 43, 44], models=['shared', 'region', 'task_id'],
              arms=list(ARMS), split_seed=42, split_policy='trainmore_v6', abdomen_split='ct7_2',
              task_ids=list(fair.PAPER_ALIGNED_TASK_IDS), dtype='float32', gradient_clip_norm=5.,
              lr_schedule='constant', augmentation='none', auxiliary_loss_weights=0.,
              threshold=.5, optimizer='AdamW', optimizer_betas=[.9, .999], optimizer_eps=1e-8,
              primary_selection='best_validation_macro_positive_dice',
              diagnostic_epochs=[0, 1, 10, 20, 40, 60, 80])


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def make_draw_plan(groups, seeds, epochs, quota, batch_size):
    plan = {}
    for seed in seeds:
        rows = []
        for epoch in range(1, epochs + 1):
            indices = domain_draws(groups, quota, seed, epoch)
            rows.append(dict(epoch=epoch, indices=indices,
                             sample_sha256=fair._digest([groups[i][0]['image'] for i in indices]),
                             effective_batch_sizes=[min(batch_size, len(indices) - start)
                                                    for start in range(0, len(indices), batch_size)]))
        plan[str(seed)] = rows
    return plan


def identities(protocol):
    c = protocol['config']
    return [(kind, seed, arm) for seed in c['seeds'] for kind in c['models'] for arm in c['arms']]


def run_folder(root, kind, seed, arm):
    return Path(root) / arm / kind / f'seed_{seed}'


def source_manifest(root):
    paths = sorted((root / 'unet_moe').rglob('*.py'))
    paths += [root / 'scripts' / name for name in ('run_organ_weighted.py', 'preflight_fair80.py')]
    return {str(path.relative_to(root)): file_sha(path) for path in paths}


def check_prerequisites(root):
    evidence = {}
    for name in PREREQUISITES:
        path = root / name / 'summary.json'
        summary = read_json(path)
        runs = summary.get('runs', [])
        if 'union_eval' in name:
            p = summary.get('protocol', {})
            wanted = {(kind, seed) for kind in CONFIG['models'] for seed in CONFIG['seeds']}
            fixed = dict(split='test', fair_record_sha256=RECORD_SHA, image_size=512, threshold=.5,
                         images=1618)
            if (len(runs) != 9 or {(r.get('kind'), r.get('seed')) for r in runs} != wanted
                    or any(p.get(key) != value for key, value in fixed.items())
                    or any(not r.get('metrics') for r in runs)):
                raise ValueError(f'Incomplete prerequisite union evaluation: {name}')
        else:
            count = 3 if name.startswith('single') else 9
            if len(runs) != count or any(r.get('epochs_completed') != 80 or
                                        r.get('test_evaluated') is not True for r in runs):
                raise ValueError(f'Incomplete prerequisite training/test: {name}')
            if summary.get('record_sha256') != RECORD_SHA:
                raise ValueError(f'Prerequisite records differ: {name}')
        evidence[name] = {'summary': str(path), 'sha256': file_sha(path)}
    return evidence


def prepare(args):
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError('Prepare requires a new experiment directory')
    previous = check_prerequisites(args.prerequisite_root)
    records, skipped = fair.build_records(args.data_root, 42, 'ct7_2', None, 'trainmore_v6')
    fair.validate_splits(records)
    records = sorted((r for r in records if r['task'] in fair.PAPER_ALIGNED_TASK_IDS),
                     key=lambda r: json.dumps(r, sort_keys=True))
    if fair._digest(records) != RECORD_SHA:
        raise ValueError('Q01 records do not match the frozen 80-epoch comparison')
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'train'], 512,
                                 fair.PAPER_ALIGNED_TASK_IDS)
    statistics = collect_statistics(dataset)
    if set(statistics['domain_image_counts']) != set(fair.DOMAINS) or statistics['all_ignored_images']:
        raise ValueError('Require all nine domains and valid labels on every training image')
    draws = make_draw_plan(dataset.groups, CONFIG['seeds'], 80, 500, 16)
    files = sorted({r[key] for r in records for key in ('image', 'mask')})
    content = {path: dict(sha256=file_sha(path), bytes=Path(path).stat().st_size) for path in files}
    initial = {}
    for kind in CONFIG['models']:
        initial[kind] = {}
        for seed in CONFIG['seeds']:
            fair._seed(seed)
            model = fair.create_model(kind, 24, 1, 9)
            initial[kind][str(seed)] = state_digest(model.state_dict())
            del model
    source_root = Path(__file__).resolve().parents[1]
    sources = source_manifest(source_root)
    config = {**CONFIG, 'data_root': str(args.data_root.resolve())}
    protocol = dict(protocol_version=VERSION, config=config,
                    config_sha256=fair._digest(config), record_sha256=RECORD_SHA,
                    source_files=sources, source_sha256=fair._digest(sources),
                    content_sha256=fair._digest(content), statistics_sha256=fair._digest(statistics),
                    draws_sha256=fair._digest(draws), initialization_sha256=initial,
                    guardrail=dict(GUARDRAIL_CONFIG),
                    diagnostic_indices=statistics['diagnostic_indices'],
                    budget=dict(images_per_epoch=4500, optimizer_steps_per_epoch=282,
                                epochs=80, runs=27, images_per_run=360000),
                    prerequisites=previous, skipped=skipped,
                    environment=dict(python=sys.version, torch=str(torch.__version__),
                                     numpy=np.__version__, cuda=torch.version.cuda),
                    input_difference={'task_id': 'oracle annotated task ID; repeated image queries',
                                      'shared': 'image only', 'region': 'image only'},
                    test_policy='all 27 training/validation selections complete before any Q01 test; '
                                'historically inspected test, not untouched external validation',
                    weights_policy='train-only fixed analytic weights; no validation hyperparameter search')
    protocol['protocol_sha256'] = fair._digest(protocol)
    root.mkdir(parents=True)
    for name, data in [('records.json', records), ('train_statistics.json', statistics),
                       ('sampling_plan.json', draws), ('data_content.json', content),
                       ('protocol.json', protocol)]:
        fair._write_json(root / name, data)
    snapshot = root / 'source_snapshot'
    for relative, digest in sources.items():
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
        if file_sha(destination) != digest:
            raise ValueError('Source changed while freezing snapshot')
    print(json.dumps(dict(prepared=str(root), protocol_sha256=protocol['protocol_sha256'],
                          train_images=len(dataset), probe_images=len(protocol['diagnostic_indices']))), flush=True)
    return protocol


def load_protocol(root, check_content=True):
    root = Path(root)
    protocol = read_json(root / 'protocol.json')
    expected = dict(protocol)
    digest = expected.pop('protocol_sha256')
    if fair._digest(expected) != digest or protocol['protocol_version'] != VERSION:
        raise ValueError('Frozen protocol hash/version differs')
    if any(protocol['config'].get(key) != value for key, value in CONFIG.items()):
        raise ValueError('Frozen Q01 configuration differs')
    sources = source_manifest(Path(__file__).resolve().parents[1])
    if sources != protocol['source_files'] or fair._digest(sources) != protocol['source_sha256']:
        raise ValueError('Source changed; run the frozen source_snapshot entry point')
    artifacts = [('records.json', 'record_sha256'), ('train_statistics.json', 'statistics_sha256'),
                 ('sampling_plan.json', 'draws_sha256'), ('data_content.json', 'content_sha256')]
    data = {}
    for name, key in artifacts:
        data[name] = read_json(root / name)
        if fair._digest(data[name]) != protocol[key]:
            raise ValueError(f'{name} differs from frozen protocol')
    if check_content:
        for path, recorded in data['data_content.json'].items():
            if Path(path).stat().st_size != recorded['bytes'] or file_sha(path) != recorded['sha256']:
                raise ValueError(f'Data content changed: {path}')
    return protocol, data['records.json'], data['sampling_plan.json'], data['train_statistics.json']


def loader(dataset, config, seed, sampler=None):
    return torch.utils.data.DataLoader(dataset, batch_size=config['micro_batch_size'],
                                      sampler=sampler, shuffle=False, num_workers=0,
                                      generator=torch.Generator().manual_seed(seed))


def train_effective_batches(model, kind, batches, optimizer, args, weights, image_total):
    count = queries = steps = window_count = 0
    total_loss = 0.
    for image, target in batches:
        image, target = image.to(args.device), target.to(args.device)
        if not target.ge(0).any((1, 2, 3)).all():
            raise ValueError('Training image without valid annotation')
        if window_count == 0:
            optimizer.zero_grad(set_to_none=True)
            window_size = min(args.batch_size, image_total - count)
        if window_size <= 0 or len(image) > window_size - window_count:
            raise ValueError('Microbatch crosses effective-batch boundary')
        output = fair.forward_model(model, kind, image, target)
        loss = weighted_loss(output['logits'], target, args.arm, weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite weighted training loss')
        (loss * (len(image) / window_size)).backward()
        total_loss += float(loss.detach()) * len(image)
        count += len(image)
        queries += int(target.ge(0).any((2, 3)).sum())
        window_count += len(image)
        if window_count == window_size:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            window_count = 0
        del output, loss, image, target
    if count != image_total or window_count or steps != math.ceil(image_total / args.batch_size):
        raise ValueError('Training did not consume the exact paired image budget')
    return dict(train_loss=total_loss / count, sample_count=count,
                annotated_channel_queries=queries, optimizer_steps=steps)


def check_identity(artifact, protocol, kind, seed, arm):
    expected = dict(protocol_sha256=protocol['protocol_sha256'], kind=kind, seed=seed, arm=arm,
                    initialization_sha256=protocol['initialization_sha256'][kind][str(seed)])
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError('Run identity, initialization, or frozen protocol differs')


def check_history(history, protocol, draws, kind, seed, arm):
    if (len(history) > len(draws[str(seed)]) or
            [r['epoch'] for r in history] != list(range(1, len(history) + 1))):
        raise ValueError('History is not a contiguous paired epoch sequence')
    for row, draw in zip(history, draws[str(seed)]):
        check_identity(row, protocol, kind, seed, arm)
        if (row['sample_sha256'] != draw['sample_sha256'] or row['sample_count'] != len(draw['indices'])
                or row['optimizer_steps'] != len(draw['effective_batch_sizes'])
                or not math.isfinite(row['macro_positive_dice'])):
            raise ValueError('History differs from paired draw/update budget')


def run_model(args, protocol, records, draws, weights):
    c = protocol['config']
    kind, seed, arm = args.kind, args.seed, args.arm
    if (kind, seed, arm) not in identities(protocol):
        raise ValueError('Run is outside the frozen matrix')
    folder = run_folder(args.output, kind, seed, arm)
    if folder.exists() and not args.resume:
        raise FileExistsError('Existing run requires --resume')
    folder.mkdir(parents=True, exist_ok=True)
    fair._seed(seed)
    model = fair.create_model(kind, c['base'], c['top_k'], c['experts']).to(args.device)
    init_sha = state_digest(model.state_dict())
    identity = dict(protocol_sha256=protocol['protocol_sha256'], kind=kind, seed=seed, arm=arm,
                    initialization_sha256=init_sha)
    check_identity(identity, protocol, kind, seed, arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    history, best_score, best_epoch, guard_score, guard_epoch, elapsed = [], -math.inf, None, -math.inf, None, 0.
    if (folder / 'last.pt').exists():
        checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        check_identity(checkpoint, protocol, kind, seed, arm)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']
        if [r['epoch'] for r in history] != list(range(1, checkpoint['epoch'] + 1)):
            raise ValueError('Checkpoint history is not contiguous')
        check_history(history, protocol, draws, kind, seed, arm)
        best_score, best_epoch = checkpoint['best_score'], checkpoint['best_epoch']
        guard_score, guard_epoch = checkpoint['guardrail_score'], checkpoint['guardrail_epoch']
        elapsed = checkpoint['elapsed_seconds']
        for name, epoch in [('best', best_epoch), ('guardrail', guard_epoch)]:
            if epoch == checkpoint['epoch']:
                fair._save_checkpoint(folder / f'{name}.pt', checkpoint)
            elif epoch is not None and not (folder / f'{name}.pt').is_file():
                raise FileNotFoundError(f'Missing selected {name} checkpoint')
        fair._restore_rng(checkpoint['rng'])
        fair._write_history(folder / 'metrics.jsonl', history)
        del checkpoint
    elif any((folder / name).exists() for name in ('best.pt', 'guardrail.pt', 'metrics.jsonl')):
        raise ValueError('Run artifacts exist without last checkpoint')
    datasets = {split: fair.RegionDataset([r for r in records if r['split'] == split], c['size'],
                                          c['task_ids']) for split in ('train', 'val')}
    val_loader = loader(datasets['val'], c, seed)
    training_args = argparse.Namespace(batch_size=c['batch_size'], device=args.device, arm=arm)
    start_epoch = len(history) + 1
    if not history:
        report = diagnose_gradients(model, kind, datasets['train'], protocol['diagnostic_indices'],
                                    args.device, arm, weights)
        fair._write_json(folder / 'diagnostics' / 'epoch_000.json', {**identity, 'epoch': 0, **report})
    final = min(c['epochs'], args.stop_after_epoch or c['epochs'])
    for epoch in range(start_epoch, final + 1):
        draw = draws[str(seed)][epoch - 1]
        actual_hash = fair._digest([datasets['train'].groups[i][0]['image'] for i in draw['indices']])
        if actual_hash != draw['sample_sha256']:
            raise ValueError('Draw no longer refers to the same training images')
        batches = loader(datasets['train'], c, seed + epoch, draw['indices'])
        model.train()
        fair._synchronize(args.device)
        started = time.perf_counter()
        train = train_effective_batches(model, kind, batches, optimizer, training_args, weights, len(draw['indices']))
        fair._synchronize(args.device)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        metrics = evaluate(model, kind, val_loader, args.device, groups=datasets['val'].groups)
        fair._synchronize(args.device)
        validation_seconds = time.perf_counter() - started
        score = metrics['macro_positive_dice']
        if score is None or not math.isfinite(score):
            raise ValueError('Validation score is not finite')
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        passed = guardrail_passes(metrics, protocol['guardrail'])
        guard_improved = passed and score > guard_score
        if guard_improved:
            guard_score, guard_epoch = score, epoch
        diagnostic_seconds = 0.
        if epoch in c['diagnostic_epochs']:
            started = time.perf_counter()
            report = diagnose_gradients(model, kind, datasets['train'], protocol['diagnostic_indices'],
                                        args.device, arm, weights)
            fair._write_json(folder / 'diagnostics' / f'epoch_{epoch:03d}.json',
                             {**identity, 'epoch': epoch, **report})
            diagnostic_seconds = time.perf_counter() - started
        elapsed += train_seconds + validation_seconds + diagnostic_seconds
        metrics.update(**identity, **train, epoch=epoch, lr=c['lr'], sample_sha256=actual_hash,
                       train_seconds=train_seconds, validation_seconds=validation_seconds,
                       diagnostic_seconds=diagnostic_seconds, elapsed_seconds=elapsed,
                       guardrail_passed=passed, physical_micro_batches=math.ceil(len(draw['indices']) / c['micro_batch_size']))
        history.append(metrics)
        checkpoint = dict(**identity, protocol_version=VERSION, config=c, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), rng=fair._rng_state(), epoch=epoch,
                          history=history, metrics=metrics, best_score=best_score, best_epoch=best_epoch,
                          guardrail_score=guard_score, guardrail_epoch=guard_epoch, elapsed_seconds=elapsed)
        fair._save_checkpoint(folder / 'last.pt', checkpoint)
        if improved:
            fair._save_checkpoint(folder / 'best.pt', checkpoint)
        if guard_improved:
            fair._save_checkpoint(folder / 'guardrail.pt', checkpoint)
        fair._write_history(folder / 'metrics.jsonl', history)
        result = dict(**identity, epochs_completed=epoch, best_epoch=best_epoch,
                      guardrail_epoch=guard_epoch, guardrail_selected=guard_epoch is not None,
                      best_validation_macro_positive_dice=best_score,
                      elapsed_seconds=elapsed, test_evaluated=False,
                      parameter_count=sum(p.numel() for p in model.parameters()))
        fair._write_json(folder / 'run_summary.json', result)
        print(json.dumps(dict(kind=kind, arm=arm, seed=seed, epoch=epoch, **train,
                              val_macro_positive_dice=score,
                              val_abdomen_macro_positive_dice=metrics['abdomen_macro_positive_dice'],
                              guardrail_passed=passed, train_seconds=train_seconds)), flush=True)
    # Recover a missing/stale summary after an interrupted atomic checkpoint write.
    if history:
        result = dict(**identity, epochs_completed=len(history), best_epoch=best_epoch,
                      guardrail_epoch=guard_epoch, guardrail_selected=guard_epoch is not None,
                      best_validation_macro_positive_dice=best_score, elapsed_seconds=elapsed,
                      test_evaluated=False, parameter_count=sum(p.numel() for p in model.parameters()))
        existing = read_json(folder / 'run_summary.json') if (folder / 'run_summary.json').exists() else {}
        if existing.get('test_evaluated') and existing.get('epochs_completed') == len(history):
            result = existing
        fair._write_json(folder / 'run_summary.json', result)
    return result


def validate_run(root, protocol, draws, kind, seed, arm, require_test=False):
    folder = run_folder(root, kind, seed, arm)
    result = read_json(folder / 'run_summary.json')
    check_identity(result, protocol, kind, seed, arm)
    epochs = protocol['config']['epochs']
    if result['epochs_completed'] != epochs:
        raise ValueError('Run training is incomplete')
    history = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in history] != list(range(1, epochs + 1)):
        raise ValueError('Incomplete epoch history')
    check_history(history, protocol, draws, kind, seed, arm)
    selected = {'last': epochs, 'best': result['best_epoch'], 'guardrail': result['guardrail_epoch']}
    checkpoint_shas = {}
    for name, epoch in selected.items():
        if epoch is None:
            if (folder / f'{name}.pt').exists():
                raise ValueError('Unselected guardrail checkpoint unexpectedly exists')
            continue
        path = folder / f'{name}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        check_identity(checkpoint, protocol, kind, seed, arm)
        if checkpoint['epoch'] != epoch or checkpoint['history'] != history[:epoch]:
            raise ValueError(f'{name} checkpoint does not match recorded history')
        if name == 'last' and (checkpoint['best_epoch'] != result['best_epoch'] or
                               checkpoint['guardrail_epoch'] != result['guardrail_epoch']):
            raise ValueError('Selected epoch differs from final checkpoint')
        checkpoint_shas[name] = file_sha(path)
        del checkpoint
    expected_best = max(history, key=lambda r: r['macro_positive_dice'])['epoch']
    guard = [r for r in history if guardrail_passes(r, protocol['guardrail'])]
    expected_guard = max(guard, key=lambda r: r['macro_positive_dice'])['epoch'] if guard else None
    if result['best_epoch'] != expected_best or result['guardrail_epoch'] != expected_guard:
        raise ValueError('Checkpoint selection violates frozen validation rules')
    for epoch in protocol['config']['diagnostic_epochs']:
        diagnostic = read_json(folder / 'diagnostics' / f'epoch_{epoch:03d}.json')
        check_identity(diagnostic, protocol, kind, seed, arm)
        if (diagnostic['epoch'] != epoch or diagnostic.get('probe_indices') != protocol['diagnostic_indices']
                or diagnostic.get('probe_images') != len(protocol['diagnostic_indices'])):
            raise ValueError('Diagnostic epoch or fixed probe plan differs')
    if require_test:
        if result.get('test_evaluated') is not True:
            raise ValueError('Test not evaluated')
        for name, epoch in selected.items():
            metric = result.get('test', {}).get(name)
            if epoch is None:
                if metric is not None:
                    raise ValueError('Unselected guardrail has test metrics')
                continue
            if (not metric or metric.get('checkpoint_epoch') != epoch or
                    metric.get('checkpoint_sha256') != checkpoint_shas[name] or
                    metric.get('protocol_sha256') != protocol['protocol_sha256'] or
                    metric != read_json(folder / f'test_{name}.json')):
                raise ValueError('Test result does not match selected checkpoint/protocol')
    return result


def evaluate_test(args, protocol, records, draws):
    # An authoritative barrier, also enforced for direct CLI invocation.
    for kind, seed, arm in identities(protocol):
        validate_run(args.output, protocol, draws, kind, seed, arm)
    folder = run_folder(args.output, args.kind, args.seed, args.arm)
    result = read_json(folder / 'run_summary.json')
    c = protocol['config']
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'test'], c['size'], c['task_ids'])
    batches = loader(dataset, c, args.seed)
    model = fair.create_model(args.kind, c['base'], c['top_k'], c['experts']).to(args.device)
    test = {}
    for name in ('best', 'last', 'guardrail'):
        if name == 'guardrail' and result['guardrail_epoch'] is None:
            test[name] = None
            continue
        path = folder / f'{name}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        check_identity(checkpoint, protocol, args.kind, args.seed, args.arm)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        del checkpoint
        metric = evaluate(model, args.kind, batches, args.device, groups=dataset.groups)
        metric.update(checkpoint_epoch=epoch, checkpoint_selection=name,
                      checkpoint_sha256=file_sha(path), protocol_sha256=protocol['protocol_sha256'])
        test[name] = metric
        fair._write_json(folder / f'test_{name}.json', metric)
    result.update(test_evaluated=True, test=test)
    fair._write_json(folder / 'run_summary.json', result)
    return result


def metric_scalars(value, prefix=''):
    result = {}
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(metric_scalars(item, path))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and (
                'dice' in key or key.endswith('_rate') or key.endswith('_fraction')):
            result[path] = item
    return result


def moments(values):
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)


def paired_comparison(runs, selection='best'):
    results = {}
    for kind in sorted({r['kind'] for r in runs}):
        by_arm = {arm: {r['seed']: r.get('test', {}).get(selection) for r in runs
                        if r['kind'] == kind and r['arm'] == arm} for arm in ARMS}
        comparisons = {}
        for left, right in ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[2])):
            pairs = []
            for seed in sorted(set(by_arm[left]) & set(by_arm[right])):
                if by_arm[left][seed] is not None and by_arm[right][seed] is not None:
                    a, b = metric_scalars(by_arm[left][seed]), metric_scalars(by_arm[right][seed])
                    pairs.append((seed, {key: b[key] - a[key] for key in a.keys() & b.keys()}))
            fields = sorted(set().union(*(row.keys() for _, row in pairs)))
            comparisons[f'{right}_minus_{left}'] = {
                key: {**moments([row[key] for _, row in pairs if key in row]),
                      'pairs': [dict(seed=seed, delta=row[key]) for seed, row in pairs if key in row]}
                for key in fields}
        results[kind] = comparisons
    return results


def summarize(root, protocol, draws):
    runs = [validate_run(root, protocol, draws, kind, seed, arm, True)
            for kind, seed, arm in identities(protocol)]
    result = dict(protocol_sha256=protocol['protocol_sha256'], runs=runs, complete=True,
                  paired_test={name: paired_comparison(runs, name) for name in ('best', 'last', 'guardrail')},
                  interpretation='paired loss effects; three seeds, historically inspected test; '
                                 'guardrail unselected runs are missing candidates, never substituted')
    aggregate = {}
    for kind in protocol['config']['models']:
        aggregate[kind] = {}
        for arm in ARMS:
            aggregate[kind][arm] = {}
            for name in ('best', 'last', 'guardrail'):
                candidates = [r['test'][name] for r in runs if r['kind'] == kind and r['arm'] == arm
                              and r['test'][name] is not None]
                scalars = [metric_scalars(m) for m in candidates]
                keys = set().union(*(s.keys() for s in scalars))
                aggregate[kind][arm][name] = {key: moments([m[key] for m in scalars if key in m])
                                              for key in sorted(keys)}
    result['test_across_seeds'] = aggregate
    fair._write_json(Path(root) / 'summary.json', result)
    return result


def preflight(args, protocol, records, weights):
    from scripts.preflight_fair80 import worst_batch
    c = protocol['config']
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'train'], c['size'], c['task_ids'])
    batch = worst_batch(dataset.groups, seeds=c['seeds'], epochs=c['epochs'], quota=c['quota_per_domain'],
                        batch_size=c['batch_size'], micro_batch_size=c['micro_batch_size'])
    fair._seed(batch['seed'])
    torch.cuda.mem_get_info(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    model = fair.create_model(args.kind, c['base'], 1, 9).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    run_args = argparse.Namespace(batch_size=c['batch_size'], device=args.device, arm=ARMS[2])
    for _ in range(2):
        model.train()
        report = train_effective_batches(model, args.kind, loader(dataset, c, 42, batch['indices']),
                                         optimizer, run_args, weights, len(batch['indices']))
    diagnose_gradients(model, args.kind, dataset, protocol['diagnostic_indices'],
                       args.device, ARMS[2], weights)
    torch.cuda.synchronize(args.device)
    peak = math.ceil(torch.cuda.max_memory_reserved(args.device) / 2 ** 20)
    report.update(kind=args.kind, protocol_sha256=protocol['protocol_sha256'], selection=batch,
                  passed=True, optimizer_updates=2,
                  peak_reserved_mib=peak,
                  peak_allocated_mib=math.ceil(torch.cuda.max_memory_allocated(args.device) / 2 ** 20),
                  required_free_mib=peak + 3072)
    fair._write_json(args.output / 'preflight' / f'{args.kind}.json', report)
    print(json.dumps(report), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--verify-run', action='store_true')
    modes.add_argument('--summarize', action='store_true')
    parser.add_argument('--data-root', type=Path, default=Path('/ssd1/data/Multi-Organ_Foundation_Model/pointed_data'))
    parser.add_argument('--prerequisite-root', type=Path, default=RUNS_ROOT)
    parser.add_argument('--kind', choices=CONFIG['models'])
    parser.add_argument('--seed', type=int, choices=CONFIG['seeds'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args(argv)
    if not args.prepare and not args.summarize and not args.kind:
        parser.error('--kind required')
    if not any((args.prepare, args.preflight, args.summarize)) and (args.seed is None or args.arm is None):
        parser.error('--seed and --arm required')
    if not 0 <= args.stop_after_epoch <= 80 or (args.evaluate_test and args.stop_after_epoch):
        parser.error('Invalid stop epoch or test combined with stopping early')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.prepare:
        prepare(args)
        return
    protocol, records, draws, weights = load_protocol(args.output, check_content=not args.verify_run)
    if args.verify_run:
        result = validate_run(args.output, protocol, draws, args.kind, args.seed, args.arm, args.evaluate_test)
        print(json.dumps(dict(valid=True, kind=args.kind, seed=args.seed, arm=args.arm,
                              epochs_completed=result['epochs_completed'], test_evaluated=result['test_evaluated'])))
    elif args.summarize:
        summarize(args.output, protocol, draws)
    elif args.preflight:
        preflight(args, protocol, records, weights)
    elif args.evaluate_test:
        evaluate_test(args, protocol, records, draws)
    else:
        run_model(args, protocol, records, draws, weights)


if __name__ == '__main__':
    main()
