"""Q05 paired visual context and Task-FiLM refinement with fixed Q02 auxiliary objectives.

Both arms use Region coarse supervision / Task-ID weak balance. Only the
refined model differs. Q02 validation-selected main loss is copied and bound
without re-selection. Frozen snapshots keep Q02/Q04 and historical runs intact.
"""
import argparse
from collections import Counter
import fcntl
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
from .refinement_models import create_model
from . import routing_suite as q02
from .organ_weighted_suite import (file_sha, state_digest, read_json, make_draw_plan,
                                   metric_scalars, moments)
from .organ_weighting import ARMS as LOSS_ARMS
from .organ_diagnostics import GUARDRAIL_CONFIG, guardrail_passes, evaluate

VERSION = 'q05_moe_refinement_v1'
ARMS = ('control', 'refined')
RUNS_ROOT = Path('/ssd1/data/Multi-Organ_Foundation_Model/runs')
REFERENCE_ROOT = RUNS_ROOT / 'q02_routing_80_512_20261013'
AFTER_ROOT = REFERENCE_ROOT
RECORD_SHA = 'f621896f10234777d3067a019d96a7cb1dae3b6add305bca32815298744315e0'
CONFIG = dict(size=512, base=24, epochs=80, batch_size=16, micro_batch_size=4,
              quota_per_domain=500, workers=0, lr=1e-4, weight_decay=1e-4,
              experts=9, top_k=1, seeds=[42, 43, 44], models=['region', 'task_id'],
              arms=list(ARMS), split_seed=42, split_policy='trainmore_v6', abdomen_split='ct7_2',
              task_ids=list(fair.PAPER_ALIGNED_TASK_IDS), dtype='float32', gradient_clip_norm=5.,
              lr_schedule='constant', augmentation='none', threshold=.5, optimizer='AdamW',
              optimizer_betas=[.9, .999], optimizer_eps=1e-8,
              primary_selection='best_validation_macro_positive_dice',
              diagnostic_epochs=[0, 1, 10, 20, 40, 60, 80],
              region_coarse_weight=.1, task_id_balance_weight=.01,
              routing_diagnostics='full_validation_and_fixed_train_gradient_probes',
              architecture_comparison='refined_minus_control', auxiliary_in_both_arms=True)
ARTIFACT_FIELDS = {'records.json': 'record_sha256', 'sampling_plan.json': 'draws_sha256',
                   'data_content.json': 'content_sha256', 'train_statistics.json': 'statistics_sha256'}


SELECTION_RULE = dict(version='reuse_frozen_q02_validation_choice_v1',
    source='Q02 main_loss_selection.json, copied byte-for-byte',
    prerequisite='valid frozen Q02 protocol and completed validation-only main-loss choice',
    expected_main_loss='organ_exposure_fg_bg', test_used=False,
    reuse='one unchanged Q02 choice for all twelve Q05 runs; no outcome-based re-selection')


def verify_reference_selection(reference_protocol, selection):
    """Validate the existing Q02 choice using only its recorded validation proof."""
    q02.verify_protocol_hash(reference_protocol, 'q02_routing_v1')
    expected = dict(selection)
    digest = expected.pop('selection_sha256', None)
    if (not digest or fair._digest(expected) != digest
            or selection.get('protocol_sha256') != reference_protocol['protocol_sha256']
            or selection.get('reference_protocol_sha256') != reference_protocol.get('reference_protocol_sha256')
            or selection.get('rule') != reference_protocol.get('main_loss_selection_rule')
            or selection.get('rule') != q02.SELECTION_RULE
            or selection.get('test_used') is not False):
        raise ValueError('Q02 validation-only main-loss provenance is invalid')
    selected, candidates = q02.select_main_loss(selection['scoring_evidence'])
    if (selection.get('main_loss_arm') != selected or selection.get('candidates') != candidates
            or selected != SELECTION_RULE['expected_main_loss']):
        raise ValueError('Q05 requires the already frozen Q02 C main loss')
    return selection


def initial_digest(protocol, kind, seed, arm):
    return protocol['initialization_sha256'][kind][str(seed)][arm]


def identities(protocol):
    c = protocol['config']
    return [(kind, seed, arm) for seed in c['seeds'] for kind in c['models'] for arm in c['arms']]


def run_folder(root, kind, seed, arm):
    return Path(root) / arm / kind / f'seed_{seed}'


def source_manifest(root):
    root = Path(root)
    paths = sorted((root / 'unet_moe').rglob('*.py'))
    paths += [root / 'scripts' / name for name in (
        '__init__.py', 'run_moe_refinement.py', 'run_routing_optimization.py',
        'run_organ_weighted.py', 'preflight_fair80.py')]
    return {str(path.relative_to(root)): file_sha(path) for path in paths}


def verify_protocol_hash(protocol, version=None):
    expected = dict(protocol)
    digest = expected.pop('protocol_sha256', None)
    if not digest or fair._digest(expected) != digest or (version and protocol.get('protocol_version') != version):
        raise ValueError('Frozen protocol hash/version differs')


def verify_content(content, records=None, splits=('train', 'val')):
    paths = (set(content) if records is None else
             {row[key] for row in records if row['split'] in splits for key in ('image', 'mask')})
    for path in sorted(paths):
        recorded = content[path]
        if Path(path).stat().st_size != recorded['bytes'] or file_sha(path) != recorded['sha256']:
            raise ValueError(f'Data content changed: {path}')


def prepare(args):
    """Freeze Q02 inputs, its existing validation-only loss choice and new paired models."""
    root, reference = Path(args.output).resolve(), Path(args.reference_root).resolve()
    if root.exists():
        raise FileExistsError('Prepare requires a new experiment directory')
    reference_protocol = read_json(reference / 'protocol.json')
    verify_protocol_hash(reference_protocol, 'q02_routing_v1')
    after_root = Path(args.after_root).resolve()
    if after_root != reference or root == reference:
        raise ValueError('Q05 must use the same Q02 reference/predecessor and a new output root')
    after_protocol = read_json(after_root / 'protocol.json')
    verify_protocol_hash(after_protocol, 'q02_routing_v1')
    reference_selection = verify_reference_selection(reference_protocol, read_json(reference / 'main_loss_selection.json'))
    fields = ('size', 'base', 'epochs', 'batch_size', 'micro_batch_size', 'quota_per_domain',
              'workers', 'lr', 'weight_decay', 'experts', 'top_k', 'seeds', 'task_ids',
              'dtype', 'gradient_clip_norm', 'augmentation', 'lr_schedule', 'threshold',
              'optimizer', 'optimizer_betas', 'optimizer_eps', 'split_seed', 'split_policy', 'abdomen_split')
    if any(reference_protocol['config'].get(key) != CONFIG[key] for key in fields):
        raise ValueError('Reference Q02 training/data configuration differs')
    artifacts = {name: read_json(reference / name) for name in ARTIFACT_FIELDS}
    for name, key in ARTIFACT_FIELDS.items():
        if fair._digest(artifacts[name]) != reference_protocol[key]:
            raise ValueError(f'Reference {name} fingerprint differs')
    records, draws = artifacts['records.json'], artifacts['sampling_plan.json']
    if fair._digest(records) != RECORD_SHA:
        raise ValueError('Reference records differ from the formal 80-epoch comparison')
    fair.validate_splits(records)
    if {r[key] for r in records for key in ('image', 'mask')} != set(artifacts['data_content.json']):
        raise ValueError('Reference data content manifest differs from records')
    verify_content(artifacts['data_content.json'], records)
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'train'], CONFIG['size'], CONFIG['task_ids'])
    if set(r[0]['dataset'] for r in dataset.groups) != set(fair.DOMAINS):
        raise ValueError('All nine domains are required')
    if draws != make_draw_plan(dataset.groups, CONFIG['seeds'], CONFIG['epochs'],
                               CONFIG['quota_per_domain'], CONFIG['batch_size']):
        raise ValueError('Sampling plan differs from paired Q01 draws')
    initial, common = {}, {}
    for kind in CONFIG['models']:
        initial[kind], common[kind] = {}, {}
        for seed in CONFIG['seeds']:
            initial[kind][str(seed)] = {}
            fair._seed(seed)
            control = create_model(kind, CONFIG['base'], CONFIG['top_k'], CONFIG['experts'], arm='control')
            control_state = control.state_dict()
            control_sha = state_digest(control_state)
            if control_sha != reference_protocol['initialization_sha256'][kind][str(seed)]:
                raise ValueError('Control initialization no longer reproduces Q02')
            fair._seed(seed)
            refined = create_model(kind, CONFIG['base'], CONFIG['top_k'], CONFIG['experts'], arm='refined')
            refined_state = refined.state_dict()
            if not control_state.keys() <= refined_state.keys():
                raise ValueError('Refined model changed common parameter keys')
            if any(not torch.equal(value, refined_state[key]) for key, value in control_state.items()):
                raise ValueError('Paired common initialization differs')
            additions = refined_state.keys() - control_state.keys()
            if not additions or any(torch.count_nonzero(refined_state[key]).item() for key in additions):
                raise ValueError('Refined additions must be nonempty and identically zero initialized')
            initial[kind][str(seed)] = dict(control=control_sha, refined=state_digest(refined_state))
            common[kind][str(seed)] = control_sha
            del control, refined, control_state, refined_state
    sources = source_manifest(Path(__file__).resolve().parents[1])
    config = {**CONFIG, 'data_root': reference_protocol['config']['data_root']}
    protocol = dict(protocol_version=VERSION, config=config, config_sha256=fair._digest(config),
                    architecture_policy=dict(region='image-conditioned only: foreground/background visual contrast through zero-initialized CxC adapter; no task/domain/organ ID',
                                             task_id='four zero-initialized Task-FiLM adapters after MoE outputs; same oracle annotated task queries',
                                             control='original Q02 architecture with the same auxiliary objective',
                                             refined='only visual-contrast or Task-FiLM addition; common state identical at initialization'),
                    reference_root=str(reference), reference_protocol_sha256=reference_protocol['protocol_sha256'],
                    after_root=str(after_root), after_protocol_sha256=after_protocol['protocol_sha256'],
                    source_files=sources, source_sha256=fair._digest(sources),
                    initialization_sha256=initial, common_initialization_sha256=common, guardrail=dict(GUARDRAIL_CONFIG),
                    reference_selection_sha256=reference_selection['selection_sha256'],
                    reference_selection_file_sha256=file_sha(reference / 'main_loss_selection.json'),
                    reference_protocol_file_sha256=file_sha(reference / 'protocol.json'),
                    diagnostic_indices=artifacts['train_statistics.json']['diagnostic_indices'],
                    main_loss_selection_rule=dict(SELECTION_RULE),
                    budget=dict(images_per_epoch=4500, optimizer_steps_per_epoch=282, epochs=80,
                                runs=12, images_per_run=360000),
                    auxiliary_policy=dict(region='four upsampled coarse losses use frozen main weighting then layer mean',
                                          task_id='Switch balance over all queries in each physical microbatch; image-count accumulation',
                                          control='same fixed Q02 auxiliary objectives in BOTH arms; refined adds only the specified visual/Task-FiLM adapter'),
                    test_policy='all 12 complete and independently verified before any test',
                    selection_policy='best structure validation macro; epoch80; original Q01 guardrail best, never substituted',
                    environment=dict(python=sys.version, torch=str(torch.__version__),
                                     numpy=np.__version__, cuda=torch.version.cuda))
    protocol.update({key: fair._digest(artifacts[name]) for name, key in ARTIFACT_FIELDS.items()})
    protocol['protocol_sha256'] = fair._digest(protocol)
    root.mkdir(parents=True)
    for name, artifact in artifacts.items():
        fair._write_json(root / name, artifact)
    fair._write_json(root / 'protocol.json', protocol)
    shutil.copy2(reference / 'protocol.json', root / 'q02_protocol.json')
    shutil.copy2(reference / 'main_loss_selection.json', root / 'q02_main_loss_selection.json')
    selection = dict(protocol_sha256=protocol['protocol_sha256'],
                     rule=dict(SELECTION_RULE), reference_protocol_sha256=reference_protocol['protocol_sha256'],
                     reference_selection_sha256=reference_selection['selection_sha256'],
                     reference_selection_file_sha256=protocol['reference_selection_file_sha256'],
                     main_loss_arm=reference_selection['main_loss_arm'], test_used=False)
    selection['selection_sha256'] = fair._digest(selection)
    fair._write_json(root / 'main_loss_selection.json', selection)
    load_selection(root, protocol)
    source_root = Path(__file__).resolve().parents[1]
    for relative, digest in sources.items():
        destination = root / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
        if file_sha(destination) != digest:
            raise ValueError('Source changed while freezing snapshot')
    print(json.dumps(dict(prepared=str(root), protocol_sha256=protocol['protocol_sha256'],
                          state='prepared_not_enqueued')), flush=True)
    return protocol


def load_protocol(root, check_content=True):
    root = Path(root)
    protocol = read_json(root / 'protocol.json')
    verify_protocol_hash(protocol, VERSION)
    if (any(protocol['config'].get(key) != value for key, value in CONFIG.items()) or
            fair._digest(protocol['config']) != protocol['config_sha256'] or
            protocol.get('main_loss_selection_rule') != SELECTION_RULE):
        raise ValueError('Frozen Q05 configuration or selection rule differs')
    sources = source_manifest(Path(__file__).resolve().parents[1])
    if sources != protocol['source_files'] or fair._digest(sources) != protocol['source_sha256']:
        raise ValueError('Source changed; run the frozen source_snapshot entry point')
    artifacts = {name: read_json(root / name) for name in ARTIFACT_FIELDS}
    for name, key in ARTIFACT_FIELDS.items():
        if fair._digest(artifacts[name]) != protocol[key]:
            raise ValueError(f'{name} differs from frozen protocol')
    if check_content:
        verify_content(artifacts['data_content.json'], artifacts['records.json'])
    return protocol, artifacts['records.json'], artifacts['sampling_plan.json'], artifacts['train_statistics.json']


def load_selection(root, protocol):
    """Bind Q05 to exact copied Q02 protocol/selection; no re-selection from outcomes."""
    root = Path(root)
    if (file_sha(root / 'q02_protocol.json') != protocol['reference_protocol_file_sha256']
            or file_sha(root / 'q02_main_loss_selection.json') != protocol['reference_selection_file_sha256']):
        raise ValueError('Copied Q02 provenance file hash differs')
    reference = read_json(root / 'q02_protocol.json')
    source = verify_reference_selection(reference, read_json(root / 'q02_main_loss_selection.json'))
    if (reference['protocol_sha256'] != protocol['reference_protocol_sha256']
            or source['selection_sha256'] != protocol['reference_selection_sha256']):
        raise ValueError('Q02 provenance is bound to a different reference')
    selection = read_json(root / 'main_loss_selection.json')
    expected = dict(selection)
    digest = expected.pop('selection_sha256', None)
    fields = dict(protocol_sha256=protocol['protocol_sha256'], rule=SELECTION_RULE,
                  reference_protocol_sha256=protocol['reference_protocol_sha256'],
                  reference_selection_sha256=source['selection_sha256'],
                  reference_selection_file_sha256=protocol['reference_selection_file_sha256'],
                  main_loss_arm=source['main_loss_arm'], test_used=False)
    if (not digest or fair._digest(expected) != digest
            or any(selection.get(key) != value for key, value in fields.items())):
        raise ValueError('Main-loss selection is invalid or differs from frozen Q02 provenance')
    return selection


def with_selection(root, protocol):
    """Attach verified runtime state without modifying the immutable protocol file."""
    return {**protocol, '_selection': load_selection(root, protocol)}


def loader(dataset, config, seed, sampler=None):
    return torch.utils.data.DataLoader(dataset, batch_size=config['micro_batch_size'],
                                      sampler=sampler, shuffle=False, num_workers=0,
                                      generator=torch.Generator().manual_seed(seed))


def train_effective_batches(model, kind, batches, optimizer, args, weights, image_total):
    count = queries = steps = window_count = 0
    total_loss = 0.
    component_totals = Counter()
    for image, target in batches:
        image, target = image.to(args.device), target.to(args.device)
        if not target.ge(0).any((1, 2, 3)).all():
            raise ValueError('Training image without valid annotation')
        if window_count == 0:
            optimizer.zero_grad(set_to_none=True)
            window_size = min(args.batch_size, image_total - count)
        if window_size <= 0 or len(image) > window_size - window_count:
            raise ValueError('Microbatch crosses effective-batch boundary')
        from .routing_objective import compute_objective
        loss, components = compute_objective(model, kind, image, target, 'auxiliary', args.main_loss_arm, weights)
        for key, value in components.items():
            component_totals[key] += float(value.detach() if torch.is_tensor(value) else value) * len(image)
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
        del loss, components, image, target
    if count != image_total or window_count or steps != math.ceil(image_total / args.batch_size):
        raise ValueError('Training did not consume the exact paired image budget')
    return dict(train_loss=total_loss / count, sample_count=count,
                train_components={key: value / count for key, value in component_totals.items()},
                annotated_channel_queries=queries, optimizer_steps=steps)


def check_identity(artifact, protocol, kind, seed, arm):
    expected = dict(protocol_sha256=protocol['protocol_sha256'], kind=kind, seed=seed, arm=arm,
                    initialization_sha256=initial_digest(protocol, kind, seed, arm),
                    main_selection_sha256=protocol['_selection']['selection_sha256'],
                    main_loss_arm=protocol['_selection']['main_loss_arm'])
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
    model = create_model(kind, c['base'], c['top_k'], c['experts'], arm=arm).to(args.device)
    init_sha = state_digest(model.state_dict())
    identity = dict(protocol_sha256=protocol['protocol_sha256'], kind=kind, seed=seed, arm=arm,
                    initialization_sha256=init_sha,
                    main_selection_sha256=protocol['_selection']['selection_sha256'],
                    main_loss_arm=protocol['_selection']['main_loss_arm'])
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
    training_args = argparse.Namespace(batch_size=c['batch_size'], device=args.device, arm=arm,
                                       main_loss_arm=protocol['_selection']['main_loss_arm'])
    start_epoch = len(history) + 1
    if not history:
        report = diagnose_run(model, kind, datasets, protocol, args.device, weights, arm)
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
            report = diagnose_run(model, kind, datasets, protocol, args.device, weights, arm)
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
            for split in ('val', 'test'):
                report = read_json(folder / 'evaluation' / f'{name}_{split}.json')
                expected = dict(kind=kind, seed=seed, arm=arm, checkpoint_selection=name, split=split,
                                checkpoint_epoch=epoch, checkpoint_sha256=checkpoint_shas[name],
                                protocol_sha256=protocol['protocol_sha256'],
                                main_selection_sha256=protocol['_selection']['selection_sha256'],
                                main_loss_arm=protocol['_selection']['main_loss_arm'])
                if any(report.get(key) != value for key, value in expected.items()):
                    raise ValueError('Evaluation identity or selection differs')
                row_path = folder / 'evaluation' / f'{name}_{split}_per_image.jsonl'
                if report.get('per_image_file') != str(row_path) or report.get('per_image_sha256') != file_sha(row_path):
                    raise ValueError('Per-image metric artifact changed')
                if split == 'test' and (report['surface'] != metric['surface'] or
                                        any(metric.get(key) != value for key, value in report['diagnostics'].items())):
                    raise ValueError('Stored test metrics differ from full evaluation report')
    return result


def evaluate_test(args, protocol, records, draws):
    from .refinement_evaluation import evaluate as evaluate_complete
    # Test records are not decoded until all twelve runs pass full verification.
    for kind, seed, arm in identities(protocol):
        validate_run(args.output, protocol, draws, kind, seed, arm)
    verify_content(read_json(Path(args.output) / 'data_content.json'), records, splits=('test',))
    folder = run_folder(args.output, args.kind, args.seed, args.arm)
    result = read_json(folder / 'run_summary.json')
    c = protocol['config']
    model = create_model(args.kind, c['base'], c['top_k'], c['experts'], arm=args.arm).to(args.device)
    test = {}
    for name in ('best', 'last', 'guardrail'):
        if name == 'guardrail' and result['guardrail_epoch'] is None:
            test[name] = None
            continue
        path = folder / f'{name}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        check_identity(checkpoint, protocol, args.kind, args.seed, args.arm)
        model.load_state_dict(checkpoint['model'], strict=True)
        epoch = checkpoint['epoch']
        del checkpoint
        identity = dict(kind=args.kind, seed=args.seed, arm=args.arm, checkpoint_epoch=epoch,
                        checkpoint_selection=name, checkpoint_sha256=file_sha(path),
                        protocol_sha256=protocol['protocol_sha256'],
                        main_selection_sha256=protocol['_selection']['selection_sha256'],
                        main_loss_arm=protocol['_selection']['main_loss_arm'])
        for split in ('val', 'test'):
            dataset = fair.RegionDataset([r for r in records if r['split'] == split], c['size'], c['task_ids'])
            report = evaluate_complete(model, args.kind, loader(dataset, c, args.seed), args.device, groups=dataset.groups)
            rows = report.pop('rows')
            report.update(**identity, split=split)
            rows_path = folder / 'evaluation' / f'{name}_{split}_per_image.jsonl'
            fair._write_history(rows_path, rows)
            report['per_image_file'] = str(rows_path)
            report['per_image_sha256'] = file_sha(rows_path)
            fair._write_json(folder / 'evaluation' / f'{name}_{split}.json', report)
            if split == 'test':
                metric = {**report['diagnostics'], 'surface': report['surface'], **identity,
                          'evaluation_file': str(folder / 'evaluation' / f'{name}_{split}.json')}
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
                'dice' in key or 'dsc' in key.lower() or 'hd95' in key.lower() or 'assd' in key.lower()
                or key.endswith('_rate') or key.endswith('_fraction')):
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
        for left, right in ((ARMS[0], ARMS[1]),):
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
    result = dict(protocol_sha256=protocol['protocol_sha256'],
                  main_selection_sha256=protocol['_selection']['selection_sha256'],
                  main_loss_arm=protocol['_selection']['main_loss_arm'], runs=runs, complete=True,
                  paired_test={name: paired_comparison(runs, name) for name in ('best', 'last', 'guardrail')},
                  interpretation='paired architecture effects with identical fixed auxiliary losses; three seeds, historically inspected test; '
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
    from .refinement_evaluation import summarize as summarize_evaluation
    result['evaluation'] = summarize_evaluation(root, protocol)
    result['main_loss_selection'] = protocol['_selection']
    fair._write_json(Path(root) / 'summary.json', result)
    return result



def diagnose_run(model, kind, datasets, protocol, device, weights, arm):
    from .refinement_diagnostics import diagnose, evaluate_routing
    report = diagnose(model, kind, datasets['train'], protocol['diagnostic_indices'], device,
                      protocol['_selection']['main_loss_arm'], weights, arm=arm)
    report['arm'] = arm
    report['objective_arm'] = 'q02_auxiliary'
    report['validation'] = evaluate_routing(model, kind, loader(datasets['val'], protocol['config'], 42),
                                            device, groups=datasets['val'].groups)
    return report


def worst_planned_batch(groups, draws, config):
    """Find the true largest physical-query microbatch in the frozen plan."""
    best = None
    for seed in config['seeds']:
        for draw in draws[str(seed)]:
            for start in range(0, len(draw['indices']), config['batch_size']):
                indices = draw['indices'][start:start + config['batch_size']]
                queries = sum(len(groups[index]) for index in indices)
                micros = [sum(len(groups[index]) for index in indices[m:m + config['micro_batch_size']])
                          for m in range(0, len(indices), config['micro_batch_size'])]
                if best is None or (max(micros), queries) > (best['max_microbatch_queries'], best['annotated_queries']):
                    best = dict(seed=seed, epoch=draw['epoch'], batch_index=start // config['batch_size'],
                                indices=indices, images=len(indices), annotated_queries=queries,
                                max_microbatch_queries=max(micros), microbatch_queries=micros,
                                sample_sha256=fair._digest([groups[index][0]['image'] for index in indices]))
    if best is None:
        raise ValueError('No frozen training batches')
    return best


def nested_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_state(item) for item in value)
    return value


def preflight(args, protocol, records, draws, weights):
    """Two real effective updates including optimizer residency and strict reload.

    CPU checks are stored separately and never admit GPU training. Both arms
    require their own CUDA report, tied to the finalized main-loss selection.
    """
    c = protocol['config']
    cuda = torch.device(args.device).type == 'cuda'
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'train'], c['size'], c['task_ids'])
    batch = worst_planned_batch(dataset.groups, draws, c)
    if len(batch['indices']) != c['batch_size']:
        raise ValueError('Preflight requires a complete effective batch')
    fair._seed(batch['seed'])
    if cuda:
        torch.cuda.mem_get_info(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    report_folder = Path(args.output) / ('preflight' if cuda else 'preflight_cpu')
    report_folder.mkdir(parents=True, exist_ok=True)
    temporary = report_folder / f'.{args.kind}_{args.arm}_reload.pt'
    identity = dict(protocol_sha256=protocol['protocol_sha256'], kind=args.kind, arm=args.arm,
                    main_selection_sha256=protocol['_selection']['selection_sha256'],
                    main_loss_arm=protocol['_selection']['main_loss_arm'])
    report = dict(**identity, passed=False, device=str(args.device), selection=batch,
                  effective_batch=c['batch_size'], micro_batch=c['micro_batch_size'], size=c['size'])
    try:
        model = create_model(args.kind, c['base'], c['top_k'], c['experts'], arm=args.arm).to(args.device)
        if state_digest(model.state_dict()) != initial_digest(protocol, args.kind, batch['seed'], args.arm):
            raise ValueError('Preflight initialization differs from frozen plan')
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        run_args = argparse.Namespace(batch_size=c['batch_size'], device=args.device, arm=args.arm,
                                      main_loss_arm=protocol['_selection']['main_loss_arm'])
        updates = []
        for _ in range(2):
            model.train()
            updates.append(train_effective_batches(model, args.kind, loader(dataset, c, 42, batch['indices']),
                                                    optimizer, run_args, weights, len(batch['indices'])))
        image, target = next(iter(loader(dataset, c, 42, batch['indices'])))
        image, target = image.to(args.device), target.to(args.device)
        model.eval()
        with torch.no_grad():
            expected = fair.forward_model(model, args.kind, image, target)['logits'].cpu()
        digest = state_digest(model.state_dict())
        fair._save_checkpoint(temporary, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=fair._rng_state()))
        # One continuation step is executed on both branches, proving that
        # restoration resumes the AdamW trajectory, not only predictions.
        model.train()
        continued = train_effective_batches(model, args.kind, loader(dataset, c, 43, batch['indices']),
                                            optimizer, run_args, weights, len(batch['indices']))
        continued_model_sha = state_digest(model.state_dict())
        continued_optimizer = _cpu_state(optimizer.state_dict())
        continued_rng = _cpu_state(fair._rng_state())
        del model, optimizer
        checkpoint = torch.load(temporary, map_location='cpu', weights_only=True)
        restored = create_model(args.kind, c['base'], c['top_k'], c['experts'], arm=args.arm).to(args.device)
        restored.load_state_dict(checkpoint['model'], strict=True)
        optimizer = torch.optim.AdamW(restored.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        fair._restore_rng(checkpoint['rng'])
        if (state_digest(restored.state_dict()) != digest or
                not nested_equal(optimizer.state_dict(), checkpoint['optimizer']) or
                not nested_equal(fair._rng_state(), checkpoint['rng'])):
            raise ValueError('Model/optimizer/RNG state failed strict reload')
        restored.eval()
        with torch.no_grad():
            actual = fair.forward_model(restored, args.kind, image, target)['logits'].cpu()
        if not torch.equal(actual, expected):
            raise ValueError('Reloaded prediction differs')
        restored.train()
        resumed = train_effective_batches(restored, args.kind, loader(dataset, c, 43, batch['indices']),
                                          optimizer, run_args, weights, len(batch['indices']))
        if (state_digest(restored.state_dict()) != continued_model_sha
                or not nested_equal(optimizer.state_dict(), continued_optimizer)
                or not nested_equal(fair._rng_state(), continued_rng)
                or not nested_equal(continued, resumed)):
            raise ValueError('Resumed optimizer continuation differs')
        fair._synchronize(args.device)
        peak = math.ceil(torch.cuda.max_memory_reserved(args.device) / 2 ** 20) if cuda else None
        report.update(passed=True, updates=updates, real_optimizer_updates=sum(row['optimizer_steps'] for row in updates),
                      real_query_images=sum(row['sample_count'] for row in updates),
                      strict_model_optimizer_rng_reload=True, identical_reload_prediction=True,
                      identical_resume_update=True, continuation_verification_updates=2,
                      continuation_sample_count=2 * len(batch['indices']),
                      peak_reserved_mib=peak,
                      peak_allocated_mib=math.ceil(torch.cuda.max_memory_allocated(args.device) / 2 ** 20) if cuda else None,
                      required_free_mib=peak + 3072 if cuda else None,
                      gpu_training_admissible=cuda)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        fair._write_json(report_folder / f'{args.kind}_{args.arm}.json', report)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    fair._write_json(report_folder / f'{args.kind}_{args.arm}.json', report)
    print(json.dumps(report), flush=True)
    return report


def check_preflight(root, protocol, kind, arm):
    report = read_json(Path(root) / 'preflight' / f'{kind}_{arm}.json')
    expected = dict(protocol_sha256=protocol['protocol_sha256'], kind=kind, arm=arm, passed=True,
                    main_selection_sha256=protocol['_selection']['selection_sha256'],
                    main_loss_arm=protocol['_selection']['main_loss_arm'],
                    real_optimizer_updates=2, effective_batch=16, micro_batch=4, size=512,
                    real_query_images=32, strict_model_optimizer_rng_reload=True,
                    identical_reload_prediction=True, identical_resume_update=True,
                    continuation_verification_updates=2, gpu_training_admissible=True)
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('CUDA preflight is missing or differs from run protocol/selection')
    if not isinstance(report.get('required_free_mib'), int) or report['required_free_mib'] <= 0:
        raise ValueError('CUDA memory admission requirement is invalid')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action', choices=('prepare', 'preflight', 'train', 'verify', 'evaluate', 'summarize'), required=True)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--after-root', type=Path, default=AFTER_ROOT)
    parser.add_argument('--kind', choices=CONFIG['models'])
    parser.add_argument('--seed', type=int, choices=CONFIG['seeds'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--require-test', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args(argv)
    if args.action not in ('prepare', 'summarize') and (not args.kind or not args.arm):
        parser.error('--kind and --arm required')
    if args.action in ('train', 'verify', 'evaluate') and args.seed is None:
        parser.error('--seed required')
    if not 0 <= args.stop_after_epoch <= 80 or (args.stop_after_epoch and args.action != 'train'):
        parser.error('Invalid stop epoch')
    if args.require_test and args.action != 'verify':
        parser.error('--require-test only applies to verify')
    if args.resume and args.action != 'train':
        parser.error('--resume only applies to train')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.action == 'prepare':
        prepare(args)
        return
    protocol, records, draws, weights = load_protocol(args.output, check_content=args.action not in ('verify', 'summarize'))
    protocol = with_selection(args.output, protocol)
    if args.action == 'verify':
        result = validate_run(args.output, protocol, draws, args.kind, args.seed, args.arm, args.require_test)
        print(json.dumps(dict(valid=True, kind=args.kind, seed=args.seed, arm=args.arm,
                              epochs_completed=result['epochs_completed'], test_evaluated=result['test_evaluated'])))
    elif args.action == 'summarize':
        summarize(args.output, protocol, draws)
    elif args.action == 'preflight':
        preflight(args, protocol, records, draws, weights)
    elif args.action == 'evaluate':
        evaluate_test(args, protocol, records, draws)
    else:
        if torch.device(args.device).type == 'cuda':
            check_preflight(args.output, protocol, args.kind, args.arm)
        run_model(args, protocol, records, draws, weights)


if __name__ == '__main__':
    main()
