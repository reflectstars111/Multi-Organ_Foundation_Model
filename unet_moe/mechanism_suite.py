"""Frozen Q06 routing and Q07 expert-count mechanism experiments.

The experiments reuse Q05's data, seed42 image draws, C main loss, visual/FiLM
architecture and validation checkpoint rules. Stage-specific matrices have
independent completion/test barriers and immutable provenance.
"""
import argparse
from collections import Counter
from itertools import combinations
import gc
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
from .mechanism_models import create_model, common_state_dict, expert_count_for_arm
from . import refinement_suite as q05
from . import routing_suite as q02
from .organ_weighted_suite import (file_sha, state_digest, read_json, make_draw_plan,
                                   metric_scalars, moments)
from .organ_weighting import ARMS as LOSS_ARMS
from .organ_diagnostics import GUARDRAIL_CONFIG, guardrail_passes, evaluate

VERSION = 'q06_q07_mechanism_v1'
STAGES = {'q06': dict(models=['region'], arms=['control', 'routing'], seeds=[42]),
          'q07': dict(models=['task_id'], arms=['experts0', 'experts1', 'experts9'], seeds=[42])}
ARMS = ('control', 'routing', 'experts0', 'experts1', 'experts9')
RUNS_ROOT = Path('/ssd1/data/Multi-Organ_Foundation_Model/runs')
REFERENCE_ROOT = RUNS_ROOT / 'q05_visual_context_task_film_80_512_20261014'
REVIEW_ROOT = RUNS_ROOT / 'moe_followup_review_20261016'
RECORD_SHA = q05.RECORD_SHA
CONFIG = {key: value for key, value in q05.CONFIG.items()
          if key not in ('models', 'arms', 'seeds', 'architecture_comparison', 'auxiliary_in_both_arms')}
CONFIG.update(region_routing_weight=.1, region_routing_layer=3,
              region_routing_scope='annotated_positive_queries_per_physical_microbatch',
              region_routing_divergence='KL(uniform_experts || mean_positive_query_softmax)',
              seed_policy='future_experiments_seed42_only')
ARTIFACT_FIELDS = dict(q05.ARTIFACT_FIELDS)
SELECTION_RULE = dict(version='reuse_frozen_q05_validation_choice_v1',
    source='completed Q05 selection and its exact Q02 validation-only provenance chain',
    expected_main_loss='organ_exposure_fg_bg', test_used=False,
    reuse='one unchanged C choice for every arm; no outcome-based re-selection')
NUMERICAL_SCOPE = dict(version='native_updates_strict_resume_v1',
    native_training_updates=2, deterministic_continuation_updates=2,
    checkpoint_prediction_scope='strict_deterministic',
    production_training_scope='native_fp32_no_bitwise_continuation_claim',
    cublas_workspace_config=':4096:8')


def stage_config(stage):
    if stage not in STAGES:
        raise ValueError('Unknown mechanism stage')
    return {**CONFIG, **{k: list(v) for k, v in STAGES[stage].items()}}


def _self_hash(value, field):
    body = dict(value)
    sha = body.pop(field, None)
    if not sha or fair._digest(body) != sha:
        raise ValueError(f'Invalid provenance digest: {field}')
    return sha


def validate_q05_evidence(root, protocol):
    """Validate frozen validation-only review and recompute its routing trigger."""
    root = Path(root)
    for relative, sha in protocol['review_evidence_files'].items():
        if file_sha(root / relative) != sha:
            raise ValueError('Frozen Q05 review evidence changed')
    review = read_json(root / 'review_provenance/q05_validation_review.json')
    next_steps = read_json(root / 'review_provenance/next_steps.json')
    manifest = read_json(root / 'review_provenance/review_manifest.json')
    _self_hash(review, 'report_sha256')
    _self_hash(manifest, 'manifest_sha256')
    if (review.get('complete') is not True or review.get('split') != 'val'
            or review.get('test_metrics_used') is not False
            or review.get('protocol_sha256') != protocol['reference_protocol_sha256']
            or next_steps.get('test_metrics_used') is not False
            or next_steps.get('q05_review_sha256') != review['report_sha256']
            or next_steps.get('manifest_sha256') != manifest['manifest_sha256']
            or review.get('manifest_sha256') != manifest['manifest_sha256']
            or manifest['stages']['Q05']['protocol_sha256'] != protocol['reference_protocol_sha256']):
        raise ValueError('Q05 review is incomplete, not validation-only, or bound elsewhere')
    trigger = next_steps['q06']
    refined = {r['seed']: r for r in review['run_reviews']
               if r['kind'] == 'region' and r['arm'] == 'refined'}
    if set(refined) != {42, 43, 44}:
        raise ValueError('Historical three-seed trigger evidence is incomplete')
    evidence = []
    for seed, row in sorted(refined.items()):
        epochs = {str(epoch): row['routing'][str(epoch)]['layers'][-1]['all'] for epoch in (60, 80)}
        strict = []
        for epoch, route in epochs.items():
            counts = route['selected_counts']
            qualifies = (len(counts) == 9 and route['queries'] > 0
                         and sum(counts) == route['queries']
                         and sum(value > 0 for value in counts) == 1)
            if route['strictly_single_expert'] != qualifies:
                raise ValueError('Recorded routing trigger differs from actual counts')
            strict.append(qualifies)
        evidence.append(dict(seed=seed, strict_exclusivity_at_both_epochs=all(strict), epochs=epochs))
    qualifying = sum(row['strict_exclusivity_at_both_epochs'] for row in evidence)
    if (trigger.get('triggered') is not True or trigger.get('required_seeds') != 2
            or trigger.get('qualifying_seeds') != qualifying or qualifying < 2
            or trigger.get('evidence') != evidence):
        raise ValueError('Q06 trigger has not been proved by the frozen validation review')
    return dict(complete=True, test_used=False, qualifying_seeds=qualifying,
                review_sha256=review['report_sha256'])


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
        'run_organ_weighted.py', 'preflight_fair80.py', 'run_moe_mechanisms.py')]
    optional = root / 'scripts/benchmark_mechanisms.py'
    if optional.exists():
        paths.append(optional)
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
    root, reference = Path(args.output).resolve(), Path(args.reference_root).resolve()
    if root.exists():
        raise FileExistsError('Prepare requires a new experiment directory')
    stage = args.stage
    config = stage_config(stage)
    after_root = Path(args.after_root or reference).resolve()
    if root in (reference, after_root):
        raise ValueError('Output must be independent of reference and predecessor')
    reference_protocol = read_json(reference / 'protocol.json')
    verify_protocol_hash(reference_protocol, q05.VERSION)
    reference_selection = q05.load_selection(reference, reference_protocol)
    if reference_selection['main_loss_arm'] != SELECTION_RULE['expected_main_loss']:
        raise ValueError('Q05 frozen C main loss is required')
    reference_summary = read_json(reference / 'summary.json')
    if (reference_summary.get('complete') is not True
            or reference_summary.get('protocol_sha256') != reference_protocol['protocol_sha256']
            or reference_summary.get('main_selection_sha256') != reference_selection['selection_sha256']
            or len(reference_summary.get('runs', [])) != 12
            or any(r.get('epochs_completed') != 80 or r.get('test_evaluated') is not True
                   for r in reference_summary['runs'])):
        raise ValueError('Completed Q05 reference is required')
    after_protocol = read_json(after_root / 'protocol.json')
    verify_protocol_hash(after_protocol, q05.VERSION if stage == 'q06' else VERSION)
    if ((stage == 'q06' and after_root != reference)
            or (stage == 'q07' and (after_protocol.get('stage') != 'q06'
                or after_protocol.get('reference_protocol_sha256') != reference_protocol['protocol_sha256']))):
        raise ValueError('Q06 must follow Q05; Q07 must follow matching Q06')
    fields = ('size', 'base', 'epochs', 'batch_size', 'micro_batch_size', 'quota_per_domain',
              'workers', 'lr', 'weight_decay', 'experts', 'top_k', 'task_ids',
              'dtype', 'gradient_clip_norm', 'augmentation', 'lr_schedule', 'threshold',
              'optimizer', 'optimizer_betas', 'optimizer_eps', 'split_seed', 'split_policy', 'abdomen_split')
    if any(reference_protocol['config'].get(key) != config[key] for key in fields):
        raise ValueError('Reference Q05 training/data configuration differs')
    artifacts = {name: read_json(reference / name) for name in ARTIFACT_FIELDS}
    for name, key in ARTIFACT_FIELDS.items():
        if fair._digest(artifacts[name]) != reference_protocol[key]:
            raise ValueError(f'Reference {name} fingerprint differs')
    records = artifacts['records.json']
    if fair._digest(records) != RECORD_SHA:
        raise ValueError('Reference records differ from formal comparison')
    fair.validate_splits(records)
    if {r[key] for r in records for key in ('image', 'mask')} != set(artifacts['data_content.json']):
        raise ValueError('Data manifest differs from records')
    verify_content(artifacts['data_content.json'], records)
    dataset = fair.RegionDataset([r for r in records if r['split'] == 'train'], config['size'], config['task_ids'])
    if set(r[0]['dataset'] for r in dataset.groups) != set(fair.DOMAINS):
        raise ValueError('All nine training domains required')
    artifacts['sampling_plan.json'] = {'42': artifacts['sampling_plan.json']['42']}
    if artifacts['sampling_plan.json'] != make_draw_plan(dataset.groups, [42], 80, 500, 16):
        raise ValueError('Copied seed42 draws differ from the frozen paired image plan')
    initial, common, counts = {}, {}, {}
    for kind in config['models']:
        initial[kind], common[kind], counts[kind] = {'42': {}}, {}, {}
        common_sha = None
        for arm in config['arms']:
            fair._seed(42)
            model = create_model(kind, config['base'], config['top_k'], config['experts'], arm=arm)
            initial[kind]['42'][arm] = state_digest(model.state_dict())
            sha = state_digest(common_state_dict(model))
            if common_sha is not None and sha != common_sha:
                raise ValueError('Paired common initialization differs')
            common_sha = sha
            counts[kind][arm] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                     experts=expert_count_for_arm(kind, arm))
            if arm in ('control', 'routing', 'experts9') and initial[kind]['42'][arm] != reference_protocol['initialization_sha256'][kind]['42']['refined']:
                raise ValueError('New model initialization does not reproduce Q05 refined architecture')
            del model
        common[kind]['42'] = common_sha
    review_root = Path(args.review_root).resolve()
    review_files = {f'review_provenance/{name}': file_sha(review_root / name)
                   for name in ('q05_validation_review.json', 'next_steps.json', 'review_manifest.json')}
    provenance_files = {f'reference_provenance/{name}': file_sha(reference / name)
                        for name in ('protocol.json', 'main_loss_selection.json',
                                     'q02_protocol.json', 'q02_main_loss_selection.json')}
    sources = source_manifest(Path(__file__).resolve().parents[1])
    config['data_root'] = reference_protocol['config']['data_root']
    protocol = dict(protocol_version=VERSION, stage=stage, config=config, config_sha256=fair._digest(config),
        reference_root=str(reference), after_root=str(after_root),
        reference_protocol_sha256=reference_protocol['protocol_sha256'],
        reference_protocol_file_sha256=file_sha(reference / 'protocol.json'),
        after_protocol_sha256=after_protocol['protocol_sha256'],
        reference_selection_sha256=reference_selection['selection_sha256'],
        reference_selection_file_sha256=file_sha(reference / 'main_loss_selection.json'),
        reference_provenance_files=provenance_files, review_evidence_files=review_files,
        source_files=sources, source_sha256=fair._digest(sources),
        initialization_sha256=initial, common_initialization_sha256=common, model_counts=counts,
        guardrail=dict(GUARDRAIL_CONFIG), diagnostic_indices=artifacts['train_statistics.json']['diagnostic_indices'],
        main_loss_selection_rule=dict(SELECTION_RULE),
        budget=dict(images_per_epoch=4500, optimizer_steps_per_epoch=282, epochs=80,
                    runs=len(config['arms']), images_per_run=360000),
        architecture_policy=dict(region='Q05 refined visual foreground/background contrast; router never receives organ/domain ID',
            task_id='Q05 refined shared backbone/Task-FiLM; construct all 9 experts then retain 0/1/9',
            initialization='Q05 refined from-scratch initialization; never fine-tune completed Q05 weights'),
        auxiliary_policy=dict(region='main + 0.1 coarse; routing arm additionally 0.1 final-layer positive-query reverse KL',
            task_id='main + 0.01 original balance for 9 experts; balance omitted for 0/1 experts'),
        test_policy='all declared stage runs fully trained and verified before any stage test decode; no test tuning',
        selection_policy='original best validation positive structure macro; last; unchanged guardrail; no substitution',
        numerical_scope=dict(NUMERICAL_SCOPE),
        environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda))
    protocol.update({key: fair._digest(artifacts[name]) for name, key in ARTIFACT_FIELDS.items()})
    protocol['protocol_sha256'] = fair._digest(protocol)
    root.mkdir(parents=True)
    for name, artifact in artifacts.items():
        fair._write_json(root / name, artifact)
    for relative in review_files:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(review_root / destination.name, destination)
    for relative in provenance_files:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / destination.name, destination)
    validate_q05_evidence(root, protocol)
    fair._write_json(root / 'protocol.json', protocol)
    selection = dict(protocol_sha256=protocol['protocol_sha256'], rule=dict(SELECTION_RULE),
        reference_protocol_sha256=reference_protocol['protocol_sha256'],
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
            raise ValueError('Source changed during freeze')
    print(json.dumps(dict(prepared=str(root), stage=stage, protocol_sha256=protocol['protocol_sha256'],
                          state='prepared_not_enqueued')), flush=True)
    return protocol


def load_protocol(root, check_content=True):
    root = Path(root)
    protocol = read_json(root / 'protocol.json')
    verify_protocol_hash(protocol, VERSION)
    config = stage_config(protocol['stage'])
    if (any(protocol['config'].get(key) != value for key, value in config.items())
            or fair._digest(protocol['config']) != protocol['config_sha256']
            or protocol.get('main_loss_selection_rule') != SELECTION_RULE
            or protocol.get('numerical_scope') != NUMERICAL_SCOPE):
        raise ValueError('Frozen mechanism configuration differs')
    sources = source_manifest(Path(__file__).resolve().parents[1])
    if sources != protocol['source_files'] or fair._digest(sources) != protocol['source_sha256']:
        raise ValueError('Source changed; use the frozen source_snapshot')
    artifacts = {name: read_json(root / name) for name in ARTIFACT_FIELDS}
    for name, key in ARTIFACT_FIELDS.items():
        if fair._digest(artifacts[name]) != protocol[key]:
            raise ValueError(f'Frozen {name} changed')
    validate_q05_evidence(root, protocol)
    if check_content:
        verify_content(artifacts['data_content.json'], artifacts['records.json'])
    return protocol, artifacts['records.json'], artifacts['sampling_plan.json'], artifacts['train_statistics.json']


def load_selection(root, protocol):
    root = Path(root)
    for relative, sha in protocol['reference_provenance_files'].items():
        if file_sha(root / relative) != sha:
            raise ValueError('Copied Q05 selection provenance changed')
    reference = read_json(root / 'reference_provenance/protocol.json')
    verify_protocol_hash(reference, q05.VERSION)
    source = q05.load_selection(root / 'reference_provenance', reference)
    if (reference['protocol_sha256'] != protocol['reference_protocol_sha256']
            or source['selection_sha256'] != protocol['reference_selection_sha256']
            or source['main_loss_arm'] != SELECTION_RULE['expected_main_loss']):
        raise ValueError('Q05 reference selection is bound elsewhere')
    selection = read_json(root / 'main_loss_selection.json')
    _self_hash(selection, 'selection_sha256')
    expected = dict(protocol_sha256=protocol['protocol_sha256'], rule=SELECTION_RULE,
        reference_protocol_sha256=protocol['reference_protocol_sha256'],
        reference_selection_sha256=source['selection_sha256'],
        reference_selection_file_sha256=protocol['reference_selection_file_sha256'],
        main_loss_arm=source['main_loss_arm'], test_used=False)
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError('Main loss differs from immutable Q05 C selection')
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
        from .mechanism_objective import compute_objective
        loss, components = compute_objective(model, kind, image, target, args.arm, args.main_loss_arm, weights)
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
    if (kind, seed, arm) not in identities(protocol):
        raise ValueError('Run is outside frozen stage matrix')
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
    if require_test and protocol.get('stage') == 'q07':
        benchmark = read_json(folder / 'benchmark.json')
        expected = dict(schema='q07_cost_benchmark_v1', complete=True,
            protocol_sha256=protocol['protocol_sha256'], kind=kind, seed=seed, arm=arm,
            checkpoint_sha256=checkpoint_shas['best'], input_split='val',
            input_domain='AbdomenUS', query_counts=[5] * 4, dtype='float32', size=512)
        if any(benchmark.get(key) != value for key, value in expected.items()):
            raise ValueError('Q07 benchmark is incomplete or bound to a different checkpoint')
        workloads = benchmark.get('workloads', [])
        names = set(workloads) if isinstance(workloads, dict) else {row['workload'] for row in workloads}
        if names != {'single_task_batch1', 'complete_union_batch1', 'complete_union_batch4'}:
            raise ValueError('Q07 complete-query benchmark workloads differ')
    return result


def evaluate_test(args, protocol, records, draws):
    from .mechanism_evaluation import evaluate as evaluate_complete
    # Test records are not decoded until all stage runs pass full verification.
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
    if protocol['stage'] == 'q07':
        del model
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        from .mechanism_benchmark import benchmark_run
        benchmark_run(args.output, protocol, records, args.kind, args.seed, args.arm, args.device)
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


def paired_comparison(runs, selection='best', arms=None):
    arms = list(arms or sorted({r['arm'] for r in runs}))
    results = {}
    for kind in sorted({r['kind'] for r in runs}):
        by_arm = {arm: {r['seed']: r.get('test', {}).get(selection) for r in runs
                        if r['kind'] == kind and r['arm'] == arm} for arm in arms}
        comparisons = {}
        for left, right in combinations(arms, 2):
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
                  paired_test={name: paired_comparison(runs, name, protocol['config']['arms']) for name in ('best', 'last', 'guardrail')},
                  stage=protocol['stage'], evaluation_ready=True,
                  interpretation='single seed42 mechanism comparisons; sample SD is null; historically inspected test; '
                                 'guardrail unselected runs are missing candidates, never substituted')
    aggregate = {}
    for kind in protocol['config']['models']:
        aggregate[kind] = {}
        for arm in protocol['config']['arms']:
            aggregate[kind][arm] = {}
            for name in ('best', 'last', 'guardrail'):
                candidates = [r['test'][name] for r in runs if r['kind'] == kind and r['arm'] == arm
                              and r['test'][name] is not None]
                scalars = [metric_scalars(m) for m in candidates]
                keys = set().union(*(s.keys() for s in scalars))
                aggregate[kind][arm][name] = {key: moments([m[key] for m in scalars if key in m])
                                              for key in sorted(keys)}
    result['test_across_seeds'] = aggregate
    from .mechanism_evaluation import summarize as summarize_evaluation
    result['evaluation'] = summarize_evaluation(root, protocol)
    result['main_loss_selection'] = protocol['_selection']
    fair._write_json(Path(root) / 'summary.json', result)
    return result



def diagnose_run(model, kind, datasets, protocol, device, weights, arm):
    from .mechanism_diagnostics import diagnose, evaluate_routing
    report = diagnose(model, kind, datasets['train'], protocol['diagnostic_indices'], device,
                      protocol['_selection']['main_loss_arm'], weights, arm=arm)
    report['arm'] = arm
    report['objective_arm'] = arm
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
    if (args.kind, 42, args.arm) not in identities(protocol):
        raise ValueError('Preflight outside frozen stage matrix')
    c = protocol['config']
    cuda = torch.device(args.device).type == 'cuda'
    if cuda and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('CUDA preflight requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization')
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
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
                  effective_batch=c['batch_size'], micro_batch=c['micro_batch_size'], size=c['size'],
                  numerical_scope=dict(NUMERICAL_SCOPE), update_deterministic_modes=[],
                  update_cudnn_deterministic=[])
    try:
        torch.use_deterministic_algorithms(False)
        model = create_model(args.kind, c['base'], c['top_k'], c['experts'], arm=args.arm).to(args.device)
        if state_digest(model.state_dict()) != initial_digest(protocol, args.kind, batch['seed'], args.arm):
            raise ValueError('Preflight initialization differs from frozen plan')
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        run_args = argparse.Namespace(batch_size=c['batch_size'], device=args.device, arm=args.arm,
                                      main_loss_arm=protocol['_selection']['main_loss_arm'])
        updates = []
        for _ in range(2):
            report['update_deterministic_modes'].append(torch.are_deterministic_algorithms_enabled())
            report['update_cudnn_deterministic'].append(torch.backends.cudnn.deterministic)
            model.train()
            updates.append(train_effective_batches(model, args.kind, loader(dataset, c, 42, batch['indices']),
                                                    optimizer, run_args, weights, len(batch['indices'])))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
        report['update_deterministic_modes'].append(torch.are_deterministic_algorithms_enabled())
        report['update_cudnn_deterministic'].append(torch.backends.cudnn.deterministic)
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
        report['update_deterministic_modes'].append(torch.are_deterministic_algorithms_enabled())
        report['update_cudnn_deterministic'].append(torch.backends.cudnn.deterministic)
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
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
    fair._write_json(report_folder / f'{args.kind}_{args.arm}.json', report)
    print(json.dumps(report), flush=True)
    return report


def check_preflight(root, protocol, kind, arm):
    if (kind, 42, arm) not in identities(protocol):
        raise ValueError('Preflight is outside stage matrix')
    report = read_json(Path(root) / 'preflight' / f'{kind}_{arm}.json')
    expected = dict(protocol_sha256=protocol['protocol_sha256'], kind=kind, arm=arm, passed=True,
                    main_selection_sha256=protocol['_selection']['selection_sha256'],
                    main_loss_arm=protocol['_selection']['main_loss_arm'],
                    real_optimizer_updates=2, effective_batch=16, micro_batch=4, size=512,
                    real_query_images=32, strict_model_optimizer_rng_reload=True,
                    identical_reload_prediction=True, identical_resume_update=True,
                    continuation_verification_updates=2, continuation_sample_count=32, gpu_training_admissible=True,
                    numerical_scope=NUMERICAL_SCOPE, update_deterministic_modes=[False, False, True, True],
                    update_cudnn_deterministic=[True, True, True, True])
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
    parser.add_argument('--after-root', type=Path)
    parser.add_argument('--stage', choices=tuple(STAGES))
    parser.add_argument('--review-root', type=Path, default=REVIEW_ROOT)
    parser.add_argument('--kind', choices=('region', 'task_id'))
    parser.add_argument('--seed', type=int, choices=(42,))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--require-test', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args(argv)
    if args.action == 'prepare' and not args.stage:
        parser.error('--stage required for prepare')
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
    if args.action == 'preflight':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
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
