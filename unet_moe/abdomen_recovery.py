"""Paired, validation-only Region fine-tuning experiments at 512 x 512.

Every arm starts from the same completed fair-suite Region checkpoint for its
seed. The fair-suite sources and artifacts are read but never modified. The
training test set is never opened by this module.

Prepare the frozen plan after the nine fair-suite runs and their tests finish::

    python -m unet_moe.abdomen_recovery --fair-root FAIR --output RECOVERY --prepare

Run (or resume) the eighteen fine-tunes on a physical GPU::

    python -m unet_moe.abdomen_recovery --fair-root FAIR --output RECOVERY --gpu 1
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from scripts import fair_parallel
from . import fair_suite
from .data import TASKS
from .empty_loss import empty_top_fraction_loss
from .region import DOMAINS, PAPER_ALIGNED_TASK_IDS, RegionDataset, masked_loss
from .sampling import domain_draws


VERSION = 'abdomen_recovery_v1'
SEEDS = (42, 43, 44)
EPOCHS = 8
QUOTA = 128
BATCH_SIZE = 4
SIZE = 512
BASE = 16
TOP_K = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EMPTY_FRACTION = .01
MINIMUM_FREE_MIB = 22 * 1024
ABDOMEN_TASKS = (13, 14, 16, 18, 20)
RARE_TASKS = (16, 20)
ARMS = (
    dict(name='control', empty_weight=0., rare_positive_weight=1., abdomen_sampling_weight=1.),
    dict(name='empty_0p002', empty_weight=.002, rare_positive_weight=1., abdomen_sampling_weight=1.),
    dict(name='empty_0p01', empty_weight=.01, rare_positive_weight=1., abdomen_sampling_weight=1.),
    dict(name='rare_positive_x2', empty_weight=0., rare_positive_weight=2., abdomen_sampling_weight=1.),
    dict(name='abdomen_positive_sampling_x3', empty_weight=0., rare_positive_weight=1., abdomen_sampling_weight=3.),
    dict(name='rare_positive_x2_empty_0p002', empty_weight=.002, rare_positive_weight=2., abdomen_sampling_weight=1.),
)
SOURCE_NAMES = ('fair_suite.py', 'baseline.py', 'model.py', 'region.py', 'normalization.py', 'data.py',
                'sampling.py', 'schedule.py', 'empty_loss.py')
SELECTION_GUARDS = dict(overall_positive_dice_decline=.015,
                        abdomen_empty_fp_area_increase=.015,
                        abdomen_empty_fp_rate_increase=.05,
                        abdomen_zero_overlap_increase=.02,
                        kidney_empty_fp_area_increase=.005,
                        kidney_positive_dice_decline=.02)
COMPARISON_GUARDS = dict(min_mean_abdomen_positive_dice_gain=.01,
                         max_mean_overall_positive_dice_decline=.01,
                         max_mean_abdomen_empty_fp_area_increase=0.,
                         max_mean_abdomen_empty_fp_rate_increase=.03,
                         max_mean_abdomen_zero_overlap_increase=.01,
                         max_individual_seed_abdomen_positive_dice_decline=.01,
                         max_mean_liver_empty_fp_rate_increase=.03,
                         max_mean_kidney_empty_fp_area_increase=.002,
                         max_mean_kidney_positive_dice_decline=.02)


def digest(value):
    return fair_suite._digest(value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    fair_suite._write_json(path, value)


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def train_val_content_digest(records):
    """Freeze all train/validation bytes without opening any test file."""
    paths = sorted({row[key] for row in records if row['split'] in ('train', 'val')
                    for key in ('image', 'mask')})
    if not paths:
        raise ValueError('No train/validation files to hash')
    value = hashlib.sha256()
    for path in paths:
        value.update(path.encode('utf-8'))
        value.update(b'\0')
        value.update(file_sha256(path).encode('ascii'))
        value.update(b'\n')
    return dict(files=len(paths), sha256=value.hexdigest())


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def completed_fair(fair_root):
    """Verify the immutable fair plan, both test markers, and root summary.

    Reading test result metadata here gates on completion; no test image or mask
    is opened, and no test metric enters any treatment or selection decision.
    """
    fair_root = Path(fair_root).resolve()
    raw = read_json(fair_root / 'parallel_plan.json')
    plan = fair_parallel.load_plan(fair_root, raw['common_config']['data_root'])
    metadata = {gpu: fair_parallel.validate_marker(plan, gpu, 'test') for gpu in ('0', '1')}
    expected = fair_parallel.aggregate_shards(plan, metadata)
    summary = read_json(fair_root / 'summary.json')
    if (summary.get('runs') != expected['runs'] or
            summary.get('record_sha256') != expected['record_sha256'] or
            summary.get('source_sha256') != expected['source_sha256'] or
            summary.get('parallel', {}).get('training_barrier_verified') is not True or
            summary.get('test_evaluation_requested') is not True):
        raise ValueError('Fair root summary does not match two verified completed test shards')
    for gpu in ('0', '1'):
        status = read_json(fair_root / f'worker_gpu{gpu}.json')
        if status.get('state') != 'complete' or status.get('plan_sha256') != plan['plan_sha256']:
            raise RuntimeError(f'Fair GPU {gpu} has not reported complete')
    return plan, summary


def fair_records_and_checkpoints(fair_plan):
    shard = fair_parallel.shard_path(fair_plan, '1')
    records = read_json(shard / 'records.json')
    protocol = read_json(shard / 'protocol.json')
    if digest(records) != protocol['record_sha256']:
        raise ValueError('Fair Region records differ from its protocol')
    runs = {}
    for seed in SEEDS:
        path = shard / 'region' / f'seed_{seed}' / 'best.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing fair Region best checkpoint: {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        runs[str(seed)] = dict(path=str(path.resolve()), sha256=file_sha256(path),
                              best_epoch=checkpoint['epoch'],
                              best_validation_macro_positive_dice=
                                  checkpoint['metrics']['macro_positive_dice'])
    return records, protocol, runs


def abdomen_positive_map(groups):
    """Read training masks only; mark images containing vessel or spleen."""
    mapping, source_hashes = {}, {}
    for rows in groups:
        if rows[0]['dataset'] != 'AbdomenUS':
            continue
        image = rows[0]['image']
        rare_rows = [row for row in rows if row['task'] in RARE_TASKS]
        positive = False
        for row in rare_rows:
            mask = row['mask']
            if mask not in source_hashes:
                source_hashes[mask] = file_sha256(mask)
            with Image.open(mask) as stream:
                indexed = np.asarray(stream)
            if (indexed.ndim != 2 or row['class_id'] != row['task'] - 12 or
                    not set(np.unique(indexed)).issubset(set(range(9)) | {255})):
                raise ValueError(f'Invalid abdominal indexed training mask or class ID: {mask}')
            positive |= bool(np.any(indexed == row['class_id']))
        mapping[image] = positive
    if not mapping or not any(mapping.values()) or all(mapping.values()):
        raise ValueError('Abdomen rare-positive sampling has no positive/negative training strata')
    return mapping, source_hashes


def prepare_plan(fair_root, output):
    fair_plan, fair_summary = completed_fair(fair_root)
    records, fair_protocol, initial = fair_records_and_checkpoints(fair_plan)
    kept = [row for row in records if row['split'] in ('train', 'val')]
    if not kept or len(kept) == len(records):
        raise ValueError('Expected training/validation records and a held-out test split')
    train = RegionDataset([row for row in kept if row['split'] == 'train'], SIZE, PAPER_ALIGNED_TASK_IDS)
    val = RegionDataset([row for row in kept if row['split'] == 'val'], SIZE, PAPER_ALIGNED_TASK_IDS)
    if Counter(group[0]['dataset'] for group in train.groups) != Counter(fair_protocol['split_images']['train']):
        raise ValueError('Training image counts differ from fair protocol')
    if Counter(group[0]['dataset'] for group in val.groups) != Counter(fair_protocol['split_images']['val']):
        raise ValueError('Validation image counts differ from fair protocol')
    positive, mask_hashes = abdomen_positive_map(train.groups)
    source = Path(__file__).parent
    source_files = {name: file_sha256(source / name) for name in SOURCE_NAMES}
    if source_files != fair_plan['source_files']:
        raise ValueError('Fair training source has changed since the frozen comparison')
    plan = dict(schema=VERSION, fair_root=str(Path(fair_root).resolve()),
                output=str(Path(output).resolve()), fair_plan_sha256=fair_plan['plan_sha256'],
                fair_record_sha256=fair_protocol['record_sha256'],
                fair_source_sha256=fair_plan['source_sha256'],
                fair_summary_sha256=digest(fair_summary),
                train_val_record_sha256=digest(kept),
                train_val_content=train_val_content_digest(kept),
                source_files=source_files, extension_sha256=file_sha256(__file__),
                initial_checkpoints=initial, seeds=list(SEEDS), arms=list(ARMS),
                design=dict(size=SIZE, base=BASE, top_k=TOP_K, batch_size=BATCH_SIZE,
                            epochs=EPOCHS, quota_per_domain=QUOTA, workers=2,
                            images_per_epoch=len(DOMAINS) * QUOTA,
                            optimizer_steps_per_epoch=math.ceil(len(DOMAINS) * QUOTA / BATCH_SIZE),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                            empty_fraction=EMPTY_FRACTION, gradient_clip_norm=5.,
                            all_arms_start_from='each seed fair Region best.pt; AdamW optimizer reset',
                            checkpoint_selection='max abdomen validation positive Dice among epochs 0..8 satisfying fixed baseline guards',
                            random_seed_policy='same seed and epoch train draws across arms except declared abdomen sampling arm',
                            test_policy='no test image or mask opened; no test metric used for model selection',
                            selection_guards=SELECTION_GUARDS,
                            comparison_guards=COMPARISON_GUARDS,
                            mask_positive_definition='indexed original training mask has class 4 vessel or class 8 spleen',
                            abdomen_positive_images=sum(positive.values()),
                            abdomen_train_images=len(positive),
                            abdomen_positive_mapping_sha256=digest(positive),
                            abdomen_mask_sha256=digest(mask_hashes),
                            minimum_free_mib=MINIMUM_FREE_MIB))
    plan['plan_sha256'] = digest(plan)
    output = Path(output).resolve()
    if output.exists():
        existing = output / 'plan.json'
        if existing.is_file():
            if read_json(existing) != plan:
                raise ValueError('Existing recovery output differs from this frozen plan')
        else:
            allowed_controller_files = {'controller_status.json', 'recovery.log'}
            unexpected = {item.name for item in output.iterdir()} - allowed_controller_files
            if unexpected:
                raise ValueError(f'Unplanned artifacts precede recovery plan: {sorted(unexpected)}')
            write_json(existing, plan)
    else:
        output.mkdir(parents=True)
        write_json(output / 'plan.json', plan)
    return plan


def load_plan(fair_root, output):
    output = Path(output).resolve()
    plan = read_json(output / 'plan.json')
    unsigned = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    if plan.get('plan_sha256') != digest(unsigned) or plan.get('schema') != VERSION:
        raise ValueError('Recovery plan hash or schema differs')
    if plan != prepare_plan(fair_root, output):
        raise ValueError('Recovery plan differs from current fair artifacts or source')
    return plan


def load_datasets(plan):
    fair_plan = read_json(Path(plan['fair_root']) / 'parallel_plan.json')
    records = read_json(fair_parallel.shard_path(fair_plan, '1') / 'records.json')
    kept = [row for row in records if row['split'] in ('train', 'val')]
    if digest(records) != plan['fair_record_sha256'] or digest(kept) != plan['train_val_record_sha256']:
        raise ValueError('Record metadata changed since recovery preparation')
    train = RegionDataset([row for row in kept if row['split'] == 'train'], SIZE, PAPER_ALIGNED_TASK_IDS)
    val = RegionDataset([row for row in kept if row['split'] == 'val'], SIZE, PAPER_ALIGNED_TASK_IDS)
    positive, hashes = abdomen_positive_map(train.groups)
    if (digest(positive) != plan['design']['abdomen_positive_mapping_sha256'] or
            digest(hashes) != plan['design']['abdomen_mask_sha256']):
        raise ValueError('Abdominal sampling masks changed since recovery preparation')
    return train, val, positive


def verify_train_val_contents(plan):
    fair_plan = read_json(Path(plan['fair_root']) / 'parallel_plan.json')
    records = read_json(fair_parallel.shard_path(fair_plan, '1') / 'records.json')
    if train_val_content_digest(records) != plan['train_val_content']:
        raise ValueError('Train/validation file contents changed during the recovery screen')


def sampled_indices(groups, positive_by_image, arm, seed, epoch, quota=QUOTA):
    """Keep all non-abdomen indices identical, even in the sampling arm."""
    indices = domain_draws(groups, quota, seed + 271828, epoch)
    if arm['abdomen_sampling_weight'] == 1.:
        return indices
    abdomen = [index for index, rows in enumerate(groups) if rows[0]['dataset'] == 'AbdomenUS']
    weights = [arm['abdomen_sampling_weight'] if positive_by_image[groups[index][0]['image']] else 1.
               for index in abdomen]
    rng = random.Random(seed + 271828 + 100003 * epoch + 7919 + 197)
    replacements = iter(rng.choices(abdomen, weights=weights, k=quota))
    result = [next(replacements) if groups[index][0]['dataset'] == 'AbdomenUS' else index
              for index in indices]
    if sum(groups[index][0]['dataset'] == 'AbdomenUS' for index in result) != quota:
        raise AssertionError('Abdomen sampling changed the fixed domain quota')
    return result


def objective(output, target, arm):
    if arm['rare_positive_weight'] == 1.:
        loss = fair_suite.common_loss(output['logits'], target)
    else:
        positive_weights = torch.ones(len(PAPER_ALIGNED_TASK_IDS), device=target.device)
        for task in RARE_TASKS:
            positive_weights[PAPER_ALIGNED_TASK_IDS.index(task)] = arm['rare_positive_weight']
        loss = masked_loss(output['logits'], target, positive_weights)
    if arm['empty_weight']:
        abdominal_channels = [PAPER_ALIGNED_TASK_IDS.index(task) for task in ABDOMEN_TASKS]
        loss = loss + arm['empty_weight'] * empty_top_fraction_loss(
            output['logits'], target, EMPTY_FRACTION, abdominal_channels)
    return loss


def abdomen_metrics(metrics):
    per = metrics['per_structure']
    rows = [per[TASKS[task]] for task in ABDOMEN_TASKS]
    def mean(field):
        values = [row[field] for row in rows if row[field] is not None]
        if not values:
            raise ValueError(f'Validation has no abdominal {field} denominator')
        return statistics.mean(values)
    return dict(overall_positive_dice=metrics['macro_positive_dice'],
                abdomen_positive_dice=mean('positive_dice'),
                abdomen_empty_fp_rate=mean('empty_false_positive_rate'),
                abdomen_empty_fp_area=mean('empty_mean_false_positive_area_fraction'),
                abdomen_zero_overlap=mean('positive_zero_overlap_rate'),
                liver_empty_fp_rate=per['abdomen_liver']['empty_false_positive_rate'],
                kidney_empty_fp_area=per['abdomen_kidney']['empty_mean_false_positive_area_fraction'],
                kidney_positive_dice=per['abdomen_kidney']['positive_dice'],
                per_structure={TASKS[task]: per[TASKS[task]] for task in ABDOMEN_TASKS})


@torch.no_grad()
def validation_ct_diagnostics(model, validation, device):
    """Describe the two already-used validation CTs, never an independent holdout."""
    abdomen_records = [row for rows in validation.groups if rows[0]['dataset'] == 'AbdomenUS'
                       for row in rows]
    abdomen = RegionDataset(abdomen_records, SIZE, PAPER_ALIGNED_TASK_IDS)
    loader = DataLoader(abdomen, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    vessel_channel = PAPER_ALIGNED_TASK_IDS.index(16)
    kidney_channel = PAPER_ALIGNED_TASK_IDS.index(14)
    liver_channel = PAPER_ALIGNED_TASK_IDS.index(13)
    totals = {}
    model.eval()
    offset = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predicted = model(images)['logits'].sigmoid() >= .5
        for local in range(len(images)):
            ct = abdomen.groups[offset + local][0]['group'].split('/')[-1]
            row = totals.setdefault(ct, dict(images=0, vessel_positive=0,
                vessel_zero_overlap=0, vessel_no_prediction=0,
                vessel_recall_sum=0., vessel_dice_sum=0., kidney_empty=0,
                kidney_empty_fp=0, kidney_empty_fp_area_sum=0.,
                liver_empty=0, liver_empty_fp=0, liver_empty_fp_area_sum=0.))
            row['images'] += 1
            valid = targets[local, vessel_channel] >= 0
            truth = targets[local, vessel_channel] > 0
            pred = predicted[local, vessel_channel] & valid
            if truth.any():
                overlap = int((pred & truth).sum())
                count_pred = int(pred.sum())
                count_truth = int(truth.sum())
                row['vessel_positive'] += 1
                row['vessel_zero_overlap'] += int(overlap == 0)
                row['vessel_no_prediction'] += int(count_pred == 0)
                row['vessel_recall_sum'] += overlap / count_truth
                row['vessel_dice_sum'] += 2 * overlap / (count_pred + count_truth)
            for channel, label in ((kidney_channel, 'kidney'), (liver_channel, 'liver')):
                valid = targets[local, channel] >= 0
                truth = targets[local, channel] > 0
                if valid.any() and not truth.any():
                    pixels = int((predicted[local, channel] & valid).sum())
                    row[f'{label}_empty'] += 1
                    row[f'{label}_empty_fp'] += int(pixels > 0)
                    row[f'{label}_empty_fp_area_sum'] += pixels / int(valid.sum())
        offset += len(images)
    if offset != len(abdomen) or not totals:
        raise ValueError('Validation abdominal CT diagnostics did not cover all images')
    result = {}
    for ct, row in sorted(totals.items()):
        positive = row['vessel_positive']
        result[ct] = dict(images=row['images'], vessel_positive=positive,
                          vessel_zero_overlap=row['vessel_zero_overlap'],
                          vessel_zero_overlap_rate=row['vessel_zero_overlap'] / positive if positive else None,
                          vessel_no_prediction=row['vessel_no_prediction'],
                          vessel_no_prediction_rate=row['vessel_no_prediction'] / positive if positive else None,
                          vessel_mean_recall=row['vessel_recall_sum'] / positive if positive else None,
                          vessel_mean_dice=row['vessel_dice_sum'] / positive if positive else None)
        for label in ('kidney', 'liver'):
            empty = row[f'{label}_empty']
            result[ct].update({f'{label}_empty': empty,
                               f'{label}_empty_fp_rate': row[f'{label}_empty_fp'] / empty if empty else None,
                               f'{label}_empty_mean_fp_area_fraction':
                                   row[f'{label}_empty_fp_area_sum'] / empty if empty else None})
    return result


def eligible_checkpoint(candidate, initial, guards=SELECTION_GUARDS):
    return (candidate['overall_positive_dice'] >= initial['overall_positive_dice'] - guards['overall_positive_dice_decline'] and
            candidate['abdomen_empty_fp_area'] <= initial['abdomen_empty_fp_area'] + guards['abdomen_empty_fp_area_increase'] and
            candidate['abdomen_empty_fp_rate'] <= initial['abdomen_empty_fp_rate'] + guards['abdomen_empty_fp_rate_increase'] and
            candidate['abdomen_zero_overlap'] <= initial['abdomen_zero_overlap'] + guards['abdomen_zero_overlap_increase'] and
            candidate['kidney_empty_fp_area'] <= initial['kidney_empty_fp_area'] + guards['kidney_empty_fp_area_increase'] and
            candidate['kidney_positive_dice'] >= initial['kidney_positive_dice'] - guards['kidney_positive_dice_decline'])


def _checkpoint_identity(checkpoint, plan, seed, arm):
    if (checkpoint.get('plan_sha256') != plan['plan_sha256'] or
            checkpoint.get('initial_sha256') != plan['initial_checkpoints'][str(seed)]['sha256'] or
            checkpoint.get('seed') != seed or checkpoint.get('arm') != arm['name']):
        raise ValueError(f'Recovery checkpoint identity differs: {arm["name"]}/seed_{seed}')


def ensure_best_checkpoint(best_path, checkpoint, plan, seed, arm):
    """Reconcile a committed last.pt with a possibly stale best.pt."""
    best_path = Path(best_path)
    if checkpoint['best_epoch'] == checkpoint['epoch']:
        fair_suite._save_checkpoint(best_path, checkpoint)
    elif not best_path.is_file():
        raise FileNotFoundError(f'Best recovery checkpoint missing: {best_path}')
    else:
        best = torch.load(best_path, map_location='cpu', weights_only=True, mmap=True)
        _checkpoint_identity(best, plan, seed, arm)
        if best['epoch'] != checkpoint['best_epoch']:
            raise ValueError('Recovery best checkpoint does not match last checkpoint')


def _run_result(checkpoint, folder):
    return dict(seed=checkpoint['seed'], arm=checkpoint['arm'],
                plan_sha256=checkpoint['plan_sha256'],
                epochs_completed=checkpoint['epoch'], selected_epoch=checkpoint['best_epoch'],
                validation=checkpoint['best_metrics'], initial_validation=checkpoint['initial_metrics'],
                initial_sha256=checkpoint['initial_sha256'],
                checkpoint_directory=str(folder), test_evaluated=False)


def run_arm(plan, train, val, positive, arm, seed, device, progress=None):
    folder = Path(plan['output']) / arm['name'] / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    fair_suite._seed(seed)
    model = fair_suite.create_model('region', BASE, TOP_K).to(device)
    initial_path = Path(plan['initial_checkpoints'][str(seed)]['path'])
    if file_sha256(initial_path) != plan['initial_checkpoints'][str(seed)]['sha256']:
        raise ValueError(f'Fair seed {seed} initial checkpoint bytes changed')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)
    if (initial.get('kind') != 'region' or initial.get('seed') != seed or
            initial.get('source_sha256') != plan['fair_source_sha256'] or
            initial.get('record_sha256') != plan['fair_record_sha256'] or
            initial.get('epoch') != initial.get('best_epoch')):
        raise ValueError(f'Fair seed {seed} checkpoint identity changed')
    model.load_state_dict(initial['model'])
    del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                            generator=torch.Generator().manual_seed(seed))
    last_path, best_path = folder / 'last.pt', folder / 'best.pt'
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        _checkpoint_identity(checkpoint, plan, seed, arm)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        fair_suite._restore_rng(checkpoint['rng'])
        ensure_best_checkpoint(best_path, checkpoint, plan, seed, arm)
        history = checkpoint['history']
        if [row['epoch'] for row in history] != list(range(checkpoint['epoch'] + 1)):
            raise ValueError('Recovery history epochs are incomplete')
        fair_suite._write_history(folder / 'metrics.jsonl', history)
    else:
        if best_path.exists() or (folder / 'metrics.jsonl').exists():
            raise ValueError('Recovery artifacts exist without last.pt')
        raw = fair_suite.evaluate(model, 'region', val_loader, device)
        initial_metrics = abdomen_metrics(raw)
        expected_score = plan['initial_checkpoints'][str(seed)]['best_validation_macro_positive_dice']
        if not math.isclose(initial_metrics['overall_positive_dice'], expected_score,
                            rel_tol=0., abs_tol=1e-4):
            raise ValueError(f'Fair seed {seed} warm-start validation differs from its selected checkpoint')
        history = [dict(epoch=0, validation=initial_metrics, sample_count=0,
                        sample_sha256=None, train_loss=None, eligible=True)]
        checkpoint = dict(schema=VERSION, plan_sha256=plan['plan_sha256'],
                          initial_sha256=plan['initial_checkpoints'][str(seed)]['sha256'],
                          seed=seed, arm=arm['name'], model=model.state_dict(),
                          optimizer=optimizer.state_dict(), rng=fair_suite._rng_state(),
                          epoch=0, history=history, initial_metrics=initial_metrics,
                          best_epoch=0, best_metrics=initial_metrics)
        fair_suite._save_checkpoint(last_path, checkpoint)
        fair_suite._save_checkpoint(best_path, checkpoint)
        fair_suite._write_history(folder / 'metrics.jsonl', history)
    if checkpoint['epoch'] > EPOCHS:
        raise ValueError('Recovery run exceeds fixed epoch budget')
    result_path = folder / 'result.json'
    if checkpoint['epoch'] == EPOCHS and result_path.is_file():
        previous = read_json(result_path)
        if (previous.get('plan_sha256') != plan['plan_sha256'] or
                previous.get('seed') != seed or previous.get('arm') != arm['name'] or
                previous.get('selected_epoch') != checkpoint['best_epoch'] or
                previous.get('initial_sha256') != plan['initial_checkpoints'][str(seed)]['sha256'] or
                previous.get('test_evaluated') is not False or
                'validation_ct_diagnostics' not in previous.get('validation', {})):
            raise ValueError('Completed recovery result differs from checkpoint or plan')
        return previous
    for epoch in range(checkpoint['epoch'] + 1, EPOCHS + 1):
        indices = sampled_indices(train.groups, positive, arm, seed, epoch)
        sampled_images = [train.groups[index][0]['image'] for index in indices]
        loader = DataLoader(train, batch_size=BATCH_SIZE, sampler=indices, shuffle=False,
                            num_workers=2, generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        total_loss = 0.
        image_count = 0
        query_count = 0
        started = time.perf_counter()
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if not target.ge(0).any((1, 2, 3)).all():
                raise ValueError('A sampled image lacks an annotated channel')
            output = model(image)
            loss = objective(output, target, arm)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite recovery loss: {arm["name"]}/seed_{seed}/epoch_{epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            image_count += len(image)
            query_count += int(target.ge(0).any((2, 3)).sum())
            total_loss += float(loss.detach()) * len(image)
        fair_suite._synchronize(device)
        train_seconds = time.perf_counter() - started
        raw = fair_suite.evaluate(model, 'region', val_loader, device)
        metrics = abdomen_metrics(raw)
        eligible = eligible_checkpoint(metrics, checkpoint['initial_metrics'])
        improved = (eligible and
                    (metrics['abdomen_positive_dice'], metrics['overall_positive_dice']) >
                    (checkpoint['best_metrics']['abdomen_positive_dice'],
                     checkpoint['best_metrics']['overall_positive_dice']))
        if improved:
            checkpoint['best_epoch'] = epoch
            checkpoint['best_metrics'] = metrics
        non_abdomen_images = [train.groups[index][0]['image'] for index in indices
                              if train.groups[index][0]['dataset'] != 'AbdomenUS']
        abdomen_images = [train.groups[index][0]['image'] for index in indices
                          if train.groups[index][0]['dataset'] == 'AbdomenUS']
        history.append(dict(epoch=epoch, validation=metrics, sample_count=image_count,
                            annotated_channel_queries=query_count,
                            sample_sha256=digest(sampled_images),
                            non_abdomen_sample_sha256=digest(non_abdomen_images),
                            abdomen_sample_sha256=digest(abdomen_images),
                            train_loss=total_loss / image_count,
                            train_seconds=train_seconds, eligible=eligible))
        checkpoint.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          rng=fair_suite._rng_state(), epoch=epoch, history=history)
        fair_suite._save_checkpoint(last_path, checkpoint)
        if improved:
            fair_suite._save_checkpoint(best_path, checkpoint)
        fair_suite._write_history(folder / 'metrics.jsonl', history)
        event = dict(seed=seed, arm=arm['name'], epoch=epoch,
                     abdomen_positive_dice=metrics['abdomen_positive_dice'],
                     overall_positive_dice=metrics['overall_positive_dice'],
                     abdomen_empty_fp_rate=metrics['abdomen_empty_fp_rate'],
                     abdomen_empty_fp_area=metrics['abdomen_empty_fp_area'],
                     abdomen_zero_overlap=metrics['abdomen_zero_overlap'],
                     eligible=eligible, selected_epoch=checkpoint['best_epoch'],
                     sample_sha256=history[-1]['sample_sha256'])
        print(json.dumps(event), flush=True)
        if progress is not None:
            progress(event)
    best = torch.load(best_path, map_location=device, weights_only=True)
    _checkpoint_identity(best, plan, seed, arm)
    if best['epoch'] != checkpoint['best_epoch']:
        raise ValueError('Selected recovery checkpoint changed before CT diagnostics')
    model.load_state_dict(best['model'])
    selected = dict(checkpoint['best_metrics'])
    selected['validation_ct_diagnostics'] = validation_ct_diagnostics(model, val, device)
    result = _run_result(checkpoint, folder)
    result['validation'] = selected
    if result['epochs_completed'] != EPOCHS:
        raise AssertionError('Recovery run did not reach the fixed budget')
    write_json(result_path, result)
    return result


def compare_runs(results, guards=COMPARISON_GUARDS):
    by_key = {(row['arm'], row['seed']): row for row in results}
    expected = {(arm['name'], seed) for arm in ARMS for seed in SEEDS}
    if set(by_key) != expected or len(by_key) != len(results):
        raise ValueError('Paired screen requires exactly six arms and all three seeds')
    control = {seed: by_key['control', seed]['validation'] for seed in SEEDS}
    candidates = {}
    for arm in ARMS[1:]:
        name = arm['name']
        fields = ('abdomen_positive_dice', 'overall_positive_dice', 'abdomen_empty_fp_area',
                  'abdomen_empty_fp_rate', 'abdomen_zero_overlap', 'liver_empty_fp_rate',
                  'kidney_empty_fp_area', 'kidney_positive_dice')
        differences = {field: [by_key[name, seed]['validation'][field] - control[seed][field]
                               for seed in SEEDS] for field in fields}
        means = {field: statistics.mean(values) for field, values in differences.items()}
        checks = dict(
            abdomen_positive_dice=means['abdomen_positive_dice'] >= guards['min_mean_abdomen_positive_dice_gain'],
            overall_positive_dice=means['overall_positive_dice'] >= -guards['max_mean_overall_positive_dice_decline'],
            abdomen_empty_fp_area=means['abdomen_empty_fp_area'] <= guards['max_mean_abdomen_empty_fp_area_increase'],
            abdomen_empty_fp_rate=means['abdomen_empty_fp_rate'] <= guards['max_mean_abdomen_empty_fp_rate_increase'],
            abdomen_zero_overlap=means['abdomen_zero_overlap'] <= guards['max_mean_abdomen_zero_overlap_increase'],
            each_seed_abdomen_positive_dice=min(differences['abdomen_positive_dice']) >=
                -guards['max_individual_seed_abdomen_positive_dice_decline'],
            liver_empty_fp_rate=means['liver_empty_fp_rate'] <= guards['max_mean_liver_empty_fp_rate_increase'],
            kidney_empty_fp_area=means['kidney_empty_fp_area'] <= guards['max_mean_kidney_empty_fp_area_increase'],
            kidney_positive_dice=means['kidney_positive_dice'] >= -guards['max_mean_kidney_positive_dice_decline'])
        candidates[name] = dict(paired_differences=differences, mean_difference=means,
                                checks=checks, selected=all(checks.values()))
    accepted = [name for name, item in candidates.items() if item['selected']]
    selected = (max(accepted, key=lambda name: (candidates[name]['mean_difference']['abdomen_positive_dice'],
                                                candidates[name]['mean_difference']['overall_positive_dice']))
                if accepted else None)
    structure_averages = {}
    for arm in ARMS:
        per_structure = {}
        for task in ABDOMEN_TASKS:
            name = TASKS[task]
            metrics = [by_key[arm['name'], seed]['validation']['per_structure'][name]
                       for seed in SEEDS]
            per_structure[name] = {}
            for field in ('positive_dice', 'empty_false_positive_rate',
                          'empty_mean_false_positive_area_fraction', 'positive_zero_overlap_rate'):
                values = [row[field] for row in metrics if row[field] is not None]
                per_structure[name][field] = statistics.mean(values) if values else None
        structure_averages[arm['name']] = per_structure
    return dict(schema=VERSION, selection_basis='validation only',
                selection_guards=SELECTION_GUARDS, comparison_guards=guards,
                candidate_comparisons=candidates, selected_candidate=selected,
                validation_structure_means=structure_averages,
                test_evaluated=False,
                limitation='Historical AbdomenUS test images were previously inspected; this validation screen is exploratory')


def validate_paired_histories(plan):
    """Check fixed budgets and matched draws before interpreting arm effects."""
    output = Path(plan['output'])
    for seed in SEEDS:
        histories = {}
        for arm in ARMS:
            folder = output / arm['name'] / f'seed_{seed}'
            result = read_json(folder / 'result.json')
            rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()
                    if line.strip()]
            if (result['seed'] != seed or result['arm'] != arm['name'] or
                    result['epochs_completed'] != EPOCHS or result['test_evaluated'] or
                    result.get('plan_sha256') != plan['plan_sha256'] or
                    result['initial_sha256'] != plan['initial_checkpoints'][str(seed)]['sha256'] or
                    len(rows) != EPOCHS + 1 or [row['epoch'] for row in rows] != list(range(EPOCHS + 1))):
                raise ValueError(f'Incomplete or mismatched paired history: {arm["name"]}/seed_{seed}')
            if any(row['sample_count'] != len(DOMAINS) * QUOTA or
                   not isinstance(row.get('sample_sha256'), str) or
                   len(row['sample_sha256']) != 64 for row in rows[1:]):
                raise ValueError(f'Paired history has a different image budget: {arm["name"]}/seed_{seed}')
            histories[arm['name']] = rows
        base = histories['control']
        for arm in ARMS[1:]:
            for epoch in range(1, EPOCHS + 1):
                row, control = histories[arm['name']][epoch], base[epoch]
                if (row['sample_count'] != control['sample_count'] or
                        row['annotated_channel_queries'] < row['sample_count'] or
                        row['non_abdomen_sample_sha256'] != control['non_abdomen_sample_sha256']):
                    raise ValueError(f'Paired non-abdomen draws differ: {arm["name"]}/seed_{seed}/epoch_{epoch}')
                if arm['abdomen_sampling_weight'] == 1. and (
                        row['sample_sha256'] != control['sample_sha256'] or
                        row['abdomen_sample_sha256'] != control['abdomen_sample_sha256'] or
                        row['annotated_channel_queries'] != control['annotated_channel_queries']):
                    raise ValueError(f'Undeclared sampling difference: {arm["name"]}/seed_{seed}/epoch_{epoch}')


@contextmanager
def gpu_lock(gpu):
    path = Path(f'/tmp/multi_organ_fair_gpu_{gpu}.lock')
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'GPU {gpu} is reserved by another project controller') from error
        yield


def check_gpu(gpu):
    import subprocess
    result = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        text=True).strip()
    free = int(result)
    if free < MINIMUM_FREE_MIB:
        raise RuntimeError(f'GPU {gpu} has {free} MiB free; need at least {MINIMUM_FREE_MIB} MiB')
    return free


def run(plan, gpu):
    output = Path(plan['output'])
    comparison_path = output / 'comparison.json'
    if comparison_path.is_file():
        validate_paired_histories(plan)
        verify_train_val_contents(plan)
        existing = read_json(comparison_path)
        expected = compare_runs(existing['runs'])
        if (existing.get('plan_sha256') != plan['plan_sha256'] or
                existing.get('candidate_comparisons') != expected['candidate_comparisons'] or
                existing.get('selected_candidate') != expected['selected_candidate'] or
                existing.get('test_evaluated') is not False):
            raise ValueError('Existing recovery comparison differs from frozen paired results')
        return existing
    with gpu_lock(gpu):
        free = check_gpu(gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        os.environ['OMP_NUM_THREADS'] = '4'
        device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            raise RuntimeError('Requested GPU is unavailable')
        status = dict(schema=VERSION, plan_sha256=plan['plan_sha256'], pid=os.getpid(),
                      gpu=gpu, gpu_free_mib_at_start=free, started_at=timestamp())
        def update(state, **extra):
            status.update(state=state, updated_at=timestamp(), **extra)
            write_json(output / 'status.json', status)
        update('running')
        try:
            train, val, positive = load_datasets(plan)
            results = []
            for seed in SEEDS:
                for arm in ARMS:
                    update('running', seed=seed, arm=arm['name'])
                    result = run_arm(plan, train, val, positive, arm, seed, device,
                                     progress=lambda event: update('running', **event))
                    results.append(result)
            validate_paired_histories(plan)
            verify_train_val_contents(plan)
            comparison = compare_runs(results)
            comparison.update(plan_sha256=plan['plan_sha256'], runs=results,
                              completed_at=timestamp())
            write_json(comparison_path, comparison)
            update('complete', seed=None, arm=None, selected_candidate=comparison['selected_candidate'])
            return comparison
        except BaseException as error:
            update('failed', error=f'{type(error).__name__}: {error}')
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', default='42,43,44', help='Fixed paired seed set: 42,43,44')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--gpu', type=int)
    args = parser.parse_args(argv)
    if args.seeds != '42,43,44':
        parser.error('This prespecified paired screen requires --seeds 42,43,44')
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be a nonnegative physical GPU index')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.prepare:
        plan = prepare_plan(args.fair_root, args.output)
        print(json.dumps(dict(plan=str(Path(plan['output']) / 'plan.json'),
                              plan_sha256=plan['plan_sha256'])), flush=True)
    else:
        # Fail immediately if the fair comparison is incomplete; an external
        # queue may wait for it without holding a GPU lock.
        plan = load_plan(args.fair_root, args.output)
        result = run(plan, args.gpu)
        print(json.dumps(dict(comparison=str(Path(plan['output']) / 'comparison.json'),
                              selected_candidate=result['selected_candidate'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'abdomen_recovery: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
