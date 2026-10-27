"""Frozen image-matched comparison of shared, task-ID, and region U-Nets.

Each architecture sees the same image draws and annotated channel/pixel targets.
Task-ID conditioning remains an intentional input difference, and its repeated
image forwards cost more compute: this protocol does not claim equal FLOPs.
Test images are opened only with the explicit --evaluate-test flag.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .baseline import PlainUNet
from .data import TASKS, build_records, validate_splits
from .empty_loss import empty_top_fraction_loss
from .model import UNetMoE
from .region import DOMAINS, PAPER_ALIGNED_TASK_IDS, RegionDataset, RegionUNet
from .sampling import domain_draws
from .schedule import epoch_lr


MODELS = ('shared', 'task_id', 'region')
PROTOCOL_VERSION = 'fair_image_matched_v1'


def create_model(kind, base=16, top_k=1, experts=9, task_ids=None):
    task_ids = PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids
    if kind == 'shared':
        return PlainUNet(outputs=len(task_ids), base=base)
    if kind == 'task_id':
        return UNetMoE(tasks=len(TASKS), base=base, experts=experts, top_k=top_k)
    if kind == 'region':
        return RegionUNet(base=base, top_k=top_k, task_ids=task_ids)
    raise ValueError(f'Unknown model: {kind}')


def forward_model(model, kind, image, target, task_ids=None):
    """Scatter task-conditioned predictions into the common channel layout.

The label *availability* chooses scored task queries; foreground presence never
does. Empty annotated channels are therefore included in both train and eval.
Unannotated outputs are placeholders and are ignored by every loss and metric.
"""
    task_ids = PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids
    if kind == 'shared':
        return {'logits': model(image)}
    if kind == 'region':
        return model(image)
    if kind != 'task_id':
        raise ValueError(f'Unknown model: {kind}')
    active = target.ge(0).any((2, 3))
    batch, channel = active.nonzero(as_tuple=True)
    if not batch.numel():
        raise ValueError('Batch has no annotated target pixels')
    task = torch.as_tensor(task_ids, device=image.device)[channel]
    output = model(image[batch], task)
    # index_copy is differentiable, retaining each queried prediction's gradient.
    flat = image.new_zeros((target.shape[0] * target.shape[1], *target.shape[2:]))
    flat = flat.index_copy(0, batch * target.shape[1] + channel, output['logits'][:, 0])
    return {**output, 'logits': flat.reshape_as(target)}


def common_loss(logits, target):
    """BCE + soft Dice, equal valid channels/image, then equal images/batch."""
    if logits.shape != target.shape or target.ndim != 4:
        raise ValueError('Expected matching B,C,H,W logits and targets')
    valid = target >= 0
    truth = torch.where(valid, target, 0.)
    active = valid.any((2, 3))
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction='none') * valid).sum((2, 3))
    bce = bce / valid.sum((2, 3)).clamp_min(1)
    probability = logits.sigmoid() * valid
    dice = 1 - (2 * (probability * truth).sum((2, 3)) + 1) / (
        probability.sum((2, 3)) + truth.sum((2, 3)) + 1)
    per_image = ((bce + dice) * active).sum(1) / active.sum(1).clamp_min(1)
    return per_image.mean()


def objective(model, kind, output, target, args):
    loss = common_loss(output['logits'], target)
    if args.empty_weight:
        abdomen_channels = [index for index, task in enumerate(PAPER_ALIGNED_TASK_IDS) if 13 <= task <= 20]
        loss = loss + args.empty_weight * empty_top_fraction_loss(
            output['logits'], target, fraction=args.empty_fraction, channel_indices=abdomen_channels)
    if kind == 'task_id':
        if args.balance_weight:
            loss = loss + args.balance_weight * output['balance']
        if args.router_z_weight:
            loss = loss + args.router_z_weight * output['router_z']
    if kind == 'region':
        if args.coarse_weight:
            loss = loss + args.coarse_weight * torch.stack([
                common_loss(coarse, target) for coarse in output['coarse']]).mean()
        if args.route_weight:
            present = target.gt(0).any((2, 3))
            labels = torch.as_tensor(model.expert_for_region, device=target.device).expand(target.shape[0], -1)
            route = torch.stack([
                (F.cross_entropy(scores.transpose(1, 2), labels, reduction='none') * present).sum()
                / present.sum().clamp_min(1) for scores in output['routing']]).mean()
            loss = loss + args.route_weight * route
    return loss


@torch.no_grad()
def evaluate(model, kind, loader, device, task_ids=None):
    """Per-image/channel metrics; all areas exclude ignore pixels.

    Positive miss means no predicted foreground on a positive image. Area is
    reported as raw resized-image pixels and as a fraction of valid pixels.
    Macro metrics average only structures with an applicable denominator.
    """
    task_ids = PAPER_ALIGNED_TASK_IDS if task_ids is None else task_ids
    model.eval()
    totals = [dict(samples=0, positive_samples=0, empty_samples=0, dice=0., positive_dice=0.,
                   empty_fp=0, positive_miss=0, positive_zero_overlap=0, fp_pixels=0., fp_fraction=0.,
                   empty_fp_pixels=0., empty_fp_fraction=0.) for _ in task_ids]
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        valid, truth = target.ge(0), target.gt(0)
        pred = (forward_model(model, kind, image, target, task_ids)['logits'].sigmoid() >= .5) & valid
        pred_pixels, truth_pixels = pred.sum((2, 3)), truth.sum((2, 3))
        denominator = pred_pixels + truth_pixels
        dice = torch.where(denominator > 0, 2 * (pred & truth).sum((2, 3)) / denominator.clamp_min(1), 1.)
        false_positive = (pred & ~truth).sum((2, 3))
        valid_pixels = valid.sum((2, 3))
        for batch, channel in valid.any((2, 3)).nonzero().tolist():
            row = totals[channel]
            row['samples'] += 1
            row['dice'] += float(dice[batch, channel])
            row['fp_pixels'] += float(false_positive[batch, channel])
            row['fp_fraction'] += float(false_positive[batch, channel] / valid_pixels[batch, channel])
            if truth_pixels[batch, channel] > 0:
                row['positive_samples'] += 1
                row['positive_dice'] += float(dice[batch, channel])
                row['positive_miss'] += int(pred_pixels[batch, channel] == 0)
                row['positive_zero_overlap'] += int(not (pred[batch, channel] & truth[batch, channel]).any())
            else:
                row['empty_samples'] += 1
                row['empty_fp'] += int(pred_pixels[batch, channel] > 0)
                row['empty_fp_pixels'] += float(false_positive[batch, channel])
                row['empty_fp_fraction'] += float(false_positive[batch, channel] / valid_pixels[batch, channel])
    per_structure = {}
    for task, row in zip(task_ids, totals):
        samples, positive, empty = row['samples'], row['positive_samples'], row['empty_samples']
        per_structure[TASKS[task]] = dict(
            dice=row['dice'] / samples if samples else None,
            positive_dice=row['positive_dice'] / positive if positive else None,
            samples=samples, positive_samples=positive, empty_samples=empty,
            empty_false_positive_rate=row['empty_fp'] / empty if empty else None,
            positive_miss_rate=row['positive_miss'] / positive if positive else None,
            positive_zero_overlap_rate=row['positive_zero_overlap'] / positive if positive else None,
            empty_mean_false_positive_pixels=row['empty_fp_pixels'] / empty if empty else None,
            empty_mean_false_positive_area_fraction=row['empty_fp_fraction'] / empty if empty else None,
            mean_false_positive_pixels=row['fp_pixels'] / samples if samples else None,
            mean_false_positive_area_fraction=row['fp_fraction'] / samples if samples else None)
    result = dict(per_structure=per_structure, threshold=.5)
    for output_key, field in [('macro_positive_dice', 'positive_dice'), ('macro_all_dice', 'dice'),
                              ('macro_empty_false_positive_rate', 'empty_false_positive_rate'),
                              ('macro_positive_miss_rate', 'positive_miss_rate'),
                              ('macro_positive_zero_overlap_rate', 'positive_zero_overlap_rate'),
                              ('macro_empty_false_positive_pixels', 'empty_mean_false_positive_pixels'),
                              ('macro_empty_false_positive_area_fraction', 'empty_mean_false_positive_area_fraction'),
                              ('macro_false_positive_pixels', 'mean_false_positive_pixels'),
                              ('macro_false_positive_area_fraction', 'mean_false_positive_area_fraction')]:
        values = [row[field] for row in per_structure.values() if row[field] is not None]
        result[output_key] = sum(values) / len(values) if values else None
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb' if binary else 'w', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False,
                                         **({} if binary else {'encoding': 'utf-8'})) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_write(path, lambda stream: json.dump(value, stream, indent=2, allow_nan=False))


def _save_checkpoint(path, value):
    _atomic_write(path, lambda stream: torch.save(value, stream), binary=True)


def _write_history(path, history):
    _atomic_write(path, lambda stream: stream.write(''.join(json.dumps(row, allow_nan=False) + '\n' for row in history)))


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def _restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def _loader(dataset, args, seed, sampler=None):
    # Isolate DataLoader worker/base-seed draws from the model's RNG stream.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                      sampler=sampler, shuffle=False, generator=generator)


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _configuration(args):
    operational = {'output', 'resume', 'evaluate_test', 'dry_run', 'stop_after_epoch'}
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in operational}
    config.update(protocol_version=PROTOCOL_VERSION, abdomen_split='ct7_2',
                  split_policy='trainmore_v6', task_ids=list(PAPER_ALIGNED_TASK_IDS),
                  labels='paper5', augmentation='none',
                  objective='mean_images(mean_annotated_channels(BCE_valid_pixels + soft_Dice_smooth_1))',
                  optimizer={'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': args.weight_decay},
                  gradient_clip_norm=5., checkpoint_selection='best_validation_macro_positive_dice',
                  threshold=.5)
    return config


def _prepare_protocol(args):
    records, skipped = build_records(args.data_root, args.split_seed, 'ct7_2', None, 'trainmore_v6')
    validate_splits(records)
    excluded = dict(Counter(f"{row['dataset']}:task_{row['task']}" for row in records
                            if row['task'] not in PAPER_ALIGNED_TASK_IDS))
    records = [row for row in records if row['task'] in PAPER_ALIGNED_TASK_IDS]
    # Sort records before grouping so manifest ordering cannot alter a draw's image.
    records = sorted(records, key=lambda row: json.dumps(row, sort_keys=True))
    config = _configuration(args)
    protocol = dict(config=config, config_sha256=_digest(config), record_sha256=_digest(records),
                    record_hash_scope='ordered full record metadata including paths and label versions; not PNG content',
                    skipped=skipped, excluded_tasks=excluded,
                    input_difference={'shared': 'image only', 'task_id': 'image + oracle annotated task ID; not automatic task routing',
                                                      'region': 'image only'},
                    compute_comparison='matched images and primary objective; parameters and elapsed time recorded, FLOPs not matched',
                    test_policy='held out; explicit --evaluate-test only; no threshold or checkpoint tuning on test')
    source_names = ['fair_suite.py', 'baseline.py', 'model.py', 'region.py', 'normalization.py', 'data.py',
                    'sampling.py', 'schedule.py', 'empty_loss.py']
    source_files = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in source_names}
    protocol.update(source_files=source_files, source_sha256=_digest(source_files),
                    environment=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                                     numpy=np.__version__, cuda=torch.version.cuda))
    try:
        protocol['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        protocol['git_commit'] = None
    datasets = {split: RegionDataset([row for row in records if row['split'] == split], args.size,
                                     PAPER_ALIGNED_TASK_IDS) for split in ('train', 'val')}
    if not len(datasets['train']) or not len(datasets['val']):
        raise ValueError('Nonempty train and validation splits are required')
    counts = Counter(rows[0]['dataset'] for rows in datasets['train'].groups)
    missing = set(DOMAINS) - counts.keys()
    if missing:
        raise ValueError(f'All nine training domains are required; missing {sorted(missing)}')
    hc = [row for row in records if row['dataset'] == 'Fetal_HC']
    if any(row.get('label_version') != 'hc18_filled_v1' for row in hc):
        raise ValueError('Fetal_HC requires filled hc18_filled_v1 masks')
    protocol['split_images'] = {split: dict(Counter(row['dataset'] for row in {
        row['image']: row for row in records if row['split'] == split and row['task'] in PAPER_ALIGNED_TASK_IDS}.values()))
                               for split in ('train', 'val', 'test')}
    protocol['split_groups'] = {split: {domain: sorted({row['group'] for row in records
        if row['split'] == split and row['dataset'] == domain}) for domain in DOMAINS}
                               for split in ('train', 'val', 'test')}
    protocol['budget'] = dict(images_per_epoch=len(DOMAINS) * args.quota_per_domain,
                             optimizer_steps_per_epoch=(len(DOMAINS) * args.quota_per_domain + args.batch_size - 1) // args.batch_size,
                             planned_epochs=args.epochs, planned_images_per_run=len(DOMAINS) * args.quota_per_domain * args.epochs,
                             seeds=args.seeds, models=args.models)
    protocol_path = args.output / 'protocol.json'
    if args.resume:
        if not protocol_path.is_file():
            raise FileNotFoundError('--resume requires an existing protocol.json')
        previous = json.loads(protocol_path.read_text())
        if previous['config_sha256'] != protocol['config_sha256']:
            raise ValueError('Config hash differs from frozen protocol; use a new output directory')
        if previous['record_sha256'] != protocol['record_sha256']:
            raise ValueError('Record hash differs from frozen protocol; refusing to resume')
        if previous.get('source_sha256') != protocol['source_sha256']:
            raise ValueError('Source hash differs from frozen protocol; use the original code to resume')
        if _digest(json.loads((args.output / 'records.json').read_text())) != protocol['record_sha256']:
            raise ValueError('Stored records.json does not match frozen record hash')
    else:
        if args.output.exists():
            raise FileExistsError('Output exists; use a new directory or --resume')
        args.output.mkdir(parents=True)
        _write_json(args.output / 'records.json', records)
        _write_json(protocol_path, protocol)
    return protocol, records, datasets


def _check_checkpoint(checkpoint, protocol, kind, seed):
    for key in ('config_sha256', 'record_sha256', 'source_sha256'):
        if checkpoint.get(key) != protocol[key]:
            raise ValueError(f'Checkpoint {key} does not match frozen protocol')
    if checkpoint['kind'] != kind or checkpoint['seed'] != seed:
        raise ValueError('Checkpoint model or training seed does not match run')


def _run_model(args, protocol, records, datasets, kind, seed):
    _seed(seed)
    model = create_model(kind, args.base, args.top_k, args.experts).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    folder = args.output / kind / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    history, start_epoch, best_score, best_epoch = [], 1, -float('inf'), None
    elapsed_seconds = 0.
    if args.resume and (folder / 'last.pt').is_file():
        checkpoint = torch.load(folder / 'last.pt', map_location=args.device, weights_only=True)
        _check_checkpoint(checkpoint, protocol, kind, seed)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']
        start_epoch, best_score, best_epoch = checkpoint['epoch'] + 1, checkpoint['best_score'], checkpoint['best_epoch']
        elapsed_seconds = checkpoint['elapsed_seconds']
        # Recover a crash between committing last.pt and committing best.pt/logs.
        if best_epoch == checkpoint['epoch']:
            _save_checkpoint(folder / 'best.pt', checkpoint)
        if not (folder / 'best.pt').is_file():
            raise FileNotFoundError('Previous best.pt is missing and cannot be reconstructed from last.pt')
        _write_history(folder / 'metrics.jsonl', history)
        _restore_rng(checkpoint['rng'])
    elif any((folder / name).exists() for name in ('best.pt', 'metrics.jsonl')):
        raise ValueError('Run artifacts exist without last.pt; refusing an ambiguous restart')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    val_loader = _loader(datasets['val'], args, seed)
    final_epoch = min(args.epochs, args.stop_after_epoch or args.epochs)
    for epoch in range(start_epoch, final_epoch + 1):
        indices = domain_draws(datasets['train'].groups, args.quota_per_domain, seed, epoch)
        sampled_images = [datasets['train'].groups[index][0]['image'] for index in indices]
        train_loader = _loader(datasets['train'], args, seed + epoch, indices)
        lr = epoch_lr(args.lr, epoch, args.epochs, args.lr_schedule, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        total_loss, image_count, query_count = 0., 0, 0
        _synchronize(args.device)
        started = time.perf_counter()
        for image, target in train_loader:
            image, target = image.to(args.device), target.to(args.device)
            if not target.ge(0).any((1, 2, 3)).all():
                raise ValueError('A sampled image has no valid annotated pixels')
            output = forward_model(model, kind, image, target)
            loss = objective(model, kind, output, target, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {kind}, seed {seed}, epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            image_count += len(image)
            query_count += int(target.ge(0).any((2, 3)).sum())
            total_loss += float(loss.detach()) * len(image)
        _synchronize(args.device)
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        metrics = evaluate(model, kind, val_loader, args.device)
        _synchronize(args.device)
        validation_seconds = time.perf_counter() - validation_started
        elapsed_seconds += train_seconds + validation_seconds
        score = metrics['macro_positive_dice']
        if score is None:
            raise ValueError('Validation has no foreground samples for checkpoint selection')
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        metrics.update(epoch=epoch, kind=kind, seed=seed, train_loss=total_loss / image_count, lr=lr,
                       sample_count=image_count, annotated_channel_queries=query_count,
                       model_input_images=query_count if kind == 'task_id' else image_count,
                       sample_sha256=_digest(sampled_images), parameter_count=parameters,
                       train_seconds=train_seconds, validation_seconds=validation_seconds,
                       elapsed_seconds=elapsed_seconds)
        history.append(metrics)
        checkpoint = dict(protocol_version=PROTOCOL_VERSION, architecture=f'fair_{kind}_unet_v1',
                          kind=kind, seed=seed, config=protocol['config'],
                          config_sha256=protocol['config_sha256'], record_sha256=protocol['record_sha256'],
                          source_sha256=protocol['source_sha256'],
                          task_ids=list(PAPER_ALIGNED_TASK_IDS), structures=[TASKS[i] for i in PAPER_ALIGNED_TASK_IDS],
                          model=model.state_dict(), optimizer=optimizer.state_dict(), rng=_rng_state(),
                          epoch=epoch, metrics=metrics, history=history, best_score=best_score, best_epoch=best_epoch,
                          elapsed_seconds=elapsed_seconds, parameter_count=parameters)
        _save_checkpoint(folder / 'last.pt', checkpoint)
        if improved:
            _save_checkpoint(folder / 'best.pt', checkpoint)
        _write_history(folder / 'metrics.jsonl', history)
        print(json.dumps(dict(kind=kind, seed=seed, epoch=epoch, train_loss=metrics['train_loss'],
                              val_macro_positive_dice=score, sample_sha256=metrics['sample_sha256'],
                              train_seconds=train_seconds)), flush=True)
    result = dict(kind=kind, seed=seed, parameter_count=parameters,
                  epochs_completed=history[-1]['epoch'] if history else 0,
                  best_epoch=best_epoch, best_validation_macro_positive_dice=best_score if history else None,
                  elapsed_seconds=elapsed_seconds, checkpoint_directory=str(folder), test_evaluated=False)
    if args.evaluate_test:
        if result['epochs_completed'] < args.epochs:
            raise ValueError('Test evaluation requires completing the frozen training budget')
        checkpoint = torch.load(folder / 'best.pt', map_location=args.device, weights_only=True)
        _check_checkpoint(checkpoint, protocol, kind, seed)
        model.load_state_dict(checkpoint['model'])
        test_dataset = RegionDataset([row for row in records if row['split'] == 'test'], args.size,
                                     PAPER_ALIGNED_TASK_IDS)
        if not len(test_dataset):
            raise ValueError('Explicit test evaluation requested but no test images exist')
        test = evaluate(model, kind, _loader(test_dataset, args, seed), args.device)
        test.update(checkpoint_epoch=checkpoint['epoch'], checkpoint_selection='best_validation_macro_positive_dice',
                    record_sha256=protocol['record_sha256'], config_sha256=protocol['config_sha256'])
        _write_json(folder / 'test_metrics.json', test)
        result.update(test_evaluated=True, test=test)
    return result


def run_suite(args):
    """Freeze records once, then reset each architecture to each training seed."""
    protocol, records, datasets = _prepare_protocol(args)
    summary = dict(protocol_version=PROTOCOL_VERSION, config_sha256=protocol['config_sha256'],
                   record_sha256=protocol['record_sha256'], dry_run=args.dry_run,
                   test_evaluation_requested=args.evaluate_test, runs=[])
    if args.dry_run:
        if not args.resume:
            _write_json(args.output / 'summary.json', summary)
        print(json.dumps(dict(protocol=str(args.output / 'protocol.json'), budget=protocol['budget'])), flush=True)
        return summary
    for seed in args.seeds:
        for kind in args.models:
            summary['runs'].append(_run_model(args, protocol, records, datasets, kind, seed))
            _write_json(args.output / 'summary.json', summary)
    aggregate = {}
    for kind in args.models:
        values = [run['best_validation_macro_positive_dice'] for run in summary['runs']
                  if run['kind'] == kind and run['best_validation_macro_positive_dice'] is not None]
        aggregate[kind] = dict(seeds=len(values), mean=float(np.mean(values)) if values else None,
                               sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    summary['validation_across_seeds'] = aggregate
    test_aggregate = {}
    for kind in args.models:
        evaluated = [run['test'] for run in summary['runs'] if run['kind'] == kind and run['test_evaluated']]
        if evaluated:
            test_aggregate[kind] = {}
            for field in evaluated[0]:
                if not field.startswith('macro_'):
                    continue
                values = [metrics[field] for metrics in evaluated if metrics[field] is not None]
                test_aggregate[kind][field] = dict(seeds=len(values), mean=float(np.mean(values)) if values else None,
                                                  sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    summary['test_across_seeds'] = test_aggregate
    _write_json(args.output / 'summary.json', summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--seeds', default='42,43,44', help='Comma-separated independent training seeds')
    parser.add_argument('--split-seed', type=int, choices=[42], default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quota-per-domain', type=int, default=500)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--experts', type=int, default=9, help='Task-ID expert count; region architecture has nine experts')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--min-lr-ratio', type=float, default=.1)
    parser.add_argument('--coarse-weight', type=float, default=0.)
    parser.add_argument('--route-weight', type=float, default=0.)
    parser.add_argument('--balance-weight', type=float, default=0.)
    parser.add_argument('--router-z-weight', type=float, default=0.)
    parser.add_argument('--empty-weight', type=float, default=0., help='Optional abdomen-only empty-channel hard-negative loss')
    parser.add_argument('--empty-fraction', type=float, default=.01)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Freeze protocol and records without reading PNGs or training')
    parser.add_argument('--stop-after-epoch', type=int, default=0,
                        help='Pause after this epoch without changing the frozen planned epoch budget')
    args = parser.parse_args(argv)
    args.models = args.models.split(',')
    try:
        args.seeds = [int(seed) for seed in args.seeds.split(',')]
    except ValueError:
        parser.error('--seeds must be comma-separated integers')
    if not set(args.models).issubset(MODELS) or len(set(args.models)) != len(args.models):
        parser.error('--models must contain unique choices from shared,task_id,region')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('--seeds must be unique integers in 0..2**32-1')
    if (args.epochs < 1 or args.quota_per_domain < 1 or args.size < 32 or args.base < 4 or args.base % 4
            or args.batch_size < 1 or args.workers < 0 or not 1 <= args.top_k <= min(args.experts, 9)
            or args.lr <= 0 or not np.isfinite(args.lr) or not 0 <= args.min_lr_ratio <= 1
            or not 0 < args.empty_fraction <= 1 or args.weight_decay < 0 or not np.isfinite(args.weight_decay)
            or args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs
            or any(value < 0 or not np.isfinite(value) for value in
                   (args.coarse_weight, args.route_weight, args.balance_weight, args.router_z_weight, args.empty_weight))):
        parser.error('Invalid size, base, budget, learning rate, routing, or auxiliary-loss setting')
    if args.evaluate_test and args.stop_after_epoch and args.stop_after_epoch < args.epochs:
        parser.error('--evaluate-test cannot be combined with an incomplete training budget')
    return args


def main(argv=None):
    run_suite(parse_args(argv))


if __name__ == '__main__':
    main()
