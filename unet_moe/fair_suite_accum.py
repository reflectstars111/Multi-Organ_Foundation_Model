"""Image-matched fair suite with an effective batch formed by gradient accumulation.

``--batch-size`` is the number of images per optimizer update; the physical
DataLoader batch is ``--micro-batch-size``. Primary losses are means per image,
so each microbatch contributes its actual image count divided by the current
effective batch count. Validation and test also use the physical batch size.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import fair_suite as fair
from .sampling import domain_draws
from .schedule import epoch_lr


PROTOCOL_VERSION = 'fair_image_matched_accum_v1'
SOURCE_NAMES = ('fair_suite_accum.py', 'fair_suite.py', 'baseline.py', 'model.py',
                'region.py', 'normalization.py', 'data.py', 'sampling.py', 'schedule.py', 'empty_loss.py')


def _configuration(args):
    config = fair._configuration(args)
    config.update(protocol_version=PROTOCOL_VERSION,
                  effective_batch_size=args.batch_size,
                  physical_micro_batch_size=args.micro_batch_size,
                  accumulation='per-image weighted microbatch losses; one optimizer update and one gradient clip per effective batch')
    return config


def _prepare_protocol(args):
    records, skipped = fair.build_records(args.data_root, args.split_seed, 'ct7_2', None, 'trainmore_v6')
    fair.validate_splits(records)
    excluded = dict(Counter(f"{row['dataset']}:task_{row['task']}" for row in records
                            if row['task'] not in fair.PAPER_ALIGNED_TASK_IDS))
    records = sorted((row for row in records if row['task'] in fair.PAPER_ALIGNED_TASK_IDS),
                     key=lambda row: json.dumps(row, sort_keys=True))
    config = _configuration(args)
    protocol = dict(config=config, config_sha256=fair._digest(config),
                    record_sha256=fair._digest(records),
                    record_hash_scope='ordered full record metadata including paths and label versions; not PNG content',
                    skipped=skipped, excluded_tasks=excluded,
                    input_difference={'shared': 'image only',
                                      'task_id': 'image + oracle annotated task ID; not automatic task routing',
                                      'region': 'image only'},
                    compute_comparison='matched images and primary objective; parameters and elapsed time recorded, FLOPs not matched',
                    test_policy='held out; explicit --evaluate-test only; no threshold or checkpoint tuning on test')
    source_files = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in SOURCE_NAMES}
    protocol.update(source_files=source_files, source_sha256=fair._digest(source_files),
                    environment=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                                     numpy=np.__version__, cuda=torch.version.cuda))
    try:
        protocol['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        protocol['git_commit'] = None
    datasets = {split: fair.RegionDataset([row for row in records if row['split'] == split], args.size,
                                          fair.PAPER_ALIGNED_TASK_IDS)
                for split in ('train', 'val')}
    if not len(datasets['train']) or not len(datasets['val']):
        raise ValueError('Nonempty train and validation splits are required')
    counts = Counter(rows[0]['dataset'] for rows in datasets['train'].groups)
    missing = set(fair.DOMAINS) - counts.keys()
    if missing:
        raise ValueError(f'All nine training domains are required; missing {sorted(missing)}')
    if any(row.get('label_version') != 'hc18_filled_v1' for row in records
           if row['dataset'] == 'Fetal_HC'):
        raise ValueError('Fetal_HC requires filled hc18_filled_v1 masks')
    protocol['split_images'] = {split: dict(Counter(row['dataset'] for row in {
        row['image']: row for row in records if row['split'] == split}.values()))
                               for split in ('train', 'val', 'test')}
    protocol['split_groups'] = {split: {domain: sorted({row['group'] for row in records
        if row['split'] == split and row['dataset'] == domain}) for domain in fair.DOMAINS}
                               for split in ('train', 'val', 'test')}
    images_per_epoch = len(fair.DOMAINS) * args.quota_per_domain
    protocol['budget'] = dict(images_per_epoch=images_per_epoch,
                             optimizer_steps_per_epoch=(images_per_epoch + args.batch_size - 1) // args.batch_size,
                             physical_micro_batches_per_epoch=(images_per_epoch + args.micro_batch_size - 1)
                                 // args.micro_batch_size,
                             planned_epochs=args.epochs,
                             planned_images_per_run=images_per_epoch * args.epochs,
                             seeds=args.seeds, models=args.models)
    protocol_path = args.output / 'protocol.json'
    if args.resume:
        if not protocol_path.is_file():
            raise FileNotFoundError('--resume requires an existing protocol.json')
        previous = json.loads(protocol_path.read_text())
        for key in ('config_sha256', 'record_sha256', 'source_sha256'):
            if previous.get(key) != protocol[key]:
                raise ValueError(f'{key} differs from frozen accumulation protocol')
        if fair._digest(json.loads((args.output / 'records.json').read_text())) != protocol['record_sha256']:
            raise ValueError('Stored records.json differs from frozen record hash')
    else:
        if args.output.exists():
            raise FileExistsError('Output exists; use a new directory or --resume')
        args.output.mkdir(parents=True)
        fair._write_json(args.output / 'records.json', records)
        fair._write_json(protocol_path, protocol)
    return protocol, records, datasets


def _physical_loader(dataset, args, seed, sampler=None):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=args.micro_batch_size,
                                       num_workers=args.workers, sampler=sampler,
                                       shuffle=False, generator=generator)


def train_effective_batches(model, kind, loader, optimizer, args, image_total, seed, epoch):
    """Train physical microbatches; return totals after whole effective updates."""
    total_loss = 0.
    image_count = query_count = optimizer_steps = 0
    window_count = 0
    window_size = 0
    for image, target in loader:
        image, target = image.to(args.device), target.to(args.device)
        if not target.ge(0).any((1, 2, 3)).all():
            raise ValueError('A sampled image has no valid annotated pixels')
        if window_count == 0:
            optimizer.zero_grad(set_to_none=True)
            window_size = min(args.batch_size, image_total - image_count)
            if window_size <= 0:
                raise ValueError('Loader has more images than the sampled draw')
        if len(image) > window_size - window_count:
            raise ValueError('Physical microbatch crosses an effective-batch boundary')
        output = fair.forward_model(model, kind, image, target)
        loss = fair.objective(model, kind, output, target, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss in {kind}, seed {seed}, epoch {epoch}')
        (loss * (len(image) / window_size)).backward()
        total_loss += float(loss.detach()) * len(image)
        image_count += len(image)
        query_count += int(target.ge(0).any((2, 3)).sum())
        window_count += len(image)
        if window_count == window_size:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            optimizer_steps += 1
            window_count = 0
        # Match the preflight: do not retain the previous query outputs while
        # the next microbatch constructs its forward graph.
        del output, loss, image, target
    if image_count != image_total or window_count:
        raise ValueError('Loader failed to finish the exact sampled image budget')
    expected_steps = (image_total + args.batch_size - 1) // args.batch_size
    if optimizer_steps != expected_steps:
        raise ValueError('Optimizer step count differs from effective-batch budget')
    return total_loss, image_count, query_count, optimizer_steps


def _check_checkpoint(checkpoint, protocol, kind, seed):
    fair._check_checkpoint(checkpoint, protocol, kind, seed)
    if checkpoint.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Checkpoint uses a different accumulation protocol')


def _run_model(args, protocol, records, datasets, kind, seed):
    fair._seed(seed)
    model = fair.create_model(kind, args.base, args.top_k, args.experts).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    folder = args.output / kind / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    history, start_epoch, best_score, best_epoch, elapsed_seconds = [], 1, -float('inf'), None, 0.
    if args.resume and (folder / 'last.pt').is_file():
        checkpoint = torch.load(folder / 'last.pt', map_location=args.device, weights_only=True)
        _check_checkpoint(checkpoint, protocol, kind, seed)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']
        if [row['epoch'] for row in history] != list(range(1, checkpoint['epoch'] + 1)):
            raise ValueError('Checkpoint epoch/history mismatch')
        start_epoch, best_score, best_epoch = checkpoint['epoch'] + 1, checkpoint['best_score'], checkpoint['best_epoch']
        elapsed_seconds = checkpoint['elapsed_seconds']
        if best_epoch == checkpoint['epoch']:
            fair._save_checkpoint(folder / 'best.pt', checkpoint)
        if not (folder / 'best.pt').is_file():
            raise FileNotFoundError('Previous best.pt is missing and cannot be reconstructed')
        fair._write_history(folder / 'metrics.jsonl', history)
        fair._restore_rng(checkpoint['rng'])
    elif any((folder / name).exists() for name in ('best.pt', 'metrics.jsonl')):
        raise ValueError('Run artifacts exist without last.pt; refusing an ambiguous restart')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    val_loader = _physical_loader(datasets['val'], args, seed)
    final_epoch = min(args.epochs, args.stop_after_epoch or args.epochs)
    for epoch in range(start_epoch, final_epoch + 1):
        indices = domain_draws(datasets['train'].groups, args.quota_per_domain, seed, epoch)
        sampled_images = [datasets['train'].groups[index][0]['image'] for index in indices]
        train_loader = _physical_loader(datasets['train'], args, seed + epoch, indices)
        lr = epoch_lr(args.lr, epoch, args.epochs, args.lr_schedule, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        fair._synchronize(args.device)
        started = time.perf_counter()
        total_loss, image_count, query_count, steps = train_effective_batches(
            model, kind, train_loader, optimizer, args, len(indices), seed, epoch)
        fair._synchronize(args.device)
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        metrics = fair.evaluate(model, kind, val_loader, args.device)
        fair._synchronize(args.device)
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
                       sample_sha256=fair._digest(sampled_images), parameter_count=parameters,
                       optimizer_steps=steps, physical_micro_batches=(image_count + args.micro_batch_size - 1)
                           // args.micro_batch_size,
                       train_seconds=train_seconds, validation_seconds=validation_seconds,
                       elapsed_seconds=elapsed_seconds)
        history.append(metrics)
        checkpoint = dict(protocol_version=PROTOCOL_VERSION, architecture=f'fair_{kind}_unet_accum_v1',
                          kind=kind, seed=seed, config=protocol['config'],
                          config_sha256=protocol['config_sha256'], record_sha256=protocol['record_sha256'],
                          source_sha256=protocol['source_sha256'],
                          task_ids=list(fair.PAPER_ALIGNED_TASK_IDS),
                          structures=[fair.TASKS[i] for i in fair.PAPER_ALIGNED_TASK_IDS],
                          model=model.state_dict(), optimizer=optimizer.state_dict(), rng=fair._rng_state(),
                          epoch=epoch, metrics=metrics, history=history, best_score=best_score,
                          best_epoch=best_epoch, elapsed_seconds=elapsed_seconds,
                          parameter_count=parameters)
        fair._save_checkpoint(folder / 'last.pt', checkpoint)
        if improved:
            fair._save_checkpoint(folder / 'best.pt', checkpoint)
        fair._write_history(folder / 'metrics.jsonl', history)
        print(json.dumps(dict(kind=kind, seed=seed, epoch=epoch,
                              train_loss=metrics['train_loss'],
                              val_macro_positive_dice=score,
                              sample_sha256=metrics['sample_sha256'],
                              optimizer_steps=steps, train_seconds=train_seconds)), flush=True)
    result = dict(kind=kind, seed=seed, parameter_count=parameters,
                  epochs_completed=history[-1]['epoch'] if history else 0,
                  best_epoch=best_epoch,
                  best_validation_macro_positive_dice=best_score if history else None,
                  elapsed_seconds=elapsed_seconds, checkpoint_directory=str(folder), test_evaluated=False)
    if args.evaluate_test:
        if result['epochs_completed'] < args.epochs:
            raise ValueError('Test evaluation requires completing the frozen training budget')
        checkpoint = torch.load(folder / 'best.pt', map_location=args.device, weights_only=True)
        _check_checkpoint(checkpoint, protocol, kind, seed)
        model.load_state_dict(checkpoint['model'])
        test_dataset = fair.RegionDataset([row for row in records if row['split'] == 'test'],
                                          args.size, fair.PAPER_ALIGNED_TASK_IDS)
        if not len(test_dataset):
            raise ValueError('Explicit test evaluation requested but no test images exist')
        test = fair.evaluate(model, kind, _physical_loader(test_dataset, args, seed), args.device)
        test.update(checkpoint_epoch=checkpoint['epoch'],
                    checkpoint_selection='best_validation_macro_positive_dice',
                    record_sha256=protocol['record_sha256'],
                    config_sha256=protocol['config_sha256'])
        fair._write_json(folder / 'test_metrics.json', test)
        result.update(test_evaluated=True, test=test)
    return result


def run_suite(args):
    protocol, records, datasets = _prepare_protocol(args)
    summary = dict(protocol_version=PROTOCOL_VERSION, config_sha256=protocol['config_sha256'],
                   record_sha256=protocol['record_sha256'], dry_run=args.dry_run,
                   test_evaluation_requested=args.evaluate_test, runs=[])
    if args.dry_run:
        if not args.resume:
            fair._write_json(args.output / 'summary.json', summary)
        print(json.dumps(dict(protocol=str(args.output / 'protocol.json'),
                              budget=protocol['budget'])), flush=True)
        return summary
    for seed in args.seeds:
        for kind in args.models:
            summary['runs'].append(_run_model(args, protocol, records, datasets, kind, seed))
            fair._write_json(args.output / 'summary.json', summary)
    validation_aggregate = {}
    for kind in args.models:
        values = [run['best_validation_macro_positive_dice'] for run in summary['runs']
                  if run['kind'] == kind and run['best_validation_macro_positive_dice'] is not None]
        validation_aggregate[kind] = dict(seeds=len(values),
                                          mean=float(np.mean(values)) if values else None,
                                          sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    summary['validation_across_seeds'] = validation_aggregate
    test_aggregate = {}
    for kind in args.models:
        evaluated = [run['test'] for run in summary['runs'] if run['kind'] == kind and run['test_evaluated']]
        if evaluated:
            test_aggregate[kind] = {}
            for field in evaluated[0]:
                if not field.startswith('macro_'):
                    continue
                values = [metrics[field] for metrics in evaluated if metrics[field] is not None]
                test_aggregate[kind][field] = dict(seeds=len(values),
                                                   mean=float(np.mean(values)) if values else None,
                                                   sample_std=float(np.std(values, ddof=1))
                                                       if len(values) > 1 else None)
    summary['test_across_seeds'] = test_aggregate
    fair._write_json(args.output / 'summary.json', summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', default=','.join(fair.MODELS))
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--split-seed', type=int, choices=[42], default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quota-per-domain', type=int, default=500)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16, help='Effective images per optimizer step')
    parser.add_argument('--micro-batch-size', type=int, default=4, help='Physical images per forward/backward')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--experts', type=int, default=9)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--min-lr-ratio', type=float, default=.1)
    parser.add_argument('--coarse-weight', type=float, default=0.)
    parser.add_argument('--route-weight', type=float, default=0.)
    parser.add_argument('--balance-weight', type=float, default=0.)
    parser.add_argument('--router-z-weight', type=float, default=0.)
    parser.add_argument('--empty-weight', type=float, default=0.)
    parser.add_argument('--empty-fraction', type=float, default=.01)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args(argv)
    args.models = args.models.split(',')
    try:
        args.seeds = [int(seed) for seed in args.seeds.split(',')]
    except ValueError:
        parser.error('--seeds must be comma-separated integers')
    if (not args.models or not set(args.models).issubset(fair.MODELS)
            or len(set(args.models)) != len(args.models)
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**32 for seed in args.seeds)):
        parser.error('Models and seeds must be unique valid choices')
    if (args.epochs < 1 or args.quota_per_domain < 1 or args.size < 32
            or args.base < 4 or args.base % 4 or args.batch_size < 1
            or args.micro_batch_size < 1 or args.micro_batch_size > args.batch_size
            or args.batch_size % args.micro_batch_size != 0 or args.workers < 0
            or not 1 <= args.top_k <= min(args.experts, 9)
            or args.lr <= 0 or not np.isfinite(args.lr)
            or args.weight_decay < 0 or not np.isfinite(args.weight_decay)
            or not 0 <= args.min_lr_ratio <= 1
            or not 0 < args.empty_fraction <= 1
            or args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs):
        parser.error('Invalid model, microbatch, optimization, or budget setting')
    if any(getattr(args, name) != 0 for name in ('coarse_weight', 'route_weight',
                                                'balance_weight', 'router_z_weight',
                                                'empty_weight')):
        parser.error('This fair accumulation protocol fixes all auxiliary losses to zero')
    if args.evaluate_test and args.stop_after_epoch and args.stop_after_epoch < args.epochs:
        parser.error('--evaluate-test cannot be combined with an incomplete training budget')
    return args


def main(argv=None):
    run_suite(parse_args(argv))


if __name__ == '__main__':
    main()
