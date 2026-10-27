"""One-output, image-only U-Net for the union of available paper5 foregrounds.

This is a different target from the 16-structure fair comparison. Its Dice is
never a per-organ Dice and must not be ranked against 16-channel macro Dice.
Test PNGs are opened only with the explicit --evaluate-test flag, after all
configured training seeds have completed their planned epoch budget.
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
from PIL import Image

from .baseline import PlainUNet
from .data import build_records, validate_splits
from .fair_suite import (_digest, _loader, _restore_rng, _rng_state, _save_checkpoint,
                         _seed, _synchronize, _write_history, _write_json, common_loss)
from .region import DOMAINS, PAPER_ALIGNED_TASK_IDS, RegionDataset
from .sampling import domain_draws
from .schedule import epoch_lr


PROTOCOL_VERSION = 'single_channel_paper5_union_v1'
EXCLUDED_ABDOMEN_CLASSES = (3, 5, 7, 255)


def partial_union_images(records):
    """List images with mass annotation but no AUL liver/outline annotation."""
    by_image = {}
    for row in records:
        by_image.setdefault((row['dataset'], row['image']), []).append(row)
    return [dict(dataset=domain, image=image, split=rows[0]['split'],
                 tasks=sorted({row['task'] for row in rows}),
                 known_rule='mass positive is foreground; all other pixels unknown')
            for (domain, image), rows in sorted(by_image.items())
            if domain == 'AUL' and {row['task'] for row in rows} == {4}]


def union_target(target, abdomen_indexed_mask=None, unknown_background=False):
    """Convert C,H,W or B,C,H,W partial channel labels into one union mask.

    Any annotated positive proves union foreground. Union background is known
    only where *every active annotated channel* is valid and negative. Absent
    channels (entirely -1) do not constrain the union. The indexed abdomen
    mask optionally hides excluded paper5 classes 3/5/7 and void 255.
    """
    if target.ndim not in (3, 4):
        raise ValueError('Expected C,H,W or B,C,H,W target')
    if not target.is_floating_point():
        raise ValueError('Expected floating point targets with -1 ignore')
    active = target.ge(0).any((-2, -1))
    if not active.any(-1).all():
        raise ValueError('An image has no active annotated channel')
    expanded = active[..., None, None]
    positive = (target.gt(0) & expanded).any(-3)
    fully_known_negative = ((target.ge(0) | ~expanded).all(-3) & ~positive)
    result = torch.full_like(positive, -1, dtype=target.dtype)
    result[positive] = 1
    if not unknown_background:
        result[fully_known_negative] = 0
    if abdomen_indexed_mask is not None:
        indexed = torch.as_tensor(abdomen_indexed_mask, device=target.device)
        if indexed.shape != result.shape:
            raise ValueError('Indexed abdomen mask must match spatial/batch target shape')
        ignored = torch.zeros_like(indexed, dtype=torch.bool)
        for value in EXCLUDED_ABDOMEN_CLASSES:
            ignored |= indexed.eq(value)
        result[ignored] = -1
    return result.unsqueeze(-3)


class UnionDataset(torch.utils.data.Dataset):
    """Exactly one example per image, with a single paper5 union target."""

    def __init__(self, records, size=512, return_multi_target=False):
        self.region = RegionDataset(records, size, PAPER_ALIGNED_TASK_IDS)
        self.groups = self.region.groups  # Required by the fair domain_draws sampler.
        self.size = size
        self.return_multi_target = return_multi_target

    def __len__(self):
        return len(self.region)

    def __getitem__(self, index):
        rows = self.groups[index]
        image, multi_target = self.region[index]
        domain = rows[0]['dataset']
        source = rows[0]['group'].split('/')[-1] if domain == 'AbdomenUS' else ''
        indexed = None
        if domain == 'AbdomenUS':
            masks = {row['mask'] for row in rows}
            if len(masks) != 1:
                raise ValueError('Abdomen channels for an image must share one indexed mask')
            with Image.open(next(iter(masks))) as mask:
                if mask.mode not in ('L', 'P'):
                    raise ValueError('Expected indexed abdomen mask')
                indexed = np.array(mask.resize((self.size, self.size), Image.Resampling.NEAREST), copy=True)
        mass_only_aul = domain == 'AUL' and {row['task'] for row in rows} == {4}
        target = union_target(multi_target, indexed, unknown_background=mass_only_aul)
        if self.return_multi_target:
            return image, target, domain, source, multi_target
        return image, target, domain, source


def _new_totals():
    return dict(samples=0, positive_samples=0, empty_samples=0, dice_sum=0.,
                positive_dice_sum=0., empty_fp=0, empty_fp_pixels=0.,
                empty_fp_fraction=0., positive_miss=0, positive_zero_overlap=0)


def _add_sample(row, truth, pred, valid):
    truth_pixels = int(truth.sum())
    pred_pixels = int(pred.sum())
    intersection = int((pred & truth).sum())
    dice = 2 * intersection / (truth_pixels + pred_pixels) if truth_pixels + pred_pixels else 1.
    row['samples'] += 1
    row['dice_sum'] += dice
    if truth_pixels:
        row['positive_samples'] += 1
        row['positive_dice_sum'] += dice
        row['positive_miss'] += int(pred_pixels == 0)
        row['positive_zero_overlap'] += int(intersection == 0)
    else:
        row['empty_samples'] += 1
        row['empty_fp'] += int(pred_pixels > 0)
        row['empty_fp_pixels'] += pred_pixels
        row['empty_fp_fraction'] += pred_pixels / int(valid.sum())


def _finish_totals(row):
    n, p, e = row['samples'], row['positive_samples'], row['empty_samples']
    return dict(samples=n, positive_samples=p, empty_samples=e,
                all_union_dice=row['dice_sum'] / n if n else None,
                positive_union_dice=row['positive_dice_sum'] / p if p else None,
                empty_false_positive_rate=row['empty_fp'] / e if e else None,
                empty_mean_false_positive_pixels=row['empty_fp_pixels'] / e if e else None,
                empty_mean_false_positive_area_fraction=row['empty_fp_fraction'] / e if e else None,
                positive_miss_rate=row['positive_miss'] / p if p else None,
                positive_zero_overlap_rate=row['positive_zero_overlap'] / p if p else None)


@torch.no_grad()
def evaluate(model, loader, device):
    """One union prediction/image; overall, nine-domain, and abdomen CT metrics."""
    model.eval()
    overall = _new_totals()
    per_domain = {domain: _new_totals() for domain in DOMAINS}
    per_abdomen_ct = {}
    for image, target, domains, sources in loader:
        image, target = image.to(device), target.to(device)
        valid, truth = target.ge(0), target.gt(0)
        if not valid.any((1, 2, 3)).all():
            raise ValueError('Evaluation image has no valid union pixels')
        logits = model(image)
        if logits.shape != target.shape:
            raise ValueError('Single-channel output shape does not match union target')
        pred = logits.sigmoid().ge(.5) & valid
        for index, domain in enumerate(domains):
            if domain not in per_domain:
                raise ValueError(f'Unknown domain {domain}')
            for row in (overall, per_domain[domain]):
                _add_sample(row, truth[index, 0], pred[index, 0], valid[index, 0])
            if domain == 'AbdomenUS':
                row = per_abdomen_ct.setdefault(sources[index], _new_totals())
                _add_sample(row, truth[index, 0], pred[index, 0], valid[index, 0])
    domains = {name: _finish_totals(row) for name, row in per_domain.items()}
    positive = [row['positive_union_dice'] for row in domains.values()
                if row['positive_union_dice'] is not None]
    all_dice = [row['all_union_dice'] for row in domains.values() if row['all_union_dice'] is not None]
    return dict(target='paper5_available_foreground_union', threshold=.5,
                overall=_finish_totals(overall), per_domain=domains,
                per_abdomen_ct={name: _finish_totals(row) for name, row in sorted(per_abdomen_ct.items())},
                macro_domain_positive_union_dice=sum(positive) / len(positive) if positive else None,
                macro_domain_all_union_dice=sum(all_dice) / len(all_dice) if all_dice else None,
                positive_domains=len(positive))


def _hash_files(paths):
    manifest = []
    for path in sorted({str(Path(path).resolve()) for path in paths}):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        manifest.append(dict(path=path, sha256=digest.hexdigest()))
    return manifest, _digest(manifest)


def _content_paths(records, splits):
    return [row[key] for row in records if row['split'] in splits for key in ('image', 'mask')]


def _reference_draws(args):
    """Read the completed Region fair shard as the paired draw reference."""
    if args.reference_records is None:
        return None
    shard = args.reference_records.parent
    fair_protocol = json.loads((shard / 'protocol.json').read_text())
    fair_config = fair_protocol['config']
    for key in ('size', 'base', 'batch_size', 'workers', 'quota_per_domain', 'epochs',
                'split_seed', 'lr', 'weight_decay', 'lr_schedule', 'min_lr_ratio'):
        if fair_config[key] != getattr(args, key):
            raise ValueError(f'{key} differs from fair reference protocol')
    if fair_config['seeds'] != args.seeds:
        raise ValueError('Training seeds differ from fair reference protocol')
    result = {}
    for seed in args.seeds:
        history_path = shard / 'region' / f'seed_{seed}' / 'metrics.jsonl'
        if not history_path.is_file():
            raise FileNotFoundError(f'Completed fair Region history required: {history_path}')
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        if [row['epoch'] for row in rows] != list(range(1, args.epochs + 1)):
            raise ValueError(f'Fair Region seed {seed} has an incomplete or nonsequential history')
        result[str(seed)] = {str(row['epoch']): dict(sample_sha256=row['sample_sha256'],
            sample_count=row['sample_count'],
            annotated_channel_queries=row['annotated_channel_queries']) for row in rows}
    return result


def _config(args):
    operational = {'output', 'resume', 'evaluate_test', 'dry_run', 'stop_after_epoch', 'device'}
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in operational}
    config.update(protocol_version=PROTOCOL_VERSION, architecture='PlainUNet(outputs=1)',
                  split_policy='trainmore_v6', abdomen_split='ct7_2',
                  label_scope='paper5 (abdomen indexed classes 1,2,4,6,8)',
                  union_rule='positive if any active annotated channel positive; background if all active annotated channels valid and negative; otherwise ignore',
                  excluded_abdomen_classes=list(EXCLUDED_ABDOMEN_CLASSES),
                  partial_annotation_policy='AUL mass-only images remain in draws: mass positive pixels are foreground and other pixels ignored',
                  objective='mean_images(BCE_valid_pixels + soft_Dice_smooth_1) on one union target',
                  optimizer=dict(name='AdamW', betas=[.9, .999], eps=1e-8, weight_decay=args.weight_decay),
                  gradient_clip_norm=5., threshold=.5,
                  checkpoint_selection='validation_macro_9_domain_positive_union_dice',
                  comparison_warning='Union Dice is a different target and cannot be directly ranked against 16-structure Dice')
    return config


def _prepare(args):
    records, skipped = build_records(args.data_root, args.split_seed, 'ct7_2', None, 'trainmore_v6')
    validate_splits(records)
    excluded_tasks = Counter(f"{row['dataset']}:task_{row['task']}" for row in records
                             if row['task'] not in PAPER_ALIGNED_TASK_IDS)
    records = [row for row in records if row['task'] in PAPER_ALIGNED_TASK_IDS]
    partial_images = partial_union_images(records)
    records = sorted(records, key=lambda row: json.dumps(row, sort_keys=True))
    if args.reference_records is not None:
        reference = json.loads(args.reference_records.read_text())
        if _digest(reference) != _digest(records):
            raise ValueError('Records differ from reference fair records')
    reference_draws = _reference_draws(args)
    config = _config(args)
    sources = ['single_channel_suite.py', 'fair_suite.py', 'baseline.py', 'model.py',
               'region.py', 'data.py', 'sampling.py', 'schedule.py', 'empty_loss.py']
    source_files = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in sources}
    content_manifest, content_hash = _hash_files(_content_paths(records, {'train', 'val'}))
    protocol = dict(config=config, config_sha256=_digest(config),
                    record_sha256=_digest(records), source_files=source_files,
                    source_sha256=_digest(source_files), content_sha256=content_hash,
                    content_hash_scope='unique train/val image and mask PNG file bytes; test bytes only at explicit test evaluation',
                    skipped=skipped, excluded_tasks=dict(excluded_tasks),
                    partial_annotation_images=partial_images,
                    reference_records_sha256=_digest(json.loads(args.reference_records.read_text()))
                        if args.reference_records is not None else None,
                    reference_draws=reference_draws,
                    reference_draws_sha256=_digest(reference_draws) if reference_draws is not None else None,
                    budget=dict(images_per_epoch=len(DOMAINS) * args.quota_per_domain,
                                optimizer_steps_per_epoch=(len(DOMAINS) * args.quota_per_domain + args.batch_size - 1) // args.batch_size,
                                planned_epochs=args.epochs,
                                planned_images_per_seed=len(DOMAINS) * args.quota_per_domain * args.epochs,
                                seeds=args.seeds),
                    environment=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                                     numpy=np.__version__, cuda=torch.version.cuda),
                    test_policy='held out; no test image/mask content read before explicit --evaluate-test after all seeds finish')
    try:
        protocol['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        protocol['git_commit'] = None
    datasets = {split: UnionDataset([row for row in records if row['split'] == split], args.size,
                                   return_multi_target=(split == 'train'))
                for split in ('train', 'val')}
    for split in ('train', 'val'):
        counts = Counter(rows[0]['dataset'] for rows in datasets[split].groups)
        if set(counts) != set(DOMAINS):
            raise ValueError(f'{split} must contain all nine domains; found {sorted(counts)}')
        protocol[f'{split}_images_by_domain'] = dict(counts)
    protocol['test_images_by_domain'] = dict(Counter(row['dataset'] for row in {
        row['image']: row for row in records if row['split'] == 'test'}.values()))
    if any(row.get('label_version') != 'hc18_filled_v1' for row in records if row['dataset'] == 'Fetal_HC'):
        raise ValueError('Fetal_HC requires filled hc18_filled_v1 masks')
    if args.resume:
        previous_path = args.output / 'protocol.json'
        if not previous_path.is_file():
            raise FileNotFoundError('--resume requires an existing protocol.json')
        previous = json.loads(previous_path.read_text())
        for key in ('config_sha256', 'record_sha256', 'source_sha256', 'content_sha256',
                    'reference_draws_sha256'):
            if previous.get(key) != protocol[key]:
                raise ValueError(f'{key} differs from frozen single-channel protocol')
        if _digest(json.loads((args.output / 'records.json').read_text())) != protocol['record_sha256']:
            raise ValueError('Stored records.json differs from frozen records')
        if _digest(json.loads((args.output / 'content_manifest.json').read_text())) != content_hash:
            raise ValueError('Stored content manifest differs from frozen train/val contents')
    else:
        if args.output.exists():
            raise FileExistsError('Output exists; use --resume or a new output directory')
        args.output.mkdir(parents=True)
        _write_json(args.output / 'records.json', records)
        _write_json(args.output / 'content_manifest.json', content_manifest)
        _write_json(args.output / 'protocol.json', protocol)
    return protocol, records, datasets


def _check_checkpoint(checkpoint, protocol, seed):
    for key in ('config_sha256', 'record_sha256', 'source_sha256', 'content_sha256'):
        if checkpoint.get(key) != protocol[key]:
            raise ValueError(f'Checkpoint {key} differs from frozen protocol')
    if checkpoint.get('seed') != seed or checkpoint.get('architecture') != 'single_channel_plain_unet_v1':
        raise ValueError('Checkpoint architecture or seed mismatch')


def _run_seed(args, protocol, datasets, seed):
    _seed(seed)
    model = PlainUNet(outputs=1, base=args.base).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    folder = args.output / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    history, start_epoch, best_score, best_epoch, elapsed = [], 1, -float('inf'), None, 0.
    if args.resume and (folder / 'last.pt').is_file():
        checkpoint = torch.load(folder / 'last.pt', map_location=args.device, weights_only=True)
        _check_checkpoint(checkpoint, protocol, seed)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']
        if not history or [row['epoch'] for row in history] != list(range(1, checkpoint['epoch'] + 1)):
            raise ValueError('Checkpoint epoch/history mismatch')
        start_epoch, best_score, best_epoch, elapsed = (checkpoint['epoch'] + 1,
            checkpoint['best_score'], checkpoint['best_epoch'], checkpoint['elapsed_seconds'])
        if best_epoch == checkpoint['epoch']:
            _save_checkpoint(folder / 'best.pt', checkpoint)
        if not (folder / 'best.pt').is_file():
            raise FileNotFoundError('Previous best.pt is missing')
        _write_history(folder / 'metrics.jsonl', history)
        _restore_rng(checkpoint['rng'])
    elif any((folder / name).exists() for name in ('best.pt', 'metrics.jsonl')):
        raise ValueError('Run artifacts exist without last.pt; refusing ambiguous restart')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    val_loader = _loader(datasets['val'], args, seed)
    end_epoch = min(args.epochs, args.stop_after_epoch or args.epochs)
    for epoch in range(start_epoch, end_epoch + 1):
        indices = domain_draws(datasets['train'].groups, args.quota_per_domain, seed, epoch)
        sampled_images = [datasets['train'].groups[index][0]['image'] for index in indices]
        sample_hash = _digest(sampled_images)
        reference = (protocol['reference_draws'] or {}).get(str(seed), {}).get(str(epoch))
        if reference is not None and (sample_hash != reference['sample_sha256'] or
                                      len(indices) != reference['sample_count']):
            raise ValueError(f'Seed {seed} epoch {epoch} image draws differ from fair reference')
        train_loader = _loader(datasets['train'], args, seed + epoch, indices)
        lr = epoch_lr(args.lr, epoch, args.epochs, args.lr_schedule, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        total_loss, count = 0., 0
        _synchronize(args.device)
        started = time.perf_counter()
        query_count = 0
        for image, target, _, _, multi_target in train_loader:
            image, target = image.to(args.device), target.to(args.device)
            if not target.ge(0).any((1, 2, 3)).all():
                raise ValueError('Sampled image has no valid union pixels')
            logits = model(image)
            loss = common_loss(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in seed {seed}, epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(image)
            count += len(image)
            query_count += int(multi_target.ge(0).any((2, 3)).sum())
        _synchronize(args.device)
        train_seconds = time.perf_counter() - started
        if reference is not None and query_count != reference['annotated_channel_queries']:
            raise ValueError(f'Seed {seed} epoch {epoch} annotated channel count differs from fair reference')
        started = time.perf_counter()
        metrics = evaluate(model, val_loader, args.device)
        _synchronize(args.device)
        validation_seconds = time.perf_counter() - started
        elapsed += train_seconds + validation_seconds
        if metrics['positive_domains'] != len(DOMAINS):
            raise ValueError('All nine validation domains need positive union examples')
        score = metrics['macro_domain_positive_union_dice']
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        metrics.update(epoch=epoch, seed=seed, train_loss=total_loss / count, lr=lr,
                       sample_count=count, annotated_channel_queries=query_count,
                       sample_sha256=sample_hash,
                       parameter_count=parameters, train_seconds=train_seconds,
                       validation_seconds=validation_seconds, elapsed_seconds=elapsed)
        history.append(metrics)
        checkpoint = dict(protocol_version=PROTOCOL_VERSION, architecture='single_channel_plain_unet_v1',
                          seed=seed, config=protocol['config'], config_sha256=protocol['config_sha256'],
                          record_sha256=protocol['record_sha256'], source_sha256=protocol['source_sha256'],
                          content_sha256=protocol['content_sha256'], model=model.state_dict(),
                          optimizer=optimizer.state_dict(), rng=_rng_state(), epoch=epoch,
                          metrics=metrics, history=history, best_score=best_score,
                          best_epoch=best_epoch, elapsed_seconds=elapsed, parameter_count=parameters)
        _save_checkpoint(folder / 'last.pt', checkpoint)
        if improved:
            _save_checkpoint(folder / 'best.pt', checkpoint)
        _write_history(folder / 'metrics.jsonl', history)
        print(json.dumps(dict(seed=seed, epoch=epoch, train_loss=metrics['train_loss'],
                              val_macro_domain_positive_union_dice=score,
                              sample_sha256=metrics['sample_sha256'], train_seconds=train_seconds)), flush=True)
    best_metrics = next((row for row in history if row['epoch'] == best_epoch), None)
    return dict(seed=seed, parameter_count=parameters,
                epochs_completed=history[-1]['epoch'] if history else 0,
                best_epoch=best_epoch,
                best_validation_macro_domain_positive_union_dice=best_score if history else None,
                best_validation_overall_positive_union_dice=(best_metrics['overall']['positive_union_dice']
                    if best_metrics else None),
                best_validation_overall_all_union_dice=(best_metrics['overall']['all_union_dice']
                    if best_metrics else None),
                elapsed_seconds=elapsed, checkpoint_directory=str(folder), test_evaluated=False)


def _aggregate(runs, key):
    values = [run[key] for run in runs if run.get(key) is not None]
    return dict(seeds=len(values), mean=float(np.mean(values)) if values else None,
                sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None)


def _evaluate_test(args, protocol, records, runs):
    if any(run['epochs_completed'] != args.epochs for run in runs):
        raise ValueError('Test requires every seed to finish the planned training budget')
    test_records = [row for row in records if row['split'] == 'test']
    manifest, content_hash = _hash_files(_content_paths(test_records, {'test'}))
    test_manifest_path = args.output / 'test_content_manifest.json'
    if test_manifest_path.exists():
        if _digest(json.loads(test_manifest_path.read_text())) != content_hash:
            raise ValueError('Test PNG contents changed since previous evaluation')
    else:
        _write_json(test_manifest_path, manifest)
    dataset = UnionDataset(test_records, args.size)
    if not len(dataset):
        raise ValueError('No held-out test images')
    for run in runs:
        seed = run['seed']
        folder = args.output / f'seed_{seed}'
        test_path = folder / 'test_metrics.json'
        if test_path.exists():
            test = json.loads(test_path.read_text())
            if test.get('test_content_sha256') != content_hash:
                raise ValueError('Existing test metrics use different test content')
        else:
            checkpoint = torch.load(folder / 'best.pt', map_location=args.device, weights_only=True)
            _check_checkpoint(checkpoint, protocol, seed)
            model = PlainUNet(outputs=1, base=args.base).to(args.device)
            model.load_state_dict(checkpoint['model'])
            test = evaluate(model, _loader(dataset, args, seed), args.device)
            test.update(checkpoint_epoch=checkpoint['epoch'],
                        checkpoint_selection='validation_macro_9_domain_positive_union_dice',
                        record_sha256=protocol['record_sha256'],
                        config_sha256=protocol['config_sha256'],
                        test_content_sha256=content_hash)
            _write_json(test_path, test)
        run.update(test_evaluated=True, test=test)


def run_suite(args):
    protocol, records, datasets = _prepare(args)
    summary = dict(protocol_version=PROTOCOL_VERSION, target='paper5_available_foreground_union',
                   config_sha256=protocol['config_sha256'], record_sha256=protocol['record_sha256'],
                   content_sha256=protocol['content_sha256'], dry_run=args.dry_run,
                   test_evaluation_requested=args.evaluate_test, runs=[])
    if args.dry_run:
        if not args.resume:
            _write_json(args.output / 'summary.json', summary)
        print(json.dumps(dict(protocol=str(args.output / 'protocol.json'), budget=protocol['budget'])), flush=True)
        return summary
    for seed in args.seeds:
        summary['runs'].append(_run_seed(args, protocol, datasets, seed))
        _write_json(args.output / 'summary.json', summary)
    summary['validation_across_seeds'] = _aggregate(
        summary['runs'], 'best_validation_macro_domain_positive_union_dice')
    summary['validation_overall_across_seeds'] = {
        key: _aggregate(summary['runs'], f'best_validation_{key}')
        for key in ('overall_positive_union_dice', 'overall_all_union_dice')}
    if args.evaluate_test:
        _evaluate_test(args, protocol, records, summary['runs'])
    summary['test_across_seeds'] = {
        key: _aggregate([run['test'] for run in summary['runs'] if run['test_evaluated']], key)
        for key in ('macro_domain_positive_union_dice', 'macro_domain_all_union_dice')
    } if args.evaluate_test else {}
    if args.evaluate_test:
        for key in ('positive_union_dice', 'all_union_dice'):
            summary['test_across_seeds'][f'overall_{key}'] = _aggregate(
                [run['test']['overall'] for run in summary['runs'] if run['test_evaluated']], key)
    _write_json(args.output / 'summary.json', summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-records', type=Path,
                        help='Optional fair shard records.json; checks exact records and draw hashes')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--split-seed', type=int, choices=[42], default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quota-per-domain', type=int, default=500)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--min-lr-ratio', type=float, default=.1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        args.seeds = [int(seed) for seed in args.seeds.split(',')]
    except ValueError:
        parser.error('--seeds must be comma-separated integers')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('--seeds must be unique integers in 0..2**32-1')
    if (args.epochs < 1 or args.quota_per_domain < 1 or args.size < 32 or args.base < 4 or args.base % 4
            or args.batch_size < 1 or args.workers < 0 or args.lr <= 0 or not np.isfinite(args.lr)
            or args.weight_decay < 0 or not np.isfinite(args.weight_decay)
            or not 0 <= args.min_lr_ratio <= 1 or args.stop_after_epoch < 0
            or args.stop_after_epoch > args.epochs):
        parser.error('Invalid model, optimization, or budget setting')
    if args.evaluate_test and (args.dry_run or (args.stop_after_epoch and args.stop_after_epoch < args.epochs)):
        parser.error('--evaluate-test requires a complete training budget')
    return args


def main(argv=None):
    run_suite(parse_args(argv))


if __name__ == '__main__':
    main()
