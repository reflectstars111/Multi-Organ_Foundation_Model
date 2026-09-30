"""Train image-only region MoE; independent of the legacy task-ID experiments."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from .data import build_records
from .region import (RegionDataset, RegionUNet, region_loss, STRUCTURES, DOMAINS,
                     TASK_IDS, PAPER_ALIGNED_TASK_IDS)
from .sampling import domain_draws
from .schedule import epoch_lr


def abdomen_positive_weights(dataset, max_weight=8.):
    """Balance positive/empty images per abdominal class using training masks only."""
    groups = [rows for rows in dataset.groups if rows[0]['dataset'] == 'AbdomenUS']
    if not groups:
        raise ValueError('AbdomenUS training images are required')
    task_ids = list(getattr(dataset, 'task_ids', TASK_IDS))
    abdomen_classes = [(task, task - 12) for task in task_ids if 13 <= task <= 20]
    counts = np.zeros(len(abdomen_classes), dtype=np.int64)
    for rows in groups:
        if len({row['mask'] for row in rows}) != 1:
            raise ValueError('AbdomenUS labels for one image must share an indexed mask')
        with Image.open(rows[0]['mask']) as mask:
            indices = np.asarray(mask.resize((dataset.size, dataset.size), Image.Resampling.NEAREST))
        if indices.ndim != 2 or not set(np.unique(indices)).issubset(set(range(9)) | {255}):
            raise ValueError('Unexpected AbdomenUS class indices')
        present = set(np.unique(indices))
        counts += np.array([class_id in present for _, class_id in abdomen_classes], dtype=np.int64)
    if (counts == 0).any():
        raise ValueError('At least one abdominal class has no positive training image')
    weights = torch.ones(len(task_ids), dtype=torch.float32)
    stats = {}
    for (task, _), count in zip(abdomen_classes, counts):
        name = STRUCTURES[TASK_IDS.index(task)]
        weight = min(max_weight, max(1., (len(groups) - int(count)) / int(count)))
        weights[task_ids.index(task)] = weight
        stats[name] = dict(positive_images=int(count), images=len(groups), positive_weight=weight)
    return weights, stats


def domain_sampling_weights(dataset, positive_weights, abdomen_strength=0.):
    """Equal domain mass; optionally equal CT mass with foreground-aware abdominal draws."""
    if abdomen_strength < 0:
        raise ValueError('AbdomenUS sampling strength must be nonnegative')
    domain_counts = Counter(rows[0]['dataset'] for rows in dataset.groups)
    weights = [1 / domain_counts[rows[0]['dataset']] for rows in dataset.groups]
    if not abdomen_strength:
        return weights
    ct_scores = {}
    for index, rows in enumerate(dataset.groups):
        if rows[0]['dataset'] != 'AbdomenUS':
            continue
        with Image.open(rows[0]['mask']) as mask:
            indices = np.asarray(mask.resize((dataset.size, dataset.size), Image.Resampling.NEAREST))
        present = set(np.unique(indices))
        excess = [float(positive_weights[dataset.task_ids.index(task)]) - 1
                  for task in dataset.task_ids if 13 <= task <= 20 and task - 12 in present]
        score = 1 + abdomen_strength * max(excess, default=0.)
        ct_scores.setdefault(rows[0]['group'], []).append((index, score))
    for members in ct_scores.values():
        total = sum(score for _, score in members)
        for index, score in members:
            weights[index] = score / total / len(ct_scores)
    return weights


@torch.no_grad()
def evaluate(model, loader, device, structures=None):
    model.eval()
    structures = STRUCTURES if structures is None else structures
    all_scores = [[] for _ in structures]
    positive_scores = [[] for _ in structures]
    empty_false_positives = [[] for _ in structures]
    for image, target in loader:
        target = target.to(device)
        valid = target >= 0
        pred = (model(image.to(device))['logits'].sigmoid() >= .5) & valid
        truth = target > 0
        denominator = pred.sum((2, 3)) + truth.sum((2, 3))
        scores = torch.where(denominator > 0, 2*(pred & truth).sum((2, 3))/denominator.clamp_min(1), 1.)
        for b, c in valid.any((2, 3)).nonzero().tolist():
            all_scores[c].append(scores[b, c].item())
            if truth[b, c].any():
                positive_scores[c].append(scores[b, c].item())
            else:
                empty_false_positives[c].append(float(pred[b, c].any().item()))
    per_class = {name: dict(dice=float(np.mean(a)) if a else None,
                           positive_dice=float(np.mean(p)) if p else None,
                           samples=len(a), positive_samples=len(p),
                           empty_samples=len(e),
                           empty_false_positive_rate=float(np.mean(e)) if e else None)
                 for name, a, p, e in zip(structures, all_scores, positive_scores,
                                          empty_false_positives)}
    values = [v['positive_dice'] for v in per_class.values() if v['positive_dice'] is not None]
    all_values = [v['dice'] for v in per_class.values() if v['dice'] is not None]
    empty_values = [v['empty_false_positive_rate'] for v in per_class.values()
                    if v['empty_false_positive_rate'] is not None]
    return dict(macro_positive_dice=float(np.mean(values)) if values else None,
                macro_all_dice=float(np.mean(all_values)) if all_values else None,
                macro_empty_false_positive_rate=float(np.mean(empty_values)) if empty_values else None,
                per_structure=per_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--steps-per-epoch', type=int, default=0)
    parser.add_argument('--quota-per-domain', type=int, default=0,
                        help='Deterministic per-domain draws per epoch; 0 keeps legacy sampler')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--min-lr-ratio', type=float, default=.1)
    parser.add_argument('--coarse-weight', type=float, default=.3)
    parser.add_argument('--route-weight', type=float, default=.1)
    parser.add_argument('--max-positive-weight', type=float, default=8.,
                        help='Cap on rare abdominal positive-image loss weight')
    parser.add_argument('--abdomen-split', choices=('legacy', 'ct7_2', 'ct_cv5', 'ct_final'), default='legacy')
    parser.add_argument('--abdomen-fold', type=int)
    parser.add_argument('--split-policy', choices=('legacy', 'trainmore_v6'), default='legacy')
    parser.add_argument('--abdomen-labels', choices=('all8', 'paper5'), default='all8')
    parser.add_argument('--abdomen-positive-sampling', type=float, default=0.,
                        help='Training-only within-CT foreground sampling strength')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.epochs < 1 or args.size < 32 or args.batch_size < 1 or
            args.steps_per_epoch < 0 or args.quota_per_domain < 0 or args.max_positive_weight < 1 or
            args.abdomen_positive_sampling < 0 or args.coarse_weight < 0 or
            args.route_weight < 0 or not 0 <= args.min_lr_ratio <= 1):
        parser.error('Invalid epochs, size, batch size or steps')
    if (args.abdomen_split == 'ct_cv5') != (args.abdomen_fold is not None):
        parser.error('--abdomen-fold is required only for ct_cv5')
    if args.abdomen_fold is not None and not 0 <= args.abdomen_fold < 5:
        parser.error('--abdomen-fold must be 0..4')
    if args.quota_per_domain and args.steps_per_epoch:
        parser.error('Use either quota-per-domain or steps-per-epoch')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a new output directory (or --resume)')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records, skipped = build_records(args.data_root, args.seed, args.abdomen_split,
                                     args.abdomen_fold, args.split_policy)
    hc = [r for r in records if r['dataset'] == 'Fetal_HC']
    if len(hc) != 999 or any(r['label_version'] != 'hc18_filled_v1' for r in hc):
        raise ValueError('Expected all 999 HC18 records with filled masks')
    task_ids = TASK_IDS if args.abdomen_labels == 'all8' else PAPER_ALIGNED_TASK_IDS
    excluded = Counter(f"{r['dataset']}:region_excluded_task_{r['task']}" for r in records if r['task'] not in task_ids)
    datasets = {split: RegionDataset([r for r in records if r['split'] == split], args.size, task_ids)
                for split in ['train', 'val', 'test']}
    if not len(datasets['train']) or not len(datasets['val']):
        raise ValueError('Empty train or validation split')
    positive_weights, abdomen_stats = abdomen_positive_weights(datasets['train'], args.max_positive_weight)
    used = [r for r in records if r['task'] in task_ids]
    digest = hashlib.sha256(json.dumps(used, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    config = dict(base=args.base, top_k=args.top_k, task_ids=task_ids)
    model = RegionUNet(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = domain_sampling_weights(datasets['train'], positive_weights,
                                      args.abdomen_positive_sampling)
    positive_weights = positive_weights.to(args.device)
    sampler = (WeightedRandomSampler(weights, args.steps_per_epoch*args.batch_size or len(weights),
                                    replacement=True) if not args.quota_per_domain else None)
    loaders = {s: DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
               for s, ds in datasets.items() if s != 'train'}
    if sampler is not None:
        loaders['train'] = DataLoader(datasets['train'], batch_size=args.batch_size,
                                      num_workers=args.workers, sampler=sampler)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    training_config = dict(base=args.base, top_k=args.top_k, task_ids=task_ids,
                           size=args.size, epochs=args.epochs, batch_size=args.batch_size,
                           quota_per_domain=args.quota_per_domain, steps_per_epoch=args.steps_per_epoch,
                           seed=args.seed, lr=args.lr, lr_schedule=args.lr_schedule,
                           min_lr_ratio=args.min_lr_ratio, coarse_weight=args.coarse_weight,
                           route_weight=args.route_weight,
                           max_positive_weight=args.max_positive_weight,
                           abdomen_split=args.abdomen_split, abdomen_labels=args.abdomen_labels,
                           abdomen_fold=args.abdomen_fold, split_policy=args.split_policy,
                           abdomen_positive_sampling=args.abdomen_positive_sampling,
                           loss='per_image_valid_channel_bce_dice_v2',
                           abdomen_positive_balance=abdomen_stats, record_sha256=digest)
    if args.resume:
        saved = json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        if saved != training_config:
            raise ValueError('Resume configuration or frozen records changed')
        completion = 'cv_complete.json' if args.abdomen_split == 'ct_cv5' else 'test_metrics.json'
        if (args.output/completion).exists():
            print('Region MoE already complete; skipping', flush=True)
            return
    else:
        (args.output/'config.json').write_text(json.dumps(training_config, indent=2), encoding='utf-8')
        (args.output/'records.json').write_text(json.dumps(dict(records=used, skipped=skipped,
            region_excluded=dict(excluded), split_version=records[0]['split_version'], sha256=digest),
            ensure_ascii=False), encoding='utf-8')
    print('images per split:', {s:len(ds) for s,ds in datasets.items()}, flush=True)
    print('AbdomenUS positive-image balance:', json.dumps(abdomen_stats), flush=True)
    best = -1
    start_epoch = 1
    if args.resume and (args.output/'last.pt').exists():
        checkpoint = torch.load(args.output/'last.pt', map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_checkpoint = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
        best = best_checkpoint['metrics']['macro_positive_dice']
        print(f'Region MoE resuming at epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, args.epochs+1):
        lr = epoch_lr(args.lr, epoch, args.epochs, args.lr_schedule, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group['lr'] = lr
        if args.quota_per_domain:
            indices = domain_draws(datasets['train'].groups, args.quota_per_domain,
                                   args.seed, epoch, weights=weights)
            train_loader = DataLoader(datasets['train'], batch_size=args.batch_size,
                                      num_workers=args.workers, sampler=indices)
        else:
            train_loader = loaders['train']
        model.train()
        total = 0.; loads = []
        for step, (image, target) in enumerate(train_loader, 1):
            output = model(image.to(args.device))
            loss = region_loss(output, target.to(args.device), positive_weights,
                               model.expert_for_region, args.coarse_weight, args.route_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item(); loads.append(output['expert_load'].detach().cpu())
            if step % 50 == 0:
                print(f'epoch={epoch} step={step}/{len(train_loader)} loss={total/step:.4f}', flush=True)
        metrics = evaluate(model, loaders['val'], args.device, model.structures)
        score = metrics['macro_positive_dice']
        if score is None:
            raise ValueError('No foreground validation samples')
        metrics.update(epoch=epoch, train_loss=total/step, lr=lr,
                       expert_load=torch.stack(loads).mean(0).tolist())
        with (args.output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics)+'\n')
        checkpoint = dict(architecture='region_unet_v1', model=model.state_dict(), config=config,
                          structures=model.structures, domains=DOMAINS, size=args.size, epoch=epoch,
                          optimizer=optimizer.state_dict(), metrics=metrics)
        torch.save(checkpoint, args.output/'last.pt')
        if score > best:
            best = score; torch.save(checkpoint, args.output/'best.pt')
        print(json.dumps(metrics), flush=True)
    if args.abdomen_split == 'ct_cv5':
        (args.output/'cv_complete.json').write_text(json.dumps(dict(
            best_validation_macro_positive_dice=best,
            best_epoch=torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)['epoch'],
            abdomen_fold=args.abdomen_fold, record_sha256=digest), indent=2), encoding='utf-8')
        print('CV fold complete; frozen test set was not evaluated', flush=True)
        return
    selected_file = 'last.pt' if args.abdomen_split == 'ct_final' else 'best.pt'
    checkpoint = torch.load(args.output/selected_file, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test = evaluate(model, loaders['test'], args.device, model.structures)
    test['checkpoint_epoch'] = checkpoint['epoch']
    test['checkpoint_selection'] = 'fixed_final_epoch' if args.abdomen_split == 'ct_final' else 'best_validation'
    test['record_sha256'] = digest
    (args.output/'test_metrics.json').write_text(json.dumps(test, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
