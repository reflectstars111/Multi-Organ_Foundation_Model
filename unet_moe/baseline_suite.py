"""Run frozen-data independent and shared plain U-Net comparison."""
import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .baseline import PlainUNet, evaluate, image_balanced_loss
from .data import build_records, TASKS
from .region import (DOMAINS, RegionDataset, TASK_IDS,
                     PAPER_ALIGNED_TASK_IDS, EXPERT_FOR_REGION)
from .region_train import abdomen_positive_weights, domain_sampling_weights
from .sampling import domain_draws
from .schedule import epoch_lr


class ChannelView(Dataset):
    def __init__(self, source, channels):
        self.source = source
        self.channels = channels
        self.groups = source.groups

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, target = self.source[index]
        return image, target[self.channels]


def frozen_records(root, seed, task_ids=TASK_IDS, abdomen_split='legacy',
                   abdomen_fold=None, split_policy='legacy'):
    records, skipped = build_records(root, seed, abdomen_split, abdomen_fold, split_policy)
    used = [r for r in records if r['task'] in task_ids]
    for row in used:
        if not Path(row['image']).is_file() or not Path(row['mask']).is_file():
            raise FileNotFoundError(row)
    hc = [r for r in used if r['dataset'] == 'Fetal_HC']
    if len(hc) != 999 or any(r['label_version'] != 'hc18_filled_v1' for r in hc):
        raise ValueError('Expected all 999 HC18 records with filled masks')
    encoded = json.dumps(used, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return used, skipped, hashlib.sha256(encoded).hexdigest()


def run_model(mode, domain, datasets, root, args, digest, positive_weights=None):
    name = 'shared' if mode == 'shared' else domain
    folder = root/name
    folder.mkdir(exist_ok=args.resume)
    task_ids = TASK_IDS if args.abdomen_labels == 'all8' else PAPER_ALIGNED_TASK_IDS
    structures = [TASKS[i] for i in task_ids]
    owners = [EXPERT_FOR_REGION[TASK_IDS.index(i)] for i in task_ids]
    channels = (list(range(len(structures))) if mode == 'shared' else
                [i for i, owner in enumerate(owners) if owner == DOMAINS.index(domain)])
    subsets = {}
    for split, dataset in datasets.items():
        if mode == 'shared':
            chosen = dataset
        else:
            domain_rows = [rows for rows in dataset.groups if rows[0]['dataset'] == domain]
            chosen = RegionDataset([r for rows in domain_rows for r in rows], args.size, task_ids)
        subsets[split] = ChannelView(chosen, channels)
    has_validation = len(subsets['val']) > 0
    if not has_validation and args.abdomen_split != 'ct_final':
        raise ValueError(f'No validation images for {name}')
    torch.manual_seed(args.seed)
    model = PlainUNet(outputs=len(channels), base=args.base).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    selected_weights = positive_weights[channels].to(args.device) if positive_weights is not None else None
    sampling_weights = (domain_sampling_weights(subsets['train'].source, positive_weights,
                        args.abdomen_positive_sampling) if positive_weights is not None else None)
    params = sum(p.numel() for p in model.parameters())
    best = -1.
    config = dict(mode=mode, domain=domain, channels=channels, structures=[structures[c] for c in channels],
                  base=args.base, size=args.size, epochs=args.epochs, quota=args.quota,
                  batch_size=args.batch_size, seed=args.seed, lr=args.lr,
                  lr_schedule=args.lr_schedule, min_lr_ratio=args.min_lr_ratio,
                  abdomen_fold=args.abdomen_fold, split_policy=args.split_policy,
                  record_sha256=digest, parameters=params, loss='per_image_mean_valid_channel_bce_plus_dice')
    if args.abdomen_split != 'legacy' or args.abdomen_labels != 'all8':
        config.update(abdomen_split=args.abdomen_split, abdomen_labels=args.abdomen_labels,
                      abdomen_positive_sampling=args.abdomen_positive_sampling,
                      positive_weights=selected_weights.cpu().tolist(),
                      loss='per_image_valid_channel_bce_dice_positive_balanced_v2')
    config_path = folder/'config.json'
    start_epoch = 1
    if args.resume and config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding='utf-8'))
        if saved_config != config:
            raise ValueError(f'Resume configuration mismatch: {folder}')
        completion = 'cv_complete.json' if args.abdomen_split == 'ct_cv5' else 'test_metrics.json'
        if (folder/completion).exists():
            result = json.loads((folder/completion).read_text(encoding='utf-8'))
            selected_file = 'last.pt' if args.abdomen_split == 'ct_final' else 'best.pt'
            best_cp = torch.load(folder/selected_file, map_location='cpu', weights_only=True)
            print(f'{name} already complete; skipping', flush=True)
            return dict(name=name, parameters=params, best_epoch=best_cp['epoch'],
                        **({'validation_positive_macro_dice': result['best_validation_macro_positive_dice']}
                           if args.abdomen_split == 'ct_cv5' else
                           {'test_positive_macro_dice': result['macro_positive_dice']}))
        if (folder/'last.pt').exists():
            checkpoint = torch.load(folder/'last.pt', map_location=args.device, weights_only=True)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            if (folder/'best.pt').exists():
                best = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)['validation']['macro_positive_dice']
            print(f'{name} resuming at epoch {start_epoch}', flush=True)
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    for epoch in range(start_epoch, args.epochs + 1):
        lr = epoch_lr(args.lr, epoch, args.epochs, args.lr_schedule, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group['lr'] = lr
        indices = domain_draws(subsets['train'].groups, args.quota, args.seed, epoch,
                               domain if mode == 'independent' else None, sampling_weights)
        train = DataLoader(subsets['train'], batch_size=args.batch_size, sampler=indices,
                           num_workers=args.workers)
        model.train()
        total = 0.
        for step, (image, target) in enumerate(train, 1):
            logits = model(image.to(args.device))
            loss = image_balanced_loss(logits, target.to(args.device), selected_weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {name} epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item()
            if step % 100 == 0:
                print(f'{name} epoch={epoch}/{args.epochs} step={step}/{len(train)} loss={total/step:.4f}', flush=True)
        if has_validation:
            val = DataLoader(subsets['val'], batch_size=args.batch_size, num_workers=args.workers)
            metrics = evaluate(model, val, args.device, channels, structures)
            score = metrics['macro_positive_dice']
            if score is None:
                raise ValueError(f'No positive validation labels: {name}')
        else:
            metrics = dict(macro_positive_dice=None, per_structure={})
            score = None
        metrics.update(epoch=epoch, train_loss=total/step, lr=lr)
        with (folder/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics)+'\n')
        checkpoint = dict(architecture='plain_unet_v1', model=model.state_dict(),
                          optimizer=optimizer.state_dict(), config=config, epoch=epoch,
                          validation=metrics)
        torch.save(checkpoint, folder/'last.pt')
        if score is not None and score > best:
            best = score
            torch.save(checkpoint, folder/'best.pt')
        print(f'{name} epoch={epoch} val_positive_macro_dice={score if score is not None else "not_available"}', flush=True)
    if args.abdomen_split == 'ct_cv5':
        best_cp = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        (folder/'cv_complete.json').write_text(json.dumps(dict(
            best_validation_macro_positive_dice=best,
            best_epoch=best_cp['epoch'], abdomen_fold=args.abdomen_fold,
            record_sha256=digest), indent=2), encoding='utf-8')
        print(f'{name} CV fold complete; frozen test set was not evaluated', flush=True)
        return dict(name=name, parameters=params, best_epoch=best_cp['epoch'],
                    validation_positive_macro_dice=best)
    selected_file = 'last.pt' if args.abdomen_split == 'ct_final' else 'best.pt'
    selected = torch.load(folder/selected_file, map_location=args.device, weights_only=True)
    model.load_state_dict(selected['model'])
    test = DataLoader(subsets['test'], batch_size=args.batch_size, num_workers=args.workers)
    results = evaluate(model, test, args.device, channels, structures)
    results.update(checkpoint_epoch=selected['epoch'], record_sha256=digest,
                   checkpoint_selection='fixed_final_epoch' if args.abdomen_split == 'ct_final' else 'best_validation')
    (folder/'test_metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{name} test_positive_macro_dice={results["macro_positive_dice"]:.4f}', flush=True)
    return dict(name=name, parameters=params, best_epoch=selected['epoch'],
                test_positive_macro_dice=results['macro_positive_dice'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['all', 'shared', 'independent', 'summarize'], default='all')
    parser.add_argument('--domains', default='', help='Comma-separated subset for independent mode')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--quota', type=int, default=500, help='image draws per dataset per epoch')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    parser.add_argument('--min-lr-ratio', type=float, default=.1)
    parser.add_argument('--abdomen-split', choices=('legacy', 'ct7_2', 'ct_cv5', 'ct_final'), default='legacy')
    parser.add_argument('--abdomen-fold', type=int)
    parser.add_argument('--split-policy', choices=('legacy', 'trainmore_v6'), default='legacy')
    parser.add_argument('--abdomen-labels', choices=('all8', 'paper5'), default='all8')
    parser.add_argument('--abdomen-positive-sampling', type=float, default=0.)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    selected_domains = args.domains.split(',') if args.domains else DOMAINS
    if len(set(selected_domains)) != len(selected_domains) or any(name not in DOMAINS for name in selected_domains):
        parser.error('Invalid --domains list')
    if args.domains and args.mode != 'independent':
        parser.error('--domains is only valid with --mode independent')
    if (args.output.exists() and not args.resume) or min(args.epochs, args.size, args.base, args.batch_size, args.quota) < 1:
        parser.error('Use a new output directory (or --resume) and positive hyperparameters')
    if args.size < 32:
        parser.error('size must be >= 32')
    if args.abdomen_positive_sampling < 0:
        parser.error('AbdomenUS positive sampling must be nonnegative')
    if not 0 <= args.min_lr_ratio <= 1:
        parser.error('Minimum LR ratio must be in 0..1')
    if (args.abdomen_split == 'ct_cv5') != (args.abdomen_fold is not None):
        parser.error('--abdomen-fold is required only for ct_cv5')
    if args.abdomen_fold is not None and not 0 <= args.abdomen_fold < 5:
        parser.error('--abdomen-fold must be 0..4')
    if args.abdomen_split == 'ct_cv5' and args.mode in ('all', 'summarize'):
        parser.error('CV folds support --mode shared or independent; no test comparison')
    if args.abdomen_positive_sampling and args.abdomen_split == 'legacy' and args.abdomen_labels == 'all8':
        parser.error('Positive sampling requires the new AbdomenUS protocol')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    task_ids = TASK_IDS if args.abdomen_labels == 'all8' else PAPER_ALIGNED_TASK_IDS
    structures = [TASKS[i] for i in task_ids]
    records, skipped, digest = frozen_records(args.data_root, args.seed, task_ids,
                                              args.abdomen_split, args.abdomen_fold,
                                              args.split_policy)
    datasets = {split: RegionDataset([r for r in records if r['split'] == split], args.size, task_ids)
                for split in ('train', 'val', 'test')}
    positive_weights = None
    abdomen_balance = None
    if args.abdomen_split != 'legacy' or args.abdomen_labels != 'all8':
        positive_weights, abdomen_balance = abdomen_positive_weights(datasets['train'])
    if any(not len(dataset) for dataset in datasets.values()):
        raise ValueError('Empty split')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    record_path = args.output/'records.json'
    if args.resume:
        saved = json.loads(record_path.read_text(encoding='utf-8'))
        if saved['sha256'] != digest:
            raise ValueError('Frozen records changed; refusing resume')
    else:
        record_path.write_text(json.dumps(dict(records=records, skipped=skipped,
            split_version=records[0]['split_version'], sha256=digest), ensure_ascii=False), encoding='utf-8')
    counts = {split:dict(Counter(rows[0]['dataset'] for rows in ds.groups)) for split,ds in datasets.items()}
    if not args.resume:
        (args.output/'protocol.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v
            for k,v in vars(args).items()}, image_counts=counts, structures=structures,
            abdomen_positive_balance=abdomen_balance, record_sha256=digest,
            selection=('validation-only CV, no test' if args.abdomen_split == 'ct_cv5' else
                       'fixed final epoch; test once' if args.abdomen_split == 'ct_final' else
                       'best validation positive macro Dice; test once'), threshold=.5,
            caveats=['HC18 uses filled labels', 'AbdomenUS includes simulated AUS only',
                     '9 independent models have larger aggregate parameter count']),
            ensure_ascii=False, indent=2), encoding='utf-8')
    print('Frozen records:',digest,'images:',counts,flush=True)
    results = []
    if args.mode in ('all','shared'):
        results.append(run_model('shared', None, datasets, args.output, args, digest, positive_weights))
    if args.mode in ('all','independent'):
        for domain in selected_domains:
            results.append(run_model('independent', domain, datasets, args.output, args, digest, positive_weights))
            if args.mode == 'all':
                (args.output/'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    if args.mode in ('all', 'shared'):
        (args.output/'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    if args.mode in ('all', 'summarize'):
        shared = json.loads((args.output/'shared'/'test_metrics.json').read_text(encoding='utf-8'))
        independent = {}
        for domain in DOMAINS:
            result = json.loads((args.output/domain/'test_metrics.json').read_text(encoding='utf-8'))
            independent.update(result['per_structure'])
        if set(independent) != set(shared['per_structure']):
            raise ValueError('Structure mismatch between baselines')
        standalone = np.mean([r['positive_dice'] for r in independent.values()])
        parameter_counts = {name:json.loads((args.output/name/'config.json').read_text(encoding='utf-8'))['parameters']
                            for name in ['shared'] + DOMAINS}
        comparison = dict(record_sha256=digest, structures=structures,
                          shared_macro_positive_dice=shared['macro_positive_dice'],
                          independent_macro_positive_dice=float(standalone),
                          shared_parameters=parameter_counts['shared'],
                          independent_total_parameters=sum(parameter_counts[name] for name in DOMAINS),
                          per_structure={name: dict(shared=shared['per_structure'][name],
                                                    independent=independent[name]) for name in structures})
        (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        print('COMPARISON',json.dumps({k:comparison[k] for k in
              ('shared_macro_positive_dice','independent_macro_positive_dice',
               'shared_parameters','independent_total_parameters')}),flush=True)


if __name__ == '__main__':
    main()
