import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import TASKS, SegmentationDataset, build_records
from .model import UNetMoE, segmentation_loss
from .region import PAPER_ALIGNED_TASK_IDS


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores = defaultdict(list)
    for image, target, task in loader:
        valid = target.to(device) >= 0
        prediction = (model(image.to(device), task.to(device))['logits'].sigmoid() >= .5) & valid
        truth = target.to(device) >= .5
        intersection = (prediction & truth).sum((1, 2, 3))
        denominator = prediction.sum((1, 2, 3)) + truth.sum((1, 2, 3))
        dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.)
        for t, score in zip(task.tolist(), dice.tolist()):
            scores[TASKS[t]].append(score)
    per_task = {name: float(np.mean(values)) for name, values in scores.items()}
    return {'macro_dice': float(np.mean(list(per_task.values()))), 'per_task_dice': per_task}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/unet_moe'))
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--base', type=int, default=24)
    parser.add_argument('--experts', type=int, default=9)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--decoder', choices=['moe', 'shared'], default='moe')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps-per-epoch', type=int, default=0)
    parser.add_argument('--abdomen-split', choices=('legacy', 'ct7_2', 'ct_cv5', 'ct_final'), default='legacy')
    parser.add_argument('--abdomen-fold', type=int)
    parser.add_argument('--split-policy', choices=('legacy', 'trainmore_v6'), default='legacy')
    parser.add_argument('--abdomen-labels', choices=('all8', 'paper5'), default='all8')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.size < 32 or args.batch_size < 1 or args.epochs < 1 or args.steps_per_epoch < 0:
        parser.error('Require size >= 32, batch-size/epochs >= 1 and steps-per-epoch >= 0')
    if (args.abdomen_split == 'ct_cv5') != (args.abdomen_fold is not None):
        parser.error('--abdomen-fold is required only for ct_cv5')
    if args.abdomen_fold is not None and not 0 <= args.abdomen_fold < 5:
        parser.error('--abdomen-fold must be 0..4')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError('Use a new output directory to preserve existing checkpoints')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records, skipped = build_records(args.data_root, args.seed, args.abdomen_split,
                                     args.abdomen_fold, args.split_policy)
    task_ids = set(PAPER_ALIGNED_TASK_IDS) if args.abdomen_labels == 'paper5' else set(range(len(TASKS)))
    records = [r for r in records if r['task'] in task_ids]
    digest = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'val']
    if not train or not val:
        raise ValueError('Train and validation splits must both be nonempty')
    config_record = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    config_record['record_sha256'] = digest
    if args.resume:
        saved = json.loads((args.output / 'training_config.json').read_text(encoding='utf-8'))
        if saved != config_record:
            raise ValueError('Resume configuration or frozen records changed')
        completion = 'cv_complete.json' if args.abdomen_split == 'ct_cv5' else 'test_metrics.json'
        if (args.output / completion).exists():
            print('Run already complete; skipping', flush=True)
            return
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'training_config.json').write_text(json.dumps(config_record, indent=2), encoding='utf-8')
        (args.output / 'records.json').write_text(json.dumps({'records': records, 'skipped': skipped,
            'sha256': digest}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('splits:', Counter(r['split'] for r in records), 'skipped:', skipped, flush=True)
    counts = Counter(r['task'] for r in train)
    samples = args.steps_per_epoch * args.batch_size if args.steps_per_epoch else len(train)
    sampler = WeightedRandomSampler([1 / counts[r['task']] for r in train], samples, replacement=True)
    train_loader = DataLoader(SegmentationDataset(train, args.size), batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.workers)
    val_loader = DataLoader(SegmentationDataset(val, args.size), batch_size=args.batch_size,
                            num_workers=args.workers)
    config = dict(tasks=len(TASKS), base=args.base, experts=args.experts, top_k=args.top_k,
                  decoder=args.decoder)
    model = UNetMoE(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1
    start_epoch = 0
    if args.resume and (args.output / 'last.pt').exists():
        checkpoint = torch.load(args.output / 'last.pt', map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        if (args.output / 'best.pt').exists():
            best = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)['metrics']['macro_dice']
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        print(f'Resuming from epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        model.train()
        total, batches = 0., 0
        loads = []
        for image, target, task in train_loader:
            output = model(image.to(args.device), task.to(args.device))
            loss = segmentation_loss(output['logits'], target.to(args.device)) + .01 * output['balance'] + .001 * output['router_z']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item(); batches += 1
            loads.append(output['expert_load'].cpu())
            if batches % 50 == 0:
                print(f'epoch={epoch+1}/{args.epochs} batch={batches}/{len(train_loader)} loss={total/batches:.4f} elapsed={time.monotonic()-started:.1f}s', flush=True)
        metrics = evaluate(model, val_loader, args.device)
        metrics.update(epoch=epoch + 1, train_loss=total / batches,
                       elapsed_seconds=time.monotonic()-started,
                       expert_load=torch.stack(loads).mean(0).tolist())
        print(json.dumps(metrics), flush=True)
        with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics) + '\n')
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                          tasks=TASKS, size=args.size, epoch=epoch + 1, metrics=metrics,
                          seed=args.seed, trained_tasks=sorted(counts), record_sha256=digest,
                          rng_state=torch.get_rng_state(),
                          cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(checkpoint, args.output / 'last.pt')
        if metrics['macro_dice'] > best:
            best = metrics['macro_dice']
            torch.save(checkpoint, args.output / 'best.pt')
    if args.abdomen_split == 'ct_cv5':
        (args.output / 'cv_complete.json').write_text(json.dumps(dict(
            best_validation_macro_dice=best,
            best_epoch=torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)['epoch'],
            abdomen_fold=args.abdomen_fold, record_sha256=digest), indent=2), encoding='utf-8')
        print('CV fold complete; frozen test was not evaluated', flush=True)
        return
    # Final-fit protocol selects the fixed last epoch, not a checkpoint selected
    # using validation sets that have no abdominal examples.
    selected_file = 'last.pt' if args.abdomen_split == 'ct_final' else 'best.pt'
    selected = torch.load(args.output / selected_file, map_location=args.device, weights_only=True)
    model.load_state_dict(selected['model'])
    test = [r for r in records if r['split'] == 'test']
    if test:
        test_loader = DataLoader(SegmentationDataset(test, args.size), batch_size=args.batch_size,
                                 num_workers=args.workers)
        result = evaluate(model, test_loader, args.device)
        result.update(checkpoint_epoch=selected['epoch'], records=len(test),
                      checkpoint_selection='fixed_final_epoch' if args.abdomen_split == 'ct_final' else 'best_validation',
                      record_sha256=digest,
                      note='Resized-grid task-wise Dice; datasets mix official and project splits. HC18 test is held out from the labelled release training set, not the official challenge test set.')
        (args.output / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('FINAL TEST:', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
