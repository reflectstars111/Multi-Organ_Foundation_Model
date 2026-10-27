"""Train-only MOFO anatomical prior; exact batch contrast via gradient caching.

The independently budgeted stage never loads validation/test images. Resume
restores optimizer and all RNG states at an atomic completed-epoch boundary.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..fair_suite import common_loss, _rng_state, _restore_rng
from ..region import DOMAINS
from ..sampling import domain_draws
from ..single_channel_suite import UnionDataset
from .mofo import sha256
from .vendor.mofo.prior import pUNet


def multi_scale_consistency(features, labels):
    """Eq8 positive-vs-negative cosine log-ratio, no diagonal/self positive.

    The paper uses a negatives-only denominator (not InfoNCE). For anchors
    lacking a same-domain partner or negative in a sampled batch, the term is
    undefined and contributes zero. Each scale is averaged over the full B.
    """
    labels = labels.reshape(-1)
    same = labels[:, None].eq(labels[None])
    positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    negative = ~same
    active = positive.any(1) & negative.any(1)
    losses = []
    for feature in features:
        z = F.normalize(feature.float(), dim=1)
        if not active.any():
            losses.append(z.sum() * 0)
            continue
        similarities = z @ z.T
        denominator = similarities.masked_fill(~negative, float('-inf')).logsumexp(1)
        # Evaluate active rows only: infinity * zero would produce NaN.
        similarity = similarities[active]
        positives = positive[active]
        # Eq8 divides by n_k; the self pair remains excluded from the sum.
        terms = ((denominator[active, None] - similarity) * positives).sum(1)
        terms = terms / same[active].sum(1)
        losses.append(terms.sum() / len(labels))
    return torch.stack(losses).mean()


def cached_prior_step(model, optimizer, masks, domains, *, micro_batch_size=1,
                      consistency_weight=.5):
    """Exact logical-batch contrast with bounded activations and two passes.

    pUNet uses per-image InstanceNorm and no stochastic layers. First collect
    pooled features without a graph, differentiate batch contrast w.r.t. those
    features, then replay each microbatch with its feature-vector VJP and its
    reconstruction loss. No approximate independent microbatch contrast.
    """
    if (masks < 0).any():
        raise ValueError('The prior must never encode unknown labels as background')
    if micro_batch_size < 1:
        raise ValueError('micro_batch_size must be positive')
    optimizer.zero_grad(set_to_none=True)
    cached = [[] for _ in range(8)]
    with torch.no_grad():
        for start in range(0, len(masks), micro_batch_size):
            _, features, _ = model(masks[start:start + micro_batch_size])
            for storage, feature in zip(cached, features, strict=True):
                storage.append(feature.detach())
    feature_leaves = [torch.cat(chunks).requires_grad_(True) for chunks in cached]
    contrast = multi_scale_consistency(feature_leaves, domains)
    derivatives = torch.autograd.grad(consistency_weight * contrast, feature_leaves)
    reconstruction_sum = 0.
    for start in range(0, len(masks), micro_batch_size):
        target = masks[start:start + micro_batch_size]
        reconstruction, features, _ = model(target)
        reconstruction_loss = common_loss(reconstruction, target) * len(target) / len(masks)
        surrogate = sum((feature * gradient[start:start + len(target)]).sum()
                        for feature, gradient in zip(features, derivatives, strict=True))
        (reconstruction_loss + surrogate).backward()
        reconstruction_sum += float(reconstruction_loss.detach())
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Non-finite anatomical-prior gradient')
    optimizer.step()
    return {'reconstruction': reconstruction_sum, 'consistency': float(contrast.detach()),
            'loss': reconstruction_sum + consistency_weight * float(contrast.detach())}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_state(path, model, optimizer, protocol, epoch, history=None):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'protocol': protocol, 'epoch': epoch, 'complete': epoch == protocol['epochs'],
             'history': list(history or []), 'rng': _rng_state()}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_state(path, model, optimizer, protocol, history=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['protocol'] != protocol:
        raise ValueError('Prior resume protocol mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    _restore_rng(state['rng'])
    if history is not None:
        history[:] = state['history']
    return state['epoch']


def prepare_pool(dataset, manifest=None):
    """Reuse a verified support pool, or scan train targets only."""
    if manifest is None:
        pools = {domain: [] for domain in DOMAINS}
        for index, group in enumerate(dataset.groups):
            _, target, domain, _ = dataset[index]
            from .valid_region import valid_roi
            roi = valid_roi(target[0])
            if roi is not None:
                top, left, bottom, right = roi
                truth = target[..., top:bottom, left:right]
                if truth.gt(0).any() and truth.eq(0).any():
                    pools[domain].append({'index': index, 'roi': list(roi)})
    else:
        raw = manifest.get('pools', manifest)
        pools = {domain: [{**entry, 'roi': entry.get('roi', [0, 0, dataset.size, dataset.size])}
                          if isinstance(entry, dict) else {'index': entry, 'roi': [0, 0, dataset.size, dataset.size]}
                          for entry in raw.get(domain, [])] for domain in DOMAINS}
    for domain, entries in pools.items():
        for entry in entries:
            index = entry['index']
            row = dataset.groups[index][0]
            if row['dataset'] != domain:
                raise ValueError('Prior pool domain/order differs from frozen train records')
            for field in ('image', 'group'):
                if field in entry and row[field] != entry[field]:
                    raise ValueError('Prior pool image/group differs from frozen train records')
    if any(not indices for indices in pools.values()):
        raise ValueError(f'No fully annotated prior data for domains: '
                         f'{[k for k, v in pools.items() if not v]}')
    return pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool-manifest', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--quota-per-domain', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.micro_batch_size, args.quota_per_domain) < 1:
        parser.error('Budgets must be positive')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = json.loads(args.records.read_text())
    dataset = UnionDataset([r for r in records if r['split'] == 'train'], args.size)
    manifest = json.loads(args.pool_manifest.read_text()) if args.pool_manifest else None
    pools = prepare_pool(dataset, manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    pool_digest = hashlib.sha256((json.dumps(pools, indent=2) + '\n').encode()).hexdigest()
    protocol = {'schema': 'mofo_prior_paper_v1', 'split': 'train',
                'target_policy': 'fully_observed_union_roi', 'domains': DOMAINS,
                'records_sha256': sha256(args.records), 'pool_sha256': pool_digest,
                'support_manifest_sha256': sha256(args.pool_manifest) if args.pool_manifest else None,
                'size': args.size, 'seed': args.seed, 'epochs': args.epochs,
                'batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size,
                'quota_per_domain': args.quota_per_domain, 'optimizer': 'AdamW',
                'lr': args.lr, 'weight_decay': args.weight_decay,
                'consistency_weight': .5, 'loss': 'BCE+Dice + 0.5*Eq8 negative-denominator cosine contrast',
                'cost': 'Two pUNet forwards per mask draw; exact gradient caching',
                'pool_counts': {k: len(v) for k, v in pools.items()}}
    source_files = [Path(__file__), Path(__file__).with_name('valid_region.py'),
                    Path(__file__).parent / 'vendor/mofo/prior.py']
    protocol['source_sha256'] = {str(path.relative_to(Path(__file__).parent)): sha256(path)
                               for path in source_files}
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing prior protocol differs; output will not be overwritten')
    if not args.resume and (args.output / 'last.pt').exists():
        raise ValueError('Existing prior checkpoint requires explicit --resume')
    if args.prepare_only:
        atomic_json(args.output / 'prior_pool.json', pools)
        atomic_json(protocol_path, protocol)
        print(json.dumps(protocol, indent=2)); return
    device = torch.device(args.device)
    model = pUNet(1).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    start_epoch = restore_state(args.resume, model, optimizer, protocol, history) if args.resume else 0
    atomic_json(args.output / 'prior_pool.json', pools)
    atomic_json(protocol_path, protocol)
    # The checkpoint is authoritative after an interrupted JSONL write.
    history_path = args.output / 'history.jsonl'
    history_path.write_text(''.join(json.dumps(row) + '\n' for row in history))
    last = args.output / 'last.pt'
    selected = [entry for domain in DOMAINS for entry in pools[domain]]
    groups = [dataset.groups[entry['index']] for entry in selected]
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        draws = domain_draws(groups, args.quota_per_domain, args.seed, epoch)
        losses = []
        for offset in range(0, len(draws), args.batch_size):
            from .valid_region import crop_resize
            items = []
            for index in draws[offset:offset + args.batch_size]:
                entry = selected[index]
                image, target, domain, source = dataset[entry['index']]
                if 'roi_target_sha256' in entry:
                    top, left, bottom, right = entry['roi']
                    observed = target[..., top:bottom, left:right].numpy().astype(np.int8).tobytes()
                    if hashlib.sha256(observed).hexdigest() != entry['roi_target_sha256']:
                        raise ValueError('Prior training ROI pixels changed after protocol freeze')
                image, target = crop_resize(image, target, entry['roi'], args.size)
                if not target.gt(0).any() or not target.eq(0).any():
                    raise ValueError('Prior support ROI must contain foreground and background')
                items.append((image, target, domain, source))
            masks = torch.stack([item[1] for item in items]).to(device)
            domains = torch.tensor([DOMAINS.index(item[2]) for item in items], device=device)
            value = cached_prior_step(model, optimizer, masks, domains,
                                      micro_batch_size=args.micro_batch_size)
            losses.append(value['loss'])
            atomic_json(args.output / 'status.json', {'state': 'training', 'epoch': epoch + 1,
                        'completed_epochs': epoch, 'epochs': args.epochs,
                        'step': offset // args.batch_size + 1,
                        'steps': (len(draws) + args.batch_size - 1) // args.batch_size,
                        'loss': value, 'pid': os.getpid()})
        row = {'epoch': epoch + 1, 'loss': sum(losses) / len(losses),
               'seconds': time.monotonic() - started}
        history.append(row)
        save_state(last, model, optimizer, protocol, epoch + 1, history)
        with history_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    atomic_json(args.output / 'status.json', {'state': 'complete', 'completed_epochs': args.epochs,
                                            'epochs': args.epochs, 'checkpoint': str(last)})


if __name__ == '__main__':
    main()
