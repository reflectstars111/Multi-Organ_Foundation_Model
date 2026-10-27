"""Check the worst paired 512px microbatch before the 80-epoch suite.

The frozen fair trainer expands each image into its annotated task queries for
the Task-ID model. Sixteen images can therefore require substantially more
than sixteen model forwards. This command performs two real training updates
on the most query-heavy planned batch, without opening validation or test PNGs.
The second update measures activation memory with AdamW state resident.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from unet_moe.fair_suite import (_digest, _seed, _write_json, create_model,
                                 forward_model, objective)
from unet_moe.region import PAPER_ALIGNED_TASK_IDS, RegionDataset
from unet_moe.sampling import domain_draws


PROFILE = dict(seeds=[42, 43, 44], split_seed=42, epochs=80,
               quota_per_domain=500, size=512, base=24, batch_size=16,
               workers=0, lr=.0001, weight_decay=.0001,
               lr_schedule='constant', min_lr_ratio=.1)


def worst_batch(groups, seeds=(42, 43, 44), epochs=80, quota=500,
                batch_size=16, micro_batch_size=4):
    """Find the effective batch containing the heaviest physical microbatch."""
    best = None
    counts = [len(rows) for rows in groups]
    for seed in seeds:
        for epoch in range(1, epochs + 1):
            draws = domain_draws(groups, quota, seed, epoch)
            for start in range(0, len(draws), batch_size):
                indices = draws[start:start + batch_size]
                queries = sum(counts[index] for index in indices)
                micro_queries = [sum(counts[index] for index in indices[m:m + micro_batch_size])
                                 for m in range(0, len(indices), micro_batch_size)]
                peak_queries = max(micro_queries)
                if best is None or (peak_queries, queries) > (
                        best['max_microbatch_queries'], best['annotated_queries']):
                    best = dict(seed=seed, epoch=epoch, batch_index=start // batch_size,
                                indices=indices, images=len(indices),
                                annotated_queries=queries,
                                max_microbatch_queries=peak_queries,
                                sample_sha256=_digest([groups[index][0]['image'] for index in indices]))
    if best is None:
        raise ValueError('No planned training batches')
    return best


def _load_fair_training(records_path):
    records_path = Path(records_path)
    protocol = json.loads((records_path.parent / 'protocol.json').read_text())
    config = protocol['config']
    for key, expected in PROFILE.items():
        if config.get(key) != expected:
            raise ValueError(f'Fair protocol {key}={config.get(key)!r}, expected {expected!r}')
    if config.get('task_ids') != list(PAPER_ALIGNED_TASK_IDS):
        raise ValueError('Expected frozen paper5 task IDs')
    records = json.loads(records_path.read_text())
    if _digest(records) != protocol['record_sha256']:
        raise ValueError('Fair protocol record digest differs from records.json')
    dataset = RegionDataset([row for row in records if row['split'] == 'train'],
                            PROFILE['size'], PAPER_ALIGNED_TASK_IDS)
    if len(dataset) != 6737:
        raise ValueError(f'Unexpected training image count: {len(dataset)}')
    return dataset, protocol


def prepare_cuda_memory_stats(device, report):
    """Initialize the CUDA context before resetting allocator peak counters."""
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    report['gpu_free_mib_before'] = free // 2**20
    report['gpu_total_mib'] = total // 2**20


def run(args):
    dataset, protocol = _load_fair_training(args.records)
    batch = worst_batch(dataset.groups, micro_batch_size=args.micro_batch_size)
    report = dict(schema='fair80_worst_batch_preflight_v1', kind=args.kind,
                  record_sha256=protocol['record_sha256'],
                  source_sha256=protocol['source_sha256'],
                  image_size=PROFILE['size'], base=PROFILE['base'],
                  effective_batch_size=PROFILE['batch_size'],
                  physical_micro_batch_size=args.micro_batch_size,
                  selection=batch, device=args.device, status='started')
    _write_json(args.output, report)
    _seed(batch['seed'])
    device = torch.device(args.device)
    try:
        if device.type == 'cuda':
            prepare_cuda_memory_stats(device, report)
        examples = [dataset[index] for index in batch['indices']]
        image = torch.stack([item[0] for item in examples])
        target = torch.stack([item[1] for item in examples])
        model = create_model(args.kind, base=PROFILE['base'], top_k=1, experts=9).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROFILE['lr'],
                                      weight_decay=PROFILE['weight_decay'])
        model.train()
        weights = SimpleNamespace(empty_weight=0., balance_weight=0., router_z_weight=0.,
                                  coarse_weight=0., route_weight=0.)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for start in range(0, len(image), args.micro_batch_size):
                micro_image = image[start:start + args.micro_batch_size].to(device)
                micro_target = target[start:start + args.micro_batch_size].to(device)
                output = forward_model(model, args.kind, micro_image, micro_target)
                loss = objective(model, args.kind, output, micro_target, weights)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite preflight loss')
                share = len(micro_image) / len(image)
                (loss * share).backward()
                total_loss += float(loss.detach()) * share
                del output, loss, micro_image, micro_target
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) // 2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) // 2**20
        report.update(status='passed', loss=total_loss,
                      optimizer_step_ok=True, optimizer_steps=2)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        _write_json(args.output, report)
        raise
    _write_json(args.output, report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True,
                        help='Prepared fair shard records.json, with protocol.json beside it')
    parser.add_argument('--kind', choices=('shared', 'task_id', 'region'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--micro-batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a new preflight report path')
    if args.micro_batch_size < 1 or args.micro_batch_size > PROFILE['batch_size']:
        parser.error('Invalid micro-batch-size')
    run(args)


if __name__ == '__main__':
    main()
