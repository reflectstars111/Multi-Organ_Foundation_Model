"""Check that a physical batch of 16 union U-Net images fits GPU memory.

Only training images from the prepared 80-epoch fair records are opened. The
probe performs two BCE+Dice backward/AdamW steps before long training, so
the measured peak includes activation memory with AdamW state resident.
"""

import argparse
import json
from pathlib import Path

import torch

from unet_moe.baseline import PlainUNet
from unet_moe.fair_suite import _digest, _seed, _write_json, common_loss
from unet_moe.single_channel_suite import UnionDataset
from unet_moe.sampling import domain_draws
from .preflight_fair80 import prepare_cuda_memory_stats


def planned_first_batch(groups):
    indices = domain_draws(groups, 500, 42, 1)[:16]
    return dict(indices=indices, images=len(indices),
                sample_sha256=_digest([groups[index][0]['image'] for index in indices]))


def run(args):
    records_path = Path(args.records)
    protocol = json.loads((records_path.parent / 'protocol.json').read_text())
    config = protocol['config']
    expected = dict(size=512, base=24, batch_size=16, epochs=80,
                    quota_per_domain=500, lr=.0001, weight_decay=.0001,
                    split_seed=42, seeds=[42, 43, 44])
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Fair reference {key} differs from single-channel 80 protocol')
    records = json.loads(records_path.read_text())
    if _digest(records) != protocol['record_sha256']:
        raise ValueError('Fair reference records digest mismatch')
    dataset = UnionDataset([row for row in records if row['split'] == 'train'], 512)
    if len(dataset) != 6737:
        raise ValueError(f'Unexpected training image count: {len(dataset)}')
    selected = planned_first_batch(dataset.groups)
    report = dict(schema='single80_physical_batch_preflight_v1',
                  record_sha256=protocol['record_sha256'],
                  source_sha256=protocol['source_sha256'],
                  architecture='PlainUNet(outputs=1)',
                  image_size=512, base=24, physical_batch_size=16,
                  selection=selected, device=args.device, status='started')
    _write_json(args.output, report)
    _seed(42)
    device = torch.device(args.device)
    try:
        if device.type == 'cuda':
            prepare_cuda_memory_stats(device, report)
        examples = [dataset[index] for index in selected['indices']]
        image = torch.stack([row[0] for row in examples]).to(device)
        target = torch.stack([row[1] for row in examples]).to(device)
        model = PlainUNet(outputs=1, base=24).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
        model.train()
        for _ in range(2):
            logits = model(image)
            loss = common_loss(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite preflight loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) // 2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) // 2**20
        report.update(status='passed', optimizer_step_ok=True, optimizer_steps=2,
                      loss=float(loss.detach()))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        _write_json(args.output, report)
        raise
    _write_json(args.output, report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a new preflight report path')
    run(args)


if __name__ == '__main__':
    main()
