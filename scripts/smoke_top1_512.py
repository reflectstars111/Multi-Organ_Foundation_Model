"""One real-data forward/backward at the proposed v6 Top-1 settings; no writes."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from unet_moe.data import SegmentationDataset, TASKS, build_records
from unet_moe.model import UNetMoE, segmentation_loss
from unet_moe.region import DOMAINS, PAPER_ALIGNED_TASK_IDS, RegionDataset, RegionUNet, region_loss


def diverse_indices(items, key, count):
    first = {}
    for index, item in enumerate(items):
        first.setdefault(key(item), index)
    selected = list(first.values())
    selected.extend(i for i in range(len(items)) if i not in first.values())
    return selected[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('task_id', 'region'), required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--base', type=int, default=24)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.75)
    rows, _ = build_records(args.data_root, 42, 'ct_final', split_policy='trainmore_v6')
    rows = [r for r in rows if r['split'] == 'train' and r['task'] in PAPER_ALIGNED_TASK_IDS]
    if args.mode == 'task_id':
        dataset = SegmentationDataset(rows, args.size)
        indices = diverse_indices(rows, lambda r: r['task'], args.batch_size)
        model = UNetMoE(tasks=len(TASKS), base=args.base, experts=9, top_k=1).to(args.device)
        image, target, task = next(iter(DataLoader(Subset(dataset, indices), batch_size=args.batch_size)))
        output = model(image.to(args.device), task.to(args.device))
        loss = segmentation_loss(output['logits'], target.to(args.device))
        loss = loss + .01 * output['balance'] + .001 * output['router_z']
    else:
        dataset = RegionDataset(rows, args.size, PAPER_ALIGNED_TASK_IDS)
        indices = diverse_indices(dataset.groups, lambda group: group[0]['dataset'], args.batch_size)
        model = RegionUNet(base=args.base, top_k=1, task_ids=PAPER_ALIGNED_TASK_IDS).to(args.device)
        image, target = next(iter(DataLoader(Subset(dataset, indices), batch_size=args.batch_size)))
        output = model(image.to(args.device))
        loss = region_loss(output, target.to(args.device), expert_for_region=model.expert_for_region)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite real-data loss')
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith('cuda') else 0
    print(dict(mode=args.mode, batch=len(indices), size=args.size, base=args.base,
               top_k=1, loss=float(loss), peak_allocated_gib=round(peak, 2)), flush=True)


if __name__ == '__main__':
    main()
