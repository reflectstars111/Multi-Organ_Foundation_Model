"""Validation-only diagnostic for image-conditioned expert routing."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from unet_moe.region import RegionDataset, RegionUNet
from unet_moe.region_train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ablate-experts', action='store_true',
                        help='Compare intact model with expert outputs zeroed on validation only')
    args = parser.parse_args()
    checkpoint = torch.load(args.run / 'best.pt', map_location=args.device, weights_only=True)
    model = RegionUNet(**checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    records = json.loads((args.run / 'records.json').read_text(encoding='utf-8'))['records']
    dataset = RegionDataset([r for r in records if r['split'] == 'val'],
                            checkpoint['size'], checkpoint['config']['task_ids'])
    loader = DataLoader(dataset, batch_size=2, num_workers=2)
    counts = defaultdict(lambda: [0, 0])
    with torch.inference_mode():
        for image, target in loader:
            output = model(image.to(args.device))
            present = (target > 0).any((2, 3))
            for stage, scores in enumerate(output['routing']):
                chosen = scores.argmax(-1).cpu()
                for b, c in present.nonzero().tolist():
                    entry = counts[(stage, model.structures[c])]
                    entry[0] += int(chosen[b, c] == model.expert_for_region[c])
                    entry[1] += 1
    for stage in range(4):
        total = [sum(counts[(stage, name)][i] for name in model.structures) for i in (0, 1)]
        print(f'stage={stage} route_match={total[0]}/{total[1]}={total[0]/total[1]:.4f}')
        for name in model.structures:
            correct, count = counts[(stage, name)]
            print(f'  {name}: {correct}/{count}={correct/count:.4f}' if count else f'  {name}: no positives')
    if args.ablate_experts:
        intact = evaluate(model, loader, args.device, model.structures)
        hooks = [expert.register_forward_hook(lambda _m, _a, out: torch.zeros_like(out))
                 for moe in model.moes for expert in moe.experts]
        try:
            ablated = evaluate(model, loader, args.device, model.structures)
        finally:
            for hook in hooks:
                hook.remove()
        print('validation_positive_dice_intact', intact['macro_positive_dice'])
        print('validation_positive_dice_experts_zeroed', ablated['macro_positive_dice'])
        for name in model.structures:
            before = intact['per_structure'][name]['positive_dice']
            after = ablated['per_structure'][name]['positive_dice']
            print(f'  {name}: {before:.4f} -> {after:.4f}')


if __name__ == '__main__':
    main()
