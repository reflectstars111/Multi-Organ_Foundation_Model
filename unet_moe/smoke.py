"""Exercise every available task using real files, gradients and serialization."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .data import TASKS, SegmentationDataset, build_records
from .model import UNetMoE, segmentation_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/moe_smoke'))
    args = parser.parse_args()
    torch.manual_seed(42)
    records, skipped = build_records(args.data_root)
    groups = {split: {r['group'] for r in records if r['split'] == split}
              for split in ('train', 'val', 'test')}
    assert not groups['train'] & groups['val']
    assert not groups['train'] & groups['test']
    selected = {r['task']: r for r in reversed(records)}
    dataset = SegmentationDataset(list(selected.values()), size=64)
    config = dict(tasks=len(TASKS), base=4, experts=3, top_k=2)
    model = UNetMoE(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for image, target, task in dataset:
        result = model(image[None], task[None])
        loss = segmentation_loss(result['logits'], target[None]) + .01 * result['balance']
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        assert torch.isfinite(loss)
        losses.append(loss.item())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), config=config, tasks=TASKS, size=64), args.output/'smoke.pt')
    restored = UNetMoE(**config)
    restored.load_state_dict(torch.load(args.output/'smoke.pt', weights_only=True)['model'])
    model.eval(); restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image[None], task[None])['logits'], restored(image[None], task[None])['logits'])
    report = dict(tasks_tested=[TASKS[i] for i in selected], losses=losses,
                  splits=dict(Counter(r['split'] for r in records)), skipped=skipped,
                  checkpoint_reload_equal=True, group_overlap=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
