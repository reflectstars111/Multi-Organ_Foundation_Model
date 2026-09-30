"""Validate and summarize the completed paper-five three-way comparison."""
import argparse
import json
from pathlib import Path

from unet_moe.region import DOMAINS, RegionUNet


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', type=Path, required=True)
    args = parser.parse_args()
    moe = args.runs_root / 'region_paper5_ct7_2_v1'
    baselines = args.runs_root / 'unet_paper5_ct7_2_v1'
    region = load(moe / 'test_metrics.json')
    comparison = load(baselines / 'comparison.json')
    if region['record_sha256'] != comparison['record_sha256']:
        raise ValueError('Models were evaluated on different frozen records')
    if set(region['per_structure']) != set(comparison['per_structure']):
        raise ValueError('Models have different output structures')
    config = load(moe / 'config.json')
    model = RegionUNet(base=config['base'], top_k=config['top_k'], task_ids=config['task_ids'])
    result = dict(protocol='paper5_ct7_2_v1', record_sha256=region['record_sha256'],
                  class_count=len(comparison['structures']), expert_count=len(DOMAINS),
                  positive_macro_dice={
                      'region_top1_moe': region['macro_positive_dice'],
                      'shared_unet': comparison['shared_macro_positive_dice'],
                      'nine_independent_unets': comparison['independent_macro_positive_dice'],
                  },
                  parameters={
                      'region_top1_moe': sum(p.numel() for p in model.parameters()),
                      'shared_unet': comparison['shared_parameters'],
                      'nine_independent_unets_total': comparison['independent_total_parameters'],
                  },
                  per_structure={name: dict(
                      region_top1_moe=region['per_structure'][name],
                      shared_unet=comparison['per_structure'][name]['shared'],
                      independent_unet=comparison['per_structure'][name]['independent'])
                      for name in comparison['structures']})
    output = args.runs_root / 'paper5_ct7_2_v1_comparison.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output),
                          positive_macro_dice=result['positive_macro_dice']), ensure_ascii=False))


if __name__ == '__main__':
    main()
