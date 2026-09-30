"""Refresh local README statistics from actual processed_png files (read-only data)."""
import csv
from collections import Counter
from pathlib import Path


def main():
    root = next(p for p in Path.cwd().iterdir() if '(Datasets)' in p.name) / 'pointed_data'
    # Produce patches; file edits are applied through apply_patch by the caller.
    sections = {}
    for dataset in sorted(root.iterdir()):
        output = dataset / 'processed_png'
        readme = dataset / 'README_DATASET.md'
        if not readme.exists():
            continue
        images = list((output / 'images').rglob('*.png'))
        masks = list((output / 'masks').rglob('*.png'))
        with (output / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        paired = sum(bool(r['image'] and r['mask']) and (output/r['image']).is_file() and (output/r['mask']).is_file() for r in rows)
        missing = sum(any(r[k] and not (output/r[k]).is_file() for k in ('image','mask')) for r in rows)
        text = '## processed_png 本地统计（优先使用）\n\n统计日期：2026-09-16；图像与 mask 数量来自当前磁盘文件，配对按 manifest 路径实际存在性复核。\n\n'
        text += '| 项目 | 数量 |\n|---|---:|\n'
        for key, value in [('原图 PNG',len(images)),('mask PNG',len(masks)),('PNG 合计',len(images)+len(masks)),('manifest 记录',len(rows)),('图像与 mask 均存在的配对记录',paired),('引用缺失文件的记录',missing)]:
            text += f'| {key} | {value} |\n'
        text += '\n配对记录按标签计数：一张图有多个目标标签时会有多条记录，因此配对数不等于原图数。\n\n### 实际目录分布\n\n| 相对于 processed_png 的目录 | PNG 数量 |\n|---|---:|\n'
        for directory, count in sorted(Counter(str(p.parent.relative_to(output)).replace('\\','/') for p in images+masks).items()):
            if 'CAMUS' not in dataset.name:
                text += f'| `{directory}` | {count} |\n'
        if 'CAMUS' in dataset.name:
            text += f'| `images/patient*`（{len(set(p.parent for p in images))} 个患者目录） | {len(images)} |\n| `masks/patient*` | {len(masks)} |\n'
        sections[dataset.name] = text
    import json
    print(json.dumps(sections,ensure_ascii=False))


if __name__ == '__main__':
    main()
