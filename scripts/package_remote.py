"""Package only the files needed to reproduce the nine-dataset experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from unet_moe.data import build_records
from unet_moe.region import TASK_IDS


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[1]
    root = args.data_root.resolve()
    records, _ = build_records(root)
    files = {Path(r[key]).resolve() for r in records if r['task'] in TASK_IDS for key in ('image', 'mask')}
    for dataset in root.iterdir():
        if not dataset.is_dir() or not (dataset/'processed_png'/'manifest.csv').exists():
            continue
        files.update(dataset.glob('processed_png/manifest*.csv'))
        files.update(dataset.glob('processed_png/report*.json'))
        readme = dataset/'README_DATASET.md'
        if readme.exists():
            files.add(readme)
        if 'CAMUS' in dataset.name:
            files.update(dataset.glob('**/database_split/subgroup_*.txt'))
    code = set((repo/'unet_moe').glob('*.py'))
    code.update((repo/'unet_moe').glob('*.md'))
    code.update((repo/'unet_moe'/'image').rglob('*.png'))
    code.update((repo/'scripts').glob('*.py'))
    code.update((repo/'tests').glob('*.py'))
    code.add(repo/'requirements-unet-moe.txt')
    args.output.mkdir(parents=True)
    data_tar = args.output/'pointed_data.tar'
    code_tar = args.output/'code.tar'
    with tarfile.open(data_tar, 'w') as archive:
        for path in sorted(files):
            if not path.is_file() or not path.is_relative_to(root):
                raise ValueError(f'Invalid dataset path: {path}')
            archive.add(path, arcname=(Path('pointed_data')/path.relative_to(root)).as_posix(), recursive=False)
    with tarfile.open(code_tar, 'w') as archive:
        for path in sorted(code):
            archive.add(path, arcname=path.relative_to(repo).as_posix(), recursive=False)
    report = dict(data_files=len(files), code_files=len(code), paired_records=sum(r['task'] in TASK_IDS for r in records),
                  archives={path.name:dict(bytes=path.stat().st_size,sha256=sha256(path)) for path in (data_tar,code_tar)})
    (args.output/'package.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
