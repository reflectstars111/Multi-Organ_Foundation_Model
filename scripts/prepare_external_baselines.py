"""Freeze Q04 sources, shared draws, observed support regions and real assets."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from unet_moe.external_baselines import METHODS
from unet_moe.external_baselines.protocol import file_sha, read_json, write_json
from unet_moe.external_baselines.support import support_manifest
from unet_moe.fair_suite import _digest
from unet_moe.region import DOMAINS
from unet_moe.sampling import domain_draws
from unet_moe.single_channel_suite import UnionDataset


def prepare(args):
    if args.output.exists():
        raise FileExistsError('Freeze requires a fresh directory; existing experiments are immutable')
    reference = read_json(args.reference_root / 'protocol.json')
    records = read_json(args.reference_root / 'records.json')
    if _digest(records) != reference['record_sha256']:
        raise ValueError('Reference records differ from the Q01 frozen protocol')
    source_content = read_json(args.reference_root / 'data_content.json')
    if _digest(source_content) != reference['content_sha256']:
        raise ValueError('Reference content manifest differs from the frozen Q01 protocol')
    evaluation_reference = read_json(args.evaluation_reference_root / 'protocol.json')
    evaluation_sha = evaluation_reference['protocol_sha256']
    if len(evaluation_sha) != 64:
        raise ValueError('Metric-backfill reference has no valid protocol fingerprint')
    paths = {r[k] for r in records if r['split'] in ('train', 'val') for k in ('image', 'mask')}
    content = {path: source_content[path] for path in sorted(paths)}
    # Copy only the previously frozen identities. Do not open test images or
    # masks until every Q04 training run and checkpoint selection is complete.
    test_paths = {r[k] for r in records if r['split'] == 'test' for k in ('image', 'mask')}
    expected_test_content = {path: source_content[path] for path in sorted(test_paths)}
    for path, item in content.items():
        if Path(path).stat().st_size != item['bytes'] or file_sha(path) != item['sha256']:
            raise ValueError(f'Train/validation file changed since Q01: {path}')
    sampling = read_json(args.reference_root / 'sampling_plan.json')
    train = UnionDataset([r for r in records if r['split'] == 'train'], 512)
    for seed in (42, 43, 44):
        for epoch in range(1, 81):
            indices = domain_draws(train.groups, 500, seed, epoch)
            expected = sampling[str(seed)][epoch - 1]
            if (indices != expected['indices']
                    or _digest([train.groups[i][0]['image'] for i in indices]) != expected['sample_sha256']):
                raise ValueError('Current data loading order differs from reference query sampling')
    supports = support_manifest(records, size=512, support_size=8)
    # Text prompt file is a real fixed CLIP embedding; never generate random
    # semantic assets during training. Its own provenance accompanies the tensor.
    prompts = args.assets_dir / 'text_prompts.pt'
    metadata = args.assets_dir / 'text_prompts.json'
    if not prompts.is_file() or not metadata.is_file():
        raise FileNotFoundError('Generate real MOFO CLIP text assets before freezing Q04')
    assets = {str(path.resolve()): file_sha(path) for path in (prompts, metadata)}
    args.output.mkdir(parents=True)
    write_json(args.output / 'records.json', records)
    write_json(args.output / 'sampling_plan.json', sampling)
    write_json(args.output / 'support_manifest.json', supports)
    write_json(args.output / 'data_content.json', content)
    write_json(args.output / 'test_content_expected.json', expected_test_content)
    snapshot = args.output / 'source_snapshot'
    sources = {}
    candidates = list((CODE_ROOT / 'unet_moe').rglob('*'))
    # Queue helpers import the earlier controllers and process utilities. Keep
    # every Python script in the immutable snapshot so those imports cannot
    # fall back to a mutable checkout after a long dependency wait.
    candidates += list((CODE_ROOT / 'scripts').rglob('*.py'))
    for path in sorted(candidates):
        if (not path.is_file() or '__pycache__' in path.parts
                or (path.suffix not in ('.py', '.json', '.md') and not path.name.startswith('LICENSE'))):
            continue
        relative = path.relative_to(CODE_ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(relative)] = file_sha(destination)
    shutil.copy2(CODE_ROOT / 'requirements-q04.txt', args.output / 'requirements-q04.txt')
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (args.output / 'environment.lock.txt').write_text(packages)
    config = dict(methods=list(METHODS), seeds=[42, 43, 44], size=512, epochs=80,
        batch_size=16, micro_batch_sizes=dict(lanzhou=2, aimagelab=2, universeg=1, smart=2, mofo=1),
        lr=1e-4, weight_decay=1e-4, optimizer='AdamW', lr_schedule='constant',
        augmentation='none', dtype='float32', quota_per_domain=500, workers=0,
        gradient_clip_norm=5., split_seed=42, selection='validation_positive_union_9_domain_macro_dsc',
        checkpoint_selections=['best', 'last'], test_barrier='all15_complete',
        reference_root=str(args.reference_root.resolve()),
        evaluation_reference_root=str(args.evaluation_reference_root.resolve()),
        assets_dir=str(args.assets_dir.resolve()), python=str(Path(sys.executable).resolve()),
        segmentation_loss='mean_images(BCE_valid_pixels + soft_Dice_smooth_1)',
        smart_aux_weight=.4, mofo_aux=dict(domain_ce=.5, prior_euclidean_l2=.2),
        mofo_prior=dict(epochs=80, batch_size=16, micro_batch_size=1, seeds=[42,43,44],
                        additional_stage=True, train_only=True),
        universeg_support_size=8, support_seed=1729,
        support_policy='train_only_valid_ROI_no_query_subject_no_replacement',
        evaluation=dict(metrics=['DSC (%)','HD95 (pixel)','ASSD (pixel)'],
                        threshold=.5, empty_policy='diagonal', grid=[512,512]))
    protocol = dict(schema='q04_external_baselines_v1', config=config,
        record_sha256=_digest(records), reference_protocol_sha256=reference['protocol_sha256'],
        evaluation_reference_protocol_sha256=evaluation_sha,
        input_files={n: file_sha(args.output / n) for n in
                     ('sampling_plan.json', 'support_manifest.json', 'data_content.json', 'test_content_expected.json')},
        source_files=sources, assets=assets, environment_sha256=file_sha(args.output / 'environment.lock.txt'),
        runtime_versions={name: importlib.metadata.version(name) for name in
                          ('torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'timm', 'einops')},
        content_hash_scope='train and validation only; test content verified after matrix completion',
        images_by_split={split: dict(Counter(r['dataset'] for r in
            {r['image']: r for r in records if r['split'] == split}.values())) for split in ('train','val','test')},
        source_architecture_registry=read_json(CODE_ROOT / 'relate_baseline' / 'baseline_registry.json'),
        budget=dict(primary_trainings=15, auxiliary_prior_trainings=3, query_images_per_epoch=4500,
            optimizer_updates_per_epoch=282, query_images_per_training=360000,
            universeg_support_images_per_training=2880000),
        created_at=datetime.now(timezone.utc).isoformat())
    protocol['protocol_sha256'] = _digest(protocol)
    write_json(args.output / 'protocol.json', protocol)
    write_json(args.output / 'preparation.json', dict(complete=True,
        protocol_sha256=protocol['protocol_sha256'], cpu_implementation_ready=True,
        gpu_preflight='mandatory queued stage before each architecture starts primary training',
        auxiliary_prior='three separate training stages, not pretrained substitutes'))
    print(f'Frozen Q04: {args.output}\nSHA256: {protocol["protocol_sha256"]}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--evaluation-reference-root', type=Path, required=True)
    p.add_argument('--assets-dir', type=Path, required=True)
    args = p.parse_args()
    prepare(args)


if __name__ == '__main__':
    main()
