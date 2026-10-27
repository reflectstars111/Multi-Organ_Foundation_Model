"""Focused behavioral checks for the one-output foreground-union protocol."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from unet_moe import single_channel_suite as single
from unet_moe.region import DOMAINS, PAPER_ALIGNED_TASK_IDS, RegionDataset
from unet_moe.sampling import domain_draws


def test_union_label_requires_all_active_channels_for_background():
    target = torch.full((3, 2, 3), -1.)
    target[0] = 0
    target[1] = 0
    target[0, 0, 0] = 1
    target[1, 0, 1] = -1
    target[0, 0, 1] = 1  # Positive wins even if another channel is ignored.
    target[1, 1, 0] = -1  # No positive and not all channels valid -> unknown.
    union = single.union_target(target)
    assert union.shape == (1, 2, 3)
    assert union[0, 0].tolist() == [1., 1., 0.]
    assert union[0, 1].tolist() == [-1., 0., 0.]
    assert single.union_target(target, unknown_background=True)[0, 1, 1] == -1
    target[2] = -1  # Absent channel imposes no condition.
    assert torch.equal(single.union_target(target), union)
    indexed = torch.tensor([[1, 3, 4], [5, 7, 255]])
    masked = single.union_target(target, indexed)
    assert masked[0, 0, 0] == 1
    assert masked[0, 0, 1] == -1
    assert masked[0, 0, 2] == 0
    assert torch.all(masked[0, 1] == -1)


def _make_records(tmp_path):
    rows = []
    tasks = {'BUSI': [0], 'MMOTU': [1], 'DDTI': [2], 'AUL': [3, 4],
             'FALLMUD': [6], 'Fetal_HC': [8], 'CCA': [9],
             'CAMUS': [10, 11, 12], 'AbdomenUS': [13, 14, 16, 18, 20]}
    for split in ('train', 'val', 'test'):
        for domain in DOMAINS:
            image = tmp_path / f'{domain}_{split}.png'
            mask = tmp_path / f'{domain}_{split}_mask.png'
            Image.fromarray(np.arange(1024, dtype=np.uint8).reshape(32, 32)).save(image)
            indices = np.zeros((32, 32), dtype=np.uint8)
            indices[8:24, 8:24] = 1 if domain in ('CAMUS', 'AbdomenUS') else 255
            if domain == 'AbdomenUS':
                indices[0:3] = 3
                indices[3:6] = 5
                indices[6:8] = 7
                indices[24:27] = 255
            Image.fromarray(indices).save(mask)
            for task in tasks[domain]:
                rows.append(dict(image=str(image), mask=str(mask), task=task,
                                 class_id=(task - 12 if domain == 'AbdomenUS' else
                                           task - 9 if domain == 'CAMUS' else 0),
                                 dataset=domain, split=split, group=f'{domain}/{split}',
                                 split_protocol='synthetic_fixed', split_version='synthetic_v1',
                                 label_version='hc18_filled_v1' if domain == 'Fetal_HC' else 'original'))
    image = tmp_path / 'AUL_mass_only_train.png'
    mask = tmp_path / 'AUL_mass_only_train_mask.png'
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image)
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[10:14, 10:14] = 255
    Image.fromarray(pixels).save(mask)
    rows.append(dict(image=str(image), mask=str(mask), task=4, class_id=0,
                     dataset='AUL', split='train', group='AUL/mass_only',
                     split_protocol='synthetic_fixed', split_version='synthetic_v1',
                     label_version='original'))
    return rows


def _args(tmp_path, *more):
    return single.parse_args(['--data-root', str(tmp_path), '--output', str(tmp_path / 'run'),
                              '--size', '32', '--base', '4', '--batch-size', '3',
                              '--quota-per-domain', '1', '--epochs', '1', '--seeds', '42',
                              '--workers', '0', '--device', 'cpu', *more])


def test_union_dataset_keeps_exact_images_and_handles_partial_labels(tmp_path):
    records = _make_records(tmp_path)
    dataset = single.UnionDataset([row for row in records if row['split'] == 'train'], 32,
                                  return_multi_target=True)
    fair = RegionDataset([row for row in records if row['split'] == 'train'], 32,
                         PAPER_ALIGNED_TASK_IDS)
    assert [[r['image'] for r in group] for group in dataset.groups] == [
        [r['image'] for r in group] for group in fair.groups]
    assert len(dataset) == 10
    abdomen_index = next(i for i, group in enumerate(dataset.groups) if group[0]['dataset'] == 'AbdomenUS')
    _, target, domain, source, multi = dataset[abdomen_index]
    assert domain == 'AbdomenUS' and source == 'train'
    assert multi.shape[0] == 16 and target.shape == (1, 32, 32)
    assert target[0, 10, 10] == 1
    assert target[0, 0, 0] == target[0, 4, 0] == target[0, 7, 0] == -1
    assert target[0, 25, 0] == -1
    assert target[0, 30, 0] == 0
    mass_index = next(i for i, group in enumerate(dataset.groups)
                      if group[0]['image'].endswith('AUL_mass_only_train.png'))
    _, target, _, _, _ = dataset[mass_index]
    assert int((target == 1).sum()) == 16
    assert int((target == 0).sum()) == 0
    assert int((target == -1).sum()) == 32 * 32 - 16


def test_train_resume_and_test_gate_preserve_fair_draws(tmp_path, monkeypatch):
    records = _make_records(tmp_path)
    monkeypatch.setattr(single, 'build_records', lambda *a, **kw: (records, {}))
    args = _args(tmp_path, '--epochs', '2', '--stop-after-epoch', '1')
    single.run_suite(args)
    protocol = json.loads((args.output / 'protocol.json').read_text())
    assert protocol['config']['architecture'] == 'PlainUNet(outputs=1)'
    assert len(protocol['partial_annotation_images']) == 1
    assert protocol['train_images_by_domain']['AUL'] == 2
    assert not (args.output / 'seed_42/test_metrics.json').exists()
    checkpoint = torch.load(args.output / 'seed_42/last.pt', weights_only=True)
    assert checkpoint['model']['head.weight'].shape[0] == 1
    assert checkpoint['epoch'] == 1
    groups = single.UnionDataset([row for row in sorted(records, key=lambda r: json.dumps(r, sort_keys=True))
                                  if row['split'] == 'train'], 32).groups
    expected = [groups[i][0]['image'] for i in domain_draws(groups, 1, 42, 1)]
    assert checkpoint['metrics']['sample_sha256'] == single._digest(expected)
    assert checkpoint['metrics']['sample_count'] == 9
    assert checkpoint['metrics']['annotated_channel_queries'] >= 9
    resumed = _args(tmp_path, '--epochs', '2', '--resume', '--evaluate-test')
    single.run_suite(resumed)
    checkpoint = torch.load(args.output / 'seed_42/last.pt', weights_only=True)
    assert checkpoint['epoch'] == 2
    assert len((args.output / 'seed_42/metrics.jsonl').read_text().splitlines()) == 2
    test = json.loads((args.output / 'seed_42/test_metrics.json').read_text())
    assert test['checkpoint_selection'] == 'validation_macro_9_domain_positive_union_dice'
    assert test['positive_domains'] == 9
    assert 'per_structure' not in test
    assert test['per_abdomen_ct']['test']['samples'] == 1
    summary = json.loads((args.output / 'summary.json').read_text())
    assert summary['test_across_seeds']['macro_domain_positive_union_dice']['seeds'] == 1


def test_train_val_content_change_blocks_resume_without_touching_test(tmp_path, monkeypatch):
    records = _make_records(tmp_path)
    monkeypatch.setattr(single, 'build_records', lambda *a, **kw: (records, {}))
    args = _args(tmp_path, '--dry-run')
    # A missing test file is intentionally irrelevant until explicit test evaluation.
    Path(next(row['image'] for row in records if row['split'] == 'test')).unlink()
    single.run_suite(args)
    changed = next(row['mask'] for row in records if row['split'] == 'train')
    Image.fromarray(np.full((32, 32), 1, dtype=np.uint8)).save(changed)
    with pytest.raises(ValueError, match='content_sha256'):
        single.run_suite(_args(tmp_path, '--resume', '--dry-run'))


def test_reference_records_and_epoch_draws_are_enforced(tmp_path, monkeypatch):
    records = _make_records(tmp_path)
    monkeypatch.setattr(single, 'build_records', lambda *a, **kw: (records, {}))
    reference_dir = tmp_path / 'fair_shard'
    reference_dir.mkdir()
    reference_records = sorted(records, key=lambda row: json.dumps(row, sort_keys=True))
    records_path = reference_dir / 'records.json'
    records_path.write_text(json.dumps(reference_records))
    args = _args(tmp_path, '--reference-records', str(records_path))
    config_keys = ('size', 'base', 'batch_size', 'workers', 'quota_per_domain', 'epochs',
                   'split_seed', 'lr', 'weight_decay', 'lr_schedule', 'min_lr_ratio', 'seeds')
    (reference_dir / 'protocol.json').write_text(json.dumps({
        'config': {key: getattr(args, key) for key in config_keys}}))
    groups = RegionDataset([row for row in reference_records if row['split'] == 'train'],
                           32, PAPER_ALIGNED_TASK_IDS).groups
    indices = domain_draws(groups, 1, 42, 1)
    expected = dict(epoch=1,
                    sample_sha256=single._digest([groups[i][0]['image'] for i in indices]),
                    sample_count=9,
                    annotated_channel_queries=sum(len(groups[i]) for i in indices))
    history = reference_dir / 'region/seed_42/metrics.jsonl'
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps(expected) + '\n')
    single.run_suite(args)
    protocol = json.loads((args.output / 'protocol.json').read_text())
    assert protocol['reference_draws']['42']['1']['sample_sha256'] == expected['sample_sha256']
    assert json.loads((args.output / 'seed_42/metrics.jsonl').read_text())['annotated_channel_queries'] == expected['annotated_channel_queries']
    broken = dict(expected, sample_sha256='bad')
    history.write_text(json.dumps(broken) + '\n')
    with pytest.raises(ValueError, match='reference_draws_sha256'):
        single.run_suite(_args(tmp_path, '--reference-records', str(records_path), '--resume'))
