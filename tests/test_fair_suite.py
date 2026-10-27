"""Behavioral checks for the frozen, image-matched architecture comparison."""
import importlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from PIL import Image

from unet_moe.data import TASKS
from unet_moe.region import DOMAINS, PAPER_ALIGNED_TASK_IDS


@pytest.fixture
def fair():
    assert importlib.util.find_spec('unet_moe.fair_suite') is not None, 'fair comparison entrypoint is missing'
    return importlib.import_module('unet_moe.fair_suite')


@pytest.fixture
def records(tmp_path):
    """Real PNGs with positive, empty, and ignored labels in every domain."""
    rows = []
    tasks = [[0], [1], [2], [3, 4], [6], [8], [9], [10, 11, 12], [13, 14, 16, 18, 20]]
    for split in ('train', 'val', 'test'):
        for domain, ids in zip(DOMAINS, tasks):
            image = tmp_path / f'{domain}_{split}.png'
            mask = tmp_path / f'{domain}_{split}_mask.png'
            Image.fromarray(np.arange(1024, dtype=np.uint8).reshape(32, 32)).save(image)
            indices = np.zeros((32, 32), dtype=np.uint8)
            indices[8:24, 8:24] = 1 if domain in ('CAMUS', 'AbdomenUS') else 255
            if domain == 'AbdomenUS':
                indices[:4] = 255
            Image.fromarray(indices).save(mask)
            for task in ids:
                rows.append(dict(image=str(image), mask=str(mask), task=task,
                                 class_id=task - 12 if domain == 'AbdomenUS' else task - 9 if domain == 'CAMUS' else 0,
                                 dataset=domain, split=split, group=f'{domain}/{split}',
                                 split_protocol='synthetic_fixed', split_version='synthetic_v1',
                                 label_version='hc18_filled_v1' if domain == 'Fetal_HC' else 'original'))
    return rows


def args_for(fair, tmp_path, *extra):
    return fair.parse_args(['--data-root', str(tmp_path), '--output', str(tmp_path / 'run'),
                            '--size', '32', '--base', '4', '--batch-size', '3',
                            '--quota-per-domain', '1', '--epochs', '1', '--seeds', '42',
                            '--device', 'cpu', *extra])


def test_missing_labels_have_zero_gradient_and_channels_are_averaged_per_image(fair):
    logits = torch.zeros(2, 3, 2, 2, requires_grad=True)
    target = torch.full_like(logits, -1)
    target[0, 0] = 0
    target[1, 0] = 1
    target[1, 1] = 0
    target[1, 0, 0, 0] = -1
    loss = fair.common_loss(logits, target)
    # For zero logits: empty-channel Dice loss=2/3; positive 3px=1-4/5.5.
    expected = np.log(2) + ((2 / 3) + ((1 - 4 / 5.5) + 2 / 3) / 2) / 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[target < 0]) == 0
    assert torch.count_nonzero(logits.grad[target >= 0]) == (target >= 0).sum()


@pytest.mark.parametrize('kind', ['shared', 'task_id', 'region'])
def test_all_architectures_train_on_same_partial_targets(fair, kind):
    torch.manual_seed(10)
    model = fair.create_model(kind, base=4, top_k=2, experts=9)
    image = torch.rand(2, 1, 32, 32)
    target = torch.full((2, len(PAPER_ALIGNED_TASK_IDS), 32, 32), -1.)
    target[0, 0] = 0
    target[0, 0, 8:16, 8:16] = 1
    target[1, -1] = 1
    target[1, -2] = 0
    target[1, -1, :4] = -1
    output = fair.forward_model(model, kind, image, target)
    assert output['logits'].shape == target.shape
    output['logits'].retain_grad()
    loss = fair.common_loss(output['logits'], target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(output['logits'].grad[target < 0]) == 0
    assert model.head.weight.grad.abs().sum() > 0


def test_task_id_forward_uses_only_annotated_image_task_pairs(fair):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, image, task):
            self.seen = (image[:, 0, 0, 0].tolist(), task.tolist())
            return {'logits': image + task[:, None, None, None]}

    model = RecordingModel()
    image = torch.tensor([1., 2.])[:, None, None, None].expand(2, 1, 2, 2)
    target = torch.full((2, len(PAPER_ALIGNED_TASK_IDS), 2, 2), -1.)
    target[0, 0] = 0
    target[0, -1] = 1
    target[1, 1] = 0
    result = fair.forward_model(model, 'task_id', image, target)['logits']
    assert model.seen == ([1., 1., 2.], [0, 20, 1])
    assert result[0, -1, 0, 0] == 21
    assert result[1, 1, 0, 0] == 3


def test_metrics_mask_ignore_and_report_positive_misses_and_fp_area(fair):
    target = torch.tensor([[[[1., 0.], [-1., -1.]]], [[[0., 0.], [0., -1.]]]])
    logits = torch.tensor([[[[-10., -10.], [10., 10.]]], [[[10., -10.], [-10., 10.]]]])
    class FixedModel(torch.nn.Module):
        def forward(self, image):
            return logits
    metrics = fair.evaluate(FixedModel(), 'shared', [(torch.zeros(2, 1, 2, 2), target)],
                            'cpu', task_ids=[0])
    row = metrics['per_structure'][TASKS[0]]
    assert row['positive_dice'] == 0
    assert row['dice'] == 0
    assert row['empty_false_positive_rate'] == 1
    assert row['positive_miss_rate'] == 1
    assert row['positive_zero_overlap_rate'] == 1
    assert row['mean_false_positive_pixels'] == .5
    assert row['mean_false_positive_area_fraction'] == pytest.approx(1 / 6)
    assert row['empty_mean_false_positive_pixels'] == 1
    assert row['empty_mean_false_positive_area_fraction'] == pytest.approx(1 / 3)
    assert row['positive_samples'] == row['empty_samples'] == 1


def test_suite_identical_sampling_and_test_gate(fair, records, tmp_path, monkeypatch):
    monkeypatch.setattr(fair, 'build_records', lambda *a, **kw: (records, {}))
    # If default training touches test images, opening these missing files fails.
    for row in records:
        if row['split'] == 'test':
            Path(row['image']).unlink(missing_ok=True)
    args = args_for(fair, tmp_path)
    fair.run_suite(args)
    protocol = json.loads((args.output / 'protocol.json').read_text())
    assert protocol['config']['split_seed'] == 42
    assert protocol['config']['coarse_weight'] == protocol['config']['route_weight'] == 0
    hashes = []
    for kind in ['shared', 'task_id', 'region']:
        folder = args.output / kind / 'seed_42'
        row = json.loads((folder / 'metrics.jsonl').read_text())
        hashes.append(row['sample_sha256'])
        assert row['sample_count'] == 9
        assert not (folder / 'test_metrics.json').exists()
        ckpt = torch.load(folder / 'last.pt', weights_only=True)
        assert ckpt['optimizer']['state']
        assert ckpt['rng']['torch'].numel() > 0
    assert len(set(hashes)) == 1


def test_split_seed_is_independent_of_training_seeds_and_dry_run(fair, records, tmp_path, monkeypatch):
    received = []
    def build(root, seed, abdomen_split, abdomen_fold, split_policy):
        received.append((seed, abdomen_split, split_policy))
        return records, {}
    monkeypatch.setattr(fair, 'build_records', build)
    args = args_for(fair, tmp_path, '--seeds', '43,44', '--dry-run')
    fair.run_suite(args)
    assert received == [(42, 'ct7_2', 'trainmore_v6')]
    assert (args.output / 'protocol.json').exists()
    assert not list(args.output.rglob('*.pt'))
    with pytest.raises(SystemExit):
        args_for(fair, tmp_path, '--split-seed', '43')


def test_resume_restores_exact_training_and_rejects_protocol_changes(fair, records, tmp_path, monkeypatch):
    monkeypatch.setattr(fair, 'build_records', lambda *a, **kw: (records, {}))
    args = args_for(fair, tmp_path, '--models', 'shared', '--epochs', '2', '--stop-after-epoch', '1')
    fair.run_suite(args)
    resumed = args_for(fair, tmp_path, '--models', 'shared', '--epochs', '2', '--resume')
    fair.run_suite(resumed)
    resumed_ckpt = torch.load(args.output / 'shared/seed_42/last.pt', weights_only=True)
    full = args_for(fair, tmp_path, '--models', 'shared', '--epochs', '2',
                    '--output', str(tmp_path / 'full'))
    fair.run_suite(full)
    full_ckpt = torch.load(full.output / 'shared/seed_42/last.pt', weights_only=True)
    for name, value in full_ckpt['model'].items():
        torch.testing.assert_close(resumed_ckpt['model'][name], value, rtol=0, atol=0)
    assert resumed_ckpt['epoch'] == 2
    assert len((args.output / 'shared/seed_42/metrics.jsonl').read_text().splitlines()) == 2
    changed = args_for(fair, tmp_path, '--models', 'shared', '--epochs', '2', '--resume', '--lr', '.001')
    with pytest.raises(ValueError, match='[Cc]onfig|protocol'):
        fair.run_suite(changed)
    records[0]['label_version'] = 'changed'
    with pytest.raises(ValueError, match='[Rr]ecord|protocol'):
        fair.run_suite(resumed)


def test_explicit_test_evaluation_uses_validation_selected_checkpoint(fair, records, tmp_path, monkeypatch):
    monkeypatch.setattr(fair, 'build_records', lambda *a, **kw: (records, {}))
    args = args_for(fair, tmp_path, '--models', 'shared', '--evaluate-test')
    fair.run_suite(args)
    folder = args.output / 'shared/seed_42'
    metrics = json.loads((folder / 'test_metrics.json').read_text())
    best = torch.load(folder / 'best.pt', weights_only=True)
    assert metrics['checkpoint_epoch'] == best['epoch']
    assert metrics['checkpoint_selection'] == 'best_validation_macro_positive_dice'
    summary = json.loads((args.output / 'summary.json').read_text())
    assert summary['test_across_seeds']['shared']['macro_positive_dice']['mean'] == metrics['macro_positive_dice']


def test_excluded_labels_are_absent_from_frozen_records(fair, records, tmp_path, monkeypatch):
    excluded = dict(next(row for row in records if row['dataset'] == 'AbdomenUS'), task=15, class_id=3)
    monkeypatch.setattr(fair, 'build_records', lambda *a, **kw: (records + [excluded], {}))
    args = args_for(fair, tmp_path, '--dry-run')
    fair.run_suite(args)
    frozen = json.loads((args.output / 'records.json').read_text())
    assert len(frozen) == len(records)
    assert all(row['task'] != 15 for row in frozen)


def test_optional_empty_loss_is_abdomen_only_and_default_disabled(fair, tmp_path):
    target = torch.full((1, len(PAPER_ALIGNED_TASK_IDS), 2, 2), -1.)
    target[0, 0] = 0  # Non-abdominal negative must not receive the auxiliary term.
    target[0, -1] = 0
    logits = torch.zeros_like(target, requires_grad=True)
    args = args_for(fair, tmp_path, '--empty-weight', '.1')
    output = {'logits': logits}
    main = fair.common_loss(logits, target)
    loss = fair.objective(None, 'shared', output, target, args)
    extra_gradient = torch.autograd.grad(loss - main, logits)[0]
    assert loss.item() == pytest.approx(main.item() + .1 * np.log(2))
    assert extra_gradient[0, 0].count_nonzero() == 0
    assert extra_gradient[0, -1].sum() > 0
    defaults = args_for(fair, tmp_path)
    torch.testing.assert_close(fair.objective(None, 'shared', output, target, defaults), main)


def test_rng_roundtrip_restores_python_numpy_and_torch(fair):
    fair._seed(42)
    state = fair._rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    fair._seed(99)
    fair._restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_resume_rejects_changed_source_hash(fair, records, tmp_path, monkeypatch):
    monkeypatch.setattr(fair, 'build_records', lambda *a, **kw: (records, {}))
    args = args_for(fair, tmp_path, '--dry-run')
    fair.run_suite(args)
    protocol_file = args.output / 'protocol.json'
    protocol = json.loads(protocol_file.read_text())
    assert protocol['source_sha256']
    protocol['source_sha256'] = 'changed'
    protocol_file.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='[Ss]ource'):
        fair.run_suite(args_for(fair, tmp_path, '--resume', '--dry-run'))
