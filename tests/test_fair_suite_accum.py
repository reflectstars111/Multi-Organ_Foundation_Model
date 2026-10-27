"""CPU-only checks for effective-batch gradient accumulation and fair artifacts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from unet_moe import fair_suite_accum as suite
from unet_moe.region import DOMAINS


def _records(tmp_path):
    rows = []
    tasks = [[0], [1], [2], [3, 4], [6], [8], [9], [10, 11, 12], [13, 14, 16, 18, 20]]
    for split in ('train', 'val', 'test'):
        for domain, ids in zip(DOMAINS, tasks):
            image = tmp_path / f'{domain}_{split}.png'
            mask = tmp_path / f'{domain}_{split}_mask.png'
            Image.fromarray(np.arange(1024, dtype=np.uint8).reshape(32, 32)).save(image)
            labels = np.zeros((32, 32), dtype=np.uint8)
            labels[8:24, 8:24] = 1 if domain in ('CAMUS', 'AbdomenUS') else 255
            if domain == 'AbdomenUS':
                labels[:4] = 255
            Image.fromarray(labels).save(mask)
            for task in ids:
                rows.append(dict(image=str(image), mask=str(mask), task=task,
                                 class_id=task - 12 if domain == 'AbdomenUS' else
                                          task - 9 if domain == 'CAMUS' else 0,
                                 dataset=domain, split=split, group=f'{domain}/{split}',
                                 split_protocol='synthetic_fixed', split_version='synthetic_v1',
                                 label_version='hc18_filled_v1' if domain == 'Fetal_HC' else 'original'))
    return rows


def _args(tmp_path, *extra):
    return suite.parse_args(['--data-root', str(tmp_path), '--output', str(tmp_path / 'run'),
                             '--size', '32', '--base', '4', '--batch-size', '4',
                             '--micro-batch-size', '2', '--quota-per-domain', '1',
                             '--epochs', '1', '--seeds', '42', '--device', 'cpu', *extra])


def test_microbatch_weighting_matches_full_batch_adamw_even_for_final_partial_batch():
    torch.manual_seed(10)
    images = torch.randn(18, 1, 4, 4)
    targets = torch.randint(0, 2, (18, 1, 4, 4), dtype=torch.float32)
    targets[0, :, 0, 0] = -1
    initial = torch.nn.Conv2d(1, 1, kernel_size=1)
    full = torch.nn.Conv2d(1, 1, kernel_size=1)
    micro = torch.nn.Conv2d(1, 1, kernel_size=1)
    full.load_state_dict(initial.state_dict())
    micro.load_state_dict(initial.state_dict())
    full_optimizer = torch.optim.AdamW(full.parameters(), lr=1e-3, weight_decay=1e-4)
    micro_optimizer = torch.optim.AdamW(micro.parameters(), lr=1e-3, weight_decay=1e-4)
    for start in (0, 16):
        chunk_images, chunk_targets = images[start:start + 16], targets[start:start + 16]
        full_optimizer.zero_grad(set_to_none=True)
        suite.fair.common_loss(full(chunk_images), chunk_targets).backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 5.)
        full_optimizer.step()
    args = SimpleNamespace(device='cpu', batch_size=16, micro_batch_size=4,
                           empty_weight=0., balance_weight=0., router_z_weight=0.,
                           coarse_weight=0., route_weight=0.)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, targets),
                                         batch_size=4, shuffle=False)
    total, count, queries, steps = suite.train_effective_batches(
        micro, 'shared', loader, micro_optimizer, args, len(images), 42, 1)
    assert (count, queries, steps) == (18, 18, 2)
    assert np.isfinite(total)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(micro.state_dict()[key], value, rtol=1e-5, atol=1e-6)


def test_all_three_models_keep_identical_draws_and_explicit_test_gate(tmp_path, monkeypatch):
    records = _records(tmp_path)
    monkeypatch.setattr(suite.fair, 'build_records', lambda *args: (records, {}))
    args = _args(tmp_path)
    suite.run_suite(args)
    protocol = json.loads((args.output / 'protocol.json').read_text())
    assert protocol['config']['protocol_version'] == suite.PROTOCOL_VERSION
    assert protocol['config']['effective_batch_size'] == 4
    assert protocol['config']['physical_micro_batch_size'] == 2
    assert 'fair_suite_accum.py' in protocol['source_files']
    assert protocol['budget']['optimizer_steps_per_epoch'] == 3
    hashes = []
    for kind in suite.fair.MODELS:
        folder = args.output / kind / 'seed_42'
        metric = json.loads((folder / 'metrics.jsonl').read_text())
        hashes.append(metric['sample_sha256'])
        assert metric['sample_count'] == 9
        assert metric['optimizer_steps'] == 3
        assert metric['physical_micro_batches'] == 5
        assert not (folder / 'test_metrics.json').exists()
        checkpoint = torch.load(folder / 'last.pt', weights_only=True)
        assert checkpoint['protocol_version'] == suite.PROTOCOL_VERSION
        assert checkpoint['optimizer']['state']
    assert len(set(hashes)) == 1
    tested = _args(tmp_path, '--resume', '--evaluate-test')
    result = suite.run_suite(tested)
    assert len(result['runs']) == 3
    assert all(run['test_evaluated'] for run in result['runs'])
    assert all((args.output / kind / 'seed_42/test_metrics.json').is_file()
               for kind in suite.fair.MODELS)


def test_resume_is_exact_and_rejects_changed_source_hash(tmp_path, monkeypatch):
    records = _records(tmp_path)
    monkeypatch.setattr(suite.fair, 'build_records', lambda *args: (records, {}))
    first = _args(tmp_path, '--models', 'shared', '--epochs', '2', '--stop-after-epoch', '1')
    suite.run_suite(first)
    resumed = _args(tmp_path, '--models', 'shared', '--epochs', '2', '--resume')
    suite.run_suite(resumed)
    resumed_checkpoint = torch.load(first.output / 'shared/seed_42/last.pt', weights_only=True)
    complete = _args(tmp_path, '--models', 'shared', '--epochs', '2',
                     '--output', str(tmp_path / 'full'))
    suite.run_suite(complete)
    complete_checkpoint = torch.load(complete.output / 'shared/seed_42/last.pt', weights_only=True)
    for key, value in complete_checkpoint['model'].items():
        torch.testing.assert_close(resumed_checkpoint['model'][key], value, rtol=0, atol=0)
    assert resumed_checkpoint['epoch'] == 2
    protocol_file = first.output / 'protocol.json'
    protocol = json.loads(protocol_file.read_text())
    protocol['source_sha256'] = 'tampered'
    protocol_file.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='source_sha256'):
        suite.run_suite(_args(tmp_path, '--models', 'shared', '--epochs', '2',
                              '--resume', '--dry-run'))


def test_cli_rejects_wrong_microbatch_or_auxiliary_loss(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, '--batch-size', '5')
    with pytest.raises(SystemExit):
        _args(tmp_path, '--coarse-weight', '.1')
