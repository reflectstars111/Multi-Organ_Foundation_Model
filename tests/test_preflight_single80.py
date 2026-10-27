"""The union resource probe must use the paired training image draw."""

import json
from types import SimpleNamespace

import pytest

from scripts import preflight_single80 as probe
from scripts.preflight_single80 import planned_first_batch
from unet_moe.region import DOMAINS
from unet_moe.sampling import domain_draws


def test_single80_preflight_uses_first_paired_batch():
    groups = [[{'dataset': domain, 'image': f'{domain}.png'}]
              for domain in DOMAINS]
    selection = planned_first_batch(groups)
    assert selection['indices'] == domain_draws(groups, 500, 42, 1)[:16]
    assert selection['images'] == 16
    assert len(selection['sample_sha256']) == 64


def test_single80_initializes_cuda_before_reset_and_records_probe_failure(tmp_path, monkeypatch):
    records = [{'split': 'train'}]
    (tmp_path / 'records.json').write_text(json.dumps(records))
    (tmp_path / 'protocol.json').write_text(json.dumps({
        'config': dict(size=512, base=24, batch_size=16, epochs=80,
                       quota_per_domain=500, lr=.0001, weight_decay=.0001,
                       split_seed=42, seeds=[42, 43, 44]),
        'record_sha256': probe._digest(records), 'source_sha256': 'source',
    }))

    class TrainingImages:
        groups = []

        def __init__(self, *args):
            pass

        def __len__(self):
            return 6737

        def __getitem__(self, index):
            raise RuntimeError('stop after CUDA initialization')

    initialized = False

    def memory_info(device):
        nonlocal initialized
        initialized = True
        return 40 * 2**30, 48 * 2**30

    def reset(device):
        if not initialized:
            raise RuntimeError('allocator not initialized')

    monkeypatch.setattr(probe, 'UnionDataset', TrainingImages)
    monkeypatch.setattr(probe, 'planned_first_batch', lambda _: {'indices': [0]})
    monkeypatch.setattr(probe, '_seed', lambda _: None)
    monkeypatch.setattr(probe.torch.cuda, 'mem_get_info', memory_info)
    monkeypatch.setattr(probe.torch.cuda, 'reset_peak_memory_stats', reset)
    args = SimpleNamespace(records=tmp_path / 'records.json',
                           output=tmp_path / 'probe.json', device='cuda:0')
    with pytest.raises(RuntimeError, match='stop after CUDA initialization'):
        probe.run(args)
    report = json.loads(args.output.read_text())
    assert report['status'] == 'failed'
    assert report['gpu_total_mib'] == 49152
    assert report['error'] == 'RuntimeError: stop after CUDA initialization'
