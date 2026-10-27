"""The resource probe must cover the heaviest planned Task-ID image batch."""

import pytest

pytest.importorskip('unet_moe.fair_suite')

from scripts.preflight_fair80 import prepare_cuda_memory_stats, worst_batch
from unet_moe.region import DOMAINS
from unet_moe.sampling import domain_draws


def test_worst_batch_covers_maximum_query_expansion():
    groups = []
    for index, domain in enumerate(DOMAINS):
        groups.append([{'dataset': domain, 'image': f'{domain}.png'}]
                      * (1 + index % 5))
    best = worst_batch(groups, seeds=(42, 43), epochs=3, quota=4,
                       batch_size=5, micro_batch_size=2)
    actual_max = max(
        sum(len(groups[index]) for index in drawn[start + offset:start + offset + 2])
        for seed in (42, 43) for epoch in range(1, 4)
        for drawn in (domain_draws(groups, 4, seed, epoch),)
        for start in range(0, len(drawn), 5)
        for offset in range(0, min(5, len(drawn) - start), 2))
    assert best['max_microbatch_queries'] == actual_max
    assert best['images'] <= 5
    assert len(best['indices']) == best['images']
    assert len(best['sample_sha256']) == 64


def test_cuda_preflight_initializes_context_before_reset(monkeypatch):
    calls = []

    def memory_info(device):
        calls.append(('mem_get_info', device))
        return 40 * 2**30, 48 * 2**30

    def reset(device):
        calls.append(('reset_peak_memory_stats', device))

    monkeypatch.setattr('torch.cuda.mem_get_info', memory_info)
    monkeypatch.setattr('torch.cuda.reset_peak_memory_stats', reset)
    report = {}
    prepare_cuda_memory_stats('cuda:0', report)
    assert calls == [('mem_get_info', 'cuda:0'),
                     ('reset_peak_memory_stats', 'cuda:0')]
    assert report['gpu_free_mib_before'] == 40960
    assert report['gpu_total_mib'] == 49152
