import pytest
import torch
from unet_moe.mechanism_benchmark import benchmark_indices, latency_statistics, invoked_parameter_count


def test_benchmark_uses_fixed_validation_abdomen_only():
    groups = [[dict(split='val', dataset='BUSI')]] + [[dict(split='val', dataset='AbdomenUS')] for _ in range(5)]
    assert benchmark_indices(groups) == [1, 2, 3, 4]
    groups[2][0]['split'] = 'test'
    with pytest.raises(ValueError, match='validation'):
        benchmark_indices(groups)


def test_benchmark_never_silently_changes_input_budget():
    with pytest.raises(ValueError, match='four'):
        benchmark_indices([[dict(split='val', dataset='AbdomenUS')]])


def test_invoked_parameters_counts_only_executed_modules_and_deduplicates():
    model = torch.nn.ModuleDict({'used': torch.nn.Linear(2, 3), 'unused': torch.nn.Linear(2, 7)})
    x = torch.ones(1, 2)
    def run():
        return model['used'](x) + model['used'](x)
    assert invoked_parameter_count(model, run) == 9
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_hooks_are_removed_on_failed_measurement():
    model = torch.nn.Linear(2, 3)
    with pytest.raises(RuntimeError):
        invoked_parameter_count(model, lambda: model(torch.ones(1, 8)))
    assert not model._forward_pre_hooks


def test_latency_statistics_rejects_invalid_samples():
    assert latency_statistics([1., 2., 3.])['median_ms'] == 2.
    for values in ([], [float('nan')], [-1.], [float('inf')]):
        with pytest.raises(ValueError):
            latency_statistics(values)
