"""Q01 diagnostics must reveal organ failure without perturbing training."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from unet_moe.data import TASKS
from unet_moe.region import PAPER_ALIGNED_TASK_IDS
from unet_moe import organ_diagnostics as diagnostics


def test_single_pass_metrics_stratify_sources_and_ignore_pixels():
    target = torch.full((3, 16, 2, 2), -1.)
    channel = PAPER_ALIGNED_TASK_IDS.index(16)
    target[:, channel] = torch.tensor([[[1., 0.], [-1., -1.]],
                                      [[0., 0.], [0., -1.]],
                                      [[1., 0.], [0., 0.]]])
    logits = torch.full_like(target, -10.)
    logits[0, channel, 0, 1] = 10.  # Nonempty but zero intersection.
    logits[0, channel, 1] = 10.  # Ignored pixels must contribute nothing.
    logits[1, channel, 0, 0] = 10.
    logits[2, channel, 0, 0] = 10.

    class Fixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, image):
            self.calls += 1
            return logits[image[:, 0, 0, 0].long()]

    model = Fixed()
    image = torch.arange(3.)[:, None, None, None].expand(3, 1, 2, 2)
    loader = DataLoader(TensorDataset(image, target), batch_size=2)
    groups = [[{'dataset': 'AbdomenUS', 'group': f'AbdomenUS/{source}'}]
              for source in ('ct3', 'ct3', 'ct4')]
    result = diagnostics.evaluate(model, 'shared', loader, 'cpu', groups=groups)
    assert model.calls == 2
    row = result['per_structure']['abdomen_vessels']
    assert row['positive_dice'] == .5
    assert row['positive_zero_overlap_rate'] == .5
    assert row['positive_miss_rate'] == 0
    assert row['empty_false_positive_rate'] == 1
    assert row['empty_mean_false_positive_area_fraction'] == pytest.approx(1 / 3)
    assert result['abdomen_macro_positive_dice'] == .5
    ct3 = result['abdomen_by_source']['ct3']['per_structure']
    assert set(ct3) == {TASKS[task] for task in (13, 14, 16, 18, 20)}
    assert ct3['abdomen_vessels']['positive_dice'] == 0
    assert result['abdomen_by_source']['ct4']['per_structure']['abdomen_vessels']['positive_dice'] == 1
    from unet_moe.fair_suite import evaluate as fair_evaluate
    reference = fair_evaluate(model, 'shared', loader, 'cpu')
    for key, value in reference.items():
        if key == 'per_structure':
            for name, row in value.items():
                for field, expected in row.items():
                    actual = result[key][name][field]
                    assert actual is None if expected is None else actual == pytest.approx(expected)
        else:
            assert result[key] is None if value is None else result[key] == pytest.approx(value)


def test_evaluation_refuses_misaligned_or_random_group_metadata():
    image = torch.zeros(2, 1, 2, 2)
    target = torch.zeros(2, 16, 2, 2)
    model = nn.Conv2d(1, 16, 1)
    groups = [[{'dataset': 'AbdomenUS', 'group': 'AbdomenUS/ct3'}]]
    with pytest.raises(ValueError, match='groups'):
        diagnostics.evaluate(model, 'shared', DataLoader(TensorDataset(image, target)), 'cpu', groups)
    with pytest.raises(ValueError, match='sequential'):
        diagnostics.evaluate(model, 'shared', DataLoader(TensorDataset(image, target), shuffle=True),
                             'cpu', groups * 2)


def test_guardrail_rejects_missing_denominators_and_single_organ_collapse():
    metrics = {'abdomen_macro_positive_dice': .6, 'per_structure': {
        TASKS[task]: {'positive_dice': .6, 'positive_zero_overlap_rate': .1,
                     'empty_mean_false_positive_area_fraction': .01}
        for task in (13, 14, 16, 18, 20)}}
    assert diagnostics.guardrail_passes(metrics)
    for field, value in [('positive_dice', .19), ('positive_zero_overlap_rate', .51),
                         ('empty_mean_false_positive_area_fraction', .0201),
                         ('empty_mean_false_positive_area_fraction', None),
                         ('positive_dice', float('nan'))]:
        candidate = copy.deepcopy(metrics)
        candidate['per_structure']['abdomen_vessels'][field] = value
        assert not diagnostics.guardrail_passes(candidate)
    candidate = copy.deepcopy(metrics)
    del candidate['per_structure']['abdomen_spleen']
    assert not diagnostics.guardrail_passes(candidate)
    metrics['abdomen_macro_positive_dice'] = .39
    assert not diagnostics.guardrail_passes(metrics)


def test_gradient_probe_preserves_rng_modes_buffers_parameters_and_existing_grads():
    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.Sequential(nn.Conv2d(1, 4, 1), nn.BatchNorm2d(4), nn.Dropout(.4))
            self.head = nn.Conv2d(4, 16, 1)
            self.register_buffer('probe_counter', torch.zeros(()))

        def forward(self, image):
            # Deliberate random consumption tests isolation beyond dropout.eval().
            random.random()
            np.random.random()
            torch.rand(1)
            self.probe_counter.add_(1)
            return self.head(self.encoders(image))

    model = ProbeModel().train()
    model.encoders[2].eval()  # Preserve heterogeneous submodule mode flags.
    image = torch.randn(2, 1, 4, 4)
    target = torch.full((2, 16, 4, 4), -1.)
    target[:, -5:] = 0
    target[0, -3, :2, :2] = 1
    dataset = TensorDataset(image, target)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    state = copy.deepcopy(model.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    modes = [module.training for module in model.modules()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    weights = {'organ_weights': [2.] * 16, 'foreground_weights': [1.] * 16,
               'foreground_normalizers': [1.] * 16, 'foreground_proportions': [.1] * 16}
    result = diagnostics.diagnose_gradients(model, 'shared', dataset, [0, 1], 'cpu',
                                            'organ_exposure', weights)
    assert random.getstate() == rng[0]
    after_np = np.random.get_state()
    assert after_np[0] == rng[1][0] and np.array_equal(after_np[1], rng[1][1])
    assert after_np[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(p.grad, old) for p, old in zip(model.parameters(), gradients))
    vessels = result['per_structure']['abdomen_vessels']
    assert vessels['samples'] == 2 and vessels['positive_samples'] == vessels['empty_samples'] == 1
    assert vessels['weighted_loss_contribution_mean'] == pytest.approx(
        2 * vessels['unweighted_loss_contribution_mean'])
    assert vessels['weighted_encoder_gradient_norm_mean'] == pytest.approx(
        2 * vessels['unweighted_encoder_gradient_norm_mean'], rel=1e-5)
    assert 'single-image' in result['gradient_aggregation']
    assert result['probe_indices'] == [0, 1]


@pytest.mark.parametrize('kind', ['shared', 'task_id', 'region'])
def test_gradient_probe_supports_all_architectures_without_populating_grads(kind):
    from unet_moe.fair_suite import create_model

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_model(kind, base=4, experts=9, top_k=1)
        image = torch.randn(1, 1, 32, 32)
        target = torch.full((1, 16, 32, 32), -1.)
        target[:, -5:] = 0
        target[:, -3, 8:16, 8:16] = 1
        weights = {'organ_weights': [1.] * 16, 'foreground_weights': [1.] * 16,
                   'foreground_normalizers': [1.] * 16, 'foreground_proportions': [.1] * 16}
        result = diagnostics.diagnose_gradients(model, kind, TensorDataset(image, target),
                                                [0], 'cpu', 'image_mean', weights)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert model.training
        for name in diagnostics.ABDOMEN_STRUCTURES:
            row = result['per_structure'][name]
            assert row['samples'] == 1
            assert row['unweighted_encoder_gradient_norm_mean'] > 0
            assert row['weighted_encoder_gradient_norm_mean'] == pytest.approx(
                row['unweighted_encoder_gradient_norm_mean'])
    finally:
        torch.set_num_threads(threads)
