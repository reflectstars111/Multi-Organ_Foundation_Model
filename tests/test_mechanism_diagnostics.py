import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from unet_moe import mechanism_models as fair
from unet_moe import organ_diagnostics as organs
from unet_moe.mechanism_diagnostics import diagnose, evaluate_routing, zero_expert_residual


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def data(split='val'):
    image = torch.randn(2, 1, 32, 32)
    target = torch.full((2, 16, 32, 32), -1.)
    target[:, -5:] = 0
    target[0, -3, 8:16, 8:16] = 1
    target[1, -1, 8:16, 8:16] = 1
    dataset = TensorDataset(image, target)
    dataset.groups = [[dict(dataset='AbdomenUS', group='AbdomenUS/ct3', split=split)] for _ in image]
    return dataset


@pytest.mark.parametrize('kind,arm', [('region','control'), ('region','routing'),
    ('task_id','experts0'), ('task_id','experts1'), ('task_id','experts9')])
def test_routing_complete_counts_strata_and_read_only(kind, arm):
    torch.manual_seed(92)
    model = fair.create_model(kind, base=4, experts=9, top_k=1, arm=arm).train()
    model.encoders[0].eval()
    ds = data()
    loader = DataLoader(ds, batch_size=1, generator=torch.Generator().manual_seed(42))
    loader_state = loader.generator.get_state().clone()
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    before = copy.deepcopy(model.state_dict())
    grads = [(p.grad, p.grad.clone()) for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    state = torch.get_rng_state().clone()
    result = evaluate_routing(model, kind, loader, 'cpu', groups=ds.groups)
    assert torch.equal(loader_state, loader.generator.get_state())
    assert torch.equal(state, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(before[n], v) for n, v in model.state_dict().items())
    assert all(p.grad is obj and torch.equal(p.grad, value) for p, (obj, value) in zip(model.parameters(), grads))
    assert all(not m._forward_hooks for m in model.modules())
    reference = organs.evaluate(model, kind, loader, 'cpu', groups=ds.groups)
    assert result['normal'] == reference
    for layer in result['layers']:
        if arm == 'experts0':
            assert layer['all']['state'] == 'not_applicable'
            assert layer['all']['selected_counts'] == []
            assert result['zero_expert_residual'] is None
            assert result['normal_minus_zero']['macro_positive_dice'] is None
            continue
        assert len(layer['all']['selected_counts']) == (1 if arm == 'experts1' else 9)
        assert layer['all']['queries'] == (32 if kind == 'region' else 10)
        assert sum(layer['all']['selected_counts']) == layer['all']['queries']
        assert layer['by_domain']['AbdomenUS']['queries'] == layer['all']['queries']
        vessel = layer['by_structure']['abdomen_vessels']
        assert vessel['all']['queries'] == 2
        assert vessel['positive']['queries'] == vessel['empty']['queries'] == 1
        assert layer['by_structure']['breast_lesion']['all']['queries'] == 0
        if kind == 'region':
            assert layer['proposal_metrics']['per_structure']['abdomen_vessels']['positive_samples'] == 1


@pytest.mark.parametrize('kind,arm', [('region','control'), ('region','routing'),
    ('task_id','experts0'), ('task_id','experts1'), ('task_id','experts9')])
def test_fixed_expert_gradient_probe_preserves_existing_gradients(kind, arm):
    model = fair.create_model(kind, base=4, experts=9, top_k=1, arm=arm).train()
    ds = data('train')
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    gradients = [p.grad.clone() for p in model.parameters()]
    state = torch.get_rng_state().clone()
    result = diagnose(model, kind, ds, [0, 1], 'cpu', 'image_mean', None, arm=arm)
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), gradients))
    assert model.training
    assert result['probe_images'] == 2
    assert result['objective_arm'] == arm
    adapters = [v for key, v in result['modules'].items() if '/contrast_projection' in key or '/film' in key]
    assert len(adapters) == 4
    if adapters:
        assert all(row['nonzero_images'] > 0 for row in adapters)
    if arm == 'experts0':
        assert not any('/expert_' in key or '/router' in key for key in result['modules'])
    else:
        assert any(v['nonzero_images'] > 0 for key, v in result['modules'].items() if '/expert_' in key)
    assert all(v['mean_gradient_norm'] >= 0 for v in result['modules'].values())
    if kind == 'region':
        assert result['mean_objective']['coarse'] > 0
    elif arm == 'experts9':
        assert result['mean_objective']['balance'] > 0
    else:
        assert result['mean_objective']['balance'] == 0


def test_residual_hooks_removed_on_exception_and_test_ablation_rejected():
    model = fair.create_model('region', base=4, arm='control')
    with pytest.raises(RuntimeError):
        with zero_expert_residual(model, 'region'):
            assert any(m._forward_hooks for m in model.modules())
            raise RuntimeError('probe failed')
    assert all(not m._forward_hooks for m in model.modules())
    ds = data('test')
    with pytest.raises(ValueError, match='validation'):
        evaluate_routing(model, 'region', DataLoader(ds), 'cpu', groups=ds.groups)
