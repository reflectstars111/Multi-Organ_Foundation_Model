import copy

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe.organ_weighting import weighted_loss
from unet_moe.routing_objective import compute_objective, objective_from_output


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('kind', ['region', 'task_id'])
@pytest.mark.parametrize('main_arm', ['image_mean', 'organ_exposure', 'organ_exposure_fg_bg'])
def test_control_exactly_reproduces_q01_objective_and_gradients(kind, main_arm):
    torch.manual_seed(17)
    model = fair.create_model(kind, base=4, experts=9, top_k=1)
    reference = copy.deepcopy(model)
    image = torch.randn(2, 1, 32, 32)
    target = torch.full((2, 16, 32, 32), -1.)
    target[0, -5:] = 0
    target[0, -3, 5:15, 5:15] = 1
    target[1, 0] = 0
    weights = dict(organ_weights=[2.] * 16, foreground_weights=[3.] * 16,
                   foreground_normalizers=[1.2] * 16)
    actual, components = compute_objective(model, kind, image, target, 'control', main_arm, weights)
    expected = weighted_loss(fair.forward_model(reference, kind, image, target)['logits'],
                             target, main_arm, weights)
    assert torch.equal(actual, expected)
    assert components['coarse'] == components['balance'] == 0
    actual.backward()
    expected.backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize('main_arm', ['image_mean', 'organ_exposure', 'organ_exposure_fg_bg'])
def test_region_coarse_uses_identical_weights_and_ignores_unknown(main_arm):
    target = torch.full((2, 16, 4, 4), -1.)
    target[0, -5:, :2] = 0
    target[0, -3, 0, 0] = 1
    target[1, 0] = 0
    logits = torch.randn_like(target, requires_grad=True)
    proposals = [torch.randn_like(target, requires_grad=True) for _ in range(4)]
    weights = dict(organ_weights=[2.] * 16, foreground_weights=[3.] * 16,
                   foreground_normalizers=[1.2] * 16)
    loss, terms = objective_from_output({'logits': logits, 'coarse': proposals}, target,
                                       'region', 'auxiliary', main_arm, weights)
    expected_aux = sum(weighted_loss(p, target, main_arm, weights) for p in proposals) / 4
    torch.testing.assert_close(loss, weighted_loss(logits, target, main_arm, weights) + .1 * expected_aux)
    loss.backward()
    assert terms['coarse'] > 0
    for p in proposals:
        assert p.grad[target < 0].count_nonzero() == 0
        assert p.grad[target >= 0].abs().sum() > 0


def test_task_balance_has_router_gradient_even_when_expert_assignment_is_fixed():
    model = fair.create_model('task_id', base=4, experts=9, top_k=1)
    image = torch.randn(2, 1, 32, 32)
    target = torch.full((2, 16, 32, 32), -1.)
    target[:, 0] = 0
    output = fair.forward_model(model, 'task_id', image, target)
    loss, terms = objective_from_output(output, target, 'task_id', 'auxiliary', 'image_mean', None)
    expected = weighted_loss(output['logits'], target) + .01 * output['balance']
    torch.testing.assert_close(loss, expected)
    grads = torch.autograd.grad(output['balance'], [layer.router.weight for layer in model.decoders])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert terms['balance'] > 0 and terms['coarse'] == 0


def test_missing_region_proposals_is_not_silently_accepted():
    with pytest.raises(ValueError, match='four'):
        objective_from_output({'logits': torch.zeros(1, 16, 4, 4)}, torch.zeros(1, 16, 4, 4),
                              'region', 'auxiliary', 'image_mean', None)
