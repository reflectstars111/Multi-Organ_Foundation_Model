"""Stable Q06 routing gradients and Q07 loss ablations retain label semantics."""
import copy
import math

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe import routing_objective as q02
from unet_moe.mechanism_models import create_model
from unet_moe.mechanism_objective import (
    REGION_ROUTING_WEIGHT, compute_objective, objective_from_output, positive_reverse_kl,
)
from unet_moe.organ_weighting import weighted_loss


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(size=32):
    g = torch.Generator().manual_seed(312)
    image = torch.randn(2, 1, size, size, generator=g)
    target = torch.full((2, 16, size, size), -1.)
    target[0, 0] = 0
    target[1, -5:] = 0
    target[1, -3, 6:18, 8:20] = 1
    target[:, :, :2] = -1
    return image, target


def weights():
    return dict(organ_weights=[2.] * 16, foreground_weights=[3.] * 16,
                foreground_normalizers=[1.2] * 16)


@pytest.mark.parametrize('kind,arm', [('region', 'control'), ('task_id', 'experts9')])
def test_retained_objectives_exactly_match_q05_auxiliary_gradients(kind, arm):
    torch.manual_seed(42)
    model = create_model(kind, base=4, arm=arm)
    reference = copy.deepcopy(model)
    image, target = sample()
    actual, terms = compute_objective(model, kind, image, target, arm, 'organ_exposure_fg_bg', weights())
    expected, old_terms = q02.compute_objective(reference, kind, image, target, 'auxiliary', 'organ_exposure_fg_bg', weights())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert terms['routing'] == 0
    for key in old_terms:
        assert terms[key] == old_terms[key]
    actual.backward()
    expected.backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_reverse_kl_uses_mean_distribution_only_over_positive_annotated_queries():
    scores = torch.randn(2, 3, 9, requires_grad=True)
    target = torch.full((2, 3, 4, 4), -1.)
    target[0, 0] = 0
    target[0, 1] = 0; target[0, 1, 1, 2] = 1
    target[1, 2] = 0; target[1, 2, 2, 1] = 1
    actual = positive_reverse_kl(scores, target)
    expected = -scores[[0, 1], [1, 2]].softmax(-1).mean(0).log().mean() - math.log(9)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.backward()
    positive = target.gt(0).any((2, 3))
    assert scores.grad[~positive].count_nonzero() == 0
    assert scores.grad[positive].abs().sum() > 0


def test_collapsed_large_logits_have_finite_corrective_gradients_without_clamping():
    scores = torch.full((2, 3, 9), -1000.)
    scores[..., 0] = 1000.
    scores.requires_grad_()
    target = torch.ones(2, 3, 2, 2)
    loss = positive_reverse_kl(scores, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()
    assert torch.all(scores.grad[..., 0] > 0)
    assert torch.all(scores.grad[..., 1:] < 0)
    assert scores.grad.abs().sum() > 1
    after = positive_reverse_kl(scores.detach() - scores.grad, target)
    assert after < loss.detach()


def test_balanced_marginal_does_not_force_every_individual_query_uniform():
    scores = torch.eye(9).mul(30).unsqueeze(0).requires_grad_()
    loss = positive_reverse_kl(scores, torch.ones(1, 9, 2, 2))
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert scores.grad.abs().max() < 1e-6


@pytest.mark.parametrize('fill', [-1., 0.])
def test_no_positive_query_is_connected_zero_with_zero_gradient(fill):
    scores = torch.randn(2, 3, 9, requires_grad=True)
    penalty = positive_reverse_kl(scores, torch.full((2, 3, 2, 2), fill))
    assert penalty.item() == 0 and penalty.requires_grad
    penalty.backward()
    assert scores.grad is not None and scores.grad.count_nonzero() == 0


def test_q06_only_last_layer_scores_receive_added_route_gradient():
    _, target = sample(size=24)
    logits = torch.randn_like(target, requires_grad=True)
    coarse = [torch.randn_like(target, requires_grad=True) for _ in range(4)]
    routes = [torch.randn(2, 16, 9, requires_grad=True) for _ in range(4)]
    output = dict(logits=logits, coarse=coarse, routing=routes)
    control, _ = objective_from_output(output, target, 'region', 'control', 'organ_exposure_fg_bg', weights())
    routed, terms = objective_from_output(output, target, 'region', 'routing', 'organ_exposure_fg_bg', weights())
    expected = REGION_ROUTING_WEIGHT * positive_reverse_kl(routes[-1], target)
    torch.testing.assert_close(routed - control, expected, rtol=1e-5, atol=1e-6)
    gradient = torch.autograd.grad(routed - control, routes, allow_unused=True, retain_graph=True)
    assert all(g is None for g in gradient[:-1])
    assert gradient[-1].abs().sum() > 0
    routed.backward()
    assert logits.grad[target < 0].count_nonzero() == 0
    for c in coarse:
        assert c.grad[target < 0].count_nonzero() == 0
    assert terms['routing'] > 0


@pytest.mark.parametrize('arm', ['experts0', 'experts1'])
def test_q07_zero_and_one_expert_have_no_balance_regularization(arm):
    image, target = sample()
    model = create_model('task_id', base=4, arm=arm)
    output = fair.forward_model(model, 'task_id', image, target)
    balance = torch.tensor(42., requires_grad=True)
    output['balance'] = balance
    actual, terms = objective_from_output(output, target, 'task_id', arm, 'organ_exposure_fg_bg', weights())
    expected = weighted_loss(output['logits'], target, 'organ_exposure_fg_bg', weights())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert terms['balance'] == terms['coarse'] == terms['routing'] == 0
    actual.backward()
    assert balance.grad is None


@pytest.mark.parametrize('scores,target', [
    (torch.zeros(1, 16, 8), torch.ones(1, 16, 2, 2)),
    (torch.zeros(1, 15, 9), torch.ones(1, 16, 2, 2)),
    (torch.zeros(1, 16, 9), torch.ones(1, 16, 2)),
])
def test_wrong_route_contract_rejected(scores, target):
    with pytest.raises(ValueError):
        positive_reverse_kl(scores, target)


def test_nonfinite_route_is_rejected():
    with pytest.raises(FloatingPointError):
        positive_reverse_kl(torch.full((1, 2, 9), float('nan')), torch.ones(1, 2, 2, 2))


@pytest.mark.parametrize('kind,arm', [('task_id', 'routing'), ('region', 'experts0')])
def test_wrong_objective_pair_is_rejected(kind, arm):
    with pytest.raises(ValueError):
        objective_from_output({'logits': torch.zeros(1, 16, 2, 2)}, torch.zeros(1, 16, 2, 2),
                              kind, arm, 'image_mean', None)
