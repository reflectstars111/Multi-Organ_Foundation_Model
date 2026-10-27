import pytest
import torch
from torch.nn import functional as F

from unet_moe.empty_loss import empty_top_fraction_loss


def test_only_hardest_valid_empty_pixels_receive_gradients():
    logits = torch.tensor([[[[10., -3.], [-2., 1.]]]], requires_grad=True)
    target = torch.tensor([[[[-1., 0.], [0., 0.]]]])
    loss = empty_top_fraction_loss(logits, target, fraction=.01)
    torch.testing.assert_close(loss, F.softplus(torch.tensor(1.)))
    loss.backward()
    expected = torch.zeros_like(logits)
    expected[0, 0, 1, 1] = torch.sigmoid(torch.tensor(1.))
    torch.testing.assert_close(logits.grad, expected)


def test_positive_missing_and_excluded_channels_get_no_gradient():
    logits = torch.zeros((1, 4, 2, 2), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, 0, 0, 0] = 1
    target[:, 1] = -1
    empty_top_fraction_loss(logits, target, channel_indices=[0, 1, 2]).backward()
    assert logits.grad[:, :2].abs().sum() == 0
    assert logits.grad[:, 2].abs().sum() > 0
    assert logits.grad[:, 3].abs().sum() == 0


def test_top_count_rounds_up_using_valid_area_only():
    logits = torch.arange(12., requires_grad=True).reshape(1, 1, 3, 4)
    target = torch.zeros_like(logits)
    target[0, 0, -1, -1] = -1
    loss = empty_top_fraction_loss(logits, target, fraction=.1)
    torch.testing.assert_close(loss, F.softplus(torch.tensor([9., 10.])).mean())


def test_pair_average_does_not_depend_on_mask_area():
    logits = torch.zeros((2, 1, 2, 2), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 0] = -1
    loss = empty_top_fraction_loss(logits, target, fraction=1)
    torch.testing.assert_close(loss, F.softplus(torch.tensor(0.)))
    loss.backward()
    torch.testing.assert_close(logits.grad[0].sum(), logits.grad[1].sum())


@pytest.mark.parametrize('selection', [None, []])
def test_no_eligible_pairs_is_differentiable_zero(selection):
    logits = torch.randn((2, 3, 2, 2), requires_grad=True)
    target = -torch.ones_like(logits)
    loss = empty_top_fraction_loss(logits, target, channel_indices=selection)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad.abs().sum() == 0


@pytest.mark.parametrize('fraction', [0, -1, 1.01, float('nan')])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        empty_top_fraction_loss(torch.zeros((1, 1, 1, 1)), torch.zeros((1, 1, 1, 1)), fraction)


def test_invalid_channel_selection_is_rejected():
    x = torch.zeros((1, 2, 1, 1))
    for indices in ([0, 0], [2], [-1], [.5]):
        with pytest.raises(ValueError):
            empty_top_fraction_loss(x, x, channel_indices=indices)
