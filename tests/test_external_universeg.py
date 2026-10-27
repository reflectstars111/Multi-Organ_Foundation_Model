import pytest
import torch

from unet_moe.external_baselines.universeg import UniverSegAdapter
from unet_moe.external_baselines.vendor.universeg.model import UniverSeg


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_full_original_width_forward_and_gradients_match_without_checkpoint():
    torch.manual_seed(42)
    direct = UniverSeg(encoder_blocks=[64,64,64,64])
    wrapped = UniverSegAdapter(activation_checkpointing=False)
    wrapped.network.load_state_dict(direct.state_dict(), strict=True)
    query = torch.randn(2, 1, 16, 16)
    support = torch.randn(2, 3, 1, 16, 16)
    mask = torch.randint(0, 2, support.shape).float()
    expected = direct(query, support, mask)
    actual = wrapped(query, support_images=support, support_labels=mask)['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for parameter, reference in zip(wrapped.network.parameters(), direct.parameters(), strict=True):
        assert (parameter.grad is None) == (reference.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0)


def test_checkpoint_preserves_outputs_and_backward_and_support_has_effect():
    torch.manual_seed(43)
    original = UniverSegAdapter(activation_checkpointing=False)
    checkpointed = UniverSegAdapter(activation_checkpointing=True)
    checkpointed.load_state_dict(original.state_dict(), strict=True)
    query = torch.randn(1, 1, 16, 16)
    support = torch.randn(1, 8, 1, 16, 16)
    mask = torch.randint(0, 2, support.shape).float()
    args = dict(support_images=support, support_labels=mask)
    first = original(query, **args)['logits']
    second = checkpointed(query, **args)['logits']
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    first.mean().backward()
    second.mean().backward()
    for a, b in zip(original.parameters(), checkpointed.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
            assert torch.isfinite(b.grad).all()
    changed = original(query, support_images=support, support_labels=1-mask)['logits']
    assert not torch.allclose(first, changed)


def test_unknown_support_labels_are_rejected():
    model = UniverSegAdapter()
    with pytest.raises(ValueError, match='Unknown'):
        model(torch.zeros(1,1,16,16), support_images=torch.zeros(1,8,1,16,16),
              support_labels=torch.full((1,8,1,16,16), -1.0))
