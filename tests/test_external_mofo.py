"""MOFO invariants: actual stripe masking, continuous prior, exact batch cache."""
import copy

import pytest
import torch
from torch.nn import functional as F

from unet_moe.external_baselines.mofo import MOFOAdapter, prior_encode
from unet_moe.external_baselines.mofo_prior import (
    cached_prior_step, multi_scale_consistency, restore_state, save_state,
)
from unet_moe.external_baselines.vendor.mofo.network import LePEAttention, MOFO, prompt_decoder
from unet_moe.external_baselines.vendor.mofo.prior import pUNet
from unet_moe.fair_suite import common_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('idx', [0, 1])
def test_stripe_attention_masks_padded_keys(idx):
    torch.manual_seed(3)
    attention = LePEAttention(8, resolution=10, idx=idx, split_size=7, num_heads=2)
    with torch.no_grad():
        attention.get_v.weight.zero_()
        attention.get_v.bias.zero_()
    qkv = torch.zeros(3, 1, 100, 8, requires_grad=True)
    # Uniform q,k; real v = 1. Padded keys must not dilute means below one.
    with torch.no_grad():
        qkv[2].fill_(1)
    result = attention(qkv)
    torch.testing.assert_close(result, torch.ones_like(result))
    result.square().sum().backward()
    assert torch.isfinite(qkv.grad).all()
    assert qkv.grad[2].abs().sum() > 0


def test_unpadded_attention_matches_original_operations():
    torch.manual_seed(12)
    layer = LePEAttention(8, resolution=14, idx=0, split_size=7, num_heads=2)
    qkv = torch.randn(3, 2, 196, 8)
    q, k = [layer.im2cswin(x) for x in qkv[:2]]
    v, lepe = layer.get_lepe(qkv[2], layer.get_v)
    original = ((q * layer.scale @ k.transpose(-2, -1)).softmax(-1) @ v) + lepe
    from unet_moe.external_baselines.vendor.mofo.network import windows2img
    original = original.transpose(1, 2).reshape(-1, 14 * 7, 8)
    original = windows2img(original, 14, 7, 14, 14).reshape(2, 196, 8)
    torch.testing.assert_close(layer(qkv), original)


def test_prompt_reduction_uses_true_sixteen_by_sixteen_grid():
    layer = prompt_decoder(64, reduction_factor=3, reduction_channel=256)
    feature = torch.randn(1, 64, 128, 128)
    assert layer.reduction_layer(feature).shape == (1, 256, 256)
    assert layer(feature, torch.randn(1, 9, 256)).shape == (1, 9, 256)


def test_frozen_prior_keeps_continuous_input_gradient():
    torch.manual_seed(7)
    prior = pUNet(1).requires_grad_(False).eval()
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    truth = torch.zeros_like(logits)
    truth[:, :, 8:20, 12:24] = 1
    prediction = prior_encode(prior, logits.sigmoid())
    with torch.no_grad():
        reference = prior_encode(prior, truth)
    loss = torch.linalg.vector_norm(prediction - reference, dim=1).mean()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert all(p.grad is None for p in prior.parameters())


def test_partial_roi_prior_never_supervises_unknown_pixels():
    torch.manual_seed(19)
    # Isolate the adapter's auxiliary objective without allocating an unused
    # CSWin backbone. The actual full network has a separate 512 preflight.
    adapter = MOFOAdapter.__new__(MOFOAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.image_size = 64
    adapter.prior = pUNet(1).requires_grad_(False).eval()
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, :, 48:] = -1
    target[:, :, 12:42, 8:36] = 1
    logits = torch.randn_like(target, requires_grad=True)
    classification = torch.randn(1, 9, requires_grad=True)
    loss = adapter.auxiliary_loss({'logits': logits, 'classification_logits': classification},
                                  target, torch.tensor([8]))
    loss.backward()
    assert logits.grad[:, :, :, 48:].count_nonzero() == 0
    assert logits.grad[:, :, :, :48].abs().sum() > 0
    assert torch.isfinite(logits.grad).all()
    assert all(p.grad is None for p in adapter.prior.parameters())


def test_cached_contrast_matches_full_batch_parameter_update():
    torch.manual_seed(4)
    full = pUNet(1).double()
    cached = copy.deepcopy(full)
    masks = torch.zeros(4, 1, 32, 32, dtype=torch.float64)
    for i in range(4):
        masks[i, :, 4+i:19+i, 7:23] = 1
    domains = torch.tensor([0, 0, 1, 1])
    opt_full = torch.optim.SGD(full.parameters(), lr=.01)
    opt_cached = torch.optim.SGD(cached.parameters(), lr=.01)
    prediction, features, _ = full(masks)
    loss = common_loss(prediction, masks) + .5 * multi_scale_consistency(features, domains)
    loss.backward()
    opt_full.step()
    value = cached_prior_step(cached, opt_cached, masks, domains, micro_batch_size=1)
    assert value['loss'] == pytest.approx(float(loss.detach()), rel=2e-5, abs=2e-5)
    for p, q in zip(full.parameters(), cached.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-8, rtol=1e-6)


def test_prior_checkpoint_restores_optimizer_and_rng(tmp_path):
    model = pUNet(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    protocol = {'epochs': 2}
    path = tmp_path / 'prior.pt'
    save_state(path, model, optimizer, protocol, 1)
    expected = torch.rand(4)
    torch.rand(40)
    assert restore_state(path, model, optimizer, protocol) == 1
    torch.testing.assert_close(torch.rand(4), expected)
    with pytest.raises(ValueError, match='protocol'):
        restore_state(path, model, optimizer, {'epochs': 3})
