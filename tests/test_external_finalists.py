"""Architecture, conditioning and gradient checks for Q04 finalists.

These use the complete published widths on small CPU images. The separate
Q04 preflight checks the formal 512 grid and effective batch on the real data.
"""
from pathlib import Path
import hashlib
import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from unet_moe.external_baselines.aimagelab import build_model as build_aimagelab
from unet_moe.external_baselines.lanzhou import (
    build_model as build_lanzhou,
    foreground_logit,
)
from unet_moe.external_baselines.vendor.aimagelab.net import FiLM2d
from unet_moe.external_baselines.vendor.aimagelab.utils import pad_to_2d, unpad_2d


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_lanzhou_binary_conversion_preserves_probability_and_ce_gradient():
    torch.manual_seed(1)
    logits = torch.randn(2, 2, 13, 17, requires_grad=True)
    target = torch.randint(0, 2, (2, 13, 17))
    binary = foreground_logit(logits)
    torch.testing.assert_close(binary.sigmoid(), logits.softmax(1)[:, 1:2])
    binary_loss = F.binary_cross_entropy_with_logits(binary[:, 0], target.float())
    ce_loss = F.cross_entropy(logits, target)
    torch.testing.assert_close(binary_loss, ce_loss)
    binary_grad = torch.autograd.grad(binary_loss, logits, retain_graph=True)[0]
    ce_grad = torch.autograd.grad(ce_loss, logits)[0]
    torch.testing.assert_close(binary_grad, ce_grad)


def test_film_domain_condition_is_learnable_and_identity_initialized():
    film = FiLM2d(n_organs=9, in_channels=4)
    x = torch.randn(2, 4, 8, 8)
    domains = torch.tensor([0, 8])
    torch.testing.assert_close(film(x, domains), x)
    # A nonzero learned projection makes domain embeddings affect predictions.
    with torch.no_grad():
        film.mlp[-1].weight.normal_(std=0.1)
    output = film(x, domains)
    output.square().mean().backward()
    assert film.embed.weight.grad[0].abs().sum() > 0
    assert film.embed.weight.grad[8].abs().sum() > 0
    assert film.embed.weight.grad[1:8].abs().sum() == 0


def test_padding_roundtrip_covers_divisible_and_odd_grids():
    for shape in [(64, 64), (63, 71)]:
        x = torch.randn(1, 3, *shape)
        padded, pads = pad_to_2d(x, 32)
        assert padded.shape[-2] % 32 == 0 and padded.shape[-1] % 32 == 0
        torch.testing.assert_close(unpad_2d(padded, pads), x)


def test_aimagelab_complete_fusion_branches_receive_segmentation_gradient():
    torch.manual_seed(2)
    model = build_aimagelab().train()
    net = model.network
    assert net.segm_net.size == 32 and net.segm_net.depth == 5
    assert net.ll1.in_features == 512 and net.ll1.out_features == 2048
    assert net.alpha.requires_grad and net.alpha.item() == 0.0
    assert isinstance(net.segm_net.out_layer.norm, nn.InstanceNorm2d)
    assert isinstance(net.segm_net.out_layer.activ, nn.LeakyReLU)
    assert net.cls_net.use_cbam and not net.cls_net.use_film
    assert all(m.embed.num_embeddings == 9 for m in net.segm_net.modules() if isinstance(m, FiLM2d))
    # With the original zero gate, the first update learns alpha while feature
    # gradients are zero. Probe the feature path after that gate has opened.
    with torch.no_grad():
        net.alpha.fill_(0.25)
    image = torch.rand(1, 1, 64, 64)
    cls_input = []
    handle = net.cls_net.stem[0].register_forward_pre_hook(
        lambda module, args: cls_input.append(args[0].detach())
    )
    result = model(image, torch.tensor([8]))
    handle.remove()
    assert result["logits"].shape == image.shape
    assert result["auxiliary_logits"] == []
    torch.testing.assert_close(cls_input[0], (4 * image - 2).expand(-1, 3, -1, -1))
    target = (image > 0.5).float()
    F.binary_cross_entropy_with_logits(result["logits"], target).backward()
    for parameter in [net.alpha, net.ll1.weight, net.cls_net.stem[0].weight,
                      net.segm_net.encoder["0"].conv_blocks[0].conv.weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_lanzhou_final_architecture_and_all_deep_heads_receive_gradient():
    torch.manual_seed(3)
    model = build_lanzhou().train()
    net = model.network
    assert isinstance(net.feature_enhancer[1], nn.BatchNorm2d)
    assert net.deep_blocks[0].scale3[0].dilation == (2, 2)
    assert net.deep_blocks[0].scale4[0].dilation == (3, 3)
    assert not hasattr(net, "ds_weights") and not hasattr(net, "att1")
    assert len(net.classification_heads) == 7
    image = torch.rand(1, 1, 64, 64)
    result = model(image, torch.tensor([8]))
    assert result["logits"].shape == image.shape
    assert result["auxiliary_logits"] == []
    loss = F.binary_cross_entropy_with_logits(result["logits"], (image > 0.5).float())
    loss.backward()
    for layer in [net.final_conv, net.ds_conv1, net.ds_conv2, net.ds_conv3]:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    # The first 64/4=16 token grid is not divisible by the retained window7.
    # A successful backward here exercises torchvision's dynamic padding path.
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(image, torch.tensor([0]))["logits"],
            model(image, torch.tensor([8]))["logits"],
        )


def test_vendor_lanzhou_is_exact_published_source():
    root = Path(__file__).resolve().parents[1]
    vendor = root / "unet_moe/external_baselines/vendor/lanzhou"
    metadata = json.loads((vendor / "SOURCE.json").read_text())
    assert hashlib.sha256((vendor / "exact_model.py").read_bytes()).hexdigest() == metadata["source_sha256"]


@pytest.mark.parametrize("bad_shape", [(2, 17, 17), (2, 1, 17, 17), (2, 3, 17, 17)])
def test_binary_conversion_rejects_ambiguous_class_axes(bad_shape):
    with pytest.raises(ValueError, match="B,2,H,W"):
        foreground_logit(torch.zeros(bad_shape))
