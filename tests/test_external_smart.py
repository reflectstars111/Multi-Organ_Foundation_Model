"""QMOS parity, boundary correctness and actual segmentation gradient tests."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from unet_moe.external_baselines.smart import SMARTAdapter, build_model
from unet_moe.external_baselines.vendor.smart.QMOS import (
    SwinTransformerBlock, WindowAttention, build_attention_mask, window_partition,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class Zero(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


@pytest.mark.parametrize("shape", [(8, 12), (16, 16), (14, 14)])
@pytest.mark.parametrize("shift", [0, 3])
def test_window_means_follow_true_spatial_groups_without_padding_or_wrap(shape, shift):
    """Uniform attention has an independently computable spatial average."""
    h, w = shape
    block = SwinTransformerBlock(4, shape, 1, window_size=7, shift_size=shift)
    block.norm1 = nn.Identity()
    block.mlp = Zero()
    with torch.no_grad():
        block.attn.qkv.weight.zero_()
        block.attn.qkv.bias.zero_()
        block.attn.qkv.weight[8:] = torch.eye(4)
        block.attn.proj.weight.copy_(torch.eye(4))
        block.attn.proj.bias.zero_()
        block.attn.relative_position_bias_table.zero_()
    torch.manual_seed(24)
    grid = torch.randn(h, w, 4)
    expected = torch.empty_like(grid)
    for row in range(h):
        for col in range(w):
            coords = [
                grid[r, c] for r in range(h) for c in range(w)
                if (r - shift) // 7 == (row - shift) // 7
                and (c - shift) // 7 == (col - shift) // 7
            ]
            expected[row, col] = grid[row, col] + torch.stack(coords).mean(0)
    actual = block(grid.reshape(1, h * w, 4)).reshape(h, w, 4)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-5)


@pytest.mark.parametrize("resolution", [128, 64, 32, 16])
@pytest.mark.parametrize("shift", [0, 3])
def test_all_512_stage_masks_remove_invalid_keys_without_nan(resolution, shift):
    ws = 7
    padded = ((resolution + ws - 1) // ws) * ws
    validity = torch.zeros(1, padded, padded, 1, dtype=torch.bool)
    validity[:, :resolution, :resolution] = True
    validity = torch.roll(validity, (-shift, -shift), dims=(1, 2))
    validity = window_partition(validity, ws).reshape(-1, ws * ws)
    mask = build_attention_mask(resolution, resolution, ws, shift)
    assert torch.isfinite(mask.softmax(-1)).all()
    valid_query_invalid_key = validity.unsqueeze(2) & ~validity.unsqueeze(1)
    assert torch.isneginf(mask[valid_query_invalid_key]).all()
    # Test the actual attention, including its projection biases.
    attention = WindowAttention(4, (ws, ws), 1)
    original = torch.randn(validity.shape[0], ws * ws, 4)
    perturbed = original.clone()
    perturbed[~validity] = 1000.0
    out = attention(original, mask)
    changed = attention(perturbed, mask)
    assert torch.isfinite(changed).all()
    torch.testing.assert_close(out[validity], changed[validity], atol=0, rtol=0)


def test_preserves_widths_depths_queries_dynamic_heads_and_classifier():
    model = build_model(use_checkpoint=False)
    backbone = model.network.swin_unet
    assert backbone.embed_dim == 96
    assert [len(layer.blocks) for layer in backbone.layers] == [2, 2, 6, 2]
    assert model.network.query_tokens.shape == (1, 16, 384)
    assert len(model.network.q_seg_head.qformer_layers) == 4
    assert model.network.q_seg_head.decoder1[-1].out_channels == 449
    assert model.network.q_seg_head.decoder2[-1].out_channels == 449
    assert all(block.window_size == 7 for layer in backbone.layers for block in layer.blocks)
    assert model.network.q_cls_head.classifier_c2[-1].out_features == 2
    assert model.network.q_cls_head.classifier_c4[-1].out_features == 4
    assert not any(p.requires_grad for p in model.network.q_cls_head.parameters())
    assert model.network.query_tokens.requires_grad


def test_both_segmentation_heads_and_shared_query_receive_gradients():
    # Same released widths/depths; smaller spatial grid makes this a CPU test.
    torch.manual_seed(42)
    model = SMARTAdapter(image_size=32, use_checkpoint=True)
    images = torch.randn(2, 1, 32, 32)
    result = model(images, torch.tensor([0, 8]))
    assert result["logits"].shape == (2, 1, 32, 32)
    assert result["auxiliary_logits"][0].shape == (2, 1, 32, 32)
    target = torch.rand_like(result["logits"])
    loss = nn.functional.binary_cross_entropy_with_logits(result["logits"], target)
    loss = loss + 0.4 * nn.functional.binary_cross_entropy_with_logits(result["auxiliary_logits"][0], target)
    loss.backward()
    for param in [model.network.query_tokens,
                  model.network.q_seg_head.decoder1[-1].weight,
                  model.network.q_seg_head.decoder2[-1].weight,
                  model.network.swin_unet.patch_embed.proj.weight]:
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.network.q_cls_head.parameters())
    assert not model.network.q_cls_head.training


def test_domain_ids_do_not_condition_smart_outputs():
    model = SMARTAdapter(image_size=32, use_checkpoint=False).eval()
    image = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        a = model(image, torch.tensor([0]))
        b = model(image, torch.tensor([8]), compute_classification=True)
    torch.testing.assert_close(a["logits"], b["logits"], atol=0, rtol=0)
    assert b["classification_binary_logits"].shape == (1, 2)
    assert b["classification_four_logits"].shape == (1, 4)


def test_original_224_full_network_numerical_parity():
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "relate_baseline").glob("*/UUSIC 2025 第一名源码解析*/utils/QMOS.py"))
    if not sources:
        pytest.skip("Original independently distributed SMART source is not present")
    package_name = "_q04_original_smart_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(sources[0].parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.QMOS", sources[0])
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    torch.manual_seed(42)
    reference = upstream.QMOS().eval()
    adapted = SMARTAdapter(image_size=224, use_checkpoint=False).eval()
    adapted.network.load_state_dict(reference.state_dict(), strict=True)
    images = torch.rand(1, 1, 224, 224)
    with torch.no_grad():
        original = reference(images)
        result = adapted(images, compute_classification=True)
    torch.testing.assert_close(result["logits"], original[0], atol=0, rtol=0)
    torch.testing.assert_close(result["auxiliary_logits"][0], nn.functional.interpolate(
        original[1], size=(224, 224), mode="bilinear", align_corners=False), atol=0, rtol=0)
    torch.testing.assert_close(result["classification_binary_logits"], original[2], atol=0, rtol=0)
    torch.testing.assert_close(result["classification_four_logits"], original[3], atol=0, rtol=0)
