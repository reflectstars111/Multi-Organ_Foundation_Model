"""Independent Q05 candidates preserve the original model at initialization."""
import io

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe.refinement_models import (
    added_parameter_count,
    added_parameter_names,
    create_model,
    original_state_dict,
)
from unet_moe.region import PAPER_ALIGNED_TASK_IDS
from unet_moe.routing_diagnostics import _router_scores, zero_expert_residual


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def paired_models(kind, base=4):
    torch.manual_seed(734)
    original = create_model(kind, base=base, arm='control')
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(734)
    refined = create_model(kind, base=base, arm='refined')
    assert torch.equal(original_rng, torch.get_rng_state())
    return original, refined


def data(size):
    generator = torch.Generator().manual_seed(159)
    image = torch.randn(2, 1, size, size, generator=generator)
    target = torch.full((2, len(PAPER_ALIGNED_TASK_IDS), size, size), -1.)
    target[0, 0] = 0
    target[0, 0, 5:14, 7:18] = 1
    target[1, -5:] = 0
    target[1, -5, 4:18, 9:21] = 1
    target[1, -3, 16:21, 3:10] = 1
    # Ignored pixels must remain ignored in both objectives.
    target[:, :, :2] = -1
    return image, target


@pytest.mark.parametrize('kind', ['region', 'task_id'])
def test_control_is_exact_original_factory(kind):
    torch.manual_seed(821)
    expected = fair.create_model(kind, base=4)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(821)
    actual = create_model(kind, base=4, arm='control')
    assert type(actual) is type(expected)
    assert torch.equal(rng, torch.get_rng_state())
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name
    assert added_parameter_count(actual) == 0


@pytest.mark.parametrize('kind,expected_extra', [('region', 48960), ('task_id', 23760)])
def test_zero_initialization_preserves_all_original_state_and_rng(kind, expected_extra):
    original, refined = paired_models(kind, base=24)
    projected = original_state_dict(refined)
    assert projected.keys() == original.state_dict().keys()
    for name, tensor in original.state_dict().items():
        assert torch.equal(projected[name], tensor), name
    assert added_parameter_count(refined) == expected_extra
    extra = added_parameter_names(refined)
    assert extra
    parameters = dict(refined.named_parameters())
    for name in extra:
        assert torch.count_nonzero(parameters[name]) == 0
    assert (sum(p.numel() for p in refined.parameters())
            - sum(p.numel() for p in original.parameters())) == expected_extra


@pytest.mark.parametrize('kind', ['region', 'task_id'])
@pytest.mark.parametrize('size', [32, 64])
def test_real_forward_backward_zero_equivalence_and_new_gradients(kind, size):
    original, refined = paired_models(kind)
    image, target = data(size)
    left, right = image.clone().requires_grad_(), image.clone().requires_grad_()
    original_output = fair.forward_model(original, kind, left, target)
    refined_output = fair.forward_model(refined, kind, right, target)
    torch.testing.assert_close(original_output['logits'], refined_output['logits'], rtol=0, atol=0)
    fair.common_loss(original_output['logits'], target).backward()
    fair.common_loss(refined_output['logits'], target).backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
    current = dict(refined.named_parameters())
    for name, parameter in original.named_parameters():
        actual = current[name].grad
        if parameter.grad is None:
            assert actual is None, name
        else:
            torch.testing.assert_close(parameter.grad, actual, rtol=1e-5, atol=1e-7,
                                       msg=lambda msg: f'{name}: {msg}')
    for name in added_parameter_names(refined):
        gradient = current[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name
    layers = refined.moes if kind == 'region' else refined.decoders
    for layer in layers:
        assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                   for p in layer.router.parameters())


@pytest.mark.parametrize('kind', ['region', 'task_id'])
def test_checkpoint_strict_roundtrip_after_nonzero_refinement(kind):
    _, model = paired_models(kind)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in added_parameter_names(model):
            parameters[name].fill_(.025)
    image, target = data(32)
    expected = fair.forward_model(model.eval(), kind, image, target)['logits']
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = create_model(kind, base=4, arm='refined').eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    actual = fair.forward_model(restored, kind, image, target)['logits']
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(original_state_dict(model), strict=True)
    control = create_model(kind, base=4, arm='control')
    with pytest.raises(RuntimeError, match='Unexpected key'):
        control.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize('kind', ['region', 'task_id'])
def test_existing_diagnostic_hooks_include_refinement_and_restore(kind):
    _, model = paired_models(kind)
    image, target = data(32)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in added_parameter_names(model):
            parameters[name].fill_(.03)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with _router_scores(model, kind) as scores:
        expected = fair.forward_model(model, kind, image, target)['logits']
        assert set(scores) == {0, 1, 2, 3}
        for value in scores.values():
            assert value.shape[-1] == 9
            assert torch.isfinite(value).all()
    with zero_expert_residual(model, kind):
        without = fair.forward_model(model, kind, image, target)['logits']
    assert not torch.equal(expected, without)
    after = fair.forward_model(model, kind, image, target)['logits']
    torch.testing.assert_close(expected, after, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_task_film_conditions_predictions_even_with_task_blind_router():
    model = create_model('task_id', base=4, arm='refined')
    with torch.no_grad():
        for decoder in model.decoders:
            decoder.router.weight[:, -32:] = 0
        model.embedding.weight[0, 0] = 1
        model.embedding.weight[1, 0] = -1
        model.head.weight.zero_()
        model.head.weight[0, 0] = 1
        model.head.bias.zero_()
    image = torch.randn(1, 1, 32, 32).expand(2, -1, -1, -1)
    task = torch.tensor([0, 1])
    with torch.no_grad():
        before = model(image, task)['logits']
        torch.testing.assert_close(before[0], before[1], rtol=0, atol=0)
        # Only beta for feature zero depends on embedding coordinate zero.
        model.decoders[-1].film.weight[4, 0] = .2
        after = model(image, task)['logits']
    torch.testing.assert_close(after[0] - after[1], torch.full_like(after[0], .4),
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('bias', [-1000., 1000.])
def test_region_saturated_proposals_have_finite_forward_and_backward(bias):
    model = create_model('region', base=4, arm='refined')
    layer = model.moes[-1]
    with torch.no_grad():
        layer.proposals.weight.zero_()
        layer.proposals.bias.fill_(bias)
        layer.contrast_projection.weight.fill_(.1)
    image = torch.randn(2, 4, 8, 8, requires_grad=True)
    output, _, scores, _ = layer(image)
    assert torch.isfinite(output).all() and torch.isfinite(scores).all()
    output.square().mean().backward()
    assert torch.isfinite(image.grad).all()
    for parameter in layer.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    # Undefined foreground/background contrast has no fabricated projection signal.
    assert torch.count_nonzero(layer.contrast_projection.weight.grad) == 0


def test_region_contrast_is_region_permutation_equivariant_without_ids():
    model = create_model('region', base=4, arm='refined')
    layer = model.moes[-1]
    with torch.no_grad():
        layer.contrast_projection.weight.copy_(torch.eye(4) * .3)
    image = torch.randn(2, 4, 8, 8)
    permutation = torch.arange(len(PAPER_ALIGNED_TASK_IDS) - 1, -1, -1)
    first_output, first_coarse, first_scores, _ = layer(image)
    with torch.no_grad():
        layer.proposals.weight.copy_(layer.proposals.weight[permutation].clone())
        layer.proposals.bias.copy_(layer.proposals.bias[permutation].clone())
    second_output, second_coarse, second_scores, _ = layer(image)
    torch.testing.assert_close(first_coarse[:, permutation], second_coarse, rtol=0, atol=0)
    torch.testing.assert_close(first_scores[:, permutation], second_scores, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(first_output, second_output, rtol=1e-5, atol=1e-6)
    assert not any('identity' in name or 'task' in name for name in layer.state_dict())


def test_region_contrast_matches_visual_formula_and_both_pool_gradients():
    layer = create_model('region', base=4, arm='refined').moes[-1]
    with torch.no_grad():
        layer.contrast_projection.weight.copy_(torch.eye(4) * .4)
    image = torch.randn(2, 4, 8, 8, requires_grad=True)
    masks = layer.proposals(image).sigmoid()
    fg = torch.einsum('brhw,bchw->brc', masks, image) / masks.sum((2, 3))[..., None]
    bg = torch.einsum('brhw,bchw->brc', 1 - masks, image) / (1 - masks).sum((2, 3))[..., None]
    expected = layer.router(fg + layer.contrast_projection(fg - bg))
    actual = layer(image)[2]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    variables = (image, layer.proposals.weight)
    expected_grad = torch.autograd.grad(expected.square().sum(), variables)
    actual_grad = torch.autograd.grad(actual.square().sum(), variables)
    for left, right in zip(expected_grad, actual_grad):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
        assert torch.count_nonzero(right) > 0


def test_region_full_model_equivariant_when_proposals_and_head_are_reordered():
    model = create_model('region', base=4, arm='refined')
    with torch.no_grad():
        for layer in model.moes:
            layer.contrast_projection.weight.copy_(torch.eye(layer.proposals.in_channels) * .2)
    image = torch.randn(1, 1, 32, 32)
    permutation = torch.arange(len(PAPER_ALIGNED_TASK_IDS) - 1, -1, -1)
    with torch.no_grad():
        original = model(image)
        for layer in model.moes:
            layer.proposals.weight.copy_(layer.proposals.weight[permutation].clone())
            layer.proposals.bias.copy_(layer.proposals.bias[permutation].clone())
        model.head.weight.copy_(model.head.weight[permutation].clone())
        model.head.bias.copy_(model.head.bias[permutation].clone())
        reordered = model(image)
    torch.testing.assert_close(original['logits'][:, permutation], reordered['logits'],
                               rtol=1e-4, atol=1e-5)
    for left, right in zip(original['routing'], reordered['routing']):
        torch.testing.assert_close(left[:, permutation], right, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('kwargs', [
    {'kind': 'shared'}, {'kind': 'unknown'}, {'kind': 'region', 'arm': 'semantic'},
    {'kind': 'task_id', 'arm': 'unknown'}, {'kind': 'region', 'experts': 4},
])
def test_factory_rejects_protocol_ambiguity(kwargs):
    with pytest.raises(ValueError):
        create_model(base=4, **kwargs)
