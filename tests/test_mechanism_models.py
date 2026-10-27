"""Q06/Q07 isolate the intended mechanism and preserve shared initialization."""
import io

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe import refinement_models as q05
from unet_moe.mechanism_models import create_model, common_state_dict, expert_count_for_arm


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def seeded(kind, arm):
    torch.manual_seed(42)
    model = create_model(kind, base=4, arm=arm)
    return model, torch.get_rng_state().clone()


@pytest.mark.parametrize('arm', ['control', 'routing'])
def test_q06_is_exact_q05_refined_model_with_no_new_router_input(arm):
    model, rng = seeded('region', arm)
    torch.manual_seed(42)
    reference = q05.create_model('region', base=4, arm='refined')
    assert torch.equal(rng, torch.get_rng_state())
    assert type(model) is type(reference)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key
    image = torch.randn(2, 1, 32, 32)
    torch.testing.assert_close(model(image)['logits'], reference(image)['logits'], rtol=0, atol=0)


def test_q07_prunes_after_full_reference_initialization_without_consuming_rng():
    models = {n: seeded('task_id', f'experts{n}') for n in (0, 1, 9)}
    reference = models[9][0].state_dict()
    common = common_state_dict(models[9][0])
    assert common and not any('.router.' in k or '.experts.' in k for k in common)
    for n, (model, rng) in models.items():
        assert torch.equal(rng, models[9][1])
        assert expert_count_for_arm('task_id', f'experts{n}') == n
        assert common_state_dict(model).keys() == common.keys()
        for key, value in common_state_dict(model).items():
            assert torch.equal(value, common[key]), key
        for layer in model.decoders:
            assert len(layer.experts) == n
            assert hasattr(layer, 'film')
            assert hasattr(layer, 'router') is (n > 0)
        if n == 1:
            for key, value in model.state_dict().items():
                expected = reference[key][:1] if '.router.' in key else reference[key]
                assert torch.equal(value, expected), key
    torch.manual_seed(42)
    q05_reference = q05.create_model('task_id', base=4, arm='refined')
    for key, value in q05_reference.state_dict().items():
        assert torch.equal(reference[key], value), key


@pytest.mark.parametrize('n', [0, 1, 9])
def test_actual_query_expansion_backward_and_film_gradients(n):
    model, _ = seeded('task_id', f'experts{n}')
    image = torch.randn(2, 1, 32, 32)
    target = torch.full((2, 16, 32, 32), -1.)
    target[0, 0] = 0
    target[1, -5:] = 0
    target[1, -2, 5:15, 6:19] = 1
    output = fair.forward_model(model, 'task_id', image, target)
    assert output['logits'].shape == target.shape
    assert output['expert_load'].shape == (4, n)
    assert output['logits'][target.eq(-1)].count_nonzero() == 0
    assert torch.isfinite(output['logits']).all()
    fair.common_loss(output['logits'], target).backward()
    for layer in model.decoders:
        assert layer.film.weight.grad is not None
        assert torch.isfinite(layer.film.weight.grad).all()
        assert layer.film.weight.grad.abs().sum() > 0
    if n == 0:
        assert output['balance'].item() == output['router_z'].item() == 0
        assert not any('.experts.' in key or '.router.' in key for key in model.state_dict())
    if n == 1:
        assert output['balance'].item() == 1
        for layer in model.decoders:
            assert layer.router.weight.grad.count_nonzero() == 0


def test_zero_expert_film_is_direct_conditioning_not_old_conditional_decoder():
    model, _ = seeded('task_id', 'experts0')
    with torch.no_grad():
        model.embedding.weight[0, 0] = 1
        model.embedding.weight[1, 0] = -1
        model.head.weight.zero_()
        model.head.weight[0, 0] = 1
        model.head.bias.zero_()
    image = torch.randn(1, 1, 32, 32).expand(2, -1, -1, -1)
    tasks = torch.tensor([0, 1])
    with torch.no_grad():
        before = model(image, tasks)['logits']
        torch.testing.assert_close(before[0], before[1], rtol=0, atol=0)
        model.decoders[-1].film.weight[4, 0] = .2
        after = model(image, tasks)['logits']
    torch.testing.assert_close(after[0] - after[1], torch.full_like(after[0], .4), rtol=1e-5, atol=1e-6)
    assert not any('.condition.' in name for name in model.state_dict())


def test_one_expert_prediction_cannot_depend_on_router_scores():
    model, _ = seeded('task_id', 'experts1')
    image, tasks = torch.randn(2, 1, 32, 32), torch.tensor([0, 13])
    expected = model(image, tasks)['logits']
    with torch.no_grad():
        for layer in model.decoders:
            layer.router.weight.fill_(42)
            layer.router.bias.fill_(-91)
    actual = model(image, tasks)['logits']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('n', [0, 1, 9])
def test_q07_strict_checkpoint_roundtrip_after_real_optimizer_step(n):
    model, _ = seeded('task_id', f'experts{n}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image, tasks = torch.randn(2, 1, 32, 32), torch.tensor([0, 13])
    model(image, tasks)['logits'].square().mean().backward()
    optimizer.step()
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored, _ = seeded('task_id', f'experts{n}')
    restored.load_state_dict(state['model'], strict=True)
    opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    opt.load_state_dict(state['optimizer'])
    torch.testing.assert_close(model(image, tasks)['logits'], restored(image, tasks)['logits'], rtol=0, atol=0)
    assert len(opt.state) == len(optimizer.state)


@pytest.mark.parametrize('kind,arm', [('region', 'experts9'), ('task_id', 'routing'), ('shared', 'control')])
def test_invalid_model_arm_cannot_silently_select_a_different_network(kind, arm):
    with pytest.raises(ValueError):
        create_model(kind, base=4, arm=arm)


@pytest.mark.parametrize('kwargs', [dict(experts=1), dict(experts=0), dict(top_k=2)])
def test_factory_requires_common_nine_expert_top1_initialization(kwargs):
    with pytest.raises(ValueError):
        create_model('task_id', base=4, arm='experts1', **kwargs)
