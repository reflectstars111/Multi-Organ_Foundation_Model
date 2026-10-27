"""Region normalization contracts: backward compatibility and isolated domain state."""
import copy
import io
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from unet_moe.model import block
from unet_moe.region import RegionUNet


ARMS = ('shared_gn', 'domain_gn', 'shared_bn', 'domain_bn')


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def legacy_region():
    """Pre-normalization-experiment topology, independent of Region constructors."""
    model = nn.Module()
    widths = [4 * 2**i for i in range(5)]
    model.encoders = nn.ModuleList([block(1 if i == 0 else widths[i-1], w)
                                   for i, w in enumerate(widths)])
    model.fusions = nn.ModuleList([block(widths[i+1]+widths[i], widths[i])
                                  for i in reversed(range(4))])
    model.moes = nn.ModuleList()
    for width in reversed(widths[:-1]):
        moe = nn.Module()
        moe.proposals = nn.Conv2d(width, 2, 1)
        moe.router = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 9))
        moe.shared = block(width, width)
        moe.experts = nn.ModuleList([block(width, width) for _ in range(9)])
        model.moes.append(moe)
    model.head = nn.Conv2d(4, 2, 1)
    return model


def legacy_forward(model, image):
    features, x = [], image
    for index, encoder in enumerate(model.encoders):
        x = encoder(F.max_pool2d(x, 2) if index else x)
        features.append(x)
    for skip, fusion, moe in zip(reversed(features[:-1]), model.fusions, model.moes):
        x = fusion(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                                           align_corners=False), skip], 1))
        masks = moe.proposals(x).sigmoid()
        pooled = torch.einsum('brhw,bchw->brc', masks, x) / masks.sum((2, 3)).clamp_min(1e-6)[..., None]
        probs = moe.router(pooled).float().softmax(-1)
        values, indices = probs.topk(1, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, indices, values)
        weights = torch.einsum('brhw,bre->behw', masks, gates.to(x.dtype))
        weights = weights / masks.sum(1, keepdim=True).clamp_min(1e-6)
        output = moe.shared(x)
        for index, expert in enumerate(moe.experts):
            batch = torch.where((indices == index).any((1, 2)))[0]
            if batch.numel():
                output = output.index_add(0, batch, expert(x[batch]) * weights[batch, index:index+1])
        x = output
    return model.head(x)


def test_default_keeps_legacy_initialization_forward_and_strict_checkpoints():
    torch.manual_seed(21)
    old = legacy_region()
    torch.manual_seed(21)
    new = RegionUNet(base=4, top_k=1, task_ids=[0, 1], normalization='shared_gn')
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    new.load_state_dict(old.state_dict(), strict=True)
    old.load_state_dict(new.state_dict(), strict=True)
    image = torch.randn(2, 1, 32, 32)
    assert torch.equal(legacy_forward(old, image), new(image)['logits'])
    assert torch.equal(new(image)['logits'], new(image, torch.tensor([0, 8]))['logits'])


@pytest.mark.parametrize('arm', ARMS)
def test_non_normalization_initialization_remains_paired_and_experts_keep_gn(arm):
    torch.manual_seed(31)
    reference = legacy_region()
    torch.manual_seed(31)
    model = RegionUNet(base=4, top_k=1, task_ids=[0, 1], normalization=arm)
    actual = model.state_dict()
    for name, module in reference.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            for parameter in ('weight', 'bias'):
                key = f'{name}.{parameter}'
                assert torch.equal(actual[key], reference.state_dict()[key]), key
    for moe in model.moes:
        for expert in moe.experts:
            assert isinstance(expert[1], nn.GroupNorm)
            assert isinstance(expert[4], nn.GroupNorm)
            assert not any(isinstance(part, nn.BatchNorm2d) for part in expert.modules())


def test_domain_gn_matches_shared_gn_when_affines_match():
    torch.manual_seed(73)
    shared = RegionUNet(base=4, top_k=1, task_ids=[0, 1], normalization='shared_gn')
    torch.manual_seed(73)
    domain = RegionUNet(base=4, top_k=1, task_ids=[0, 1], normalization='domain_gn')
    image = torch.randn(3, 1, 32, 32)
    torch.testing.assert_close(domain(image, torch.tensor([8, 0, 8]))['logits'],
                               shared(image)['logits'], rtol=2e-5, atol=2e-6)


def test_domain_bn_separates_statistics_restores_batch_order_and_reaches_selected_affines():
    from unet_moe.normalization import DomainSpecificNorm2d
    norm = DomainSpecificNorm2d(4, num_domains=9, kind='batch', momentum=1.)
    x = torch.arange(64, dtype=torch.float32).reshape(4, 4, 2, 2).requires_grad_()
    ids = torch.tensor([8, 1, 8, 1])
    output = norm(x, ids)
    # Independent PyTorch layers define expected per-domain statistics/order.
    expected = torch.empty_like(output)
    for domain in (1, 8):
        mask = ids == domain
        independent = nn.BatchNorm2d(4, momentum=1.)
        expected[mask] = independent(x[mask])
        torch.testing.assert_close(norm.norms[domain].running_mean, independent.running_mean)
        torch.testing.assert_close(norm.norms[domain].running_var, independent.running_var)
    torch.testing.assert_close(output, expected)
    output.mul(torch.arange(4.).view(4, 1, 1, 1)).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for domain, branch in enumerate(norm.norms):
        assert int(branch.num_batches_tracked) == int(domain in (1, 8))
        if domain in (1, 8):
            assert branch.weight.grad is not None and branch.weight.grad.abs().sum() > 0
        else:
            assert branch.weight.grad is None
            assert torch.equal(branch.running_mean, torch.zeros(4))
            assert torch.equal(branch.running_var, torch.ones(4))


def test_domain_bn_eval_is_batch_independent_and_checkpoint_restores_buffers():
    from unet_moe.normalization import DomainSpecificNorm2d
    norm = DomainSpecificNorm2d(4, num_domains=9, kind='batch')
    image = torch.randn(4, 4, 3, 3)
    ids = torch.tensor([8, 2, 8, 2])
    norm(image, ids)
    norm.eval()
    expected = norm(image, ids)
    before = copy.deepcopy(norm.state_dict())
    for index in range(4):
        torch.testing.assert_close(norm(image[index:index+1], ids[index:index+1]),
                                   expected[index:index+1], rtol=0, atol=0)
    for name, value in before.items():
        assert torch.equal(value, norm.state_dict()[name]), name
    buffer = io.BytesIO()
    torch.save(norm.state_dict(), buffer)
    buffer.seek(0)
    restored = DomainSpecificNorm2d(4, num_domains=9, kind='batch')
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(image, ids), expected, rtol=0, atol=0)


@pytest.mark.parametrize('ids', [None, [0, 1], torch.tensor([0., 1.]),
                               torch.tensor([True, False]), torch.tensor([[0], [1]]),
                               torch.tensor([0]), torch.tensor([-1, 0]), torch.tensor([0, 9])])
def test_domain_model_rejects_missing_or_invalid_ids_before_updating_statistics(ids):
    model = RegionUNet(base=4, task_ids=[0, 1], normalization='domain_bn')
    before = copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError, TypeError)):
        model(torch.randn(2, 1, 32, 32), ids)
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name]), name


@pytest.mark.parametrize('kwargs', [{'normalization': 'bn'}, {'norm_eps': 0},
                                    {'norm_eps': float('nan')}, {'norm_momentum': -1},
                                    {'norm_momentum': 1.1}])
def test_rejects_invalid_normalization_configuration(kwargs):
    with pytest.raises(ValueError):
        RegionUNet(base=4, task_ids=[0, 1], **kwargs)


def test_normalization_diagnostics_are_read_only_and_cover_dense_layers():
    from unet_moe.normalization import normalization_diagnostics
    model = RegionUNet(base=4, task_ids=[0, 1], normalization='domain_bn')
    model(torch.randn(2, 1, 32, 32), torch.tensor([0, 8]))
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    report = normalization_diagnostics(model)
    assert report['normalization'] == 'domain_bn'
    assert len(report['layers']) == 26 * 9
    assert all('experts' not in row['name'] for row in report['layers'])
    assert all(row['num_batches_tracked'] == int(row['domain_index'] in (0, 8))
               for row in report['layers'])
    assert all(row['running_var']['min'] >= 0 for row in report['layers'])
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name]), name


def source_fixture(tmp_path):
    """Real temporary source/data files for provenance checks; no training or test reads."""
    from PIL import Image
    from unet_moe.region import DOMAINS
    source = tmp_path / 'source' / 'unet_moe'
    source.mkdir(parents=True)
    for path in (Path(__file__).resolve().parents[1] / 'unet_moe').glob('*.py'):
        shutil.copyfile(path, source / path.name)
    records = []
    for split in ('train', 'val', 'test'):
        for domain, task in zip(DOMAINS, (0, 1, 2, 3, 6, 8, 9, 10, 13)):
            image = tmp_path / f'{domain}_{split}.png'
            mask = tmp_path / f'{domain}_{split}_mask.png'
            # Source provenance setup uses only metadata for held-out images.
            if split != 'test':
                Image.new('L', (4, 4), 128).save(image)
                Image.new('L', (4, 4), 1).save(mask)
            records.append(dict(dataset=domain, task=task, image=str(image), mask=str(mask),
                                split=split, group=f'{domain}/{split}', class_id=1,
                                label_version='hc18_filled_v1' if domain == 'Fetal_HC' else 'original'))
    positive_image, positive_mask = tmp_path / 'abdomen_positive.png', tmp_path / 'abdomen_positive_mask.png'
    Image.new('L', (4, 4), 128).save(positive_image)
    Image.new('L', (4, 4), 4).save(positive_mask)
    records.append(dict(dataset='AbdomenUS', task=16, image=str(positive_image),
                        mask=str(positive_mask), split='train', group='AbdomenUS/train_positive',
                        class_id=4, label_version='original'))
    return source, records


@pytest.mark.parametrize('accumulated', [False, True])
def test_fair_resume_rejects_changed_normalization_dependency(tmp_path, monkeypatch, accumulated):
    from unet_moe import fair_suite, fair_suite_accum
    suite = fair_suite_accum if accumulated else fair_suite
    source, records = source_fixture(tmp_path)
    monkeypatch.setattr(suite, '__file__', str(source / Path(suite.__file__).name))
    monkeypatch.setattr(fair_suite, 'build_records', lambda *args: (records, {}))
    args = suite.parse_args(['--data-root', str(tmp_path), '--output', str(tmp_path / 'run'),
                             '--size', '32', '--base', '4', '--epochs', '1', '--device', 'cpu'])
    suite._prepare_protocol(args)
    with (source / 'normalization.py').open('a') as stream:
        stream.write('\n# changed dependency after protocol freeze\n')
    args.resume = True
    with pytest.raises(ValueError, match='[Ss]ource'):
        suite._prepare_protocol(args)


def test_parallel_freeze_rejects_changed_normalization_dependency(tmp_path, monkeypatch):
    from scripts import fair_parallel
    source, _ = source_fixture(tmp_path)
    monkeypatch.setattr(fair_parallel, 'CODE_ROOT', source.parent)
    output = tmp_path / 'parallel'
    fair_parallel.prepare_plan(output, tmp_path)
    with (source / 'normalization.py').open('a') as stream:
        stream.write('\n# changed dependency after plan freeze\n')
    with pytest.raises(ValueError, match='frozen plan'):
        fair_parallel.prepare_plan(output, tmp_path)


def test_recovery_rejects_normalization_change_after_fair_freeze(tmp_path, monkeypatch):
    from scripts import fair_parallel
    from unet_moe import abdomen_recovery as recovery
    from unet_moe.region import DOMAINS
    source, records = source_fixture(tmp_path)
    monkeypatch.setattr(fair_parallel, 'CODE_ROOT', source.parent)
    monkeypatch.setattr(recovery, '__file__', str(source / 'abdomen_recovery.py'))
    fair_plan = fair_parallel.build_plan(tmp_path / 'fair', tmp_path)
    protocol = dict(record_sha256='synthetic-records',
                    split_images={split: dict.fromkeys(DOMAINS, 1) for split in ('train', 'val')})
    protocol['split_images']['train']['AbdomenUS'] = 2
    monkeypatch.setattr(recovery, 'completed_fair', lambda *_: (fair_plan, {}))
    monkeypatch.setattr(recovery, 'fair_records_and_checkpoints', lambda *_: (records, protocol, {}))
    recovery.prepare_plan(tmp_path / 'fair', tmp_path / 'recovery')
    with (source / 'normalization.py').open('a') as stream:
        stream.write('\n# changed dependency after source comparison\n')
    with pytest.raises(ValueError, match='source has changed'):
        recovery.prepare_plan(tmp_path / 'fair', tmp_path / 'second_recovery')
