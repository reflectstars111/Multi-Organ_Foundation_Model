"""Contract tests for the independent, paired Region normalization study."""
from types import SimpleNamespace

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe import region_normalization_suite as suite


def test_domain_dataset_uses_metadata_even_when_targets_are_identical(monkeypatch):
    class Base:
        def __init__(self, records, size, task_ids):
            self.groups = [[row] for row in records]
        def __len__(self):
            return len(self.groups)
        def __getitem__(self, index):
            return torch.zeros(1, 4, 4), torch.zeros(16, 4, 4)
    monkeypatch.setattr(fair, 'RegionDataset', Base)
    dataset = suite.DomainDataset([{'dataset': 'BUSI'}, {'dataset': 'AbdomenUS'}], 4, [0])
    assert dataset[0][2] == 0
    assert dataset[1][2] == 8
    assert torch.equal(dataset[0][1], dataset[1][1])
    with pytest.raises(ValueError, match='domain'):
        suite.DomainDataset([{'dataset': 'unknown'}], 4, [0])


def test_accumulation_preserves_effective_batch_scaling_and_domain_order():
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 2, 1)
            self.seen = []
        def forward(self, image, domain_ids):
            self.seen.extend(domain_ids.tolist())
            return {'logits': self.conv(image)}
    torch.manual_seed(7)
    model, reference = Conv(), Conv()
    reference.load_state_dict(model.state_dict())
    images = torch.randn(6, 1, 4, 4)
    targets = torch.randint(0, 2, (6, 2, 4, 4)).float()
    targets[1, 1] = -1
    domains = torch.tensor([0, 8, 0, 8, 2, 2])
    batches = [(images[i:i+2], targets[i:i+2], domains[i:i+2]) for i in range(0, 6, 2)]
    opt = torch.optim.SGD(model.parameters(), lr=.03)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.03)
    for start in (0, 4):
        ref_opt.zero_grad()
        loss = fair.common_loss(reference(images[start:start+4], domains[start:start+4])['logits'],
                                targets[start:start+4])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.)
        ref_opt.step()
    result = suite.train_effective_batches(model, batches, opt,
                                           SimpleNamespace(batch_size=4, device='cpu'), 6)
    assert result['optimizer_steps'] == 2
    assert result['sample_count'] == 6
    assert result['domain_image_counts']['BUSI'] == 2
    assert result['domain_subbatch_sizes']['AbdomenUS'] == {'1': 2}
    assert model.seen == domains.tolist()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_evaluation_preserves_bn_buffers_and_uses_metadata():
    from unet_moe.region import RegionUNet
    model = RegionUNet(base=4, top_k=1, task_ids=[0], normalization='domain_bn')
    model.train()
    image = torch.rand(2, 1, 32, 32)
    domains = torch.tensor([0, 8])
    model(image, domains)
    before = suite.state_digest(dict(model.named_buffers()))
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:16, 8:16] = 1
    result = suite.evaluate(model, [(image, target, domains)], 'cpu', task_ids=[0])
    assert result['per_structure']['breast_lesion']['samples'] == 2
    assert before == suite.state_digest(dict(model.named_buffers()))
    assert model.training
    assert len(result['routing']) == 4
    assert sum(result['routing'][0]['top1_expert_counts']) == 2


def test_all_arms_have_identical_non_normalization_initialization():
    c = {**suite.CONFIG, 'base': 4}
    hashes = []
    for arm in suite.ARMS:
        fair._seed(42)
        model = suite.create_model(c, arm)
        hashes.append(suite.non_norm_digest(model))
    assert len(set(hashes)) == 1


def test_checkpoint_identity_rejects_wrong_norm_or_non_norm_initialization():
    protocol = {'protocol_sha256': 'frozen', 'initialization_sha256': {'domain_bn': {'42': 'init'}},
                'non_norm_initialization_sha256': {'42': 'conv'}}
    row = {'protocol_sha256': 'frozen', 'kind': 'region', 'seed': 42, 'arm': 'domain_bn',
           'initialization_sha256': 'init', 'non_norm_initialization_sha256': 'conv'}
    suite.check_identity(row, protocol, 'region', 42, 'domain_bn')
    for key in ('arm', 'protocol_sha256', 'initialization_sha256', 'non_norm_initialization_sha256'):
        with pytest.raises(ValueError):
            suite.check_identity({**row, key: 'bad'}, protocol, 'region', 42, 'domain_bn')


def test_test_barrier_precedes_opening_test_dataset(monkeypatch, tmp_path):
    protocol = {'config': {'models': ['region'], 'seeds': [42], 'arms': list(suite.ARMS)}}
    visited = []
    def verify(root, protocol, draws, kind, seed, arm):
        visited.append(arm)
        if arm == 'domain_gn':
            raise ValueError('incomplete run')
    def forbidden(*args, **kwargs):
        pytest.fail('Test dataset constructed before training barrier')
    monkeypatch.setattr(suite, 'validate_run', verify)
    monkeypatch.setattr(suite, 'DomainDataset', forbidden)
    with pytest.raises(ValueError, match='incomplete'):
        suite.evaluate_test(SimpleNamespace(output=tmp_path), protocol, [], {})
    assert visited == ['shared_gn', 'domain_gn']


@pytest.mark.parametrize('tail', [
    [], ['--prepare', '--arm', 'shared_bn'], ['--prepare', '--resume'],
    ['--preflight'], ['--preflight', '--arm', 'shared_bn', '--seed', '42'],
    ['--arm', 'domain_bn', '--seed', '42', '--stop-after-epoch', '81'],
    ['--arm', 'domain_bn', '--seed', '42', '--evaluate-test', '--resume'],
    ['--summarize', '--arm', 'shared_gn'], ['--summarize', '--evaluate-test'],
    ['--arm', 'domain_bn', '--seed', '42', '--reference-root', '/tmp/ref'],
    ['--arm', 'shared_gn', '--seed', '42', '--device', 'cuda:wrong'],
])
def test_invalid_cli_is_rejected(tail):
    with pytest.raises(SystemExit):
        suite.parse_args(['--output', '/tmp/experiment', *tail])


def test_paired_results_compare_same_seed_in_requested_direction():
    runs = []
    for seed, baseline in ((42, .4), (43, .6), (44, .5)):
        for arm, delta in zip(suite.ARMS, (0, .01, -.02, .03)):
            runs.append({'kind': 'region', 'seed': seed, 'arm': arm,
                         'test': {'best': {'macro_positive_dice': baseline + delta}}})
    result = suite.paired_comparison(runs)
    assert result['domain_bn_minus_shared_bn']['macro_positive_dice']['mean'] == pytest.approx(.05)
    assert result['domain_gn_minus_shared_gn']['macro_positive_dice']['mean'] == pytest.approx(.01)


def test_union_evaluation_passes_domain_names_without_using_target_channels():
    class Union(torch.utils.data.Dataset):
        groups = [[{'image': 'image_busi'}], [{'image': 'image_abdomen'}]]
        def __len__(self):
            return 2
        def __getitem__(self, index):
            target = torch.zeros(1, 8, 8)
            target[:, 2:6, 2:6] = 1
            return (torch.ones(1, 8, 8), target, ['BUSI', 'AbdomenUS'][index],
                    ['', 'ct3'][index], target.clone())
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []
        def forward(self, image, domain_ids):
            self.seen.extend(domain_ids.tolist())
            return {'logits': image}
    model = Model()
    result = suite.evaluate_union(model, Union(), 'cpu', {**suite.CONFIG, 'micro_batch_size': 2}, 42)
    assert model.seen == [0, 8]
    assert result['metrics']['per_domain']['BUSI']['samples'] == 1
    assert result['metrics']['abdomen_by_ct']['ct3']['samples'] == 1
    assert result['normalization_buffers_unchanged']


def test_frozen_protocol_detects_artifact_and_content_tampering(tmp_path, monkeypatch):
    source = {'unet_moe/region.py': 'code'}
    monkeypatch.setattr(suite, 'source_manifest', lambda root: source)
    data_path = tmp_path / 'image.png'
    data_path.write_bytes(b'original image')
    artifacts = {
        'records.json': [], 'sampling_plan.json': {}, 'train_statistics.json': {},
        'data_content.json': {str(data_path): {'sha256': suite.file_sha(data_path), 'bytes': data_path.stat().st_size}}}
    protocol = {'protocol_version': suite.VERSION, 'config': dict(suite.CONFIG),
                'config_sha256': fair._digest(suite.CONFIG), 'source_files': source,
                'source_sha256': fair._digest(source)}
    fields = {'records.json': 'record_sha256', 'sampling_plan.json': 'draws_sha256',
              'train_statistics.json': 'statistics_sha256', 'data_content.json': 'content_sha256'}
    for name, artifact in artifacts.items():
        fair._write_json(tmp_path / name, artifact)
        protocol[fields[name]] = fair._digest(artifact)
    protocol['protocol_sha256'] = fair._digest(protocol)
    fair._write_json(tmp_path / 'protocol.json', protocol)
    loaded, _, _, _ = suite.load_protocol(tmp_path)
    assert loaded == protocol
    data_path.write_bytes(b'tampered image')
    with pytest.raises(ValueError, match='Data content'):
        suite.load_protocol(tmp_path)
    fair._write_json(tmp_path / 'records.json', [{'changed': True}])
    with pytest.raises(ValueError, match='records.json'):
        suite.load_protocol(tmp_path, check_content=False)


def test_manifest_snapshot_scripts_override_checkout_regular_package(tmp_path):
    """A namespace snapshot can lose to a later checkout regular package."""
    from importlib.machinery import PathFinder
    from pathlib import Path
    import shutil

    source_root = Path(suite.__file__).resolve().parents[1]
    snapshot = tmp_path / 'snapshot'
    for relative in suite.source_manifest(source_root):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    package = PathFinder.find_spec('scripts', [str(snapshot), str(source_root)])
    helper = PathFinder.find_spec('scripts.run_organ_weighted', package.submodule_search_locations)
    assert Path(helper.origin) == snapshot / 'scripts' / 'run_organ_weighted.py'


def test_tiny_bn_resume_matches_continuous_training(tmp_path, monkeypatch):
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, records, size, task_ids):
            self.groups = [[row] for row in records]
            self.task_ids = task_ids
        def __len__(self):
            return len(self.groups)
        def __getitem__(self, index):
            image = torch.rand(1, 32, 32, generator=torch.Generator().manual_seed(index + 9))
            target = torch.full((16, 32, 32), -1.)
            channels = list(range(11, 16)) if self.groups[index][0]['dataset'] == 'AbdomenUS' else [0]
            target[channels] = 0
            target[channels, 8:16, 8:16] = 1
            return image, target
    monkeypatch.setattr(fair, 'RegionDataset', TinyDataset)
    c = {**suite.CONFIG, 'base': 4, 'size': 32, 'epochs': 2, 'seeds': [42],
         'arms': ['domain_bn'], 'batch_size': 4, 'micro_batch_size': 2, 'quota_per_domain': 1}
    records = [{'split': split, 'dataset': domain, 'image': f'{split}/{domain}',
                'group': 'AbdomenUS/ct3' if domain == 'AbdomenUS' else domain}
               for split in ('train', 'val') for domain in fair.DOMAINS]
    dataset = suite.DomainDataset([r for r in records if r['split'] == 'train'], 32, c['task_ids'])
    draws = suite.make_draw_plan(dataset.groups, [42], 2, 1, 4)
    fair._seed(42)
    model = suite.create_model(c, 'domain_bn')
    protocol = {'config': c, 'protocol_sha256': 'tiny', 'guardrail': suite.GUARDRAIL_CONFIG,
                'initialization_sha256': {'domain_bn': {'42': suite.state_digest(model.state_dict())}},
                'non_norm_initialization_sha256': {'42': suite.non_norm_digest(model)}}
    args = SimpleNamespace(output=tmp_path / 'resume', kind='region', seed=42, arm='domain_bn',
                           device='cpu', resume=False, stop_after_epoch=1)
    suite.run_model(args, protocol, records, draws)
    args.resume, args.stop_after_epoch = True, 0
    suite.run_model(args, protocol, records, draws)
    suite.validate_run(args.output, protocol, draws, 'region', 42, 'domain_bn')
    resumed = torch.load(suite.run_folder(args.output, 'region', 42, 'domain_bn') / 'last.pt', weights_only=True)
    args.output, args.resume = tmp_path / 'continuous', False
    suite.run_model(args, protocol, records, draws)
    continuous = torch.load(suite.run_folder(args.output, 'region', 42, 'domain_bn') / 'last.pt', weights_only=True)
    assert suite.state_digest(resumed['model']) == suite.state_digest(continuous['model'])
    assert [r['train_loss'] for r in resumed['history']] == [r['train_loss'] for r in continuous['history']]
    assert torch.equal(resumed['rng']['torch'], continuous['rng']['torch'])
    assert resumed['normalization_exposure'] == continuous['normalization_exposure']
    folder = suite.run_folder(args.output, 'region', 42, 'domain_bn')
    summary = suite.read_json(folder / 'run_summary.json')
    summary['non_norm_initialization_sha256'] = 'corrupt'
    fair._write_json(folder / 'run_summary.json', summary)
    with pytest.raises(ValueError):
        suite.validate_run(args.output, protocol, draws, 'region', 42, 'domain_bn')
