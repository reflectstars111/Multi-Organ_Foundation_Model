from types import SimpleNamespace

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe import organ_weighted_suite as suite


def test_draw_plan_is_paired_and_preserves_partial_batch():
    groups = [[{'dataset': d, 'image': f'{d}/{i}'}]
              for d in fair.DOMAINS for i in range(3)]
    plan = suite.make_draw_plan(groups, seeds=[42, 43], epochs=2, quota=2, batch_size=16)
    assert len(plan['42']) == 2
    assert len(plan['42'][0]['indices']) == 18
    assert plan['42'][0]['effective_batch_sizes'] == [16, 2]
    assert plan == suite.make_draw_plan(groups, seeds=[42, 43], epochs=2, quota=2, batch_size=16)
    assert plan['42'][0]['sample_sha256'] != plan['43'][0]['sample_sha256']


def test_model_state_digest_detects_parameter_change():
    model = torch.nn.Linear(2, 1)
    before = suite.state_digest(model.state_dict())
    with torch.no_grad():
        model.weight[0, 0] += .1
    assert suite.state_digest(model.state_dict()) != before


def test_accumulation_matches_two_full_updates_with_short_final_batch(monkeypatch):
    torch.manual_seed(4)
    model = torch.nn.Conv2d(1, 2, 1)
    reference = torch.nn.Conv2d(1, 2, 1)
    reference.load_state_dict(model.state_dict())
    images = torch.randn(6, 1, 4, 4)
    targets = torch.randint(0, 2, (6, 2, 4, 4)).float()
    targets[1, 1] = -1
    loader = [(images[i:i+2], targets[i:i+2]) for i in range(0, 6, 2)]
    args = SimpleNamespace(batch_size=4, device='cpu', arm='image_mean')
    opt = torch.optim.SGD(model.parameters(), lr=.03)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.03)
    monkeypatch.setattr(fair, 'forward_model', lambda model, kind, x, y: {'logits': model(x)})
    for i in (0, 4):
        ref_opt.zero_grad()
        loss = fair.common_loss(reference(images[i:i+4]), targets[i:i+4])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.)
        ref_opt.step()
    totals = suite.train_effective_batches(model, 'shared', loader, opt, args, {}, 6)
    assert totals['optimizer_steps'] == 2
    assert totals['sample_count'] == 6
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_identity_rejects_wrong_arm_protocol_and_initialization():
    protocol = {'protocol_sha256': 'frozen', 'initialization_sha256': {'shared': {'42': 'init'}}}
    checkpoint = {'protocol_sha256': 'frozen', 'kind': 'shared', 'seed': 42,
                  'arm': 'image_mean', 'initialization_sha256': 'init'}
    suite.check_identity(checkpoint, protocol, 'shared', 42, 'image_mean')
    for key, value in [('protocol_sha256', 'changed'), ('arm', 'organ_exposure'),
                       ('initialization_sha256', 'different')]:
        with pytest.raises(ValueError):
            suite.check_identity({**checkpoint, key: value}, protocol, 'shared', 42, 'image_mean')


def test_paired_differences_use_same_seed():
    runs = []
    for seed, a, b, c in [(42, .1, .2, .4), (43, .8, .9, .85), (44, .5, .6, .7)]:
        for arm, score in zip(suite.ARMS, (a, b, c)):
            runs.append({'kind': 'shared', 'seed': seed, 'arm': arm,
                         'test': {'best': {'macro_positive_dice': score}}})
    result = suite.paired_comparison(runs, selection='best')
    ab = result['shared']['organ_exposure_minus_image_mean']['macro_positive_dice']
    assert ab['mean'] == pytest.approx(.1)
    assert ab['sample_std'] == pytest.approx(0, abs=1e-10)
    assert [row['seed'] for row in ab['pairs']] == [42, 43, 44]


def test_history_validation_rejects_corrupt_draw_before_resume():
    protocol = {'protocol_sha256': 'frozen', 'initialization_sha256': {'shared': {'42': 'init'}}}
    row = {'protocol_sha256': 'frozen', 'kind': 'shared', 'seed': 42, 'arm': 'image_mean',
           'initialization_sha256': 'init', 'epoch': 1, 'sample_count': 2,
           'sample_sha256': 'changed', 'optimizer_steps': 1, 'macro_positive_dice': .5}
    draws = {'42': [{'epoch': 1, 'indices': [0, 1], 'sample_sha256': 'original',
                     'effective_batch_sizes': [2]}]}
    with pytest.raises(ValueError, match='paired'):
        suite.check_history([row], protocol, draws, 'shared', 42, 'image_mean')


def test_tiny_training_resume_matches_uninterrupted_and_verifies_artifacts(tmp_path, monkeypatch):
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, records, size, task_ids):
            self.task_ids = task_ids
            self.groups = [[row] for row in records]

        def __len__(self):
            return len(self.groups)

        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index + 100)
            x = torch.rand(1, 32, 32, generator=generator)
            y = torch.full((16, 32, 32), -1.)
            channels = list(range(11, 16)) if self.groups[index][0]['dataset'] == 'AbdomenUS' else [0]
            y[channels] = 0
            if index % 2:
                y[channels, 8:20, 8:20] = 1
            return x, y

    monkeypatch.setattr(fair, 'RegionDataset', TinyDataset)
    c = {**suite.CONFIG, 'base': 4, 'size': 32, 'epochs': 2, 'seeds': [42], 'models': ['shared'],
         'batch_size': 4, 'micro_batch_size': 2, 'quota_per_domain': 1, 'diagnostic_epochs': [0, 1, 2]}
    records = [{'split': split, 'dataset': domain, 'image': f'{split}/{domain}',
                'group': 'AbdomenUS/ct3' if domain == 'AbdomenUS' else domain}
               for split in ('train', 'val') for domain in fair.DOMAINS]
    dataset = TinyDataset([r for r in records if r['split'] == 'train'], 32, c['task_ids'])
    draws = suite.make_draw_plan(dataset.groups, [42], 2, 1, 4)
    fair._seed(42)
    init = suite.state_digest(fair.create_model('shared', 4, 1, 9).state_dict())
    protocol = {'config': c, 'protocol_sha256': 'tiny', 'diagnostic_indices': [0, 1],
                'initialization_sha256': {'shared': {'42': init}}, 'guardrail': suite.GUARDRAIL_CONFIG}
    args = SimpleNamespace(output=tmp_path / 'resumed', kind='shared', seed=42, arm='image_mean',
                           device='cpu', resume=False, stop_after_epoch=1)
    suite.run_model(args, protocol, records, draws, {})
    args.resume, args.stop_after_epoch = True, 0
    resumed = suite.run_model(args, protocol, records, draws, {})
    suite.validate_run(args.output, protocol, draws, 'shared', 42, 'image_mean')
    saved = torch.load(suite.run_folder(args.output, 'shared', 42, 'image_mean') / 'last.pt',
                       weights_only=True)
    args.output, args.resume = tmp_path / 'continuous', False
    uninterrupted = suite.run_model(args, protocol, records, draws, {})
    whole = torch.load(suite.run_folder(args.output, 'shared', 42, 'image_mean') / 'last.pt',
                       weights_only=True)
    assert resumed['best_epoch'] == uninterrupted['best_epoch']
    assert suite.state_digest(saved['model']) == suite.state_digest(whole['model'])
    assert [r['train_loss'] for r in saved['history']] == [r['train_loss'] for r in whole['history']]
    assert torch.equal(saved['rng']['torch'], whole['rng']['torch'])
    path = suite.run_folder(args.output, 'shared', 42, 'image_mean') / 'diagnostics' / 'epoch_000.json'
    diagnostic = suite.read_json(path)
    diagnostic['probe_indices'] = [1, 0]
    fair._write_json(path, diagnostic)
    with pytest.raises(ValueError, match='fixed probe'):
        suite.validate_run(args.output, protocol, draws, 'shared', 42, 'image_mean')


def test_test_mode_enforces_training_barrier_before_opening_test_labels(monkeypatch, tmp_path):
    protocol = {'config': {'models': ['shared'], 'seeds': [42], 'arms': list(suite.ARMS)}}
    visited = []

    def incomplete(root, protocol, draws, kind, seed, arm):
        visited.append(arm)
        if arm == 'organ_exposure':
            raise ValueError('incomplete run')

    def forbidden(*args, **kwargs):
        pytest.fail('Test labels were opened before all training completed')

    monkeypatch.setattr(suite, 'validate_run', incomplete)
    monkeypatch.setattr(fair, 'RegionDataset', forbidden)
    with pytest.raises(ValueError, match='incomplete'):
        suite.evaluate_test(SimpleNamespace(output=tmp_path), protocol, [], {})
    assert visited == ['image_mean', 'organ_exposure']
