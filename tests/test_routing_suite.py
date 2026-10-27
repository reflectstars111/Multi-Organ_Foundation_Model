from types import SimpleNamespace

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe import routing_suite as suite
from unet_moe import routing_objective


def selection_evidence():
    return [dict(kind=kind, seed=seed, arm=arm, guardrail_selected=False,
                 best_abdomen_macro_positive_dice=.5, best_macro_positive_dice=.7)
            for kind in suite.CONFIG['models'] for seed in suite.CONFIG['seeds'] for arm in suite.LOSS_ARMS]


def test_selection_guardrails_take_priority_over_scores_and_fixed_tie_order():
    rows = selection_evidence()
    assert suite.select_main_loss(rows)[0] == 'image_mean'
    for row in rows:
        if row['arm'] == 'organ_exposure':
            row['best_abdomen_macro_positive_dice'] = .9
            row['best_macro_positive_dice'] = .9
        if row['arm'] == 'organ_exposure_fg_bg':
            row['guardrail_selected'] = True
    assert suite.select_main_loss(rows)[0] == 'organ_exposure_fg_bg'
    for row in rows:
        row['guardrail_selected'] = True
    assert suite.select_main_loss(rows)[0] == 'organ_exposure'


def test_selection_macro_is_only_third_key_and_missing_or_nan_fails():
    rows = selection_evidence()
    for row in rows:
        if row['arm'] == 'organ_exposure_fg_bg':
            row['best_macro_positive_dice'] = .8
    assert suite.select_main_loss(rows)[0] == 'organ_exposure_fg_bg'
    with pytest.raises(ValueError, match='eighteen'):
        suite.select_main_loss(rows[:-1])
    rows[0]['best_macro_positive_dice'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        suite.select_main_loss(rows)


def test_effective_batch_accumulation_keeps_short_final_window(monkeypatch):
    torch.manual_seed(4)
    model, reference = torch.nn.Conv2d(1, 2, 1), torch.nn.Conv2d(1, 2, 1)
    reference.load_state_dict(model.state_dict())
    images, targets = torch.randn(6, 1, 4, 4), torch.randint(0, 2, (6, 2, 4, 4)).float()
    targets[1, 1] = -1
    def objective(model, kind, image, target, arm, main_loss_arm, weights):
        loss = fair.common_loss(model(image), target)
        return loss, dict(main=float(loss.detach()), coarse=0., balance=0., total=float(loss.detach()))
    monkeypatch.setattr(routing_objective, 'compute_objective', objective)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    for i in (0, 4):
        ref_optimizer.zero_grad()
        fair.common_loss(reference(images[i:i+4]), targets[i:i+4]).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.)
        ref_optimizer.step()
    args = SimpleNamespace(batch_size=4, device='cpu', arm='control', main_loss_arm='image_mean')
    totals = suite.train_effective_batches(model, 'region', [(images[i:i+2], targets[i:i+2]) for i in range(0, 6, 2)], optimizer, args, {}, 6)
    assert totals['optimizer_steps'] == 2 and totals['sample_count'] == 6
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_worst_batch_is_selected_from_frozen_draws_not_resampled():
    groups = [[{'image': f'{i}'}] * n for i, n in enumerate([1, 1, 5, 5, 1, 1, 1, 1])]
    c = dict(seeds=[42], batch_size=4, micro_batch_size=2)
    draws = {'42': [dict(epoch=1, indices=[0, 1, 2, 3, 4, 5, 6, 7]), dict(epoch=2, indices=[2, 0, 3, 1])]}
    batch = suite.worst_planned_batch(groups, draws, c)
    assert batch['epoch'] == 1 and batch['indices'] == [0, 1, 2, 3]
    assert batch['max_microbatch_queries'] == 10


def tiny_protocol():
    return dict(protocol_sha256='p', initialization_sha256={'region': {'42': 'init'}},
                _selection=dict(main_loss_arm='image_mean', selection_sha256='selected'))


def test_run_identity_binds_main_loss_selection():
    p = tiny_protocol()
    row = dict(protocol_sha256='p', kind='region', seed=42, arm='control',
               initialization_sha256='init', main_loss_arm='image_mean', main_selection_sha256='selected')
    suite.check_identity(row, p, 'region', 42, 'control')
    for patch in ({'main_selection_sha256': 'other'}, {'main_loss_arm': 'organ_exposure'}, {'arm': 'auxiliary'}):
        with pytest.raises(ValueError):
            suite.check_identity({**row, **patch}, p, 'region', 42, 'control')


def test_cpu_preflight_cannot_admit_gpu_training(tmp_path):
    p = tiny_protocol()
    report = dict(protocol_sha256='p', kind='region', arm='control', passed=True,
                  main_selection_sha256='selected', main_loss_arm='image_mean',
                  real_optimizer_updates=2, effective_batch=16, micro_batch=4, size=512,
                  real_query_images=32, strict_model_optimizer_rng_reload=True,
                  identical_reload_prediction=True, gpu_training_admissible=False, required_free_mib=None)
    fair._write_json(tmp_path / 'preflight' / 'region_control.json', report)
    with pytest.raises(ValueError, match='preflight'):
        suite.check_preflight(tmp_path, p, 'region', 'control')


def test_test_barrier_fails_before_decoding_test_records(monkeypatch, tmp_path):
    p = dict(config=dict(models=['region', 'task_id'], seeds=[42, 43, 44], arms=list(suite.ARMS)))
    visited = []
    def verify(root, protocol, draws, kind, seed, arm):
        visited.append((kind, seed, arm))
        raise ValueError('unfinished')
    monkeypatch.setattr(suite, 'validate_run', verify)
    monkeypatch.setattr(fair, 'RegionDataset', lambda *a, **k: pytest.fail('test labels were opened'))
    with pytest.raises(ValueError, match='unfinished'):
        suite.evaluate_test(SimpleNamespace(output=tmp_path), p, [], {})
    assert visited == [('region', 42, 'control')]


def test_paired_comparison_does_not_mix_seeds():
    rows = [dict(kind='region', seed=s, arm=a, test={'best': dict(macro_positive_dice=v)})
            for s, values in ((42, (.2, .3)), (43, (.8, .7)), (44, (.5, .6)))
            for a, v in zip(suite.ARMS, values)]
    result = suite.paired_comparison(rows)['region']['auxiliary_minus_control']['macro_positive_dice']
    assert result['mean'] == pytest.approx(.1 / 3)
    assert [row['seed'] for row in result['pairs']] == [42, 43, 44]


def test_tiny_resume_exactly_matches_uninterrupted(tmp_path, monkeypatch):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, records, size, task_ids):
            self.groups = [[r] for r in records]
        def __len__(self):
            return len(self.groups)
        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index + 3)
            return torch.rand(1, 8, 8, generator=generator), torch.randint(0, 2, (16, 8, 8), generator=generator).float()
    monkeypatch.setattr(fair, 'RegionDataset', Dataset)
    monkeypatch.setattr(fair, 'create_model', lambda *a: torch.nn.Conv2d(1, 16, 1))
    def objective(model, kind, image, target, arm, main_loss_arm, weights):
        loss = fair.common_loss(model(image), target)
        return loss, {'main': float(loss.detach())}
    monkeypatch.setattr(routing_objective, 'compute_objective', objective)
    def evaluate(model, *a, **k):
        value = float(model.weight.detach().mean().sigmoid())
        return dict(macro_positive_dice=value, abdomen_macro_positive_dice=value)
    monkeypatch.setattr(suite, 'evaluate', evaluate)
    monkeypatch.setattr(suite, 'guardrail_passes', lambda *a: False)
    monkeypatch.setattr(suite, 'diagnose_run', lambda model, kind, datasets, protocol, *a:
                        dict(probe_indices=protocol['diagnostic_indices'], probe_images=len(protocol['diagnostic_indices'])))
    c = {**suite.CONFIG, 'epochs': 2, 'seeds': [42], 'models': ['region'], 'arms': ['control'],
         'batch_size': 4, 'micro_batch_size': 2, 'diagnostic_epochs': [0, 1, 2]}
    records = [dict(image=f'{split}/{i}', dataset='BUSI', split=split) for split in ('train', 'val') for i in range(6)]
    dataset = Dataset([r for r in records if r['split'] == 'train'], 8, c['task_ids'])
    draws = {'42': [dict(epoch=e, indices=list(range(6)), sample_sha256=fair._digest([g[0]['image'] for g in dataset.groups]),
                         effective_batch_sizes=[4, 2]) for e in (1, 2)]}
    fair._seed(42)
    initial = suite.state_digest(fair.create_model('region').state_dict())
    p = dict(config=c, protocol_sha256='tiny', diagnostic_indices=[0, 1], guardrail=suite.GUARDRAIL_CONFIG,
             initialization_sha256={'region': {'42': initial}}, _selection=dict(main_loss_arm='image_mean', selection_sha256='s'))
    args = SimpleNamespace(output=tmp_path / 'resumed', kind='region', seed=42, arm='control', device='cpu', resume=False, stop_after_epoch=1)
    suite.run_model(args, p, records, draws, {})
    args.resume, args.stop_after_epoch = True, 0
    suite.run_model(args, p, records, draws, {})
    suite.validate_run(args.output, p, draws, 'region', 42, 'control')
    resumed = torch.load(suite.run_folder(args.output, 'region', 42, 'control') / 'last.pt', weights_only=True)
    args.output, args.resume = tmp_path / 'continuous', False
    suite.run_model(args, p, records, draws, {})
    uninterrupted = torch.load(suite.run_folder(args.output, 'region', 42, 'control') / 'last.pt', weights_only=True)
    assert suite.state_digest(resumed['model']) == suite.state_digest(uninterrupted['model'])
    assert suite.nested_equal(resumed['optimizer'], uninterrupted['optimizer'])
    assert suite.nested_equal(resumed['rng'], uninterrupted['rng'])
    assert [r['train_loss'] for r in resumed['history']] == [r['train_loss'] for r in uninterrupted['history']]


def test_content_verification_defaults_to_train_val_then_explicit_test(tmp_path, monkeypatch):
    records, content = [], {}
    for split in ('train', 'val', 'test'):
        image, mask = str(tmp_path / f'{split}.png'), str(tmp_path / f'{split}_mask.png')
        records.append(dict(split=split, image=image, mask=mask))
        for path in (image, mask):
            from pathlib import Path
            Path(path).write_bytes(b'pixels')
            content[path] = dict(bytes=6, sha256='expected')
    visited = []
    def fingerprint(path):
        visited.append(str(path))
        return 'expected'
    monkeypatch.setattr(suite, 'file_sha', fingerprint)
    suite.verify_content(content, records)
    assert len(visited) == 4 and all(not path.rsplit('/', 1)[-1].startswith('test') for path in visited)
    visited.clear()
    suite.verify_content(content, records, splits=('test',))
    assert len(visited) == 2 and all(path.rsplit('/', 1)[-1].startswith('test') for path in visited)


def test_finalization_is_validation_only_and_immutable_after_creation(tmp_path, monkeypatch):
    from unet_moe import organ_weighted_suite as q01
    root, reference = tmp_path / 'q02', tmp_path / 'q01'
    root.mkdir()
    reference.mkdir()
    original = dict(protocol_version='q01_organ_weighted_v1',
                    config=dict(models=['shared', 'region', 'task_id'], seeds=[42, 43, 44],
                                arms=list(suite.LOSS_ARMS), epochs=80), guardrail=suite.GUARDRAIL_CONFIG)
    original['protocol_sha256'] = fair._digest(original)
    fair._write_json(reference / 'protocol.json', original)
    protocol = dict(protocol_sha256='q02', reference_root=str(reference),
                    reference_protocol_sha256=original['protocol_sha256'], main_loss_selection_rule=suite.SELECTION_RULE)
    monkeypatch.setattr(suite, 'load_protocol', lambda *a, **k: (protocol, [], {}, {}))
    monkeypatch.setattr(suite, 'guardrail_passes', lambda *a: False)
    verified = []
    def verify(*args, require_test=False):
        assert not require_test
        verified.append(args[3:6])
    monkeypatch.setattr(q01, 'validate_run', verify)
    for kind, seed, arm in q01.identities(original):
        folder = q01.run_folder(reference, kind, seed, arm)
        fair._write_history(folder / 'metrics.jsonl', [dict(epoch=e, macro_positive_dice=.5,
                              abdomen_macro_positive_dice=.4) for e in range(1, 81)])
        (folder / 'test_best.json').write_text('MUST NEVER BE PARSED')
    selection = suite.finalize_selection(root)
    assert selection['test_used'] is False and selection['main_loss_arm'] == 'image_mean'
    assert len(verified) == 27
    assert len(selection['scoring_evidence']) == 18
    for path in reference.rglob('metrics.jsonl'):
        path.unlink()
    assert suite.finalize_selection(root) == selection
    assert len(verified) == 27
