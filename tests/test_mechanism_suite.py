from types import SimpleNamespace
from pathlib import Path
import json

import pytest
import torch

from unet_moe import fair_suite as fair
from unet_moe import mechanism_suite as suite
from unet_moe import mechanism_objective


def test_effective_batch_accumulation_keeps_short_final_window(monkeypatch):
    torch.manual_seed(4)
    model, reference = torch.nn.Conv2d(1, 2, 1), torch.nn.Conv2d(1, 2, 1)
    reference.load_state_dict(model.state_dict())
    images, targets = torch.randn(6, 1, 4, 4), torch.randint(0, 2, (6, 2, 4, 4)).float()
    targets[1, 1] = -1
    def objective(model, kind, image, target, arm, main_loss_arm, weights):
        loss = fair.common_loss(model(image), target)
        return loss, dict(main=float(loss.detach()), coarse=0., balance=0., total=float(loss.detach()))
    monkeypatch.setattr(mechanism_objective, 'compute_objective', objective)
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
    return dict(protocol_sha256='p', initialization_sha256={'region': {'42': {'control': 'init', 'refined': 'init2'}}},
                _selection=dict(main_loss_arm='image_mean', selection_sha256='selected'), config=suite.stage_config('q06'))

def test_run_identity_binds_main_loss_selection():
    p = tiny_protocol()
    row = dict(protocol_sha256='p', kind='region', seed=42, arm='control',
               initialization_sha256='init', main_loss_arm='image_mean', main_selection_sha256='selected')
    suite.check_identity(row, p, 'region', 42, 'control')
    for patch in ({'main_selection_sha256': 'other'}, {'main_loss_arm': 'organ_exposure'}, {'arm': 'refined'}):
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

def test_all_q07_pairs_keep_single_seed_sd_undefined():
    arms = ['experts0', 'experts1', 'experts9']
    rows = [dict(kind='task_id', seed=42, arm=a, test={'best': dict(macro_positive_dice=v)})
            for a, v in zip(arms, (.2, .3, .4))]
    result = suite.paired_comparison(rows, arms=arms)['task_id']
    assert set(result) == {'experts1_minus_experts0', 'experts9_minus_experts0', 'experts9_minus_experts1'}
    assert result['experts9_minus_experts0']['macro_positive_dice']['mean'] == pytest.approx(.2)
    assert all(r['macro_positive_dice']['sample_std'] is None for r in result.values())


@pytest.mark.parametrize('arm', suite.ARMS)
def test_tiny_resume_exactly_matches_uninterrupted(tmp_path, monkeypatch, arm):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, records, size, task_ids):
            self.groups = [[r] for r in records]
        def __len__(self):
            return len(self.groups)
        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index + 3)
            return torch.rand(1, 8, 8, generator=generator), torch.randint(0, 2, (16, 8, 8), generator=generator).float()
    monkeypatch.setattr(fair, 'RegionDataset', Dataset)
    monkeypatch.setattr(suite, 'create_model', lambda *a, **k: torch.nn.Conv2d(1, 16, 1))
    def objective(model, kind, image, target, arm, main_loss_arm, weights):
        loss = fair.common_loss(model(image), target)
        return loss, {'main': float(loss.detach())}
    monkeypatch.setattr(mechanism_objective, 'compute_objective', objective)
    def evaluate(model, *a, **k):
        value = float(model.weight.detach().mean().sigmoid())
        return dict(macro_positive_dice=value, abdomen_macro_positive_dice=value)
    monkeypatch.setattr(suite, 'evaluate', evaluate)
    monkeypatch.setattr(suite, 'guardrail_passes', lambda *a: False)
    monkeypatch.setattr(suite, 'diagnose_run', lambda model, kind, datasets, protocol, *a:
                        dict(probe_indices=protocol['diagnostic_indices'], probe_images=len(protocol['diagnostic_indices'])))
    c = {**suite.stage_config('q06'), 'epochs': 2, 'seeds': [42], 'models': ['region'], 'arms': [arm],
         'batch_size': 4, 'micro_batch_size': 2, 'diagnostic_epochs': [0, 1, 2]}
    records = [dict(image=f'{split}/{i}', dataset='BUSI', split=split) for split in ('train', 'val') for i in range(6)]
    dataset = Dataset([r for r in records if r['split'] == 'train'], 8, c['task_ids'])
    draws = {'42': [dict(epoch=e, indices=list(range(6)), sample_sha256=fair._digest([g[0]['image'] for g in dataset.groups]),
                         effective_batch_sizes=[4, 2]) for e in (1, 2)]}
    fair._seed(42)
    initial = suite.state_digest(suite.create_model('region').state_dict())
    p = dict(config=c, protocol_sha256='tiny', diagnostic_indices=[0, 1], guardrail=suite.GUARDRAIL_CONFIG,
             initialization_sha256={'region': {'42': {arm: initial}}}, _selection=dict(main_loss_arm='image_mean', selection_sha256='s'))
    args = SimpleNamespace(output=tmp_path / 'resumed', kind='region', seed=42, arm=arm, device='cpu', resume=False, stop_after_epoch=1)
    suite.run_model(args, p, records, draws, {})
    args.resume, args.stop_after_epoch = True, 0
    suite.run_model(args, p, records, draws, {})
    suite.validate_run(args.output, p, draws, 'region', 42, arm)
    resumed = torch.load(suite.run_folder(args.output, 'region', 42, arm) / 'last.pt', weights_only=True)
    args.output, args.resume = tmp_path / 'continuous', False
    suite.run_model(args, p, records, draws, {})
    uninterrupted = torch.load(suite.run_folder(args.output, 'region', 42, arm) / 'last.pt', weights_only=True)
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


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('stage,kind,arms', [
    ('q06', 'region', ['control', 'routing']),
    ('q07', 'task_id', ['experts0', 'experts1', 'experts9'])])
def test_frozen_matrices(stage, kind, arms):
    c = suite.stage_config(stage)
    assert suite.identities({'config': c}) == [(kind, 42, arm) for arm in arms]
    assert c['size'] == 512 and c['epochs'] == 80
    assert c['batch_size'] == 16 and c['micro_batch_size'] == 4
    assert c['workers'] == 0


@pytest.mark.parametrize('stage,kind,arm', [('q06','region','control'), ('q06','region','routing'),
    ('q07','task_id','experts0'), ('q07','task_id','experts1'), ('q07','task_id','experts9')])
def test_real_small_preflight_checks_scoped_optimizer_continuation(tmp_path, monkeypatch, stage, kind, arm):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *a):
            self.groups = [[dict(image=str(i), task=13)] for i in range(4)]
        def __len__(self):
            return 4
        def __getitem__(self, i):
            generator = torch.Generator().manual_seed(i)
            image = torch.randn(1, 16, 16, generator=generator)
            target = torch.full((16, 16, 16), -1.)
            target[11] = (torch.rand(16, 16, generator=generator) > .75).float()
            return image, target
    monkeypatch.setattr(fair, 'RegionDataset', Dataset)
    c = {**suite.stage_config(stage), 'base': 4, 'size': 16, 'batch_size': 4, 'micro_batch_size': 2}
    fair._seed(42)
    digest = suite.state_digest(suite.create_model(kind, base=4, experts=9, top_k=1, arm=arm).state_dict())
    p = dict(config=c, protocol_sha256='tiny', initialization_sha256={kind: {'42': {arm: digest}}},
             _selection=dict(main_loss_arm='image_mean', selection_sha256='selected'))
    draws = {'42': [dict(epoch=1, indices=[0, 1, 2, 3])]}
    before = torch.are_deterministic_algorithms_enabled()
    report = suite.preflight(SimpleNamespace(output=tmp_path, kind=kind, arm=arm, device='cpu'), p, [], draws, {})
    assert report['passed'] and report['identical_resume_update']
    assert report['continuation_verification_updates'] == report['real_optimizer_updates'] == 2
    assert report['update_deterministic_modes'] == [False, False, True, True]
    assert report['update_cudnn_deterministic'] == [True, True, True, True]
    assert report['numerical_scope'] == suite.NUMERICAL_SCOPE
    assert torch.are_deterministic_algorithms_enabled() == before
    assert not report['gpu_training_admissible'] and not (tmp_path / 'preflight').exists()


def test_new_arm_reaches_its_exact_objective(monkeypatch):
    observed = []
    model = torch.nn.Conv2d(1, 1, 1)
    def objective(model, kind, image, target, arm, main_loss_arm, weights):
        observed.append((arm, main_loss_arm))
        loss = model(image).square().mean()
        return loss, {'main': float(loss.detach())}
    monkeypatch.setattr(mechanism_objective, 'compute_objective', objective)
    for arm in suite.ARMS:
        args = SimpleNamespace(device='cpu', batch_size=1, arm=arm, main_loss_arm='organ_exposure_fg_bg')
        suite.train_effective_batches(model, 'task_id', [(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))],
                                      torch.optim.SGD(model.parameters(), lr=.01), args, {}, 1)
    assert observed == [(arm, 'organ_exposure_fg_bg') for arm in suite.ARMS]


def test_q07_releases_evaluation_model_before_benchmark(monkeypatch, tmp_path):
    """A second model must not contaminate the benchmark's GPU allocation."""
    import weakref
    from unet_moe import mechanism_evaluation, mechanism_benchmark
    created = []
    template = torch.nn.Conv2d(1, 16, 1).state_dict()
    def factory(*args, **kwargs):
        model = torch.nn.Conv2d(1, 16, 1)
        created.append(weakref.ref(model))
        return model
    class Dataset:
        def __init__(self, *args):
            self.groups = []
    monkeypatch.setattr(suite, 'create_model', factory)
    monkeypatch.setattr(suite, 'validate_run', lambda *a, **k: {})
    monkeypatch.setattr(suite, 'verify_content', lambda *a, **k: None)
    monkeypatch.setattr(suite, 'check_identity', lambda *a, **k: None)
    monkeypatch.setattr(suite, 'file_sha', lambda *a: 'checkpoint')
    monkeypatch.setattr(suite, 'loader', lambda *a, **k: [])
    monkeypatch.setattr(fair, 'RegionDataset', Dataset)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: dict(model=template, epoch=80))
    monkeypatch.setattr(mechanism_evaluation, 'evaluate',
        lambda *a, **k: dict(rows=[], diagnostics={}, surface=[]))
    observed = []
    def benchmark(*args):
        assert created[0]() is None
        observed.append(args[3:6])
    monkeypatch.setattr(mechanism_benchmark, 'benchmark_run', benchmark)
    p = dict(stage='q07', protocol_sha256='p', config=suite.stage_config('q07'),
             _selection=dict(selection_sha256='s', main_loss_arm='organ_exposure_fg_bg'))
    folder = suite.run_folder(tmp_path, 'task_id', 42, 'experts0')
    fair._write_json(folder / 'run_summary.json', dict(best_epoch=80, guardrail_epoch=None))
    fair._write_json(tmp_path / 'data_content.json', {})
    args = SimpleNamespace(output=tmp_path, kind='task_id', seed=42, arm='experts0', device='cpu')
    result = suite.evaluate_test(args, p, [], {})
    assert result['test_evaluated'] is True
    assert observed == [('task_id', 42, 'experts0')]


def _selection_chain(root):
    """A self-contained Q02 -> Q05 -> new-stage selection proof."""
    ref = root / 'reference_provenance'
    q2 = dict(protocol_version=suite.q02.VERSION, reference_protocol_sha256='q01',
              main_loss_selection_rule=suite.q02.SELECTION_RULE)
    q2['protocol_sha256'] = fair._digest(q2)
    evidence = [dict(kind=kind, seed=seed, arm=arm, guardrail_selected=arm=='organ_exposure_fg_bg',
                    best_abdomen_macro_positive_dice=.5, best_macro_positive_dice=.7)
                for kind in suite.q05.CONFIG['models'] for seed in suite.q05.CONFIG['seeds']
                for arm in suite.LOSS_ARMS]
    selected, candidates = suite.q02.select_main_loss(evidence)
    choice2 = dict(protocol_sha256=q2['protocol_sha256'], reference_protocol_sha256='q01',
        rule=suite.q02.SELECTION_RULE, scoring_evidence=evidence, candidates=candidates,
        main_loss_arm=selected, test_used=False)
    choice2['selection_sha256'] = fair._digest(choice2)
    fair._write_json(ref / 'q02_protocol.json', q2)
    fair._write_json(ref / 'q02_main_loss_selection.json', choice2)
    q5 = dict(protocol_version=suite.q05.VERSION, reference_protocol_sha256=q2['protocol_sha256'],
        reference_selection_sha256=choice2['selection_sha256'],
        reference_protocol_file_sha256=suite.file_sha(ref / 'q02_protocol.json'),
        reference_selection_file_sha256=suite.file_sha(ref / 'q02_main_loss_selection.json'))
    q5['protocol_sha256'] = fair._digest(q5)
    choice5 = dict(protocol_sha256=q5['protocol_sha256'], rule=suite.q05.SELECTION_RULE,
        reference_protocol_sha256=q2['protocol_sha256'], reference_selection_sha256=choice2['selection_sha256'],
        reference_selection_file_sha256=q5['reference_selection_file_sha256'], main_loss_arm=selected, test_used=False)
    choice5['selection_sha256'] = fair._digest(choice5)
    fair._write_json(ref / 'protocol.json', q5)
    fair._write_json(ref / 'main_loss_selection.json', choice5)
    p = dict(protocol_sha256='new-stage', reference_protocol_sha256=q5['protocol_sha256'],
        reference_selection_sha256=choice5['selection_sha256'],
        reference_selection_file_sha256=suite.file_sha(ref / 'main_loss_selection.json'),
        reference_provenance_files={str(path.relative_to(root)): suite.file_sha(path) for path in ref.iterdir()})
    choice = dict(protocol_sha256='new-stage', rule=suite.SELECTION_RULE,
        reference_protocol_sha256=q5['protocol_sha256'], reference_selection_sha256=choice5['selection_sha256'],
        reference_selection_file_sha256=p['reference_selection_file_sha256'], main_loss_arm=selected, test_used=False)
    choice['selection_sha256'] = fair._digest(choice)
    fair._write_json(root / 'main_loss_selection.json', choice)
    return p, choice


def test_selection_copies_q05_and_q02_chain_without_reselection(tmp_path):
    p, choice = _selection_chain(tmp_path)
    assert suite.load_selection(tmp_path, p) == choice
    assert choice['main_loss_arm'] == 'organ_exposure_fg_bg' and choice['test_used'] is False


@pytest.mark.parametrize('field,value', [('test_used', True), ('main_loss_arm', 'image_mean'),
                                       ('reference_selection_sha256', 'other')])
def test_rehashed_selection_cannot_bypass_q05_source_binding(tmp_path, field, value):
    p, choice = _selection_chain(tmp_path)
    choice[field] = value
    choice.pop('selection_sha256')
    choice['selection_sha256'] = fair._digest(choice)
    fair._write_json(tmp_path / 'main_loss_selection.json', choice)
    with pytest.raises(ValueError, match='Main loss'):
        suite.load_selection(tmp_path, p)
