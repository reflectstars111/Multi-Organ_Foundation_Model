"""The Q05 recovery overlay must scope verification without changing training.

These tests exercise the real flag-switching wrapper with a CPU-only fake suite:
``cuda:0`` is an admission label only; no CUDA tensor or context is created.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_q05_verified_resume as recovery


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(autouse=True)
def restore_torch_flags(monkeypatch):
    mode = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', recovery.SCOPE['cublas_workspace_config'])
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(mode, warn_only=warn)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


@pytest.fixture
def bound_manifest(tmp_path):
    root = tmp_path / 'q05'
    write_json(root / 'protocol.json', {'frozen': 'protocol'})
    write_json(root / 'main_loss_selection.json', {'frozen': 'selection'})
    payload = dict(schema=recovery.SCHEMA, root=str(root.resolve()),
                   scope=copy.deepcopy(recovery.SCOPE),
                   overlay_sha256=recovery.sha(recovery.__file__),
                   protocol_file_sha256=recovery.sha(root / 'protocol.json'),
                   selection_file_sha256=recovery.sha(root / 'main_loss_selection.json'))
    manifest = dict(payload, manifest_sha256=recovery.digest(payload))
    path = root / 'recovery_manifest.json'
    write_json(path, manifest)
    return root, path, manifest


def fake_suite(root, *, updates=4, failure=None, failure_after=None, result=None):
    calls = []
    args = SimpleNamespace(device='cuda:0', output=root, kind='region', arm='control')
    report_path = root / 'preflight/region_control.json'

    def original_update(*a, **kw):
        calls.append(dict(mode=torch.are_deterministic_algorithms_enabled(),
                          warn=torch.is_deterministic_algorithms_warn_only_enabled(),
                          args=a, kwargs=kw))
        if failure is not None and failure_after == len(calls):
            raise failure
        return dict(result or {'optimizer_steps': 1, 'sample_count': 16})

    suite = SimpleNamespace(torch=torch, train_effective_batches=original_update,
                            fair=SimpleNamespace(_write_json=write_json))

    def original_preflight(*a):
        assert a == (args, 'protocol', 'records', 'draws', 'weights')
        for index in range(updates):
            suite.train_effective_batches('unchanged-model', index=index)
        if failure is not None and failure_after is None:
            raise failure
        report = dict(passed=True, strict_model_optimizer_rng_reload=True,
                      identical_reload_prediction=True, identical_resume_update=True)
        write_json(report_path, report)
        return report

    suite.preflight = original_preflight
    return suite, args, calls, original_update, report_path


def run_preflight(suite, args, manifest):
    return recovery.scoped_preflight(suite, args, 'protocol', 'records', 'draws', 'weights', manifest)


def test_preflight_runs_two_native_then_two_strict_updates_and_retains_exact_proofs(bound_manifest):
    root, _, manifest = bound_manifest
    suite, args, calls, original, report_path = fake_suite(root)
    report = run_preflight(suite, args, manifest)
    assert [call['mode'] for call in calls] == [False, False, True, True]
    assert [call['warn'] for call in calls] == [False] * 4
    assert [call['kwargs'] for call in calls] == [{'index': i} for i in range(4)]
    assert all(call['args'] == ('unchanged-model',) for call in calls)
    assert report['strict_model_optimizer_rng_reload']
    assert report['identical_reload_prediction'] and report['identical_resume_update']
    assert report['recovery_manifest_sha256'] == manifest['manifest_sha256']
    assert report['numerical_scope'] == recovery.SCOPE
    assert recovery.read(report_path) == report
    recovery.validate_scoped_report(report, manifest)
    assert suite.train_effective_batches is original
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_original_exact_equality_failure_is_never_suppressed(bound_manifest):
    root, _, manifest = bound_manifest
    error = ValueError('Resumed optimizer continuation differs')
    suite, args, calls, original, report_path = fake_suite(root, failure=error)
    with pytest.raises(ValueError, match='Resumed optimizer continuation differs') as captured:
        run_preflight(suite, args, manifest)
    assert captured.value is error
    report = recovery.read(report_path)
    assert report['passed'] is False
    assert report['update_deterministic_modes'] == [False, False, True, True]
    assert suite.train_effective_batches is original
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize('updates', [0, 1, 2, 3, 5])
def test_missing_or_extra_updates_fail_and_overwrite_original_passing_report(bound_manifest, updates):
    root, _, manifest = bound_manifest
    suite, args, calls, original, report_path = fake_suite(root, updates=updates)
    with pytest.raises(ValueError, match='preflight update count|Missing native admission'):
        run_preflight(suite, args, manifest)
    assert len(calls) == min(updates, 4)
    assert recovery.read(report_path)['passed'] is False
    assert suite.train_effective_batches is original
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize('result', [
    {'optimizer_steps': 0, 'sample_count': 16},
    {'optimizer_steps': 2, 'sample_count': 16},
    {'optimizer_steps': 1, 'sample_count': 15},
    {'optimizer_steps': 1, 'sample_count': 32},
])
def test_each_call_must_really_perform_one_batch16_update(bound_manifest, result):
    root, _, manifest = bound_manifest
    suite, args, calls, original, report_path = fake_suite(root, result=result)
    with pytest.raises(ValueError, match='one real batch16 update'):
        run_preflight(suite, args, manifest)
    assert len(calls) == 1
    assert recovery.read(report_path)['passed'] is False
    assert suite.train_effective_batches is original


@pytest.mark.parametrize('failure_after', [1, 3, 4])
@pytest.mark.parametrize('error_class', [RuntimeError, KeyboardInterrupt])
def test_even_interrupted_update_restores_function_and_original_algorithm_flags(
        bound_manifest, failure_after, error_class):
    root, _, manifest = bound_manifest
    # warn_only can be enabled even while deterministic algorithms are disabled.
    torch.use_deterministic_algorithms(False, warn_only=True)
    error = error_class('injected update interruption')
    suite, args, _, original, report_path = fake_suite(
        root, failure=error, failure_after=failure_after)
    with pytest.raises(error_class) as captured:
        run_preflight(suite, args, manifest)
    assert captured.value is error
    assert suite.train_effective_batches is original
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert recovery.read(report_path)['passed'] is False


@pytest.mark.parametrize('invalid', ['cpu', 'workspace', 'already_deterministic'])
def test_preconditions_fail_before_original_suite_runs(bound_manifest, monkeypatch, invalid):
    root, _, manifest = bound_manifest
    suite, args, calls, original, _ = fake_suite(root)
    if invalid == 'cpu':
        args.device = 'cpu'
    elif invalid == 'workspace':
        monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    else:
        torch.use_deterministic_algorithms(True)
    with pytest.raises(ValueError):
        run_preflight(suite, args, manifest)
    assert calls == [] and suite.train_effective_batches is original


def test_manifest_accepts_bound_unchanged_files(bound_manifest):
    root, path, manifest = bound_manifest
    assert recovery.validate_manifest(path, root) == manifest


@pytest.mark.parametrize('field', [
    'schema', 'root', 'scope', 'overlay_sha256',
    'protocol_file_sha256', 'selection_file_sha256',
])
def test_even_resigned_manifest_cannot_change_bound_contract(bound_manifest, field):
    root, path, manifest = bound_manifest
    manifest[field] = {'other_scope': True} if field == 'scope' else 'changed'
    payload = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    manifest['manifest_sha256'] = recovery.digest(payload)
    write_json(path, manifest)
    with pytest.raises(ValueError, match='manifest/source/protocol/selection differs'):
        recovery.validate_manifest(path, root)


@pytest.mark.parametrize('filename', ['protocol.json', 'main_loss_selection.json'])
def test_changed_frozen_input_file_invalidates_manifest(bound_manifest, filename):
    root, path, _ = bound_manifest
    with (root / filename).open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError):
        recovery.validate_manifest(path, root)


def test_changed_overlay_source_invalidates_manifest(bound_manifest, monkeypatch, tmp_path):
    root, path, _ = bound_manifest
    other_source = tmp_path / 'changed_overlay.py'
    other_source.write_bytes(Path(recovery.__file__).read_bytes() + b'\n# changed source\n')
    monkeypatch.setattr(recovery, '__file__', str(other_source))
    with pytest.raises(ValueError):
        recovery.validate_manifest(path, root)


def test_unsigned_manifest_edit_is_rejected(bound_manifest):
    root, path, manifest = bound_manifest
    manifest['extra_note'] = 'changed after signing'
    write_json(path, manifest)
    with pytest.raises(ValueError):
        recovery.validate_manifest(path, root)


class FrozenController:
    def __init__(self, root):
        self.root = root
        self.status = {}
        self.commands = []
        self.checked_reports = []

    def command(self, task=None, action=None, device='cuda:0'):
        self.commands.append((task, action, device))
        return ['frozen-worker', repr(task), str(action), device]

    def read_preflight(self, kind, arm):
        self.checked_reports.append((kind, arm))
        return 'validated-by-original-controller'


@pytest.fixture
def controller(bound_manifest):
    root, path, _ = bound_manifest
    frozen = SimpleNamespace(Controller=FrozenController)
    return recovery.controller_type(frozen, path)(root)


def test_controller_replaces_only_preflight_command(controller, bound_manifest):
    root, path, manifest = bound_manifest
    task = ('task_id', 43, 'refined', 'preflight')
    command = controller.command(task, device='cuda:1')
    assert controller.commands == []
    assert command[1] == str(Path(recovery.__file__).resolve())
    assert '--worker' in command
    for option, expected in [('--output', str(root)), ('--manifest', str(path.resolve())),
                             ('--kind', 'task_id'), ('--seed', '43'),
                             ('--arm', 'refined'), ('--device', 'cuda:1')]:
        assert command[command.index(option) + 1] == expected
    assert controller.status['recovery_manifest_sha256'] == manifest['manifest_sha256']


@pytest.mark.parametrize('phase', ['train', 'evaluate', 'verify'])
@pytest.mark.parametrize('explicit_action', [False, True])
def test_training_evaluation_and_verification_keep_original_command(
        controller, phase, explicit_action):
    task = ('region', 42, 'control', 'preflight' if explicit_action else phase)
    action = phase if explicit_action else None
    result = controller.command(task, action=action, device='cpu')
    assert result == ['frozen-worker', repr(task), str(action), 'cpu']
    assert controller.commands == [(task, action, 'cpu')]


def test_action_only_verification_still_uses_frozen_controller(controller):
    assert controller.command(action='verify', device='cpu') == [
        'frozen-worker', 'None', 'verify', 'cpu']


def test_controller_validates_overlay_evidence_before_original_preflight_checks(
        controller, bound_manifest):
    root, _, manifest = bound_manifest
    suite, args, _, _, report_path = fake_suite(root)
    report = run_preflight(suite, args, manifest)
    assert controller.read_preflight('region', 'control') == 'validated-by-original-controller'
    assert controller.checked_reports == [('region', 'control')]
    report['update_deterministic_modes'] = [False] * 4
    write_json(report_path, report)
    with pytest.raises(ValueError, match='scoped determinism evidence'):
        controller.read_preflight('region', 'control')
    assert controller.checked_reports == [('region', 'control')]


@pytest.mark.parametrize('field,value', [
    ('deterministic_algorithms', False), ('warn_only', True),
    ('cudnn_deterministic', False), ('cudnn_benchmark', True),
    ('cublas_workspace_config', ':16:8'),
])
def test_changed_backend_evidence_is_rejected(bound_manifest, field, value):
    root, _, manifest = bound_manifest
    suite, args, _, _, _ = fake_suite(root)
    report = run_preflight(suite, args, manifest)
    report['update_backend_flags'][2][field] = value
    with pytest.raises(ValueError, match='backend flag evidence'):
        recovery.validate_scoped_report(report, manifest)
