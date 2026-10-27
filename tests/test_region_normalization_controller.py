"""Process-level checks for the deferred Region normalization experiment queue."""
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ARMS = ('shared_gn', 'domain_gn', 'shared_bn', 'domain_bn')
REF_ARMS = ('image_mean', 'organ_exposure', 'organ_exposure_fg_bg')
REF_MODELS = ('shared', 'region', 'task_id')
SEEDS = (42, 43, 44)


@pytest.fixture
def queue():
    return importlib.import_module('scripts.run_region_normalization')


@pytest.fixture
def prepared(tmp_path):
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'source_snapshot' / 'unet_moe').mkdir(parents=True)
    (reference / 'source_snapshot' / 'unet_moe' / 'organ_weighted_suite.py').touch()
    (reference / 'protocol.json').write_text(json.dumps(dict(
        protocol_sha256='b' * 64,
        config=dict(epochs=80, models=REF_MODELS, seeds=SEEDS, arms=REF_ARMS))))
    runs = []
    for seed in SEEDS:
        for kind in REF_MODELS:
            for arm in REF_ARMS:
                run = dict(kind=kind, seed=seed, arm=arm, protocol_sha256='b' * 64,
                           epochs_completed=80, test_evaluated=True, guardrail_epoch=None)
                folder = reference / arm / kind / f'seed_{seed}'
                folder.mkdir(parents=True)
                (folder / 'run_summary.json').write_text(json.dumps(run))
                for name in ('last.pt', 'best.pt', 'metrics.jsonl'):
                    (folder / name).write_text('complete')
                runs.append(run)
    (reference / 'summary.json').write_text(json.dumps(dict(
        protocol_sha256='b' * 64, complete=True, runs=runs)))
    root = tmp_path / 'normalization'
    root.mkdir()
    (root / 'protocol.json').write_text(json.dumps(dict(
        protocol_sha256='a' * 64, reference_root=str(reference),
        reference_protocol_sha256='b' * 64,
        config=dict(epochs=80, models=['region'], seeds=SEEDS, arms=ARMS))))
    return root, reference


@pytest.fixture
def toy_suite(tmp_path):
    path = tmp_path / 'toy_normalization_suite.py'
    path.write_text('''import argparse, json, os, pathlib, sys, time
p = argparse.ArgumentParser()
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--kind')
p.add_argument('--seed', type=int)
p.add_argument('--arm')
p.add_argument('--device')
p.add_argument('--resume', action='store_true')
p.add_argument('--preflight', action='store_true')
p.add_argument('--evaluate-test', action='store_true')
p.add_argument('--verify-run', action='store_true')
p.add_argument('--summarize', action='store_true')
a = p.parse_args()
protocol = json.loads((a.output / 'protocol.json').read_text())
h = protocol['protocol_sha256']
c = protocol['config']
count = len(c['arms']) * len(c['seeds']) * len(c['models'])
if a.summarize:
    runs = [json.loads(x.read_text()) for x in a.output.glob('*/*/seed_*/run_summary.json')]
    assert len(runs) == count and all(x['test_evaluated'] for x in runs)
    assert not (a.output / 'forged_evidence').exists(), 'forged checkpoint evidence'
    (a.output / 'summary.json').write_text(json.dumps(dict(protocol_sha256=h, runs=runs, complete=True)))
    (a.output / 'verified_summary').touch()
    sys.exit(0)
if a.preflight:
    assert a.device == 'cuda:0' and os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1')
    assert pathlib.Path(protocol['reference_root'], 'verified_summary').exists()
    if (a.output / 'fail_preflight').exists():
        sys.exit(8)
    (a.output / 'preflight').mkdir(exist_ok=True)
    (a.output / 'preflight' / f'{a.arm}.json').write_text(json.dumps(dict(
        protocol_sha256=h, kind='region', arm=a.arm, passed=True, required_free_mib=30000)))
    with (a.output / 'preflight_calls.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(arm=a.arm, gpu=os.environ['CUDA_VISIBLE_DEVICES'])) + '\\n')
    sys.exit(0)
d = a.output / a.arm / a.kind / f'seed_{a.seed}'
s = d / 'run_summary.json'
if a.verify_run:
    value = json.loads(s.read_text())
    assert (d / 'evidence').read_text() == 'complete'
    assert value['protocol_sha256'] == h and value['epochs_completed'] == 80
    assert not a.evaluate_test or value['test_evaluated']
    sys.exit(0)
if (a.output / 'fail_first').exists():
    if a.seed == 42 and a.arm == 'shared_gn':
        deadline = time.monotonic() + 5
        while not list(a.output.glob('live_child_*')) and time.monotonic() < deadline:
            time.sleep(.01)
        sys.exit(7)
    (a.output / f'live_child_{os.getpid()}').touch()
    time.sleep(30)
if (a.output / 'slow').exists():
    (a.output / f'live_child_{os.getpid()}').touch()
    time.sleep(30)
if a.evaluate_test:
    runs = [json.loads(x.read_text()) for x in a.output.glob('*/*/seed_*/run_summary.json')]
    assert len(runs) == 12 and all(x['epochs_completed'] == 80 for x in runs), 'early test'
    assert not a.resume
d.mkdir(parents=True, exist_ok=True)
(d / 'evidence').write_text('complete')
s.write_text(json.dumps(dict(protocol_sha256=h, kind=a.kind, seed=a.seed, arm=a.arm,
                             epochs_completed=80, test_evaluated=a.evaluate_test)))
with (a.output / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps(dict(arm=a.arm, seed=a.seed, test=a.evaluate_test,
                                 gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))) + '\\n')
''')
    return path


def controller(queue, prepared, toy_suite, tmp_path, **kwargs):
    root, reference = prepared
    worker = queue.Controller(root, after_root=reference, poll_seconds=.005, **kwargs)
    worker.suite_prefix = [sys.executable, str(toy_suite)]
    worker.reference_prefix = [sys.executable, str(toy_suite)]
    worker.gpu_memory = lambda gpu: 48000
    worker.gpu_lock_path = lambda gpu: tmp_path / f'gpu{gpu}.lock'
    return worker


def test_all_twelve_runs_train_then_test_and_restart_is_idempotent(queue, prepared, toy_suite, tmp_path):
    root, reference = prepared
    worker = controller(queue, prepared, toy_suite, tmp_path)
    worker.run()
    calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 24
    assert not any(x['test'] for x in calls[:12])
    assert all(x['test'] for x in calls[12:])
    assert {x['gpu'] for x in calls} == {'0', '1'}
    assert len((root / 'preflight_calls.jsonl').read_text().splitlines()) == 4
    assert (reference / 'verified_summary').exists()
    status = json.loads((root / 'controller_status.json').read_text())
    assert status['state'] == 'complete'
    assert all(j['train_state'] == j['test_state'] == 'complete' for j in status['jobs'])
    controller(queue, prepared, toy_suite, tmp_path).run()
    assert len((root / 'calls.jsonl').read_text().splitlines()) == 24
    assert len((root / 'preflight_calls.jsonl').read_text().splitlines()) == 4


def test_waiting_reference_does_not_query_or_lock_gpu(queue, prepared, toy_suite, tmp_path, monkeypatch):
    root, reference = prepared
    (reference / 'summary.json').unlink()
    (reference / 'controller_status.json').write_text('{"state": "complete"}')
    worker = controller(queue, prepared, toy_suite, tmp_path)
    def unexpected(*args):
        pytest.fail('GPU touched while waiting for Q01')
    worker.gpu_memory = worker.gpu_lock_path = unexpected
    def stop_wait(*args):
        assert worker.status['state'] == 'waiting_reference'
        raise InterruptedError('test wait stopped')
    monkeypatch.setattr(queue.time, 'sleep', stop_wait)
    with pytest.raises(InterruptedError, match='test wait stopped'):
        worker.run()
    assert not (root / 'calls.jsonl').exists()
    assert not (root / 'preflight_calls.jsonl').exists()


@pytest.mark.parametrize('damage', ['missing_run', 'duplicate', 'untested', 'epoch',
                                  'wrong_hash', 'missing_checkpoint', 'forged_run'])
def test_incomplete_reference_never_ready(queue, prepared, damage):
    root, reference = prepared
    path = reference / 'summary.json'
    data = json.loads(path.read_text())
    if damage == 'missing_run':
        data['runs'].pop()
    elif damage == 'duplicate':
        data['runs'][-1] = dict(data['runs'][0])
    elif damage == 'untested':
        data['runs'][0]['test_evaluated'] = False
    elif damage == 'epoch':
        data['runs'][0]['epochs_completed'] = 79
    elif damage == 'wrong_hash':
        data['runs'][0]['protocol_sha256'] = 'c' * 64
    elif damage == 'missing_checkpoint':
        (reference / REF_ARMS[0] / REF_MODELS[0] / 'seed_42' / 'best.pt').unlink()
    else:
        run_path = reference / REF_ARMS[0] / REF_MODELS[0] / 'seed_42' / 'run_summary.json'
        value = json.loads(run_path.read_text())
        value['epochs_completed'] = 79
        run_path.write_text(json.dumps(value))
    path.write_text(json.dumps(data))
    assert not queue.Controller(root, after_root=reference).reference_ready()


def test_reference_summaries_cannot_forge_checkpoint_evidence(queue, prepared, toy_suite, tmp_path):
    root, reference = prepared
    (reference / 'forged_evidence').touch()
    worker = controller(queue, prepared, toy_suite, tmp_path)
    assert worker.reference_ready()
    with pytest.raises(RuntimeError, match='reference'):
        worker.run()
    assert not (root / 'calls.jsonl').exists()
    assert not (root / 'preflight_calls.jsonl').exists()


@pytest.mark.parametrize('damage', ['failed', 'hash', 'memory', 'arm'])
def test_invalid_cached_preflight_fails_before_training(queue, prepared, toy_suite, tmp_path, damage):
    root, _ = prepared
    folder = root / 'preflight'
    folder.mkdir()
    report = dict(protocol_sha256='a' * 64, kind='region', arm='shared_gn', passed=True,
                  required_free_mib=30000)
    report.update({'failed': dict(passed=False), 'hash': dict(protocol_sha256='c' * 64),
                   'memory': dict(required_free_mib=0), 'arm': dict(arm='domain_bn')}[damage])
    (folder / 'shared_gn.json').write_text(json.dumps(report))
    worker = controller(queue, prepared, toy_suite, tmp_path)
    with pytest.raises(ValueError, match='preflight'):
        worker.run()
    assert not (root / 'calls.jsonl').exists()


def test_cuda_preflight_failure_never_starts_training(queue, prepared, toy_suite, tmp_path):
    root, _ = prepared
    (root / 'fail_preflight').touch()
    worker = controller(queue, prepared, toy_suite, tmp_path)
    with pytest.raises(RuntimeError, match='preflight.*exited 8'):
        worker.run()
    assert not (root / 'calls.jsonl').exists()
    assert json.loads((root / 'controller_status.json').read_text())['state'] == 'failed'
    with queue.exclusive_lock(tmp_path / 'gpu0.lock'):
        pass


def test_forged_own_completed_summary_fails_verification(queue, prepared, toy_suite, tmp_path):
    root, _ = prepared
    folder = root / 'shared_gn' / 'region' / 'seed_42'
    folder.mkdir(parents=True)
    (folder / 'run_summary.json').write_text(json.dumps(dict(
        protocol_sha256='a' * 64, kind='region', seed=42, arm='shared_gn',
        epochs_completed=80, test_evaluated=False)))
    worker = controller(queue, prepared, toy_suite, tmp_path)
    with pytest.raises(RuntimeError, match='verification'):
        worker.run()
    assert not (root / 'calls.jsonl').exists()


def test_failure_stops_owned_peer_and_leaves_external_child_alive(queue, prepared, toy_suite, tmp_path):
    root, _ = prepared
    (root / 'fail_first').touch()
    external = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                start_new_session=True)
    try:
        worker = controller(queue, prepared, toy_suite, tmp_path)
        with pytest.raises(RuntimeError, match='exited 7'):
            worker.run()
        assert external.poll() is None
        children = [int(path.name.rsplit('_', 1)[1]) for path in root.glob('live_child_*')]
        assert children
        for pid in children:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert not worker.active
        status = json.loads((root / 'controller_status.json').read_text())
        assert status['state'] == 'failed'
        assert all(j['test_state'] == 'pending' for j in status['jobs'])
    finally:
        queue.terminate_children([external], timeout=1)


def test_root_and_gpu_locks_remain_held_by_inheriting_child(queue, tmp_path):
    root_path, gpu_path = tmp_path / 'root.lock', tmp_path / 'gpu.lock'
    with queue.exclusive_lock(root_path) as root_lock, queue.exclusive_lock(gpu_path) as gpu_lock:
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 pass_fds=(root_lock.fileno(), gpu_lock.fileno()), start_new_session=True)
    try:
        for path in (root_path, gpu_path):
            with pytest.raises(RuntimeError, match='already'):
                with queue.exclusive_lock(path):
                    pass
    finally:
        queue.terminate_children([child], timeout=1)
    for path in (root_path, gpu_path):
        with queue.exclusive_lock(path):
            pass


def test_root_lock_prevents_duplicate_controller(queue, prepared, toy_suite, tmp_path):
    root, _ = prepared
    with queue.exclusive_lock(root / '.controller.lock'):
        with pytest.raises(RuntimeError, match='already'):
            controller(queue, prepared, toy_suite, tmp_path).run()
    assert not (root / 'preflight_calls.jsonl').exists()


def test_admission_respects_each_arm_memory_and_existing_gpu_lock(queue, prepared, tmp_path):
    root, reference = prepared
    worker = queue.Controller(root, after_root=reference, gpus=[0])
    lock_path = tmp_path / 'gpu.lock'
    worker.gpu_lock_path = lambda gpu: lock_path
    worker.gpu_memory = lambda gpu: 28000
    worker.required_mib.update(shared_gn=29000, domain_gn=27000)
    pending = [worker.jobs[0], worker.jobs[1]]
    admitted = worker.admit(0, pending)
    assert admitted is not None
    job, lock, free = admitted
    try:
        assert job['arm'] == 'domain_gn' and free == 28000
        assert pending[0]['arm'] == 'shared_gn'
    finally:
        lock.close()
    worker.gpu_memory = lambda gpu: 20000
    assert worker.admit(0, [dict(arm='shared_gn')], preflight=True) is None
    def never_query(*args):
        pytest.fail('locked GPU must not be queried/admitted')
    worker.gpu_memory = never_query
    with queue.exclusive_lock(lock_path):
        assert worker.admit(0, pending) is None


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'hash', 'untested', 'epochs'])
def test_final_summary_requires_complete_twelve_run_matrix(queue, prepared, damage):
    root, reference = prepared
    worker = queue.Controller(root, after_root=reference)
    runs = [dict(kind=j['kind'], seed=j['seed'], arm=j['arm'], protocol_sha256='a' * 64,
                 epochs_completed=80, test_evaluated=True) for j in worker.jobs]
    if damage == 'missing':
        runs.pop()
    elif damage == 'duplicate':
        runs[-1] = dict(runs[0])
    else:
        runs[-1].update({'hash': dict(protocol_sha256='b' * 64),
                         'untested': dict(test_evaluated=False),
                         'epochs': dict(epochs_completed=79)}[damage])
    (root / 'summary.json').write_text(json.dumps(dict(
        protocol_sha256='a' * 64, complete=True, runs=runs)))
    with pytest.raises(ValueError, match='summary'):
        worker.validate_summary()


@pytest.mark.parametrize('phase,artifacts', [('train', False), ('train', True), ('test', True)])
def test_launch_arguments_obey_the_real_suite_parser(
        queue, prepared, tmp_path, monkeypatch, phase, artifacts):
    from unet_moe.region_normalization_suite import parse_args

    root, reference = prepared
    worker = queue.Controller(root, after_root=reference, gpus=[0])
    job = worker.jobs[0]
    if artifacts:
        folder = worker.folder(job)
        folder.mkdir(parents=True)
        (folder / 'last.pt').write_text('artifact')
    captured = []
    def fake_popen(command, **kwargs):
        parsed = parse_args(command[len(worker.suite_prefix):])
        captured.append(parsed)
        return SimpleNamespace(pid=999999, poll=lambda: 0)
    monkeypatch.setattr(queue.subprocess, 'Popen', fake_popen)
    with queue.exclusive_lock(root / '.controller.lock') as root_lock:
        worker.root_lock = root_lock
        with queue.exclusive_lock(tmp_path / 'gpu.lock') as gpu_lock:
            try:
                worker.launch(0, job, gpu_lock, 48000, phase)
            finally:
                worker.stop()
    assert len(captured) == 1
    assert captured[0].evaluate_test is (phase == 'test')
    assert captured[0].resume is (phase == 'train' and artifacts)


def test_snapshot_helpers_win_when_pythonpath_already_contains_snapshot(queue, tmp_path):
    checkout = tmp_path / 'mutable_checkout'
    snapshot = tmp_path / 'source_snapshot'
    for root in (checkout, snapshot):
        (root / 'scripts').mkdir(parents=True)
        (root / 'scripts' / '__init__.py').touch()
    (checkout / 'scripts' / 'run_organ_weighted.py').write_text(
        'raise AssertionError("mutable checkout helper was imported")\n')
    for name in ('run_region_normalization.py', 'run_organ_weighted.py'):
        shutil.copyfile(queue.CODE_ROOT / 'scripts' / name, snapshot / 'scripts' / name)
    probe = '''import json, runpy, sys
namespace = runpy.run_path(sys.argv[1], run_name='snapshot_import_test')
print(json.dumps(dict(first_path=sys.path[0], helper=namespace['read_json'].__code__.co_filename)))
'''
    result = subprocess.run(
        [sys.executable, '-c', probe, str(snapshot / 'scripts' / 'run_region_normalization.py')],
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(checkout), str(snapshot)])})
    assert result.returncode == 0, result.stderr
    imported = json.loads(result.stdout)
    assert imported['first_path'] == str(snapshot)
    assert Path(imported['helper']).parent.parent == snapshot


def test_sigterm_stops_both_training_groups_and_releases_locks(queue, prepared, toy_suite, tmp_path):
    root, reference = prepared
    (root / 'slow').touch()
    runner = tmp_path / 'normalization_runner.py'
    runner.write_text('''import pathlib, sys
from scripts.run_region_normalization import Controller
worker = Controller(sys.argv[1], after_root=sys.argv[2], poll_seconds=.005)
worker.suite_prefix = worker.reference_prefix = [sys.executable, sys.argv[3]]
worker.gpu_memory = lambda gpu: 48000
worker.gpu_lock_path = lambda gpu: pathlib.Path(sys.argv[4]) / f'gpu{gpu}.lock'
worker.run()
''')
    log_path = tmp_path / 'controller.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, str(runner), str(root), str(reference),
                                    str(toy_suite), str(tmp_path)], stdout=log, stderr=subprocess.STDOUT,
                                   cwd=queue.CODE_ROOT, start_new_session=True,
                                   env={**os.environ, 'PYTHONPATH': str(queue.CODE_ROOT)})
    try:
        deadline = time.monotonic() + 10
        children = []
        while time.monotonic() < deadline:
            children = [int(path.name.rsplit('_', 1)[1]) for path in root.glob('live_child_*')]
            if len(children) == 2:
                break
            assert process.poll() is None, log_path.read_text()
            time.sleep(.01)
        assert len(children) == 2, log_path.read_text()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) != 0
        status = json.loads((root / 'controller_status.json').read_text())
        assert status['state'] == 'interrupted'
        assert all(job['pid'] is None for job in status['jobs'])
        for pid in children:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        for path in [root / '.controller.lock', tmp_path / 'gpu0.lock', tmp_path / 'gpu1.lock']:
            with queue.exclusive_lock(path):
                pass
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)


@pytest.mark.parametrize('gpus', [[0, 0], [], [-1], [True]])
def test_invalid_gpu_indices_rejected(queue, prepared, gpus):
    root, reference = prepared
    with pytest.raises(ValueError, match='GPU'):
        queue.Controller(root, after_root=reference, gpus=gpus)


@pytest.mark.parametrize('poll', [0, -1, 61, float('inf'), float('nan')])
def test_poll_bound_rejected(queue, prepared, poll):
    root, reference = prepared
    with pytest.raises(ValueError, match='poll'):
        queue.Controller(root, after_root=reference, poll_seconds=poll)


def test_reference_argument_must_match_frozen_protocol(queue, prepared, tmp_path):
    root, _ = prepared
    with pytest.raises(ValueError, match='reference_root'):
        queue.Controller(root, after_root=tmp_path)


@pytest.mark.parametrize('damage', ['hash', 'epochs', 'arms', 'seeds'])
def test_frozen_protocol_rejects_wrong_budget_or_matrix(queue, prepared, damage):
    root, reference = prepared
    path = root / 'protocol.json'
    data = json.loads(path.read_text())
    if damage == 'hash':
        data['protocol_sha256'] = 'x' * 64
    else:
        data['config'][damage] = {'epochs': 79, 'arms': ARMS[:-1], 'seeds': [42]}[damage]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='protocol'):
        queue.Controller(root, after_root=reference)
