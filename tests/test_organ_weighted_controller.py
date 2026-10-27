"""CPU-only process tests for the Q01 GPU queue and train/test barrier."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def test_controller_entry_point_exists():
    # A missing standalone entry point must not silently use the historical queue.
    assert importlib.util.find_spec('scripts.run_organ_weighted') is not None


@pytest.fixture
def queue():
    return importlib.import_module('scripts.run_organ_weighted')


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / 'q01'
    root.mkdir()
    (root / 'protocol.json').write_text(json.dumps({
        'protocol_sha256': 'a' * 64, 'config': {'epochs': 80}}))
    (root / 'preflight.json').write_text(json.dumps({
        'protocol_sha256': 'a' * 64,
        'models': {kind: {'passed': True, 'required_free_mib': memory}
                   for kind, memory in [('shared', 1000), ('region', 2000), ('task_id', 3000)]}}))
    return root


@pytest.fixture
def toy_suite(tmp_path):
    """Real subprocess fixture; only the expensive GPU training boundary is replaced."""
    path = tmp_path / 'toy_suite.py'
    path.write_text('''import argparse, json, os, pathlib, sys, time
p = argparse.ArgumentParser()
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--kind')
p.add_argument('--seed', type=int)
p.add_argument('--arm')
p.add_argument('--device')
p.add_argument('--resume', action='store_true')
p.add_argument('--evaluate-test', action='store_true')
p.add_argument('--verify-run', action='store_true')
p.add_argument('--summarize', action='store_true')
a = p.parse_args()
if a.summarize:
    runs = [json.loads(x.read_text()) for x in a.output.glob('*/*/seed_*/run_summary.json')]
    assert len(runs) == 27 and all(x['test_evaluated'] for x in runs)
    (a.output / 'summary.json').write_text(json.dumps({'protocol_sha256': 'a' * 64, 'runs': runs, 'complete': True}))
    sys.exit(0)
d = a.output / a.arm / a.kind / f'seed_{a.seed}'
s = d / 'run_summary.json'
if a.verify_run:
    value = json.loads(s.read_text())
    assert (d / 'evidence').read_text() == 'complete'
    assert value['protocol_sha256'] == 'a' * 64
    assert value['epochs_completed'] == 80
    assert not a.evaluate_test or value['test_evaluated']
    sys.exit(0)
if (a.output / 'fail_first').exists() and a.kind == 'shared' and a.seed == 42 and a.arm == 'image_mean':
    sys.exit(7)
if (a.output / 'slow').exists():
    (a.output / f'live_child_{os.getpid()}').touch()
    time.sleep(30)
if a.evaluate_test:
    runs = [json.loads(x.read_text()) for x in a.output.glob('*/*/seed_*/run_summary.json')]
    assert len(runs) == 27 and all(x['epochs_completed'] == 80 for x in runs), 'test gate opened early'
    assert a.resume
d.mkdir(parents=True, exist_ok=True)
time.sleep(.005)
(d / 'evidence').write_text('complete')
s.write_text(json.dumps(dict(protocol_sha256='a' * 64, kind=a.kind, seed=a.seed, arm=a.arm,
                             epochs_completed=80, test_evaluated=a.evaluate_test)))
with (a.output / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps(dict(kind=a.kind, seed=a.seed, arm=a.arm, test=a.evaluate_test,
                                 gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))) + '\\n')
''')
    return path


def controller(queue, prepared, toy_suite, monkeypatch, tmp_path):
    monkeypatch.setattr(queue, 'gpu_free_mib', lambda gpu: 4000)
    monkeypatch.setattr(queue, 'gpu_lock_path', lambda gpu: tmp_path / f'gpu{gpu}.lock')
    monkeypatch.setattr(queue, 'SUITE_PREFIX', [sys.executable, str(toy_suite)])
    return queue.Controller(prepared, gpus=[0, 1], poll_seconds=.005)


def test_all_27_runs_train_before_any_test_and_restart_is_idempotent(
        queue, prepared, toy_suite, monkeypatch, tmp_path):
    worker = controller(queue, prepared, toy_suite, monkeypatch, tmp_path)
    worker.run()
    calls = [json.loads(x) for x in (prepared / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 54
    assert not any(x['test'] for x in calls[:27])
    assert all(x['test'] for x in calls[27:])
    assert {x['gpu'] for x in calls} == {'0', '1'}
    status = json.loads((prepared / 'controller_status.json').read_text())
    assert status['state'] == 'complete'
    assert all(x['train_state'] == x['test_state'] == 'complete' for x in status['jobs'])
    controller(queue, prepared, toy_suite, monkeypatch, tmp_path).run()
    assert len((prepared / 'calls.jsonl').read_text().splitlines()) == 54


def test_forged_summary_cannot_skip_unfinished_training(
        queue, prepared, toy_suite, monkeypatch, tmp_path):
    folder = prepared / 'image_mean' / 'shared' / 'seed_42'
    folder.mkdir(parents=True)
    (folder / 'run_summary.json').write_text(json.dumps(dict(
        protocol_sha256='a' * 64, kind='shared', seed=42, arm='image_mean',
        epochs_completed=80, test_evaluated=False)))
    worker = controller(queue, prepared, toy_suite, monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match='verif'):
        worker.run()
    assert not (prepared / 'calls.jsonl').exists()
    assert json.loads((prepared / 'controller_status.json').read_text())['state'] == 'failed'


@pytest.mark.parametrize('change', ['protocol', 'failed', 'memory'])
def test_invalid_preflight_never_launches(queue, prepared, change):
    path = prepared / 'preflight.json'
    data = json.loads(path.read_text())
    if change == 'protocol':
        data['protocol_sha256'] = 'b' * 64
    elif change == 'failed':
        data['models']['task_id']['passed'] = False
    else:
        data['models']['region']['required_free_mib'] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='preflight'):
        queue.Controller(prepared)
    assert not (prepared / 'controller.pid').exists()


def test_low_memory_skips_ineligible_job_but_keeps_it_pending(queue, prepared, monkeypatch, tmp_path):
    worker = queue.Controller(prepared, gpus=[0])
    monkeypatch.setattr(queue, 'gpu_lock_path', lambda gpu: tmp_path / 'gpu.lock')
    monkeypatch.setattr(queue, 'gpu_free_mib', lambda gpu: 1500)
    pending = [job for job in worker.jobs if job['kind'] in ('region', 'shared')]
    pending.sort(key=lambda job: job['kind'] != 'region')
    admitted = worker.admit(0, pending)
    assert admitted is not None
    job, lock, free = admitted
    try:
        assert job['kind'] == 'shared' and free == 1500
        assert pending[0]['kind'] == 'region'
    finally:
        lock.close()
    monkeypatch.setattr(queue, 'gpu_free_mib', lambda gpu: 500)
    assert worker.admit(0, pending) is None


def test_training_failure_stops_queue_before_test(queue, prepared, toy_suite, monkeypatch, tmp_path):
    (prepared / 'fail_first').touch()
    worker = controller(queue, prepared, toy_suite, monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match='exited 7'):
        worker.run()
    status = json.loads((prepared / 'controller_status.json').read_text())
    assert status['state'] == 'failed'
    assert any(x['train_state'] == 'failed' for x in status['jobs'])
    assert all(x['test_state'] == 'pending' for x in status['jobs'])
    assert not worker.active


def test_root_lock_rejects_second_controller(queue, prepared, toy_suite, monkeypatch, tmp_path):
    worker = controller(queue, prepared, toy_suite, monkeypatch, tmp_path)
    with queue.exclusive_lock(prepared / '.controller.lock'):
        with pytest.raises(RuntimeError, match='already'):
            worker.run()
    assert not (prepared / 'calls.jsonl').exists()


@pytest.mark.parametrize('damage', ['missing_run', 'duplicate_run', 'wrong_protocol', 'untested', 'unfinished'])
def test_final_summary_must_cover_the_frozen_complete_test_matrix(queue, prepared, damage):
    runs = [dict(kind=kind, seed=seed, arm=arm, protocol_sha256='a' * 64,
                 epochs_completed=80, test_evaluated=True)
            for seed in (42, 43, 44) for kind in ('shared', 'region', 'task_id')
            for arm in ('image_mean', 'organ_exposure', 'organ_exposure_fg_bg')]
    summary = dict(protocol_sha256='a' * 64, complete=True, runs=runs)
    if damage == 'missing_run':
        runs.pop()
    elif damage == 'duplicate_run':
        runs[-1] = dict(runs[0])
    elif damage == 'wrong_protocol':
        runs[-1]['protocol_sha256'] = 'b' * 64
    elif damage == 'untested':
        runs[-1]['test_evaluated'] = False
    else:
        summary['complete'] = False
    (prepared / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='summary'):
        queue.Controller(prepared).validate_summary()


def test_gpu_lock_is_held_until_inheriting_child_exits(queue, tmp_path):
    path = tmp_path / 'gpu.lock'
    with queue.exclusive_lock(path) as lock:
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 pass_fds=(lock.fileno(),), start_new_session=True)
    try:
        with pytest.raises(RuntimeError, match='already'):
            with queue.exclusive_lock(path):
                pass
    finally:
        queue.terminate_children([child], timeout=1)
    assert child.poll() is not None
    with queue.exclusive_lock(path):
        pass


def test_sigterm_stops_training_process_groups_and_releases_locks(
        queue, prepared, toy_suite, tmp_path):
    (prepared / 'slow').touch()
    runner = tmp_path / 'controller_runner.py'
    runner.write_text('''import pathlib, sys
from scripts import run_organ_weighted as q
q.SUITE_PREFIX = [sys.executable, sys.argv[2]]
q.gpu_free_mib = lambda gpu: 4000
q.gpu_lock_path = lambda gpu: pathlib.Path(sys.argv[3]) / f'gpu{gpu}.lock'
q.Controller(sys.argv[1], poll_seconds=.01).run()
''')
    log = tmp_path / 'controller.log'
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, str(runner), str(prepared),
                                    str(toy_suite), str(tmp_path)],
                                   cwd=queue.CODE_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONPATH': str(queue.CODE_ROOT)},
                                   start_new_session=True)
    children = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            children = [int(path.name.rsplit('_', 1)[1]) for path in prepared.glob('live_child_*')]
            if len(children) == 2:
                break
            assert process.poll() is None, log.read_text()
            time.sleep(.02)
        assert len(children) == 2, log.read_text()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) != 0
        status = json.loads((prepared / 'controller_status.json').read_text())
        assert status['state'] == 'interrupted'
        assert not any(job['pid'] is not None for job in status['jobs'])
        for pid in children:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        for path in [prepared / '.controller.lock', tmp_path / 'gpu0.lock', tmp_path / 'gpu1.lock']:
            with queue.exclusive_lock(path):
                pass
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=15)
