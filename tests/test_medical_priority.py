"""Priority handoff safety checks, separate from training/model tests."""
import copy
import importlib
import json
import sys

import pytest


@pytest.fixture
def queue():
    return importlib.import_module('scripts.run_medical_priority')


def test_worker_identity_comes_from_actual_command(queue, tmp_path):
    command = [sys.executable, '-m', 'unet_moe.organ_weighted_suite', '--output', str(tmp_path),
               '--kind', 'shared', '--seed', '44', '--arm', 'organ_exposure', '--device', 'cuda:0']
    assert queue.worker_job(command, tmp_path) == dict(kind='shared', seed=44, arm='organ_exposure')
    for bad in [command + ['--evaluate-test'], command + ['--output', '/wrong'],
                [sys.executable, '-c', 'print(1)']]:
        with pytest.raises(ValueError):
            queue.worker_job(bad, tmp_path)


def test_q03_complete_requires_entire_frozen_matrix(queue):
    runs = [dict(kind='region', seed=s, arm=a, epochs_completed=80, test_evaluated=True,
                 protocol_sha256='a' * 64)
            for s in (42, 43, 44) for a in queue.Q03_ARMS]
    summary = dict(protocol_sha256='a' * 64, complete=True, runs=runs)
    assert queue.matrix_complete(summary, 'a' * 64)
    for change in ('missing', 'duplicate', 'early_test', 'wrong_protocol'):
        bad = copy.deepcopy(summary)
        if change == 'missing':
            bad['runs'].pop()
        elif change == 'duplicate':
            bad['runs'][0] = bad['runs'][1]
        elif change == 'early_test':
            bad['runs'][0]['epochs_completed'] = 79
        else:
            bad['runs'][0]['protocol_sha256'] = 'b' * 64
        assert not queue.matrix_complete(bad, 'a' * 64)


def test_plan_rejects_changed_source_before_signals(queue, tmp_path, monkeypatch):
    source = tmp_path / 'source_snapshot' / 'scripts'
    source.mkdir(parents=True)
    module = source / 'run_medical_priority.py'
    module.write_text('frozen')
    monkeypatch.setattr(queue, '__file__', str(module))
    plan = dict(schema='medical_queue_priority_v1',
                order=['finish_active_q01', 'q03', 'remaining_q01'],
                source_files={'scripts/run_medical_priority.py': queue.file_digest(module)})
    plan['plan_sha256'] = queue.digest(plan)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    assert queue.load_plan(path) == plan
    module.write_text('modified')
    with pytest.raises(ValueError, match='source'):
        queue.load_plan(path)


def test_failed_verification_never_releases_controller(queue, monkeypatch, tmp_path):
    item = object.__new__(queue.Supervisor)
    item.plan = {'original_q01_process': {'pid': 100, 'starttime': 123},
                 'q01_root': str(tmp_path), 'q03_root': str(tmp_path / 'q03')}
    item.state = {'phase': 'draining', 'workers': [dict(identity={'pid': 101},
                  job=dict(kind='shared', seed=44, arm='organ_exposure'))]}
    item.root = tmp_path
    signals = []
    monkeypatch.setattr(queue, 'process_identity', lambda pid: dict(pid=pid, starttime=123, state='T'))
    monkeypatch.setattr(queue, 'process_running', lambda identity: False)
    monkeypatch.setattr(queue, 'children_of', lambda identity: [])
    monkeypatch.setattr(queue, 'signal_process', lambda *args: signals.append(args))
    monkeypatch.setattr(item, 'save', lambda **kwargs: item.state.update(kwargs))
    def fail(job):
        raise RuntimeError('incomplete epoch80 artifact')
    monkeypatch.setattr(item, 'verify_worker', fail)
    with pytest.raises(RuntimeError, match='incomplete'):
        item.drain()
    assert signals == []
    assert item.state['phase'] == 'draining'


def test_live_worker_blocks_release_and_verification(queue, monkeypatch, tmp_path):
    item = object.__new__(queue.Supervisor)
    item.plan = {'original_q01_process': {'pid': 100, 'starttime': 123}, 'q01_root': str(tmp_path)}
    item.state = {'phase': 'draining', 'workers': [dict(identity={'pid': 101},
                  job=dict(kind='shared', seed=44, arm='organ_exposure'))]}
    monkeypatch.setattr(queue, 'process_identity', lambda pid: dict(pid=pid, starttime=123, state='T'))
    monkeypatch.setattr(queue, 'process_running', lambda identity: True)
    monkeypatch.setattr(queue, 'children_of', lambda identity: [])
    monkeypatch.setattr(item, 'save', lambda **kwargs: item.state.update(kwargs))
    monkeypatch.setattr(item, 'verify_worker', lambda job: pytest.fail('live worker verification'))
    assert item.drain() is False
    assert item.state['phase'] == 'draining'
