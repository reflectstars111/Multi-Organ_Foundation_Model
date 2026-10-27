"""Failure-injection coverage for supervisor handoff and adoption.

All process identities, process launches and signals are fake.  Only temporary
status/gate files are written; these tests never touch experiment controllers.
"""
import importlib
import json
import os
import signal
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    queue = importlib.import_module('scripts.run_medical_priority')
    identities = {
        os.getpid(): dict(pid=os.getpid(), starttime=100, state='S'),
        11001: dict(pid=11001, starttime=101, state='T'),
        11002: dict(pid=11002, starttime=102, state='S'),
        11003: dict(pid=11003, starttime=103, state='S'),
    }
    item = object.__new__(queue.Supervisor)
    item.root = tmp_path
    item.manifest = tmp_path / 'plan.json'
    item.plan = dict(
        python=sys.executable, plan_sha256='a' * 64,
        q01_root=str(tmp_path / 'q01'), q03_root=str(tmp_path / 'q03'),
        original_q01_process=identities[11001],
        original_q03_process=identities[11002], poll_seconds=0.01,
    )
    item.state = dict(phase='q03_pending', plan_sha256='a' * 64)
    item.children = {}
    handlers, sent_signals = {}, []
    monkeypatch.setattr(item, 'start_watchdog', lambda: None)
    monkeypatch.setattr(queue, 'process_identity', lambda pid: identities.get(pid))
    monkeypatch.setattr(queue, 'process_running', lambda identity: (
        identity is not None and identity['pid'] in identities
        and identities[identity['pid']]['starttime'] == identity['starttime']
        and identities[identity['pid']]['state'] not in ('Z', 'X')
    ))
    monkeypatch.setattr(queue, 'signal_process', lambda *args: sent_signals.append(args))
    monkeypatch.setattr(queue.signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))
    return SimpleNamespace(queue=queue, item=item, identities=identities,
                           handlers=handlers, signals=sent_signals)


def test_gate_write_failure_preserves_committed_child_for_watchdog(supervisor, monkeypatch):
    """A live wrapper must remain adoptable after its gate write raises."""
    env = supervisor
    queue, item = env.queue, env.item
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(pid=11003))
    original_write = queue.atomic_write_json

    def fail_gate(path, value):
        if path.name == 'q03_launch.json':
            raise OSError('injected gate write failure')
        original_write(path, value)

    monkeypatch.setattr(queue, 'atomic_write_json', fail_gate)
    with pytest.raises(OSError, match='injected gate write'):
        item.run()

    durable = json.loads((item.root / 'status.json').read_text())
    assert durable['phase'] == 'q03_running'
    assert durable['q03_process'] == env.identities[11003]
    assert durable['recovery_pending'] is True
    assert not (item.root / 'q03_launch.json').exists()
    assert env.signals == []


def test_restarted_supervisor_releases_existing_child_without_respawn(supervisor, monkeypatch):
    """Adoption repairs the missing gate and never starts a second controller."""
    env = supervisor
    queue, item = env.queue, env.item
    item.state.update(phase='q03_running', q03_process=env.identities[11003])
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **k: pytest.fail('duplicate launch'))

    assert item.wait_controller('q03') is False
    gate = json.loads((item.root / 'q03_launch.json').read_text())
    assert gate['identity'] == env.identities[11003]
    assert gate['plan_sha256'] == item.plan['plan_sha256']
    assert item.state['phase'] == 'q03_running'
    assert env.signals == []


def test_sigterm_during_q03_retains_phase_and_live_child(supervisor, monkeypatch):
    env = supervisor
    item = env.item
    item.state.update(phase='q03_running', q03_process=env.identities[11003])

    def inject_term(name):
        assert name == 'q03'
        env.handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(item, 'wait_controller', inject_term)
    with pytest.raises(InterruptedError, match='received'):
        item.run()

    durable = json.loads((item.root / 'status.json').read_text())
    assert durable['phase'] == 'q03_running'
    assert durable['q03_process'] == env.identities[11003]
    assert durable['recovery_pending'] is True
    assert env.signals == []


@pytest.mark.parametrize('controller_state', ['recovering', 'training', 'evaluating_test'])
def test_resume_requires_worker_admission_not_just_controller_state(
        supervisor, monkeypatch, controller_state):
    """Recovery or a phase with no admitted worker must keep supervision."""
    env = supervisor
    item = env.item
    item.state.update(phase='q01_running', q01_process=env.identities[11003])
    root = item.root / 'q01'
    root.mkdir()
    (root / 'controller_status.json').write_text(json.dumps(dict(
        controller_pid=11003, state=controller_state, jobs=[])))
    observed = []

    def wait_then_interrupt(name):
        observed.append(name)
        raise InterruptedError('injected poll boundary interruption')

    monkeypatch.setattr(item, 'wait_controller', wait_then_interrupt)
    with pytest.raises(InterruptedError, match='poll boundary'):
        item.run()

    assert observed == ['q01']
    assert item.state['phase'] == 'q01_running'
    assert item.state['recovery_pending'] is True
    assert 'completed_at' not in item.state
    assert env.signals == []


def test_admitted_q01_training_worker_completes_handoff(supervisor, monkeypatch):
    env = supervisor
    item = env.item
    item.state.update(phase='q01_running', q01_process=env.identities[11003])
    root = item.root / 'q01'
    root.mkdir()
    (root / 'controller_status.json').write_text(json.dumps(dict(
        controller_pid=11003, state='training',
        jobs=[dict(pid=11004, train_state='running', test_state='pending')],
    )))
    monkeypatch.setattr(item, 'wait_controller', lambda name: pytest.fail('already admitted'))

    item.run()

    assert item.state['phase'] == 'complete'
    assert 'completed_at' in item.state
    assert env.signals == []
