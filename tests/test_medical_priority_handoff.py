"""Real temporary processes exercise Q01 -> Q03 -> Q01 without GPUs or real runs."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('Temporary process did not reach expected state')


@pytest.fixture
def live_handoff(tmp_path):
    scheduler, q01, q03 = (tmp_path / name for name in ('scheduler', 'q01', 'q03'))
    snapshot = scheduler / 'source_snapshot'
    (snapshot / 'scripts').mkdir(parents=True)
    for name in ('__init__.py', 'run_medical_priority.py', 'medical_priority_process.py'):
        shutil.copy2(SOURCE_ROOT / 'scripts' / name, snapshot / 'scripts' / name)
    original_path = list(sys.path)
    spec = importlib.util.spec_from_file_location(
        '_temporary_medical_supervisor', snapshot / 'scripts/run_medical_priority.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q01_source = q01 / 'source_snapshot'
    (q01_source / 'unet_moe').mkdir(parents=True)
    (q01_source / 'unet_moe/__init__.py').write_text('')
    q03.mkdir()
    worker = q01_source / 'unet_moe/organ_weighted_suite.py'
    worker.write_text('''import argparse, json, pathlib, time
p = argparse.ArgumentParser()
p.add_argument('--output', type=pathlib.Path)
p.add_argument('--kind'); p.add_argument('--seed'); p.add_argument('--arm')
a = p.parse_args()
(a.output / (a.arm + '_ready')).touch()
while not (a.output / ('allow_' + a.arm)).exists(): time.sleep(.01)
(a.output / (a.arm + '_done')).write_text(json.dumps({'epochs_completed':80}))
''')
    parent = q01_source / 'parent.py'
    parent.write_text('''import fcntl, json, os, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
lock = (root / '.controller.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
marker = root / 'parent_started'
if marker.exists():
    (root / 'resumed_command.json').write_text(json.dumps([sys.executable] + sys.argv))
    (root / 'controller_status.json').write_text(json.dumps({'state':'recovering', 'controller_pid':os.getpid()}))
else:
    marker.touch()
    children = [subprocess.Popen([sys.executable, '-m', 'unet_moe.organ_weighted_suite',
                 '--output',str(root),'--kind','task_id','--seed','44','--arm',arm],
                 start_new_session=True, pass_fds=(lock.fileno(),))
                for arm in ('image_mean', 'organ_exposure')]
    (root / 'children.json').write_text(json.dumps([child.pid for child in children]))
    while any(child.poll() is None for child in children): time.sleep(.01)
    (root / 'remaining_started_too_early').touch()
while True: time.sleep(.1)
''')
    waiting = q03 / 'waiting.py'
    waiting.write_text('''import fcntl, json, os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
lock = (root / '.controller.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(root / 'controller_status.json').write_text(json.dumps({'state':'waiting_reference', 'controller_pid':os.getpid()}))
while True: time.sleep(.1)
''')
    fake_adapter = snapshot / 'scripts/run_region_normalization_priority.py'
    fake_adapter.write_text('''import argparse, json, pathlib, time
p=argparse.ArgumentParser()
p.add_argument('--output',type=pathlib.Path);p.add_argument('--after-root',type=pathlib.Path)
p.add_argument('--priority-manifest');p.add_argument('--gpus');p.add_argument('--poll-seconds')
a=p.parse_args(); root=a.output
assert all((a.after_root/(arm+'_done')).exists() for arm in ('image_mean','organ_exposure'))
assert not (a.after_root/'remaining_started_too_early').exists()
(root/'priority_started').touch()
(root/'controller_status.json').write_text(json.dumps({'state':'training'}))
while not (root/'allow_q03_complete').exists(): time.sleep(.01)
sha=json.loads((root/'protocol.json').read_text())['protocol_sha256']
runs=[dict(kind='region',seed=s,arm=a,epochs_completed=80,test_evaluated=True,protocol_sha256=sha)
      for s in (42,43,44) for a in ('shared_gn','domain_gn','shared_bn','domain_bn')]
if (root/'incomplete_matrix').exists(): runs.pop()
(root/'summary.json').write_text(json.dumps({'complete':True,'protocol_sha256':sha,'runs':runs}))
(root/'controller_status.json').write_text(json.dumps({'state':'complete'}))
''')
    processes = []
    supervisor_instances = []
    try:
        parent_process = subprocess.Popen([sys.executable, str(parent), '--output', str(q01)],
                                         cwd=q01_source, start_new_session=True,
                                         env={**os.environ, 'PYTHONPATH': str(q01_source)})
        processes.append(parent_process)
        waiting_process = subprocess.Popen([sys.executable, str(waiting), str(q03)],
                                          start_new_session=True)
        processes.append(waiting_process)
        wait_until(lambda: all((q01 / (arm + '_ready')).exists()
                               for arm in ('image_mean', 'organ_exposure')))
        wait_until(lambda: (q03 / 'controller_status.json').exists())
        protocols = {}
        for name, root in (('q01', q01), ('q03', q03)):
            protocols[name] = module.signed({'experiment': name}, 'protocol_sha256')
            (root / 'protocol.json').write_text(json.dumps(protocols[name]))
        source_hashes = {str(path.relative_to(snapshot)): module.file_digest(path)
                         for path in snapshot.rglob('*.py')}
        plan = module.signed(dict(schema='medical_queue_priority_v1', order=module.ORDER,
                                  q01_root=str(q01), q03_root=str(q03),
                                  q01_protocol_sha256=protocols['q01']['protocol_sha256'],
                                  q03_protocol_sha256=protocols['q03']['protocol_sha256'],
                                  source_files=source_hashes, python=sys.executable,
                                  original_q01_process=module.process_identity(parent_process.pid),
                                  original_q03_process=module.process_identity(waiting_process.pid),
                                  gpus=[0, 1], poll_seconds=.01), 'plan_sha256')
        manifest = scheduler / 'plan.json'
        manifest.write_text(json.dumps(plan))

        class CheckedSupervisor(module.Supervisor):
            def verify_worker(self, job):
                record = json.loads((q01 / (job['arm'] + '_done')).read_text())
                assert record['epochs_completed'] == 80
                (scheduler / (job['arm'] + '_verified')).touch()

            def start_watchdog(self):
                # Tests explicitly recreate this class to exercise durable state.
                pass

        def supervisor(cls=CheckedSupervisor):
            item = cls(manifest)
            supervisor_instances.append(item)
            return item

        yield module, supervisor, q01, q03, scheduler
    finally:
        # Only PIDs created in this temporary fixture are ever touched.
        for item in supervisor_instances:
            processes.extend(item.children.values())
        children_file = q01 / 'children.json'
        if children_file.exists():
            for pid in json.loads(children_file.read_text()):
                identity = module.process_identity(pid)
                if identity and module.process_running(identity):
                    module.signal_process(identity, signal.SIGKILL)
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        sys.path[:] = original_path


def complete_workers(module, supervisor, q01):
    supervisor.prepare()
    assert supervisor.state['phase'] == 'draining'
    assert len(supervisor.state['workers']) == 2
    assert supervisor.drain() is False
    (q01 / 'allow_image_mean').touch()
    wait_until(lambda: (q01 / 'image_mean_done').exists())
    assert supervisor.drain() is False
    assert module.process_running(supervisor.plan['original_q01_process'])
    assert not (supervisor.root / 'handoff.json').exists()
    (q01 / 'allow_organ_exposure').touch()
    wait_until(lambda: (q01 / 'organ_exposure_done').exists())
    wait_until(lambda: not any(module.process_running(w['identity']) for w in supervisor.state['workers']))
    assert supervisor.drain() is True
    assert supervisor.state['phase'] == 'releasing_q01'
    assert not (q01 / 'remaining_started_too_early').exists()


def test_drains_both_then_q03_full_matrix_then_original_q01_command(live_handoff):
    module, new_supervisor, q01, q03, scheduler = live_handoff
    supervisor = new_supervisor()
    original_command = supervisor.plan['original_q01_process']['cmdline']
    complete_workers(module, supervisor, q01)
    # Restart from the durable state immediately before destructive release.
    recovered = new_supervisor()
    recovered.release()
    assert not module.process_running(recovered.plan['original_q01_process'])
    assert not module.process_running(recovered.plan['original_q03_process'])
    handoff = json.loads((scheduler / 'handoff.json').read_text())
    assert handoff['active_q01_verified'] and handoff['q01_controller_released']
    assert all((scheduler / (arm + '_verified')).exists()
               for arm in ('image_mean', 'organ_exposure'))
    recovered.launch('q03')
    wait_until(lambda: (q03 / 'priority_started').exists())
    assert recovered.wait_controller('q03') is False
    assert not (q01 / 'resumed_command.json').exists()
    assert not (q01 / 'remaining_started_too_early').exists()
    (q03 / 'allow_q03_complete').touch()
    wait_until(lambda: not module.process_running(recovered.state['q03_process']))
    # A second supervisor adopts the already finished Q03 child without Popen.
    resumed = new_supervisor()
    assert resumed.wait_controller('q03') is True
    assert resumed.state['phase'] == 'q01_pending'
    assert len(json.loads((q03 / 'summary.json').read_text())['runs']) == 12
    resumed.launch('q01')
    wait_until(lambda: (q01 / 'resumed_command.json').exists())
    assert json.loads((q01 / 'resumed_command.json').read_text()) == original_command
    assert module.process_running(resumed.state['q01_process'])


def test_partial_q03_summary_cannot_release_remaining_q01(live_handoff):
    module, new_supervisor, q01, q03, _ = live_handoff
    supervisor = new_supervisor()
    complete_workers(module, supervisor, q01)
    supervisor.release()
    (q03 / 'incomplete_matrix').touch()
    supervisor.launch('q03')
    wait_until(lambda: (q03 / 'priority_started').exists())
    (q03 / 'allow_q03_complete').touch()
    wait_until(lambda: not module.process_running(supervisor.state['q03_process']))
    with pytest.raises(RuntimeError, match='full verified matrix'):
        supervisor.wait_controller('q03')
    assert supervisor.state['phase'] == 'q03_running'
    assert not (q01 / 'resumed_command.json').exists()


def test_verification_failure_resumes_original_parent_without_handoff(live_handoff):
    module, new_supervisor, q01, q03, scheduler = live_handoff
    base = type(new_supervisor())

    class FailingVerifier(base):
        def prepare(self):
            super().prepare()
            (q01 / 'allow_image_mean').touch()
            (q01 / 'allow_organ_exposure').touch()

        def verify_worker(self, job):
            raise RuntimeError('checkpoint evidence rejected')

    supervisor = new_supervisor(FailingVerifier)
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(RuntimeError, match='checkpoint evidence rejected'):
            supervisor.run()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    assert supervisor.state['phase'] == 'failed'
    assert supervisor.state['original_q01_resumed'] is True
    parent = module.process_identity(supervisor.plan['original_q01_process']['pid'])
    assert parent['state'] not in ('T', 't', 'Z')
    assert module.process_running(supervisor.plan['original_q03_process'])
    assert not (scheduler / 'handoff.json').exists()
    assert not (q03 / 'priority_started').exists()
