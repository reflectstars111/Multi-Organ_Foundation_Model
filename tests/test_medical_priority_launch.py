"""Real gated-exec and watchdog tests confined to temporary toy queues."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from scripts.medical_priority_process import atomic_write_json, process_identity


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.025)
    raise AssertionError("toy queue did not reach its expected state")


def cleanup(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


@pytest.fixture
def toy_queue(tmp_path):
    source = tmp_path / 'source_snapshot'
    scripts = source / 'scripts'
    scripts.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    for name in ('run_medical_priority.py', 'medical_priority_process.py'):
        shutil.copy2(repository / 'scripts' / name, scripts / name)
    (scripts / '__init__.py').write_text('')
    spec = importlib.util.spec_from_file_location('toy_medical_priority', scripts / 'run_medical_priority.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = tmp_path / 'executed'
    toy_command = [sys.executable, '-c',
                   'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("executed")', str(marker)]
    plan = dict(schema='medical_queue_priority_v1', order=module.ORDER,
                python=sys.executable, gpus=[], poll_seconds=0.025,
                original_q01_process=dict(pid=0, starttime=0, cmdline=toy_command),
                source_files={f'scripts/{name}': module.file_digest(scripts / name)
                              for name in ('run_medical_priority.py', 'medical_priority_process.py')})
    for name in ('q01', 'q03'):
        root = tmp_path / name
        (root / 'source_snapshot').mkdir(parents=True)
        protocol = module.signed(dict(test_fixture=name), 'protocol_sha256')
        atomic_write_json(root / 'protocol.json', protocol)
        plan[f'{name}_root'] = str(root)
        plan[f'{name}_protocol_sha256'] = protocol['protocol_sha256']
    plan = module.signed(plan, 'plan_sha256')
    manifest = tmp_path / 'plan.json'
    atomic_write_json(manifest, plan)
    atomic_write_json(tmp_path / 'status.json', dict(phase='q01_pending', plan_sha256=plan['plan_sha256']))
    return dict(module=module, root=tmp_path, manifest=manifest, plan=plan,
                script=scripts / 'run_medical_priority.py', marker=marker)


def start_child(toy, owner):
    return subprocess.Popen([sys.executable, str(toy['script']), '--manifest', str(toy['manifest']),
                             '--child', 'q01', '--owner', json.dumps(process_identity(owner.pid))],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def test_owner_death_before_identity_persistence_never_executes(toy_queue):
    toy = toy_queue
    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    child = None
    try:
        child = start_child(toy, owner)
        owner.terminate()
        owner.wait(timeout=3)
        _, error = child.communicate(timeout=4)
        assert child.returncode != 0
        assert 'before child identity was persisted' in error
        assert not toy['marker'].exists()
    finally:
        cleanup(child)
        cleanup(owner)


def test_durable_identity_survives_owner_death_and_missing_gate_is_repaired(toy_queue):
    toy = toy_queue
    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    child = None
    try:
        child = start_child(toy, owner)
        identity = process_identity(child.pid)
        atomic_write_json(toy['root'] / 'status.json', dict(
            phase='q01_running', plan_sha256=toy['plan']['plan_sha256'], q01_process=identity))
        owner.terminate()
        owner.wait(timeout=3)
        time.sleep(0.25)
        assert child.poll() is None
        assert not toy['marker'].exists()
        supervisor = toy['module'].Supervisor(toy['manifest'])
        assert supervisor.wait_controller('q01') is False
        child.wait(timeout=4)
        assert child.returncode == 0, child.stderr.read()
        assert toy['marker'].read_text() == 'executed'
        gate = json.loads((toy['root'] / 'q01_launch.json').read_text())
        assert gate['identity']['pid'] == child.pid
        assert gate['identity']['starttime'] == identity['starttime']
    finally:
        cleanup(child)
        cleanup(owner)


@pytest.mark.parametrize('corruption', ['pid', 'starttime', 'plan'])
def test_wrong_gate_never_executes(toy_queue, corruption):
    toy = toy_queue
    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    child = None
    try:
        child = start_child(toy, owner)
        identity = process_identity(child.pid)
        gate = dict(plan_sha256=toy['plan']['plan_sha256'], identity=dict(identity))
        if corruption == 'plan':
            gate['plan_sha256'] = 'f' * 64
        else:
            gate['identity'][corruption] += 1
        atomic_write_json(toy['root'] / 'q01_launch.json', gate)
        time.sleep(0.25)
        assert child.poll() is None
        assert not toy['marker'].exists()
        owner.terminate()
        owner.wait(timeout=3)
        child.wait(timeout=4)
        assert child.returncode != 0
        assert not toy['marker'].exists()
    finally:
        cleanup(child)
        cleanup(owner)


def test_launch_persists_identity_then_executes_only_toy_controller(toy_queue):
    toy = toy_queue
    supervisor = toy['module'].Supervisor(toy['manifest'])
    child = None
    try:
        supervisor.launch('q01')
        child = supervisor.children['q01']
        child.wait(timeout=4)
        assert child.returncode == 0
        assert toy['marker'].read_text() == 'executed'
        state = json.loads((toy['root'] / 'status.json').read_text())
        gate = json.loads((toy['root'] / 'q01_launch.json').read_text())
        assert state['phase'] == 'q01_running'
        assert state['q01_process']['pid'] == child.pid
        assert gate['identity'] == state['q01_process']
    finally:
        cleanup(child)


def test_adoption_repairs_gate_with_wrong_plan_digest(toy_queue):
    toy = toy_queue
    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    child = None
    try:
        child = start_child(toy, owner)
        identity = process_identity(child.pid)
        atomic_write_json(toy['root'] / 'status.json', dict(
            phase='q01_running', plan_sha256=toy['plan']['plan_sha256'], q01_process=identity))
        gate_path = toy['root'] / 'q01_launch.json'
        atomic_write_json(gate_path, dict(plan_sha256='f' * 64, identity=identity))
        supervisor = toy['module'].Supervisor(toy['manifest'])
        assert supervisor.wait_controller('q01') is False
        assert json.loads(gate_path.read_text())['plan_sha256'] == toy['plan']['plan_sha256']
        child.wait(timeout=4)
        assert child.returncode == 0, child.stderr.read()
        assert toy['marker'].read_text() == 'executed'
    finally:
        cleanup(child)
        cleanup(owner)


def test_watchdog_restarts_dead_toy_supervisor_once(toy_queue):
    toy = toy_queue
    module = toy['module']
    # The watchdog itself is real; its replacement executable only records a
    # startup and terminal state. No experiment controller can be invoked here.
    harness = toy['root'] / 'toy_python'
    harness.write_text('#!' + sys.executable + '\n' +
                       'import json,pathlib\n' +
                       f'root=pathlib.Path({str(toy["root"])!r})\n' +
                       '(root/"restarts").open("a").write("started\\n")\n' +
                       'p=root/"status.json"\n' +
                       's=json.loads(p.read_text()); s["phase"]="complete"\n' +
                       'p.write_text(json.dumps(s))\n')
    harness.chmod(0o700)
    plan = {k: v for k, v in toy['plan'].items() if k != 'plan_sha256'}
    plan['python'] = str(harness)
    plan = module.signed(plan, 'plan_sha256')
    atomic_write_json(toy['manifest'], plan)
    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    watchdog = None
    try:
        atomic_write_json(toy['root'] / 'status.json', dict(
            phase='draining', plan_sha256=plan['plan_sha256'], supervisor=process_identity(owner.pid)))
        watchdog = subprocess.Popen([sys.executable, str(toy['script']), '--manifest',
                                     str(toy['manifest']), '--watchdog'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        wait_until((toy['root'] / 'watchdog.json').exists)
        assert not (toy['root'] / 'restarts').exists()
        owner.kill()
        owner.wait(timeout=3)
        watchdog.wait(timeout=12)
        assert watchdog.returncode == 0, watchdog.stderr.read()
        assert (toy['root'] / 'restarts').read_text().splitlines() == ['started']
        assert json.loads((toy['root'] / 'status.json').read_text())['phase'] == 'complete'
        assert not toy['marker'].exists()
    finally:
        cleanup(watchdog)
        cleanup(owner)
