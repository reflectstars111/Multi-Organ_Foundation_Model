"""Durable Q01-active -> Q03 -> Q01-remaining scheduling, outside frozen suites.

Only the old Q01 controller is stopped; its training processes finish normally.
The old controller is released only after frozen epoch80 verification. A separate
watchdog restarts this supervisor after unexpected death. Durable gated exec
prevents the spawn/status-write gap from producing an untracked controller.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
while str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))
from scripts.medical_priority_process import (  # noqa: E402
    atomic_write_json, children_of, exclusive_lock, locks_available,
    process_identity, process_running, signal_process,
    stop_process, wait_exited,
)

ORDER = ['finish_active_q01', 'q03', 'remaining_q01']
Q03_ARMS = ('shared_gn', 'domain_gn', 'shared_bn', 'domain_bn')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(value, key):
    return {**value, key: digest(value)}


def load_plan(path):
    path = Path(path).resolve()
    plan = read(path)
    if (plan.get('schema') != 'medical_queue_priority_v1' or plan.get('order') != ORDER
            or plan.get('plan_sha256') != digest({k: v for k, v in plan.items() if k != 'plan_sha256'})):
        raise ValueError('Invalid priority plan identity/order/digest')
    snapshot = path.parent / 'source_snapshot'
    if Path(__file__).resolve() != snapshot / 'scripts/run_medical_priority.py':
        raise ValueError('Run the frozen priority supervisor source')
    sources = plan.get('source_files', {})
    if 'scripts/run_medical_priority.py' not in sources:
        raise ValueError('Missing frozen supervisor source')
    for relative, expected in sources.items():
        target = snapshot / relative
        if (not target.resolve().is_relative_to(snapshot.resolve())
                or file_digest(target) != expected):
            raise ValueError(f'Priority source changed: {relative}')
    return plan


def worker_job(command, root):
    if (command[1:3] != ['-m', 'unet_moe.organ_weighted_suite']
            or '--evaluate-test' in command):
        raise ValueError(f'Unexpected Q01 child command: {command}')
    values = {}
    for key in ('output', 'kind', 'seed', 'arm'):
        flag = '--' + key
        if command.count(flag) != 1:
            raise ValueError(f'Expected one {flag} in Q01 worker')
        values[key] = command[command.index(flag) + 1]
    if (Path(values.pop('output')).resolve() != Path(root).resolve()
            or values['kind'] not in ('shared', 'region', 'task_id')
            or values['arm'] not in ('image_mean', 'organ_exposure', 'organ_exposure_fg_bg')
            or values['seed'] not in ('42', '43', '44')):
        raise ValueError('Q01 worker outside frozen matrix')
    values['seed'] = int(values['seed'])
    return values


def matrix_complete(summary, protocol_hash):
    expected = {('region', s, a) for s in (42, 43, 44) for a in Q03_ARMS}
    runs = summary.get('runs', [])
    return (summary.get('complete') is True and summary.get('protocol_sha256') == protocol_hash
            and len(runs) == 12
            and {(r.get('kind'), r.get('seed'), r.get('arm')) for r in runs} == expected
            and all(r.get('epochs_completed') == 80 and r.get('test_evaluated') is True
                    and r.get('protocol_sha256') == protocol_hash for r in runs))


class Supervisor:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.plan = load_plan(self.manifest)
        for name in ('q01', 'q03'):
            protocol = read(Path(self.plan[f'{name}_root']) / 'protocol.json')
            if protocol.get('protocol_sha256') != self.plan[f'{name}_protocol_sha256']:
                raise ValueError(f'{name} protocol changed')
            if protocol['protocol_sha256'] != digest({k: v for k, v in protocol.items()
                                                     if k != 'protocol_sha256'}):
                raise ValueError(f'{name} protocol digest invalid')
        self.state = (read(self.root / 'status.json') if (self.root / 'status.json').exists()
                      else dict(phase='prepared', plan_sha256=self.plan['plan_sha256']))
        if self.state.get('plan_sha256') != self.plan['plan_sha256']:
            raise ValueError('Status belongs to another priority plan')
        self.children = {}

    def save(self, **values):
        self.state.update(values, updated_at=now())
        atomic_write_json(self.root / 'status.json', self.state)

    def collect_workers(self):
        previous = {w['identity']['pid']: w for w in self.state.get('workers', [])}
        for identity in children_of(self.plan['original_q01_process']):
            old = previous.get(identity['pid'])
            if old and old['identity']['starttime'] == identity['starttime']:
                continue
            previous[identity['pid']] = dict(
                identity=identity, job=worker_job(identity['cmdline'], self.plan['q01_root']))
        if not previous:
            raise RuntimeError('No active Q01 worker identities captured; refuse empty handoff')
        self.save(workers=list(previous.values()))

    def prepare(self):
        parent = self.plan['original_q01_process']
        if not process_running(parent):
            raise RuntimeError('Original Q01 controller is no longer running')
        # Save before STOP, then re-enumerate after STOP to cover Popen/status races.
        self.collect_workers()
        self.save(phase='stopping_q01')
        if not stop_process(parent):
            raise RuntimeError('Q01 controller exited before STOP')
        self.collect_workers()
        self.save(phase='draining', drain_started_at=now(),
                  policy='finish active Q01 workers; Q03 matrix; remaining Q01')

    def verify_worker(self, job):
        source = Path(self.plan['q01_root']) / 'source_snapshot'
        command = [self.plan['python'], '-m', 'unet_moe.organ_weighted_suite',
                   '--output', self.plan['q01_root'], '--kind', job['kind'],
                   '--seed', str(job['seed']), '--arm', job['arm'], '--verify-run']
        with (self.root / 'verification.log').open('a') as stream:
            stream.write(json.dumps(dict(time=now(), command=command)) + '\n')
            stream.flush()
            subprocess.run(command, cwd=source, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(source),
                                'OMP_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'})

    def drain(self):
        parent = self.plan['original_q01_process']
        current = process_identity(parent['pid'])
        if not current or current['starttime'] != parent['starttime'] or current['state'] not in ('T', 't'):
            raise RuntimeError('Q01 controller must remain stopped during drain')
        self.collect_workers()
        running = [w for w in self.state['workers'] if process_running(w['identity'])]
        self.save(active_workers_remaining=len(running))
        if running:
            return False
        for worker in self.state['workers']:
            self.verify_worker(worker['job'])
        # This durable state allows recovery from SIGKILL between release and handoff.
        self.save(phase='releasing_q01', active_q01_verified=True, verified_at=now())
        return True

    def release(self):
        parent = self.plan['original_q01_process']
        if process_running(parent):
            current = process_identity(parent['pid'])
            if current['state'] not in ('T', 't'):
                raise RuntimeError('Refusing to release an unstopped Q01 controller')
            if any(process_running(w['identity']) for w in self.state['workers']):
                raise RuntimeError('Q01 worker restarted before release')
            if any(process_running(p) for p in children_of(parent)):
                raise RuntimeError('Unexpected live Q01 controller child')
            signal_process(parent, signal.SIGKILL)  # Single PID, never a process group.
            if not wait_exited(parent, 5):
                raise RuntimeError('Q01 controller did not release')
        # A waiting Q03 has no GPU workers; terminate it before the priority adapter.
        waiting = self.plan['original_q03_process']
        if process_running(waiting):
            if children_of(waiting):
                raise RuntimeError('Original Q03 controller unexpectedly has children')
            status = read(Path(self.plan['q03_root']) / 'controller_status.json')
            if status['state'] != 'waiting_reference':
                raise RuntimeError('Original Q03 controller no longer waiting for reference')
            signal_process(waiting, signal.SIGTERM)
            if not wait_exited(waiting, 30):
                raise RuntimeError('Original Q03 waiting controller has not exited')
        roots = [Path(self.plan[f'{name}_root']) / '.controller.lock' for name in ('q01', 'q03')]
        if not locks_available(roots):
            raise RuntimeError('Experiment controller lock still held; refuse handoff')
        handoff = {key: self.plan[key] for key in ('plan_sha256', 'q01_root', 'q03_root',
                                                  'q01_protocol_sha256', 'q03_protocol_sha256')}
        handoff.update(active_q01_verified=True, q01_controller_released=True,
                       verified_workers=self.state['workers'], released_at=now())
        atomic_write_json(self.root / 'handoff.json', signed(handoff, 'handoff_sha256'))
        self.save(phase='q03_pending', q01_controller_released=True)

    def launch(self, name):
        # A gated wrapper cannot exec until its identity is durable. If the owner
        # dies before that write the wrapper exits, so a restart cannot duplicate it.
        owner = process_identity(os.getpid())
        gate = self.root / f'{name}_launch.json'
        command = [self.plan['python'], str(Path(__file__).resolve()),
                   '--manifest', str(self.manifest), '--child', name,
                   '--owner', json.dumps(owner)]
        with (self.root / f'{name}_controller.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=CODE_ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, 'PYTHONPATH': str(CODE_ROOT),
                                            'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4'})
        identity = process_identity(process.pid)
        if identity is None:
            raise RuntimeError('Gated child exited before identity could be persisted')
        self.children[name] = process
        self.save(phase=f'{name}_running', **{f'{name}_process': identity})
        atomic_write_json(gate, dict(plan_sha256=self.plan['plan_sha256'], identity=identity))

    def wait_controller(self, name):
        identity = self.state[f'{name}_process']
        process = self.children.get(name)
        if process is not None:
            result = process.poll()
            if result is not None:
                self.save(**{f'{name}_exit_code': result})
        if process_running(identity):
            # Repair a crash after identity persistence but before gate release.
            gate = self.root / f'{name}_launch.json'
            value = read(gate) if gate.exists() else {}
            if (value.get('identity') != identity
                    or value.get('plan_sha256') != self.plan['plan_sha256']):
                atomic_write_json(gate, dict(plan_sha256=self.plan['plan_sha256'], identity=identity))
            return False
        root = Path(self.plan[f'{name}_root'])
        if name == 'q03':
            status = read(root / 'controller_status.json')
            summary = read(root / 'summary.json') if (root / 'summary.json').exists() else {}
            if (self.state.get('q03_exit_code', 0) != 0 or status.get('state') != 'complete'
                    or not matrix_complete(summary, self.plan['q03_protocol_sha256'])):
                raise RuntimeError('Q03 did not complete its full verified matrix; see q03_controller.log')
            self.save(phase='q01_pending', q03_completed_at=now())
        return True

    def start_watchdog(self):
        with (self.root / 'watchdog.log').open('a') as stream:
            subprocess.Popen([self.plan['python'], str(Path(__file__).resolve()),
                              '--manifest', str(self.manifest), '--watchdog'],
                             cwd=CODE_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            path = self.root / 'watchdog.json'
            if path.exists() and process_running(read(path)['identity']):
                return
            time.sleep(0.05)
        raise RuntimeError('Watchdog did not become ready; Q01 was not stopped')

    def run(self):
        with exclusive_lock(self.root / '.supervisor.lock'):
            if self.state['phase'] in ('complete', 'failed'):
                raise RuntimeError(f"Priority handoff already {self.state['phase']}; inspect status first")
            self.save(supervisor=process_identity(os.getpid()))
            self.start_watchdog()
            def interrupted(signum, frame):
                raise InterruptedError(f'Priority supervisor received {signum}')
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, interrupted)
            try:
                while True:
                    phase = self.state['phase']
                    if phase in ('prepared', 'stopping_q01'):
                        self.prepare()
                    elif phase == 'draining':
                        self.drain()
                    elif phase == 'releasing_q01':
                        self.release()
                    elif phase == 'q03_pending':
                        self.launch('q03')
                    elif phase == 'q03_running':
                        self.wait_controller('q03')
                    elif phase == 'q01_pending':
                        self.launch('q01')
                    elif phase == 'q01_running':
                        root = Path(self.plan['q01_root'])
                        status = read(root / 'controller_status.json')
                        identity = self.state['q01_process']
                        if (status.get('controller_pid') == identity['pid']
                                and status.get('state') in ('training', 'evaluating_test')
                                and any(j.get('pid') and (j.get('train_state') == 'running'
                                                        or j.get('test_state') == 'running')
                                        for j in status.get('jobs', []))
                                and process_running(identity)):
                            self.save(phase='complete', completed_at=now(),
                                      outcome='Q03 complete; original Q01 controller resumed')
                            return
                        if not process_running(identity):
                            raise RuntimeError('Resumed Q01 controller exited before startup confirmation')
                        self.wait_controller('q01')
                    else:
                        raise ValueError(f'Unknown priority phase: {phase}')
                    time.sleep(self.plan.get('poll_seconds', 15))
            except BaseException as error:
                # A committed gated launch or a live controller still needs its
                # successor scheduled. Keep the resumable phase; the watchdog
                # adopts it instead of abandoning a child behind a closed gate.
                phase = self.state['phase']
                live_child = (phase in ('q03_running', 'q01_running')
                              and process_running(self.state.get(f'{phase[:3]}_process')))
                if live_child or isinstance(error, (InterruptedError, KeyboardInterrupt)):
                    self.save(last_supervisor_error=f'{type(error).__name__}: {error}',
                              recovery_pending=True)
                    raise
                # A verification failure must not strand the
                # stopped parent. After release, a failed experiment stays failed.
                parent = self.plan['original_q01_process']
                current = process_identity(parent['pid'])
                restored = False
                if (current and current['starttime'] == parent['starttime']
                        and current['state'] in ('T', 't')):
                    restored = signal_process(parent, signal.SIGCONT)
                self.save(phase='failed', failed_at=now(), error=f'{type(error).__name__}: {error}',
                          original_q01_resumed=restored)
                raise


def child(manifest, name, owner):
    plan = load_plan(manifest)
    root = Path(manifest).resolve().parent
    identity = process_identity(os.getpid())
    gate = root / f'{name}_launch.json'
    while True:
        if gate.exists():
            value = read(gate)
            permitted = value.get('identity', {})
            if (value.get('plan_sha256') == plan['plan_sha256']
                    and (permitted.get('pid'), permitted.get('starttime')) ==
                    (identity['pid'], identity['starttime'])):
                break
        if not process_running(owner):
            # A persisted identity can be adopted by the restarted supervisor.
            state = read(root / 'status.json')
            recorded = state.get(f'{name}_process', {})
            if (recorded.get('pid'), recorded.get('starttime')) != (identity['pid'], identity['starttime']):
                raise RuntimeError('Owner died before child identity was persisted')
        time.sleep(0.1)
    gpu_arg = ','.join(str(g) for g in plan['gpus'])
    if name == 'q03':
        command = [plan['python'], str(root / 'source_snapshot/scripts/run_region_normalization_priority.py'),
                   '--output', plan['q03_root'], '--after-root', plan['q01_root'],
                   '--priority-manifest', str(Path(manifest).resolve()), '--gpus', gpu_arg,
                   '--poll-seconds', str(plan.get('poll_seconds', 15))]
        cwd = root / 'source_snapshot'
    else:
        command = plan['original_q01_process']['cmdline']
        cwd = Path(plan['q01_root']) / 'source_snapshot'
    os.chdir(cwd)
    os.environ['PYTHONPATH'] = str(cwd)
    os.execv(command[0], command)


def watchdog(manifest):
    plan = load_plan(manifest)
    root = Path(manifest).resolve().parent
    with exclusive_lock(root / '.watchdog.lock'):
        atomic_write_json(root / 'watchdog.json', dict(identity=process_identity(os.getpid()),
                                                       started_at=now()))
        spawned = None
        while True:
            if spawned is not None:
                spawned.poll()
            state = read(root / 'status.json')
            if state['phase'] in ('complete', 'failed'):
                return
            if not process_running(state['supervisor']):
                # Durable source and stages allow adoption after SIGKILL. The
                # supervisor's exclusive lock rejects concurrent manual restart.
                with (root / 'supervisor.log').open('a') as stream:
                    spawned = subprocess.Popen([plan['python'], str(Path(__file__).resolve()),
                                                '--manifest', str(Path(manifest).resolve())],
                                               cwd=CODE_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                time.sleep(3)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--child', choices=['q01', 'q03'])
    parser.add_argument('--owner', type=json.loads)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.manifest)
    elif args.child:
        child(args.manifest, args.child, args.owner)
    else:
        Supervisor(args.manifest).run()


if __name__ == '__main__':
    main()
