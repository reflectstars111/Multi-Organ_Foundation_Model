"""Run the frozen 12-job Region normalization matrix after Q01 is complete.

Launch the frozen script with ``--output ROOT --after-root Q01_ROOT``. The queue
does not claim a GPU until Q01's full train/test matrix has passed its own frozen
summary validator. Four CUDA preflights precede training. All twelve verified
training runs must finish before any normalization test evaluation starts.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


CODE_ROOT = Path(__file__).resolve().parents[1]
# A launching shell may put checkout:snapshot on PYTHONPATH. Even when our
# snapshot is already present, it must precede the mutable checkout.
while str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from scripts.run_organ_weighted import (  # noqa: E402
    exclusive_lock, gpu_free_mib, gpu_lock_path, read_json, terminate_children,
    timestamp, write_json,
)


ARMS = ('shared_gn', 'domain_gn', 'shared_bn', 'domain_bn')
SEEDS = (42, 43, 44)
REFERENCE_MODELS = ('shared', 'region', 'task_id')
REFERENCE_ARMS = ('image_mean', 'organ_exposure', 'organ_exposure_fg_bg')


def valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def matrix_complete(summary, expected, protocol_sha256):
    """Cheap identity check; full checkpoint verification remains mandatory."""
    if not isinstance(summary, dict):
        return False
    runs = summary.get('runs')
    if (summary.get('complete') is not True or not isinstance(runs, list)
            or len(runs) != len(expected)
            or summary.get('protocol_sha256') != protocol_sha256
            or any(not isinstance(run, dict) for run in runs)):
        return False
    actual = set()
    for run in runs:
        identity = (run.get('kind'), run.get('seed'), run.get('arm'))
        if (any(isinstance(value, (list, dict)) for value in identity)
                or type(run.get('seed')) is not int
                or run.get('protocol_sha256') != protocol_sha256
                or type(run.get('epochs_completed')) is not int
                or run['epochs_completed'] != 80 or run.get('test_evaluated') is not True):
            return False
        actual.add(identity)
    return actual == expected


class Controller:
    def __init__(self, output, after_root, gpus=(0, 1), poll_seconds=15,
                 preflight_min_free_mib=24576):
        self.root = Path(output).resolve()
        self.reference_root = Path(after_root).resolve()
        self.gpus = list(gpus)
        if (not self.gpus or len(self.gpus) != len(set(self.gpus))
                or any(type(gpu) is not int or gpu < 0 for gpu in self.gpus)):
            raise ValueError('GPU indices must be distinct nonnegative integers')
        if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
            raise ValueError('poll_seconds must be positive and at most 60')
        if type(preflight_min_free_mib) is not int or preflight_min_free_mib < 16384:
            raise ValueError('CUDA preflight requires at least 16384 MiB admission memory')
        self.poll_seconds = poll_seconds
        self.preflight_min_free_mib = preflight_min_free_mib
        self.protocol = read_json(self.root / 'protocol.json')
        self.protocol_sha256 = self.protocol.get('protocol_sha256')
        self.reference_sha256 = self.protocol.get('reference_protocol_sha256')
        config = self.protocol.get('config', {})
        if (not valid_sha256(self.protocol_sha256) or not valid_sha256(self.reference_sha256)
                or type(config.get('epochs')) is not int or config['epochs'] != 80
                or config.get('models') != ['region']
                or config.get('seeds') != list(SEEDS) or config.get('arms') != list(ARMS)):
            raise ValueError('Region normalization needs the frozen 80-epoch, four-arm protocol')
        frozen_reference = self.protocol.get('reference_root')
        if (not isinstance(frozen_reference, str)
                or Path(frozen_reference).resolve() != self.reference_root):
            raise ValueError('--after-root must equal the frozen protocol reference_root')
        if self.root == self.reference_root:
            raise ValueError('Normalization output must differ from reference_root')

        self.suite_prefix = [sys.executable, '-m', 'unet_moe.region_normalization_suite']
        self.reference_prefix = [sys.executable, '-m', 'unet_moe.organ_weighted_suite']
        self.reference_code_root = self.reference_root / 'source_snapshot'
        # Instance boundaries permit subprocess tests without changing the old
        # production controller or monkeypatching any of its module constants.
        self.gpu_memory = gpu_free_mib
        self.gpu_lock_path = gpu_lock_path
        self.required_mib = {}
        self.jobs = [dict(kind='region', seed=seed, arm=arm, train_state='pending',
                          test_state='pending', pid=None, gpu=None)
                     for seed in SEEDS for arm in ARMS]
        self.active = {}
        self.control_child = None
        self.root_lock = None
        self.status = dict(
            schema='region_normalization_controller_v1', state='starting',
            output=str(self.root), reference_root=str(self.reference_root),
            protocol_sha256=self.protocol_sha256, reference_protocol_sha256=self.reference_sha256,
            controller_pid=os.getpid(), started_at=timestamp(), gpu_indices=self.gpus,
            preflight_min_free_mib=preflight_min_free_mib, required_free_mib=self.required_mib,
            preflights={}, jobs=self.jobs, gpu_observations={})

    def update(self, state=None, **extra):
        if state is not None:
            self.status['state'] = state
        self.status.update(updated_at=timestamp(), **extra)
        write_json(self.root / 'controller_status.json', self.status)

    def folder(self, job):
        return self.root / job['arm'] / job['kind'] / f'seed_{job["seed"]}'

    def arguments(self, job):
        return ['--kind', job['kind'], '--seed', str(job['seed']), '--arm', job['arm']]

    def run_command(self, command, label, *, cwd=CODE_ROOT, gpu=None, lock=None):
        path = self.root / 'logs' / 'verification.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptors = [self.root_lock.fileno()]
        if lock is not None:
            descriptors.append(lock.fileno())
        with path.open('a') as log:
            log.write(json.dumps(dict(time=timestamp(), operation=label, command=command,
                                      cwd=str(cwd), gpu=gpu)) + '\n')
            log.flush()
            # Pin import resolution to the corresponding snapshot even when the
            # launching shell has PYTHONPATH pointing at the mutable checkout.
            self.control_child = subprocess.Popen(
                command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu),
                     'PYTHONPATH': str(cwd), 'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'},
                start_new_session=True, pass_fds=tuple(descriptors))
            self.update(control_child_pid=self.control_child.pid)
            try:
                result = self.control_child.wait()
            finally:
                if self.control_child.poll() is None:
                    terminate_children([self.control_child])
                self.control_child = None
                self.update(control_child_pid=None)
        if result:
            raise RuntimeError(f'{label} exited {result}; see {path}')

    def run_cpu(self, arguments, label):
        self.run_command([*self.suite_prefix, '--output', str(self.root), *arguments], label)

    def reference_ready(self):
        """Do not infer completion from a controller state or a vanished PID."""
        reason = 'Q01 summary or per-run evidence is incomplete'
        try:
            protocol = read_json(self.reference_root / 'protocol.json')
            config = protocol.get('config', {})
            if (protocol.get('protocol_sha256') != self.reference_sha256
                    or config.get('epochs') != 80 or config.get('models') != list(REFERENCE_MODELS)
                    or config.get('arms') != list(REFERENCE_ARMS)
                    or config.get('seeds') != list(SEEDS)):
                raise ValueError('Q01 reference protocol differs from the frozen reference')
            expected = {(kind, seed, arm) for kind in REFERENCE_MODELS
                        for seed in SEEDS for arm in REFERENCE_ARMS}
            summary = read_json(self.reference_root / 'summary.json')
            if not matrix_complete(summary, expected, self.reference_sha256):
                return False
            for run in summary['runs']:
                folder = (self.reference_root / run['arm'] / run['kind']
                          / f'seed_{run["seed"]}')
                recorded = read_json(folder / 'run_summary.json')
                for key in ('kind', 'seed', 'arm', 'protocol_sha256', 'epochs_completed', 'test_evaluated'):
                    if recorded.get(key) != run[key]:
                        return False
                names = ['last.pt', 'best.pt', 'metrics.jsonl']
                if recorded.get('guardrail_epoch') is not None:
                    names.append('guardrail.pt')
                if any(not (folder / name).is_file() or (folder / name).stat().st_size == 0
                       for name in names):
                    return False
            reason = None
            return True
        except (FileNotFoundError, json.JSONDecodeError) as error:
            reason = f'Waiting for complete Q01 evidence: {error}'
            return False
        finally:
            self.status['reference_wait_reason'] = reason

    def wait_for_reference(self):
        self.update('waiting_reference')
        while not self.reference_ready():
            self.update()
            time.sleep(self.poll_seconds)
        if not (self.reference_code_root / 'unet_moe' / 'organ_weighted_suite.py').is_file():
            raise ValueError('Q01 frozen reference validator is missing')
        self.update('verifying_reference')
        self.run_command([*self.reference_prefix, '--output', str(self.reference_root), '--summarize'],
                         'reference complete train/test verification', cwd=self.reference_code_root)
        if not self.reference_ready():
            raise ValueError('Q01 reference changed while verifying its completed matrix')
        self.update(reference_verified_at=timestamp(), reference_wait_reason=None)

    def read_preflight(self, arm):
        path = self.root / 'preflight' / f'{arm}.json'
        report = read_json(path)
        required = report.get('required_free_mib')
        if (report.get('protocol_sha256') != self.protocol_sha256 or report.get('kind') != 'region'
                or report.get('arm') != arm or report.get('passed') is not True
                or type(required) is not int or required <= 0):
            raise ValueError(f'Invalid or failed preflight for {arm}: {path}')
        self.required_mib[arm] = required
        self.status['preflights'][arm] = dict(state='complete', path=str(path), required_free_mib=required)
        return report

    def admit(self, gpu, pending, *, preflight=False):
        lock = self.gpu_lock_path(gpu).open('a+')
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.status['gpu_observations'][str(gpu)] = dict(state='locked', time=timestamp())
                lock.close()
                return None
            free = self.gpu_memory(gpu)
            self.status['gpu_observations'][str(gpu)] = dict(
                state='available', free_mib=free, time=timestamp())
            for job in pending:
                required = self.preflight_min_free_mib if preflight else self.required_mib[job['arm']]
                if free >= required:
                    lock.seek(0)
                    lock.truncate()
                    lock.write(json.dumps(dict(pid=os.getpid(), output=str(self.root), gpu=gpu,
                                               arm=job['arm'], seed=job.get('seed'),
                                               phase='preflight' if preflight else 'run')) + '\n')
                    lock.flush()
                    return job, lock, free
            self.status['gpu_observations'][str(gpu)]['state'] = 'waiting_for_memory'
            lock.close()
            return None
        except BaseException:
            lock.close()
            raise

    def run_preflights(self):
        self.update('preflighting')
        # Validate all cached reports first. A failed/corrupt report requires an
        # explicit repair; silently overwriting it would hide a failed check.
        for arm in ARMS:
            if (self.root / 'preflight' / f'{arm}.json').exists():
                self.read_preflight(arm)
        for arm in ARMS:
            if arm in self.required_mib:
                continue
            self.status['preflights'][arm] = dict(state='waiting_for_gpu')
            admitted = None
            while admitted is None:
                for gpu in self.gpus:
                    admitted = self.admit(gpu, [dict(arm=arm)], preflight=True)
                    if admitted is not None:
                        break
                self.update()
                if admitted is None:
                    time.sleep(self.poll_seconds)
            _, lock, free = admitted
            try:
                self.status['preflights'][arm] = dict(state='running', gpu=gpu, free_mib_at_admission=free)
                self.update()
                self.run_command([*self.suite_prefix, '--output', str(self.root), '--preflight',
                                  '--arm', arm, '--device', 'cuda:0'],
                                 f'CUDA preflight {arm}', gpu=gpu, lock=lock)
                self.read_preflight(arm)
                self.update()
            finally:
                lock.close()

    def verify(self, job, require_test=False):
        arguments = self.arguments(job) + ['--verify-run']
        if require_test:
            arguments.append('--evaluate-test')
        self.run_cpu(arguments, f'verification {job["arm"]}/{job["kind"]}/{job["seed"]}')

    def recover(self):
        tested = False
        for job in self.jobs:
            path = self.folder(job) / 'run_summary.json'
            if not path.exists():
                continue
            summary = read_json(path)
            for key in ('kind', 'seed', 'arm'):
                if summary.get(key) != job[key]:
                    raise ValueError(f'Run summary has the wrong {key}: {path}')
            if summary.get('protocol_sha256') != self.protocol_sha256:
                raise ValueError(f'Run summary belongs to another protocol: {path}')
            epochs = summary.get('epochs_completed')
            if type(epochs) is not int or not 0 <= epochs <= 80:
                raise ValueError(f'Run summary has an invalid epoch budget: {path}')
            if epochs < 80:
                if summary.get('test_evaluated'):
                    raise ValueError(f'Test exists before training completion: {path}')
                continue
            require_test = summary.get('test_evaluated') is True
            self.verify(job, require_test=require_test)
            job['train_state'] = 'complete'
            if require_test:
                job['test_state'] = 'complete'
                tested = True
        if tested and any(job['train_state'] != 'complete' for job in self.jobs):
            raise ValueError('Existing test results violate the all-12-runs training barrier')

    def launch(self, gpu, job, lock, free, phase):
        command = [*self.suite_prefix, '--output', str(self.root), *self.arguments(job), '--device', 'cuda:0']
        folder = self.folder(job)
        has_artifacts = folder.exists() and any(folder.iterdir())
        if phase == 'test':
            if not has_artifacts:
                raise ValueError('Test evaluation requires existing training artifacts')
            command.append('--evaluate-test')
        elif phase == 'train':
            if has_artifacts:
                command.append('--resume')
        else:
            raise ValueError(f'Unknown job phase: {phase}')
        path = self.root / 'logs' / f'gpu{gpu}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open('a')
        try:
            log.write(json.dumps(dict(time=timestamp(), phase=phase, command=command,
                                     free_mib_at_admission=free)) + '\n')
            log.flush()
            process = subprocess.Popen(
                command, cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(CODE_ROOT),
                     'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}, start_new_session=True,
                pass_fds=(self.root_lock.fileno(), lock.fileno()))
        except BaseException:
            log.close()
            lock.close()
            raise
        self.active[gpu] = dict(process=process, job=job, lock=lock, log=log, phase=phase)
        job.update(pid=process.pid, gpu=gpu, log=str(path), free_mib_at_admission=free)
        job[f'{phase}_state'] = 'running'
        self.update()

    def reap(self):
        for gpu, item in list(self.active.items()):
            result = item['process'].poll()
            if result is None:
                continue
            self.active.pop(gpu)
            item['log'].close()
            item['lock'].close()
            job, phase = item['job'], item['phase']
            job['pid'] = None
            if result:
                job[f'{phase}_state'] = 'failed'
                job['error'] = f'{phase} subprocess exited {result}; see {job["log"]}'
                raise RuntimeError(job['error'])
            try:
                self.verify(job, require_test=phase == 'test')
            except BaseException as error:
                job[f'{phase}_state'] = 'failed'
                job['error'] = str(error)
                raise
            job[f'{phase}_state'] = 'complete'
            self.update()

    def run_phase(self, phase):
        self.update('training' if phase == 'train' else 'evaluating_test')
        while True:
            self.reap()
            pending = [job for job in self.jobs if job[f'{phase}_state'] == 'pending']
            if not pending and not self.active:
                return
            for gpu in self.gpus:
                if gpu in self.active or not pending:
                    continue
                admitted = self.admit(gpu, pending)
                if admitted is None:
                    continue
                job, lock, free = admitted
                try:
                    self.launch(gpu, job, lock, free, phase)
                except BaseException:
                    lock.close()
                    raise
                pending.remove(job)
            self.update(waiting_jobs=len(pending))
            time.sleep(self.poll_seconds)

    def validate_summary(self):
        summary = read_json(self.root / 'summary.json')
        expected = {(job['kind'], job['seed'], job['arm']) for job in self.jobs}
        if not matrix_complete(summary, expected, self.protocol_sha256):
            raise ValueError('Final summary does not cover the complete frozen test matrix')

    def stop(self):
        children = [item['process'] for item in self.active.values()]
        if self.control_child is not None:
            children.append(self.control_child)
        terminate_children(children)
        self.control_child = None
        for item in self.active.values():
            item['lock'].close()
            item['log'].close()
            item['job']['pid'] = None
            item['job'][f'{item["phase"]}_state'] = 'interrupted'
        self.active.clear()

    def run(self):
        with exclusive_lock(self.root / '.controller.lock') as root_lock:
            self.root_lock = root_lock
            previous_handlers = {}

            def interrupted(signum, frame):
                raise InterruptedError(f'Controller received signal {signum}')

            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    previous_handlers[signum] = signal.signal(signum, interrupted)
                previous_status = self.root / 'controller_status.json'
                if (previous_status.exists()
                        and read_json(previous_status).get('protocol_sha256') != self.protocol_sha256):
                    raise ValueError('Controller status belongs to another protocol')
                (self.root / 'controller.pid').write_text(f'{os.getpid()}\n')
                self.wait_for_reference()
                self.update('recovering')
                self.recover()
                self.run_preflights()
                self.run_phase('train')
                if any(job['train_state'] != 'complete' for job in self.jobs):
                    raise RuntimeError('Training barrier incomplete')
                self.update('training_barrier_verified', training_barrier_verified_at=timestamp())
                self.run_phase('test')
                if any(job['test_state'] != 'complete' for job in self.jobs):
                    raise RuntimeError('Test matrix incomplete')
                self.update('summarizing')
                self.run_cpu(['--summarize'], 'paired normalization summary')
                self.validate_summary()
                self.update('complete', completed_at=timestamp(), waiting_jobs=0,
                            summary=str(self.root / 'summary.json'))
            except BaseException as error:
                for signum in previous_handlers:
                    signal.signal(signum, signal.SIG_IGN)
                self.stop()
                self.update('interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
                            error=f'{type(error).__name__}: {error}', control_child_pid=None)
                raise
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                self.root_lock = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-root', type=Path, required=True,
                        help='Q01 root, exactly matching protocol.reference_root')
    parser.add_argument('--gpus', default='0,1', help='Comma-separated physical GPU indices')
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--preflight-min-free-mib', type=int, default=24576,
                        help='Initial CUDA preflight admission floor; at least 16384 MiB')
    args = parser.parse_args(argv)
    Controller(args.output, after_root=args.after_root,
               gpus=[int(value) for value in args.gpus.split(',')], poll_seconds=args.poll_seconds,
               preflight_min_free_mib=args.preflight_min_free_mib).run()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'run_region_normalization: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
