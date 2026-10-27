"""Run the frozen Q01 matrix with shared GPU locks and a global test barrier.

Launch from the experiment's source snapshot, for example::

    python ROOT/source_snapshot/scripts/run_organ_weighted.py --output ROOT

The same command resumes after interruption. Run validation is delegated to the
suite's CPU ``--verify-run`` command, so a summary alone never proves completion.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


CODE_ROOT = Path(__file__).resolve().parents[1]
SUITE_PREFIX = [sys.executable, '-m', 'unet_moe.organ_weighted_suite']
MODELS = ('shared', 'region', 'task_id')
ARMS = ('image_mean', 'organ_exposure', 'organ_exposure_fg_bg')
SEEDS = (42, 43, 44)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path):
    # Closing the parent descriptor does not unlock an orphan child's inherited
    # descriptor; never explicitly LOCK_UN before the child has stopped.
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another controller already owns lock {path}') from error
        yield stream


def gpu_lock_path(gpu):
    return Path(f'/tmp/multi_organ_fair_gpu_{gpu}.lock')


def gpu_free_mib(gpu):
    value = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
         '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    return int(value)


def terminate_children(processes, timeout=10):
    processes = [process for process in processes if process.poll() is None]
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


class Controller:
    def __init__(self, output, gpus=(0, 1), poll_seconds=15):
        self.root = Path(output).resolve()
        self.gpus = list(gpus)
        if (not self.gpus or len(self.gpus) != len(set(self.gpus))
                or any(type(gpu) is not int or gpu < 0 for gpu in self.gpus)):
            raise ValueError('GPU indices must be distinct nonnegative integers')
        if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
            raise ValueError('poll_seconds must be positive and at most 60')
        self.poll_seconds = poll_seconds
        self.protocol = read_json(self.root / 'protocol.json')
        self.protocol_sha256 = self.protocol.get('protocol_sha256')
        if (not isinstance(self.protocol_sha256, str) or len(self.protocol_sha256) != 64
                or any(char not in '0123456789abcdef' for char in self.protocol_sha256)
                or self.protocol.get('config', {}).get('epochs') != 80):
            raise ValueError('Q01 needs a frozen 80-epoch protocol with its SHA-256')
        preflight = read_json(self.root / 'preflight.json')
        if (preflight.get('protocol_sha256') != self.protocol_sha256
                or set(preflight.get('models', {})) != set(MODELS)):
            raise ValueError('GPU preflight does not match the frozen Q01 protocol/models')
        self.required_mib = {}
        for kind in MODELS:
            report = preflight['models'][kind]
            required = report.get('required_free_mib')
            if report.get('passed') is not True or type(required) is not int or required <= 0:
                raise ValueError(f'Invalid or failed preflight for {kind}')
            self.required_mib[kind] = required
        # Model/seed trios remain adjacent. The first eligible pending job is
        # claimed by either GPU, avoiding fixed partitions with idle peers.
        self.jobs = [dict(kind=kind, seed=seed, arm=arm, train_state='pending',
                          test_state='pending', pid=None, gpu=None)
                     for seed in SEEDS for kind in MODELS for arm in ARMS]
        self.active = {}
        self.control_child = None
        self.root_lock = None
        self.status = dict(schema='organ_weighted_controller_v1', state='starting',
                           output=str(self.root), protocol_sha256=self.protocol_sha256,
                           controller_pid=os.getpid(), started_at=timestamp(),
                           gpu_indices=self.gpus, required_free_mib=self.required_mib,
                           jobs=self.jobs, gpu_observations={})

    def update(self, state=None, **extra):
        if state is not None:
            self.status['state'] = state
        self.status.update(updated_at=timestamp(), **extra)
        write_json(self.root / 'controller_status.json', self.status)

    def folder(self, job):
        return self.root / job['arm'] / job['kind'] / f'seed_{job["seed"]}'

    def arguments(self, job):
        return ['--output', str(self.root), '--kind', job['kind'],
                '--seed', str(job['seed']), '--arm', job['arm']]

    def run_cpu(self, arguments, label):
        path = self.root / 'logs' / 'verification.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [*SUITE_PREFIX, '--output', str(self.root), *arguments]
        with path.open('a') as log:
            log.write(json.dumps(dict(time=timestamp(), operation=label, command=command)) + '\n')
            log.flush()
            self.control_child = subprocess.Popen(
                command, cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
                     'PYTHONUNBUFFERED': '1'}, start_new_session=True,
                pass_fds=(self.root_lock.fileno(),))
            self.update(control_child_pid=self.control_child.pid)
            result = self.control_child.wait()
            self.control_child = None
            self.update(control_child_pid=None)
        if result:
            raise RuntimeError(f'{label} exited {result}; see {path}')

    def verify(self, job, require_test=False):
        arguments = self.arguments(job)[2:] + ['--verify-run']
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
            raise ValueError('Existing test results violate the all-27-runs training barrier')

    def admit(self, gpu, pending):
        lock = gpu_lock_path(gpu).open('a+')
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.status['gpu_observations'][str(gpu)] = dict(state='locked', time=timestamp())
                lock.close()
                return None
            free = gpu_free_mib(gpu)
            self.status['gpu_observations'][str(gpu)] = dict(
                state='available', free_mib=free, time=timestamp())
            for job in pending:
                if free >= self.required_mib[job['kind']]:
                    lock.seek(0)
                    lock.truncate()
                    lock.write(json.dumps(dict(pid=os.getpid(), output=str(self.root), gpu=gpu,
                                               kind=job['kind'], arm=job['arm'], seed=job['seed'])) + '\n')
                    lock.flush()
                    return job, lock, free
            self.status['gpu_observations'][str(gpu)]['state'] = 'waiting_for_memory'
            lock.close()
            return None
        except BaseException:
            lock.close()
            raise

    def launch(self, gpu, job, lock, free, phase):
        command = [*SUITE_PREFIX, *self.arguments(job), '--device', 'cuda:0']
        folder = self.folder(job)
        if folder.exists() and any(folder.iterdir()):
            command.append('--resume')
        if phase == 'test':
            if '--resume' not in command:
                raise ValueError('Test evaluation requires existing training artifacts')
            command.append('--evaluate-test')
        path = self.root / 'logs' / f'gpu{gpu}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open('a')
        try:
            log.write(json.dumps(dict(time=timestamp(), phase=phase, command=command,
                                     free_mib_at_admission=free)) + '\n')
            log.flush()
            process = subprocess.Popen(command, cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
                                            'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'},
                                       start_new_session=True,
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
        runs = summary.get('runs', [])
        expected = {(job['kind'], job['seed'], job['arm']) for job in self.jobs}
        actual = {(run.get('kind'), run.get('seed'), run.get('arm')) for run in runs}
        if (summary.get('protocol_sha256') != self.protocol_sha256
                or summary.get('complete') is not True or len(runs) != 27 or actual != expected
                or any(run.get('protocol_sha256') != self.protocol_sha256
                       or run.get('epochs_completed') != 80 or run.get('test_evaluated') is not True
                       for run in runs)):
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
                self.update('recovering')
                self.recover()
                self.run_phase('train')
                # Every completion came from full suite verification; a resumed
                # job cannot open this barrier using just run_summary.json.
                if any(job['train_state'] != 'complete' for job in self.jobs):
                    raise RuntimeError('Training barrier incomplete')
                self.update('training_barrier_verified', training_barrier_verified_at=timestamp())
                self.run_phase('test')
                if any(job['test_state'] != 'complete' for job in self.jobs):
                    raise RuntimeError('Test matrix incomplete')
                self.update('summarizing')
                self.run_cpu(['--summarize'], 'paired summary')
                self.validate_summary()
                self.update('complete', completed_at=timestamp(), waiting_jobs=0,
                            summary=str(self.root / 'summary.json'))
            except BaseException as error:
                # Prevent a second signal from interrupting peer shutdown.
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
    parser.add_argument('--gpus', default='0,1', help='Comma-separated physical GPU indices')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args(argv)
    gpus = [int(value) for value in args.gpus.split(',')]
    Controller(args.output, gpus=gpus, poll_seconds=args.poll_seconds).run()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'run_organ_weighted: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
