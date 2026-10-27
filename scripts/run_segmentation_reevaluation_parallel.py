"""Additive dual-GPU scheduler for an unchanged frozen medical evaluator.

Freeze this file and its manifest outside ``source_snapshot``. Every inference
worker still executes the original evaluator, original protocol and job scope.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def durable_lock(path):
    """Close only: an inherited worker descriptor must keep an orphan locked."""
    handle = Path(path).open('a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle
    finally:
        handle.close()


def load_original(root, manifest_path):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    manifest = read(manifest_path)
    scheduler = Path(__file__).resolve()
    worker = root / 'source_snapshot/scripts/run_segmentation_reevaluation.py'
    if scheduler.parent != manifest_path.parent:
        raise ValueError('Run the scheduler snapshot beside its manifest')
    if manifest.get('scheduler_file_sha256') != file_sha(scheduler):
        raise ValueError('Scheduler snapshot changed')
    if manifest.get('worker_file_sha256') != file_sha(worker):
        raise ValueError('Original frozen worker changed')
    protocol = read(root / 'protocol.json')
    if manifest.get('evaluation_protocol_sha256') != protocol.get('protocol_sha256'):
        raise ValueError('Scheduler belongs to another evaluation protocol')
    spec = importlib.util.spec_from_file_location('_frozen_medical_reevaluation', worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The original validator also checks the complete source manifest and scope.
    module.load_protocol(root)
    return module, manifest, worker


class Controller:
    def __init__(self, root, original, manifest, worker, gpus=(0, 1), poll_seconds=15):
        self.root = Path(root).resolve()
        self.original, self.manifest, self.worker = original, manifest, Path(worker).resolve()
        self.gpus = list(gpus)
        if (not self.gpus or len(set(self.gpus)) != len(self.gpus)
                or any(type(gpu) is not int or gpu < 0 for gpu in self.gpus)):
            raise ValueError('GPU indices must be unique nonnegative integers')
        if self.gpus != manifest.get('gpus'):
            raise ValueError('GPU allocation differs from scheduler manifest')
        if not 1 <= poll_seconds <= 60:
            raise ValueError('Polling interval must be between 1 and 60 seconds')
        self.poll_seconds = poll_seconds
        self.base = original.Controller(self.root, gpu=self.gpus[0], poll_seconds=poll_seconds)
        self.protocol = self.base.protocol
        self.active = {}
        self.root_lock = None
        self.observations = {}

    def gpu_lock_path(self, gpu):
        return Path(f'/tmp/multi_organ_fair_gpu_{gpu}.lock')

    def gpu_memory(self, gpu):
        return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15).strip())

    def update(self, state, **extra):
        self.original.atomic_write_json(self.root / 'status.json', dict(
            state=state, updated_at=self.original.timestamp(), controller_pid=os.getpid(),
            gpu_indices=self.gpus, gpu=self.gpus[0] if len(self.gpus) == 1 else None,
            protocol_sha256=self.protocol['protocol_sha256'],
            scheduler_sha256=self.manifest['scheduler_file_sha256'],
            completed_jobs=len(self.base.completed), failed_jobs=len(self.base.failed),
            active_jobs=[dict(gpu=gpu, worker_pid=item['process'].pid,
                **{key: item['job'].get(key) for key in
                   ('job_id', 'experiment', 'kind', 'arm', 'seed', 'selection', 'split')})
                for gpu, item in self.active.items()],
            gpu_observations=self.observations, **extra))

    def inspect(self):
        """Read-only preview, including while another controller owns the root."""
        inventory = read(self.root / 'inventory.json')
        return dict(read_only=True, protocol_sha256=self.protocol['protocol_sha256'],
            scheduler_sha256=self.manifest['scheduler_file_sha256'], gpu_indices=self.gpus,
            current_status=read(self.root / 'status.json'),
            inventory_counts=inventory.get('evaluation_counts', inventory.get('counts')),
            yielding_to_training=self.original.training_needs_gpu(self.protocol['runs_root']),
            worker=str(self.worker))

    def refresh(self):
        inventory = self.base.refresh()
        return self.mark_active(inventory)

    def mark_active(self, inventory):
        active_ids = {item['job']['job_id'] for item in self.active.values()}
        for job in inventory['jobs']:
            if job['job_id'] in active_ids:
                job['evaluation_state'] = 'running'
        inventory['evaluation_counts'] = dict(Counter(j['evaluation_state'] for j in inventory['jobs']))
        self.original.atomic_write_json(self.root / 'inventory.json', inventory)
        return inventory

    def failure(self, item, code, error=None):
        path = self.root / 'jobs' / item['job']['job_id'] / 'failure.json'
        if not path.exists():
            self.original.atomic_write_json(path, dict(exit_code=code, at=self.original.timestamp(),
                log=str(item['log_path']), error=str(error) if error is not None else None))

    def reap(self):
        for gpu, item in list(self.active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            try:
                if code:
                    self.failure(item, code)
                elif not (self.root / 'jobs' / item['job']['job_id'] / 'result.json').exists():
                    self.failure(item, code, 'Worker exited successfully without a completed result')
                # Verify result fingerprints using the unchanged original validator
                # before releasing the GPU lock or declaring a successful job.
                self.base.refresh()
                if not code and item['job']['job_id'] not in self.base.completed:
                    if item['job']['job_id'] not in self.base.failed:
                        raise RuntimeError('Exited worker has no verified result or persisted failure')
            finally:
                del self.active[gpu]
                item['log'].close()
                item['lock'].close()

    def launch(self, gpu, job, lock, free_mib):
        if (gpu in self.active or any(item['job']['job_id'] == job['job_id'] for item in self.active.values())
                or job['job_id'] in self.base.completed or job['job_id'] in self.base.failed
                or job.get('evaluation_state') != 'ready'):
            raise ValueError('Job or GPU already claimed, complete, failed, or not ready')
        folder = self.root / 'jobs' / job['job_id']
        folder.mkdir(exist_ok=True)
        if (folder / 'result.json').exists() or (folder / 'failure.json').exists():
            raise ValueError('Refusing to overwrite a completed or failed job')
        job = dict(job)
        job.pop('evaluation_state', None)
        job['inference_source_candidates'] = self.protocol.get('inference_source_candidates', [])
        checkpoint = Path(job['checkpoint'])
        stat = checkpoint.stat()
        key = (str(checkpoint), stat.st_size, stat.st_mtime_ns)
        if key not in self.base.sha_cache:
            self.base.sha_cache[key] = self.original.file_sha(checkpoint)
        sha = self.base.sha_cache[key]
        if job.get('checkpoint_sha256') and job['checkpoint_sha256'] != sha:
            raise ValueError('Source checkpoint digest differs from original experiment')
        job.update(checkpoint_sha256=sha, evaluation_protocol_sha256=self.protocol['protocol_sha256'])
        job['job_manifest_sha256'] = self.original.digest(job)
        manifest = folder / 'job.json'
        self.original.atomic_write_json(manifest, job)
        path = self.root / 'logs' / f'{job["job_id"]}.log'
        log = path.open('a')
        command = [sys.executable, str(self.worker), '--output', str(self.root),
                   '--evaluate-job', str(manifest), '--device', 'cuda:0']
        try:
            process = subprocess.Popen(command, cwd=self.original.CODE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(), self.root_lock.fileno()),
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(self.original.CODE_ROOT), 'PYTHONUNBUFFERED': '1'})
        except BaseException:
            log.close()
            raise
        self.active[gpu] = dict(process=process, job=job, lock=lock, log=log, log_path=path)
        self.observations[str(gpu)] = dict(state='evaluating', free_mib_at_admission=free_mib,
                                          job_id=job['job_id'], worker_pid=process.pid)

    def admit(self, gpu, job):
        lock = self.gpu_lock_path(gpu).open('a+')
        transferred = False
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.observations[str(gpu)] = dict(state='waiting_gpu_lock')
                return False
            free = self.gpu_memory(gpu)
            if free < 18000:
                self.observations[str(gpu)] = dict(state='waiting_gpu_memory', free_mib=free)
                return False
            if self.original.training_needs_gpu(self.protocol['runs_root']):
                self.observations[str(gpu)] = dict(state='yielding_to_training')
                return False
            self.launch(gpu, job, lock, free)
            transferred = True
            return True
        finally:
            if not transferred:
                lock.close()

    def tick(self, launch=True):
        self.reap()
        inventory = self.refresh()
        pending = sorted([j for j in inventory['jobs'] if j['evaluation_state'] == 'ready'],
                         key=self.original.job_priority)
        deferred = sum(j['evaluation_state'] == 'deferred' for j in inventory['jobs'])
        yielding = self.original.training_needs_gpu(self.protocol['runs_root'])
        if launch and not yielding:
            for gpu in self.gpus:
                if gpu not in self.active and pending and self.admit(gpu, pending[0]):
                    pending.pop(0)
        self.mark_active(inventory)
        self.base.reports(inventory)
        if self.active:
            state = 'evaluating'
        elif not pending and not deferred:
            unavailable = inventory['evaluation_counts'].get('unavailable', 0)
            state = ('complete_with_failures' if self.base.failed else
                     'complete_with_unavailable' if unavailable else 'complete')
        elif yielding:
            state = 'yielding_to_training'
        elif not pending:
            state = 'waiting_training_barriers'
        else:
            state = 'waiting_gpu_admission'
        self.update(state, counts=inventory['evaluation_counts'], pending_jobs=len(pending))
        return state.startswith('complete')

    def stop(self):
        # These Popen children have private process groups. Never signal a
        # training controller, existing evaluator, or arbitrary recorded PID.
        for item in self.active.values():
            if item['process'].poll() is None:
                try:
                    os.killpg(item['process'].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        for item in self.active.values():
            try:
                item['process'].wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(item['process'].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                item['process'].wait()
            item['log'].close()
            item['lock'].close()
        self.active.clear()

    def run(self, once=False):
        with durable_lock(self.root / '.controller.lock') as root_lock:
            self.root_lock = root_lock
            handlers = {}

            def interrupted(signum, frame):
                raise InterruptedError(f'Scheduler received signal {signum}')

            try:
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, interrupted)
                while True:
                    complete = self.tick(launch=not once)
                    if once or complete:
                        return 0
                    time.sleep(self.poll_seconds)
            except BaseException as error:
                self.stop()
                self.update('interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed',
                            error=f'{type(error).__name__}: {error}')
                raise
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
                self.root_lock = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--poll-seconds', type=int, default=15)
    parser.add_argument('--scheduler-manifest', type=Path, default=Path(__file__).resolve().with_name('manifest.json'))
    parser.add_argument('--dry-run', action='store_true', help='Read-only preview; safe while another controller runs')
    parser.add_argument('--once', action='store_true', help='Take root lock, refresh verified reports once, never launch a worker')
    args = parser.parse_args(argv)
    module, manifest, worker = load_original(args.output, args.scheduler_manifest)
    controller = Controller(args.output, module, manifest, worker,
                            gpus=[int(value) for value in args.gpus.split(',')], poll_seconds=args.poll_seconds)
    if args.dry_run:
        print(json.dumps(controller.inspect(), ensure_ascii=False, indent=2))
        return 0
    return controller.run(once=args.once)


if __name__ == '__main__':
    raise SystemExit(main())
