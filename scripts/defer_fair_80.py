"""Queue the separate 80-epoch, effective-batch-16 fair comparison.

The existing 50-epoch fair suite is immutable. After its nine tests finish,
this controller prepares a separate plan, records and protocols; runs a two-update
GPU preflight per architecture; then starts independent GPU
workers. GPU 0 runs shared/task_id and GPU 1 runs region. Both workers finish
all training before either opens the held-out test split.

The physical batch is four images. Four microbatches accumulate gradients for
one effective batch of 16 images, identically for all three architectures.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import fair_parallel as parallel
from .defer_abdomen_recovery import CODE_ROOT, fair_state, write_status


SCHEMA = 'fair_parallel_80_accum_v1'
TRAINER = 'unet_moe.fair_suite_accum'
EPOCHS = 80
EFFECTIVE_BATCH = 16
MICRO_BATCH = 4
IMAGES_PER_EPOCH = 4500
MINIMUM_FREE_MIB = 22 * 1024
PREFLIGHT_MINIMUM_FREE_MIB = 34 * 1024
PREFLIGHT_HEADROOM_MIB = 2048
DEFAULT_OUTPUT = Path('/ssd1/data/Multi-Organ_Foundation_Model/runs/fair_80_base24_batch16_512_20260930')
SOURCE_NAMES = (*parallel.SOURCE_NAMES, 'fair_suite_accum.py')


def suite_arguments(data_root, output, models):
    """Every argument that defines the new frozen training protocol."""
    return ['--data-root', str(Path(data_root).resolve()),
            '--output', str(Path(output).resolve()),
            '--models', ','.join(models), '--seeds', '42,43,44',
            '--split-seed', '42', '--size', '512', '--base', '24',
            '--batch-size', str(EFFECTIVE_BATCH), '--micro-batch-size', str(MICRO_BATCH),
            '--epochs', str(EPOCHS), '--top-k', '1', '--experts', '9',
            '--quota-per-domain', '500', '--lr', '0.0001',
            '--weight-decay', '0.0001', '--lr-schedule', 'constant',
            '--min-lr-ratio', '0.1', '--workers', '0',
            '--coarse-weight', '0', '--route-weight', '0',
            '--balance-weight', '0', '--router-z-weight', '0',
            '--empty-weight', '0', '--empty-fraction', '0.01',
            '--device', 'cuda:0']


def expected_budget(models):
    return dict(images_per_epoch=IMAGES_PER_EPOCH,
                optimizer_steps_per_epoch=math.ceil(IMAGES_PER_EPOCH / EFFECTIVE_BATCH),
                physical_micro_batches_per_epoch=math.ceil(IMAGES_PER_EPOCH / MICRO_BATCH),
                planned_epochs=EPOCHS,
                planned_images_per_run=IMAGES_PER_EPOCH * EPOCHS,
                seeds=list(parallel.SEEDS), models=list(models))


def build_plan(output, data_root):
    from unet_moe import fair_suite_accum
    output = Path(output).resolve()
    arguments = suite_arguments(data_root, output, parallel.MODELS)
    parsed = fair_suite_accum.parse_args(arguments)
    config = fair_suite_accum._configuration(parsed)
    config.pop('models')
    sources = {name: hashlib.sha256((CODE_ROOT / 'unet_moe' / name).read_bytes()).hexdigest()
               for name in SOURCE_NAMES}
    preflight_source = hashlib.sha256((CODE_ROOT / 'scripts' / 'preflight_fair80.py').read_bytes()).hexdigest()
    plan = dict(schema=SCHEMA, output=str(output), common_config=config,
                source_files=sources, source_sha256=parallel.digest(sources),
                preflight_source_sha256=preflight_source,
                budget=expected_budget(parallel.MODELS),
                minimum_free_mib=MINIMUM_FREE_MIB,
                preflight_minimum_free_mib=PREFLIGHT_MINIMUM_FREE_MIB,
                preflight_headroom_mib=PREFLIGHT_HEADROOM_MIB,
                peer_poll_seconds=parallel.POLL_SECONDS,
                trainer=TRAINER,
                test_gate='all nine runs complete all 80 epochs before either shard evaluates test',
                timing_interpretation='GPU sharing and different devices confound architecture timing comparisons',
                shards={gpu: dict(models=models, relative_output=f'shards/gpu{gpu}',
                                  budget=expected_budget(models))
                        for gpu, models in parallel.PARTITIONS.items()})
    plan['plan_sha256'] = parallel.digest(plan)
    return plan


def validate_plan(plan):
    unsigned = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    if plan.get('plan_sha256') != parallel.digest(unsigned):
        raise ValueError('80-epoch plan hash mismatch')
    if (plan.get('schema') != SCHEMA or plan.get('trainer') != TRAINER
            or plan.get('budget') != expected_budget(parallel.MODELS)
            or set(plan.get('shards', {})) != set(parallel.PARTITIONS)
            or plan.get('minimum_free_mib') != MINIMUM_FREE_MIB
            or plan.get('preflight_minimum_free_mib') != PREFLIGHT_MINIMUM_FREE_MIB
            or plan.get('preflight_headroom_mib') != PREFLIGHT_HEADROOM_MIB
            or plan.get('peer_poll_seconds') != parallel.POLL_SECONDS):
        raise ValueError('80-epoch plan has an invalid fixed budget or worker partition')
    if (set(plan.get('source_files', {})) != set(SOURCE_NAMES)
            or plan.get('source_sha256') != parallel.digest(plan['source_files'])
            or not isinstance(plan.get('preflight_source_sha256'), str)
            or len(plan['preflight_source_sha256']) != 64):
        raise ValueError('80-epoch plan source hashes are invalid')
    for gpu, models in parallel.PARTITIONS.items():
        if plan['shards'][gpu] != dict(models=models,
                                       relative_output=f'shards/gpu{gpu}',
                                       budget=expected_budget(models)):
            raise ValueError(f'Invalid GPU {gpu} model partition')
    config = plan['common_config']
    fixed = dict(seeds=parallel.SEEDS, split_seed=42, epochs=EPOCHS,
                 quota_per_domain=500, size=512, base=24,
                 batch_size=EFFECTIVE_BATCH, micro_batch_size=MICRO_BATCH,
                 effective_batch_size=EFFECTIVE_BATCH,
                 physical_micro_batch_size=MICRO_BATCH, workers=0,
                 top_k=1, experts=9, lr=.0001, weight_decay=.0001,
                 lr_schedule='constant', min_lr_ratio=.1,
                 coarse_weight=0., route_weight=0., balance_weight=0.,
                 router_z_weight=0., empty_weight=0., empty_fraction=.01,
                 device='cuda:0', abdomen_split='ct7_2', split_policy='trainmore_v6',
                 labels='paper5', augmentation='none',
                 protocol_version='fair_image_matched_accum_v1')
    if any(config.get(key) != value for key, value in fixed.items()) or 'models' in config:
        raise ValueError('80-epoch plan changes the fixed training configuration')


@contextmanager
def configured_parallel():
    """Reuse the audited artifact/barrier code with process-local 80-epoch rules.

    No frozen source file is edited. The patch is active only in this process,
    and restored after the controller/worker exits (including CPU tests).
    """
    names = ('EPOCHS', 'MINIMUM_FREE_MIB', 'build_plan', 'validate_plan')
    previous = {name: getattr(parallel, name) for name in names}
    try:
        parallel.EPOCHS = EPOCHS
        parallel.MINIMUM_FREE_MIB = MINIMUM_FREE_MIB
        parallel.build_plan = build_plan
        parallel.validate_plan = validate_plan
        yield
    finally:
        for name, value in previous.items():
            setattr(parallel, name, value)


def log_command(command, path, environment, lock=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        return subprocess.run(command, cwd=CODE_ROOT, env=environment,
                              stdout=stream, stderr=subprocess.STDOUT, check=False,
                              pass_fds=() if lock is None else (lock.fileno(),)).returncode


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                PYTHONUNBUFFERED='1')


def gpu_environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                PYTHONUNBUFFERED='1')


@contextmanager
def handle_termination():
    """Unwind managed subprocesses on SIGTERM just as on other exceptions."""
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise InterruptedError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def wait_workers(children, poll_seconds):
    """Observe both workers, including a peer still waiting for its GPU lock."""
    while True:
        codes = {gpu: process.poll() for gpu, process in children.items()}
        if any(code is not None and code != 0 for code in codes.values()):
            raise RuntimeError(f'fair80 GPU workers exited {codes}')
        if all(code is not None for code in codes.values()):
            return codes
        time.sleep(min(poll_seconds, parallel.POLL_SECONDS))


def stop_workers(children, timeout=15):
    """Stop each owned worker group, including any trainer left by a dead worker.

    Workers are session leaders; their trainers deliberately share that process
    group. Send TERM to every group before waiting so cleanup is concurrent.
    """
    for process in children.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for process in children.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A worker can die before its trainer. Always address the owned group,
    # even when poll()/wait() has already reaped the group leader.
    for process in children.values():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {gpu: process.wait() for gpu, process in children.items()}


def _valid_prepared_shard(plan, gpu):
    folder = parallel.shard_path(plan, gpu)
    protocol = parallel.read_json(folder / 'protocol.json')
    parallel.validate_protocol(plan, gpu, protocol)
    if parallel.digest(parallel.read_json(folder / 'records.json')) != protocol['record_sha256']:
        raise ValueError(f'GPU {gpu} records.json differs from the frozen protocol')
    return protocol


def prepare_shards(plan, old_fair_root, log_path):
    """Prepare metadata on CPU; reject any change of records versus old fair."""
    old_summary = parallel.read_json(Path(old_fair_root) / 'summary.json')
    expected_records = old_summary['record_sha256']
    protocols = {}
    for gpu, models in parallel.PARTITIONS.items():
        folder = parallel.shard_path(plan, gpu)
        if not folder.exists():
            command = [sys.executable, '-m', TRAINER,
                       *suite_arguments(plan['common_config']['data_root'], folder, models),
                       '--dry-run']
            if log_command(command, log_path, cpu_environment()):
                raise RuntimeError(f'GPU {gpu} dry-run preparation failed; see {log_path}')
        protocols[gpu] = _valid_prepared_shard(plan, gpu)
        if protocols[gpu]['record_sha256'] != expected_records:
            raise ValueError(f'GPU {gpu} new records differ from the completed 50-epoch data protocol')
    if len({entry['record_sha256'] for entry in protocols.values()}) != 1:
        raise ValueError('New GPU shard records differ')
    return protocols


def report_path(plan, kind):
    return Path(plan['output']) / 'preflight' / f'{kind}.json'


def _validate_report(plan, kind, report, protocols):
    gpu = '1' if kind == 'region' else '0'
    protocol = protocols[gpu]
    selection = report.get('selection', {})
    if (report.get('schema') != 'fair80_worst_batch_preflight_v1'
            or report.get('status') != 'passed' or report.get('kind') != kind
            or report.get('record_sha256') != protocol['record_sha256']
            or report.get('source_sha256') != protocol['source_sha256']
            or report.get('effective_batch_size') != EFFECTIVE_BATCH
            or report.get('physical_micro_batch_size') != MICRO_BATCH
            or report.get('optimizer_step_ok') is not True
            or not isinstance(report.get('optimizer_steps'), int)
            or report['optimizer_steps'] < 2
            or not 1 <= selection.get('images', 0) <= EFFECTIVE_BATCH
            or selection.get('max_microbatch_queries', 0) < 1):
        raise ValueError(f'{kind} GPU preflight report is incomplete or belongs to another protocol')
    peak = report.get('peak_reserved_mib')
    if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        raise ValueError(f'{kind} GPU preflight report has no peak reserved memory')
    total = report.get('gpu_total_mib')
    if not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
        raise ValueError(f'{kind} GPU preflight report has no device memory capacity')
    required = max(MINIMUM_FREE_MIB, math.ceil(peak) + PREFLIGHT_HEADROOM_MIB)
    if required > total:
        raise ValueError(f'{kind} GPU preflight needs {required} MiB including headroom, '
                         f'exceeding the GPU capacity of {total} MiB')
    return required


def preflight_marker(plan, protocols):
    reports = {}
    thresholds = {}
    for kind in parallel.MODELS:
        path = report_path(plan, kind)
        report = parallel.read_json(path)
        gpu = '1' if kind == 'region' else '0'
        thresholds[gpu] = max(thresholds.get(gpu, MINIMUM_FREE_MIB),
                              _validate_report(plan, kind, report, protocols))
        reports[kind] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(schema='fair80_preflight_gate_v1', plan_sha256=plan['plan_sha256'],
                record_sha256=protocols['0']['record_sha256'],
                source_sha256=plan['source_sha256'], report_sha256=reports,
                minimum_free_mib_by_gpu=thresholds)


def validate_preflight(plan):
    protocols = {gpu: _valid_prepared_shard(plan, gpu) for gpu in parallel.PARTITIONS}
    expected = preflight_marker(plan, protocols)
    path = Path(plan['output']) / 'preflight_complete.json'
    if parallel.read_json(path) != expected:
        raise ValueError('GPU preflight gate differs from its plan or reports')
    return expected


def gpu_free_mib(gpu):
    output = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
         '--format=csv,noheader,nounits'], text=True).strip()
    return int(output)


def wait_for_gpu_lock(gpu, required_mib, poll_seconds, status_path, state):
    """Yield a live GPU lock only when the admission threshold is met."""
    path = Path(f'/tmp/multi_organ_fair_gpu_{gpu}.lock')
    while True:
        with path.open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                free = None
            else:
                free = gpu_free_mib(gpu)
                if free >= required_mib:
                    yield lock, free
                    return
        write_status(status_path, state, gpu=int(gpu), minimum_free_mib=required_mib,
                     observed_free_mib=free)
        time.sleep(poll_seconds)


def run_preflights(plan, old_fair_root, poll_seconds, status_path, log_path):
    protocols = {gpu: _valid_prepared_shard(plan, gpu) for gpu in parallel.PARTITIONS}
    gate_path = Path(plan['output']) / 'preflight_complete.json'
    if gate_path.exists():
        return validate_preflight(plan)
    for kind in parallel.MODELS:
        path = report_path(plan, kind)
        if path.exists():
            _validate_report(plan, kind, parallel.read_json(path), protocols)
            continue
        gpu = '1' if kind == 'region' else '0'
        for lock, free in wait_for_gpu_lock(gpu, PREFLIGHT_MINIMUM_FREE_MIB, poll_seconds,
                                            status_path, 'waiting_gpu_preflight'):
            command = [sys.executable, '-m', 'scripts.preflight_fair80',
                       '--records', str(parallel.shard_path(plan, gpu) / 'records.json'),
                       '--kind', kind, '--device', 'cuda:0',
                       '--micro-batch-size', str(MICRO_BATCH), '--output', str(path)]
            write_status(status_path, 'preflight', old_fair_root=str(old_fair_root),
                         kind=kind, gpu=int(gpu), gpu_free_mib_at_start=free,
                         report=str(path), command=command, log=str(log_path))
            if log_command(command, log_path, gpu_environment(gpu), lock):
                raise RuntimeError(f'{kind} preflight failed; see {path} and {log_path}')
            _validate_report(plan, kind, parallel.read_json(path), protocols)
    marker = preflight_marker(plan, protocols)
    parallel.write_json(gate_path, marker)
    return validate_preflight(plan)


class Worker80(parallel.Worker):
    def event(self, message, **extra):
        super().event(message.replace('50 epochs', '80 epochs'), **extra)

    def run_suite(self, evaluate_test=False):
        folder = parallel.shard_path(self.plan, self.gpu)
        arguments = suite_arguments(self.plan['common_config']['data_root'], folder,
                                    self.plan['shards'][self.gpu]['models'])
        if not (folder / 'protocol.json').is_file():
            raise FileNotFoundError(f'Prepared shard protocol missing: {folder}')
        parallel.validate_protocol(self.plan, self.gpu,
                                   parallel.read_json(folder / 'protocol.json'))
        arguments.append('--resume')
        if evaluate_test:
            arguments.append('--evaluate-test')
        command = [sys.executable, '-m', TRAINER, *arguments]
        environment = gpu_environment(self.gpu)
        self.event('Starting fair80 accumulated-batch trainer', command=command)
        with self.log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=CODE_ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       # The trainer shares the worker's dedicated
                                       # group so controller cleanup also reaches it.
                                       start_new_session=False,
                                       pass_fds=(self.lock.fileno(),))
            try:
                self.update('evaluating_test' if evaluate_test else 'training', child_pid=process.pid)
                result = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
        self.status.pop('child_pid', None)
        if result:
            raise RuntimeError(f'{TRAINER} exited {result}; see {self.log_path}')


def worker_main(args):
    with configured_parallel():
        plan = parallel.load_plan(args.output, args.data_root)
        gate = validate_preflight(plan)
        gpu = str(args.worker)
        required = gate['minimum_free_mib_by_gpu'][gpu]
        # The inherited worker checks memory again before the held-out test.
        # Use the measured preflight threshold at both admission points.
        parallel.MINIMUM_FREE_MIB = required
        status_path = Path(plan['output']) / f'worker_gpu{gpu}_admission.json'
        for lock, free in wait_for_gpu_lock(gpu, required, args.poll_seconds,
                                            status_path, 'waiting_gpu'):
            lock.seek(0)
            lock.truncate()
            lock.write(json.dumps(dict(pid=os.getpid(), gpu=gpu,
                                       output=plan['output'], minimum_free_mib=required)) + '\n')
            lock.flush()
            worker = Worker80(plan, gpu, lock)
            worker.status['preflight_minimum_free_mib'] = required
            worker.status['gpu_free_mib_at_admission'] = free
            previous = signal.getsignal(signal.SIGTERM)
            def interrupted(signum, frame):
                raise InterruptedError(f'Worker received signal {signum}')
            signal.signal(signal.SIGTERM, interrupted)
            try:
                worker.run()
            except parallel.PeerFailedError as error:
                worker.update('blocked_by_peer', error=str(error))
                worker.event('Worker blocked by peer failure; restart after peer recovers', error=str(error))
                raise
            except BaseException as error:
                failure = dict(gpu=gpu, plan_sha256=plan['plan_sha256'],
                               time=parallel.timestamp(),
                               error=f'{type(error).__name__}: {error}', worker_pid=os.getpid())
                parallel.write_json(worker.failure_path(gpu), failure)
                worker.update('failed', error=failure['error'])
                worker.event('Worker failed', error=failure['error'])
                raise
            finally:
                signal.signal(signal.SIGTERM, previous)


def complete_root(plan):
    root = Path(plan['output'])
    if not (root / 'summary.json').is_file():
        return False
    with configured_parallel():
        metadata = {gpu: parallel.validate_marker(plan, gpu, 'test')
                    for gpu in parallel.PARTITIONS}
        expected = parallel.aggregate_shards(plan, metadata)
    stored = parallel.read_json(root / 'summary.json')
    if any(stored.get(key) != value for key, value in expected.items()
           if key != 'parallel'):
        raise ValueError('Root fair80 summary differs from validated shards')
    if (stored.get('parallel', {}).get('plan_sha256') != plan['plan_sha256']
            or stored.get('parallel', {}).get('training_barrier_verified') is not True
            or any(parallel.read_json(root / f'worker_gpu{gpu}.json').get('state') != 'complete'
                   for gpu in parallel.PARTITIONS)):
        raise ValueError('Root fair80 summary lacks the complete two-GPU test barrier')
    return True


def controller_main(args):
    output = args.output.resolve()
    controller = output.parent / f'{output.name}-logs'
    controller.mkdir(parents=True, exist_ok=True)
    status_path = controller / 'controller_status.json'
    log_path = controller / 'controller.log'
    with (controller / '.controller.lock').open('a+') as controller_lock, handle_termination():
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('fair80 controller is already running') from error
        children = {}
        write_status(status_path, 'waiting_fair', old_fair_root=str(args.fair_root.resolve()),
                     output=str(output), log=str(log_path))
        try:
            while True:
                state = fair_state(args.fair_root.resolve(), args.research_status.resolve())
                if state == 'ready':
                    break
                if state != 'waiting':
                    raise RuntimeError(state)
                time.sleep(args.poll_seconds)
            with configured_parallel():
                write_status(status_path, 'preparing', old_fair_root=str(args.fair_root.resolve()),
                             output=str(output), log=str(log_path))
                plan = parallel.prepare_plan(output, args.data_root)
                prepare_shards(plan, args.fair_root, log_path)
            write_status(status_path, 'preflighting', output=str(output), log=str(log_path))
            run_preflights(plan, args.fair_root, args.poll_seconds, status_path, log_path)
            if complete_root(plan):
                write_status(status_path, 'complete', output=str(output),
                             summary=str(output / 'summary.json'), reused=True)
                return
            for gpu in (0, 1):
                command = [sys.executable, '-m', 'scripts.defer_fair_80',
                           '--output', str(output), '--data-root', str(args.data_root.resolve()),
                           '--worker', str(gpu), '--poll-seconds', str(args.poll_seconds)]
                with log_path.open('a') as stream:
                    process = subprocess.Popen(command, cwd=CODE_ROOT, env=cpu_environment(),
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                children[gpu] = process
            write_status(status_path, 'running', output=str(output),
                         worker_pids={str(gpu): process.pid for gpu, process in children.items()},
                         log=str(log_path))
            codes = wait_workers(children, args.poll_seconds)
            if not complete_root(plan):
                raise ValueError('fair80 workers exited without a validated nine-run test summary')
            write_status(status_path, 'complete', output=str(output),
                         summary=str(output / 'summary.json'), worker_exit_codes=codes,
                         log=str(log_path))
        except BaseException as error:
            try:
                write_status(status_path, 'failed', output=str(output), log=str(log_path),
                             error=f'{type(error).__name__}: {error}')
            finally:
                # Covers partial startup and cancellation without waiting for a
                # worker that has not acquired its GPU lock yet.
                stop_workers(children)
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path,
                        help='Completed existing 50-epoch fair root; required for controller')
    parser.add_argument('--research-status', type=Path,
                        help='Existing research status.json; required for controller')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--worker', type=int, choices=(0, 1),
                        help=argparse.SUPPRESS)
    parser.add_argument('--poll-seconds', type=int, default=45)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error('--poll-seconds must be positive')
    if args.worker is None and (args.fair_root is None or args.research_status is None):
        parser.error('--fair-root and --research-status are required')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker is None:
        controller_main(args)
    else:
        worker_main(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'defer_fair_80: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
