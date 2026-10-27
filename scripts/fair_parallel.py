"""Fixed two-GPU controller for the frozen fair-suite protocol.

Prepare once, then launch each GPU independently (no GPU is started implicitly)::

    python -m scripts.fair_parallel --output ROOT --prepare
    python -m scripts.fair_parallel --output ROOT --gpu 0
    python -m scripts.fair_parallel --output ROOT --gpu 1

GPU 0 runs shared/task_id; GPU 1 runs region, each for seeds 42,43,44.
Training must finish on both GPUs before either worker evaluates test images.
Full subprocess logs are appended to ROOT/logs/gpuN.log. Workers can be restarted
with the same command; all existing artifacts must match the frozen plan.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = '/ssd1/data/Multi-Organ_Foundation_Model/pointed_data'
SOURCE_NAMES = ('fair_suite.py', 'baseline.py', 'model.py', 'region.py', 'normalization.py', 'data.py',
                'sampling.py', 'schedule.py', 'empty_loss.py')
PARTITIONS = {'0': ['shared', 'task_id'], '1': ['region']}
MODELS = ['shared', 'task_id', 'region']
SEEDS = [42, 43, 44]
EPOCHS = 50
IMAGES_PER_EPOCH = 4500
MINIMUM_FREE_MIB = 22 * 1024
POLL_SECONDS = 15


class PeerFailedError(RuntimeError):
    """A peer failure blocks this worker without creating a cascading failure."""



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Commit a complete JSON file atomically, including across worker processes."""
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
def exclusive_lock(path, blocking=False):
    """Keep the descriptor open; children inherit it to prevent orphan races."""
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f'Another controller already owns lock {path}') from error
        yield stream


def suite_arguments(data_root, output, models):
    return ['--data-root', str(Path(data_root).resolve()), '--output', str(Path(output).resolve()),
            '--models', ','.join(models), '--seeds', '42,43,44', '--split-seed', '42',
            '--size', '512', '--base', '16', '--batch-size', '4', '--epochs', '50',
            '--top-k', '1', '--experts', '9', '--quota-per-domain', '500', '--lr', '0.0003',
            '--weight-decay', '0.0001', '--lr-schedule', 'constant', '--min-lr-ratio', '0.1',
            '--workers', '2', '--coarse-weight', '0', '--route-weight', '0',
            '--balance-weight', '0', '--router-z-weight', '0', '--empty-weight', '0',
            '--empty-fraction', '0.01', '--device', 'cuda:0']


def expected_budget(models):
    return dict(images_per_epoch=IMAGES_PER_EPOCH, optimizer_steps_per_epoch=1125,
                planned_epochs=EPOCHS, planned_images_per_run=225000,
                seeds=list(SEEDS), models=list(models))


def build_plan(output, data_root):
    # Parse the existing CLI so derived objective/split/optimizer fields exactly
    # match fair_suite, without creating protocol files or accessing image data.
    from unet_moe import fair_suite
    output = Path(output).resolve()
    args = fair_suite.parse_args(suite_arguments(data_root, output, MODELS))
    common_config = fair_suite._configuration(args)
    common_config.pop('models')
    sources = {name: hashlib.sha256((CODE_ROOT / 'unet_moe' / name).read_bytes()).hexdigest()
               for name in SOURCE_NAMES}
    plan = dict(schema='fair_parallel_v1', output=str(output), common_config=common_config,
                source_files=sources, source_sha256=digest(sources),
                budget=expected_budget(MODELS), minimum_free_mib=MINIMUM_FREE_MIB,
                peer_poll_seconds=POLL_SECONDS,
                test_gate='all nine runs complete all 50 epochs before either shard evaluates test',
                timing_interpretation='GPU sharing and different devices confound architecture timing comparisons',
                shards={gpu: dict(models=models, relative_output=f'shards/gpu{gpu}',
                                  budget=expected_budget(models)) for gpu, models in PARTITIONS.items()})
    plan['plan_sha256'] = digest(plan)
    return plan


def validate_plan(plan):
    unsigned = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    if plan.get('plan_sha256') != digest(unsigned):
        raise ValueError('Parallel plan hash mismatch')
    if (plan.get('schema') != 'fair_parallel_v1' or plan.get('budget') != expected_budget(MODELS)
            or set(plan.get('shards', {})) != set(PARTITIONS)
            or plan.get('minimum_free_mib') != MINIMUM_FREE_MIB
            or plan.get('peer_poll_seconds') != POLL_SECONDS):
        raise ValueError('Parallel plan has an invalid fixed budget or worker partition')
    for gpu, models in PARTITIONS.items():
        if plan['shards'][gpu] != dict(models=models, relative_output=f'shards/gpu{gpu}',
                                      budget=expected_budget(models)):
            raise ValueError(f'Parallel plan has an invalid GPU {gpu} model partition')
    config = plan['common_config']
    fixed = dict(seeds=SEEDS, split_seed=42, epochs=50, quota_per_domain=500, size=512,
                 base=16, batch_size=4, workers=2, top_k=1, experts=9, lr=.0003,
                 weight_decay=.0001, lr_schedule='constant', min_lr_ratio=.1,
                 coarse_weight=0., route_weight=0., balance_weight=0., router_z_weight=0.,
                 empty_weight=0., empty_fraction=.01, device='cuda:0')
    if any(config.get(key) != value for key, value in fixed.items()) or 'models' in config:
        raise ValueError('Parallel plan changes the fixed training configuration')
    if (set(plan.get('source_files', {})) != set(SOURCE_NAMES)
            or plan.get('source_sha256') != digest(plan['source_files'])):
        raise ValueError('Parallel plan source hashes are invalid')


def prepare_plan(output, data_root):
    output = Path(output).resolve()
    expected = build_plan(output, data_root)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / '.parallel_plan.lock', blocking=True):
        path = output / 'parallel_plan.json'
        if path.exists():
            if read_json(path) != expected:
                raise ValueError('Existing parallel_plan.json differs; refusing to change the frozen plan')
        else:
            # Never silently wrap a sequential experiment or orphan shard artifacts.
            unrelated = [item.name for item in output.iterdir() if item.name != '.parallel_plan.lock']
            if unrelated:
                raise ValueError(f'Output contains artifacts without a parallel plan: {unrelated}')
            write_json(path, expected)
    return expected


def load_plan(output, data_root):
    output = Path(output).resolve()
    plan = read_json(output / 'parallel_plan.json')
    validate_plan(plan)
    if plan != build_plan(output, data_root):
        raise ValueError('Frozen plan differs from current source/configuration/output; refusing to start')
    return plan


def shard_path(plan, gpu):
    return Path(plan['output']) / plan['shards'][str(gpu)]['relative_output']


def shard_config(plan, gpu):
    return {**plan['common_config'], 'models': plan['shards'][str(gpu)]['models']}


def validate_protocol(plan, gpu, protocol):
    config = shard_config(plan, gpu)
    if protocol.get('config') != config or protocol.get('config_sha256') != digest(config):
        raise ValueError(f'GPU {gpu} protocol configuration differs from the frozen plan')
    for key in ('source_files', 'source_sha256'):
        if protocol.get(key) != plan[key]:
            raise ValueError(f'GPU {gpu} protocol {key} differs from the frozen source')
    if protocol.get('budget') != plan['shards'][str(gpu)]['budget']:
        raise ValueError(f'GPU {gpu} protocol budget differs from the frozen plan')
    record_hash = protocol.get('record_sha256')
    if not isinstance(record_hash, str) or len(record_hash) != 64:
        raise ValueError(f'GPU {gpu} protocol has no valid record hash')


def validate_shard(plan, gpu, metadata, require_test=False):
    """Validate metadata only; suitable for CPU-only synthetic tests."""
    gpu = str(gpu)
    protocol, summary, histories = (metadata[key] for key in ('protocol', 'summary', 'histories'))
    validate_protocol(plan, gpu, protocol)
    for key in ('config_sha256', 'record_sha256'):
        if summary.get(key) != protocol[key]:
            raise ValueError(f'GPU {gpu} summary {key} differs from its protocol')
    if summary.get('protocol_version') != plan['common_config']['protocol_version'] or summary.get('dry_run'):
        raise ValueError(f'GPU {gpu} summary is not a completed training summary')
    wanted = {(kind, seed) for kind in plan['shards'][gpu]['models'] for seed in SEEDS}
    runs = summary.get('runs', [])
    pairs = [(run['kind'], run['seed']) for run in runs]
    if len(pairs) != len(wanted) or set(pairs) != wanted or set(histories) != wanted:
        raise ValueError(f'GPU {gpu} must contain exactly the planned model/seed runs')
    if require_test and summary.get('test_evaluation_requested') is not True:
        raise ValueError(f'GPU {gpu} summary has not completed explicit test evaluation')
    for run in runs:
        key = (run['kind'], run['seed'])
        if run.get('epochs_completed') != EPOCHS:
            raise ValueError(f'GPU {gpu} {key} has not completed all {EPOCHS} epochs')
        if not isinstance(run.get('best_epoch'), int) or not 1 <= run['best_epoch'] <= EPOCHS:
            raise ValueError(f'GPU {gpu} {key} has no valid validation-selected checkpoint')
        score = run.get('best_validation_macro_positive_dice')
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f'GPU {gpu} {key} has no finite validation result')
        rows = histories[key]
        if len(rows) != EPOCHS or [row.get('epoch') for row in rows] != list(range(1, EPOCHS + 1)):
            raise ValueError(f'GPU {gpu} {key} history must contain all {EPOCHS} epochs exactly once')
        for row in rows:
            if (row.get('kind'), row.get('seed')) != key or row.get('sample_count') != IMAGES_PER_EPOCH:
                raise ValueError(f'GPU {gpu} {key} history has a different run identity or sample count')
            sample_hash = row.get('sample_sha256')
            if not isinstance(sample_hash, str) or len(sample_hash) != 64:
                raise ValueError(f'GPU {gpu} {key} history has no valid sample hash')
            queries = row.get('annotated_channel_queries')
            if not isinstance(queries, int) or queries < IMAGES_PER_EPOCH:
                raise ValueError(f'GPU {gpu} {key} history has no valid annotated-channel count')
        if require_test:
            metrics = run.get('test', {})
            if run.get('test_evaluated') is not True or not metrics:
                raise ValueError(f'GPU {gpu} {key} has not completed test evaluation')
            if (metrics.get('checkpoint_epoch') != run['best_epoch']
                    or metrics.get('checkpoint_selection') != 'best_validation_macro_positive_dice'
                    or metrics.get('config_sha256') != protocol['config_sha256']
                    or metrics.get('record_sha256') != protocol['record_sha256']
                    or 'macro_positive_dice' not in metrics):
                raise ValueError(f'GPU {gpu} {key} test result differs from the frozen selected checkpoint')


def aggregate_shards(plan, shards, require_test=True):
    """Check the complete cross-GPU contract before constructing a summary."""
    validate_plan(plan)
    if set(shards) != set(PARTITIONS):
        raise ValueError('Exactly GPU 0 and GPU 1 shard metadata are required')
    for gpu, metadata in shards.items():
        validate_shard(plan, gpu, metadata, require_test)
    protocols = [shards[gpu]['protocol'] for gpu in PARTITIONS]
    if len({protocol['record_sha256'] for protocol in protocols}) != 1:
        raise ValueError('Shard record hashes differ')
    draws = {}
    for metadata in shards.values():
        for (kind, seed), rows in metadata['histories'].items():
            for row in rows:
                key = (seed, row['epoch'])
                evidence = (row['sample_sha256'], row['sample_count'], row['annotated_channel_queries'])
                if key in draws and draws[key] != evidence:
                    raise ValueError(f'Model sample hashes/counts or annotated-channel counts differ for seed {seed}, epoch {row["epoch"]}')
                draws[key] = evidence
    runs = sorted([run for metadata in shards.values() for run in metadata['summary']['runs']],
                  key=lambda run: (run['seed'], MODELS.index(run['kind'])))

    def describe(values):
        values = [value for value in values if value is not None]
        return dict(seeds=len(values), mean=statistics.mean(values) if values else None,
                    sample_std=statistics.stdev(values) if len(values) > 1 else None)

    validation = {kind: describe([run['best_validation_macro_positive_dice'] for run in runs
                                 if run['kind'] == kind]) for kind in MODELS}
    tests = {}
    for kind in MODELS:
        evaluated = [run['test'] for run in runs if run['kind'] == kind and run.get('test_evaluated')]
        if evaluated:
            fields = {field for field in evaluated[0] if field.startswith('macro_')}
            if any({field for field in metrics if field.startswith('macro_')} != fields for metrics in evaluated):
                raise ValueError(f'Test metric fields differ between {kind} seeds')
            tests[kind] = {field: describe([metrics[field] for metrics in evaluated]) for field in sorted(fields)}
    combined_config = {**plan['common_config'], 'models': MODELS}
    return dict(protocol_version=plan['common_config']['protocol_version'],
                config_sha256=digest(combined_config), record_sha256=protocols[0]['record_sha256'],
                source_sha256=plan['source_sha256'], dry_run=False,
                test_evaluation_requested=require_test, runs=runs,
                validation_across_seeds=validation, test_across_seeds=tests,
                parallel=dict(plan_sha256=plan['plan_sha256'], models_by_gpu=PARTITIONS,
                              training_barrier_verified=True,
                              timing_interpretation=plan['timing_interpretation']),
                shards={gpu: dict(output=str(shard_path(plan, gpu)),
                                  config_sha256=metadata['protocol']['config_sha256'],
                                  source_sha256=metadata['protocol']['source_sha256'])
                        for gpu, metadata in shards.items()})


def load_shard(plan, gpu, training_snapshot=False, require_test=False):
    folder = shard_path(plan, gpu)
    protocol = read_json(folder / 'protocol.json')
    validate_protocol(plan, gpu, protocol)
    if digest(read_json(folder / 'records.json')) != protocol['record_sha256']:
        raise ValueError(f'GPU {gpu} stored records differ from its frozen record hash')
    summary_name = 'training_summary.json' if training_snapshot else 'summary.json'
    summary = read_json(folder / summary_name)
    histories = {}
    for kind in plan['shards'][str(gpu)]['models']:
        for seed in SEEDS:
            run_folder = folder / kind / f'seed_{seed}'
            histories[kind, seed] = [json.loads(line) for line in
                                    (run_folder / 'metrics.jsonl').read_text().splitlines() if line.strip()]
            for name in ('last.pt', 'best.pt'):
                if not (run_folder / name).is_file() or not (run_folder / name).stat().st_size:
                    raise ValueError(f'Missing completed checkpoint: {run_folder / name}')
    metadata = dict(protocol=protocol, summary=summary, histories=histories)
    validate_shard(plan, gpu, metadata, require_test)
    # mmap keeps tensor storage off the CPU heap; inspect metadata without GPU allocation.
    import torch
    for run in summary['runs']:
        run_folder = folder / run['kind'] / f'seed_{run["seed"]}'
        for filename, expected_epoch in [('last.pt', EPOCHS), ('best.pt', run['best_epoch'])]:
            checkpoint = torch.load(run_folder / filename, map_location='cpu', weights_only=True, mmap=True)
            for key in ('config', 'config_sha256', 'record_sha256', 'source_sha256'):
                if checkpoint.get(key) != protocol[key]:
                    raise ValueError(f'GPU {gpu} checkpoint {filename} {key} differs from protocol')
            if (checkpoint.get('kind') != run['kind'] or checkpoint.get('seed') != run['seed']
                    or checkpoint.get('epoch') != expected_epoch
                    or checkpoint.get('best_epoch') != run['best_epoch']
                    or checkpoint.get('best_score') != run['best_validation_macro_positive_dice']):
                raise ValueError(f'GPU {gpu} checkpoint {filename} identity/epoch/selection differs from summary')
            if filename == 'last.pt' and checkpoint.get('history') != histories[run['kind'], run['seed']]:
                raise ValueError(f'GPU {gpu} last.pt history differs from completed epoch history')
            del checkpoint
    if require_test:
        for run in summary['runs']:
            test_path = folder / run['kind'] / f'seed_{run["seed"]}' / 'test_metrics.json'
            if read_json(test_path) != run['test']:
                raise ValueError(f'Test file differs from completed summary: {test_path}')
    return metadata


def training_evidence(metadata):
    # Do not bind to the mutable test phase's summary, elapsed times, or flags.
    return digest(dict(runs=[{key: run[key] for key in ('kind', 'seed', 'epochs_completed',
                            'best_epoch', 'best_validation_macro_positive_dice')}
                             for run in metadata['summary']['runs']],
                       histories={f'{kind}/{seed}': rows for (kind, seed), rows in metadata['histories'].items()}))


def marker_path(plan, gpu, phase):
    return shard_path(plan, gpu) / f'{phase}_complete.json'


def completion_marker(plan, gpu, phase, metadata):
    protocol = metadata['protocol']
    marker = dict(phase=phase, gpu=str(gpu), plan_sha256=plan['plan_sha256'],
                  config_sha256=protocol['config_sha256'], record_sha256=protocol['record_sha256'],
                  source_sha256=protocol['source_sha256'], training_sha256=training_evidence(metadata))
    if phase == 'test':
        marker['summary_sha256'] = digest(metadata['summary'])
    return marker


def validate_marker(plan, gpu, phase):
    marker = read_json(marker_path(plan, gpu, phase))
    if marker.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError(f'GPU {gpu} has a stale {phase} completion marker from another plan')
    metadata = load_shard(plan, gpu, training_snapshot=phase == 'training', require_test=phase == 'test')
    if marker != completion_marker(plan, gpu, phase, metadata):
        raise ValueError(f'GPU {gpu} {phase} completion marker differs from validated artifacts')
    return metadata


def gpu_snapshot(gpu):
    command = ['nvidia-smi', '-i', str(gpu),
               '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free',
               '--format=csv,noheader,nounits']
    lines = subprocess.check_output(command, text=True).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f'GPU {gpu} lookup did not return exactly one device')
    index, uuid, name, total, used, free = [part.strip() for part in next(csv.reader(lines))]
    processes = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True).strip()
    snapshot = dict(index=int(index), uuid=uuid, name=name, total_mib=int(total), used_mib=int(used),
                    free_mib=int(free), existing_compute_processes=processes,
                    shared_with_existing_work=bool(processes))
    if snapshot['free_mib'] < MINIMUM_FREE_MIB:
        raise RuntimeError(f'GPU {gpu} has {free} MiB free; at least {MINIMUM_FREE_MIB} MiB required; '
                           'existing jobs were left untouched')
    return snapshot


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class Worker:
    def __init__(self, plan, gpu, lock):
        self.plan, self.gpu, self.lock = plan, str(gpu), lock
        self.root, self.peer = Path(plan['output']), str(1 - int(gpu))
        self.log_path = self.root / 'logs' / f'gpu{gpu}.log'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.status = dict(gpu=self.gpu, worker_pid=os.getpid(), hostname=socket.gethostname(),
                           started_at=timestamp(), plan_sha256=plan['plan_sha256'],
                           source_sha256=plan['source_sha256'], log=str(self.log_path),
                           shard=str(shard_path(plan, gpu)))

    def update(self, state, **extra):
        self.status.update(state=state, updated_at=timestamp(), **extra)
        write_json(self.root / f'worker_gpu{self.gpu}.json', self.status)

    def event(self, message, **extra):
        line = json.dumps(dict(time=timestamp(), gpu=self.gpu, message=message, **extra))
        print(line, flush=True)
        with self.log_path.open('a') as stream:
            stream.write(line + '\n')

    def failure_path(self, gpu):
        return self.root / f'failed_gpu{gpu}.json'

    def check_peer_failure(self):
        path = self.failure_path(self.peer)
        if path.exists():
            failure = read_json(path)
            if failure.get('plan_sha256') != self.plan['plan_sha256']:
                raise RuntimeError(f'GPU {self.peer} has a stale failure marker from another plan')
            raise PeerFailedError(f'Peer GPU {self.peer} failed: {failure.get("error", "unknown error")}')

    def run_suite(self, evaluate_test=False):
        folder = shard_path(self.plan, self.gpu)
        arguments = suite_arguments(self.plan['common_config']['data_root'], folder,
                                    self.plan['shards'][self.gpu]['models'])
        if folder.exists():
            if not (folder / 'protocol.json').is_file():
                raise ValueError(f'Existing shard has no frozen protocol: {folder}')
            validate_protocol(self.plan, self.gpu, read_json(folder / 'protocol.json'))
            arguments.append('--resume')
        if evaluate_test:
            if '--resume' not in arguments:
                raise ValueError('Test evaluation requires an existing completed shard')
            arguments.append('--evaluate-test')
        command = [sys.executable, '-m', 'unet_moe.fair_suite', *arguments]
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': self.gpu,
                       'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
        self.event('Starting frozen fair-suite subprocess', command=command)
        with self.log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=CODE_ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(self.lock.fileno(),))
            self.update('evaluating_test' if evaluate_test else 'training', child_pid=process.pid)
            try:
                result = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        self.status.pop('child_pid', None)
        if result:
            raise RuntimeError(f'fair_suite exited {result}; see {self.log_path}')

    def wait_for_peer(self, phase):
        last_state = None
        while True:
            self.check_peer_failure()
            if marker_path(self.plan, self.peer, phase).exists():
                return validate_marker(self.plan, self.peer, phase)
            peer_status_path = self.root / f'worker_gpu{self.peer}.json'
            peer_state = 'not_started'
            if peer_status_path.exists():
                peer_status = read_json(peer_status_path)
                if peer_status.get('plan_sha256') != self.plan['plan_sha256']:
                    raise ValueError(f'GPU {self.peer} status belongs to a different plan')
                peer_state = peer_status.get('state', 'unknown')
            state = 'waiting_for_peer' if peer_state == 'not_started' else f'waiting_for_peer_{phase}'
            self.update(state, waiting_for_gpu=self.peer, waiting_for_phase=phase, peer_state=peer_state)
            if (state, peer_state) != last_state:
                self.event('Waiting for independently launched peer', phase=phase, peer_state=peer_state)
                last_state = (state, peer_state)
            time.sleep(POLL_SECONDS)

    def run(self):
        self.update('starting')
        own_failure = self.failure_path(self.gpu)
        if own_failure.exists():
            if read_json(own_failure).get('plan_sha256') != self.plan['plan_sha256']:
                raise ValueError('Own failure marker belongs to a different plan')
            own_failure.rename(self.root / f'failed_gpu{self.gpu}.previous_{time.time_ns()}.json')
        self.status['gpu_at_start'] = gpu_snapshot(self.gpu)
        self.update('ready')
        self.event('GPU admission succeeded', snapshot=self.status['gpu_at_start'])
        self.check_peer_failure()
        if marker_path(self.plan, self.gpu, 'training').exists():
            own_training = validate_marker(self.plan, self.gpu, 'training')
        else:
            self.run_suite()
            own_training = load_shard(self.plan, self.gpu)
            if (own_training['summary'].get('test_evaluation_requested')
                    or any(run.get('test_evaluated') for run in own_training['summary']['runs'])):
                raise ValueError('Test results exist before the parallel training barrier')
            write_json(shard_path(self.plan, self.gpu) / 'training_summary.json', own_training['summary'])
            write_json(marker_path(self.plan, self.gpu, 'training'),
                       completion_marker(self.plan, self.gpu, 'training', own_training))
        self.update('training_complete')
        self.event('All local runs completed 50 epochs; training completion verified')
        peer_training = self.wait_for_peer('training')
        aggregate_shards(self.plan, {self.gpu: own_training, self.peer: peer_training}, require_test=False)
        self.update('training_barrier_verified')
        if marker_path(self.plan, self.gpu, 'test').exists():
            validate_marker(self.plan, self.gpu, 'test')
        else:
            self.check_peer_failure()
            gpu_snapshot(self.gpu)  # Recheck admission after possibly waiting for days.
            self.run_suite(evaluate_test=True)
            own_test = load_shard(self.plan, self.gpu, require_test=True)
            write_json(marker_path(self.plan, self.gpu, 'test'),
                       completion_marker(self.plan, self.gpu, 'test', own_test))
        self.update('test_complete')
        self.wait_for_peer('test')
        with exclusive_lock(self.root / '.summary.lock', blocking=True):
            complete = {gpu: validate_marker(self.plan, gpu, 'test') for gpu in PARTITIONS}
            summary = aggregate_shards(self.plan, complete)
            summary['parallel']['gpu_observations'] = {
                gpu: read_json(self.root / f'worker_gpu{gpu}.json').get('gpu_at_start')
                for gpu in PARTITIONS if (self.root / f'worker_gpu{gpu}.json').exists()}
            write_json(self.root / 'summary.json', summary)
        self.update('complete', summary=str(self.root / 'summary.json'), waiting_for_gpu=None,
                    waiting_for_phase=None, peer_state=None)
        self.event('All nine runs and test evaluations complete; root summary committed')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path,
                        default=Path(os.environ.get('UNET_MOE_DATA_ROOT', DEFAULT_DATA_ROOT)))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--gpu', type=int, choices=[0, 1])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.prepare:
        plan = prepare_plan(args.output, args.data_root)
        print(json.dumps(dict(plan=str(Path(plan['output']) / 'parallel_plan.json'),
                              plan_sha256=plan['plan_sha256'])), flush=True)
        return
    plan = load_plan(args.output, args.data_root)
    lock_path = Path(f'/tmp/multi_organ_fair_gpu_{args.gpu}.lock')
    with exclusive_lock(lock_path) as lock:
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(), gpu=args.gpu, output=plan['output'])) + '\n')
        lock.flush()
        worker = Worker(plan, args.gpu, lock)
        previous = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise InterruptedError(f'Worker received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            worker.run()
        except PeerFailedError as error:
            worker.update('blocked_by_peer', error=str(error))
            worker.event('Worker blocked by peer failure; restart after the peer recovers', error=str(error))
            raise
        except BaseException as error:
            failure = dict(gpu=str(args.gpu), plan_sha256=plan['plan_sha256'],
                           time=timestamp(), error=f'{type(error).__name__}: {error}', worker_pid=os.getpid())
            write_json(worker.failure_path(args.gpu), failure)
            worker.update('failed', error=failure['error'])
            worker.event('Worker failed', error=failure['error'])
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'fair_parallel: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
