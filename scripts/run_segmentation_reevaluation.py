"""Additive, resumable DSC/HD95/ASSD backfill of the medical experiment inventory.

Prepare once from the checkout with --initialize, then run the frozen script in
OUTPUT/source_snapshot/scripts. Original artifacts are always read-only. GPU
admission shares the training controllers' file locks and yields between jobs.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
while str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from scripts.medical_priority_process import atomic_write_json, exclusive_lock, process_identity, process_running
from unet_moe.reevaluation_inventory import build_inventory
from unet_moe.reevaluation_report import csv_text, markdown_report, report_rows, seed_groups, summarize_rows
from unet_moe.surface_metrics import binary_metrics


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_text(path, text):
    path = Path(path)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize(output, runs_root, empty_policy='diagonal', metric_workers=4):
    output, runs_root = Path(output).resolve(), Path(runs_root).resolve()
    if output.exists():
        raise FileExistsError('Evaluation directory already exists; use its frozen resume command')
    if empty_policy not in ('diagonal', 'strict') or not 1 <= metric_workers <= 16:
        raise ValueError('Invalid metric policy or worker count')
    inventory = build_inventory(runs_root, project_root=CODE_ROOT)
    output.mkdir(parents=True)
    sources = sorted([p for folder in ('unet_moe', 'scripts') for p in (CODE_ROOT / folder).glob('*.py')])
    hashes = {}
    for source in sources:
        relative = source.relative_to(CODE_ROOT)
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = file_sha(target)
    # Some early 256-pixel ablations predate source_snapshot. Recover their
    # actual committed inference modules, then the adapter selects candidates
    # only when their original per-file SHA values match.
    recovered = []
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE_ROOT, text=True).strip()
        relative_root = Path('recovered_sources') / f'git_{commit}'
        for name in ('__init__.py', 'model.py', 'baseline.py', 'region.py', 'data.py'):
            relative = relative_root / 'unet_moe' / name
            content = subprocess.check_output(['git', 'show', f'{commit}:unet_moe/{name}'], cwd=CODE_ROOT)
            target = output / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hashes[str(relative)] = file_sha(target)
        recovered.append(str(output / 'source_snapshot' / relative_root))
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Absence cannot silently authorize a replacement: adapter provenance
        # validation will mark old jobs unsupported if no exact source exists.
        pass
    protocol = dict(schema='medical_surface_reevaluation_v1', created_at=timestamp(),
                    runs_root=str(runs_root), empty_policy=empty_policy, metric_workers=metric_workers,
                    source_files=hashes, source_sha256=digest(hashes),
                    inference_source_candidates=recovered,
                    job_ids=sorted(j['job_id'] for j in inventory['jobs']),
                    threshold=0.5, units={'DSC': '%', 'HD95': 'pixel', 'ASSD': 'pixel'},
                    surface_connectivity=4, spacing=[1.0, 1.0],
                    hd95_definition='95th percentile of concatenated two directed surface distances',
                    assd_definition='mean of concatenated two directed surface distances',
                    diagonal_definition='hypot(H-1,W-1), maximum Euclidean pixel-center separation',
                    ignored_boundary='exclude surface pixels touching ignore in the 4-neighborhood',
                    aggregation='image mean within structure/domain; equal group macro; all and positive cohorts plus their fixed ground-truth-surface-defined subsets',
                    cohorts=['all', 'positive', 'all_surface_defined', 'positive_surface_defined'],
                    surface_defined_cohort='fixed by ground truth only: exclude nonempty targets without an observable surface; retain all prediction failures',
                    selection='existing best/last/selected guardrail; never reselect on new metrics',
                    validation_test_policy='original splits and test barriers retained',
                    initial_inventory_sha256=digest(inventory),
                    metric_reference='https://github.com/loli/medpy/blob/master/medpy/metric/binary.py')
    protocol['protocol_sha256'] = digest(protocol)
    atomic_write_json(output / 'protocol.json', protocol)
    atomic_write_json(output / 'inventory_initial.json', inventory)
    atomic_write_json(output / 'inventory.json', inventory)
    (output / 'jobs').mkdir()
    (output / 'logs').mkdir()
    atomic_write_json(output / 'status.json', dict(state='prepared', updated_at=timestamp(),
                      protocol_sha256=protocol['protocol_sha256'], jobs=len(inventory['jobs'])))
    return protocol


def load_protocol(root):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    if protocol.get('protocol_sha256') != digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'}):
        raise ValueError('Evaluation protocol changed')
    if digest(read(root / 'inventory_initial.json')) != protocol['initial_inventory_sha256']:
        raise ValueError('Initial evaluation inventory changed')
    source = root / 'source_snapshot'
    if CODE_ROOT != source:
        raise ValueError('Run the frozen evaluator under OUTPUT/source_snapshot')
    if digest(protocol['source_files']) != protocol['source_sha256']:
        raise ValueError('Invalid evaluator source manifest')
    for relative, expected in protocol['source_files'].items():
        target = source / relative
        if not target.resolve().is_relative_to(source) or file_sha(target) != expected:
            raise ValueError(f'Evaluator source changed: {relative}')
    return protocol


def score_sample(sample, empty_policy):
    metric = binary_metrics(sample.prediction, sample.target, empty_policy=empty_policy)
    return dict(image_id=sample.image_id, domain=sample.domain, structure=sample.structure,
                view=sample.view, source=sample.source, variant=getattr(sample, 'variant', None),
                task_id=getattr(sample, 'task_id', None), **metric)


def bounded_scores(samples, workers, empty_policy):
    """Bound outstanding masks; Executor.map in Python3.11 would eagerly load all."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for sample in samples:
            pending.append(executor.submit(score_sample, sample, empty_policy))
            if len(pending) >= max(2, workers * 2):
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def evaluate_job(root, manifest, device='cpu', limit_samples=None):
    """A worker writes only its independent output folder, never training files."""
    from unet_moe.reevaluation_inference import load_evaluation
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    protocol = load_protocol(root)
    job = read(manifest)
    if job['job_id'] not in protocol['job_ids'] or job.get('evaluation_protocol_sha256') != protocol['protocol_sha256']:
        raise ValueError('Job is outside the frozen reevaluation scope')
    if job.get('job_manifest_sha256') != digest({k: v for k, v in job.items() if k != 'job_manifest_sha256'}):
        raise ValueError('Job manifest changed')
    live = {j['job_id']: j for j in build_inventory(protocol['runs_root'], project_root=CODE_ROOT)['jobs']}
    if job['job_id'] not in live or live[job['job_id']]['status'] != 'ready':
        raise ValueError('Original training/test barrier has not opened for this job')
    if file_sha(job['checkpoint']) != job['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after evaluation dispatch')
    folder = root / 'jobs' / job['job_id']
    result_file = folder / 'result.json'
    if result_file.exists():
        raise FileExistsError('Completed result exists; refuse to overwrite it')
    rows = []
    started = time.monotonic()
    session = load_evaluation(job, device=device)
    temporary = folder / '.per_image.jsonl.tmp'
    identities = set()
    try:
        with temporary.open('w') as stream:
            for row in bounded_scores(session.samples, protocol['metric_workers'], protocol['empty_policy']):
                key = (row['image_id'], row['view'], row['structure'], row['variant'])
                if key in identities:
                    raise ValueError(f'Duplicate evaluated sample: {key}')
                identities.add(key)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if len(rows) % 100 == 0:
                    stream.flush()
                    atomic_write_json(folder / 'progress.json', dict(state='evaluating', rows=len(rows),
                                      updated_at=timestamp(), elapsed_seconds=time.monotonic() - started))
                if limit_samples and len(rows) >= limit_samples:
                    raise RuntimeError('Debug sample limit reached: partial evaluation never produces result.json')
            stream.flush()
            os.fsync(stream.fileno())
        if not rows:
            raise ValueError('Evaluation produced no rows')
        expected_images = session.metadata.get('images')
        if expected_images is not None and len({r['image_id'] for r in rows}) != expected_images:
            raise ValueError('Evaluated image count differs from saved split scope')
        if 'union' in session.metadata.get('views', []):
            union_images = {r['image_id'] for r in rows if r['view'] == 'union'}
            if len(union_images) != (expected_images if expected_images is not None else len({r['image_id'] for r in rows})):
                raise ValueError('Union evaluation omitted images from the saved split')
        if file_sha(job['checkpoint']) != job['checkpoint_sha256']:
            raise ValueError('Checkpoint changed while evaluating')
        per_image = folder / 'per_image.jsonl'
        temporary.replace(per_image)
        result = dict(schema='medical_surface_metrics_result_v1', complete=True, job=job,
                      evaluation_protocol_sha256=protocol['protocol_sha256'],
                      checkpoint_sha256=job['checkpoint_sha256'], metadata=session.metadata,
                      metrics=summarize_rows(rows), rows=len(rows),
                      per_image_sha256=file_sha(per_image), completed_at=timestamp(),
                      elapsed_seconds=time.monotonic() - started)
        result['result_sha256'] = digest(result)
        atomic_write_json(result_file, result)
        atomic_write_json(folder / 'progress.json', dict(state='complete', rows=len(rows), updated_at=timestamp()))
        return result
    finally:
        temporary.unlink(missing_ok=True)


def job_priority(job):
    # Complete common 512 baselines first, then current paired matrices, followed
    # by older protocols and validation-only explorations. Every scoped job stays.
    family = job['family']
    rank = {'fair': 0, 'single_union': 0, 'region_normalization': 1, 'organ_weighted': 2,
            'abdomen_recovery': 4, 'postprocess_screen': 6}.get(family, 5)
    if job['experiment'].startswith('empty_ablation'):
        rank = 4
    return (job['selection'] != 'best', job['split'] != 'test', rank,
            -int(job.get('size') or 0), job['experiment'], str(job.get('arm') or ''),
            job.get('seed') or 0, job['kind'], job['job_id'])


def training_needs_gpu(runs_root):
    """Yield to pending jobs of living medical training controllers."""
    for filename in ('q01_organ_weighted_80_512_20261007', 'region_domain_norm_80_512_20261010'):
        path = Path(runs_root) / filename / 'controller_status.json'
        if not path.exists():
            continue
        status = read(path)
        identity = process_identity(status.get('controller_pid', 0))
        if not identity or not process_running(identity):
            continue
        state = status.get('state')
        if state in ('recovering', 'preflight', 'preflighting', 'summarizing', 'training_barrier_verified'):
            return True
        phase = 'train' if state == 'training' else 'test' if state == 'evaluating_test' else None
        if phase and any(job.get(phase + '_state') == 'pending' for job in status.get('jobs', [])):
            return True
    return False


class Controller:
    def __init__(self, root, gpu=1, poll_seconds=30):
        self.root = Path(root).resolve()
        self.protocol = load_protocol(self.root)
        self.gpu = gpu
        self.poll_seconds = poll_seconds
        self.process = None
        self.completed = {}
        self.failed = {}
        self.sha_cache = {}

    def update(self, state, **extra):
        atomic_write_json(self.root / 'status.json', dict(
            state=state, updated_at=timestamp(), controller_pid=os.getpid(), gpu=self.gpu,
            protocol_sha256=self.protocol['protocol_sha256'],
            completed_jobs=len(self.completed), failed_jobs=len(self.failed), **extra))

    def refresh(self):
        inventory = build_inventory(self.protocol['runs_root'], project_root=CODE_ROOT)
        allowed = set(self.protocol['job_ids'])
        inventory['jobs'] = [j for j in inventory['jobs'] if j['job_id'] in allowed]
        original = {j['job_id']: j for j in read(self.root / 'inventory_initial.json')['jobs']}
        present = {j['job_id'] for j in inventory['jobs']}
        for job_id in sorted(allowed - present):
            inventory['jobs'].append({**original[job_id], 'status': 'unavailable',
                                       'reason': 'frozen inventory job disappeared from live experiment inventory'})
        for job in inventory['jobs']:
            for field in ('protocol_sha256', 'records_sha256'):
                if job.get(field) != original[job['job_id']].get(field):
                    job.update(status='unavailable', reason=f'original {field} changed after evaluation scope was frozen')
            folder = self.root / 'jobs' / job['job_id']
            result_path, failure = folder / 'result.json', folder / 'failure.json'
            if job['job_id'] not in self.completed and result_path.exists():
                result = read(result_path)
                stored_job = result['job']
                checkpoint = Path(job['checkpoint'])
                stat = checkpoint.stat()
                key = (str(checkpoint), stat.st_size, stat.st_mtime_ns)
                if key not in self.sha_cache:
                    self.sha_cache[key] = file_sha(checkpoint)
                if (result.get('complete') is not True or result.get('evaluation_protocol_sha256') != self.protocol['protocol_sha256']
                        or result['job']['job_id'] != job['job_id']
                        or result.get('result_sha256') != digest({k: v for k, v in result.items() if k != 'result_sha256'})
                        or stored_job != read(folder / 'job.json')
                        or any(stored_job.get(k) != job.get(k) for k in ('protocol_sha256', 'records_sha256', 'checkpoint'))
                        or stored_job.get('job_manifest_sha256') != digest({k: v for k, v in stored_job.items() if k != 'job_manifest_sha256'})
                        or result['checkpoint_sha256'] != stored_job['checkpoint_sha256']
                        or result['checkpoint_sha256'] != self.sha_cache[key]
                        or result['per_image_sha256'] != file_sha(folder / 'per_image.jsonl')):
                    raise ValueError(f'Unverified prior evaluation result: {result_path}')
                self.completed[job['job_id']] = result
            if failure.exists() and job['job_id'] not in self.completed:
                self.failed[job['job_id']] = read(failure)
            if job['job_id'] in self.completed:
                job['evaluation_state'] = 'complete'
            elif job['job_id'] in self.failed:
                job['evaluation_state'] = 'failed'
            else:
                job['evaluation_state'] = job['status']
        inventory['evaluation_counts'] = dict(Counter(j['evaluation_state'] for j in inventory['jobs']))
        atomic_write_json(self.root / 'inventory.json', inventory)
        return inventory

    def reports(self, inventory):
        results = sorted(self.completed.values(), key=lambda r: job_priority(r['job']))
        summary = dict(updated_at=timestamp(), evaluation_protocol_sha256=self.protocol['protocol_sha256'],
                       counts=inventory['evaluation_counts'], completed_job_ids=list(self.completed),
                       failed_jobs=self.failed, seed_groups=seed_groups(results))
        atomic_write_json(self.root / 'summary.json', summary)
        write_text(self.root / 'metrics.csv', csv_text(report_rows(results)))
        write_text(self.root / 'REPORT.md', markdown_report(results, inventory, self.failed, summary['updated_at']))

    def dispatch(self, job, lock_fd):
        job = dict(job)
        job['inference_source_candidates'] = self.protocol.get('inference_source_candidates', [])
        stat = Path(job['checkpoint']).stat()
        cache_key = (job['checkpoint'], stat.st_size, stat.st_mtime_ns)
        if cache_key not in self.sha_cache:
            self.sha_cache[cache_key] = file_sha(job['checkpoint'])
        sha = self.sha_cache[cache_key]
        if job.get('checkpoint_sha256') and job['checkpoint_sha256'] != sha:
            raise ValueError('Source checkpoint digest differs from original experiment')
        job.update(checkpoint_sha256=sha, evaluation_protocol_sha256=self.protocol['protocol_sha256'])
        job['job_manifest_sha256'] = digest(job)
        folder = self.root / 'jobs' / job['job_id']
        folder.mkdir(exist_ok=True)
        manifest = folder / 'job.json'
        atomic_write_json(manifest, job)
        path = self.root / 'logs' / f'{job["job_id"]}.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(self.root),
                   '--evaluate-job', str(manifest), '--device', 'cuda:0']
        with path.open('a') as log:
            self.process = subprocess.Popen(command, cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock_fd, self.root_lock.fileno()),
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(self.gpu), 'OMP_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(CODE_ROOT), 'PYTHONUNBUFFERED': '1'})
            self.update('evaluating', current_job={k: job.get(k) for k in
                        ('job_id', 'experiment', 'kind', 'arm', 'seed', 'selection', 'split')},
                        worker_pid=self.process.pid, log=str(path))
            result = self.process.wait()
            self.process = None
        if result:
            atomic_write_json(folder / 'failure.json', dict(exit_code=result, at=timestamp(), log=str(path)))
        elif not (folder / 'result.json').exists():
            raise RuntimeError('Evaluator exited successfully without a completed result')

    def run(self):
        def interrupted(signum, frame):
            raise InterruptedError(f'Evaluation controller received signal {signum}')
        with exclusive_lock(self.root / '.controller.lock') as root_lock:
            self.root_lock = root_lock
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, interrupted)
            try:
                self.update('starting')
                while True:
                    inventory = self.refresh()
                    self.reports(inventory)
                    pending = sorted([j for j in inventory['jobs'] if j['evaluation_state'] == 'ready'], key=job_priority)
                    deferred = sum(j['evaluation_state'] == 'deferred' for j in inventory['jobs'])
                    if not pending:
                        if not deferred:
                            unavailable = inventory['evaluation_counts'].get('unavailable', 0)
                            self.update('complete_with_failures' if self.failed else 'complete_with_unavailable' if unavailable else 'complete',
                                        counts=inventory['evaluation_counts'])
                            return
                        self.update('waiting_training_barriers', counts=inventory['evaluation_counts'])
                        time.sleep(self.poll_seconds)
                        continue
                    if training_needs_gpu(self.protocol['runs_root']):
                        self.update('yielding_to_training', counts=inventory['evaluation_counts'])
                        time.sleep(self.poll_seconds)
                        continue
                    lock_path = Path(f'/tmp/multi_organ_fair_gpu_{self.gpu}.lock')
                    try:
                        with exclusive_lock(lock_path) as lock:
                            free = int(subprocess.check_output(['nvidia-smi', '-i', str(self.gpu),
                                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15).strip())
                            if free < 18000:
                                self.update('waiting_gpu_memory', free_mib=free, counts=inventory['evaluation_counts'])
                            elif training_needs_gpu(self.protocol['runs_root']):
                                self.update('yielding_to_training', counts=inventory['evaluation_counts'])
                            else:
                                self.dispatch(pending[0], lock.fileno())
                    except BlockingIOError:
                        self.update('waiting_gpu_lock', counts=inventory['evaluation_counts'])
                    time.sleep(self.poll_seconds)
            except BaseException as error:
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                self.update('interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed',
                            error=f'{type(error).__name__}: {error}')
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, default=Path('/ssd1/data/Multi-Organ_Foundation_Model/runs'))
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--empty-policy', choices=['diagonal', 'strict'], default='diagonal')
    parser.add_argument('--metric-workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--evaluate-job', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.initialize:
        initialize(args.output, args.runs_root, args.empty_policy, args.metric_workers)
    elif args.evaluate_job:
        evaluate_job(args.output, args.evaluate_job, args.device)
    else:
        if args.gpu < 0 or not 1 <= args.poll_seconds <= 60:
            raise ValueError('Invalid GPU or poll interval')
        Controller(args.output, args.gpu, args.poll_seconds).run()


if __name__ == '__main__':
    main()
