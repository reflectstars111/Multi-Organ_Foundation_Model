"""Queue the 80-epoch binary-union U-Net behind the matching fair suite.

The fair suite and this single-output baseline use the same paper5 records,
image draws, split, image size, training budget, and optimizer settings. The
binary-union target has separate metrics from the 16-structure models.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .defer_abdomen_recovery import fair_state, write_status
from .defer_single_channel import gpu_free_mib, gpu_lock_path, run_command


SEEDS = (42, 43, 44)
FAIR_MODELS = ('shared', 'task_id', 'region')
MINIMUM_FREE_MIB = 34 * 1024
EXPECTED_FAIR_CONFIG = dict(
    seeds=list(SEEDS), split_seed=42, size=512, base=24, batch_size=16,
    epochs=80, quota_per_domain=500, workers=0, top_k=1, experts=9,
    lr=1e-4, weight_decay=1e-4, lr_schedule='constant', min_lr_ratio=.1,
    abdomen_split='ct7_2', split_policy='trainmore_v6', labels='paper5',
)
EXPECTED_SINGLE_CONFIG = {
    key: value for key, value in EXPECTED_FAIR_CONFIG.items()
    if key not in ('top_k', 'experts', 'labels')
}


def reference_records(fair_root):
    return fair_root / 'shards' / 'gpu1' / 'records.json'


def check_fair_configuration(fair_root):
    """Reject a complete but differently configured reference experiment."""
    summary = json.loads((fair_root / 'summary.json').read_text())
    if len(summary.get('runs', [])) != len(SEEDS) * len(FAIR_MODELS):
        raise ValueError('Fair summary has an unexpected run count')
    if not reference_records(fair_root).is_file():
        raise FileNotFoundError('The Region fair shard has no records.json')
    protocol = json.loads((fair_root / 'shards' / 'gpu1' / 'protocol.json').read_text())
    config = protocol['config']
    for key, expected in EXPECTED_FAIR_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f'Fair reference {key} differs from the 80-epoch protocol')
    if config.get('models') != ['region']:
        raise ValueError('GPU1 reference shard must contain the Region model')
    if summary.get('record_sha256') != protocol.get('record_sha256'):
        raise ValueError('Fair summary and Region shard record hashes differ')
    return summary['record_sha256']


def validate_complete(output, fair_record_sha256):
    """Only a fully tested, paired three-seed output can complete the queue."""
    output = Path(output)
    summary_path = output / 'summary.json'
    protocol = json.loads((output / 'protocol.json').read_text())
    summary = json.loads(summary_path.read_text())
    if (summary.get('protocol_version') != 'single_channel_paper5_union_v1'
            or summary.get('test_evaluation_requested') is not True
            or summary.get('config_sha256') != protocol.get('config_sha256')
            or summary.get('record_sha256') != fair_record_sha256
            or protocol.get('record_sha256') != fair_record_sha256
            or protocol.get('reference_records_sha256') != fair_record_sha256):
        raise ValueError('Single-channel output is not paired with the 80-epoch fair records')
    for key, expected in EXPECTED_SINGLE_CONFIG.items():
        if protocol.get('config', {}).get(key) != expected:
            raise ValueError(f'Single-channel {key} differs from the 80-epoch protocol')
    runs = summary.get('runs', [])
    if (len(runs) != len(SEEDS)
            or {row.get('seed') for row in runs} != set(SEEDS)
            or any(row.get('epochs_completed') != 80 or row.get('test_evaluated') is not True
                   or not (output / f"seed_{row.get('seed')}" / 'test_metrics.json').is_file()
                   for row in runs)):
        raise ValueError('Single-channel summary lacks the completed 80-epoch three-seed test barrier')
    return summary_path


def preflight_peak(report_path, fair_record_sha256):
    report = json.loads(Path(report_path).read_text())
    peak = report.get('peak_reserved_mib')
    steps = report.get('optimizer_steps')
    capacity = report.get('gpu_total_mib')
    if (report.get('schema') != 'single80_physical_batch_preflight_v1'
            or report.get('record_sha256') != fair_record_sha256
            or report.get('status') != 'passed'
            or report.get('optimizer_step_ok') is not True
            or not isinstance(steps, int) or steps < 2
            or report.get('physical_batch_size') != 16
            or not isinstance(peak, int) or peak < 1):
        raise ValueError('Single-channel GPU preflight is incomplete or mismatched')
    if (not isinstance(capacity, int)
            or max(MINIMUM_FREE_MIB, peak + 2048) > capacity):
        raise ValueError('Single-channel GPU preflight leaves an unreachable memory gate')
    return peak


def suite_command(args, fair_root, output):
    """Use the exact fixed fair data/training settings plus the union target."""
    return [sys.executable, '-m', 'unet_moe.single_channel_suite',
            '--data-root', str(args.data_root.resolve()), '--output', str(output),
            '--reference-records', str(reference_records(fair_root)),
            '--seeds', '42,43,44', '--split-seed', '42',
            '--size', '512', '--base', '24', '--batch-size', '16',
            '--epochs', '80', '--quota-per-domain', '500',
            '--lr', '0.0001', '--weight-decay', '0.0001',
            '--lr-schedule', 'constant', '--min-lr-ratio', '0.1',
            '--workers', '0', '--device', 'cuda:0']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, required=True)
    parser.add_argument('--research-status', type=Path, required=True,
                        help='80-epoch fair controller_status.json')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=45)
    parser.add_argument('--minimum-free-mib', type=int, default=MINIMUM_FREE_MIB)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or args.minimum_free_mib < MINIMUM_FREE_MIB:
        parser.error('poll seconds must be positive and free-memory gate at least 34 GiB')
    fair_root = args.fair_root.resolve()
    output = args.output.resolve()
    controller = output.parent / f'{output.name}-logs'
    status_path = controller / 'controller_status.json'
    log_path = controller / 'single_channel.log'
    preflight_path = controller / 'gpu_preflight.json'
    details = dict(fair_root=str(fair_root), gpu=0, log=str(log_path))
    write_status(status_path, 'waiting_fair', **details)
    try:
        while True:
            state = fair_state(fair_root, args.research_status)
            if state == 'ready':
                break
            if state != 'waiting':
                raise RuntimeError(state)
            if args.research_status.is_file():
                controller_status = json.loads(args.research_status.read_text())
                controller_state = controller_status.get('state', '')
                if controller_state.startswith(('failed', 'blocked')):
                    raise RuntimeError('80-epoch fair controller failed or blocked')
            time.sleep(args.poll_seconds)
        fair_record_sha256 = check_fair_configuration(fair_root)
        common = suite_command(args, fair_root, output)
        prepare = [*common, '--dry-run']
        if (output / 'protocol.json').is_file():
            prepare.append('--resume')
        environment = dict(os.environ, OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                           CUDA_VISIBLE_DEVICES='0')
        write_status(status_path, 'preparing', **details, command=prepare)
        if run_command(prepare, log_path, environment):
            raise RuntimeError(f'Single-channel preparation failed; see {log_path}')

        lock_path = gpu_lock_path(0)
        while True:
            required_free_mib = args.minimum_free_mib
            if preflight_path.is_file():
                required_free_mib = max(required_free_mib,
                    preflight_peak(preflight_path, fair_record_sha256) + 2048)
            with lock_path.open('a+') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    free = None
                else:
                    free = gpu_free_mib(0)
                    if free >= required_free_mib:
                        if not preflight_path.is_file():
                            probe = [sys.executable, '-m', 'scripts.preflight_single80',
                                     '--records', str(reference_records(fair_root)),
                                     '--device', 'cuda:0', '--output', str(preflight_path)]
                            write_status(status_path, 'preflighting', **details,
                                         gpu_free_mib_at_start=free, command=probe)
                            if run_command(probe, log_path, environment, lock):
                                raise RuntimeError(f'Single-channel GPU preflight failed; see {log_path}')
                        required_free_mib = max(required_free_mib,
                            preflight_peak(preflight_path, fair_record_sha256) + 2048)
                        free = gpu_free_mib(0)
                    if free is not None and free >= required_free_mib:
                        train = [*common, '--resume']
                        write_status(status_path, 'training', **details,
                                     gpu_free_mib_at_start=free, command=train)
                        if run_command(train, log_path, environment, lock):
                            raise RuntimeError(f'Single-channel training failed; see {log_path}')
                        test = [*common, '--resume', '--evaluate-test']
                        write_status(status_path, 'testing', **details, command=test)
                        if run_command(test, log_path, environment, lock):
                            raise RuntimeError(f'Single-channel test failed; see {log_path}')
                        summary_path = validate_complete(output, fair_record_sha256)
                        write_status(status_path, 'complete', **details,
                                     summary=str(summary_path))
                        return
            write_status(status_path, 'waiting_gpu', **details,
                         minimum_free_mib=required_free_mib,
                         observed_free_mib=free)
            time.sleep(args.poll_seconds)
    except Exception as error:
        write_status(status_path, 'failed', **details, error=str(error))
        raise


if __name__ == '__main__':
    main()
