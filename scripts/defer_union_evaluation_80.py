"""Queue the 80-epoch models' common-union test after all paired runs finish.

GPU 0 is locked through inference. Both the nine structure-model tests and
three binary U-Net tests must finish before this evaluator opens test images.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .defer_abdomen_recovery import fair_state, write_status
from .defer_single_channel import gpu_free_mib, gpu_lock_path, run_command
from .defer_single_channel_80 import validate_complete
from unet_moe import union_evaluate_80 as evaluator


EXPECTED = set(evaluator.RUNS)
MINIMUM_FREE_MIB = 22 * 1024


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prerequisite_state(fair_root, research_status, single_status, single_output):
    """Wait for both barriers; validate all frozen artifacts before admission."""
    state = fair_state(fair_root, research_status)
    if research_status.is_file():
        controller = _json(research_status)
        if controller.get('state', '').startswith(('failed', 'blocked')):
            raise RuntimeError(f'80-epoch fair controller failed or blocked: {controller.get("error", "unknown error")}')
    if state == 'waiting':
        return 'waiting_fair'
    if state != 'ready':
        raise RuntimeError(state)
    if not single_status.is_file():
        return 'waiting_single'
    controller = _json(single_status)
    if controller.get('state', '').startswith(('failed', 'blocked')):
        raise RuntimeError(f'80-epoch single-channel controller failed or blocked: {controller.get("error", "unknown error")}')
    if controller.get('state') != 'complete':
        return 'waiting_single'
    # Expensive provenance checks happen once both controllers finish, not on
    # every waiting poll. The evaluator repeats them immediately before scoring.
    _, fair_summary, _ = evaluator.load_completed_fair(fair_root)
    summary_path = validate_complete(single_output, fair_summary['record_sha256'])
    if Path(controller.get('summary', '')).resolve() != summary_path.resolve():
        raise ValueError('80-epoch single-channel controller points to another summary')
    return 'ready'


def completed_output(output, fair_root):
    """Reuse only a complete evaluation with current evaluator provenance."""
    summary_path = output / 'summary.json'
    if not summary_path.is_file():
        return False
    summary = _json(summary_path)
    protocol = _json(output / 'protocol.json')
    fair_summary = _json(fair_root / 'summary.json')
    fair_plan = _json(fair_root / 'parallel_plan.json')
    runs = summary.get('runs', [])
    source_hashes = {
        'evaluator_sha256': hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest(),
        'shared_scoring_sha256': hashlib.sha256(Path(evaluator.common.__file__).read_bytes()).hexdigest(),
        'target_source_sha256': hashlib.sha256(Path(evaluator.single_channel_suite.__file__).read_bytes()).hexdigest(),
    }
    if (summary.get('protocol') != protocol
            or protocol.get('schema') != evaluator.SCHEMA
            or protocol.get('split') != 'test' or protocol.get('image_size') != 512
            or Path(protocol.get('fair_root', '')).resolve() != fair_root.resolve()
            or protocol.get('fair_record_sha256') != evaluator.EXPECTED_RECORD_SHA256
            or protocol.get('fair_record_sha256') != fair_summary.get('record_sha256')
            or protocol.get('fair_source_sha256') != fair_summary.get('source_sha256')
            or protocol.get('fair_plan_sha256') != fair_plan.get('plan_sha256')
            or any(protocol.get(key) != value for key, value in source_hashes.items())
            or len(runs) != len(EXPECTED)
            or {(row.get('kind'), row.get('seed')) for row in runs} != EXPECTED
            or any('overall' not in row.get('metrics', {}) for row in runs)):
        raise ValueError('Existing 80-epoch union summary is incomplete or mismatched')
    for run in runs:
        path = output / f'{run["kind"]}_seed_{run["seed"]}.json'
        if not path.is_file():
            raise ValueError('Existing 80-epoch union output lacks a per-run result')
        detail = _json(path)
        if (any(detail.get(key) != value for key, value in run.items())
                or len(detail.get('per_image', [])) != protocol.get('images')):
            raise ValueError('Existing 80-epoch union result differs from its summary')
    if summary.get('across_seeds') != evaluator.common.summarize_seeds(runs):
        raise ValueError('Existing 80-epoch union seed statistics differ from its runs')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, required=True)
    parser.add_argument('--research-status', type=Path, required=True,
                        help='80-epoch fair controller_status.json')
    parser.add_argument('--single-status', type=Path, required=True,
                        help='80-epoch single-channel controller_status.json')
    parser.add_argument('--single-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=45)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error('--poll-seconds must be positive')
    fair_root = args.fair_root.resolve()
    single_output = args.single_output.resolve()
    output = args.output.resolve()
    controller_dir = output.parent / f'{output.name}-logs'
    controller_dir.mkdir(parents=True, exist_ok=True)
    status = controller_dir / 'controller_status.json'
    log_path = controller_dir / 'evaluation.log'
    details = dict(fair_root=str(fair_root), single_output=str(single_output),
                   gpu=0, log=str(log_path))
    with (controller_dir / '.controller.lock').open('a+') as controller_lock:
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('80-epoch union evaluation controller already running') from error
        write_status(status, 'waiting_fair', **details)
        try:
            while True:
                state = prerequisite_state(fair_root, args.research_status,
                                           args.single_status, single_output)
                if state == 'ready':
                    break
                write_status(status, state, **details)
                time.sleep(args.poll_seconds)
            if completed_output(output, fair_root):
                write_status(status, 'complete', **details,
                             summary=str(output / 'summary.json'), reused=True)
                return
            if output.exists():
                raise FileExistsError(f'Incomplete output exists; inspect before restarting: {output}')
            command = [sys.executable, '-m', 'unet_moe.union_evaluate_80',
                       '--fair-root', str(fair_root), '--split', 'test',
                       '--output', str(output), '--device', 'cuda:0',
                       '--batch-size', '2', '--workers', '0']
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                               OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
            while True:
                with gpu_lock_path(0).open('a+') as gpu_lock:
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        free = None
                    else:
                        free = gpu_free_mib(0)
                        if free >= MINIMUM_FREE_MIB:
                            write_status(status, 'evaluating', **details,
                                         gpu_free_mib_at_start=free, command=command)
                            exit_code = run_command(command, log_path, environment, gpu_lock)
                            if exit_code:
                                raise RuntimeError(f'80-epoch union evaluation exited {exit_code}; see {log_path}')
                            if not completed_output(output, fair_root):
                                raise ValueError('80-epoch evaluator exited without a complete nine-run summary')
                            write_status(status, 'complete', **details,
                                         summary=str(output / 'summary.json'), exit_code=0)
                            return
                write_status(status, 'waiting_gpu', **details,
                             minimum_free_mib=MINIMUM_FREE_MIB, observed_free_mib=free)
                time.sleep(args.poll_seconds)
        except Exception as error:
            write_status(status, 'failed', **details, error=str(error))
            raise


if __name__ == '__main__':
    main()
