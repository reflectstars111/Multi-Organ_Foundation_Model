"""Queue common-union scoring after the fair and single-output runs finish.

The controller holds GPU 0's fair lock through inference. It never starts
training or reads test images while either prerequisite is incomplete.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .defer_abdomen_recovery import fair_state, write_status
from .defer_single_channel import (gpu_free_mib, gpu_lock_path, run_command,
                                   validate_complete)


EXPECTED = {(kind, seed) for kind in ('shared', 'task_id', 'region')
            for seed in (42, 43, 44)}
MINIMUM_FREE_MIB = 22 * 1024


def prerequisite_state(fair_root, research_status, single_status, single_output):
    """Return a wait state or verify both experiments are complete and paired."""
    state = fair_state(fair_root, research_status)
    if state == 'waiting':
        return 'waiting_fair'
    if state != 'ready':
        raise RuntimeError(state)
    if not single_status.is_file():
        return 'waiting_single'
    controller = json.loads(single_status.read_text(encoding='utf-8'))
    if controller.get('state') == 'failed':
        raise RuntimeError(f'Single-channel controller failed: {controller.get("error", "unknown error")}')
    if controller.get('state') != 'complete':
        return 'waiting_single'
    summary_path = validate_complete(single_output)
    if Path(controller.get('summary', '')).resolve() != summary_path.resolve():
        raise ValueError('Single-channel controller points to a different summary')
    single_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    fair_summary = json.loads((fair_root / 'summary.json').read_text(encoding='utf-8'))
    single_protocol = json.loads((single_output / 'protocol.json').read_text(encoding='utf-8'))
    record_hash = fair_summary['record_sha256']
    if (single_summary.get('record_sha256') != record_hash
            or single_protocol.get('record_sha256') != record_hash
            or single_protocol.get('reference_records_sha256') != record_hash
            or single_protocol.get('config', {}).get('size') != 512):
        raise ValueError('Single-output result is not paired to the completed fair records and 512px input')
    return 'ready'


def completed_output(output, fair_root):
    """Allow a clean controller restart after successful evaluation."""
    summary_path = output / 'summary.json'
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    fair_summary = json.loads((fair_root / 'summary.json').read_text(encoding='utf-8'))
    runs = summary.get('runs', [])
    if (summary.get('protocol') != protocol
            or protocol.get('schema') != 'fair_models_union_evaluation_v1'
            or protocol.get('split') != 'test'
            or Path(protocol.get('fair_root', '')).resolve() != fair_root.resolve()
            or protocol.get('fair_record_sha256') != fair_summary.get('record_sha256')
            or len(runs) != 9
            or {(row.get('kind'), row.get('seed')) for row in runs} != EXPECTED
            or any('overall' not in row.get('metrics', {}) for row in runs)
            or any(not (output / f'{row["kind"]}_seed_{row["seed"]}.json').is_file()
                   for row in runs)):
        raise ValueError('Existing union evaluation summary is incomplete or mismatched')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, required=True)
    parser.add_argument('--research-status', type=Path, required=True)
    parser.add_argument('--single-status', type=Path, required=True)
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
    with (controller_dir / '.controller.lock').open('a+') as controller_lock:
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Union evaluation controller already running') from error
        write_status(status, 'waiting_fair', fair_root=str(fair_root),
                     single_output=str(single_output), gpu=0, log=str(log_path))
        try:
            while True:
                state = prerequisite_state(fair_root, args.research_status,
                                           args.single_status, single_output)
                if state == 'ready':
                    break
                write_status(status, state, fair_root=str(fair_root),
                             single_output=str(single_output), gpu=0, log=str(log_path))
                time.sleep(args.poll_seconds)
            if completed_output(output, fair_root):
                write_status(status, 'complete', fair_root=str(fair_root),
                             single_output=str(single_output), gpu=0, log=str(log_path),
                             summary=str(output / 'summary.json'), reused=True)
                return
            if output.exists():
                raise FileExistsError(f'Incomplete output exists; inspect before restarting: {output}')
            command = [sys.executable, '-m', 'unet_moe.union_evaluate',
                       '--fair-root', str(fair_root), '--split', 'test',
                       '--output', str(output), '--device', 'cuda:0',
                       '--batch-size', '2', '--workers', '2']
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
                            write_status(status, 'evaluating', fair_root=str(fair_root),
                                         single_output=str(single_output), gpu=0,
                                         gpu_free_mib_at_start=free,
                                         log=str(log_path), command=command)
                            exit_code = run_command(command, log_path, environment, gpu_lock)
                            if exit_code:
                                raise RuntimeError(f'Union evaluation exited {exit_code}; see {log_path}')
                            if not completed_output(output, fair_root):
                                raise ValueError('Union evaluator exited without a complete nine-run summary')
                            write_status(status, 'complete', fair_root=str(fair_root),
                                         single_output=str(single_output), gpu=0,
                                         log=str(log_path), summary=str(output / 'summary.json'),
                                         exit_code=0)
                            return
                write_status(status, 'waiting_gpu', fair_root=str(fair_root),
                             single_output=str(single_output), gpu=0,
                             minimum_free_mib=MINIMUM_FREE_MIB,
                             observed_free_mib=free, log=str(log_path))
                time.sleep(args.poll_seconds)
        except Exception as error:
            write_status(status, 'failed', fair_root=str(fair_root),
                         single_output=str(single_output), gpu=0,
                         log=str(log_path), error=str(error))
            raise


if __name__ == '__main__':
    main()
