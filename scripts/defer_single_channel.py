"""Queue the 512px single-output union U-Net after the frozen fair suite.

Training and its explicit final test share the current fair suite's image
draws, budget, split, and hyperparameters. The new target is a single union
foreground mask, so the results have their own metric name and directory.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .defer_abdomen_recovery import CODE_ROOT, fair_state, write_status


# The paired 512px U-Net has used roughly 31 GiB including loader processes.
# Leave headroom for the existing external GPU0 process before starting.
MINIMUM_FREE_MIB = 34 * 1024


def gpu_lock_path(gpu):
    return Path(f'/tmp/multi_organ_fair_gpu_{gpu}.lock')


def gpu_free_mib(gpu):
    result = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
         '--format=csv,noheader,nounits'], text=True).strip()
    return int(result)


def run_command(command, log_path, environment, lock=None):
    with Path(log_path).open('ab') as stream:
        return subprocess.run(command, cwd=CODE_ROOT, env=environment,
                              stdout=stream, stderr=subprocess.STDOUT, check=False,
                              pass_fds=() if lock is None else (lock.fileno(),)).returncode


def validate_complete(output):
    summary_path = Path(output) / 'summary.json'
    summary = json.loads(summary_path.read_text())
    runs = summary.get('runs', [])
    if (summary.get('protocol_version') != 'single_channel_paper5_union_v1'
            or summary.get('test_evaluation_requested') is not True
            or len(runs) != 3 or {run.get('seed') for run in runs} != {42, 43, 44}
            or any(run.get('epochs_completed') != 50 or run.get('test_evaluated') is not True
                   for run in runs)):
        raise ValueError('Single-channel summary lacks the completed three-seed test barrier')
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, required=True)
    parser.add_argument('--research-status', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    parser.add_argument('--poll-seconds', type=int, default=45)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error('--poll-seconds must be positive')
    fair_root = args.fair_root.resolve()
    output = args.output.resolve()
    controller = output.parent / f'{output.name}-logs'
    status_path = controller / 'controller_status.json'
    log_path = controller / 'single_channel.log'
    write_status(status_path, 'waiting_fair', fair_root=str(fair_root),
                 gpu=args.gpu, log=str(log_path))
    try:
        while True:
            state = fair_state(fair_root, args.research_status)
            if state == 'ready':
                break
            if state != 'waiting':
                raise RuntimeError(state)
            time.sleep(args.poll_seconds)
        common = [sys.executable, '-m', 'unet_moe.single_channel_suite',
                  '--data-root', str(args.data_root.resolve()), '--output', str(output),
                  '--reference-records', str(fair_root / 'shards/gpu1/records.json'),
                  '--seeds', '42,43,44', '--split-seed', '42',
                  '--size', '512', '--base', '16', '--batch-size', '4',
                  '--epochs', '50', '--quota-per-domain', '500',
                  '--lr', '0.0003', '--weight-decay', '0.0001',
                  '--lr-schedule', 'constant', '--min-lr-ratio', '0.1',
                  '--workers', '2', '--device', 'cuda:0']
        prepare = [*common, '--dry-run']
        if (output / 'protocol.json').is_file():
            prepare.append('--resume')
        write_status(status_path, 'preparing', fair_root=str(fair_root), gpu=args.gpu,
                     log=str(log_path), command=prepare)
        environment = dict(os.environ, OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                           CUDA_VISIBLE_DEVICES=str(args.gpu))
        if run_command(prepare, log_path, environment):
            raise RuntimeError(f'Single-channel preparation failed; see {log_path}')

        lock_path = gpu_lock_path(args.gpu)
        while True:
            with lock_path.open('a+') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    free = None
                else:
                    free = gpu_free_mib(args.gpu)
                    if free >= MINIMUM_FREE_MIB:
                        train = [*common, '--resume']
                        write_status(status_path, 'training', fair_root=str(fair_root),
                                     gpu=args.gpu, gpu_free_mib_at_start=free,
                                     log=str(log_path), command=train)
                        if run_command(train, log_path, environment, lock):
                            raise RuntimeError(f'Single-channel training failed; see {log_path}')
                        test = [*common, '--resume', '--evaluate-test']
                        write_status(status_path, 'testing', fair_root=str(fair_root),
                                     gpu=args.gpu, log=str(log_path), command=test)
                        if run_command(test, log_path, environment, lock):
                            raise RuntimeError(f'Single-channel test failed; see {log_path}')
                        summary_path = validate_complete(output)
                        write_status(status_path, 'complete', fair_root=str(fair_root),
                                     gpu=args.gpu, log=str(log_path),
                                     summary=str(summary_path))
                        return
            write_status(status_path, 'waiting_gpu', fair_root=str(fair_root),
                         gpu=args.gpu, minimum_free_mib=MINIMUM_FREE_MIB,
                         observed_free_mib=free, log=str(log_path))
            time.sleep(args.poll_seconds)
    except Exception as error:
        write_status(status_path, 'failed', fair_root=str(fair_root),
                     gpu=args.gpu, log=str(log_path), error=str(error))
        raise


if __name__ == '__main__':
    main()
