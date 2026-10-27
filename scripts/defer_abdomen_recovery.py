"""Start the prespecified abdominal recovery screen after the fair suite ends.

This controller does no training while the two-GPU fair comparison is active.
Its child owns the GPU lock and validates the fair-suite artifacts again before
loading any checkpoint. Status and logs live beside the experiment directory.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


CODE_ROOT = Path(__file__).resolve().parents[1]


def write_status(path, state, **details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    value = dict(state=state, pid=os.getpid(), started_at=previous.get('started_at', now),
                 updated_at=now, **details)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fair_state(fair_root, research_status):
    """Return ready, waiting, or a concrete failure reason from atomic files."""
    summary_path = fair_root / 'summary.json'
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            expected = {(kind, seed) for kind in ('shared', 'task_id', 'region')
                        for seed in (42, 43, 44)}
            runs = summary['runs']
            identities = {(row['kind'], row['seed']) for row in runs}
            workers = [json.loads((fair_root / f'worker_gpu{gpu}.json').read_text())
                       for gpu in (0, 1)]
            if (summary.get('test_evaluation_requested') is True
                    and summary.get('parallel', {}).get('training_barrier_verified') is True
                    and len(runs) == 9 and identities == expected
                    and all(row.get('test_evaluated') is True for row in runs)
                    and all(worker.get('state') == 'complete' for worker in workers)
                    and all((fair_root / 'shards' / f'gpu{gpu}' / 'test_complete.json').is_file()
                            for gpu in (0, 1))):
                return 'ready'
        except (KeyError, ValueError, OSError):
            pass
    for gpu in (0, 1):
        path = fair_root / f'worker_gpu{gpu}.json'
        if path.is_file():
            state = json.loads(path.read_text()).get('state', '')
            if state in ('failed', 'blocked_by_peer'):
                return f'fair GPU {gpu} worker {state}'
    if research_status.is_file():
        stage = json.loads(research_status.read_text()).get('stage', '')
        if stage.startswith('failed:'):
            return f'fair controller {stage}'
    return 'waiting'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--research-status', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=1)
    parser.add_argument('--poll-seconds', type=int, default=45)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error('--poll-seconds must be positive')
    fair_root = args.fair_root.resolve()
    output = args.output.resolve()
    # Keep controller files outside the immutable experiment output. Preparation
    # owns creation of that directory and may reject unknown existing files.
    controller = output.parent / f'{output.name}-logs'
    status = controller / 'controller_status.json'
    log_path = controller / 'recovery.log'
    write_status(status, 'waiting_fair', fair_root=str(fair_root), gpu=args.gpu,
                 log=str(log_path))
    try:
        while True:
            state = fair_state(fair_root, args.research_status)
            if state == 'ready':
                break
            if state != 'waiting':
                raise RuntimeError(state)
            time.sleep(args.poll_seconds)
        common = [sys.executable, '-m', 'unet_moe.abdomen_recovery',
                  '--fair-root', str(fair_root), '--output', str(output)]
        preparation = [*common, '--prepare']
        environment = dict(os.environ, OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        write_status(status, 'preparing', fair_root=str(fair_root), gpu=args.gpu,
                     log=str(log_path), command=preparation)
        with log_path.open('ab') as stream:
            prepared = subprocess.run(preparation, cwd=CODE_ROOT, env=environment,
                                      stdout=stream, stderr=subprocess.STDOUT, check=False)
        if prepared.returncode:
            raise RuntimeError(f'recovery preparation exited {prepared.returncode}; see {log_path}')
        command = [*common, '--gpu', str(args.gpu)]
        write_status(status, 'running', fair_root=str(fair_root), gpu=args.gpu,
                     log=str(log_path), command=command)
        gpu_environment = dict(environment, CUDA_VISIBLE_DEVICES=str(args.gpu))
        with log_path.open('ab') as stream:
            completed = subprocess.run(command, cwd=CODE_ROOT, env=gpu_environment,
                                       stdout=stream, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f'recovery process exited {completed.returncode}; see {log_path}')
        write_status(status, 'complete', fair_root=str(fair_root), gpu=args.gpu,
                     log=str(log_path), exit_code=0)
    except Exception as error:
        write_status(status, 'failed', fair_root=str(fair_root), gpu=args.gpu,
                     log=str(log_path), error=str(error))
        raise


if __name__ == '__main__':
    main()
