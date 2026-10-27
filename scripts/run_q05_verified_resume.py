"""Recover Q05 with scoped deterministic *verification*, leaving training frozen.

The original preflight assumed bitwise CUDA backward replay without enabling
deterministic algorithms. Its two memory/admission updates still run in the
original numerical mode; only the save/reload/continuation proof enables strict
determinism. No tolerances or original equality checks are removed. A separate
manifest binds this overlay, and the original controller manages locks, workers,
dependencies, resume, and the twelve-training-runs-before-test barrier.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

SCHEMA = 'q05_scoped_deterministic_preflight_v1'
SCOPE = dict(
    admission_updates='original_algorithm_flags_with_verification_cublas_workspace',
    strict_reload_and_continuation='deterministic_algorithms',
    formal_training='unchanged_frozen_source_and_numerical_mode',
    formal_training_bitwise_reproducible=False,
    cublas_workspace_config=':4096:8',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_manifest(path, root):
    root, path = Path(root).resolve(), Path(path).resolve()
    value = read(path)
    payload = {k: v for k, v in value.items() if k != 'manifest_sha256'}
    if (value.get('schema') != SCHEMA or value.get('manifest_sha256') != digest(payload)
            or value.get('root') != str(root) or value.get('scope') != SCOPE
            or value.get('overlay_sha256') != sha(__file__)
            or value.get('protocol_file_sha256') != sha(root / 'protocol.json')
            or value.get('selection_file_sha256') != sha(root / 'main_loss_selection.json')):
        raise ValueError('Q05 recovery manifest/source/protocol/selection differs')
    return value


def load_frozen(root):
    snapshot = Path(root).resolve() / 'source_snapshot'
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location(
        '_q05_verified_frozen_controller', snapshot / 'scripts/run_moe_refinement.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = importlib.import_module('unet_moe.refinement_suite')
    if not Path(suite.__file__).resolve().is_relative_to(snapshot):
        raise ValueError('Q05 imported mutable source instead of its frozen snapshot')
    return module, suite


def scoped_preflight(suite, args, protocol, records, draws, weights, manifest):
    """Exercise native updates first, then the unmodified exact restoration test."""
    torch = suite.torch
    if torch.device(args.device).type != 'cuda':
        raise ValueError('Q05 recovery requires real CUDA preflight')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != SCOPE['cublas_workspace_config']:
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization')
    old_mode = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    if old_mode:
        raise ValueError('Admission updates must start in the frozen native numerical mode')
    original = suite.train_effective_batches
    modes, backend_flags = [], []

    def checked_updates(*a, **kw):
        index = len(modes)
        mode = torch.are_deterministic_algorithms_enabled()
        if index >= 4 or mode != (index >= 2):
            raise ValueError('Unexpected preflight update count or determinism mode')
        backend_flags.append(dict(deterministic_algorithms=mode,
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
        result = original(*a, **kw)
        if result['optimizer_steps'] != 1 or result['sample_count'] != 16:
            raise ValueError('Preflight must execute one real batch16 update per call')
        modes.append(mode)
        if len(modes) == 2:
            torch.use_deterministic_algorithms(True, warn_only=False)
        return result

    suite.train_effective_batches = checked_updates
    report_path = Path(args.output) / 'preflight' / f'{args.kind}_{args.arm}.json'
    try:
        report = suite.preflight(args, protocol, records, draws, weights)
        if modes != [False, False, True, True]:
            raise ValueError('Missing native admission or exact deterministic continuation updates')
        report.update(recovery_schema=SCHEMA, recovery_manifest_sha256=manifest['manifest_sha256'],
                      numerical_scope=SCOPE, update_deterministic_modes=modes,
                      update_backend_flags=backend_flags,
                      torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
        suite.fair._write_json(report_path, report)
        return report
    except BaseException as error:
        # Never leave a passing original report after an overlay-level failure.
        report = read(report_path) if report_path.exists() else {}
        report.update(passed=False, error=f'{type(error).__name__}: {error}',
                      recovery_schema=SCHEMA, recovery_manifest_sha256=manifest['manifest_sha256'],
                      numerical_scope=SCOPE, update_deterministic_modes=modes,
                      update_backend_flags=backend_flags)
        suite.fair._write_json(report_path, report)
        raise
    finally:
        suite.train_effective_batches = original
        torch.use_deterministic_algorithms(old_mode, warn_only=old_warn)


def validate_scoped_report(report, manifest):
    if (report.get('recovery_schema') != SCHEMA
            or report.get('recovery_manifest_sha256') != manifest['manifest_sha256']
            or report.get('numerical_scope') != SCOPE
            or report.get('update_deterministic_modes') != [False, False, True, True]):
        raise ValueError('Preflight lacks bound scoped determinism evidence')
    flags = report.get('update_backend_flags', [])
    if len(flags) != 4 or any(
            f.get('deterministic_algorithms') != (i >= 2)
            or f.get('warn_only') is not False
            or f.get('cudnn_deterministic') is not True
            or f.get('cudnn_benchmark') is not False
            or f.get('cublas_workspace_config') != SCOPE['cublas_workspace_config']
            for i, f in enumerate(flags)):
        raise ValueError('Preflight backend flag evidence differs')


def controller_type(frozen, manifest_path):
    class VerifiedController(frozen.Controller):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recovery_manifest = validate_manifest(manifest_path, self.root)
            self.status.update(recovery_schema=SCHEMA,
                               recovery_manifest_sha256=self.recovery_manifest['manifest_sha256'],
                               numerical_scope=SCOPE)

        def command(self, task=None, action=None, device='cuda:0'):
            phase = action or (task[3] if task else None)
            if phase == 'preflight':
                kind, seed, arm, _ = task
                return [sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--output', str(self.root), '--manifest', str(Path(manifest_path).resolve()),
                        '--kind', kind, '--seed', str(seed), '--arm', arm, '--device', device]
            return super().command(task, action=action, device=device)

        def read_preflight(self, kind, arm):
            manifest = validate_manifest(manifest_path, self.root)
            if manifest != self.recovery_manifest:
                raise ValueError('Recovery manifest changed while controller was running')
            report = read(self.root / 'preflight' / f'{kind}_{arm}.json')
            validate_scoped_report(report, manifest)
            return super().read_preflight(kind, arm)

    return VerifiedController


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--kind', choices=('region', 'task_id'))
    parser.add_argument('--arm', choices=('control', 'refined'))
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--after-root', type=Path)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    manifest = validate_manifest(args.manifest, args.output)
    if args.worker:
        if not args.kind or not args.arm:
            parser.error('--worker requires --kind and --arm')
        # This is a fresh preflight subprocess. Do not set this in the controller
        # environment, which is inherited by unmodified formal training workers.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = SCOPE['cublas_workspace_config']
    elif not args.reference_root or not args.after_root:
        parser.error('Controller requires --reference-root and --after-root')
    frozen, suite = load_frozen(args.output)
    if args.worker:
        protocol, records, draws, weights = suite.load_protocol(args.output, check_content=True)
        protocol = suite.with_selection(args.output, protocol)
        scoped_preflight(suite, args, protocol, records, draws, weights, manifest)
        return 0
    cls = controller_type(frozen, args.manifest)
    return cls(args.output, args.reference_root, args.after_root,
               gpus=[int(v) for v in args.gpus.split(',')], poll_seconds=args.poll_seconds,
               retry_failed=args.retry_failed).run(once=args.once, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
