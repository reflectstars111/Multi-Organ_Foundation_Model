"""Run frozen Q03 after a verified Q01 drain under an explicit priority plan.

The adapter replaces only the frozen controller's reference-completion gate.
All preflights, GPU locks, twelve training jobs, the test barrier, and summary
verification remain in the original controller. Q01 is resumed by its external
supervisor; its incomplete training matrix is never represented as complete.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


ADAPTER_SOURCE_KEY = 'scripts/run_region_normalization_priority.py'
ORDER = ['finish_active_q01', 'q03', 'remaining_q01']
ARMS = ['shared_gn', 'domain_gn', 'shared_bn', 'domain_bn']
ARTIFACTS = {'records.json': 'record_sha256', 'sampling_plan.json': 'draws_sha256',
             'data_content.json': 'content_sha256', 'train_statistics.json': 'statistics_sha256'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def signed_json(path, field):
    value = read_json(path)
    expected = dict(value)
    stored = expected.pop(field, None)
    if not isinstance(stored, str) or digest(expected) != stored:
        raise ValueError(f'{field} differs from its content: {path}')
    return value


def verify_manifest(path, q03_root, q01_root):
    plan = signed_json(path, 'plan_sha256')
    if plan.get('schema') != 'medical_queue_priority_v1' or plan.get('order') != ORDER:
        raise ValueError('Only the explicitly authorized Q01-drain/Q03/Q01 order is supported')
    for key, root in (('q01_root', q01_root), ('q03_root', q03_root)):
        recorded = plan.get(key)
        if not isinstance(recorded, str) or Path(recorded).resolve() != root:
            raise ValueError(f'Priority manifest {key} differs from the requested root')
    sources = plan.get('source_files')
    if (not isinstance(sources, dict) or
            sources.get(ADAPTER_SOURCE_KEY) != file_sha(Path(__file__).resolve())):
        raise ValueError('Priority adapter source differs from the signed plan')
    return plan


def verify_handoff(path, plan):
    handoff = signed_json(path, 'handoff_sha256')
    for key in ('plan_sha256', 'q01_root', 'q03_root',
                'q01_protocol_sha256', 'q03_protocol_sha256'):
        if handoff.get(key) != plan.get(key):
            raise ValueError(f'Handoff {key} differs from the priority plan')
    if (handoff.get('active_q01_verified') is not True or
            handoff.get('q01_controller_released') is not True):
        raise ValueError('Q01 workers must be verified and its controller released before Q03')
    return handoff


def verify_scope(protocol):
    config = protocol.get('config', {})
    if (protocol.get('protocol_version') != 'region_domain_normalization_v1'
            or type(config.get('epochs')) is not int or config['epochs'] != 80
            or config.get('models') != ['region'] or config.get('seeds') != [42, 43, 44]
            or config.get('arms') != ARMS):
        raise ValueError('Priority scope is limited to the frozen twelve Q03 runs')


def verify_snapshot(snapshot, protocol):
    """Verify scripts as well as model files before executing a frozen import."""
    sources = protocol.get('source_files')
    required = {'scripts/__init__.py', 'scripts/run_region_normalization.py',
                'unet_moe/__init__.py', 'unet_moe/region_normalization_suite.py'}
    if (not isinstance(sources, dict) or not required.issubset(sources)
            or digest(sources) != protocol.get('source_sha256')):
        raise ValueError('Frozen source manifest is incomplete or has changed')
    for name, expected in sources.items():
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('Frozen source paths must be relative')
        source = (snapshot / name).resolve()
        if not source.is_relative_to(snapshot) or file_sha(source) != expected:
            raise ValueError(f'Frozen source differs: {name}')


@contextmanager
def snapshot_imports(snapshot):
    """Isolate cached package imports as well as search paths from the checkout."""
    def owned(name):
        return name in ('scripts', 'unet_moe') or name.startswith(('scripts.', 'unet_moe.'))

    saved_modules = {name: module for name, module in sys.modules.copy().items() if owned(name)}
    saved_path = list(sys.path)
    for name in saved_modules:
        del sys.modules[name]
    sys.path[:] = [str(snapshot)] + [path for path in sys.path if path != str(snapshot)]
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        importlib.invalidate_caches()


def load_frozen_controller(snapshot):
    path = snapshot / 'scripts/run_region_normalization.py'
    spec = importlib.util.spec_from_file_location('_q03_priority_frozen_controller', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load frozen controller: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_reference_data(q01_root, protocol, plan):
    reference = signed_json(q01_root / 'protocol.json', 'protocol_sha256')
    if (reference.get('protocol_version') != 'q01_organ_weighted_v1' or
            reference['protocol_sha256'] != plan.get('q01_protocol_sha256') or
            reference['protocol_sha256'] != protocol.get('reference_protocol_sha256')):
        raise ValueError('Q01 reference protocol differs from the frozen identity')
    for filename, key in ARTIFACTS.items():
        with (q01_root / filename).open() as stream:
            content = json.load(stream)
        if reference.get(key) != protocol.get(key) or digest(content) != reference.get(key):
            raise ValueError(f'Q01 and Q03 frozen data identity differs: {filename}')
    return reference


def run(output, after_root, priority_manifest, gpus=(0, 1), poll_seconds=15):
    root, reference_root = Path(output).resolve(), Path(after_root).resolve()
    manifest_path = Path(priority_manifest).resolve()
    plan = verify_manifest(manifest_path, root, reference_root)
    protocol = signed_json(root / 'protocol.json', 'protocol_sha256')
    verify_scope(protocol)
    if (protocol['protocol_sha256'] != plan.get('q03_protocol_sha256') or
            protocol.get('reference_root') != str(reference_root)):
        raise ValueError('Q03 protocol differs from the priority plan or reference root')
    handoff_path = manifest_path.parent / 'handoff.json'
    verify_handoff(handoff_path, plan)
    snapshot = (root / 'source_snapshot').resolve()
    verify_snapshot(snapshot, protocol)
    with snapshot_imports(snapshot):
        frozen = load_frozen_controller(snapshot)
        suite = importlib.import_module('unet_moe.region_normalization_suite')
        if not Path(suite.__file__).resolve().is_relative_to(snapshot):
            raise ValueError('Normalization suite was not imported from the frozen snapshot')

        class PriorityController(frozen.Controller):
            def wait_for_reference(self):
                self.update('verifying_priority_handoff',
                            reference_policy='verified_priority_handoff', priority_order=ORDER,
                            priority_manifest=str(manifest_path), priority_plan_sha256=plan['plan_sha256'])
                current_plan = verify_manifest(manifest_path, root, reference_root)
                if current_plan != plan:
                    raise ValueError('Priority manifest changed while starting its controller')
                handoff = verify_handoff(handoff_path, plan)
                verified, *_ = suite.load_protocol(root, check_content=True)
                if verified != protocol:
                    raise ValueError('Q03 protocol changed while verifying its dependencies')
                verify_reference_data(reference_root, verified, plan)
                # Initialization fingerprints remain checked by the unchanged
                # frozen CUDA preflights and per-run training entry points.
                # Recheck authorization after the potentially long content scan.
                if verify_handoff(handoff_path, plan) != handoff:
                    raise ValueError('Priority handoff changed during data verification')
                self.update('priority_handoff_verified', reference_wait_reason=None,
                            priority_handoff_sha256=handoff['handoff_sha256'],
                            reference_data_verified=True, reference_training_completion_required=False)

        PriorityController(root, after_root=reference_root, gpus=gpus,
                           poll_seconds=poll_seconds).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-root', type=Path, required=True)
    parser.add_argument('--priority-manifest', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args(argv)
    run(args.output, args.after_root, args.priority_manifest,
        gpus=[int(value) for value in args.gpus.split(',')], poll_seconds=args.poll_seconds)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'run_region_normalization_priority: {type(error).__name__}: {error}',
              file=sys.stderr, flush=True)
        sys.exit(1)
