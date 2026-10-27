"""Immutable Q04 inputs and atomic artifacts, independent of model construction."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from ..fair_suite import _digest


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_protocol(root, *, verify_content=True):
    root = Path(root)
    protocol = read_json(root / 'protocol.json')
    unsigned = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    if _digest(unsigned) != protocol['protocol_sha256']:
        raise ValueError('Q04 protocol hash mismatch')
    records = read_json(root / 'records.json')
    if _digest(records) != protocol['record_sha256']:
        raise ValueError('Frozen Q04 records changed')
    for name in protocol['input_files']:
        if file_sha(root / name) != protocol['input_files'][name]:
            raise ValueError(f'Frozen Q04 input changed: {name}')
    if file_sha(root / 'environment.lock.txt') != protocol['environment_sha256']:
        raise ValueError('Frozen Q04 environment record changed')
    for name, version in protocol.get('runtime_versions', {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Q04 runtime dependency changed: {name} must be {version}')
    source = Path(__file__).resolve().parents[2]
    for relative, digest in protocol['source_files'].items():
        if file_sha(source / relative) != digest:
            raise ValueError(f'Q04 source changed: {relative}; run the frozen snapshot')
    for path, digest in protocol['assets'].items():
        if file_sha(path) != digest:
            raise ValueError(f'Frozen Q04 asset changed: {path}')
    if verify_content:
        for path, item in read_json(root / 'data_content.json').items():
            if Path(path).stat().st_size != item['bytes'] or file_sha(path) != item['sha256']:
                raise ValueError(f'Train/validation content changed: {path}')
    return protocol, records


def run_folder(root, method, seed):
    return Path(root) / method / f'seed_{seed}'


def assert_checkpoint(checkpoint, protocol, method, seed):
    if (checkpoint.get('protocol_sha256') != protocol['protocol_sha256']
            or checkpoint.get('method') != method or checkpoint.get('seed') != seed):
        raise ValueError('Q04 checkpoint method/seed/protocol mismatch')


def assert_matrix_complete(root, protocol):
    # Summary flags alone cannot open test. Each last checkpoint is validated.
    # Reuse the same complete history, sampling and selected-checkpoint
    # validator used by the controller. A manually invoked evaluate command
    # must respect the same barrier, not only an epoch=80 field.
    from ..external_baseline_suite import verify_run
    for method in protocol['config']['methods']:
        for seed in protocol['config']['seeds']:
            verify_run(root, protocol, method, seed)
