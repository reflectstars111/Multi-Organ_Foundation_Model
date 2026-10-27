"""Frozen, exact-path compatibility adapter for four legacy Region evaluations.

This adapter never edits a split, checkpoint, old protocol or source snapshot.
It verifies an explicit old-path -> existing-file manifest, temporarily resolves
only those image opens, and executes the original frozen evaluator unchanged.
An optional separately hashed, user-authorized HC18 label policy evaluates the
legacy checkpoints against filled regions. Such results explicitly disclose the
change from the checkpoint's original contour targets; it is not a path repair.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re


SCHEMA = 'legacy_region_exact_path_repair_v1'
AMENDMENT_SCHEMA = 'hc18_filled_region_evaluation_amendment_v1'
AMENDMENT_VARIANT = 'hc18_filled_eval_after_contour_training'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_bytes(item, data_root):
    """Recheck containment and bytes at use time, including symlink changes."""
    path = Path(item['path'])
    if not path.is_absolute() or '..' in path.parts or not path.resolve().is_relative_to(data_root):
        raise ValueError(f'Mapped file escapes data_root: {path}')
    if not path.is_file():
        raise ValueError(f'Mapped file is missing: {path}')
    content = path.read_bytes()
    if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
        raise ValueError(f'Mapped file bytes changed: {path}')
    return content


def validate_amendment(manifest, records):
    """An amendment changes all saved HC18 evaluation masks, and only those."""
    amendment = manifest.get('label_policy_amendment')
    if amendment is None:
        return set()
    expected = {row['mask'] for row in records if row['split'] in ('val', 'test')
                and row.get('dataset') == 'Fetal_HC' and row['task'] == 8}
    split_counts = {split: len({row['mask'] for row in records if row['split'] == split
                               and row.get('dataset') == 'Fetal_HC' and row['task'] == 8})
                    for split in ('val', 'test')}
    if (amendment.get('schema') != AMENDMENT_SCHEMA
            or amendment.get('authorization') != '需要的就是使用填充后的区域标签'
            or amendment.get('original_training_targets') != 'HC18 contour'
            or amendment.get('evaluation_targets') != 'HC18 filled region'):
        raise ValueError('Unsupported or unauthorized HC18 label policy amendment')
    if amendment.get('amendment_sha256') != digest({
            key: value for key, value in amendment.items() if key != 'amendment_sha256'}):
        raise ValueError('HC18 label policy amendment changed')
    if (not expected or amendment.get('amended_paths') != sorted(expected)
            or type(amendment.get('count')) is not int or amendment['count'] != len(expected)
            or amendment.get('split_counts') != split_counts):
        raise ValueError('HC18 amendment must cover exactly all saved validation/test HC18 masks')
    for row in records:
        if row['image'] in expected or (row['mask'] in expected
                and (row.get('dataset') != 'Fetal_HC' or row['task'] != 8)):
            raise ValueError('HC18 amendment would alter images or another task')
    return expected


def validate_manifest(root, manifest_path):
    """Validate the complete frozen mapping without changing evaluation outputs."""
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    script = Path(__file__).resolve()
    manifest = read(manifest_path)
    if script.parent != manifest_path.parent:
        raise ValueError('Run the frozen repair script beside its manifest')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported repair manifest schema')
    if manifest.get('repair_manifest_sha256') != digest({
            k: v for k, v in manifest.items() if k != 'repair_manifest_sha256'}):
        raise ValueError('Repair manifest changed')
    if manifest.get('repair_script_sha256') != file_sha(script):
        raise ValueError('Repair script changed')
    protocol = read(root / 'protocol.json')
    if manifest.get('evaluation_protocol_sha256') != protocol.get('protocol_sha256'):
        raise ValueError('Repair belongs to another evaluation protocol')
    ids = manifest.get('allowed_job_ids', [])
    if len(ids) != 4 or len(set(ids)) != 4 or not set(ids).issubset(protocol['job_ids']):
        raise ValueError('Repair must whitelist exactly four original jobs')
    if set(manifest.get('job_fingerprints', {})) != set(ids):
        raise ValueError('Every allowed job needs its original fingerprint')
    records_path = Path(manifest['records_path'])
    if not records_path.is_absolute() or file_sha(records_path) != manifest['records_file_sha256']:
        raise ValueError('Original records bytes changed')
    records = read(records_path)
    if isinstance(records, dict):
        records = records['records']
    paths = {row[key] for row in records if row['split'] in ('val', 'test')
             for key in ('image', 'mask')}
    mapping = manifest.get('path_mapping', {})
    if not paths or set(mapping) != paths:
        raise ValueError('Mapping must cover exactly the original validation/test paths')
    data_root = Path(manifest['data_root'])
    if not data_root.is_absolute() or not data_root.is_dir():
        raise ValueError('data_root must be an existing absolute directory')
    data_root = data_root.resolve()
    prefix = PureWindowsPath(manifest['windows_data_prefix'])
    if not prefix.parts or '..' in prefix.parts:
        raise ValueError('Invalid original Windows data prefix')
    amended_paths = validate_amendment(manifest, records)
    destinations = set()
    for old, item in mapping.items():
        if (not isinstance(old, str) or not old or '\x00' in old
                or '..' in PureWindowsPath(old).parts):
            raise ValueError(f'Invalid original path: {old!r}')
        if (set(item) != {'path', 'sha256', 'bytes'} or type(item['bytes']) is not int
                or item['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
            raise ValueError(f'Invalid mapping fingerprint: {old}')
        destination = Path(item['path']).resolve()
        if destination in destinations:
            raise ValueError(f'Ambiguous mapped destination: {destination}')
        destinations.add(destination)
        try:
            relative = PureWindowsPath(old).relative_to(prefix)
        except ValueError as error:
            raise ValueError(f'Original path is outside its saved data prefix: {old}') from error
        parts = list(relative.parts)
        if old in amended_paths:
            if (len(parts) != 5 or parts[0] != '胎儿头围测量数据集 (Fetal HC - HC18 Challenge)'
                    or parts[1:4] != ['processed_png', 'masks', 'train']
                    or not parts[4].endswith('.png')):
                raise ValueError(f'HC18 amendment is not an original contour mask: {old}')
            parts[2] = 'masks_filled'
        expected_destination = data_root.joinpath(*parts)
        if Path(item['path']) != expected_destination:
            raise ValueError(f'Mapping changed the original suffix without an allowed amendment: {old}')
        checked_bytes(item, data_root)
    # Validate saved job bindings even during read-only dry runs. The original
    # scheduler can reproduce these manifests when the four failures are retried.
    for job_id in ids:
        validate_job(read(root / 'jobs' / job_id / 'job.json'), manifest)
    return manifest


def validate_job(job, manifest):
    job_id = job.get('job_id')
    if job_id not in manifest['allowed_job_ids']:
        raise ValueError('Job is not on the repair whitelist')
    if (job.get('family') != 'legacy_region' or job.get('experiment') != 'region_top1_v1'
            or job.get('kind') != 'region' or job.get('split') not in ('val', 'test')
            or job.get('selection') not in ('best', 'last')):
        raise ValueError('Repair only supports the original legacy Region jobs')
    fingerprint = digest({k: v for k, v in job.items() if k != 'job_manifest_sha256'})
    if (job.get('job_manifest_sha256') != fingerprint
            or manifest['job_fingerprints'][job_id] != fingerprint):
        raise ValueError('Original allowed job fingerprint changed')
    if (Path(job['records_path']).resolve() != Path(manifest['records_path']).resolve()
            or job.get('evaluation_protocol_sha256') != manifest['evaluation_protocol_sha256']):
        raise ValueError('Job records or evaluation protocol differs from repair scope')
    if job.get('config', {}).get('data_root') != manifest['windows_data_prefix']:
        raise ValueError('Windows data prefix differs from the original training config')


@contextmanager
def resolved_image_opens(manifest):
    """Resolve exact frozen keys; decoded bytes must match the frozen manifest."""
    from PIL import Image
    original = Image.open
    data_root = Path(manifest['data_root']).resolve()

    def open_image(fp, *args, **kwargs):
        if isinstance(fp, (str, bytes, os.PathLike)):
            key = os.fsdecode(os.fspath(fp))
            if key in manifest['path_mapping']:
                # Decode the bytes we verified, avoiding a second filesystem open
                # between hashing and PIL decoding. No pixels are transformed.
                content = checked_bytes(manifest['path_mapping'][key], data_root)
                return original(io.BytesIO(content), *args, **kwargs)
            if '\\' in key or PureWindowsPath(key).drive:
                raise ValueError(f'Windows image path is outside frozen mapping: {key}')
        return original(fp, *args, **kwargs)

    Image.open = open_image
    try:
        yield
    finally:
        Image.open = original


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime(root, manifest_path):
    manifest = validate_manifest(root, manifest_path)
    scheduler_path = Path(manifest['parallel_scheduler_script_path']).resolve()
    scheduler_manifest_path = Path(manifest['parallel_scheduler_manifest_path']).resolve()
    if scheduler_path.parent != scheduler_manifest_path.parent:
        raise ValueError('Scheduler script and manifest must share their frozen directory')
    if file_sha(scheduler_manifest_path) != manifest['parallel_scheduler_manifest_file_sha256']:
        raise ValueError('Frozen parallel scheduler manifest changed')
    scheduler_manifest = read(scheduler_manifest_path)
    if scheduler_manifest.get('scheduler_file_sha256') != file_sha(scheduler_path):
        raise ValueError('Frozen parallel scheduler changed')
    parallel = _load_module(scheduler_path, '_legacy_path_repair_parallel')
    original, scheduler_manifest, _ = parallel.load_original(root, scheduler_manifest_path)
    return manifest, parallel, original, scheduler_manifest


@contextmanager
def repaired_session_metadata(root, manifest):
    inference = importlib.import_module('unet_moe.reevaluation_inference')
    expected = Path(root).resolve() / 'source_snapshot/unet_moe/reevaluation_inference.py'
    if Path(inference.__file__).resolve() != expected:
        raise ValueError('Inference module is not from the original frozen evaluator')
    original = inference.load_evaluation

    def load(job, *args, **kwargs):
        validate_job(job, manifest)
        session = original(job, *args, **kwargs)
        amendment = manifest.get('label_policy_amendment')
        session.metadata['path_resolution'] = dict(
            method=('exact_frozen_windows_path_mapping_with_explicit_hc18_label_amendment'
                    if amendment else 'exact_frozen_windows_path_mapping'),
            repair_manifest_sha256=manifest['repair_manifest_sha256'],
            repair_script_sha256=manifest['repair_script_sha256'],
            records_file_sha256=manifest['records_file_sha256'],
            data_root=manifest['data_root'], mapping_entries=len(manifest['path_mapping']),
            original_records_preserved=True, original_source_snapshot_preserved=True,
            evaluation_targets_modified=bool(amendment),
            limitation='Historical data bytes were not hashed at training time; '
                       'these hashes bind the existing local files used for retrospective evaluation.')
        if amendment:
            session.metadata['label_policy_amendment'] = dict(amendment)
            session.metadata['evaluation_variant'] = AMENDMENT_VARIANT
            original_samples = session.samples

            def amended_samples():
                # Label the entire evaluation, not just HC18 rows. This keeps
                # every domain in the same macro denominator and makes the
                # original CSV/Markdown/seed grouping disclose the amendment.
                try:
                    for sample in original_samples:
                        if sample.variant is not None:
                            raise ValueError('Legacy Region sample unexpectedly has another evaluation variant')
                        yield replace(sample, variant=AMENDMENT_VARIANT)
                finally:
                    close = getattr(original_samples, 'close', None)
                    if close is not None:
                        close()

            session.samples = amended_samples()
        return session

    inference.load_evaluation = load
    try:
        yield
    finally:
        inference.load_evaluation = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair-manifest', type=Path,
                        default=Path(__file__).resolve().with_name('manifest.json'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--evaluate-job', type=Path)
    action.add_argument('--run-controller', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--poll-seconds', type=int, default=15)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    manifest, parallel, original, scheduler_manifest = load_runtime(root, args.repair_manifest)
    if args.dry_run:
        print(json.dumps(dict(read_only=True, repair_manifest_sha256=manifest['repair_manifest_sha256'],
                              allowed_job_ids=manifest['allowed_job_ids'],
                              mapping_entries=len(manifest['path_mapping'])), indent=2))
        return 0
    if args.evaluate_job:
        job = read(args.evaluate_job)
        validate_job(job, manifest)
        if args.evaluate_job.resolve() != root / 'jobs' / job['job_id'] / 'job.json':
            raise ValueError('Worker job manifest is outside its original output folder')
        with resolved_image_opens(manifest), repaired_session_metadata(root, manifest):
            original.evaluate_job(root, args.evaluate_job, device=args.device)
        return 0
    if args.repair_manifest.resolve() != Path(__file__).resolve().with_name('manifest.json'):
        raise ValueError('Controller workers require the default adjacent manifest.json')
    controller = parallel.Controller(root, original, scheduler_manifest, worker=Path(__file__).resolve(),
                                     gpus=[int(x) for x in args.gpus.split(',')],
                                     poll_seconds=args.poll_seconds)
    return controller.run()


if __name__ == '__main__':
    raise SystemExit(main())
