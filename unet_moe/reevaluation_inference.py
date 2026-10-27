"""Read-only checkpoint inference for retrospective segmentation evaluation.

Models and pixel preprocessing are imported from each experiment's saved source
tree under an isolated package name. No training checkpoint is modified and no
split is regenerated. Historical jobs without saved source are explicitly marked.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
from PIL import Image
import torch


@dataclass
class EvaluationSample:
    image_id: str
    domain: str
    structure: str | None
    task_id: int | None
    target: np.ndarray
    prediction: np.ndarray
    view: str
    source: str = ''
    variant: str | None = None


@dataclass
class EvaluationSession:
    metadata: dict
    samples: Iterator[EvaluationSample]

    def __iter__(self):
        return iter(self.samples)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=50000)
def _cached_sha(path, size, mtime_ns, ctime_ns):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def file_sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_sha(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _source_file(root, relative):
    path = root / relative
    if not path.is_file():
        path = root / 'unet_moe' / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Missing or invalid saved source path: {relative}')
    return path


def _load_package(root):
    root = Path(root).resolve()
    folder = root / 'unet_moe'
    name = '_reevaluation_' + hashlib.sha256(str(root).encode()).hexdigest()[:20]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, folder / '__init__.py',
                                                      submodule_search_locations=[str(folder)])
        if spec is None or spec.loader is None:
            raise ValueError(f'Cannot load model source: {folder}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name


def _module(package, module):
    return importlib.import_module(f'{package}.{module}')


def _record_digests(records):
    # Earlier training scripts used JSON's default whitespace and sometimes ASCII.
    return {_digest(records), *(
        hashlib.sha256(json.dumps(records, ensure_ascii=ascii_, sort_keys=True).encode()).hexdigest()
        for ascii_ in (False, True))}


def _assert_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f'{description} differs: {actual!r} != {expected!r}')


def _source_manifest(job, protocol):
    if job['family'] in ('abdomen_recovery', 'postprocess_screen') and job.get('model_protocol_path'):
        return _read(job['model_protocol_path']).get('source_files', {})
    return protocol.get('source_files', {})


def _inference_files(job):
    needed = {'data.py', 'model.py', 'region.py', 'normalization.py'}
    if job['kind'] in ('shared', 'independent', 'single_union'):
        needed.add('baseline.py')
    return needed


def _resolve_source(job, protocol):
    """Recover old implementations only from an exactly matching source manifest."""
    requested = Path(job['source_root']).resolve()
    sources = _source_manifest(job, protocol)
    if not sources:
        return requested, bool(job.get('source_frozen', False)), 'no_original_source_manifest'

    def matches(root, required=None):
        try:
            return all(file_sha(_source_file(root, relative)) == sha for relative, sha in sources.items()
                       if required is None or Path(relative).name in required)
        except (ValueError, FileNotFoundError):
            return False

    if matches(requested):
        return requested, bool(job.get('source_frozen', False)), 'original_manifest_matches'
    if job.get('source_frozen'):
        # A changed declared snapshot must never be quietly replaced.
        raise ValueError(f'Frozen source changed: {requested}')
    experiment = Path(job.get('experiment_root', job.get('protocol_root', requested))).resolve()
    runs_root = experiment.parent
    candidates = [Path(path).resolve() for path in job.get('inference_source_candidates', [])]
    candidates.extend(sorted(runs_root.glob('*/source_snapshot')))
    for candidate in candidates:
        if matches(candidate):
            return candidate, True, 'recovered_exact_manifest_from_another_experiment_snapshot'
    for candidate in candidates:
        if matches(candidate, _inference_files(job)):
            return candidate, True, 'recovered_matching_inference_modules_training_sources_unavailable'
    raise ValueError('Original source manifest cannot be reproduced from available snapshots')


def _verify_identity(job, checkpoint, protocol, records, source_root):
    """Check every available saved identity without inventing missing provenance."""
    family = job['family']
    for key in ('protocol_sha256', 'plan_sha256'):
        if key in protocol:
            unsigned = {k: v for k, v in protocol.items() if k != key}
            _assert_equal(_digest(unsigned), protocol[key], key)
    if 'config_sha256' in protocol:
        _assert_equal(_digest(protocol['config']), protocol['config_sha256'], 'config digest')
    record_hash = protocol.get('record_sha256', protocol.get('fair_record_sha256'))
    if record_hash and record_hash not in _record_digests(records):
        raise ValueError('Frozen records differ from the experiment protocol')
    cp_record_hash = (checkpoint.get('record_sha256') or checkpoint.get('config', {}).get('record_sha256')
                      or job.get('config', {}).get('record_sha256'))
    if cp_record_hash and cp_record_hash not in _record_digests(records):
        raise ValueError('Frozen records differ from the checkpoint')
    for key in ('config_sha256', 'record_sha256', 'source_sha256', 'protocol_sha256', 'plan_sha256'):
        if key in checkpoint and key in protocol:
            _assert_equal(checkpoint[key], protocol[key], f'checkpoint {key}')
    for key in ('seed', 'arm'):
        expected = job.get(key)
        if expected is not None and key in checkpoint:
            _assert_equal(checkpoint[key], expected, f'checkpoint {key}')
    if 'kind' in checkpoint and job.get('kind') not in ('independent', 'single_union'):
        _assert_equal(checkpoint['kind'], job['kind'], 'checkpoint kind')
    expected_epoch = job.get('expected_epoch', job.get('checkpoint_epoch'))
    if expected_epoch is not None:
        _assert_equal(checkpoint['epoch'], expected_epoch, 'checkpoint epoch')
    if checkpoint.get('config') and protocol.get('config') and family not in ('legacy_region', 'legacy_task_id', 'legacy_baseline'):
        _assert_equal(checkpoint['config'], protocol['config'], 'checkpoint configuration')
    if family in ('organ_weighted', 'region_normalization'):
        arm, seed, kind = job['arm'], str(job['seed']), job['kind']
        initial_key = arm if family == 'region_normalization' else kind
        _assert_equal(checkpoint.get('initialization_sha256'),
                      protocol['initialization_sha256'][initial_key][seed], 'initialization digest')
        if family == 'region_normalization':
            _assert_equal(checkpoint.get('non_norm_initialization_sha256'),
                          protocol['non_norm_initialization_sha256'][seed], 'non-norm initialization digest')
    if family == 'abdomen_recovery':
        initial = protocol['initial_checkpoints'][str(job['seed'])]
        _assert_equal(checkpoint.get('initial_sha256'), initial['sha256'], 'recovery initialization')
    sources = _source_manifest(job, protocol)
    # Recovery's source map describes its fine-tuner, while source_root may refer
    # to the initial fair model. The inventory provides that model protocol.
    frozen = bool(job.get('source_frozen', source_root.name == 'source_snapshot'))
    source_matches = True
    inference_matches = True
    for relative, expected in sources.items():
        required = Path(relative).name in _inference_files(job)
        try:
            actual = file_sha(_source_file(source_root, relative))
        except (ValueError, FileNotFoundError):
            source_matches = False
            inference_matches = inference_matches and not required
            if frozen and (required or not job.get('inference_only_snapshot')):
                raise
            continue
        if actual != expected:
            source_matches = False
            inference_matches = inference_matches and not required
            if frozen and (required or not job.get('inference_only_snapshot')):
                raise ValueError(f'Frozen source changed: {relative}')
    if sources and 'source_sha256' in protocol and family not in ('abdomen_recovery', 'postprocess_screen'):
        _assert_equal(_digest(sources), protocol['source_sha256'], 'source manifest')
    return dict(source_frozen=frozen, source_manifest_matches=source_matches if sources else None,
                inference_source_manifest_matches=inference_matches if sources else None,
                record_identity_verified=bool(record_hash or cp_record_hash),
                source_note='saved source snapshot' if frozen else
                'historical run has no frozen source snapshot; using identified checkout implementation')


def _content_manifest(job, protocol):
    root = Path(job.get('protocol_root') or Path(job['records_path']).parent)
    if (root / 'data_content.json').is_file():
        content = _read(root / 'data_content.json')
        if protocol.get('content_sha256'):
            _assert_equal(_digest(content), protocol['content_sha256'], 'data content manifest')
        return content
    name = 'test_content_manifest.json' if job['split'] == 'test' else 'content_manifest.json'
    if (root / name).is_file():
        content = _read(root / name)
        if name == 'content_manifest.json' and protocol.get('content_sha256'):
            _assert_equal(_digest(content), protocol['content_sha256'], 'data content manifest')
        if name == 'test_content_manifest.json' and job.get('run_dir'):
            metrics_path = Path(job['run_dir']) / 'test_metrics.json'
            if metrics_path.is_file():
                expected = _read(metrics_path).get('test_content_sha256')
                if expected:
                    _assert_equal(_digest(content), expected, 'test content manifest')
        return {row['path']: row for row in content}
    return {}


def _task_layout(checkpoint, job, data):
    config = checkpoint.get('config', job.get('config', {}))
    if job['family'] == 'legacy_baseline':
        structures = config.get('structures')
        if structures:
            return [data.TASKS.index(name) for name in structures]
    tasks = checkpoint.get('task_ids') or config.get('task_ids') or job.get('task_ids')
    if tasks is None and job['family'] == 'single_union':
        if checkpoint.get('architecture') != 'single_channel_plain_unet_v1':
            raise ValueError('Single-union checkpoint architecture is not identified')
        # This architecture's saved protocol explicitly defines the paper5
        # annotated foreground union, without storing structure output heads.
        if config.get('excluded_abdomen_classes') != [3, 5, 7, 255]:
            raise ValueError('Unsupported single-channel target scope')
        tasks = [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]
    if tasks is None and checkpoint.get('structures'):
        tasks = [data.TASKS.index(name) for name in checkpoint['structures']]
    if tasks is None and job['family'] == 'legacy_task_id':
        tasks = checkpoint.get('trained_tasks')
    if not tasks or len(set(tasks)) != len(tasks) or any(t < 0 or t >= len(data.TASKS) for t in tasks):
        raise ValueError('Missing or invalid output task mapping')
    if checkpoint.get('structures'):
        _assert_equal([data.TASKS[t] for t in tasks], checkpoint['structures'], 'structure channel names')
    return list(tasks)


def _create_model(checkpoint, job, package, tasks):
    config = checkpoint.get('config') or job.get('config')
    if not config or 'base' not in config:
        raise ValueError('Checkpoint model width is unknown')
    kind = job['kind']
    if kind in ('shared', 'independent', 'single_union'):
        outputs = 1 if kind == 'single_union' else len(tasks)
        model = _module(package, 'baseline').PlainUNet(outputs=outputs, base=config['base'])
    elif kind == 'task_id':
        if job['family'] == 'legacy_task_id':
            kwargs = {key: config[key] for key in ('tasks', 'base', 'experts', 'top_k', 'decoder') if key in config}
        else:
            kwargs = dict(tasks=len(_module(package, 'data').TASKS), base=config['base'],
                          experts=config['experts'], top_k=config['top_k'])
        model = _module(package, 'model').UNetMoE(**kwargs)
    elif kind == 'region':
        kwargs = dict(base=config['base'], top_k=config['top_k'], task_ids=tasks)
        if job['family'] == 'region_normalization':
            kwargs.update(normalization=job['arm'], norm_eps=config['norm_eps'], norm_momentum=config['norm_momentum'])
        model = _module(package, 'region').RegionUNet(**kwargs)
    else:
        raise ValueError(f'Unsupported checkpoint kind: {kind}')
    model.load_state_dict(checkpoint['model'], strict=True)
    return model


def union_for_rows(target, rows, task_ids):
    """Union of annotated tasks, retaining partial labels and excluded abdomen IDs."""
    active = target.ge(0).any((-2, -1))
    if not active.any():
        raise ValueError('Image has no annotated structures')
    positive = target.gt(0).any(0)
    negative = (target.ge(0) | ~active[:, None, None]).all(0) & ~positive
    union = torch.full_like(target[0], -1.)
    union[positive] = 1
    if not (rows[0]['dataset'] == 'AUL' and {row['task'] for row in rows} == {4}):
        union[negative] = 0
    if rows[0]['dataset'] == 'AbdomenUS':
        masks = {row['mask'] for row in rows}
        if len(masks) != 1:
            raise ValueError('Abdominal structures have inconsistent indexed masks')
        with Image.open(next(iter(masks))) as mask:
            if mask.mode not in ('L', 'P'):
                raise ValueError('Expected indexed abdomen mask')
            indexed = np.array(mask.resize((target.shape[-1], target.shape[-2]), Image.Resampling.NEAREST))
        excluded = [value for value in range(1, 9) if value + 12 not in task_ids] + [255]
        union[torch.from_numpy(np.isin(indexed, excluded))] = -1
    return union


def _forward(model, kind, image, target, tasks, domain_id, domain_conditioned):
    if kind in ('shared', 'independent', 'single_union'):
        return model(image)
    if kind == 'region':
        return model(image, domain_ids=domain_id)['logits'] if domain_conditioned else model(image)['logits']
    # Query one image-task at a time: identical eval outputs with bounded memory,
    # including annotated empty structures. No foreground-dependent query gating.
    output = image.new_full(target.shape, float('-inf'))
    for channel in target[0].ge(0).any((-2, -1)).nonzero().flatten().tolist():
        task = torch.tensor([tasks[channel]], dtype=torch.long, device=image.device)
        output[:, channel:channel + 1] = model(image, task)['logits']
    return output


def postprocess_predictions(probability, specs):
    """Match the original screen: 8-connected components before ignoring pixels."""
    import cv2
    for threshold in dict.fromkeys(spec['threshold'] for spec in specs):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray((probability >= threshold).astype(np.uint8)), connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        for spec in (item for item in specs if item['threshold'] == threshold):
            selected = np.arange(1, count)[areas[1:] >= spec['min_area']]
            if spec['largest_only'] and selected.size:
                selected = selected[np.argmax(areas[selected])][None]
            yield spec['id'], np.isin(labels, selected)


def load_evaluation(job, device='cpu'):
    """Load a trusted local artifact and return metadata plus lazy sample iterator.

    ``job`` comes from reevaluation inventory. A session uses a single model;
    consume/close ``samples`` before moving on to the next checkpoint.
    """
    if job.get('status', 'ready') != 'ready':
        raise ValueError(f'Job is not ready: {job.get("status")}')
    if job['split'] not in ('val', 'test'):
        raise ValueError('Only saved validation/test splits may be reevaluated')
    checkpoint_path = Path(job['checkpoint']).resolve()
    checkpoint_sha = file_sha(checkpoint_path)
    if job.get('checkpoint_sha256'):
        _assert_equal(checkpoint_sha, job['checkpoint_sha256'], 'checkpoint bytes')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    record_document = _read(job['records_path'])
    records = record_document.get('records', []) if isinstance(record_document, dict) else record_document
    protocol = _read(job['protocol_path']) if job.get('protocol_path') else {}
    source_root, source_frozen, source_resolution = _resolve_source(job, protocol)
    verified = _verify_identity(dict(job, source_frozen=source_frozen,
                                    inference_only_snapshot=source_resolution.startswith('recovered_matching_inference')),
                                checkpoint, protocol, records, source_root)
    verified.update(source_root=str(source_root), source_resolution=source_resolution,
                    original_source_frozen=bool(job.get('source_frozen', False)))
    package = _load_package(source_root)
    data = _module(package, 'data')
    tasks = _task_layout(checkpoint, job, data)
    config = checkpoint.get('config') or job.get('config') or {}
    size = int(checkpoint.get('size', config.get('size', job.get('size', 0))))
    if size < 16 or (job.get('size') and size != job['size']):
        raise ValueError('Checkpoint image resolution is missing or disagrees with inventory')
    if job.get('family') == 'postprocess_screen' and job['split'] != 'val':
        raise ValueError('Postprocessing screen is validation-only')
    selected = [row for row in records if row['split'] == job['split'] and row['task'] in tasks]
    if job.get('family') == 'postprocess_screen':
        selected = [row for row in selected if row['dataset'] == 'AbdomenUS']
    groups = defaultdict(list)
    for row in selected:
        groups[row['image']].append(row)
    if not groups:
        raise ValueError('No images in requested saved split/task scope')
    for rows in groups.values():
        if len({row['dataset'] for row in rows}) != 1 or len({row['task'] for row in rows}) != len(rows):
            raise ValueError('Ambiguous image domain or duplicate task annotation')
    content = _content_manifest(job, protocol)
    model = _create_model(checkpoint, job, package, tasks).to(device).eval()
    metadata = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                    checkpoint_epoch=checkpoint['epoch'], image_size=size, split=job['split'],
                    images=len(groups), task_ids=tasks, structures=[data.TASKS[t] for t in tasks],
                    threshold=.5, distance_units='pixel at checkpoint input resolution',
                    records_file_sha256=file_sha(job['records_path']),
                    content_hashes_available=bool(content), **verified)
    metadata['views'] = ['union'] if job['kind'] == 'single_union' else ['structure', 'union']
    if job['family'] == 'postprocess_screen':
        metadata.update(views=['structure'], candidate_grid=job['candidate_grid'])
    del checkpoint

    def samples():
        # Keep inference context scoped to computation, never across a yield.
        domains = _module(package, 'region').DOMAINS
        for image_id, rows in groups.items():
            for path in {row[key] for row in rows for key in ('image', 'mask')}:
                if content:
                    if path not in content or file_sha(path) != content[path]['sha256']:
                        raise ValueError(f'Data content changed or absent from manifest: {path}')
            target = torch.full((len(tasks), size, size), -1.)
            for row, (image, label, _) in zip(rows, data.SegmentationDataset(rows, size)):
                target[tasks.index(row['task'])] = label[0]
            domain = rows[0]['dataset']
            source = rows[0].get('group', '').split('/')[-1] if domain == 'AbdomenUS' else ''
            domain_id = torch.tensor([domains.index(domain)], device=device, dtype=torch.long)
            with torch.inference_mode():
                logits = _forward(model, job['kind'], image[None].to(device), target[None].to(device),
                                  tasks, domain_id, job['family'] == 'region_normalization')
                probability = logits.sigmoid()[0].cpu()
            if not torch.isfinite(probability).all():
                raise ValueError(f'Nonfinite prediction: {image_id}')
            common = dict(image_id=image_id, domain=domain, source=source)
            if job['kind'] != 'single_union':
                if probability.shape != target.shape:
                    raise ValueError('Model output shape differs from annotated channel mapping')
                active = target.ge(0).any((-2, -1))
                for channel in active.nonzero().flatten().tolist():
                    truth = target[channel].numpy().copy()
                    if job['family'] == 'postprocess_screen':
                        for variant, prediction in postprocess_predictions(probability[channel].numpy(), job['candidate_grid']):
                            yield EvaluationSample(**common, structure=data.TASKS[tasks[channel]], task_id=tasks[channel],
                                                   target=truth, prediction=prediction, view='structure', variant=variant)
                    else:
                        yield EvaluationSample(**common, structure=data.TASKS[tasks[channel]], task_id=tasks[channel],
                                               target=truth, prediction=(probability[channel] >= .5).numpy(), view='structure')
                union_prediction = (probability[active] >= .5).any(0)
            else:
                if probability.shape != (1, size, size):
                    raise ValueError('Single-channel U-Net must produce exactly one output')
                union_prediction = probability[0] >= .5
            if job['family'] != 'postprocess_screen':
                union = union_for_rows(target, rows, tasks)
                yield EvaluationSample(**common, structure=None, task_id=None, target=union.numpy(),
                                       prediction=union_prediction.numpy(), view='union')

    return EvaluationSession(metadata, samples())
