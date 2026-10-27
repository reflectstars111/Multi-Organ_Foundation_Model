"""Exact-path repairs preserve legacy split/model/metric provenance."""
from copy import deepcopy
from dataclasses import dataclass
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts import repair_legacy_region_paths as repair


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    output = tmp_path / 'evaluation'
    folder = output / 'repair_snapshot'
    folder.mkdir(parents=True)
    script = folder / 'repair_legacy_region_paths.py'
    script.write_bytes(Path(repair.__file__).read_bytes())
    monkeypatch.setattr(repair, '__file__', str(script))
    data_root = tmp_path / 'data'
    data_root.mkdir()
    image, mask = data_root / 'image.png', data_root / 'mask.png'
    Image.new('L', (3, 2), color=128).save(image)
    Image.new('L', (3, 2), color=255).save(mask)
    old_image, old_mask = r'dataset\pointed_data\image.png', r'dataset\pointed_data\mask.png'
    records = [dict(image=old_image, mask=old_mask, split='val', task=0),
               dict(image=old_image, mask=old_mask, split='test', task=0),
               dict(image='train-only.png', mask='train-only-label.png', split='train', task=0)]
    records_path = tmp_path / 'records.json'
    write_json(records_path, records)
    jobs = {}
    for selection in ('best', 'last'):
        for split in ('val', 'test'):
            job_id = selection + '-' + split
            job = dict(job_id=job_id, family='legacy_region', experiment='region_top1_v1',
                       kind='region', selection=selection, split=split,
                       config=dict(data_root=r'dataset\pointed_data'),
                       records_path=str(records_path), evaluation_protocol_sha256='protocol')
            job['job_manifest_sha256'] = repair.digest(job)
            write_json(output / 'jobs' / job_id / 'job.json', job)
            jobs[job_id] = job
    write_json(output / 'protocol.json', dict(protocol_sha256='protocol', job_ids=list(jobs)))
    mapping = {old: dict(path=str(path), sha256=repair.file_sha(path), bytes=path.stat().st_size)
               for old, path in ((old_image, image), (old_mask, mask))}
    manifest = dict(schema=repair.SCHEMA, repair_script_sha256=repair.file_sha(script),
                    evaluation_protocol_sha256='protocol', records_path=str(records_path),
                    windows_data_prefix=r'dataset\pointed_data',
                    records_file_sha256=repair.file_sha(records_path), data_root=str(data_root),
                    allowed_job_ids=list(jobs), path_mapping=mapping,
                    job_fingerprints={key: value['job_manifest_sha256'] for key, value in jobs.items()})
    manifest_path = folder / 'manifest.json'

    def save():
        manifest['repair_manifest_sha256'] = repair.digest(
            {k: v for k, v in manifest.items() if k != 'repair_manifest_sha256'})
        write_json(manifest_path, manifest)

    save()
    return SimpleNamespace(output=output, manifest_path=manifest_path, manifest=manifest,
                           save=save, script=script, image=image, mask=mask, records=records,
                           records_path=records_path, data_root=data_root,
                           old_image=old_image, old_mask=old_mask, jobs=jobs)


def test_complete_validation_is_read_only_and_uses_only_saved_val_test(frozen):
    before = {path: path.read_bytes() for path in frozen.output.rglob('*') if path.is_file()}
    assert repair.validate_manifest(frozen.output, frozen.manifest_path) == frozen.manifest
    assert {path: path.read_bytes() for path in before} == before
    assert 'train-only.png' not in frozen.manifest['path_mapping']


def test_exact_mapping_decodes_identical_pixels_without_changing_native_opens(frozen):
    original = Image.open
    expected = Image.open(frozen.image).tobytes()
    with repair.resolved_image_opens(frozen.manifest):
        assert Image.open(frozen.old_image).tobytes() == expected
        assert Image.open(Path(frozen.old_image)).tobytes() == expected
        assert Image.open(frozen.old_image.encode()).tobytes() == expected
        assert Image.open(frozen.image).tobytes() == expected
        assert Image.open(io.BytesIO(frozen.image.read_bytes())).tobytes() == expected
    assert Image.open is original


@pytest.mark.parametrize('unknown', [r'dataset\pointed_data\missing.png',
                                   r'C:\unmapped\image.png', r'other\image.png'])
def test_unmapped_windows_path_rejected_and_original_open_restored(frozen, unknown):
    original = Image.open
    with pytest.raises(ValueError, match='outside frozen mapping'):
        with repair.resolved_image_opens(frozen.manifest):
            Image.open(unknown)
    assert Image.open is original


def test_exception_in_caller_also_restores_original_open(frozen):
    original = Image.open
    with pytest.raises(RuntimeError, match='caller failed'):
        with repair.resolved_image_opens(frozen.manifest):
            raise RuntimeError('caller failed')
    assert Image.open is original


def test_map_does_not_rewrite_suffix_matches(frozen, tmp_path):
    path = tmp_path / 'native' / 'image.png'
    path.parent.mkdir()
    Image.new('L', (3, 2), color=7).save(path)
    with repair.resolved_image_opens(frozen.manifest):
        assert Image.open(path).getpixel((0, 0)) == 7


def test_changed_manifest_rejected(frozen):
    value = deepcopy(frozen.manifest)
    value['data_root'] = str(frozen.data_root.parent)
    write_json(frozen.manifest_path, value)
    with pytest.raises(ValueError, match='Repair manifest changed'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_changed_repair_script_rejected(frozen):
    frozen.script.write_text('changed')
    with pytest.raises(ValueError, match='Repair script changed'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_changed_records_rejected_even_if_semantically_equivalent(frozen):
    frozen.records_path.write_text(json.dumps(frozen.records, indent=4))
    with pytest.raises(ValueError, match='Original records bytes changed'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


@pytest.mark.parametrize('operation', ['remove', 'add'])
def test_mapping_must_equal_the_entire_original_saved_scope(frozen, operation):
    if operation == 'remove':
        frozen.manifest['path_mapping'].pop(frozen.old_mask)
    else:
        frozen.manifest['path_mapping']['train-only.png'] = frozen.manifest['path_mapping'][frozen.old_image]
    frozen.save()
    with pytest.raises(ValueError, match='exactly the original'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_missing_file_rejected(frozen):
    frozen.image.unlink()
    with pytest.raises(ValueError, match='Mapped file is missing'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_changed_bytes_rejected_at_validation_and_image_open(frozen):
    Image.new('L', (3, 2), color=100).save(frozen.image)
    with pytest.raises(ValueError, match='bytes changed'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)
    original = Image.open
    with pytest.raises(ValueError, match='bytes changed'):
        with repair.resolved_image_opens(frozen.manifest):
            Image.open(frozen.old_image)
    assert Image.open is original


@pytest.mark.parametrize('escape', ['outside', 'traversal', 'symlink'])
def test_destination_escape_rejected(frozen, tmp_path, escape):
    outside = tmp_path / 'outside.png'
    outside.write_bytes(frozen.image.read_bytes())
    if escape == 'outside':
        target = outside
    elif escape == 'traversal':
        target = frozen.data_root / '..' / 'outside.png'
    else:
        target = frozen.data_root / 'redirect.png'
        target.symlink_to(outside)
    frozen.manifest['path_mapping'][frozen.old_image]['path'] = str(target)
    frozen.save()
    with pytest.raises(ValueError, match='escapes data_root|original suffix'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_symlink_changed_after_manifest_validation_is_rejected_at_open(frozen, tmp_path):
    repair.validate_manifest(frozen.output, frozen.manifest_path)
    outside = tmp_path / 'same_bytes_outside.png'
    outside.write_bytes(frozen.image.read_bytes())
    frozen.image.unlink()
    frozen.image.symlink_to(outside)
    with repair.resolved_image_opens(frozen.manifest):
        with pytest.raises(ValueError, match='escapes data_root'):
            Image.open(frozen.old_image)


def test_two_original_paths_cannot_ambiguously_map_to_one_file(frozen):
    frozen.manifest['path_mapping'][frozen.old_mask] = deepcopy(frozen.manifest['path_mapping'][frozen.old_image])
    frozen.save()
    with pytest.raises(ValueError, match='Ambiguous mapped destination'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_original_record_path_traversal_rejected(frozen):
    invalid = r'dataset\..\image.png'
    for row in frozen.records:
        if row['image'] == frozen.old_image:
            row['image'] = invalid
    write_json(frozen.records_path, frozen.records)
    frozen.manifest['records_file_sha256'] = repair.file_sha(frozen.records_path)
    frozen.manifest['path_mapping'][invalid] = frozen.manifest['path_mapping'].pop(frozen.old_image)
    frozen.save()
    with pytest.raises(ValueError, match='Invalid original path'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_other_job_rejected(frozen):
    job = dict(next(iter(frozen.jobs.values())), job_id='other')
    with pytest.raises(ValueError, match='whitelist'):
        repair.validate_job(job, frozen.manifest)


@pytest.mark.parametrize('field,value', [('experiment', 'new_run'), ('family', 'fair'),
                                      ('kind', 'task_id'), ('split', 'train'), ('selection', 'unknown')])
def test_known_id_cannot_expand_supported_legacy_scope(frozen, field, value):
    job = dict(next(iter(frozen.jobs.values())))
    job[field] = value
    with pytest.raises(ValueError, match='original legacy Region'):
        repair.validate_job(job, frozen.manifest)


def test_job_tamper_rejected_even_with_recomputed_job_fingerprint(frozen):
    job = dict(next(iter(frozen.jobs.values())))
    job['checkpoint'] = 'different-model.pt'
    job['job_manifest_sha256'] = repair.digest({k: v for k, v in job.items() if k != 'job_manifest_sha256'})
    with pytest.raises(ValueError, match='Original allowed job fingerprint changed'):
        repair.validate_job(job, frozen.manifest)


def test_metadata_provenance_and_restore_on_exception(frozen, monkeypatch):
    job = next(iter(frozen.jobs.values()))
    original_metadata = dict(images=2)
    session = SimpleNamespace(metadata=original_metadata)
    load = lambda *args, **kwargs: session
    inference = SimpleNamespace(__file__=str(frozen.output / 'source_snapshot/unet_moe/reevaluation_inference.py'),
                                load_evaluation=load)
    monkeypatch.setattr(repair.importlib, 'import_module', lambda name: inference)
    with pytest.raises(RuntimeError):
        with repair.repaired_session_metadata(frozen.output, frozen.manifest):
            actual = inference.load_evaluation(job)
            assert actual is session
            assert actual.metadata['images'] == 2
            assert actual.metadata['path_resolution']['original_records_preserved'] is True
            assert 'not hashed at training time' in actual.metadata['path_resolution']['limitation']
            raise RuntimeError()
    assert inference.load_evaluation is load


def test_unfrozen_inference_import_is_rejected(frozen, monkeypatch):
    inference = SimpleNamespace(__file__='/checkout/unet_moe/reevaluation_inference.py')
    monkeypatch.setattr(repair.importlib, 'import_module', lambda name: inference)
    with pytest.raises(ValueError, match='original frozen evaluator'):
        with repair.repaired_session_metadata(frozen.output, frozen.manifest):
            pytest.fail('Wrong module accepted')


def add_hc18_amendment(frozen):
    folder = '胎儿头围测量数据集 (Fetal HC - HC18 Challenge)'
    paths = []
    for index, split in enumerate(('val', 'test')):
        name = f'{index}_HC.png'
        old_image = '\\'.join((frozen.manifest['windows_data_prefix'], folder,
                                'processed_png', 'images', 'train', name))
        old_mask = '\\'.join((frozen.manifest['windows_data_prefix'], folder,
                               'processed_png', 'masks', 'train', name))
        image = frozen.data_root / folder / 'processed_png/images/train' / name
        filled = frozen.data_root / folder / 'processed_png/masks_filled/train' / name
        for path, color in ((image, 80), (filled, 255)):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('L', (3, 2), color=color).save(path)
        frozen.records.append(dict(image=old_image, mask=old_mask, split=split,
                                   task=8, dataset='Fetal_HC'))
        for old, path in ((old_image, image), (old_mask, filled)):
            frozen.manifest['path_mapping'][old] = dict(path=str(path), bytes=path.stat().st_size,
                                                       sha256=repair.file_sha(path))
        paths.append(old_mask)
    write_json(frozen.records_path, frozen.records)
    frozen.manifest['records_file_sha256'] = repair.file_sha(frozen.records_path)
    amendment = dict(schema=repair.AMENDMENT_SCHEMA, authorization='需要的就是使用填充后的区域标签',
                     original_training_targets='HC18 contour', evaluation_targets='HC18 filled region',
                     amended_paths=sorted(paths), count=2, split_counts=dict(val=1, test=1))
    amendment['amendment_sha256'] = repair.digest(amendment)
    frozen.manifest['label_policy_amendment'] = amendment
    frozen.save()
    return amendment


def rehash_amendment(frozen):
    amendment = frozen.manifest['label_policy_amendment']
    amendment['amendment_sha256'] = repair.digest(
        {key: value for key, value in amendment.items() if key != 'amendment_sha256'})
    frozen.save()


def test_authorized_filled_evaluation_does_not_require_missing_contour_files(frozen):
    amendment = add_hc18_amendment(frozen)
    repair.validate_manifest(frozen.output, frozen.manifest_path)
    with repair.resolved_image_opens(frozen.manifest):
        for old in amendment['amended_paths']:
            with Image.open(old) as image:
                assert image.getextrema() == (255, 255)
        # Other domains still have exactly their original pixels.
        assert Image.open(frozen.old_image).getextrema() == (128, 128)
    assert not list(frozen.data_root.rglob('masks/train/*'))


def test_filled_substitution_without_explicit_amendment_rejected(frozen):
    add_hc18_amendment(frozen)
    del frozen.manifest['label_policy_amendment']
    frozen.save()
    with pytest.raises(ValueError, match='original suffix without an allowed amendment'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


@pytest.mark.parametrize('change', ['omit_mask', 'add_other_task', 'wrong_count', 'wrong_split_count'])
def test_amendment_cannot_change_saved_mask_scope(frozen, change):
    amendment = add_hc18_amendment(frozen)
    if change == 'omit_mask':
        amendment['amended_paths'].pop()
        amendment['count'] -= 1
    elif change == 'add_other_task':
        amendment['amended_paths'].append(frozen.old_mask)
        amendment['amended_paths'].sort()
        amendment['count'] += 1
    elif change == 'wrong_count':
        amendment['count'] = 99
    else:
        amendment['split_counts']['test'] = 0
    rehash_amendment(frozen)
    with pytest.raises(ValueError, match='exactly all saved validation/test HC18 masks'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_amendment_hash_is_checked_independently(frozen):
    amendment = add_hc18_amendment(frozen)
    amendment['count'] = 3
    frozen.save()
    with pytest.raises(ValueError, match='amendment changed'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_amendment_requires_the_recorded_user_authorization(frozen):
    amendment = add_hc18_amendment(frozen)
    amendment['authorization'] = 'inferred preference'
    rehash_amendment(frozen)
    with pytest.raises(ValueError, match='unauthorized HC18'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_hc18_masks_must_keep_the_same_filenames(frozen):
    amendment = add_hc18_amendment(frozen)
    first, second = amendment['amended_paths']
    mapping = frozen.manifest['path_mapping']
    mapping[first], mapping[second] = mapping[second], mapping[first]
    frozen.save()
    with pytest.raises(ValueError, match='original suffix without an allowed amendment'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_normal_mapping_cannot_swap_unrelated_files_inside_data_root(frozen):
    mapping = frozen.manifest['path_mapping']
    mapping[frozen.old_mask], mapping[frozen.old_image] = mapping[frozen.old_image], mapping[frozen.old_mask]
    frozen.save()
    with pytest.raises(ValueError, match='original suffix without an allowed amendment'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


def test_amended_target_bytes_still_must_match_frozen_hash(frozen):
    amendment = add_hc18_amendment(frozen)
    mask = Path(frozen.manifest['path_mapping'][amendment['amended_paths'][0]]['path'])
    Image.new('L', (3, 2), color=0).save(mask)
    with pytest.raises(ValueError, match='bytes changed'):
        repair.validate_manifest(frozen.output, frozen.manifest_path)


@dataclass
class Sample:
    domain: str
    view: str
    prediction: object
    target: object
    variant: str | None = None


def test_amendment_labels_all_domains_and_views_without_splitting_macro_scope(frozen, monkeypatch):
    amendment = add_hc18_amendment(frozen)
    samples = [Sample(domain, view, prediction=object(), target=object())
               for domain in ('Fetal_HC', 'BUSI') for view in ('structure', 'union')]
    session = SimpleNamespace(metadata=dict(images=2), samples=iter(samples))
    load = lambda *args, **kwargs: session
    inference = SimpleNamespace(__file__=str(frozen.output / 'source_snapshot/unet_moe/reevaluation_inference.py'),
                                load_evaluation=load)
    monkeypatch.setattr(repair.importlib, 'import_module', lambda name: inference)
    with repair.repaired_session_metadata(frozen.output, frozen.manifest):
        actual = inference.load_evaluation(next(iter(frozen.jobs.values())))
        assert actual.metadata['label_policy_amendment'] == amendment
        assert actual.metadata['path_resolution']['evaluation_targets_modified'] is True
        assert actual.metadata['evaluation_variant'] == repair.AMENDMENT_VARIANT
        output = list(actual.samples)
    assert inference.load_evaluation is load
    assert len(output) == len(samples)
    for before, after in zip(samples, output):
        assert after.variant == repair.AMENDMENT_VARIANT
        assert before.variant is None
        assert (after.domain, after.view) == (before.domain, before.view)
        assert after.target is before.target
        assert after.prediction is before.prediction


def test_amendment_refuses_an_unexpected_preexisting_evaluation_variant(frozen, monkeypatch):
    add_hc18_amendment(frozen)
    session = SimpleNamespace(metadata={}, samples=iter([Sample('Fetal_HC', 'union', 1, 2, 'other')]))
    inference = SimpleNamespace(__file__=str(frozen.output / 'source_snapshot/unet_moe/reevaluation_inference.py'),
                                load_evaluation=lambda *a, **kw: session)
    monkeypatch.setattr(repair.importlib, 'import_module', lambda name: inference)
    with repair.repaired_session_metadata(frozen.output, frozen.manifest):
        actual = inference.load_evaluation(next(iter(frozen.jobs.values())))
        with pytest.raises(ValueError, match='another evaluation variant'):
            next(actual.samples)
