import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from unet_moe import reevaluation_inference as inference
from unet_moe.baseline import PlainUNet
from unet_moe.data import TASKS
from unet_moe.model import UNetMoE
from unet_moe.region import RegionUNet


ROOT = Path(__file__).resolve().parents[1]


def fixture_job(tmp_path, kind='shared', family='fair', tasks=(0, 1), size=32):
    image = tmp_path / 'image.png'
    mask = tmp_path / 'mask.png'
    Image.fromarray(np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)).save(image)
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[8:16, 9:18] = 255
    Image.fromarray(pixels).save(mask)
    records = [dict(image=str(image), mask=str(mask), task=0, class_id=0,
                    split='val', dataset='BUSI', group='BUSI/test')]
    records_path = tmp_path / 'records.json'
    records_path.write_text(json.dumps(records))
    config = dict(size=size, base=4, experts=9, top_k=1, task_ids=list(tasks))
    if kind in ('shared', 'single_union', 'independent'):
        model = PlainUNet(outputs=1 if kind == 'single_union' else len(tasks), base=4)
    elif kind == 'task_id':
        model = UNetMoE(tasks=len(TASKS), base=4, experts=9, top_k=1)
    elif kind == 'region':
        model = RegionUNet(base=4, top_k=1, task_ids=list(tasks))
    checkpoint = dict(config=config, model=model.state_dict(), epoch=4,
                      task_ids=list(tasks), seed=42)
    checkpoint_path = tmp_path / 'best.pt'
    torch.save(checkpoint, checkpoint_path)
    protocol = dict(config=config, config_sha256=inference._digest(config),
                    record_sha256=inference._digest(records))
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    return dict(family=family, kind=kind, checkpoint=str(checkpoint_path),
                protocol_path=str(protocol_path), protocol_root=str(tmp_path),
                records_path=str(records_path), source_root=str(ROOT), source_frozen=False,
                split='val', size=size, seed=42, status='ready', config=config,
                task_ids=list(tasks))


@pytest.mark.parametrize('kind', ['shared', 'region', 'task_id', 'single_union'])
def test_real_model_outputs_match_direct_prediction(tmp_path, kind):
    job = fixture_job(tmp_path, kind)
    session = inference.load_evaluation(job)
    rows = list(session.samples)
    assert session.metadata['images'] == 1
    assert session.metadata['record_identity_verified']
    assert session.metadata['image_size'] == 32
    assert all(row.prediction.dtype == bool and row.target.shape == (32, 32) for row in rows)
    assert [row.view for row in rows] == (['union'] if kind == 'single_union' else ['structure', 'union'])
    if kind != 'single_union':
        assert rows[0].task_id == 0
        assert np.array_equal(rows[0].prediction, rows[1].prediction)


def test_task_query_includes_empty_annotation_and_excludes_absent_channel():
    calls = []
    class Query:
        def __call__(self, image, task):
            calls.append(task.item())
            return {'logits': torch.full_like(image, float(task.item()))}
    target = torch.full((1, 3, 4, 4), -1.)
    target[0, 0] = 0  # Empty, but annotated: still must query this task.
    target[0, 2, 1, 1] = 1
    output = inference._forward(Query(), 'task_id', torch.zeros(1, 1, 4, 4),
                                target, [13, 14, 16], None, False)
    assert calls == [13, 16]
    assert torch.isneginf(output[0, 1]).all()
    assert (output[0, 0] == 13).all()
    assert (output[0, 2] == 16).all()


def test_domain_id_is_passed_to_normalization_model():
    class Model:
        def __call__(self, image, domain_ids):
            assert domain_ids.tolist() == [8]
            return {'logits': image}
    image = torch.ones(1, 1, 4, 4)
    assert torch.equal(inference._forward(Model(), 'region', image, image, [13],
                                         torch.tensor([8]), True), image)


def test_single_channel_never_emits_organ_metric(tmp_path):
    job = fixture_job(tmp_path, 'single_union', 'single_union')
    rows = list(inference.load_evaluation(job))
    assert len(rows) == 1 and rows[0].structure is None and rows[0].task_id is None


def test_independent_head_maps_structure_names_not_global_channel_indices(tmp_path):
    job = fixture_job(tmp_path, 'independent', 'legacy_baseline', tasks=(0,))
    checkpoint = torch.load(job['checkpoint'], weights_only=True)
    checkpoint['config'].update(structures=['breast_lesion'], channels=[9], mode='independent')
    checkpoint.pop('task_ids')
    torch.save(checkpoint, job['checkpoint'])
    rows = list(inference.load_evaluation(job))
    assert rows[0].structure == 'breast_lesion' and rows[0].task_id == 0


def test_checkpoint_identity_and_strict_weights_rejected(tmp_path):
    job = fixture_job(tmp_path)
    wrong_seed = dict(job, seed=43)
    with pytest.raises(ValueError, match='seed'):
        inference.load_evaluation(wrong_seed)
    checkpoint = torch.load(job['checkpoint'], weights_only=True)
    checkpoint['model'].pop('head.bias')
    torch.save(checkpoint, job['checkpoint'])
    with pytest.raises(RuntimeError, match='Missing key'):
        inference.load_evaluation(job)


def test_changed_records_are_rejected(tmp_path):
    job = fixture_job(tmp_path)
    records = json.loads(Path(job['records_path']).read_text())
    records[0]['split'] = 'test'
    Path(job['records_path']).write_text(json.dumps(records))
    with pytest.raises(ValueError, match='records differ'):
        inference.load_evaluation(job)


def test_historical_wrapped_records_preserve_identity(tmp_path):
    job = fixture_job(tmp_path)
    path = Path(job['records_path'])
    records = json.loads(path.read_text())
    path.write_text(json.dumps(dict(records=records, skipped={})))
    assert len(list(inference.load_evaluation(job))) == 2


def test_manifest_can_be_recovered_only_from_matching_inference_files(tmp_path):
    job = fixture_job(tmp_path)
    snapshot = tmp_path / 'historical_snapshot'
    package = snapshot / 'unet_moe'
    package.mkdir(parents=True)
    for name in ('__init__.py', 'baseline.py', 'model.py', 'region.py', 'normalization.py', 'data.py'):
        shutil.copyfile(ROOT / 'unet_moe' / name, package / name)
    protocol = json.loads(Path(job['protocol_path']).read_text())
    protocol['source_files'] = {path.name: inference.file_sha(path) for path in package.iterdir()}
    protocol['source_files']['fair_suite.py'] = 'unknown_training_source_digest'
    protocol['source_sha256'] = inference._digest(protocol['source_files'])
    Path(job['protocol_path']).write_text(json.dumps(protocol))
    job['inference_source_candidates'] = [str(snapshot)]
    session = inference.load_evaluation(job)
    assert not session.metadata['source_manifest_matches']
    assert session.metadata['inference_source_manifest_matches']
    assert session.metadata['source_root'] == str(snapshot)
    session.samples.close()
    with (package / 'data.py').open('a') as stream:
        stream.write('\n# invalid source\n')
    with pytest.raises(ValueError, match='cannot be reproduced'):
        inference.load_evaluation(job)


def test_saved_source_is_isolated_and_tampering_rejected(tmp_path):
    job = fixture_job(tmp_path)
    snapshot = tmp_path / 'source_snapshot'
    package = snapshot / 'unet_moe'
    package.mkdir(parents=True)
    for name in ('__init__.py', 'baseline.py', 'model.py', 'region.py', 'normalization.py', 'data.py'):
        shutil.copyfile(ROOT / 'unet_moe' / name, package / name)
    protocol = json.loads(Path(job['protocol_path']).read_text())
    protocol['source_files'] = {f'unet_moe/{path.name}': inference.file_sha(path) for path in package.iterdir()}
    protocol['source_sha256'] = inference._digest(protocol['source_files'])
    Path(job['protocol_path']).write_text(json.dumps(protocol))
    job.update(source_root=str(snapshot), source_frozen=True)
    session = inference.load_evaluation(job)
    assert session.metadata['source_frozen']
    assert len(list(session.samples)) == 2
    with (package / 'data.py').open('a') as stream:
        stream.write('\n# changed\n')
    with pytest.raises(ValueError, match='Frozen source changed'):
        inference.load_evaluation(job)


def test_partial_union_and_excluded_abdominal_organs(tmp_path):
    target = torch.zeros(2, 4, 4)
    target[0, 0, 0] = 1
    aul = [dict(dataset='AUL', task=4)]
    union = inference.union_for_rows(target[:1], aul, [4])
    assert union[0, 0] == 1 and (union == -1).sum() == 15
    indexed = np.zeros((4, 4), dtype=np.uint8)
    indexed[1, 1] = 3
    indexed[2, 2] = 255
    mask = tmp_path / 'indexed.png'
    Image.fromarray(indexed).save(mask)
    rows = [dict(dataset='AbdomenUS', task=13, mask=str(mask))]
    union = inference.union_for_rows(target, rows, [13, 14])
    assert union[1, 1] == -1 and union[2, 2] == -1 and union[3, 3] == 0
    union = inference.union_for_rows(target, rows, list(range(13, 21)))
    assert union[1, 1] == 0 and union[2, 2] == -1


def test_postprocessing_uses_whole_prediction_before_valid_mask():
    probability = np.zeros((6, 8), dtype=np.float32)
    probability[0:2, 0:2] = .8
    probability[4, 5:8] = .8
    specs = [dict(id='all', threshold=.5, min_area=0, largest_only=False),
             dict(id='large', threshold=.5, min_area=0, largest_only=True),
             dict(id='filtered', threshold=.5, min_area=5, largest_only=False)]
    output = dict(inference.postprocess_predictions(probability, specs))
    assert output['all'].sum() == 7
    assert output['large'].sum() == 4
    assert output['filtered'].sum() == 0


def test_nonready_job_and_resolution_mismatch_rejected(tmp_path):
    job = fixture_job(tmp_path)
    with pytest.raises(ValueError, match='not ready'):
        inference.load_evaluation(dict(job, status='deferred'))
    with pytest.raises(ValueError, match='resolution'):
        inference.load_evaluation(dict(job, size=64))


def test_input_content_digest_rechecked_before_inference(tmp_path):
    job = fixture_job(tmp_path)
    records = json.loads(Path(job['records_path']).read_text())
    content = {records[0][key]: {'sha256': inference.file_sha(records[0][key])} for key in ('image', 'mask')}
    (tmp_path / 'data_content.json').write_text(json.dumps(content))
    session = inference.load_evaluation(job)
    Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(records[0]['image'])
    with pytest.raises(ValueError, match='Data content changed'):
        next(session.samples)
