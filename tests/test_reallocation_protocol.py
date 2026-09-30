from collections import defaultdict
from pathlib import Path
from argparse import Namespace

import pytest
import numpy as np
import torch
from PIL import Image

from unet_moe.baseline_suite import run_model
from unet_moe.data import (ABDOMEN_CV5_VALIDATION_CTS,
                           ABDOMEN_RELEASE_TRAIN_CTS, build_records)
from unet_moe.region import RegionDataset, PAPER_ALIGNED_TASK_IDS
from unet_moe.schedule import epoch_lr


DATA_ROOT = Path(__file__).resolve().parents[1] / '数据集 (Datasets)' / 'pointed_data'


@pytest.mark.skipif(not DATA_ROOT.exists(), reason='Local datasets not installed')
def test_more_train_keeps_every_previous_test_image_and_official_camus():
    old, _ = build_records(DATA_ROOT, 42, 'ct7_2')
    new, _ = build_records(DATA_ROOT, 42, 'ct_cv5', 0, 'trainmore_v6')
    def assignments(rows):
        return {(r['dataset'], r['image']): r['split'] for r in rows}
    before, after = assignments(old), assignments(new)
    assert before.keys() == after.keys()
    assert {key for key, split in before.items() if split == 'test'} == {
        key for key, split in after.items() if split == 'test'}
    assert all(before[key] == after[key] for key in before if key[0] == 'CAMUS')
    for dataset in {key[0] for key in before} - {'CAMUS', 'AbdomenUS'}:
        assert sum(key[0] == dataset and split == 'train' for key, split in after.items()) >= sum(
            key[0] == dataset and split == 'train' for key, split in before.items())


@pytest.mark.skipif(not DATA_ROOT.exists(), reason='Local datasets not installed')
def test_abdomen_cv_covers_all_nine_sources_without_touching_release_test():
    validated = defaultdict(int)
    test_groups = None
    for fold in range(5):
        rows, _ = build_records(DATA_ROOT, 42, 'ct_cv5', fold, 'trainmore_v6')
        abdominal = [r for r in rows if r['dataset'] == 'AbdomenUS']
        split_groups = {split: {r['group'].split('/')[-1] for r in abdominal if r['split'] == split}
                        for split in ('train', 'val', 'test')}
        assert split_groups['val'] == ABDOMEN_CV5_VALIDATION_CTS[fold]
        assert split_groups['train'] | split_groups['val'] == ABDOMEN_RELEASE_TRAIN_CTS
        assert all(not (split_groups[a] & split_groups[b]) for a, b in
                   (('train', 'val'), ('train', 'test'), ('val', 'test')))
        assert test_groups is None or test_groups == split_groups['test']
        test_groups = split_groups['test']
        for source in split_groups['val']:
            validated[source] += 1
    assert set(validated) == ABDOMEN_RELEASE_TRAIN_CTS
    final, _ = build_records(DATA_ROOT, 42, 'ct_final', split_policy='trainmore_v6')
    assert {r['group'].split('/')[-1] for r in final
            if r['dataset'] == 'AbdomenUS' and r['split'] == 'train'} == ABDOMEN_RELEASE_TRAIN_CTS
    assert not any(r['dataset'] == 'AbdomenUS' and r['split'] == 'val' for r in final)


def test_new_protocol_rejects_invalid_fold_and_schedule_is_bounded():
    with pytest.raises(ValueError, match='fold'):
        build_records('.', abdomen_split='ct_cv5')
    with pytest.raises(ValueError, match='fold'):
        build_records('.', abdomen_split='ct_final', abdomen_fold=0)
    assert epoch_lr(3e-4, 1, 80, 'cosine') == pytest.approx(3e-4)
    assert epoch_lr(3e-4, 80, 80, 'cosine') == pytest.approx(3e-5)
    assert epoch_lr(3e-4, 40, 80, 'constant') == 3e-4


def test_independent_abdomen_final_fit_needs_no_validation_images(tmp_path):
    image_path, mask_path = tmp_path / 'image.png', tmp_path / 'mask.png'
    Image.fromarray(np.full((32, 32), 128, dtype=np.uint8)).save(image_path)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    Image.fromarray(mask).save(mask_path)
    row = dict(image=str(image_path), mask=str(mask_path), dataset='AbdomenUS',
               group='AbdomenUS/ct2', task=13, class_id=1)
    datasets = {name: RegionDataset([row] if name != 'val' else [], 32,
                                    PAPER_ALIGNED_TASK_IDS)
                for name in ('train', 'val', 'test')}
    args = Namespace(resume=False, abdomen_labels='paper5', size=32, base=4,
                     device='cpu', lr=3e-4, abdomen_positive_sampling=0.,
                     epochs=1, quota=1, seed=42, batch_size=1, workers=0,
                     abdomen_split='ct_final', abdomen_fold=None,
                     split_policy='trainmore_v6', lr_schedule='constant', min_lr_ratio=.1)
    (tmp_path / 'run').mkdir()
    result = run_model('independent', 'AbdomenUS', datasets, tmp_path / 'run',
                       args, 'synthetic-records', torch.ones(len(PAPER_ALIGNED_TASK_IDS)))
    assert result['best_epoch'] == 1
    assert (tmp_path / 'run' / 'AbdomenUS' / 'test_metrics.json').exists()
    assert not (tmp_path / 'run' / 'AbdomenUS' / 'best.pt').exists()
