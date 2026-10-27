import json
from itertools import product

import numpy as np
import pytest
import torch

from unet_moe.external_baselines import support
from unet_moe.external_baselines.valid_region import crop_resize, valid_roi


def brute_force(array, min_side, min_fraction=0.25):
    height, width = array.shape
    if min(height, width) < min_side:
        return None
    if (array >= 0).all():
        return (0, 0, height, width)
    candidates = []
    for top in range(height):
        for left in range(width):
            for bottom in range(top+min_side, height+1):
                for right in range(left+min_side, width+1):
                    part = array[top:bottom, left:right]
                    positive = int((part == 1).sum())
                    if (part >= 0).all() and 0 < positive < part.size and positive >= np.ceil((array == 1).sum() * min_fraction):
                        key = (part.size, positive, -top, -left, -bottom, -right)
                        candidates.append((key, (top, left, bottom, right)))
    return max(candidates)[1] if candidates else None


def test_histogram_matches_exhaustive_rectangles_with_class_and_size_constraints():
    rng = np.random.default_rng(42)
    for _ in range(100):
        target = rng.choice([-1, 0, 1], size=(5, 6), p=[0.15, 0.5, 0.35])
        for min_side, fraction in product([1, 2, 3], [0.0, 0.25, 0.75]):
            assert valid_roi(target, min_side=min_side, minimum_foreground_fraction=fraction) == brute_force(target, min_side, fraction)


def test_foreground_fraction_is_search_constraint_not_posthoc_rejection():
    target = torch.full((14, 30), -1.0)
    target[:4, :] = 0
    target[0, 0] = 1  # largest rectangle 120 pixels but only 1/13 foreground
    target[8:14, 20:26] = 0
    target[9:12, 21:25] = 1
    assert valid_roi(target, min_side=2) == (8, 20, 14, 26)
    assert valid_roi(target, min_side=2, minimum_foreground_fraction=0) == (0, 0, 4, 30)
    assert valid_roi(target, min_side=2, minimum_foreground_fraction=1) is None


def test_full_observed_negative_preserved_but_partial_foreground_only_rejected():
    assert valid_roi(torch.zeros(40, 50)) == (0, 0, 40, 50)
    assert valid_roi(torch.ones(40, 50)) == (0, 0, 40, 50)
    partial = torch.ones(40, 50)
    partial[:2] = -1
    assert valid_roi(partial) is None
    assert valid_roi(torch.full((40, 50), -1)) is None


def test_minimum_side_is_applied_during_search_not_to_winning_thin_rectangle():
    target = torch.full((12, 30), -1.0)
    target[:2, :] = 0
    target[0, 0] = 1  # largest area 60 but short side 2
    target[7:12, 20:25] = 0
    target[8, 21] = 1
    assert valid_roi(target, min_side=4) == (7, 20, 12, 25)


def test_crop_resize_keeps_image_and_mask_coordinates_and_never_unknown():
    image = torch.arange(8*10).reshape(1, 8, 10).float()
    target = torch.full((1, 8, 10), -1.0)
    target[:, 2:6, 3:7] = 0
    target[:, 3:5, 4:6] = 1
    roi = valid_roi(target[0], min_side=2)
    cropped_image, cropped_mask = crop_resize(image, target, roi, 8)
    assert cropped_image.shape == cropped_mask.shape == (1, 8, 8)
    expected = torch.nn.functional.interpolate(image[:, 2:6, 3:7].unsqueeze(0),
                size=(8, 8), mode='bilinear', align_corners=False).squeeze(0)
    torch.testing.assert_close(cropped_image, expected)
    assert ((cropped_mask == 0) | (cropped_mask == 1)).all()
    assert cropped_mask[:, 2:6, 2:6].eq(1).all()
    assert cropped_mask.eq(1).sum() == 16
    with pytest.raises(ValueError, match='Unknown'):
        crop_resize(image, target, (0, 0, 8, 10), 8)


class FakeDataset:
    opened_splits = []
    def __init__(self, records, size):
        self.groups = [[r] for r in records]
        self.size = size
    def __len__(self):
        return len(self.groups)
    def __getitem__(self, index):
        row = self.groups[index][0]
        self.opened_splits.append(row['split'])
        target = torch.zeros(1, self.size, self.size)
        target[:, 2:6, 2:6] = 1
        if row.get('partial'):
            target[:, :, 0] = -1
        if row.get('no_background'):
            target[:] = 1
            target[:, :, 0] = -1
        return target.clamp_min(0).clone(), target, row['dataset'], ''


def records():
    rows = [dict(dataset='AUL', image=f'train{i}', group=f'patient{i//2}', split='train', partial=(i%3==0))
            for i in range(14)]
    rows += [dict(dataset='AUL', image='partial_mass_only', group='ignored', split='train', no_background=True)]
    rows += [dict(dataset='AUL', image='validation', group='patient0', split='val'),
             dict(dataset='AUL', image='test', group='patient1', split='test')]
    return rows


def test_support_manifest_is_train_only_with_frozen_roi_and_every_query_verified(monkeypatch):
    monkeypatch.setattr(support, 'UnionDataset', FakeDataset)
    FakeDataset.opened_splits = []
    rows = records()
    manifest = support.support_manifest(rows, size=8, support_size=8, min_side=2)
    assert set(FakeDataset.opened_splits) == {'train'}
    assert manifest['domain_statistics']['AUL']['cropped'] == 5
    assert manifest['domain_statistics']['AUL']['excluded']['no_known_background'] == 1
    assert manifest['query_verification']['unique_queries_by_split'] == dict(train=15, val=1, test=1)
    assert manifest['query_verification']['minimum_eligible_images']['AUL']['train'] == 12
    bank = support.SupportBank(rows, json.loads(json.dumps(manifest)), size=8)
    query = rows[0]
    selected = bank.indices(query, training_seed=42, epoch=1, draw_position=7)
    assert len(selected) == len(set(selected)) == 8
    assert all(bank.entries[i]['group'] != query['group'] and bank.entries[i]['image'] != query['image'] for i in selected)
    assert selected == bank.indices(query, training_seed=42, epoch=1, draw_position=7)
    assert selected != bank.indices(query, training_seed=42, epoch=2, draw_position=7)
    fixed = bank.indices(query)
    assert fixed == bank.indices(query, epoch=80, draw_position=100000)
    images, labels, chosen = bank.batch([query], device='cpu', training_seed=42, epoch=1, draw_positions=[7])
    assert images.shape == labels.shape == (1, 8, 1, 8, 8)
    assert chosen == [selected]
    assert ((labels == 0) | (labels == 1)).all()


def test_insufficient_pool_fails_without_replacement_or_cross_domain(monkeypatch):
    monkeypatch.setattr(support, 'UnionDataset', FakeDataset)
    with pytest.raises(ValueError, match='Insufficient train supports'):
        support.support_manifest(records()[:8], size=8, support_size=8, min_side=2)


def test_support_target_hash_detects_changes(monkeypatch):
    monkeypatch.setattr(support, 'UnionDataset', FakeDataset)
    rows = records()
    manifest = support.support_manifest(rows, size=8, support_size=8, min_side=2)
    manifest['pools']['AUL'][0]['roi_target_sha256'] = 'changed'
    bank = support.SupportBank(rows, manifest, size=8)
    with pytest.raises(ValueError, match='labels changed'):
        bank.item(0)
