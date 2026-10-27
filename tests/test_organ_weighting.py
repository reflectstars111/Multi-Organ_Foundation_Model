"""Behavioral checks for train-only Q01 exposure and foreground weighting."""
import importlib
import math

import numpy as np
from PIL import Image
import pytest
import torch

from unet_moe.fair_suite import common_loss
from unet_moe.region import DOMAINS, PAPER_ALIGNED_TASK_IDS, RegionDataset


@pytest.fixture
def weighting():
    assert importlib.util.find_spec('unet_moe.organ_weighting') is not None, 'Q01 weighting is missing'
    return importlib.import_module('unet_moe.organ_weighting')


def loss_weights(channels):
    return dict(organ_weights=[2.] * channels, foreground_weights=[4.] * channels,
                foreground_normalizers=[1.3] * channels, foreground_proportions=[.1] * channels)


def mixed_targets():
    target = torch.full((5, 3, 4, 4), -1.)
    target[:, 0] = 0
    target[1:, 0, :2] = 1
    target[2:, 1] = 0
    target[2:, 1, :1, :1] = 1
    target[3, 2] = 0
    target[4, 2] = 1
    target[4, :, :1, :1] = -1
    return target


def test_image_mean_exactly_matches_old_loss_and_gradient(weighting):
    torch.manual_seed(4)
    old = torch.randn(5, 3, 4, 4, requires_grad=True)
    new = old.detach().clone().requires_grad_()
    target = mixed_targets()
    expected = common_loss(old, target)
    actual = weighting.weighted_loss(new, target, 'image_mean', loss_weights(3))
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.equal(old.grad, new.grad)


@pytest.mark.parametrize('arm', ['organ_exposure', 'organ_exposure_fg_bg'])
def test_identity_weights_recover_original_loss_and_gradient(weighting, arm):
    torch.manual_seed(4)
    old = torch.randn(5, 3, 4, 4, requires_grad=True)
    new = old.detach().clone().requires_grad_()
    target = mixed_targets()
    weights = dict(organ_weights=[1.] * 3, foreground_weights=[1.] * 3,
                   foreground_normalizers=[1.] * 3)
    expected = common_loss(old, target)
    actual = weighting.weighted_loss(new, target, arm, weights)
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.equal(old.grad, new.grad)


@pytest.mark.parametrize('arm', ['image_mean', 'organ_exposure', 'organ_exposure_fg_bg'])
def test_ignore_pixels_have_no_gradient_even_on_all_ignored_images(weighting, arm):
    target = mixed_targets()
    target[0] = -1
    logits = torch.randn_like(target, requires_grad=True)
    loss = weighting.weighted_loss(logits, target, arm, loss_weights(3))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[target < 0]) == 0
    assert torch.count_nonzero(logits.grad[target >= 0]) > 0


def test_exposure_weight_is_not_cancelled_by_per_image_renormalization(weighting):
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    base = common_loss(logits, target)
    result = weighting.loss_components(logits, target, 'organ_exposure', loss_weights(3))
    assert result['loss'].item() == pytest.approx(2 * base.item())
    assert torch.allclose(result['per_channel_weighted'], 2 * result['per_channel_unweighted'])
    assert result['active'].shape == (1, 3)
    assert result['per_image'].shape == (1,)


def test_c_weights_only_positive_image_bce_preserving_empty_penalty_and_dice(weighting):
    logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
    target = torch.tensor([[[[1., 0.], [0., -1.]]], [[[0., 0.], [0., 0.]]]])
    weights = dict(organ_weights=[1.], foreground_weights=[4.],
                   foreground_normalizers=[1.3], foreground_proportions=[.1])
    result = weighting.loss_components(logits, target, 'organ_exposure_fg_bg', weights)
    # Positive image: BCE (4+1+1)/(3*1.3) * log(2); unchanged Dice 1-2/3.5.
    assert result['per_image'][0].item() == pytest.approx(math.log(2) * 6 / 3.9 + 1 - 2 / 3.5)
    # Empty image receives no foreground normalization or downweighting.
    assert result['per_image'][1].item() == pytest.approx(math.log(2) + 2 / 3)
    assert result['per_channel_unweighted'][0, 0].item() == pytest.approx(math.log(2) + 1 - 2 / 3.5)


@pytest.mark.parametrize('arm', ['image_mean', 'organ_exposure', 'organ_exposure_fg_bg'])
def test_uneven_microbatch_accumulation_matches_full_batch_gradient(weighting, arm):
    torch.manual_seed(8)
    initial = torch.randn(3, 4, 4)
    full = initial.clone().requires_grad_()
    accumulated = initial.clone().requires_grad_()
    target = mixed_targets()
    factors = torch.arange(1., 6.)[:, None, None, None]
    weighting.weighted_loss(full[None] * factors, target, arm, loss_weights(3)).backward()
    for start, stop in [(0, 2), (2, 4), (4, 5)]:
        value = weighting.weighted_loss(accumulated[None] * factors[start:stop], target[start:stop],
                                        arm, loss_weights(3))
        (value * ((stop - start) / 5)).backward()
    assert torch.allclose(full.grad, accumulated.grad, rtol=2e-6, atol=1e-7)


def domain_dataset(tmp_path, size=8):
    tasks = [[0], [1], [2], [3, 4], [6], [8], [9], [10, 11, 12], [13, 14, 16, 18, 20]]
    rows = []
    for domain, ids in zip(DOMAINS, tasks):
        for sample in range(2 if domain == 'AbdomenUS' else 1):
            image = tmp_path / f'{domain}_{sample}.png'
            mask = tmp_path / f'{domain}_{sample}_mask.png'
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image)
            values = np.zeros((8, 8), dtype=np.uint8)
            if sample == 0:
                values[0, 0] = 1 if domain in ('CAMUS', 'AbdomenUS') else 255
            if domain == 'AbdomenUS':
                values[-1] = 255
            Image.fromarray(values).save(mask)
            for task in ids:
                rows.append(dict(image=str(image), mask=str(mask), task=task, dataset=domain,
                                 class_id=task - 12 if domain == 'AbdomenUS' else task - 9 if domain == 'CAMUS' else 0,
                                 group=f'{domain}/source_{sample}', split='train'))
    return RegionDataset(rows, size=size, task_ids=PAPER_ALIGNED_TASK_IDS)


def test_train_statistics_balance_expected_structure_coefficients_not_dataset_sizes(weighting, tmp_path):
    dataset = domain_dataset(tmp_path)
    result = weighting.collect_statistics(dataset)
    assert result['task_ids'] == PAPER_ALIGNED_TASK_IDS
    assert result['images'] == 10
    assert result['domain_image_counts']['AbdomenUS'] == 2
    exposure = [row['exposure'] for row in result['per_structure'].values()]
    coefficients = np.asarray(exposure) * np.asarray(result['organ_weights'])
    assert np.allclose(coefficients, 1 / 16)
    assert coefficients.sum() == pytest.approx(1.)
    assert result['organ_weights'][:3] == pytest.approx([.5625] * 3)
    assert result['organ_weights'][-5:] == pytest.approx([2.8125] * 5)
    liver = result['per_structure']['abdomen_liver']
    assert liver['active_images'] == 2
    assert liver['positive_images'] == 1
    assert liver['empty_images'] == 1
    assert liver['valid_pixels'] == 112
    assert liver['valid_positive_pixels'] == 56
    assert liver['foreground_pixels'] == 1
    assert liver['foreground_fraction_positive_smoothed'] == pytest.approx(2 / 58)
    assert liver['foreground_weight'] == 4
    assert result['group_semantics']['AbdomenUS'] == 'CT source; not a verified patient identifier'
    assert result['group_source_image_counts']['AbdomenUS'] == {'AbdomenUS/source_0': 1, 'AbdomenUS/source_1': 1}


def test_statistics_use_resized_labels_and_c_expected_pixel_weight_is_one(weighting, tmp_path):
    result = weighting.collect_statistics(domain_dataset(tmp_path, size=16))
    liver = result['per_structure']['abdomen_liver']
    assert liver['valid_pixels'] == 448
    assert liver['foreground_pixels'] == 4
    for q, r, normalizer in zip(result['foreground_proportions'], result['foreground_weights'],
                                result['foreground_normalizers']):
        assert 1 <= r <= 4
        assert (q * r + (1 - q)) / normalizer == pytest.approx(1.)


def test_missing_aul_annotation_changes_exposure_using_actual_active_channel_count(weighting, tmp_path):
    dataset = domain_dataset(tmp_path)
    mass = next(row for rows in dataset.groups for row in rows if row['task'] == 4).copy()
    image = tmp_path / 'aul_mass_only.png'
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(image)
    mass['image'], mass['group'] = str(image), 'AUL/mass_only'
    dataset.groups.append([mass])
    result = weighting.collect_statistics(dataset)
    assert result['per_structure']['liver']['exposure'] == pytest.approx(1 / 36)
    assert result['per_structure']['liver_mass']['exposure'] == pytest.approx(1 / 12)
    assert result['per_structure']['liver']['organ_weight'] == pytest.approx(2.25)
    assert result['per_structure']['liver_mass']['organ_weight'] == pytest.approx(.75)


def test_statistics_select_first_available_positive_and_empty_diagnostics_without_duplicates(weighting, tmp_path):
    result = weighting.collect_statistics(domain_dataset(tmp_path))
    representatives = result['representative_indices']
    assert representatives['abdomen_liver'] == {'positive': 8, 'empty': 9}
    assert representatives['abdomen_kidney'] == {'positive': None, 'empty': 8}
    assert representatives['breast_lesion'] == {'positive': 0, 'empty': None}
    assert result['diagnostic_indices'] == list(range(10))


def test_statistics_reject_validation_data(weighting, tmp_path):
    dataset = domain_dataset(tmp_path)
    dataset.groups[0][0]['split'] = 'val'
    with pytest.raises(ValueError, match='train'):
        weighting.collect_statistics(dataset)


def test_invalid_weight_shape_and_arm_fail_before_training(weighting):
    logits = torch.zeros(1, 3, 2, 2)
    target = torch.zeros_like(logits)
    with pytest.raises(ValueError, match='arm'):
        weighting.weighted_loss(logits, target, 'unknown', loss_weights(3))
    with pytest.raises(ValueError, match='organ_weights'):
        weighting.weighted_loss(logits, target, 'organ_exposure', loss_weights(2))
