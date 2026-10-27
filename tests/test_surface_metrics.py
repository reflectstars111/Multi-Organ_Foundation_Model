"""Analytic CPU oracles for the retrospective pixel-space metric protocol."""

import json
import math

import numpy as np
import pytest

from unet_moe.surface_metrics import aggregate_rows, binary_metrics


def mask(*points, shape=(15, 15)):
    result = np.zeros(shape, dtype=bool)
    for point in points:
        result[point] = True
    return result


def test_identical_square_has_exact_metrics_and_four_connected_surface():
    square = np.zeros((9, 9), dtype=bool)
    square[2:7, 2:7] = True
    row = binary_metrics(square, square)
    assert row['dsc_percent'] == 100.0
    assert row['hd95_pixel'] == row['assd_pixel'] == 0.0
    assert row['pred_surface_pixels'] == row['target_surface_pixels'] == 16
    assert row['distance_status'] == 'finite'


def test_translated_single_pixel_uses_euclidean_pixel_distance():
    row = binary_metrics(mask((2, 2)), mask((5, 6)))
    assert row['dsc_percent'] == 0.0
    assert row['hd95_pixel'] == row['assd_pixel'] == 5.0


def test_unequal_surface_counts_use_pooled_distances_not_directional_mean():
    # Directed distances are [0] and [0, 4], hence pooled mean 4/3, not 1.
    row = binary_metrics(mask((2, 2)), mask((2, 2), (2, 6)))
    assert row['dsc_percent'] == pytest.approx(200 / 3)
    assert row['assd_pixel'] == pytest.approx(4 / 3)
    assert row['hd95_pixel'] == pytest.approx(3.6)
    assert row['hd95_pixel'] != pytest.approx(3.8)  # max(directional p95)


def test_disconnected_foreground_keeps_every_component():
    row = binary_metrics(mask((2, 2), (10, 10)), mask((2, 2), (10, 13)))
    assert row['pred_surface_pixels'] == row['target_surface_pixels'] == 2
    assert row['dsc_percent'] == 50.0
    assert row['assd_pixel'] == 1.5
    assert row['hd95_pixel'] == 3.0


def test_image_border_is_real_surface():
    full = np.ones((3, 3), dtype=bool)
    row = binary_metrics(full, full)
    assert row['pred_surface_pixels'] == row['target_surface_pixels'] == 8
    assert row['hd95_pixel'] == 0


def test_four_connected_surface_excludes_center_of_plus_shape():
    plus = mask((3, 3), (2, 3), (4, 3), (3, 2), (3, 4), shape=(7, 7))
    row = binary_metrics(plus, plus)
    # Eight-connected erosion would retain the central pixel as a boundary.
    assert row['pred_pixels'] == 5
    assert row['pred_surface_pixels'] == row['target_surface_pixels'] == 4


def test_both_empty_has_exact_perfect_score_and_zero_distance():
    row = binary_metrics(mask(), mask())
    assert row['case_status'] == 'both_empty'
    assert row['is_positive'] is False
    assert row['dsc_percent'] == 100
    assert row['hd95_pixel'] == row['assd_pixel'] == 0


@pytest.mark.parametrize('prediction_empty', [True, False])
def test_one_empty_strict_preserves_infinity_without_non_json_numbers(prediction_empty):
    pred, target = (mask(), mask((2, 2))) if prediction_empty else (mask((2, 2)), mask())
    row = binary_metrics(pred, target, empty_policy='strict')
    assert row['case_status'] == ('pred_empty' if prediction_empty else 'target_empty')
    assert row['is_positive'] is prediction_empty
    assert row['dsc_percent'] == 0
    assert row['hd95_pixel'] is row['assd_pixel'] is None
    assert row['distance_status'] == 'infinite'
    json.dumps(row, allow_nan=False)


def test_one_empty_diagonal_policy_is_explicit_finite_pixel_center_penalty():
    row = binary_metrics(mask(shape=(4, 5)), mask((2, 2), shape=(4, 5)), empty_policy='diagonal')
    assert row['empty_policy'] == 'diagonal'
    assert row['distance_status'] == 'finite'
    assert row['hd95_pixel'] == row['assd_pixel'] == 5
    assert row['distance_penalized'] is True


def test_default_diagonal_policy_and_512_scale_are_declared():
    row = binary_metrics(mask(shape=(512, 512)), mask((2, 2), shape=(512, 512)))
    assert row['empty_policy'] == 'diagonal'
    assert row['diagonal_pixel'] == pytest.approx(722.6631303726516)
    assert aggregate_rows([row])['n_penalized'] == 1


def test_all_ignored_is_excluded_instead_of_perfect():
    row = binary_metrics(np.ones((5, 5), dtype=bool), np.full((5, 5), -1))
    assert row['case_status'] == row['distance_status'] == 'excluded'
    assert row['dsc_percent'] is row['hd95_pixel'] is row['assd_pixel'] is None
    assert row['is_positive'] is None
    assert row['valid_pixels'] == 0


def test_explicit_valid_mask_and_negative_target_mask_combine():
    target = np.zeros((7, 7), dtype=np.int8)
    target[0, 0] = -1
    target[3, 3] = 1
    pred = mask((0, 0), (3, 3), (6, 6), shape=(7, 7))
    valid = np.ones((7, 7), dtype=bool)
    valid[6, 6] = False
    row = binary_metrics(pred, target, valid)
    assert row['valid_pixels'] == 47
    assert row['pred_pixels'] == row['target_pixels'] == 1
    assert row['dsc_percent'] == 100
    assert row['hd95_pixel'] == 0


def test_ignored_hole_does_not_fabricate_surface():
    target = np.ones((7, 7), dtype=np.int8)
    target[3, 3] = -1
    row = binary_metrics(np.ones((7, 7), dtype=bool), target)
    # Only the true exterior perimeter remains; the four neighbours of the
    # ignored hole must not become surface pixels.
    assert row['pred_surface_pixels'] == row['target_surface_pixels'] == 24
    assert row['distance_status'] == 'finite'


def test_surface_adjacent_to_ignore_is_undefined_when_nothing_remains():
    target = np.full((5, 5), -1, dtype=np.int8)
    target[2, 2] = 1
    row = binary_metrics(mask((2, 2), shape=(5, 5)), target)
    assert row['dsc_percent'] == 100
    assert row['pred_surface_pixels'] == row['target_surface_pixels'] == 0
    assert row['distance_status'] == 'undefined'
    assert row['hd95_pixel'] is row['assd_pixel'] is None


def test_partial_ignored_surface_is_reported():
    square = np.zeros((7, 7), dtype=np.int8)
    square[2:5, 2:5] = 1
    square[1, 3] = -1
    row = binary_metrics(square == 1, square)
    assert row['pred_surface_pixels'] == row['target_surface_pixels'] == 7
    assert row['pred_surface_removed_by_ignore'] == row['target_surface_removed_by_ignore'] == 1


def test_all_case_and_positive_only_aggregates_have_explicit_denominators():
    positive = binary_metrics(mask((2, 2)), mask((2, 2)))
    negative = binary_metrics(mask((2, 2)), mask(), empty_policy='diagonal')
    excluded = binary_metrics(mask(), np.full((15, 15), -1))
    rows = [positive, negative, excluded]
    aggregate = aggregate_rows(rows)
    assert aggregate['n_rows'] == 3
    assert aggregate['n_excluded'] == 1
    assert aggregate['n_included'] == 2
    assert aggregate['n_positive'] == 1
    assert aggregate['dsc_percent'] == 50
    assert aggregate['hd95_pixel'] == pytest.approx(math.hypot(14, 14) / 2)
    positive_aggregate = aggregate_rows(rows, positive_only=True)
    assert positive_aggregate['n_included'] == 1
    assert positive_aggregate['dsc_percent'] == 100
    assert positive_aggregate['assd_pixel'] == 0


def test_aggregate_does_not_drop_infinite_distance_failures():
    finite = binary_metrics(mask((2, 2)), mask((2, 3)))
    infinite = binary_metrics(mask(), mask((2, 2)), empty_policy='strict')
    result = aggregate_rows([finite, infinite])
    assert result['distance_status'] == 'infinite'
    assert result['n_infinite'] == 1
    assert result['n_finite'] == 1
    assert result['hd95_pixel'] is result['assd_pixel'] is None
    assert result['finite_subset_mean']['hd95_pixel'] == 1
    assert result['dsc_percent'] == 0
    json.dumps(result, allow_nan=False)


def test_undefined_surface_prevents_main_finite_average():
    target = np.full((15, 15), -1)
    target[2, 2] = 1
    undefined = binary_metrics(mask((2, 2)), target)
    finite = binary_metrics(mask((2, 2)), mask((2, 2)))
    result = aggregate_rows([undefined, finite])
    assert result['distance_status'] == 'undefined'
    assert result['n_undefined'] == 1
    assert result['hd95_pixel'] is None
    assert result['finite_subset_mean']['hd95_pixel'] == 0
    assert result['dsc_percent'] == 100


def test_empty_aggregate_has_no_invented_measurements():
    result = aggregate_rows([])
    assert result['distance_status'] == 'excluded'
    assert result['n_included'] == 0
    assert result['dsc_percent'] is result['hd95_pixel'] is result['assd_pixel'] is None


@pytest.mark.parametrize('pred,target,valid,error', [
    (np.zeros((3, 3)), np.zeros((4, 3)), None, 'shape'),
    (np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), None, '2D'),
    (np.zeros((0, 3)), np.zeros((0, 3)), None, 'nonempty'),
    (np.full((3, 3), .5), np.zeros((3, 3)), None, 'prediction'),
    (np.zeros((3, 3)), np.full((3, 3), 2), None, 'target'),
    (np.zeros((3, 3)), np.full((3, 3), np.nan), None, 'target'),
    (np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((2, 3)), 'valid'),
])
def test_invalid_inputs_fail_loudly(pred, target, valid, error):
    with pytest.raises(ValueError, match=error):
        binary_metrics(pred, target, valid)


def test_unknown_empty_policy_is_rejected_even_when_no_empty_foreground():
    with pytest.raises(ValueError, match='empty_policy'):
        binary_metrics(mask((2, 2)), mask((2, 2)), empty_policy='skip')


@pytest.mark.parametrize('patch,error', [
    ({'distance_status': 'missing'}, 'distance_status'),
    ({'case_status': 'missing'}, 'case_status'),
    ({'distance_status': 'excluded'}, 'excluded'),
    ({'hd95_pixel': float('inf')}, 'finite'),
    ({'assd_pixel': float('nan')}, 'finite'),
    ({'dsc_percent': 101}, 'dsc_percent'),
    ({'is_positive': False}, 'is_positive'),
    ({'distance_penalized': True}, 'distance_penalized'),
    ({'empty_policy': 'skip'}, 'empty_policy'),
])
def test_aggregate_rejects_corrupted_statuses_and_non_json_numbers(patch, error):
    row = binary_metrics(mask((2, 2)), mask((2, 2)))
    row.update(patch)
    with pytest.raises(ValueError, match=error):
        aggregate_rows([row])


def test_infinite_numeric_null_is_required_even_for_strict_cases():
    row = binary_metrics(mask(), mask((2, 2)), empty_policy='strict')
    row['hd95_pixel'] = float('inf')
    with pytest.raises(ValueError, match='null'):
        aggregate_rows([row])


def test_diagonal_penalty_cannot_be_falsely_reported_as_zero():
    row = binary_metrics(mask(), mask((2, 2)))
    row['hd95_pixel'] = 0.0
    with pytest.raises(ValueError, match='diagonal'):
        aggregate_rows([row])


def test_infinite_and_undefined_are_both_retained_and_infinite_takes_precedence():
    target = np.full((15, 15), -1)
    target[2, 2] = 1
    undefined = binary_metrics(mask((2, 2)), target)
    infinite = binary_metrics(mask(), mask((2, 2)), empty_policy='strict')
    result = aggregate_rows([undefined, infinite])
    assert result['distance_status'] == 'infinite'
    assert result['n_infinite'] == result['n_undefined'] == 1
    assert result['n_finite'] == result['finite_subset_mean']['n'] == 0
    assert result['hd95_pixel'] is None
    assert result['finite_subset_mean']['hd95_pixel'] is None
    assert result['dsc_percent'] == 50
