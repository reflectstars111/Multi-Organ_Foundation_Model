import numpy as np
import pytest

from unet_moe.diagnostics import binary_diagnostics, summarize_binary


def test_single_pixel_fp_and_area_are_distinct():
    p = np.zeros((4, 4)); p[0, 0] = .5
    row = binary_diagnostics(p, np.zeros_like(p))
    metrics = summarize_binary([row])
    assert metrics['empty_false_positive_rate'] == 1
    assert metrics['empty_pixel_false_positive_rate'] == 1 / 16
    assert metrics['empty_fp_single_pixel_images'] == 1
    assert metrics['empty_predicted_area_bins'] == {'1..1': 1}


def test_ignored_pixels_never_make_a_valid_false_positive():
    target = np.array([[-1, 1], [0, 0]])
    row = binary_diagnostics(np.array([[1, .8], [.1, .1]]), target)
    assert row['ignored_prediction_pixels'] == 1
    assert row['predicted_pixels'] == 1
    assert row['tp'] == 1 and row['fp'] == 0
    assert row['valid_pixels'] == 3 and row['dice'] == 1


def test_all_ignore_and_completely_missed_foreground():
    missing = binary_diagnostics(np.ones((2, 2)), -np.ones((2, 2)))
    missed = binary_diagnostics(np.zeros((2, 2)), np.array([[1, 0], [0, 0]]))
    metrics = summarize_binary([missing, missed])
    assert missing['dice'] is None
    assert metrics['images'] == 1 and metrics['empty_images'] == 0
    assert metrics['positive_zero_prediction_images'] == 1
    assert metrics['positive_zero_overlap_images'] == 1
    assert metrics['micro_confusion'] == dict(tp=0, fp=0, fn=1, tn=3)
    assert metrics['positive_dice'] == 0


def test_invalid_probabilities_fail_explicitly():
    with pytest.raises(ValueError):
        binary_diagnostics(np.full((2, 2), np.nan), np.zeros((2, 2)))
