import numpy as np

from scripts.screen_abdomen_postprocess import component_counts, eligible, specifications, summarize


def test_components_are_selected_without_using_ignore_mask():
    specs = [dict(id='raw', threshold=.5, min_area=0, largest_only=False),
             dict(id='floor', threshold=.5, min_area=3, largest_only=False),
             dict(id='largest', threshold=.5, min_area=0, largest_only=True)]
    probability = np.array([[.9, .9, .9, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, .9]], dtype=np.float32)
    target = np.zeros((3, 6), dtype=np.int8)
    target[0, 0] = 1
    target[0, 1] = -1
    counts = component_counts(probability, target, specs)
    assert counts['raw']['predicted_pixels'] == 3
    assert counts['raw']['tp'] == 1
    # The ignored center pixel links the first and third pixel for the rule.
    # It contributes neither a predicted valid pixel nor a true positive.
    assert counts['largest']['predicted_pixels'] == 2
    assert counts['floor']['predicted_pixels'] == 2
    assert counts['largest']['tp'] == 1


def test_summarize_skips_fully_ignored_masks():
    result = summarize([dict(valid_pixels=0, truth_pixels=0, predicted_pixels=0, tp=0),
                        dict(valid_pixels=10, truth_pixels=0, predicted_pixels=1, tp=0)])
    assert result['images'] == 1
    assert result['empty_images'] == 1
    assert result['empty_mean_false_positive_area_fraction'] == .1


def test_empty_fp_reduction_that_erases_positives_is_rejected():
    positive = [dict(valid_pixels=100, truth_pixels=20, predicted_pixels=20, tp=15)] * 10
    empty = [dict(valid_pixels=100, truth_pixels=0, predicted_pixels=20, tp=0)] * 10
    baseline = summarize(positive + empty)
    collapsed = summarize([
        dict(valid_pixels=100, truth_pixels=20, predicted_pixels=0, tp=0)] * 10 +
        [dict(valid_pixels=100, truth_pixels=0, predicted_pixels=0, tp=0)] * 10)
    assert collapsed['empty_false_positive_rate'] == 0
    assert not eligible(collapsed, baseline, 'precision')
    assert not eligible(collapsed, baseline, 'recall')
