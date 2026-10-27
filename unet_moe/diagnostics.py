"""Fixed-threshold segmentation diagnostics; never select deployment thresholds."""
from collections import Counter

import numpy as np


AREA_EDGES = (0, 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return dict(count=int(values.size), mean=float(values.mean()),
                **{name: float(np.quantile(values, q)) for name, q in
                   [('min', 0), ('p25', .25), ('median', .5), ('p75', .75),
                    ('p90', .9), ('p95', .95), ('max', 1)]})


def binary_diagnostics(probability, target, threshold=.5):
    """Measure one annotated channel. Negative target pixels are ignored."""
    probability, target = np.asarray(probability), np.asarray(target)
    if probability.shape != target.shape or probability.ndim != 2:
        raise ValueError('Expected equal two-dimensional probability and target arrays')
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1]')
    valid = target >= 0
    truth = target > 0
    raw = probability >= threshold
    pred = raw & valid
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth & valid).sum())
    truth_pixels, predicted_pixels = tp + fn, tp + fp
    denominator = truth_pixels + predicted_pixels
    return dict(valid_pixels=int(valid.sum()), ignored_pixels=int((~valid).sum()),
                truth_pixels=truth_pixels, predicted_pixels=predicted_pixels,
                tp=tp, fp=fp, fn=fn, tn=tn,
                dice=(2 * tp / denominator if denominator else 1.) if valid.any() else None,
                recall=tp / truth_pixels if truth_pixels else None,
                precision=tp / predicted_pixels if predicted_pixels else None,
                predicted_fraction=predicted_pixels / int(valid.sum()) if valid.any() else None,
                ignored_prediction_pixels=int((raw & ~valid).sum()),
                max_probability=float(probability[valid].max()) if valid.any() else None,
                mean_probability=float(probability[valid].mean()) if valid.any() else None,
                mean_probability_on_truth=float(probability[truth].mean()) if truth.any() else None)


def summarize_binary(rows):
    """Aggregate image records, retaining image-average and pixel-micro metrics."""
    rows = [r for r in rows if r['valid_pixels']]
    positive = [r for r in rows if r['truth_pixels']]
    empty = [r for r in rows if not r['truth_pixels']]
    empty_fp = [r for r in empty if r['predicted_pixels']]
    counts = {key: sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    bins = Counter()
    for row in empty:
        pixels = row['predicted_pixels']
        if not pixels:
            bins['0'] += 1
            continue
        for low, high in zip(AREA_EDGES[:-1], AREA_EDGES[1:]):
            if low < pixels <= high:
                bins[f'{low + 1}..{high}'] += 1
                break
        else:
            bins[f'>{AREA_EDGES[-1]}'] += 1
    empty_pixels = sum(r['predicted_pixels'] for r in empty)
    empty_valid = sum(r['valid_pixels'] for r in empty)
    return dict(images=len(rows), positive_images=len(positive), empty_images=len(empty),
                mean_dice=float(np.mean([r['dice'] for r in rows])) if rows else None,
                positive_dice=float(np.mean([r['dice'] for r in positive])) if positive else None,
                positive_recall=distribution([r['recall'] for r in positive]),
                positive_precision=distribution([r['precision'] for r in positive if r['precision'] is not None]),
                positive_predicted_pixels=distribution([r['predicted_pixels'] for r in positive]),
                positive_truth_pixels=distribution([r['truth_pixels'] for r in positive]),
                positive_zero_prediction_images=sum(r['predicted_pixels'] == 0 for r in positive),
                positive_zero_overlap_images=sum(r['tp'] == 0 for r in positive),
                positive_recall_below_half_images=sum(r['recall'] < .5 for r in positive),
                empty_false_positive_images=len(empty_fp),
                empty_false_positive_rate=len(empty_fp) / len(empty) if empty else None,
                empty_predicted_pixels=distribution([r['predicted_pixels'] for r in empty]),
                empty_fp_predicted_pixels=distribution([r['predicted_pixels'] for r in empty_fp]),
                empty_predicted_fraction=distribution([r['predicted_fraction'] for r in empty]),
                empty_predicted_area_bins=dict(bins),
                empty_fp_single_pixel_images=sum(r['predicted_pixels'] == 1 for r in empty),
                empty_fp_at_most_16_pixels_images=sum(0 < r['predicted_pixels'] <= 16 for r in empty),
                empty_predicted_pixels_total=empty_pixels,
                empty_valid_pixels_total=empty_valid,
                empty_pixel_false_positive_rate=empty_pixels / empty_valid if empty_valid else None,
                micro_confusion=counts,
                micro_dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                micro_recall=tp / (tp + fn) if tp + fn else None,
                micro_precision=tp / (tp + fp) if tp + fp else None,
                micro_specificity=tn / (tn + fp) if tn + fp else None)
