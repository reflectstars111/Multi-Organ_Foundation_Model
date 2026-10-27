"""Exact binary 2D segmentation metrics in the evaluated image's pixel grid.

This module is independent of torch and the training/checkpoint selection code.
DSC is expressed as a percentage without a smoothing constant. Surfaces use
4-connected foreground erosion with the exterior of the image as background.
HD95 is the 95th percentile of the *concatenated* directed surface distances;
ASSD is the mean of that same concatenation (not a mean of directional means).
Euclidean pixel spacing is one in both axes.

Ignored pixels do not become background: surface pixels touching an ignored
pixel in the 4-neighbourhood are removed from both surfaces. If both masks have
foreground but either retained surface is empty, the distances are undefined.

The default one-empty penalty is the maximum distance between pixel centres,
``hypot(height - 1, width - 1)``. It is a declared finite *penalty*, not a
measured surface distance. ``empty_policy='strict'`` instead retains infinity
as a status. JSON numbers are always finite: undefined/infinite/excluded
distances are represented by ``None`` and an explicit ``distance_status``.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
import math
from numbers import Real
from typing import Any

import numpy as np
from scipy import ndimage


_CONNECTIVITY = ndimage.generate_binary_structure(2, 1)
_DISTANCE_STATUSES = frozenset(('finite', 'infinite', 'undefined', 'excluded'))
_CASE_STATUSES = frozenset(('both_nonempty', 'both_empty', 'pred_empty', 'target_empty', 'excluded'))


def _binary_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must contain only boolean or 0/1 values')
    return array.astype(bool, copy=False)


def _surface(mask: np.ndarray, safe: np.ndarray) -> tuple[np.ndarray, int]:
    boundary = mask & ~ndimage.binary_erosion(mask, structure=_CONNECTIVITY, border_value=0)
    retained = boundary & safe
    return retained, int(np.count_nonzero(boundary & ~safe))


def binary_metrics(
    pred: Any,
    target: Any,
    valid: Any = None,
    *,
    empty_policy: str = 'diagonal',
) -> dict[str, Any]:
    """Evaluate one binary image with an optional per-pixel validity mask.

    ``pred`` must be a thresholded boolean/0/1 mask; callers are responsible
    for applying their declared probability threshold. ``target`` may contain
    -1 for ignored pixels, 0 for background and 1 for foreground. ``valid`` is
    combined with the target's nonignored mask. Inputs must be nonempty 2D
    arrays of the same shape. Predictions outside the validity mask have no
    effect on either the overlap or the retained surfaces.

    ``is_positive`` refers to target foreground within valid pixels. A wholly
    ignored image has ``is_positive=None`` and is excluded from aggregation.
    """
    if empty_policy not in ('strict', 'diagonal'):
        raise ValueError("empty_policy must be 'strict' or 'diagonal'")
    pred_array = np.asarray(pred)
    target_array = np.asarray(target)
    if pred_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError('prediction and target must be 2D arrays')
    if pred_array.shape != target_array.shape:
        raise ValueError('prediction and target must have the same shape')
    if 0 in pred_array.shape:
        raise ValueError('prediction and target dimensions must be nonempty')
    prediction = _binary_array(pred_array, 'prediction')
    if not np.isin(target_array, (-1, 0, 1)).all():
        raise ValueError('target must contain only -1, 0 and 1')
    validity = target_array != -1
    if valid is not None:
        valid_array = np.asarray(valid)
        if valid_array.shape != prediction.shape:
            raise ValueError('valid must have the same shape as prediction and target')
        validity &= _binary_array(valid_array, 'valid')
    prediction = prediction & validity
    truth = (target_array == 1) & validity

    n_valid = int(np.count_nonzero(validity))
    n_pred = int(np.count_nonzero(prediction))
    n_truth = int(np.count_nonzero(truth))
    n_intersection = int(np.count_nonzero(prediction & truth))
    # Out-of-image pixels count as background, not ignored observations.
    safe = ~ndimage.binary_dilation(~validity, structure=_CONNECTIVITY, border_value=0)
    pred_surface, pred_removed = _surface(prediction, safe)
    target_surface, target_removed = _surface(truth, safe)
    n_pred_surface = int(np.count_nonzero(pred_surface))
    n_target_surface = int(np.count_nonzero(target_surface))
    row: dict[str, Any] = {
        'dsc_percent': None,
        'hd95_pixel': None,
        'assd_pixel': None,
        'case_status': 'excluded',
        'distance_status': 'excluded',
        'distance_penalized': False,
        'empty_policy': empty_policy,
        'diagonal_pixel': math.hypot(prediction.shape[0] - 1, prediction.shape[1] - 1),
        'is_positive': None,
        'valid_pixels': n_valid,
        'ignored_pixels': int(validity.size - n_valid),
        'pred_pixels': n_pred,
        'target_pixels': n_truth,
        'intersection_pixels': n_intersection,
        'pred_surface_pixels': n_pred_surface,
        'target_surface_pixels': n_target_surface,
        'pred_surface_removed_by_ignore': pred_removed,
        'target_surface_removed_by_ignore': target_removed,
    }
    if not n_valid:
        return row

    row['is_positive'] = bool(n_truth)
    row['dsc_percent'] = 200.0 * n_intersection / (n_pred + n_truth) if n_pred + n_truth else 100.0
    if not n_pred and not n_truth:
        row.update(case_status='both_empty', distance_status='finite', hd95_pixel=0.0, assd_pixel=0.0)
    elif not n_pred or not n_truth:
        row['case_status'] = 'pred_empty' if not n_pred else 'target_empty'
        if empty_policy == 'strict':
            row['distance_status'] = 'infinite'
        else:
            row.update(distance_status='finite', distance_penalized=True,
                       hd95_pixel=row['diagonal_pixel'], assd_pixel=row['diagonal_pixel'])
    elif not n_pred_surface or not n_target_surface:
        row.update(case_status='both_nonempty', distance_status='undefined')
    else:
        # EDT of the inverse boundary gives the distance to the nearest actual
        # boundary pixel. Both directional samples have equal weight per pixel.
        pred_to_truth = ndimage.distance_transform_edt(~target_surface)[pred_surface]
        truth_to_pred = ndimage.distance_transform_edt(~pred_surface)[target_surface]
        distances = np.concatenate((pred_to_truth, truth_to_pred))
        row.update(case_status='both_nonempty', distance_status='finite',
                   hd95_pixel=float(np.percentile(distances, 95)),
                   assd_pixel=float(distances.mean()))
    return row


def _finite_number(value: Any, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be finite and in its valid range')
    return value


def _validate_row(row: Mapping[str, Any]) -> None:
    """Reject malformed or silently non-JSON metric rows before averaging."""
    status = row.get('distance_status')
    case = row.get('case_status')
    if status not in _DISTANCE_STATUSES:
        raise ValueError(f'unknown distance_status: {status!r}')
    if case not in _CASE_STATUSES:
        raise ValueError(f'unknown case_status: {case!r}')
    if row.get('empty_policy') not in ('strict', 'diagonal'):
        raise ValueError('unknown empty_policy in metric row')
    if (status == 'excluded') != (case == 'excluded'):
        raise ValueError('excluded case_status and distance_status must agree')
    if status == 'excluded':
        if any(row.get(key) is not None for key in ('dsc_percent', 'hd95_pixel', 'assd_pixel', 'is_positive')):
            raise ValueError('excluded rows must have null metrics and positivity')
        return
    _finite_number(row.get('dsc_percent'), 'dsc_percent', maximum=100.0)
    if type(row.get('is_positive')) is not bool:
        raise ValueError('included rows must have boolean is_positive')
    if row['is_positive'] != (case in ('both_nonempty', 'pred_empty')):
        raise ValueError('is_positive must agree with case_status')
    if status == 'finite':
        _finite_number(row.get('hd95_pixel'), 'hd95_pixel')
        _finite_number(row.get('assd_pixel'), 'assd_pixel')
    elif row.get('hd95_pixel') is not None or row.get('assd_pixel') is not None:
        raise ValueError('nonfinite distance statuses must have null numeric distances')
    if status == 'infinite' and (case not in ('pred_empty', 'target_empty') or row.get('empty_policy') != 'strict'):
        raise ValueError('infinite distance requires a strict one-empty case')
    if status == 'undefined' and case != 'both_nonempty':
        raise ValueError('undefined distance requires two nonempty masks')
    penalized = row.get('distance_penalized')
    if type(penalized) is not bool:
        raise ValueError('distance_penalized must be boolean')
    if penalized != (status == 'finite' and case in ('pred_empty', 'target_empty')):
        raise ValueError('distance_penalized must identify exactly the finite one-empty cases')
    if penalized and row.get('empty_policy') != 'diagonal':
        raise ValueError('finite one-empty distance requires diagonal policy')
    if case == 'both_empty' and (row['dsc_percent'] != 100.0 or row['hd95_pixel'] != 0.0 or row['assd_pixel'] != 0.0):
        raise ValueError('both-empty case must have DSC 100 and distances zero')
    if case in ('pred_empty', 'target_empty') and row['dsc_percent'] != 0.0:
        raise ValueError('one-empty case must have DSC zero')
    if penalized:
        diagonal = _finite_number(row.get('diagonal_pixel'), 'diagonal_pixel')
        if row['hd95_pixel'] != diagonal or row['assd_pixel'] != diagonal:
            raise ValueError('penalized distances must equal the declared image diagonal')


def aggregate_rows(rows: Iterable[Mapping[str, Any]], *, positive_only: bool = False) -> dict[str, Any]:
    """Return an image-macro mean with explicit eligibility and failure counts.

    Every valid image has equal weight. ``positive_only`` selects images with
    at least one valid target foreground pixel. Excluded/all-ignore images
    never enter a mean. Infinity propagates to the main distance status, then
    undefined propagates if there are no infinities. Only when all included
    distances are finite are the main HD95/ASSD means numeric. A separately
    labelled ``finite_subset_mean`` is conditional and includes its denominator;
    it must not be reported as the all-case result when failures exist.
    """
    materialized = list(rows)
    for row in materialized:
        _validate_row(row)
    eligible = [row for row in materialized if row['case_status'] != 'excluded']
    selected = [row for row in eligible if not positive_only or row['is_positive']]
    counts = Counter(row['distance_status'] for row in selected)
    finite = [row for row in selected if row['distance_status'] == 'finite']
    finite_means = {
        'n': len(finite),
        'hd95_pixel': math.fsum(float(row['hd95_pixel']) for row in finite) / len(finite) if finite else None,
        'assd_pixel': math.fsum(float(row['assd_pixel']) for row in finite) / len(finite) if finite else None,
    }
    status = ('infinite' if counts['infinite'] else 'undefined' if counts['undefined']
              else 'finite' if selected else 'excluded')
    return {
        'aggregation': 'image_macro_mean',
        'subset': 'positive' if positive_only else 'all',
        'n_rows': len(materialized),
        'n_excluded': len(materialized) - len(eligible),
        'n_positive': sum(row['is_positive'] for row in eligible),
        'n_negative': sum(not row['is_positive'] for row in eligible),
        'n_filtered_negative': len(eligible) - len(selected),
        'n_included': len(selected),
        'n_finite': counts['finite'],
        'n_infinite': counts['infinite'],
        'n_undefined': counts['undefined'],
        'n_penalized': sum(row['distance_penalized'] for row in selected),
        'case_counts': dict(Counter(row['case_status'] for row in selected)),
        'empty_policies': sorted({row['empty_policy'] for row in selected}),
        'dsc_percent': math.fsum(float(row['dsc_percent']) for row in selected) / len(selected) if selected else None,
        'hd95_pixel': finite_means['hd95_pixel'] if status == 'finite' else None,
        'assd_pixel': finite_means['assd_pixel'] if status == 'finite' else None,
        'distance_status': status,
        'finite_subset_mean': finite_means,
    }
