"""Frozen Q01 losses and train-only statistics for domain-balanced sampling.

The exposure weight changes the coefficient of each structure without changing
the number of valid structures in an image's denominator.  The C arm changes
only positive-image BCE pixels; empty-image BCE and soft Dice are unchanged.
"""
from collections import Counter, defaultdict
import math

import numpy as np
import torch
from torch.nn import functional as F

from .data import TASKS, SegmentationDataset


ARMS = ('image_mean', 'organ_exposure', 'organ_exposure_fg_bg')
STATISTICS_VERSION = 'q01_train_exposure_v1'
FOREGROUND_WEIGHT_CAP = 4.


def _weight_vector(weights, key, logits):
    if weights is None or key not in weights:
        raise ValueError(f'Missing {key} in frozen Q01 weights')
    raw = np.asarray(weights[key], dtype=np.float64)
    if raw.shape != (logits.shape[1],) or not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError(f'{key} must contain one finite positive value per channel')
    return torch.as_tensor(raw, dtype=logits.dtype, device=logits.device)[None, :]


def loss_components(logits, target, arm='image_mean', weights=None):
    """Return differentiable channel terms and the scalar image-mean loss.

    ``per_channel_unweighted`` is the original BCE + Dice in every arm.
    ``per_channel_weighted`` is the actual arm's BCE + Dice, with the organ
    multiplier applied, but before multiplication by ``active / K / B``.
    ``per_channel_bce_weighted`` exposes C's pixel adjustment before the organ
    multiplier.  All tensors retain autograd for fixed-batch diagnostics.
    ``weights`` is the JSON-compatible output of :func:`collect_statistics`.
    """
    if arm not in ARMS:
        raise ValueError(f'Unknown Q01 arm: {arm}')
    if logits.shape != target.shape or target.ndim != 4:
        raise ValueError('Expected matching B,C,H,W logits and targets')
    valid = target >= 0
    truth = torch.where(valid, target, 0.)
    active = valid.any((2, 3))
    pixels = F.binary_cross_entropy_with_logits(logits, truth, reduction='none')
    valid_pixels = valid.sum((2, 3)).clamp_min(1)
    bce = (pixels * valid).sum((2, 3)) / valid_pixels
    probability = logits.sigmoid() * valid
    dice = 1 - (2 * (probability * truth).sum((2, 3)) + 1) / (
        probability.sum((2, 3)) + truth.sum((2, 3)) + 1)
    adjusted_bce = bce
    if arm == 'organ_exposure_fg_bg':
        ratio = _weight_vector(weights, 'foreground_weights', logits)
        normalizer = _weight_vector(weights, 'foreground_normalizers', logits)
        positive = truth.gt(0).any((2, 3))
        pixel_weight = (1 + (ratio[..., None, None] - 1) * truth) / normalizer[..., None, None]
        pixel_weight = torch.where(positive[..., None, None], pixel_weight, 1.)
        adjusted_bce = (pixels * valid * pixel_weight).sum((2, 3)) / valid_pixels
    unweighted = bce + dice
    weighted = adjusted_bce + dice
    if arm != 'image_mean':
        weighted = weighted * _weight_vector(weights, 'organ_weights', logits)
    per_image = (weighted * active).sum(1) / active.sum(1).clamp_min(1)
    return dict(per_channel_unweighted=unweighted, per_channel_weighted=weighted,
                per_channel_bce=bce, per_channel_bce_weighted=adjusted_bce,
                per_channel_dice=dice, active=active, per_image=per_image,
                loss=per_image.mean())


def weighted_loss(logits, target, arm='image_mean', weights=None):
    """Scalar Q01 objective, reduced by valid structures then by images."""
    return loss_components(logits, target, arm, weights)['loss']


def collect_statistics(dataset):
    """Compute frozen weights from resized training labels without augmentation.

    Accepts a ``RegionDataset``.  Existing ``SegmentationDataset`` decoding is
    reused one available channel at a time, preserving all ignore semantics.
    NumPy reductions avoid allocating or reducing dense 16-channel torch
    tensors on CPU.  The expected exposure is over uniform domain choice,
    followed by uniform image choice in that domain (the actual sampler).

    ``group`` is not promoted to a patient ID: only CAMUS explicitly supplies
    patient groups; AbdomenUS supplies CT source groups.  Other groups retain
    their source/image grouping interpretation from the input records.
    """
    if not dataset.groups:
        raise ValueError('Cannot collect weights from an empty train dataset')
    task_ids = list(dataset.task_ids)
    task_set = set(task_ids)
    domain_counts = Counter()
    source_counts = defaultdict(Counter)
    for rows in dataset.groups:
        if not rows or any(row.get('split') != 'train' for row in rows):
            raise ValueError('Q01 weights require exclusively train records')
        domains = {row['dataset'] for row in rows}
        images = {row['image'] for row in rows}
        groups = {row['group'] for row in rows}
        if len(domains) != 1 or len(images) != 1 or len(groups) != 1:
            raise ValueError('Each image must have one data domain and source group')
        if not {row['task'] for row in rows}.issubset(task_set):
            raise ValueError('Record task is outside the frozen channel order')
        domain = rows[0]['dataset']
        domain_counts[domain] += 1
        source_counts[domain][rows[0]['group']] += 1
    domain_probability = {name: 1 / len(domain_counts) for name in sorted(domain_counts)}
    totals = {task: dict(task_id=task, active_images=0, positive_images=0, empty_images=0,
                        valid_pixels=0, valid_positive_pixels=0, valid_empty_pixels=0,
                        foreground_pixels=0, background_positive_pixels=0, exposure=0.,
                        sampled_active_probability=0., sampled_positive_probability=0.,
                        sampled_empty_probability=0., per_domain={}, group_source_image_counts={})
              for task in task_ids}
    representatives = {task: dict(positive=None, empty=None) for task in task_ids}
    active_histogram = Counter()
    all_ignored_images = 0
    for image_index, rows in enumerate(dataset.groups):
        domain = rows[0]['dataset']
        group = rows[0]['group']
        image_probability = domain_probability[domain] / domain_counts[domain]
        # RegionDataset overwrites repeated same-task rows; retain that behavior.
        counts = {}
        for record, (_, target, _) in zip(rows, SegmentationDataset(rows, dataset.size)):
            array = target.numpy()
            counts[record['task']] = (int(np.count_nonzero(array >= 0)), int(np.count_nonzero(array > 0)))
        active = sum(valid > 0 for valid, _ in counts.values())
        active_histogram[active] += 1
        if not active:
            all_ignored_images += 1
            continue
        for task, (valid, foreground) in counts.items():
            if not valid:
                continue
            row = totals[task]
            positive = foreground > 0
            representative_kind = 'positive' if positive else 'empty'
            if representatives[task][representative_kind] is None:
                representatives[task][representative_kind] = image_index
            row['active_images'] += 1
            row['positive_images'] += int(positive)
            row['empty_images'] += int(not positive)
            row['valid_pixels'] += valid
            row['foreground_pixels'] += foreground
            row['valid_positive_pixels'] += valid if positive else 0
            row['valid_empty_pixels'] += 0 if positive else valid
            row['background_positive_pixels'] += valid - foreground if positive else 0
            row['exposure'] += image_probability / active
            row['sampled_active_probability'] += image_probability
            row['sampled_positive_probability'] += image_probability if positive else 0
            row['sampled_empty_probability'] += 0 if positive else image_probability
            row['group_source_image_counts'][group] = row['group_source_image_counts'].get(group, 0) + 1
            within_domain = row['per_domain'].setdefault(domain, dict(active_images=0, positive_images=0,
                                                                      empty_images=0, exposure=0.))
            within_domain['active_images'] += 1
            within_domain['positive_images'] += int(positive)
            within_domain['empty_images'] += int(not positive)
            within_domain['exposure'] += image_probability / active
    organ_weights, foreground_weights, normalizers, proportions = [], [], [], []
    for task in task_ids:
        row = totals[task]
        if row['exposure'] <= 0:
            raise ValueError(f'No valid training exposure for {TASKS[task]}')
        organ_weight = 1 / (len(task_ids) * row['exposure'])
        q = (row['foreground_pixels'] + 1) / (row['valid_positive_pixels'] + 2)
        ratio = min(FOREGROUND_WEIGHT_CAP, max(1., math.sqrt((1 - q) / q)))
        normalizer = 1 + (ratio - 1) * q
        row.update(organ_weight=organ_weight, foreground_weight=ratio, foreground_normalizer=normalizer,
                   foreground_fraction_positive_smoothed=q,
                   foreground_fraction_all=row['foreground_pixels'] / row['valid_pixels'],
                   foreground_fraction_positive=(row['foreground_pixels'] / row['valid_positive_pixels']
                                                 if row['valid_positive_pixels'] else None),
                   expected_weighted_coefficient=row['exposure'] * organ_weight,
                   group_source_count=len(row['group_source_image_counts']))
        organ_weights.append(organ_weight)
        foreground_weights.append(ratio)
        normalizers.append(normalizer)
        proportions.append(q)
    semantics = {domain: ('patient ID from official CAMUS patient split lists' if domain == 'CAMUS' else
                          'CT source; not a verified patient identifier' if domain == 'AbdomenUS' else
                          'source/image group; not a verified patient identifier')
                 for domain in domain_counts}
    return dict(schema=STATISTICS_VERSION, split='train', task_ids=task_ids, image_size=dataset.size,
                images=len(dataset), domain_image_counts=dict(sorted(domain_counts.items())),
                domain_sampling_probabilities=domain_probability,
                active_channel_histogram={str(key): value for key, value in sorted(active_histogram.items())},
                all_ignored_images=all_ignored_images,
                group_semantics=semantics,
                group_source_image_counts={domain: dict(sorted(counts.items()))
                                          for domain, counts in sorted(source_counts.items())},
                group_source_counts={domain: len(counts) for domain, counts in sorted(source_counts.items())},
                per_structure={TASKS[task]: totals[task] for task in task_ids},
                representative_indices={TASKS[task]: representatives[task] for task in task_ids},
                diagnostic_indices=list(dict.fromkeys(index for task in task_ids
                                                      for index in representatives[task].values()
                                                      if index is not None)),
                organ_weights=organ_weights, foreground_weights=foreground_weights,
                foreground_normalizers=normalizers, foreground_proportions=proportions,
                formula=dict(exposure='s_c = sum_d P(d)/N_d * sum_i_in_d I_ic/K_i',
                             organ_weight='1/(C*s_c); keep the per-image K_i denominator',
                             foreground_fraction='q=(positive_foreground_pixels+1)/(positive_valid_pixels+2)',
                             foreground_ratio='clamp(sqrt((1-q)/q),1,4)',
                             normalization='normalizer=1+(ratio-1)*q; foreground=ratio/normalizer; background=1/normalizer',
                             scope='positive-image BCE pixels only; empty-image BCE and Dice unchanged',
                             foreground_ratio_cap=FOREGROUND_WEIGHT_CAP, smoothing_foreground=1, smoothing_valid=2))
