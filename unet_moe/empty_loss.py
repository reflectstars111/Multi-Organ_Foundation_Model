"""Optional hard-negative penalty for empty annotated segmentation channels."""
import math

import torch
from torch.nn import functional as F


def empty_top_fraction_loss(logits, target, fraction=.01, channel_indices=None):
    """Average top-fraction negative BCE over eligible sample/channel pairs.

    Eligible means at least one valid target pixel and no positive *valid* pixel.
    An ignored area does not assert absence there. Fully missing channels, all
    ignored images and positive channels never receive this auxiliary penalty.
    ``channel_indices`` can restrict use to abdominal outputs; absent means all
    channels. This function is unweighted; training must use an explicit weight
    (recommended experimental default: disabled/zero).
    """
    if logits.ndim != 4 or logits.shape != target.shape:
        raise ValueError('Expected matching B,C,H,W logits and target')
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    if channel_indices is not None:
        indices = list(channel_indices)
        if len(set(indices)) != len(indices) or any(not isinstance(i, int) or not 0 <= i < logits.shape[1] for i in indices):
            raise ValueError('channel_indices must be distinct valid integer channel indices')
        logits, target = logits[:, indices], target[:, indices]
    valid = target >= 0
    eligible = valid.any((2, 3)) & ~(target > 0).any((2, 3))
    terms = []
    for batch, channel in eligible.nonzero().tolist():
        values = logits[batch, channel][valid[batch, channel]]
        count = max(1, math.ceil(values.numel() * fraction))
        # Softplus is BCEWithLogits(logit, 0); selecting logits is equivalent
        # to selecting the largest BCE values and avoids a full softplus map.
        terms.append(F.softplus(values.topk(count).values).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0
