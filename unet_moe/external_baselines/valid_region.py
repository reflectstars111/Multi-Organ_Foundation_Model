"""Largest fully observed rectangle extraction without fabricating negative labels."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def valid_roi(target2d, *, min_side: int = 32, minimum_foreground_fraction: float = 0.25):
    """Return an exclusive ``(top,left,bottom,right)`` ROI, or None.

    Fully observed images retain their entire field of view, including true
    negative images. Partial images require both known foreground/background.
    A monotonic histogram stack enumerates maximal known rectangles in O(HW);
    a positive-pixel integral image checks the class constraint in O(1). The
    maximum area wins, then the most retained foreground, then top/left order.
    Minimum side and retained foreground fraction constraints are applied
    during the search, not as post-hoc rejection of a background-heavy ROI.
    """
    array = target2d.detach().cpu().numpy() if isinstance(target2d, torch.Tensor) else np.asarray(target2d)
    if array.ndim != 2 or not ((array == -1) | (array == 0) | (array == 1)).all():
        raise ValueError("Expected a 2D target containing only -1, 0, 1")
    if not isinstance(min_side, int) or min_side < 1:
        raise ValueError("min_side must be a positive integer")
    if not 0 <= minimum_foreground_fraction <= 1:
        raise ValueError("minimum_foreground_fraction must lie in [0,1]")
    height, width = array.shape
    if min(height, width) < min_side:
        return None
    known = array >= 0
    if known.all():
        return (0, 0, height, width)
    positive = array == 1
    if not positive.any() or not (array == 0).any():
        return None
    minimum_foreground = int(np.ceil(positive.sum() * minimum_foreground_fraction))
    prefix = np.zeros((height + 1, width + 1), dtype=np.int64)
    prefix[1:, 1:] = positive.cumsum(0).cumsum(1)
    heights = np.zeros(width, dtype=np.int32)
    best_key, best = (0, 0, 0, 0, 0, 0), None
    for bottom in range(1, height + 1):
        heights = (heights + 1) * known[bottom - 1]
        stack = []
        for right, current in enumerate(heights.tolist() + [0]):
            start = right
            while stack and stack[-1][1] >= current:
                left, tall = stack.pop()
                start = left
                wide = right - left
                area = tall * wide
                if tall < min_side or wide < min_side or area < best_key[0]:
                    continue
                top = bottom - tall
                count = int(prefix[bottom, right] - prefix[top, right]
                            - prefix[bottom, left] + prefix[top, left])
                if count == 0 or count == area or count < minimum_foreground:
                    continue
                key = (area, count, -top, -left, -bottom, -right)
                if key > best_key:
                    best_key, best = key, (top, left, bottom, right)
            if current:
                stack.append((start, current))
    return best


def crop_resize(image, target, roi, size=512):
    """Crop paired CHW image/1HW (or HW) label, then resize without label mixing."""
    if image.ndim != 3 or target.ndim not in (2, 3) or image.shape[-2:] != target.shape[-2:]:
        raise ValueError("Expected paired CHW image and HW or 1HW target")
    if target.ndim == 3 and target.shape[0] != 1:
        raise ValueError("Only binary single-channel targets are supported")
    if roi is None or len(roi) != 4:
        raise ValueError("A nonempty ROI is required")
    top, left, bottom, right = roi
    height, width = image.shape[-2:]
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError("ROI lies outside the image")
    cropped_image = image[:, top:bottom, left:right]
    cropped_target = target[..., top:bottom, left:right]
    if not ((cropped_target == 0) | (cropped_target == 1)).all():
        raise ValueError("Unknown support/shape-prior labels must not become background")
    output_size = (size, size) if isinstance(size, int) else tuple(size)
    resized_image = F.interpolate(cropped_image.unsqueeze(0), size=output_size,
                                  mode="bilinear", align_corners=False).squeeze(0)
    labels4d = cropped_target.reshape(1, 1, bottom - top, right - left)
    resized_target = F.interpolate(labels4d.float(), size=output_size, mode="nearest").squeeze(0)
    if target.ndim == 2:
        resized_target = resized_target.squeeze(0)
    return resized_image, resized_target.to(target.dtype)
