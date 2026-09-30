"""Exact RGB mapping; dark gray has no supplied semantics and is ignored."""
import numpy as np

COLORS = [(0, 0, 0), (100, 0, 100), (255, 255, 0), (0, 0, 255),
          (255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 255, 255), (255, 0, 255)]
NAMES = ['background', 'liver', 'kidney', 'pancreas', 'vessels', 'adrenal', 'gallbladder', 'bone', 'spleen']


def rgb_to_ids(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB mask')
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    known = np.all(rgb == (10, 10, 10), axis=-1)
    for index, color in enumerate(COLORS):
        match = np.all(rgb == color, axis=-1)
        result[match] = index
        known |= match
    if not known.all():
        raise ValueError(f'Unknown RGB colors: {np.unique(rgb[~known], axis=0).tolist()}')
    return result
