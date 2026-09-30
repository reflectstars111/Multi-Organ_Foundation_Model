import cv2
import numpy as np
from scripts.fill_hc18_masks import fill_contour


def test_closed_ellipse():
    x = np.zeros((80, 100), np.uint8)
    cv2.ellipse(x, (50, 40), (30, 20), 0, 0, 360, 255, 1)
    y, method = fill_contour(x, True)
    assert method == 'fill_holes'
    assert y[40, 50] == 255 and y[0, 0] == 0
    assert (y[x > 0] == 255).all()


def test_clipped_ellipse():
    x = np.zeros((80, 100), np.uint8)
    cv2.ellipse(x, (50, 65), (30, 30), 0, 0, 360, 255, 1)
    y, method = fill_contour(x, True)
    assert method == 'ellipse_fit_clipped_to_image'
    assert y[65, 50] == 255 and y[0, 0] == 0
    assert (y[x > 0] == 255).all()
