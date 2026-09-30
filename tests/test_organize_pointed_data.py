from pathlib import Path

import numpy as np
from PIL import Image

from scripts.organize_pointed_data import (
    combine_binary_masks,
    materialize_aligned_mask,
    materialize_png,
)


def test_materialize_png_converts_tiff_and_keeps_pixels(tmp_path: Path):
    pixels = np.arange(20, dtype=np.uint8).reshape(4, 5)
    source = tmp_path / "source.tif"
    destination = tmp_path / "processed_png" / "images" / "sample.png"
    Image.fromarray(pixels).save(source)

    materialize_png(source, destination)

    with Image.open(destination) as converted:
        np.testing.assert_array_equal(np.asarray(converted), pixels)


def test_combine_binary_masks_unions_multiple_lesions(tmp_path: Path):
    first = np.zeros((6, 7), dtype=np.uint8)
    second = np.zeros((6, 7), dtype=np.uint8)
    first[1:3, 1:3] = 255
    second[4:6, 4:7] = 255
    first_path, second_path = tmp_path / "mask.png", tmp_path / "mask_1.png"
    Image.fromarray(first).save(first_path)
    Image.fromarray(second).save(second_path)
    output = tmp_path / "processed_png" / "masks" / "sample.png"

    combine_binary_masks([first_path, second_path], output, overwrite=False)

    with Image.open(output) as combined:
        actual = np.asarray(combined)
    assert set(np.unique(actual)) == {0, 255}
    assert np.count_nonzero(actual) == 10


def test_materialize_aligned_mask_rotates_swapped_axes_clockwise(tmp_path: Path):
    pixels = np.arange(6, dtype=np.uint8).reshape(3, 2)
    source = tmp_path / "mask.jpg"
    destination = tmp_path / "mask.png"
    Image.fromarray(pixels).save(source, quality=100, subsampling=0)

    materialize_aligned_mask(source, destination, target_size=(3, 2))

    with Image.open(source) as decoded_source, Image.open(destination) as corrected:
        expected = np.asarray(decoded_source.transpose(Image.Transpose.ROTATE_270))
        assert corrected.size == (3, 2)
        np.testing.assert_array_equal(np.asarray(corrected), expected)
