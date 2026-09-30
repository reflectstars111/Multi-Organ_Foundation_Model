import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from scripts.preprocess_pointed_data import (
    CAMUS_PALETTE,
    camus_mask_to_palette,
    convert_aul,
    convert_camus,
    convert_fallmud,
    polygon_to_mask,
)


def test_camus_palette_is_visible_but_preserves_class_indices():
    indices = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    mask = camus_mask_to_palette(indices)

    assert mask.mode == "P"
    np.testing.assert_array_equal(np.asarray(mask), indices)
    palette = mask.getpalette()
    assert [tuple(palette[i * 3 : i * 3 + 3]) for i in range(4)] == list(CAMUS_PALETTE)


def test_polygon_to_mask_is_binary_and_uses_image_size():
    mask = polygon_to_mask((8, 6), [[1, 1], [6, 1], [6, 4], [1, 4]])

    assert mask.mode == "L"
    assert mask.size == (8, 6)
    assert set(np.unique(mask)) == {0, 255}
    assert mask.getpixel((3, 2)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_convert_aul_pairs_json_with_jpeg(tmp_path: Path):
    source_root = tmp_path / "pointed_data"
    case_root = source_root / "liver (AUL)" / "Benign" / "Benign"
    image_dir = case_root / "image"
    annotation_dir = case_root / "segmentation" / "liver"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    Image.new("L", (10, 7), color=20).save(image_dir / "1.jpg")
    (annotation_dir / "1.json").write_text(
        json.dumps([[2, 1], [8, 1], [8, 5], [2, 5]]), encoding="utf-8"
    )

    records, errors = convert_aul(source_root, tmp_path / "out", overwrite=False)

    assert not errors
    assert len(records) == 1
    output = tmp_path / "out" / "AUL" / "Benign" / "masks" / "liver" / "1.png"
    assert output.exists()
    with Image.open(output) as mask:
        assert mask.size == (10, 7)
        assert set(np.unique(mask)) == {0, 255}


def test_convert_camus_transposes_xy_axes_and_preserves_mask_labels(tmp_path: Path):
    source_root = tmp_path / "pointed_data"
    patient = source_root / "heart (CAMUS)" / "database_nifti" / "patient0001"
    patient.mkdir(parents=True)
    image = np.arange(12, dtype=np.uint8).reshape(4, 3)
    mask = np.zeros((4, 3), dtype=np.uint8)
    mask[1:3, 1:] = 3
    nib.save(nib.Nifti1Image(image, np.eye(4)), patient / "patient0001_2CH_ED.nii.gz")
    nib.save(
        nib.Nifti1Image(mask, np.eye(4)),
        patient / "patient0001_2CH_ED_gt.nii.gz",
    )

    records, errors = convert_camus(source_root, tmp_path / "out", overwrite=False)

    assert not errors
    assert len(records) == 2
    image_out = tmp_path / "out" / "CAMUS" / "patient0001" / "images" / "patient0001_2CH_ED.png"
    mask_out = tmp_path / "out" / "CAMUS" / "patient0001" / "masks" / "patient0001_2CH_ED.png"
    with Image.open(image_out) as converted_image:
        assert converted_image.size == (4, 3)
        np.testing.assert_array_equal(np.asarray(converted_image), image.T)
    with Image.open(mask_out) as converted_mask:
        assert converted_mask.size == (4, 3)
        assert converted_mask.mode == "P"
        assert set(np.unique(converted_mask)) == {0, 3}


def test_convert_fallmud_converts_tiff_without_changing_pixels(tmp_path: Path):
    source_root = tmp_path / "pointed_data"
    source = source_root / "muscle (FALLMUD)" / "FALLMUD" / "NeilCronin" / "images"
    source.mkdir(parents=True)
    pixels = np.arange(20, dtype=np.uint8).reshape(4, 5)
    Image.fromarray(pixels).save(source / "img_00001.tif")

    records, errors = convert_fallmud(source_root, tmp_path / "out", overwrite=False)

    assert not errors
    assert len(records) == 1
    output = tmp_path / "out" / "FALLMUD" / "NeilCronin" / "images" / "img_00001.png"
    with Image.open(output) as converted:
        np.testing.assert_array_equal(np.asarray(converted), pixels)
