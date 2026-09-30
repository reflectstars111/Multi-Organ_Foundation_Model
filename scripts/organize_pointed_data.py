#!/usr/bin/env python3
"""Build a uniform per-dataset PNG layout below every pointed_data dataset.

Each dataset receives ``processed_png/images`` and ``processed_png/masks``.
Original files remain untouched. Existing PNG files are hard-linked when possible;
JPEG/TIFF files and coordinate annotations are converted to PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from PIL import Image

if __package__:
    from .preprocess_pointed_data import _camus_uint8, camus_mask_to_palette, polygon_to_mask
else:
    from preprocess_pointed_data import _camus_uint8, camus_mask_to_palette, polygon_to_mask


Record = dict[str, Any]
MANIFEST_FIELDS = (
    "dataset",
    "sample_id",
    "split",
    "modality",
    "label",
    "image",
    "mask",
    "pairing_status",
    "source_image",
    "source_mask",
)


def find_dataset(root: Path, marker: str) -> Path:
    matches = [p for p in root.iterdir() if p.is_dir() and marker in p.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one dataset matching {marker!r}, found {len(matches)}")
    return matches[0]


def rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def materialize_png(source: Path, destination: Path, overwrite: bool = False) -> str:
    """Put an image at destination as PNG without modifying source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return "existing"
    if destination.exists():
        destination.unlink()
    if source.suffix.lower() == ".png":
        try:
            os.link(source, destination)
            return "linked"
        except OSError:
            shutil.copy2(source, destination)
            return "copied"
    with Image.open(source) as image:
        image.load()
        image.save(destination, format="PNG")
    return "converted"


def save_image(image: Image.Image, destination: Path, overwrite: bool = False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return "existing"
    image.save(destination, format="PNG")
    return "generated"


def materialize_aligned_mask(
    source: Path,
    destination: Path,
    target_size: tuple[int, int],
    overwrite: bool = False,
) -> str:
    """Convert a mask to PNG and correct FALLMUD's known swapped-axis masks."""
    if destination.exists() and not overwrite:
        with Image.open(destination) as existing:
            if existing.size == target_size:
                return "existing"
    with Image.open(source) as mask:
        mask.load()
        if mask.size == target_size:
            return materialize_png(source, destination, overwrite=True)
        if mask.size == (target_size[1], target_size[0]):
            # RyanCunningham aponeurosis masks were published 90 degrees CCW
            # relative to their paired images. Rotate clockwise without resampling.
            corrected = mask.transpose(Image.Transpose.ROTATE_270)
            return save_image(corrected, destination, overwrite=True)
        raise ValueError(f"Mask size {mask.size} cannot be aligned to image size {target_size}")


def row(
    dataset: str,
    sample_id: str,
    output_root: Path,
    source_root: Path,
    image: Path | None,
    mask: Path | None,
    source_image: Path | None,
    source_mask: Path | None,
    *,
    split: str = "",
    modality: str = "",
    label: str = "",
) -> Record:
    status = "paired" if image and mask else "image_only" if image else "mask_only"
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "split": split,
        "modality": modality,
        "label": label,
        "image": rel(image, output_root),
        "mask": rel(mask, output_root),
        "pairing_status": status,
        "source_image": rel(source_image, source_root),
        "source_mask": rel(source_mask, source_root),
    }


def organize_abdomen(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = source_root / "AbdomenUS"
    canonical = dataset / "abdominal_US" / "abdominal_US"
    out = dataset / "processed_png"
    records: list[Record] = []
    errors: list[str] = []
    for modality in ("AUS", "RUS"):
        for split in ("train", "test"):
            image_dir = canonical / modality / "images" / split
            mask_dir = canonical / modality / "annotations" / split
            images = {p.stem: p for p in image_dir.glob("*") if p.is_file()}
            masks = {p.stem: p for p in mask_dir.glob("*") if p.is_file()} if mask_dir.is_dir() else {}
            for key, source_image in sorted(images.items()):
                try:
                    image = out / "images" / modality / split / f"{key}.png"
                    materialize_png(source_image, image, overwrite)
                    source_mask = masks.get(key)
                    mask = None
                    if source_mask:
                        mask = out / "masks" / modality / split / f"{key}.png"
                        materialize_png(source_mask, mask, overwrite)
                    records.append(row("AbdomenUS", key, out, source_root, image, mask, source_image, source_mask, split=split, modality=modality))
                except Exception as exc:
                    errors.append(f"{source_image}: {type(exc).__name__}: {exc}")
            for key in sorted(set(masks) - set(images)):
                source_mask = masks[key]
                try:
                    mask = out / "masks" / modality / split / f"{key}.png"
                    materialize_png(source_mask, mask, overwrite)
                    records.append(row("AbdomenUS", key, out, source_root, None, mask, None, source_mask, split=split, modality=modality))
                except Exception as exc:
                    errors.append(f"{source_mask}: {type(exc).__name__}: {exc}")
    return dataset, records, errors


def organize_fallmud(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(FALLMUD)")
    content = dataset / "FALLMUD"
    out = dataset / "processed_png"
    records: list[Record] = []
    errors: list[str] = []
    for person in sorted(p for p in content.iterdir() if p.is_dir()):
        images = {p.stem: p for p in (person / "images").glob("*") if p.is_file()}
        mask_sets = {
            "aponeurosis": {p.stem: p for p in (person / "aponeurosis_masks").glob("*") if p.is_file()},
            "fascicle": {p.stem: p for p in (person / "fascicle_masks").glob("*") if p.is_file()},
        }
        converted_images: dict[str, Path] = {}
        for key, source_image in sorted(images.items()):
            try:
                image = out / "images" / person.name / f"{key}.png"
                materialize_png(source_image, image, overwrite)
                converted_images[key] = image
            except Exception as exc:
                errors.append(f"{source_image}: {type(exc).__name__}: {exc}")
        for label, masks in mask_sets.items():
            for key in sorted(set(images) | set(masks)):
                source_image = images.get(key)
                source_mask = masks.get(key)
                image = converted_images.get(key)
                mask = None
                try:
                    if source_mask:
                        mask = out / "masks" / label / person.name / f"{key}.png"
                        if source_image:
                            with Image.open(source_image) as original:
                                target_size = original.size
                            materialize_aligned_mask(source_mask, mask, target_size, overwrite)
                        else:
                            materialize_png(source_mask, mask, overwrite)
                    records.append(row("FALLMUD", key, out, source_root, image, mask, source_image, source_mask, modality=person.name, label=label))
                except Exception as exc:
                    errors.append(f"{source_mask}: {type(exc).__name__}: {exc}")
    return dataset, records, errors


def combine_binary_masks(sources: Iterable[Path], destination: Path, overwrite: bool) -> str:
    sources = list(sources)
    if not sources:
        raise ValueError("No masks to combine")
    arrays = []
    size = None
    for source in sources:
        with Image.open(source) as image:
            current = image.convert("L")
            if size is None:
                size = current.size
            elif current.size != size:
                raise ValueError(f"Mask size mismatch in {source}")
            arrays.append(np.asarray(current) > 0)
    combined = (np.logical_or.reduce(arrays).astype(np.uint8) * 255)
    return save_image(Image.fromarray(combined), destination, overwrite)


def organize_busi(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(BUSI)")
    content = dataset / "Dataset_BUSI_with_GT"
    out = dataset / "processed_png"
    records: list[Record] = []
    errors: list[str] = []
    for category in ("benign", "malignant", "normal"):
        files = list((content / category).glob("*.png"))
        images = {p.stem: p for p in files if "_mask" not in p.stem}
        masks: dict[str, list[Path]] = defaultdict(list)
        for source_mask in files:
            if "_mask" in source_mask.stem:
                masks[source_mask.stem.split("_mask", 1)[0]].append(source_mask)
        for key, source_image in sorted(images.items()):
            try:
                image = out / "images" / category / f"{key}.png"
                mask = out / "masks" / category / f"{key}.png"
                materialize_png(source_image, image, overwrite)
                combine_binary_masks(sorted(masks[key]), mask, overwrite)
                records.append(row("BUSI", key, out, source_root, image, mask, source_image, masks[key][0], label=category))
            except Exception as exc:
                errors.append(f"{source_image}: {type(exc).__name__}: {exc}")
    return dataset, records, errors


def organize_simple_pairs(
    dataset: Path,
    source_root: Path,
    groups: Iterable[tuple[str, str, Path, Path]],
    overwrite: bool,
    dataset_name: str,
) -> tuple[Path, list[Record], list[str]]:
    out = dataset / "processed_png"
    records: list[Record] = []
    errors: list[str] = []
    for modality, split, image_dir, mask_dir in groups:
        images = {p.stem: p for p in image_dir.glob("*") if p.is_file()}
        masks = {p.stem: p for p in mask_dir.glob("*") if p.is_file()}
        relative = Path(modality) / split if split else Path(modality)
        for key in sorted(set(images) | set(masks)):
            source_image, source_mask = images.get(key), masks.get(key)
            image = mask = None
            try:
                if source_image:
                    image = out / "images" / relative / f"{key}.png"
                    materialize_png(source_image, image, overwrite)
                if source_mask:
                    mask = out / "masks" / relative / f"{key}.png"
                    materialize_png(source_mask, mask, overwrite)
                records.append(row(dataset_name, key, out, source_root, image, mask, source_image, source_mask, split=split, modality=modality))
            except Exception as exc:
                errors.append(f"{source_image or source_mask}: {type(exc).__name__}: {exc}")
    return dataset, records, errors


def organize_mmotu(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(MMOTU)")
    base = dataset / "dataset"
    groups = (
        ("OTU_2D", "train", base / "OTU_2D/train/train_image", base / "OTU_2D/train/train_label/label"),
        ("OTU_2D", "test", base / "OTU_2D/test/image", base / "OTU_2D/test/label/black_write"),
        ("OTU_CEUS", "", base / "OTU_CEUS/image", base / "OTU_CEUS/label"),
    )
    return organize_simple_pairs(dataset, source_root, groups, overwrite, "MMOTU")


def organize_ddti(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(DDTI)")
    stage = next(p for p in dataset.rglob("stage2") if p.is_dir())
    return organize_simple_pairs(dataset, source_root, (("thyroid", "", stage / "p_image", stage / "p_mask"),), overwrite, "DDTI")


def organize_cca(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(CCA)")
    base = dataset / "Common Carotid Artery Ultrasound Images"
    return organize_simple_pairs(dataset, source_root, (("CCA", "", base / "US images", base / "Expert mask images"),), overwrite, "CCA")


def organize_fetal(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(Fetal HC")
    out = dataset / "processed_png"
    records: list[Record] = []
    errors: list[str] = []
    training = list((dataset / "training_set").glob("*.png"))
    images = {p.stem: p for p in training if not p.stem.endswith("_Annotation")}
    masks = {p.stem.removesuffix("_Annotation"): p for p in training if p.stem.endswith("_Annotation")}
    for split, split_images, split_masks in (
        ("train", images, masks),
        ("test", {p.stem: p for p in (dataset / "test_set").glob("*.png")}, {}),
    ):
        for key, source_image in sorted(split_images.items()):
            source_mask = split_masks.get(key)
            try:
                image = out / "images" / split / f"{key}.png"
                materialize_png(source_image, image, overwrite)
                mask = None
                if source_mask:
                    mask = out / "masks" / split / f"{key}.png"
                    materialize_png(source_mask, mask, overwrite)
                records.append(row("Fetal_HC", key, out, source_root, image, mask, source_image, source_mask, split=split))
            except Exception as exc:
                errors.append(f"{source_image}: {type(exc).__name__}: {exc}")
    return dataset, records, errors


def organize_aul(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(AUL)")
    out = dataset / "processed_png"
    records: list[Record] = []
    errors: list[str] = []
    converted_images: dict[Path, Path] = {}
    for annotation in sorted(dataset.rglob("*.json"), key=lambda p: str(p).casefold()):
        structure = annotation.parent.name
        case_root = annotation.parents[2]
        category = case_root.name
        source_image = case_root / "image" / f"{annotation.stem}.jpg"
        image = out / "images" / category / f"{annotation.stem}.png"
        mask = out / "masks" / structure / category / f"{annotation.stem}.png"
        try:
            if source_image not in converted_images:
                materialize_png(source_image, image, overwrite)
                converted_images[source_image] = image
            with Image.open(source_image) as original:
                size = original.size
            points = json.loads(annotation.read_text(encoding="utf-8-sig"))
            save_image(polygon_to_mask(size, points), mask, overwrite)
            records.append(row("AUL", annotation.stem, out, source_root, image, mask, source_image, annotation, label=structure, modality=category))
        except Exception as exc:
            errors.append(f"{annotation}: {type(exc).__name__}: {exc}")
    return dataset, records, errors


def organize_camus(source_root: Path, overwrite: bool) -> tuple[Path, list[Record], list[str]]:
    dataset = find_dataset(source_root, "(CAMUS)")
    out = dataset / "processed_png"
    records_by_key: dict[str, Record] = {}
    errors: list[str] = []
    central = source_root / "_preprocessed_png" / "CAMUS"
    if central.is_dir():
        patient_sources = {
            path.name: path
            for path in dataset.rglob("patient*")
            if path.is_dir() and any(path.glob("*.nii.gz"))
        }
        for patient_dir in sorted(p for p in central.iterdir() if p.is_dir()):
            image_sources = {p.name: p for p in (patient_dir / "images").glob("*.png")}
            mask_sources = {p.name: p for p in (patient_dir / "masks").glob("*.png")}
            for filename in sorted(set(image_sources) | set(mask_sources)):
                source_image_png = image_sources.get(filename)
                source_mask_png = mask_sources.get(filename)
                image = mask = source_image = source_mask = None
                try:
                    base = Path(filename).stem.rsplit("_frame_", 1)[0]
                    nifti_dir = patient_sources.get(patient_dir.name)
                    if nifti_dir:
                        candidate = nifti_dir / f"{base}.nii.gz"
                        source_image = candidate if candidate.is_file() else None
                        candidate = nifti_dir / f"{base}_gt.nii.gz"
                        source_mask = candidate if candidate.is_file() else None
                    if source_image_png:
                        image = out / "images" / patient_dir.name / filename
                        materialize_png(source_image_png, image, overwrite)
                    if source_mask_png:
                        mask = out / "masks" / patient_dir.name / filename
                        materialize_png(source_mask_png, mask, overwrite)
                    records_by_key[f"{patient_dir.name}/{filename}"] = row(
                        "CAMUS",
                        Path(filename).stem,
                        out,
                        source_root,
                        image,
                        mask,
                        source_image,
                        source_mask,
                        modality=patient_dir.name,
                    )
                except Exception as exc:
                    errors.append(f"{source_image_png or source_mask_png}: {type(exc).__name__}: {exc}")
        return dataset, [records_by_key[k] for k in sorted(records_by_key)], errors

    for source in sorted(dataset.rglob("*.nii.gz"), key=lambda p: str(p).casefold()):
        try:
            stem = source.name[:-7]
            is_mask = stem.endswith("_gt")
            output_stem = stem[:-3] if is_mask else stem
            patient = source.parent.name
            data = np.squeeze(np.asanyarray(nib.load(source).dataobj))
            if data.ndim not in (2, 3):
                raise ValueError(f"Unsupported NIfTI shape {data.shape}")
            converted = _camus_uint8(data, is_mask)
            frame_count = 1 if converted.ndim == 2 else converted.shape[2]
            for index in range(frame_count):
                suffix = "" if converted.ndim == 2 else f"_frame_{index:03d}"
                sample_id = f"{output_stem}{suffix}"
                frame = converted if converted.ndim == 2 else converted[:, :, index]
                destination = out / ("masks" if is_mask else "images") / patient / f"{sample_id}.png"
                png = camus_mask_to_palette(frame.T) if is_mask else Image.fromarray(frame.T)
                save_image(png, destination, overwrite)
                key = f"{patient}/{sample_id}"
                record = records_by_key.setdefault(
                    key,
                    row("CAMUS", sample_id, out, source_root, None, None, None, None, modality=patient),
                )
                if is_mask:
                    record["mask"] = rel(destination, out)
                    record["source_mask"] = rel(source, source_root)
                else:
                    record["image"] = rel(destination, out)
                    record["source_image"] = rel(source, source_root)
                record["pairing_status"] = "paired" if record["image"] and record["mask"] else "image_only" if record["image"] else "mask_only"
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    return dataset, [records_by_key[k] for k in sorted(records_by_key)], errors


def write_dataset_report(dataset: Path, records: list[Record], errors: list[str]) -> dict[str, Any]:
    out = dataset / "processed_png"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    counts = Counter(record["pairing_status"] for record in records)
    png_count = sum(1 for _ in out.rglob("*.png"))
    report = {
        "dataset_directory": dataset.name,
        "layout": "processed_png/images and processed_png/masks",
        "manifest_rows": len(records),
        "png_files": png_count,
        "pairing_status": dict(sorted(counts.items())),
        "errors": errors,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def default_source_root() -> Path:
    workspace = Path(__file__).resolve().parents[1]
    datasets = [p for p in workspace.iterdir() if p.is_dir() and "(Datasets)" in p.name]
    if len(datasets) != 1:
        raise FileNotFoundError("Could not uniquely locate 数据集 (Datasets)")
    return datasets[0] / "pointed_data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("abdomen", "fallmud", "busi", "mmotu", "camus", "ddti", "aul", "fetal", "cca"),
        default=("abdomen", "fallmud", "busi", "mmotu", "camus", "ddti", "aul", "fetal", "cca"),
    )
    args = parser.parse_args(argv)
    source_root = (args.source_root or default_source_root()).resolve()
    available = {
        "abdomen": organize_abdomen,
        "fallmud": organize_fallmud,
        "busi": organize_busi,
        "mmotu": organize_mmotu,
        "camus": organize_camus,
        "ddti": organize_ddti,
        "aul": organize_aul,
        "fetal": organize_fetal,
        "cca": organize_cca,
    }
    reports: dict[str, Any] = {}
    total_errors = 0
    for dataset_key in args.datasets:
        organizer = available[dataset_key]
        print(f"Running {organizer.__name__} ...", flush=True)
        dataset, records, errors = organizer(source_root, args.overwrite)
        report = write_dataset_report(dataset, records, errors)
        reports[dataset.name] = report
        total_errors += len(errors)
        print(f"  png={report['png_files']}, rows={len(records)}, errors={len(errors)}", flush=True)
    all_reports: dict[str, Any] = {}
    for dataset_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        report_path = dataset_dir / "processed_png" / "report.json"
        if report_path.is_file():
            all_reports[dataset_dir.name] = json.loads(report_path.read_text(encoding="utf-8"))
    (source_root / "processed_layout_report.json").write_text(
        json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
