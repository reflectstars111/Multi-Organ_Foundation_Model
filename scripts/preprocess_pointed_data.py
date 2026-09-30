#!/usr/bin/env python3
"""Convert non-standard pointed_data images and coordinate annotations to PNG.

The conversion is intentionally non-destructive. Source files are never changed;
outputs are written below ``pointed_data/_preprocessed_png`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


Record = dict[str, Any]
CAMUS_PALETTE = (
    (0, 0, 0),       # class 0: black (background)
    (255, 0, 0),     # class 1: red
    (0, 255, 0),     # class 2: green
    (0, 0, 255),     # class 3: blue
)


def camus_mask_to_palette(array: np.ndarray) -> Image.Image:
    """Create a visibly colored indexed PNG while preserving class IDs 0-3."""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {array.shape}")
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected CAMUS class IDs: {sorted(values)}")
    mask = Image.fromarray(array.astype(np.uint8)).convert("P")
    flat_palette = [component for color in CAMUS_PALETTE for component in color]
    mask.putpalette(flat_palette + [0] * (768 - len(flat_palette)))
    return mask


def _find_dataset(source_root: Path, marker: str) -> Path:
    matches = [path for path in source_root.iterdir() if path.is_dir() and marker in path.name]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one directory containing {marker!r} below {source_root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _save(image: Image.Image, destination: Path, overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return "existing"
    image.save(destination, format="PNG")
    return "written"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def polygon_to_mask(size: tuple[int, int], points: Iterable[Iterable[float]]) -> Image.Image:
    """Rasterize one polygon as a binary 8-bit mask (background 0, foreground 255)."""
    width, height = size
    vertices = [tuple(float(value) for value in point) for point in points]
    if len(vertices) < 3 or any(len(point) != 2 for point in vertices):
        raise ValueError("A polygon must contain at least three [x, y] points")
    coordinates = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Polygon contains a non-finite coordinate")
    if (
        (coordinates[:, 0] < 0).any()
        or (coordinates[:, 0] >= width).any()
        or (coordinates[:, 1] < 0).any()
        or (coordinates[:, 1] >= height).any()
    ):
        raise ValueError(f"Polygon coordinate lies outside image bounds {width}x{height}")

    mask = Image.new("L", size, color=0)
    ImageDraw.Draw(mask).polygon(vertices, fill=255)
    return mask


def convert_aul(
    source_root: Path, output_root: Path, overwrite: bool = False
) -> tuple[list[Record], list[Record]]:
    """Convert every AUL JSON polygon to a same-sized binary PNG mask."""
    records: list[Record] = []
    errors: list[Record] = []
    try:
        dataset_root = _find_dataset(source_root, "(AUL)")
    except Exception as exc:
        return records, [{"dataset": "AUL", "error": str(exc)}]

    for annotation in sorted(dataset_root.rglob("*.json"), key=lambda path: str(path).casefold()):
        try:
            structure = annotation.parent.name
            case_root = annotation.parents[2]
            category = case_root.name
            source_image = case_root / "image" / f"{annotation.stem}.jpg"
            if not source_image.is_file():
                raise FileNotFoundError(f"Matching image not found: {source_image}")

            points = json.loads(annotation.read_text(encoding="utf-8-sig"))
            with Image.open(source_image) as image:
                size = image.size
            mask = polygon_to_mask(size, points)
            destination = output_root / "AUL" / category / "masks" / structure / f"{annotation.stem}.png"
            status = _save(mask, destination, overwrite)
            records.append(
                {
                    "dataset": "AUL",
                    "kind": "mask",
                    "label": structure,
                    "source": _relative(annotation, source_root),
                    "source_image": _relative(source_image, source_root),
                    "output": _relative(destination, output_root),
                    "width": size[0],
                    "height": size[1],
                    "frame_index": "",
                    "status": status,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "dataset": "AUL",
                    "source": _relative(annotation, source_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, errors


def _camus_uint8(data: np.ndarray, is_mask: bool) -> np.ndarray:
    data = np.asarray(data)
    if not np.isfinite(data).all():
        raise ValueError("NIfTI array contains NaN or infinity")
    if is_mask:
        rounded = np.rint(data)
        if not np.allclose(data, rounded) or rounded.min() < 0 or rounded.max() > 255:
            raise ValueError("Mask values must be integer class IDs in [0, 255]")
        return rounded.astype(np.uint8)

    minimum = float(data.min())
    maximum = float(data.max())
    if minimum >= 0 and maximum <= 255:
        return np.rint(data).astype(np.uint8)
    if maximum == minimum:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.rint((data - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)


def convert_camus(
    source_root: Path, output_root: Path, overwrite: bool = False
) -> tuple[list[Record], list[Record]]:
    """Expand CAMUS 2D/3D NIfTI files into viewable PNG images and masks."""
    records: list[Record] = []
    errors: list[Record] = []
    try:
        dataset_root = _find_dataset(source_root, "(CAMUS)")
    except Exception as exc:
        return records, [{"dataset": "CAMUS", "error": str(exc)}]

    sources = sorted(dataset_root.rglob("*.nii.gz"), key=lambda path: str(path).casefold())
    for source in sources:
        try:
            stem = source.name[: -len(".nii.gz")]
            is_mask = stem.endswith("_gt")
            output_stem = stem[:-3] if is_mask else stem
            patient = source.parent.name
            data = np.asanyarray(nib.load(source).dataobj)
            data = np.squeeze(data)
            if data.ndim not in (2, 3):
                raise ValueError(f"Expected a 2D image or 3D sequence, got shape {data.shape}")
            converted = _camus_uint8(data, is_mask)
            frames = [converted] if converted.ndim == 2 else [converted[:, :, i] for i in range(converted.shape[2])]

            for frame_index, frame in enumerate(frames):
                suffix = "" if converted.ndim == 2 else f"_frame_{frame_index:03d}"
                kind = "masks" if is_mask else "images"
                destination = output_root / "CAMUS" / patient / kind / f"{output_stem}{suffix}.png"
                # NIfTI indexes arrays as X,Y; PNG arrays are rows(Y),columns(X).
                png = camus_mask_to_palette(frame.T) if is_mask else Image.fromarray(frame.T)
                status = _save(png, destination, overwrite)
                records.append(
                    {
                        "dataset": "CAMUS",
                        "kind": "mask" if is_mask else "image",
                        "label": "class_ids_0_to_3" if is_mask else "",
                        "source": _relative(source, source_root),
                        "source_image": "",
                        "output": _relative(destination, output_root),
                        "width": png.width,
                        "height": png.height,
                        "frame_index": "" if converted.ndim == 2 else frame_index,
                        "status": status,
                    }
                )
        except Exception as exc:
            errors.append(
                {
                    "dataset": "CAMUS",
                    "source": _relative(source, source_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, errors


def convert_fallmud(
    source_root: Path, output_root: Path, overwrite: bool = False
) -> tuple[list[Record], list[Record]]:
    """Losslessly convert FALLMUD TIFF images/masks to PNG, preserving layout."""
    records: list[Record] = []
    errors: list[Record] = []
    try:
        dataset_root = _find_dataset(source_root, "(FALLMUD)")
    except Exception as exc:
        return records, [{"dataset": "FALLMUD", "error": str(exc)}]

    content_root = dataset_root / "FALLMUD"
    if not content_root.is_dir():
        content_root = dataset_root
    sources = sorted(
        [*content_root.rglob("*.tif"), *content_root.rglob("*.tiff")],
        key=lambda path: str(path).casefold(),
    )
    for source in sources:
        try:
            relative = source.relative_to(content_root).with_suffix(".png")
            destination = output_root / "FALLMUD" / relative
            with Image.open(source) as image:
                image.load()
                converted = image.copy()
            status = _save(converted, destination, overwrite)
            records.append(
                {
                    "dataset": "FALLMUD",
                    "kind": "mask" if "mask" in source.parent.name.casefold() else "image",
                    "label": source.parent.name if "mask" in source.parent.name.casefold() else "",
                    "source": _relative(source, source_root),
                    "source_image": "",
                    "output": _relative(destination, output_root),
                    "width": converted.width,
                    "height": converted.height,
                    "frame_index": "",
                    "status": status,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "dataset": "FALLMUD",
                    "source": _relative(source, source_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, errors


def _write_reports(output_root: Path, records: list[Record], errors: list[Record]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "kind",
        "label",
        "source",
        "source_image",
        "output",
        "width",
        "height",
        "frame_index",
        "status",
    ]
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "output_root": str(output_root.resolve()),
        "records": len(records),
        "errors": len(errors),
        "by_dataset_and_kind": {
            f"{dataset}/{kind}": count
            for (dataset, kind), count in sorted(
                Counter((row["dataset"], row["kind"]) for row in records).items()
            )
        },
        "by_status": dict(sorted(Counter(row["status"] for row in records).items())),
        "error_details": errors,
    }
    (output_root / "conversion_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _default_source_root() -> Path:
    workspace = Path(__file__).resolve().parents[1]
    candidates = [
        path / "pointed_data"
        for path in workspace.iterdir()
        if path.is_dir() and "(Datasets)" in path.name
    ]
    if len(candidates) != 1 or not candidates[0].is_dir():
        raise FileNotFoundError("Could not uniquely locate 数据集 (Datasets)/pointed_data")
    return candidates[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Path to pointed_data")
    parser.add_argument("--output-root", type=Path, help="Destination; defaults to SOURCE/_preprocessed_png")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("camus", "aul", "fallmud"),
        default=("camus", "aul", "fallmud"),
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing PNG outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = (args.source_root or _default_source_root()).resolve()
    output_root = (args.output_root or source_root / "_preprocessed_png").resolve()
    if not source_root.is_dir():
        print(f"Source root not found: {source_root}", file=sys.stderr)
        return 2

    converters = {
        "camus": convert_camus,
        "aul": convert_aul,
        "fallmud": convert_fallmud,
    }
    records: list[Record] = []
    errors: list[Record] = []
    for dataset in args.datasets:
        print(f"Converting {dataset.upper()} ...", flush=True)
        converted, failed = converters[dataset](source_root, output_root, args.overwrite)
        records.extend(converted)
        errors.extend(failed)
        print(f"  outputs={len(converted)}, errors={len(failed)}", flush=True)

    _write_reports(output_root, records, errors)
    print(f"Manifest: {output_root / 'manifest.csv'}")
    print(f"Report:   {output_root / 'conversion_report.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
