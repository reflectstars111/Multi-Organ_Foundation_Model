#!/usr/bin/env python3
"""Apply a visible palette to CAMUS PNG masks while preserving indices 0-3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .preprocess_pointed_data import CAMUS_PALETTE, camus_mask_to_palette
else:
    from preprocess_pointed_data import CAMUS_PALETTE, camus_mask_to_palette


def default_mask_root() -> Path:
    workspace = Path(__file__).resolve().parents[1]
    datasets = next(p for p in workspace.iterdir() if p.is_dir() and "(Datasets)" in p.name)
    pointed = datasets / "pointed_data"
    camus = next(p for p in pointed.iterdir() if p.is_dir() and "(CAMUS)" in p.name)
    return camus / "processed_png" / "masks"


def colorize(mask_root: Path) -> dict[str, object]:
    masks = sorted(mask_root.rglob("*.png"), key=lambda path: str(path).casefold())
    errors: list[str] = []
    value_counts = {str(index): 0 for index in range(4)}
    converted = 0
    for source in masks:
        try:
            with Image.open(source) as image:
                indices = np.asarray(image).copy()
            if indices.ndim != 2:
                raise ValueError(f"Expected indexed/grayscale mask, got shape {indices.shape}")
            for value in np.unique(indices):
                value_counts[str(int(value))] += int(np.count_nonzero(indices == value))
            colored = camus_mask_to_palette(indices)
            temporary = source.with_name(source.name + ".tmp")
            colored.save(temporary, format="PNG")
            os.replace(temporary, source)
            converted += 1
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    return {
        "mask_root": str(mask_root.resolve()),
        "converted": converted,
        "errors": errors,
        "class_indices_preserved": [0, 1, 2, 3],
        "palette_rgb": {str(index): list(color) for index, color in enumerate(CAMUS_PALETTE)},
        "pixel_counts": value_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path)
    args = parser.parse_args()
    mask_root = (args.mask_root or default_mask_root()).resolve()
    report = colorize(mask_root)
    report_path = mask_root.parent / "mask_palette_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
