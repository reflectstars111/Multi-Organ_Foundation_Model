"""Export raw FCS event values to CSV without compensation or transformation.

Install the reader once with ``python -m pip install FlowIO``. By default this
script reads only .fcs files directly inside FR-FCM-Z8P9; subfolders are never
scanned.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "FR-FCM-Z8P9"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "FR-FCM-Z8P9_csv_raw"


def convert_one(source: Path, target: Path, *, chunk_rows: int, overwrite: bool) -> tuple[int, int]:
    try:
        import flowio
    except ImportError as exc:
        raise RuntimeError("缺少 FlowIO：请运行 python -m pip install FlowIO") from exc

    if target.exists() and not overwrite:
        raise FileExistsError(f"CSV 已存在，不覆盖：{target}")

    # FlowIO's as_array() defaults to preprocess=True.  False is essential:
    # it preserves the numeric values encoded in the FCS DATA segment.
    fcs = flowio.FlowData(source)
    values = fcs.as_array(preprocess=False)
    columns = list(fcs.pnn_labels)
    expected_shape = (int(fcs.event_count), int(fcs.channel_count))
    if values.shape != expected_shape or len(columns) != expected_shape[1]:
        raise ValueError(f"FCS 数据形状或通道名不匹配：{source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            csv.writer(handle, lineterminator="\n").writerow(columns)
            for start in range(0, expected_shape[0], chunk_rows):
                # 17 significant digits round-trip IEEE float64 values.
                # No arcsinh, compensation, scaling, filtering or gating.
                np.savetxt(handle, values[start:start + chunk_rows],
                           fmt="%.17g", delimiter=",", newline="\n")
        if target.exists() and not overwrite:
            raise FileExistsError(f"CSV 在导出期间出现，不覆盖：{target}")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return expected_shape


def main() -> int:
    parser = argparse.ArgumentParser(description="FCS 原始事件值逐文件导出 CSV，不做数据变换")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--limit", type=int, help="只处理前 N 个文件，便于试运行")
    parser.add_argument("--dry-run", action="store_true", help="只列出输入与目标，不读写文件")
    parser.add_argument("--overwrite", action="store_true", help="明确允许替换已有 CSV")
    args = parser.parse_args()
    if args.chunk_rows < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--chunk-rows 和 --limit 必须为正整数")
    source_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        parser.error(f"输入目录不存在：{source_dir}")
    if source_dir == output_dir:
        parser.error("输出目录必须与输入目录不同")

    sources = sorted((path for path in source_dir.iterdir()
                      if path.is_file() and path.suffix.casefold() == ".fcs"),
                     key=lambda path: path.name.casefold())
    if args.limit is not None:
        sources = sources[:args.limit]
    if not sources:
        parser.error("输入目录下没有 FCS 文件")
    targets = [output_dir / f"{source.stem}.csv" for source in sources]
    if len({path.name.casefold() for path in targets}) != len(targets):
        parser.error("输出 CSV 文件名发生冲突")
    if not args.dry_run and not args.overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            parser.error(f"已有 {len(existing)} 个目标 CSV；请换输出目录或显式加 --overwrite")

    print(f"FCS 文件数：{len(sources)}；输出目录：{output_dir}", flush=True)
    for index, (source, target) in enumerate(zip(sources, targets), 1):
        if args.dry_run:
            print(f"[{index}/{len(sources)}] {source.name} -> {target.name}")
            continue
        events, channels = convert_one(source, target, chunk_rows=args.chunk_rows,
                                        overwrite=args.overwrite)
        print(f"[{index}/{len(sources)}] {source.name} -> {target.name} "
              f"({events} 行 × {channels} 通道)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
