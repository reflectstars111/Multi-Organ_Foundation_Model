from __future__ import annotations

import argparse
import csv
import fnmatch
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ARCSINH_COLUMN = "arcsinh变换细胞测量值"
ROBUST_COLUMN = "线性变换（RobustScaler）细胞测量值"
NO_TRANSFORM_COLUMN = "无需变换细胞测量值"
CELL_COLUMN = "细胞测量值"
NON_CELL_COLUMN = "非细胞测量值"

CYTOF_COFACTOR = 5.0
FLUORESCENCE_COFACTOR = 150.0
DEFAULT_SMALL_ABS_MEAN_THRESHOLD = 10.0

FLUORESCENCE_DATASET_HINTS = {
    "11_Z3DP(no",
    "12_Z4VP(no",
    "15_ZZYA",
}

CYTOF_FILE_HINTS = (
    "levine_",
    "samusik_",
)

FLUORESCENCE_FILE_HINTS = (
    "flowcap_",
    "mosmann_",
    "nilsson_",
)

METADATA_REGEXES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^source($|_)",
        r"^source_",
        r"^folder_rule$",
        r"^dataset_base$",
        r"^is_notransform$",
        r"^event(_?id|_number)?$",
        r"^label$",
        r"^manual_label$",
        r"^gate__",
        r"^sample($|_|\.|case|tube|date|panel|count|material|markers)",
        r"^condition($|_|\.|part)",
        r"^file($|_|\.|name| number|key)",
        r"^fcs file$",
        r"^tube number$",
        r"^individual$",
        r"^subjectid$",
        r"^panelid$",
        r"^patient($|\.|_|id|\.id)",
        r"^cohort",
        r"^batch$",
        r"^plate$",
        r"^row$",
        r"^column$",
        r"^drug$",
        r"^control$",
        r"^concentration$",
        r"^diagnosis$",
        r"^comments$",
        r"^case_",
        r"^timepoint$",
        r"^months\.",
        r"^patient\.",
        r"^has_.*metadata$",
        r"^barcode$",
        r"^cohort_prefix$",
        r"^sample_code$",
        r"^ki67_code$",
        r"^clldiagnosis$",
        r"^beaddist$",
        r"^beads$",
        r"^center$",
        r"^offset$",
        r"^residual$",
        r"^190bckg$",
    ]
]

ROBUST_REGEXES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^(fsc|ssc)([-_ ]?[ahw])?$",
        r"^(fs|ss)\s+(int|peak|tof)\s+lin$",
        r"^fs\s+lin$",
        r"^ss\s+log$",
        r"^width$",
        r"^cell_length$",
        r"^event_length$",
    ]
]

FLUORESCENCE_REGEXES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bFSC(?:[-_ ]?[AHW])?\b",
        r"\bSSC(?:[-_ ]?[AHW])?\b",
        r"\bFITC\b",
        r"\bPerCP\b",
        r"\bAPC(?:[-_ ]?A?700|[-_ ]?A?750|[-_ ]?Cy7)?\b",
        r"(?:^|[\s_\-/])PE(?:$|[\s_\-/])",
        r"\bPE[-_ ]?Cy\d(?:\.\d)?\b",
        r"\bPECy\d\b",
        r"\bPE[-_ ]?CF\d+\b",
        r"\bPacific\b",
        r"\bQDot\b",
        r"\bAlexa(?:\s+Fluor)?\b",
        r"\bCFSE\b",
        r"\bBV\d+\b",
        r"\bBUV\d+\b",
        r"\bBB\d+\b",
        r"\bECD\b",
        r"\bPC5(?:\.5)?\b",
        r"\bPC7\b",
        r"\bKrOr\b",
        r"\bKO\b",
        r"\b[BRV]L\d+[-_ ]?[AH]\b",
        r"\bFS\s+(?:INT|PEAK|TOF)\s+LIN\b",
        r"\bSS\s+(?:INT|PEAK|TOF)\s+LIN\b",
        r"\bFS\s+Lin\b",
        r"\bSS\s+Log\b",
    ]
]

MASS_METAL_REGEX = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"(?:"
    r"Y89|I127|Ba138|Ce140|Gd157|Dy161|Dy163|Pt195|"
    r"\d{2,3}(?:Y|In|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Ir|Pt|Pd|Cd)"
    r")(?:Di)?"
    r"(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)

MASS_SPECIAL_REGEXES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^cell_length$",
        r"^event_length$",
        r"^dna[12]$",
        r"^bc\d+$",
        r"^cisplatin$",
        r"^beaddist$",
        r"^beads$",
        r"^center$",
        r"^offset$",
        r"^residual$",
    ]
]


@dataclass(frozen=True)
class ColumnSets:
    cell: set[str]
    non_cell: set[str]
    arcsinh: set[str]
    robust: set[str]
    no_transform: set[str]
    cell_norm: set[str]
    non_cell_norm: set[str]
    arcsinh_norm: set[str]
    robust_norm: set[str]
    no_transform_norm: set[str]


@dataclass(frozen=True)
class FlowClass:
    flow_type: str
    cofactor: float | None
    notes: str
    fluor_score: int
    cytof_score: int
    metal_score: int


@dataclass(frozen=True)
class PlannedFile:
    path: Path
    rel_path: Path
    dataset: str
    headers: list[str]
    flow_class: FlowClass
    actions: dict[str, str]


@dataclass(frozen=True)
class TransformResult:
    output_path: Path
    rows_written: int
    zero_iqr_columns: list[str]
    small_abs_mean_skipped_columns: list[str]
    small_abs_means: dict[str, float]


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


def non_empty_values(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame.columns:
        return set()
    return {
        str(value).strip()
        for value in frame[column].tolist()
        if str(value).strip()
    }


def load_column_sets(summary_csv: Path) -> ColumnSets:
    frame = pd.read_csv(summary_csv, dtype=str, keep_default_na=False)
    cell = non_empty_values(frame, CELL_COLUMN)
    non_cell = non_empty_values(frame, NON_CELL_COLUMN)
    arcsinh = non_empty_values(frame, ARCSINH_COLUMN)
    robust = non_empty_values(frame, ROBUST_COLUMN)
    no_transform = non_empty_values(frame, NO_TRANSFORM_COLUMN)

    return ColumnSets(
        cell=cell,
        non_cell=non_cell,
        arcsinh=arcsinh,
        robust=robust,
        no_transform=no_transform,
        cell_norm={normalize_name(value) for value in cell},
        non_cell_norm={normalize_name(value) for value in non_cell},
        arcsinh_norm={normalize_name(value) for value in arcsinh},
        robust_norm={normalize_name(value) for value in robust},
        no_transform_norm={normalize_name(value) for value in no_transform},
    )


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def count_regex_matches(headers: Iterable[str], regexes: Iterable[re.Pattern[str]]) -> tuple[int, list[str]]:
    matches: list[str] = []
    for header in headers:
        if any(regex.search(header) for regex in regexes):
            matches.append(header)
    return len(matches), matches


def classify_flow(headers: list[str], rel_path: Path) -> FlowClass:
    dataset = rel_path.parts[0] if rel_path.parts else ""
    rel_text = rel_path.as_posix().casefold()
    file_name = rel_path.name.casefold()

    fluor_score, fluor_matches = count_regex_matches(headers, FLUORESCENCE_REGEXES)
    cytof_special_score, cytof_matches = count_regex_matches(headers, MASS_SPECIAL_REGEXES)
    metal_matches = [header for header in headers if MASS_METAL_REGEX.search(header)]
    metal_score = len(metal_matches)
    cytof_score = cytof_special_score + metal_score

    notes: list[str] = []
    if fluor_matches:
        notes.append("fluorescence_features=" + "|".join(fluor_matches[:8]))
    if cytof_matches:
        notes.append("cytof_features=" + "|".join(cytof_matches[:8]))
    if metal_matches:
        notes.append("metal_features=" + "|".join(metal_matches[:8]))

    if any(file_name.startswith(prefix) for prefix in CYTOF_FILE_HINTS):
        notes.append("filename_hint=cytof")
        return FlowClass("cytof", CYTOF_COFACTOR, "; ".join(notes), fluor_score, cytof_score, metal_score)

    if any(file_name.startswith(prefix) for prefix in FLUORESCENCE_FILE_HINTS):
        notes.append("filename_hint=fluorescence_flow")
        return FlowClass(
            "fluorescence_flow",
            FLUORESCENCE_COFACTOR,
            "; ".join(notes),
            fluor_score,
            cytof_score,
            metal_score,
        )

    if dataset in FLUORESCENCE_DATASET_HINTS and metal_score == 0 and cytof_special_score < 3:
        notes.append("dataset_hint=fluorescence_flow")
        return FlowClass(
            "fluorescence_flow",
            FLUORESCENCE_COFACTOR,
            "; ".join(notes),
            fluor_score,
            cytof_score,
            metal_score,
        )

    if metal_score >= 1 or cytof_special_score >= 3:
        return FlowClass("cytof", CYTOF_COFACTOR, "; ".join(notes), fluor_score, cytof_score, metal_score)

    if fluor_score >= 2:
        return FlowClass(
            "fluorescence_flow",
            FLUORESCENCE_COFACTOR,
            "; ".join(notes),
            fluor_score,
            cytof_score,
            metal_score,
        )

    reason = "insufficient channel evidence"
    if "notransform" in rel_text:
        reason += "; notransform filename without platform-specific channels"
    if notes:
        reason += "; " + "; ".join(notes)
    return FlowClass("ambiguous", None, reason, fluor_score, cytof_score, metal_score)


def is_metadata_column(name: str, column_sets: ColumnSets) -> bool:
    norm = normalize_name(name)
    return norm in column_sets.non_cell_norm or any(regex.search(name) for regex in METADATA_REGEXES)


def is_no_transform_column(name: str, column_sets: ColumnSets) -> bool:
    norm = normalize_name(name)
    return norm in column_sets.no_transform_norm or norm == "time"


def is_robust_column(name: str, column_sets: ColumnSets) -> bool:
    norm = normalize_name(name)
    return norm in column_sets.robust_norm or any(regex.search(name) for regex in ROBUST_REGEXES)


def is_measurement_like(name: str) -> bool:
    norm = normalize_name(name)
    if norm in {"nan", "none", ""}:
        return False
    return not any(regex.search(name) for regex in METADATA_REGEXES)


def classify_column_actions(headers: list[str], column_sets: ColumnSets, flow_type: str) -> dict[str, str]:
    actions: dict[str, str] = {}
    for header in headers:
        norm = normalize_name(header)
        if is_metadata_column(header, column_sets):
            action = "metadata"
        elif is_no_transform_column(header, column_sets):
            action = "none"
        elif is_robust_column(header, column_sets):
            action = "robust_scaler"
        elif norm in column_sets.arcsinh_norm:
            action = "arcsinh"
        elif norm in column_sets.cell_norm:
            action = "arcsinh"
        elif flow_type != "ambiguous" and is_measurement_like(header):
            action = "arcsinh"
        else:
            action = "metadata"
        actions[header] = action
    return actions


def iter_event_files(events_dir: Path) -> list[Path]:
    files = (path for path in events_dir.rglob("*_events.csv") if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(events_dir).as_posix().casefold())


def matches_filters(path: Path, rel_path: Path, datasets: set[str], globs: list[str]) -> bool:
    if datasets and (not rel_path.parts or rel_path.parts[0] not in datasets):
        return False
    if not globs:
        return True
    rel_posix = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in globs)


def select_files(
    events_dir: Path,
    datasets: set[str],
    globs: list[str],
    max_files_per_dataset: int | None,
) -> list[Path]:
    selected: list[Path] = []
    per_dataset: defaultdict[str, int] = defaultdict(int)
    for path in iter_event_files(events_dir):
        rel_path = path.relative_to(events_dir)
        if not matches_filters(path, rel_path, datasets, globs):
            continue
        dataset = rel_path.parts[0] if rel_path.parts else ""
        if max_files_per_dataset is not None and per_dataset[dataset] >= max_files_per_dataset:
            continue
        selected.append(path)
        per_dataset[dataset] += 1
    return selected


def list_columns_by_action(actions: dict[str, str], action: str) -> list[str]:
    return [column for column, column_action in actions.items() if column_action == action]


def iter_limited_chunks(
    csv_path: Path,
    chunksize: int,
    row_limit: int | None,
    usecols: list[str] | None = None,
) -> Iterable[pd.DataFrame]:
    rows_seen = 0
    reader = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        usecols=usecols,
    )
    for chunk in reader:
        if row_limit is not None:
            remaining = row_limit - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        rows_seen += len(chunk)
        yield chunk
        if row_limit is not None and rows_seen >= row_limit:
            break


def compute_abs_means(
    csv_path: Path,
    columns: list[str],
    chunksize: int,
    row_limit: int | None,
) -> dict[str, float]:
    if not columns:
        return {}

    sums = {column: 0.0 for column in columns}
    counts = {column: 0 for column in columns}
    for chunk in iter_limited_chunks(csv_path, chunksize, row_limit, usecols=columns):
        numeric = chunk.apply(pd.to_numeric, errors="coerce").abs()
        chunk_sums = numeric.sum(axis=0, skipna=True)
        chunk_counts = numeric.count(axis=0)
        for column in columns:
            sums[column] += float(chunk_sums.get(column, 0.0))
            counts[column] += int(chunk_counts.get(column, 0))

    return {
        column: (sums[column] / counts[column] if counts[column] else float("nan"))
        for column in columns
    }


def compute_robust_stats(
    csv_path: Path,
    robust_columns: list[str],
    chunksize: int,
    row_limit: int | None,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    if not robust_columns:
        return {}, {}, []

    numeric_chunks: list[pd.DataFrame] = []
    for chunk in iter_limited_chunks(csv_path, chunksize, row_limit, usecols=robust_columns):
        numeric_chunks.append(chunk.apply(pd.to_numeric, errors="coerce"))

    if not numeric_chunks:
        return {column: 0.0 for column in robust_columns}, {column: 1.0 for column in robust_columns}, robust_columns

    numeric_data = pd.concat(numeric_chunks, ignore_index=True)
    medians = numeric_data.median(axis=0, skipna=True)
    q1 = numeric_data.quantile(0.25, axis=0, interpolation="linear")
    q3 = numeric_data.quantile(0.75, axis=0, interpolation="linear")

    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    zero_iqr_columns: list[str] = []
    for column in robust_columns:
        median = medians.get(column, np.nan)
        iqr = q3.get(column, np.nan) - q1.get(column, np.nan)
        if pd.isna(median):
            median = 0.0
        if pd.isna(iqr) or float(iqr) == 0.0:
            iqr = 1.0
            zero_iqr_columns.append(column)
        centers[column] = float(median)
        scales[column] = float(iqr)
    return centers, scales, zero_iqr_columns


def replace_numeric_values(original: pd.Series, transformed: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(original, errors="coerce")
    mask = numeric.notna()
    result = original.copy()
    result.loc[mask] = transformed.loc[mask].astype(float)
    return result


def transform_file(
    planned: PlannedFile,
    events_dir: Path,
    output_dir: Path,
    chunksize: int,
    row_limit: int | None,
    overwrite: bool,
    small_abs_mean_threshold: float | None,
) -> TransformResult:
    output_path = output_dir / planned.rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    if output_path.exists() and overwrite:
        output_path.unlink()

    arcsinh_columns = list_columns_by_action(planned.actions, "arcsinh")
    robust_columns = list_columns_by_action(planned.actions, "robust_scaler")
    candidate_transform_columns = arcsinh_columns + robust_columns
    small_abs_means = compute_abs_means(
        planned.path,
        candidate_transform_columns,
        chunksize=chunksize,
        row_limit=row_limit,
    )
    small_abs_mean_skipped_columns: list[str] = []
    if small_abs_mean_threshold is not None:
        small_abs_mean_skipped_columns = [
            column
            for column in candidate_transform_columns
            if not pd.isna(small_abs_means.get(column, np.nan))
            and small_abs_means[column] < small_abs_mean_threshold
        ]
    small_abs_mean_skipped = set(small_abs_mean_skipped_columns)
    arcsinh_columns = [column for column in arcsinh_columns if column not in small_abs_mean_skipped]
    robust_columns = [column for column in robust_columns if column not in small_abs_mean_skipped]

    centers, scales, zero_iqr_columns = compute_robust_stats(
        planned.path,
        robust_columns,
        chunksize=chunksize,
        row_limit=row_limit,
    )

    cofactor = planned.flow_class.cofactor
    if cofactor is None:
        raise ValueError("cannot transform ambiguous file without a cofactor")

    rows_written = 0
    first_chunk = True
    for chunk in iter_limited_chunks(planned.path, chunksize, row_limit):
        for column in arcsinh_columns:
            numeric = pd.to_numeric(chunk[column], errors="coerce")
            transformed = np.arcsinh(numeric / cofactor)
            chunk[column] = replace_numeric_values(chunk[column], transformed)

        for column in robust_columns:
            numeric = pd.to_numeric(chunk[column], errors="coerce")
            transformed = (numeric - centers[column]) / scales[column]
            chunk[column] = replace_numeric_values(chunk[column], transformed)

        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        first_chunk = False
        rows_written += len(chunk)

    return TransformResult(
        output_path=output_path,
        rows_written=rows_written,
        zero_iqr_columns=zero_iqr_columns,
        small_abs_mean_skipped_columns=small_abs_mean_skipped_columns,
        small_abs_means=small_abs_means,
    )


def make_manifest_row(
    planned: PlannedFile,
    status: str,
    output_path: Path | None,
    rows_written: int | None,
    notes: str,
    small_abs_mean_threshold: float | None,
    small_abs_mean_skipped_columns: list[str] | None = None,
    small_abs_means: dict[str, float] | None = None,
) -> dict[str, str | int | float]:
    actions = planned.actions
    metadata_columns = list_columns_by_action(actions, "metadata")
    unchanged_columns = list_columns_by_action(actions, "none")
    robust_columns = list_columns_by_action(actions, "robust_scaler")
    arcsinh_columns = list_columns_by_action(actions, "arcsinh")
    flow_class = planned.flow_class
    small_abs_mean_skipped_columns = small_abs_mean_skipped_columns or []
    small_abs_means = small_abs_means or {}
    small_abs_mean_skipped_values = [
        f"{column}={small_abs_means[column]:.6g}"
        for column in small_abs_mean_skipped_columns
        if column in small_abs_means and not pd.isna(small_abs_means[column])
    ]
    return {
        "dataset": planned.dataset,
        "file": planned.path.name,
        "relative_path": planned.rel_path.as_posix(),
        "flow_type": flow_class.flow_type,
        "cofactor": "" if flow_class.cofactor is None else flow_class.cofactor,
        "status": status,
        "rows_written": "" if rows_written is None else rows_written,
        "output_path": "" if output_path is None else output_path.as_posix(),
        "arcsinh_columns": ";".join(arcsinh_columns),
        "robust_scaler_columns": ";".join(robust_columns),
        "unchanged_columns": ";".join(unchanged_columns),
        "metadata_columns": ";".join(metadata_columns),
        "small_abs_mean_threshold": "" if small_abs_mean_threshold is None else small_abs_mean_threshold,
        "small_abs_mean_skipped_columns": ";".join(small_abs_mean_skipped_columns),
        "small_abs_mean_skipped_values": ";".join(small_abs_mean_skipped_values),
        "ambiguous_reason": flow_class.notes if flow_class.flow_type == "ambiguous" else "",
        "notes": notes,
        "fluor_score": flow_class.fluor_score,
        "cytof_score": flow_class.cytof_score,
        "metal_score": flow_class.metal_score,
    }


def plan_one_file(path: Path, events_dir: Path, column_sets: ColumnSets) -> PlannedFile:
    rel_path = path.relative_to(events_dir)
    dataset = rel_path.parts[0] if rel_path.parts else ""
    headers = read_csv_header(path)
    flow_class = classify_flow(headers, rel_path)
    actions = classify_column_actions(headers, column_sets, flow_class.flow_type)
    return PlannedFile(path=path, rel_path=rel_path, dataset=dataset, headers=headers, flow_class=flow_class, actions=actions)


def write_manifest_header(writer: csv.DictWriter) -> None:
    writer.writeheader()


def run(args: argparse.Namespace) -> int:
    events_dir = args.events_dir.resolve()
    output_dir = args.output_dir.resolve()
    summary_csv = args.summary_csv.resolve()
    manifest_path = args.manifest.resolve()

    if not events_dir.exists():
        raise FileNotFoundError(f"events dir not found: {events_dir}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary CSV not found: {summary_csv}")

    column_sets = load_column_sets(summary_csv)
    selected_files = select_files(
        events_dir,
        datasets=set(args.datasets or []),
        globs=args.file_glob or [],
        max_files_per_dataset=args.max_files_per_dataset,
    )
    if not selected_files:
        raise FileNotFoundError("no *_events.csv files matched the requested filters")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: Counter[str] = Counter()
    flow_counts: Counter[str] = Counter()
    dataset_type_counts: Counter[tuple[str, str]] = Counter()

    fieldnames = [
        "dataset",
        "file",
        "relative_path",
        "flow_type",
        "cofactor",
        "status",
        "rows_written",
        "output_path",
        "arcsinh_columns",
        "robust_scaler_columns",
        "unchanged_columns",
        "metadata_columns",
        "small_abs_mean_threshold",
        "small_abs_mean_skipped_columns",
        "small_abs_mean_skipped_values",
        "ambiguous_reason",
        "notes",
        "fluor_score",
        "cytof_score",
        "metal_score",
    ]

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        write_manifest_header(writer)

        for index, path in enumerate(selected_files, start=1):
            planned = plan_one_file(path, events_dir, column_sets)
            output_path: Path | None = None
            rows_written: int | None = None
            small_abs_mean_skipped_columns: list[str] = []
            small_abs_means: dict[str, float] = {}
            notes = planned.flow_class.notes
            status = "dry_run"

            if planned.flow_class.flow_type == "ambiguous":
                status = "ambiguous_skipped"
            elif not args.dry_run:
                try:
                    transform_result = transform_file(
                        planned,
                        events_dir=events_dir,
                        output_dir=output_dir,
                        chunksize=args.chunksize,
                        row_limit=args.row_limit,
                        overwrite=args.overwrite,
                        small_abs_mean_threshold=args.small_abs_mean_threshold,
                    )
                    output_path = transform_result.output_path
                    rows_written = transform_result.rows_written
                    zero_iqr_columns = transform_result.zero_iqr_columns
                    small_abs_mean_skipped_columns = transform_result.small_abs_mean_skipped_columns
                    small_abs_means = transform_result.small_abs_means
                    status = "transformed"
                    if zero_iqr_columns:
                        extra = "zero_iqr_columns=" + "|".join(zero_iqr_columns)
                        notes = f"{notes}; {extra}" if notes else extra
                    if small_abs_mean_skipped_columns:
                        extra = "small_abs_mean_skipped=" + "|".join(small_abs_mean_skipped_columns)
                        notes = f"{notes}; {extra}" if notes else extra
                except Exception as exc:  # pragma: no cover - exercised by real data failures.
                    if args.fail_fast:
                        raise
                    status = "error"
                    notes = f"{notes}; error={exc}" if notes else f"error={exc}"

            row = make_manifest_row(
                planned,
                status,
                output_path,
                rows_written,
                notes,
                args.small_abs_mean_threshold,
                small_abs_mean_skipped_columns,
                small_abs_means,
            )
            writer.writerow(row)
            status_counts[status] += 1
            flow_counts[planned.flow_class.flow_type] += 1
            dataset_type_counts[(planned.dataset, planned.flow_class.flow_type)] += 1

            if args.verbose and (index == 1 or index % args.progress_every == 0 or index == len(selected_files)):
                print(f"[{index}/{len(selected_files)}] {planned.rel_path.as_posix()} -> {status}", flush=True)

    print(f"manifest: {manifest_path}")
    print("status_counts:", dict(sorted(status_counts.items())))
    print("flow_counts:", dict(sorted(flow_counts.items())))
    print("dataset_type_counts:")
    for (dataset, flow_type), count in sorted(dataset_type_counts.items()):
        print(f"  {dataset}\t{flow_type}\t{count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify flow event CSV files and apply per-platform arcsinh/robust transformations.",
    )
    parser.add_argument("--events-dir", type=Path, default=Path("events"))
    parser.add_argument("--summary-csv", type=Path, default=Path("events_header_summary_classified.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("events_transformed"))
    parser.add_argument("--manifest", type=Path, default=Path("events_transformed") / "transform_manifest.csv")
    parser.add_argument("--dry-run", action="store_true", help="write only the manifest; do not transform CSV files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output CSV files")
    parser.add_argument("--fail-fast", action="store_true", help="stop immediately if one file fails")
    parser.add_argument("--chunksize", type=int, default=100_000, help="rows per pandas chunk while writing outputs")
    parser.add_argument("--row-limit", type=int, default=None, help="optional maximum rows per file, useful for smoke tests")
    parser.add_argument(
        "--small-abs-mean-threshold",
        type=float,
        default=DEFAULT_SMALL_ABS_MEAN_THRESHOLD,
        help="skip arcsinh/RobustScaler for numeric columns whose mean(abs(x)) is below this threshold; use a negative value to disable",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="optional top-level dataset directory names to include")
    parser.add_argument(
        "--file-glob",
        action="append",
        default=None,
        help="optional glob matched against relative paths or file names; can be passed multiple times",
    )
    parser.add_argument("--max-files-per-dataset", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    if args.row_limit is not None and args.row_limit <= 0:
        parser.error("--row-limit must be positive")
    if args.max_files_per_dataset is not None and args.max_files_per_dataset <= 0:
        parser.error("--max-files-per-dataset must be positive")
    if args.small_abs_mean_threshold < 0:
        args.small_abs_mean_threshold = None
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
