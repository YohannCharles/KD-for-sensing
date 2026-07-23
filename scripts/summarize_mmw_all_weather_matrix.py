#!/usr/bin/env python3
"""Summarize Clean MMW U0 and retained baseline evaluation rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RETAINED_METHODS = ("U0", "amber_full", "rmbp_mm")
DEFAULT_GROUP_BY = (
    "method",
    "seed",
    "eval_family",
    "pattern",
    "missing_rate",
    "available_modalities",
    "metric_profile",
)
PAIR_KEYS = (
    "seed",
    "domain_id",
    "eval_family",
    "pattern",
    "missing_rate",
    "available_modalities",
    "mask_digest",
    "sample_csv_sha256",
    "mask_cache_checksum",
    "metric_profile",
)
NON_METRIC_FIELDS = {
    "method",
    "seed",
    "domain_id",
    "condition",
    "scene",
    "eval_family",
    "pattern",
    "missing_rate",
    "available_modalities",
    "mask_digest",
    "sample_csv_sha256",
    "mask_cache_checksum",
    "checkpoint_sha256",
    "checkpoint_role",
    "metric_profile",
    "sample_count",
    "expected_sample_count",
    "expected_domain_count",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Clean MMW U0, AMBER, and RMBP evaluation CSV rows.")
    parser.add_argument("--rows-csv", "--input", dest="rows_csv", action="append", required=True, help="Input evaluation rows CSV. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-by", default=",".join(DEFAULT_GROUP_BY), help="Comma-separated rollup columns.")
    parser.add_argument("--paired-deltas", action="store_true", help="Write U0-minus-baseline paired deltas when matching rows exist.")
    args = parser.parse_args(argv)

    rows = [row for path in args.rows_csv for row in _read_csv(Path(path))]
    _validate_rows(rows)
    group_by = tuple(_csv(args.group_by))
    if not group_by:
        parser.error("--group-by must contain at least one column")
    rollups = _rollups(rows, group_by)
    paired = _paired_deltas(rows) if args.paired_deltas else []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "rollups.csv", rollups)
    if args.paired_deltas:
        _write_csv(output_dir / "paired_deltas.csv", paired, fallback_columns=("method", "baseline", *PAIR_KEYS))
    summary = {
        "methods": sorted({row["method"] for row in rows}),
        "input_rows": len(rows),
        "rollup_rows": len(rollups),
        "paired_delta_rows": len(paired),
        "group_by": list(group_by),
        "paired_deltas_enabled": bool(args.paired_deltas),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No evaluation rows were supplied.")
    unsupported = sorted({row.get("method", "") for row in rows} - set(RETAINED_METHODS))
    if unsupported:
        raise ValueError(f"Only retained methods are accepted: {', '.join(unsupported)}")
    missing = [index for index, row in enumerate(rows, start=2) if not row.get("method", "").strip()]
    if missing:
        raise ValueError(f"Rows are missing method values at CSV lines: {missing}")


def _rollups(rows: list[dict[str, str]], group_by: tuple[str, ...]) -> list[dict[str, Any]]:
    metrics = _numeric_metric_columns(rows, excluded=group_by)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(column, "") for column in group_by)].append(row)
    result = []
    for key, items in sorted(groups.items()):
        output = dict(zip(group_by, key))
        output["row_count"] = len(items)
        if "sample_count" in {column for row in items for column in row}:
            output["sample_count"] = sum(_number(row.get("sample_count")) or 0 for row in items)
        for metric in metrics:
            values = [value for row in items if (value := _number(row.get(metric))) is not None]
            output[metric] = sum(values) / len(values) if values else ""
        result.append(output)
    return result


def _paired_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metrics = _numeric_metric_columns(rows, excluded=PAIR_KEYS)
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        identity = (row["method"], *(row.get(key, "") for key in PAIR_KEYS))
        if identity in index:
            raise ValueError(f"Duplicate paired-comparison row: {identity}")
        index[identity] = row
    result = []
    for baseline in ("amber_full", "rmbp_mm"):
        for identity, u0_row in index.items():
            if identity[0] != "U0":
                continue
            baseline_row = index.get((baseline, *identity[1:]))
            if baseline_row is None:
                continue
            output = {"method": "U0", "baseline": baseline, **dict(zip(PAIR_KEYS, identity[1:]))}
            for metric in metrics:
                u0_value, baseline_value = _number(u0_row.get(metric)), _number(baseline_row.get(metric))
                output[f"delta_{metric}"] = "" if u0_value is None or baseline_value is None else u0_value - baseline_value
            result.append(output)
    return result


def _numeric_metric_columns(rows: Iterable[dict[str, str]], *, excluded: Iterable[str] = ()) -> tuple[str, ...]:
    blocked = NON_METRIC_FIELDS | set(excluded)
    columns = sorted({column for row in rows for column in row} - blocked)
    return tuple(column for column in columns if any(_number(row.get(column)) is not None for row in rows))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input rows CSV is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input rows CSV has no header: {path}")
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, Any]], fallback_columns: tuple[str, ...] = ()) -> None:
    columns = list(fallback_columns)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
