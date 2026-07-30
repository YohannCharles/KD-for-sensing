#!/usr/bin/env python3
"""Normalize SMSL's Full-mask-only F2 N/A values without hiding real failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


F2_FIELDS = ("f2_high20_g1_rate", "f2_high20_top1", "f2_risk_mean")


def _is_expected_na(value: Any) -> bool:
    return value == "not_applicable" or (isinstance(value, float) and math.isnan(value)) or str(value).lower() == "nan"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_mask_rows(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        is_full = row.get("mask") == "full"
        row["f2_applicable"] = not is_full
        for field in F2_FIELDS:
            value = row.get(field)
            if is_full:
                if not _is_expected_na(value):
                    raise ValueError(f"Full {field} is not the expected F2 N/A value: {value!r}")
                row[field] = "not_applicable"
            elif not math.isfinite(float(value)):
                raise ValueError(f"Non-Full {row.get('mask')} has non-finite {field}: {value!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalize_json(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _normalize_mask_rows(payload["masks"])
    _atomic_json(path, payload)


def normalize_process_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_summary = payload.get("result_summary")
    if not isinstance(result_summary, dict) or "masks" not in result_summary:
        raise ValueError(f"Process manifest has no result_summary masks: {path}")
    _normalize_mask_rows(result_summary["masks"])
    _atomic_json(path, payload)


def _normalize_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    _normalize_mask_rows(rows)
    if "f2_applicable" not in fields:
        fields.append("f2_applicable")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _normalize_log(path: Path) -> None:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "masks" in payload:
            _normalize_mask_rows(payload["masks"])
        rows.append(json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_fold(fold_dir: Path, log_root: Path, fold: int) -> dict[str, Any]:
    changed: list[dict[str, str]] = []
    for run_dir in sorted(path for path in fold_dir.iterdir() if path.is_dir()):
        paths = (
            run_dir / "evaluation_summary.json",
            run_dir / "evaluation_runtime_status.json",
            run_dir / "per_mask_metrics.csv",
            log_root / f"{run_dir.name}_fold{fold}.log",
        )
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            before = _sha256(path)
            if path.suffix == ".csv":
                _normalize_csv(path)
            elif path.suffix == ".log":
                _normalize_log(path)
            else:
                _normalize_json(path)
            changed.append({"path": str(path), "before_sha256": before, "after_sha256": _sha256(path)})
    manifest = {
        "status": "passed",
        "scope": "Full-mask F2 fields only",
        "replacement": "not_applicable",
        "scientific_metrics_changed": False,
        "files": changed,
    }
    _atomic_json(fold_dir / "artifact_normalization.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_fold(args.fold_dir, args.log_root, args.fold), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
