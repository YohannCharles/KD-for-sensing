from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.data.transform_ops.io import joined_resource

EXECUTION_PROGRESS_FILENAME = "execution_progress.jsonl"

def write_loso_execute_summary(output_dir: str | Path, run_records: list[dict[str, Any]], *, status: str) -> dict[str, str]:
    out_dir = Path(output_dir)
    from kd_sensing.engine.hist_beam_loso_summary import _summary_row
    rows = [_summary_row(record) for record in run_records]
    completed_count = sum(1 for record in run_records if record.get("status") == "completed")
    failed_count = sum(1 for record in run_records if record.get("status") == "failed")
    missing_count = sum(1 for record in run_records if record.get("status") == "missing")
    eligible_count = sum(1 for row in rows if bool(row.get("main_conclusion_eligible", True)))
    excluded_count = len(rows) - eligible_count
    payload = {
        "status": status,
        "generated_at": _utc_now(),
        "run_count": len(run_records),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "missing_count": missing_count,
        "eligible_run_count": eligible_count,
        "excluded_run_count": excluded_count,
        "exclusion_reason_histogram": __import__("kd_sensing.engine.hist_beam_loso_summary", fromlist=["_reason_histogram"])._reason_histogram(row.get("eligibility_reasons", []) for row in rows),
        "claim_scope": __import__("kd_sensing.engine.hist_beam_loso_summary", fromlist=["_claim_scope_from_rows"])._claim_scope_from_rows(rows),
        "cross_scene_claim_allowed": all(bool(row.get("cross_scene_claim_allowed", True)) for row in rows) if rows else False,
        "runs": rows,
    }
    json_path = out_dir / "loso_summary.json"
    csv_path = out_dir / "loso_summary.csv"
    combined_path = out_dir / "combined_summary.csv"
    _write_json(json_path, payload)
    _write_summary_csv(csv_path, rows)
    _write_combined_summary_csv(combined_path, rows)
    return {"json": str(json_path), "csv": str(csv_path), "combined_summary": str(combined_path)}


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _write_combined_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "top1",
        "top3",
        "top5",
        "within1",
        "within2",
        "within3",
        "mae",
        "bpl_db",
        "nrp",
        "unique_pred_beams",
        "top1_pred_beam_ratio",
        "top5_pred_beam_ratio",
        "eligible",
        "eligibility_reasons",
        "trainable_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.get("variant"),
                    "top1": row.get("top1"),
                    "top3": row.get("top3"),
                    "top5": row.get("top5"),
                    "within1": row.get("within1"),
                    "within2": row.get("within2"),
                    "within3": row.get("within3"),
                    "mae": row.get("mae"),
                    "bpl_db": row.get("bpl_db"),
                    "nrp": row.get("nrp"),
                    "unique_pred_beams": row.get("unique_pred_beams"),
                    "top1_pred_beam_ratio": row.get("top1_pred_beam_ratio"),
                    "top5_pred_beam_ratio": row.get("top5_pred_beam_ratio"),
                    "eligible": bool(row.get("main_conclusion_eligible", False)),
                    "eligibility_reasons": _csv_cell(row.get("eligibility_reasons", [])),
                    "trainable_ratio": row.get("trainable_ratio"),
                }
            )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _append_execution_event(output_dir: str | Path, event: str, payload: Mapping[str, Any]) -> Path:
    path = Path(output_dir) / EXECUTION_PROGRESS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "timestamp": _utc_now(), **dict(payload)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def _append_stage_progress(stage_dir: str | Path, stage: str, payload: Mapping[str, Any]) -> Path:
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "timestamp": _utc_now(), **dict(payload)}
    path = directory / "progress.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    _write_json(directory / "progress_latest.json", record)
    return path


def _read_stage_progress(stage_dir: str | Path, *, phase: str | None = None) -> list[dict[str, Any]]:
    path = Path(stage_dir) / "progress.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if phase is None or payload.get("stage") == phase or payload.get("phase") == phase:
            rows.append(payload)
    return rows


def _resolve_csv_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _resolve_resource_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return joined_resource(root, str(value)).expanduser()


def _csv_records(path: Any) -> list[dict[str, Any]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _numbered_columns(columns: list[str], prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix) :]
        if suffix.isdigit():
            selected.append(col)
    return sorted(selected, key=lambda item: int(item[len(prefix) :]))


def _first_numbered_key(records: list[dict[str, Any]], prefix: str) -> str | None:
    if not records:
        return None
    columns = _numbered_columns(list(records[0].keys()), prefix)
    return columns[0] if columns else None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

__all__ = ["EXECUTION_PROGRESS_FILENAME", "write_loso_execute_summary"]
