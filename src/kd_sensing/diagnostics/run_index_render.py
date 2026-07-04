import csv
import json
from pathlib import Path
from typing import Any


def render_run_table(index: dict[str, Any]) -> str:
    rows = [
        [
            run.get("state", ""),
            run.get("run_name", ""),
            run.get("config", {}).get("dataset_family") or "",
            run.get("config", {}).get("objective") or "",
            ",".join(str(item) for item in run.get("config", {}).get("modalities", [])),
            _format_primary_metric(run.get("metrics", {})),
            run.get("timestamps", {}).get("last_updated_at") or "",
            run.get("state_reason", ""),
        ]
        for run in index.get("runs", [])
    ]
    headers = ["state", "run", "dataset", "objective", "modalities", "metric", "updated", "reason"]
    return _render_table(headers, rows)

def render_run_csv(index: dict[str, Any]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_csv_fieldnames())
    writer.writeheader()
    for run in index.get("runs", []):
        writer.writerow(_run_csv_row(run))
    return buffer.getvalue()

def write_run_index_output(index: dict[str, Any], *, path: str | Path, format: str) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        target.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    elif format == "csv":
        target.write_text(render_run_csv(index), encoding="utf-8")
    else:
        target.write_text(render_run_table(index), encoding="utf-8")
    return target

def _format_primary_metric(metrics: dict[str, Any]) -> str:
    primary = metrics.get("primary", {}) if isinstance(metrics.get("primary"), dict) else {}
    name = primary.get("name")
    value = primary.get("value")
    if name is None or value is None:
        return ""
    if isinstance(value, float):
        return f"{name}={value:.4g}"
    return f"{name}={value}"

def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], min(len(str(value)), 80))
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        clipped = [str(value)[:80] for value in row]
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(clipped)))
    if not rows:
        lines.append("(no runs)")
    return "\n".join(lines)

def _csv_fieldnames() -> list[str]:
    return [
        "run_dir",
        "run_name",
        "state",
        "state_reason",
        "dataset_family",
        "experiment_name",
        "task",
        "objective",
        "modalities",
        "primary_metric",
        "primary_value",
        "last_updated_at",
        "pid",
    ]

def _run_csv_row(run: dict[str, Any]) -> dict[str, Any]:
    config = run.get("config", {})
    primary = run.get("metrics", {}).get("primary", {})
    return {
        "run_dir": run.get("run_dir"),
        "run_name": run.get("run_name"),
        "state": run.get("state"),
        "state_reason": run.get("state_reason"),
        "dataset_family": config.get("dataset_family"),
        "experiment_name": config.get("experiment_name"),
        "task": config.get("task"),
        "objective": config.get("objective"),
        "modalities": ",".join(str(item) for item in config.get("modalities", [])),
        "primary_metric": primary.get("name"),
        "primary_value": primary.get("value"),
        "last_updated_at": run.get("timestamps", {}).get("last_updated_at"),
        "pid": run.get("process", {}).get("pid") if run.get("process") else None,
    }
