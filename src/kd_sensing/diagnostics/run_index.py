import datetime as dt
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from kd_sensing.utils.runtime_output_layout import is_default_skipped_partition

from kd_sensing.diagnostics.run_index_artifacts import (
    infer_run_state,
    summarize_artifacts,
    summarize_checkpoints,
    summarize_cleanup_context,
    summarize_config,
    summarize_metrics,
    summarize_run_dir,
    summarize_tensorboard,
    summarize_timestamps,
)
from kd_sensing.diagnostics.run_index_base import (
    DEFAULT_STALE_AFTER,
    RUN_STATES,
    RunIndexFilters,
    _as_list,
    _ensure_utc,
    _filters_to_dict,
    _format_dt,
    _matches_filters,
    _resolve_existing_or_requested,
    _utc_now,
)
from kd_sensing.diagnostics.run_index_render import (
    render_run_csv,
    render_run_table,
    write_run_index_output,
)
from kd_sensing.diagnostics.run_index_resources import (
    _empty_resources,
    collect_python_processes,
    collect_resource_snapshot,
    redact_command,
    sanitize_process_records,
)
from kd_sensing.diagnostics.run_index_scanner import (
    collect_log_records,
    discover_run_dirs,
    parse_failure_patterns,
)


RUN_CARD_SCHEMA_VERSION = 1
DEFAULT_RUN_CARD_DIR = Path("outputs/analysis/run_cards")
RUN_CARD_JSON_SCHEMA = {
    "schema_version": RUN_CARD_SCHEMA_VERSION,
    "required": [
        "schema_version",
        "generated_at",
        "run",
        "command",
        "git",
        "config",
        "dataset_split",
        "checkpoint",
        "metrics",
        "provenance",
        "caveat",
        "warnings",
    ],
}


def build_run_index(
    *,
    outputs: str | Path | Iterable[str | Path] = "outputs",
    logs: str | Path | Iterable[str | Path] | None = "logs",
    filters: RunIndexFilters | None = None,
    include_resources: bool = True,
    stale_after: dt.timedelta = DEFAULT_STALE_AFTER,
    now: dt.datetime | None = None,
    processes: list[dict[str, Any]] | None = None,
    resources: dict[str, Any] | None = None,
    include_legacy_containers: bool = False,
) -> dict[str, Any]:
    """Build a read-only index of local experiment run directories."""

    generated_at = _utc_now() if now is None else _ensure_utc(now)
    output_roots = [_resolve_existing_or_requested(path) for path in _as_list(outputs)]
    log_roots = [_resolve_existing_or_requested(path) for path in _as_list(logs)] if logs is not None else []
    filter_spec = filters or RunIndexFilters()
    process_records = processes
    if process_records is None:
        process_records = collect_python_processes() if include_resources else []
    resource_snapshot = resources
    if resource_snapshot is None:
        resource_snapshot = collect_resource_snapshot(process_records) if include_resources else _empty_resources(process_records)
    else:
        resource_snapshot = dict(resource_snapshot)
        resource_snapshot.setdefault("processes", process_records)
    process_records = sanitize_process_records(list(resource_snapshot.get("processes", process_records)))
    resource_snapshot["processes"] = process_records

    warnings: list[str] = []
    log_records = collect_log_records(log_roots, warnings=warnings)
    run_dirs: list[Path] = []
    for root in output_roots:
        if root.exists():
            run_dirs.extend(discover_run_dirs(root, warnings=warnings, include_legacy_containers=include_legacy_containers))
        else:
            warnings.append(f"outputs root does not exist: {root}")

    runs = [
        summarize_run_dir(
            run_dir,
            logs=log_records,
            processes=process_records,
            now=generated_at,
            stale_after=stale_after,
            warnings=warnings,
        )
        for run_dir in sorted(set(run_dirs), key=lambda path: path.as_posix())
    ]
    runs = [run for run in runs if _matches_filters(run, filter_spec)]
    return {
        "generated_at": _format_dt(generated_at),
        "roots": {
            "outputs": [str(path) for path in output_roots],
            "logs": [str(path) for path in log_roots],
            "explicit_non_run_partitions": [
                str(path) for path in output_roots if is_default_skipped_partition(path)
            ],
        },
        "filters": _filters_to_dict(filter_spec),
        "runs": runs,
        "resources": resource_snapshot,
        "warnings": warnings,
    }

def build_run_card(
    run: dict[str, Any] | str | Path,
    *,
    project_root: str | Path = ".",
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    generated_at = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    summary = dict(run) if isinstance(run, dict) else summarize_run_dir(run, logs=[], processes=[], now=now, warnings=warnings)
    status = _read_json_file(summary.get("artifacts", {}).get("run_status", {}).get("path"))
    config = dict(summary.get("config", {}))
    provenance = dict(summary.get("provenance", {}))
    checkpoint = _run_card_checkpoint(summary)
    if not checkpoint.get("path"):
        warnings.append("checkpoint provenance is unavailable")
    if not config.get("config_path"):
        warnings.append("config path is unavailable")
    if not summary.get("metrics", {}).get("path"):
        warnings.append("metrics path is unavailable")
    return {
        "schema_version": RUN_CARD_SCHEMA_VERSION,
        "json_schema": RUN_CARD_JSON_SCHEMA,
        "generated_at": generated_at,
        "run": {
            "run_id": provenance.get("run_id"),
            "run_name": summary.get("run_name"),
            "run_dir": summary.get("run_dir"),
            "state": summary.get("state"),
            "state_reason": summary.get("state_reason"),
            "started_at": status.get("start_time"),
            "completed_at": status.get("end_time"),
            "output_root": summary.get("runtime_layout", {}).get("root"),
        },
        "command": _redact_command(status.get("command") or (summary.get("process") or {}).get("cmdline")),
        "git": _git_summary(project_root),
        "config": {
            "path": config.get("config_path") or status.get("config_path"),
            "digest": config.get("config_digest"),
            "experiment_name": config.get("experiment_name"),
            "task": config.get("task"),
            "objective": config.get("objective"),
            "modalities": config.get("modalities", []),
        },
        "dataset_split": {
            "dataset_family": config.get("dataset_family"),
            "scene_scope": config.get("scene_scope") or config.get("scene"),
            "split": config.get("split"),
            "sample_count": config.get("sample_count"),
            "label_space": config.get("label_space"),
            "target_source": config.get("target_source"),
            "difficulty_digest": config.get("difficulty_digest"),
        },
        "checkpoint": checkpoint,
        "metrics": {
            "path": summary.get("metrics", {}).get("path"),
            "primary": summary.get("metrics", {}).get("primary", {}),
            "scalar_count": len(summary.get("metrics", {}).get("scalars", {}) or {}),
        },
        "provenance": {
            "run_id": provenance.get("run_id"),
            "metric_profile": provenance.get("metric_profile"),
            "artifact_paths": provenance.get("artifact_paths", {}),
        },
        "caveat": "run card is a local provenance artifact; it is not a reviewed paper claim",
        "warnings": warnings,
    }

def write_run_card(card: dict[str, Any], *, output_dir: str | Path = DEFAULT_RUN_CARD_DIR) -> dict[str, str]:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(card.get("run", {}).get("run_name") or card.get("run", {}).get("run_id") or "run")
    json_path = out_dir / f"{stem}_run_card.json"
    markdown_path = out_dir / f"{stem}_run_card.md"
    json_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_run_card_markdown(card), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}

def render_run_card_markdown(card: dict[str, Any]) -> str:
    run = card.get("run", {})
    config = card.get("config", {})
    dataset = card.get("dataset_split", {})
    checkpoint = card.get("checkpoint", {})
    metrics = card.get("metrics", {})
    lines = [
        f"# Run Card: {run.get('run_name') or run.get('run_id') or 'run'}",
        "",
        f"- schema_version: {card.get('schema_version')}",
        f"- generated_at: {card.get('generated_at')}",
        f"- state: {run.get('state')} ({run.get('state_reason')})",
        f"- run_dir: {run.get('run_dir')}",
        f"- command: {card.get('command') or 'unavailable'}",
        f"- git_commit: {card.get('git', {}).get('commit')}",
        f"- git_dirty: {card.get('git', {}).get('dirty')}",
        f"- config: {config.get('path')} ({config.get('digest')})",
        f"- dataset_split: {dataset.get('dataset_family')} / {dataset.get('split')}",
        f"- label_space: {dataset.get('label_space')}",
        f"- difficulty_digest: {dataset.get('difficulty_digest')}",
        f"- metrics: {metrics.get('path')} primary={metrics.get('primary')}",
        f"- checkpoint: {checkpoint.get('path')} sidecar={checkpoint.get('sidecar_path')}",
        f"- caveat: {card.get('caveat')}",
    ]
    warnings = card.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"

def _run_card_checkpoint(summary: dict[str, Any]) -> dict[str, Any]:
    checkpoints = summary.get("checkpoints", {})
    primary = checkpoints.get("primary_checkpoint") or checkpoints.get("best_checkpoint")
    items = checkpoints.get("items", []) or []
    item = next((entry for entry in items if entry.get("path") == primary), {})
    metadata = item.get("sidecar_metadata", {}) if isinstance(item.get("sidecar_metadata"), dict) else {}
    values = metadata.get("values", {}) if isinstance(metadata.get("values"), dict) else metadata
    return {
        "path": primary,
        "sidecar_path": item.get("sidecar_path"),
        "source": "run_index_summary" if primary else "unavailable",
        "selection_metric": values.get("selection_metric") or values.get("selected_metric") or values.get("primary_metric"),
        "selected_epoch": values.get("selected_epoch") or values.get("best_epoch") or values.get("epoch"),
        "checkpoint_count": checkpoints.get("count", 0),
        "total_size_bytes": checkpoints.get("total_size_bytes", 0),
    }

def _git_summary(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    commit = _git_output(["git", "rev-parse", "HEAD"], root) or "unavailable"
    status = _git_output(["git", "status", "--short"], root)
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_summary": "dirty" if status else "clean",
    }

def _git_output(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def _read_json_file(path: Any) -> dict[str, Any]:
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

def _redact_command(command: Any) -> str | None:
    return redact_command(command)

def _safe_stem(value: Any) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return stem or "run"
