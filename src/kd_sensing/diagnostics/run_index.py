from __future__ import annotations

from dataclasses import dataclass
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.utils.runtime_output_layout import (
    DEFAULT_NON_RUN_PARTITIONS,
    is_default_outputs_root,
    is_default_skipped_partition,
    output_layout_summary,
)
from kd_sensing.utils.paths import resolve_path


RUN_STATES = {
    "running",
    "complete",
    "started_no_metrics",
    "partial",
    "failed",
    "killed",
    "waiting",
    "stale",
    "unknown",
}
RUN_STATUS_FILENAME = "run_status.json"
RUN_ARTIFACT_NAMES = {
    "final_config": "final_config.yaml",
    "resolved_config": "resolved_config.yaml",
    "startup_summary": "startup_summary.json",
    "metrics": "metrics.json",
    "train_log": "train_log.json",
    "training_outputs": "training_outputs.npz",
    "test_report": "test_report.json",
    "run_status": RUN_STATUS_FILENAME,
}
DISCOVERY_FILENAMES = set(RUN_ARTIFACT_NAMES.values())
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
DEFAULT_STALE_AFTER = dt.timedelta(hours=12)
LOG_TAIL_BYTES = 64 * 1024


@dataclass(frozen=True)
class RunIndexFilters:
    states: tuple[str, ...] = ()
    dataset_family: str | None = None
    objective: str | None = None
    run_name: str | None = None
    since: dt.datetime | None = None
    until: dt.datetime | None = None


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
    process_records = list(resource_snapshot.get("processes", process_records))

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


def discover_run_dirs(
    outputs_root: str | Path,
    *,
    warnings: list[str] | None = None,
    include_legacy_containers: bool = False,
) -> list[Path]:
    root = Path(outputs_root)
    candidates: set[Path] = set()
    skip_default_partitions = is_default_outputs_root(root)
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        if skip_default_partitions and current_path == root:
            for dirname in list(dirnames):
                if dirname in DEFAULT_NON_RUN_PARTITIONS:
                    skipped = current_path / dirname
                    if warnings is not None:
                        warnings.append(f"skipped non-run outputs partition by default: {skipped}")
                    dirnames.remove(dirname)
                elif (
                    not include_legacy_containers
                    and not _is_canonical_scan_partition_name(dirname)
                    and not _shallow_child_may_contain_run(current_path / dirname)
                ):
                    skipped = current_path / dirname
                    if warnings is not None:
                        warnings.append(f"skipped non-canonical outputs partition by default: {skipped}")
                    dirnames.remove(dirname)
        if skip_default_partitions and current_path != root:
            _prune_default_outputs_children(root, current_path, dirnames, warnings=warnings)
        if _directory_contains_run_marker(current_path, filenames):
            candidates.add(current_path)
            dirnames[:] = []
            continue
        if current_path.name == "checkpoints" and any(_is_checkpoint(current_path / name) for name in filenames):
            candidates.add(current_path.parent)
            dirnames[:] = []
            continue
        if current_path.name == "tensorboard" and any(_is_tensorboard_event(current_path / name) for name in filenames):
            candidates.add(current_path.parent)
            dirnames[:] = []
            continue
        for filename in filenames:
            path = current_path / filename
            if _is_discovery_artifact(path):
                candidates.add(path.parent)
            elif _is_checkpoint(path):
                candidates.add(path.parent.parent if path.parent.name == "checkpoints" else path.parent)
            elif _is_tensorboard_event(path):
                candidates.add(path.parent.parent if path.parent.name == "tensorboard" else path.parent)
    return sorted(candidates, key=lambda path: path.as_posix())


def summarize_run_dir(
    run_dir: str | Path,
    *,
    logs: list[dict[str, Any]] | None = None,
    processes: list[dict[str, Any]] | None = None,
    now: dt.datetime | None = None,
    stale_after: dt.timedelta = DEFAULT_STALE_AFTER,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    current_time = _utc_now() if now is None else _ensure_utc(now)
    path = Path(run_dir).resolve()
    artifacts = summarize_artifacts(path)
    timestamps = summarize_timestamps(path, artifacts=artifacts, now=current_time)
    config = summarize_config(path, artifacts=artifacts, warnings=warnings)
    metrics = summarize_metrics(path, artifacts=artifacts, warnings=warnings)
    checkpoints = summarize_checkpoints(path, warnings=warnings)
    tensorboard = summarize_tensorboard(path)
    run_logs = associate_logs(path, config=config, logs=logs or [])
    process = match_run_process(path, config=config, processes=processes or [])
    state, state_reason = infer_run_state(
        artifacts=artifacts,
        timestamps=timestamps,
        sidecar=_read_json(path / RUN_STATUS_FILENAME, warnings=warnings),
        logs=run_logs,
        process=process,
        now=current_time,
        stale_after=stale_after,
    )
    cleanup = summarize_cleanup_context(
        state=state,
        artifacts=artifacts,
        timestamps=timestamps,
        checkpoints=checkpoints,
        logs=run_logs,
        stale_after=stale_after,
    )
    layout = output_layout_summary(path)
    return {
        "run_dir": str(path),
        "run_name": path.name,
        "state": state,
        "state_reason": state_reason,
        "size_bytes": _run_size_bytes(artifacts=artifacts, checkpoints=checkpoints),
        "runtime_layout": layout,
        "config": config,
        "artifacts": artifacts,
        "metrics": metrics,
        "checkpoints": checkpoints,
        "tensorboard": tensorboard,
        "logs": run_logs,
        "process": process,
        "resources": _run_resource_summary(process),
        "timestamps": timestamps,
        "cleanup": cleanup,
    }


def summarize_artifacts(run_dir: Path) -> dict[str, Any]:
    items: dict[str, Any] = {}
    missing: list[str] = []
    for key, filename in RUN_ARTIFACT_NAMES.items():
        path = run_dir / filename
        present = path.exists()
        items[key] = {"present": present, "path": str(path) if present else None}
        if not present and key in {"final_config", "resolved_config", "startup_summary", "metrics", "train_log"}:
            missing.append(filename)
    checkpoint_files = sorted(
        str(path)
        for path in (run_dir / "checkpoints").glob("*")
        if path.is_file() and _is_checkpoint(path)
    )
    checkpoint_sidecars = sorted(
        str(path)
        for path in (run_dir / "checkpoints").glob("*.json")
        if path.is_file()
    )
    tensorboard_events = [str(path) for path in _tensorboard_event_paths(run_dir)]
    items["checkpoints"] = {"present": bool(checkpoint_files), "paths": checkpoint_files}
    items["checkpoint_sidecars"] = {"present": bool(checkpoint_sidecars), "paths": checkpoint_sidecars}
    items["tensorboard_events"] = {"present": bool(tensorboard_events), "paths": tensorboard_events}
    items["missing"] = missing
    return items


def summarize_timestamps(run_dir: Path, *, artifacts: dict[str, Any], now: dt.datetime) -> dict[str, Any]:
    paths = _artifact_paths(artifacts)
    mtimes = [_mtime(path) for path in paths if path.exists()]
    newest = max(mtimes) if mtimes else _mtime(run_dir)
    oldest = min(mtimes) if mtimes else _mtime(run_dir)
    by_artifact: dict[str, str | None] = {}
    for key, item in artifacts.items():
        if isinstance(item, dict) and item.get("path"):
            by_artifact[key] = _format_dt(_mtime(Path(item["path"])))
    return {
        "first_artifact_at": _format_dt(oldest) if oldest is not None else None,
        "last_updated_at": _format_dt(newest) if newest is not None else None,
        "age_seconds": int((now - newest).total_seconds()) if newest is not None else None,
        "by_artifact": by_artifact,
    }


def summarize_config(run_dir: Path, *, artifacts: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    config_path = _first_present_path(artifacts, ("final_config", "resolved_config"))
    cfg = _read_yaml(config_path, warnings=warnings) if config_path is not None else {}
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    dataset_cfg = _nested_dict(cfg, "data", "dataset")
    model_primary = _nested_dict(cfg, "model", "primary")
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    objective_meta = runtime.get("prediction_objective") if isinstance(runtime.get("prediction_objective"), dict) else {}
    modalities = model_primary.get("modalities") or cfg.get("model", {}).get("modalities") or cfg.get("modalities")
    if modalities is None and experiment.get("task") in {"image", "radar", "gps", "lidar", "mmwave", "csi"}:
        modalities = [experiment["task"]]
    return {
        "config_path": str(config_path) if config_path is not None else None,
        "dataset_family": _dataset_family(dataset_cfg, run_dir),
        "experiment_name": experiment.get("name"),
        "task": experiment.get("task"),
        "objective": experiment.get("objective") or objective_meta.get("name"),
        "modalities": list(modalities) if isinstance(modalities, (list, tuple)) else ([] if modalities is None else [modalities]),
        "seed": experiment.get("seed"),
        "output_run_name": _nested_dict(cfg, "output").get("run_name"),
        "scene": runtime.get("scene"),
        "scene_scope": runtime.get("scene_scope") or runtime.get("output_scope"),
    }


def summarize_metrics(run_dir: Path, *, artifacts: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    metrics_path = Path(artifacts["metrics"]["path"]) if artifacts.get("metrics", {}).get("path") else None
    report_path = Path(artifacts["test_report"]["path"]) if artifacts.get("test_report", {}).get("path") else None
    raw: dict[str, Any] = {}
    source = None
    if metrics_path is not None:
        raw = _read_json(metrics_path, warnings=warnings) or {}
        source = metrics_path
    elif report_path is not None:
        report = _read_json(report_path, warnings=warnings) or {}
        raw = report.get("metrics", report)
        source = report_path
    scalar_metrics = _scalar_metrics(raw)
    primary_name = _primary_metric_name(raw)
    primary_value = _metric_value(raw, primary_name) if primary_name else None
    return {
        "path": str(source) if source is not None else None,
        "available": bool(raw),
        "primary": {"name": primary_name, "value": primary_value},
        "scalars": scalar_metrics,
    }


def summarize_checkpoints(run_dir: Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    paths = sorted(path for path in checkpoint_dir.glob("*") if path.is_file() and _is_checkpoint(path))
    best = _best_checkpoint(paths)
    items = [_checkpoint_item(path, best=best, warnings=warnings) for path in paths]
    sidecars = [item for item in items if item.get("sidecar_present")]
    best_sidecar = None
    for item in items:
        if item.get("path") == (str(best) if best is not None else None):
            best_sidecar = item.get("sidecar_metadata")
            break
    total_size = sum(int(item.get("size_bytes") or 0) for item in items)
    return {
        "count": len(paths),
        "paths": [str(path) for path in paths],
        "total_size_bytes": total_size,
        "best_checkpoint": str(best) if best is not None else None,
        "primary_checkpoint": str(best) if best is not None else None,
        "best_metadata": best_sidecar,
        "sidecar_count": len(sidecars),
        "items": items,
        "retention": _checkpoint_retention_summary(items),
    }


def summarize_cleanup_context(
    *,
    state: str,
    artifacts: dict[str, Any],
    timestamps: dict[str, Any],
    checkpoints: dict[str, Any],
    logs: list[dict[str, Any]],
    stale_after: dt.timedelta,
) -> dict[str, Any]:
    missing = list(artifacts.get("missing", []))
    candidate_reasons: list[dict[str, Any]] = []
    protection_reasons: list[str] = []
    if state == "running":
        protection_reasons.append("run_state_running")
    elif state == "waiting":
        protection_reasons.append("run_state_waiting")
    elif state == "started_no_metrics":
        protection_reasons.append("recent_unfinished_run")

    if state in {"failed", "killed"}:
        candidate_reasons.append(
            {
                "rule_id": f"run.{state}",
                "risk": "medium",
                "reason": f"Run is {state}; review failed or killed output before deleting.",
                "missing_artifacts": missing,
                "logs": _cleanup_log_refs(logs),
            }
        )
    elif state == "stale":
        candidate_reasons.append(
            {
                "rule_id": "run.stale",
                "risk": "high",
                "reason": "Run is stale and lacks completion artifacts.",
                "missing_artifacts": missing,
                "stale_after_seconds": int(stale_after.total_seconds()),
                "last_updated_at": timestamps.get("last_updated_at"),
            }
        )
    elif state == "partial":
        candidate_reasons.append(
            {
                "rule_id": "run.partial",
                "risk": "high",
                "reason": "Run has only a partial artifact set.",
                "missing_artifacts": missing,
                "logs": _cleanup_log_refs(logs),
            }
        )

    checkpoint_candidates = [
        item
        for item in checkpoints.get("retention", {}).get("items", [])
        if item.get("registry_default_candidate")
    ]
    for item in checkpoint_candidates:
        candidate_reasons.append(
            {
                "rule_id": item.get("rule_id"),
                "risk": item.get("risk", "medium"),
                "reason": item.get("retention_reason"),
                "path": item.get("path"),
            }
        )

    return {
        "protected": bool(protection_reasons),
        "protection_reasons": protection_reasons,
        "candidate_reasons": candidate_reasons,
        "missing_artifacts": missing,
        "stale_after_seconds": int(stale_after.total_seconds()),
    }


def summarize_tensorboard(run_dir: Path) -> dict[str, Any]:
    events = _tensorboard_event_paths(run_dir)
    latest = max((_mtime(path) for path in events), default=None)
    return {
        "event_count": len(events),
        "latest_event": str(events[-1]) if events else None,
        "latest_event_at": _format_dt(latest) if latest is not None else None,
    }


def collect_log_records(log_roots: Iterable[Path], *, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in log_roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in sorted(paths):
            if path.suffix not in {".log", ".out", ".err", ".txt", ".status", ""}:
                continue
            tail = _read_text_tail(path, warnings=warnings)
            failure = parse_failure_patterns(tail)
            records.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "mtime": _format_dt(_mtime(path)),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "failure": failure,
                    "tail": tail,
                }
            )
    return records


def associate_logs(run_dir: Path, *, config: dict[str, Any], logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = {
        run_dir.name.lower(),
        str(config.get("experiment_name") or "").lower(),
        str(config.get("output_run_name") or "").lower(),
    }
    tokens = {token for token in tokens if token and token != "none"}
    matches: list[tuple[int, dict[str, Any]]] = []
    for record in logs:
        haystack = f"{record.get('path', '')}\n{record.get('tail', '')}".lower()
        score = sum(1 for token in tokens if token in haystack)
        if str(run_dir).lower() in haystack:
            score += 4
        if score:
            public = {key: value for key, value in record.items() if key != "tail"}
            matches.append((score, public))
    return [record for _, record in sorted(matches, key=lambda item: (-item[0], str(item[1].get("path"))))[:5]]


def parse_failure_patterns(text: str) -> dict[str, Any]:
    if not text:
        return {"kind": None, "message": None, "waiting_for": None}
    lower = text.lower()
    if re.search(r"(^|\n)\s*(?:.+\s)?killed\s*$", text, flags=re.IGNORECASE):
        return {"kind": "killed", "message": "log contains Killed", "waiting_for": None}
    if "waiting" in lower and "checkpoint" in lower:
        wait_path = _extract_checkpoint_path(text)
        return {
            "kind": "waiting",
            "message": "log indicates waiting for checkpoint",
            "waiting_for": wait_path,
        }
    if "traceback (most recent call last)" in lower:
        return {"kind": "traceback", "message": _last_nonempty_line(text), "waiting_for": None}
    if "error conda.cli.main_run" in lower or "condaerror" in lower:
        return {"kind": "conda_failed", "message": _last_nonempty_line(text), "waiting_for": None}
    return {"kind": None, "message": None, "waiting_for": None}


def infer_run_state(
    *,
    artifacts: dict[str, Any],
    timestamps: dict[str, Any],
    sidecar: dict[str, Any] | None,
    logs: list[dict[str, Any]],
    process: dict[str, Any] | None,
    now: dt.datetime,
    stale_after: dt.timedelta = DEFAULT_STALE_AFTER,
) -> tuple[str, str]:
    log_failure = _strongest_log_failure(logs)
    if process is not None:
        wait = _process_waiting_for_checkpoint(process) or (log_failure or {}).get("waiting_for")
        if wait:
            return "waiting", f"matched process appears to be waiting for checkpoint: {wait}"
        return "running", f"matched live process pid={process.get('pid')}"

    last_updated = _parse_dt(timestamps.get("last_updated_at"))
    stale = last_updated is not None and now - last_updated > stale_after

    if sidecar:
        sidecar_state = str(sidecar.get("state", "")).lower()
        if sidecar_state == "complete":
            return "complete", "run_status.json records complete"
        if sidecar_state in {"failed", "killed", "waiting"}:
            return sidecar_state, f"run_status.json records {sidecar_state}"
        if sidecar_state == "running" and stale:
            return "stale", "run_status.json still says running but no matching process was found"
        if sidecar_state == "running":
            return "started_no_metrics", "run_status.json records running but no matching process was found"

    if _complete_artifacts_present(artifacts):
        return "complete", "metrics, logs, configs, and training outputs are present"
    if artifacts.get("test_report", {}).get("present") and artifacts.get("metrics", {}).get("present"):
        return "complete", "evaluation report and metrics are present"

    if log_failure:
        kind = log_failure.get("kind")
        if kind == "killed":
            return "killed", "associated log contains Killed"
        if kind == "waiting":
            return "waiting", f"associated log indicates waiting for checkpoint: {log_failure.get('waiting_for')}"
        if kind in {"traceback", "conda_failed"}:
            return "failed", f"associated log indicates {kind}"

    if _started_without_metrics(artifacts):
        if stale:
            return "stale", "startup/config artifacts exist but no metrics and no matching process were found"
        return "started_no_metrics", "startup/config artifacts exist but metrics are missing"
    if _any_artifact_present(artifacts):
        return "partial", "some run artifacts are present but the run is incomplete"
    return "unknown", "no recognized run artifacts were found"


def collect_python_processes() -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    records: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        cmdline = _read_proc_cmdline(entry / "cmdline")
        if not cmdline or not _looks_like_kd_process(cmdline):
            continue
        records.append(
            {
                "pid": pid,
                "cmdline": cmdline,
                "cwd": _read_proc_cwd(entry),
                "rss_mb": _read_proc_rss_mb(entry / "status"),
                "config_path": _arg_after(cmdline, "--config") or _arg_after(cmdline, "-c"),
                "output_dir": _arg_after(cmdline, "--output-dir"),
                "run_name": _override_value(cmdline, "output.run_name"),
                "kind": _process_kind(cmdline),
            }
        )
    return records


def collect_resource_snapshot(processes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    process_records = processes if processes is not None else collect_python_processes()
    gpu_snapshot = collect_gpu_snapshot()
    process_records = _attach_gpu_usage(process_records, gpu_snapshot)
    return {
        "memory": collect_memory_snapshot(),
        "gpus": gpu_snapshot,
        "processes": process_records,
    }


def collect_memory_snapshot() -> dict[str, Any]:
    meminfo = _read_meminfo()
    if not meminfo:
        return {"available": False, "reason": "/proc/meminfo unavailable"}
    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable")
    swap_total = meminfo.get("SwapTotal")
    swap_free = meminfo.get("SwapFree")
    return {
        "available": True,
        "total_mb": _kb_to_mb(total),
        "available_mb": _kb_to_mb(available),
        "used_mb": _kb_to_mb(total - available) if total is not None and available is not None else None,
        "swap_total_mb": _kb_to_mb(swap_total),
        "swap_used_mb": _kb_to_mb(swap_total - swap_free)
        if swap_total is not None and swap_free is not None
        else None,
    }


def collect_gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found", "devices": [], "processes": []}
    try:
        gpu_rows = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc), "devices": [], "processes": []}
    if gpu_rows.returncode != 0:
        return {"available": False, "reason": gpu_rows.stderr.strip() or "nvidia-smi failed", "devices": [], "processes": []}
    devices = []
    for row in gpu_rows.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 6:
            continue
        devices.append(
            {
                "index": _int_or_none(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mb": _int_or_none(parts[3]),
                "memory_used_mb": _int_or_none(parts[4]),
                "utilization_gpu_percent": _int_or_none(parts[5]),
            }
        )
    gpu_processes = _collect_gpu_processes(executable)
    return {"available": True, "reason": None, "devices": devices, "processes": gpu_processes}


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


def _matches_filters(run: dict[str, Any], filters: RunIndexFilters) -> bool:
    config = run.get("config", {})
    if filters.states and run.get("state") not in filters.states:
        return False
    if filters.dataset_family and config.get("dataset_family") != filters.dataset_family:
        return False
    if filters.objective and config.get("objective") != filters.objective:
        return False
    if filters.run_name and filters.run_name.lower() not in str(run.get("run_name", "")).lower():
        return False
    updated = _parse_dt(run.get("timestamps", {}).get("last_updated_at"))
    if filters.since and (updated is None or updated < filters.since):
        return False
    if filters.until and (updated is None or updated > filters.until):
        return False
    return True


def match_run_process(
    run_dir: Path,
    *,
    config: dict[str, Any],
    processes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    run_name = run_dir.name.lower()
    output_run_name = str(config.get("output_run_name") or "").lower()
    experiment = str(config.get("experiment_name") or "").lower()
    for process in processes:
        cmdline = str(process.get("cmdline", "")).lower()
        score = 0
        if str(run_dir).lower() in cmdline:
            score += 6
        if run_name and run_name in cmdline:
            score += 3
        if output_run_name and output_run_name in cmdline:
            score += 3
        if experiment and experiment in cmdline:
            score += 1
        proc_run_name = str(process.get("run_name") or "").lower()
        if proc_run_name and proc_run_name in {run_name, output_run_name}:
            score += 5
        output_dir = process.get("output_dir")
        if output_dir:
            try:
                if run_dir.is_relative_to(Path(output_dir).expanduser().resolve()):
                    score += 2
            except (OSError, ValueError):
                pass
        if score:
            candidates.append((score, dict(process)))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item[0], item[1].get("pid", 0)))[0][1]


def _complete_artifacts_present(artifacts: dict[str, Any]) -> bool:
    required = ("metrics", "train_log", "training_outputs", "final_config", "resolved_config")
    return all(artifacts.get(key, {}).get("present") for key in required)


def _started_without_metrics(artifacts: dict[str, Any]) -> bool:
    return (
        artifacts.get("startup_summary", {}).get("present")
        and artifacts.get("final_config", {}).get("present")
        and artifacts.get("resolved_config", {}).get("present")
        and not artifacts.get("metrics", {}).get("present")
    )


def _any_artifact_present(artifacts: dict[str, Any]) -> bool:
    for value in artifacts.values():
        if isinstance(value, dict) and value.get("present"):
            return True
    return False


def _strongest_log_failure(logs: list[dict[str, Any]]) -> dict[str, Any] | None:
    priority = {"killed": 4, "waiting": 3, "traceback": 2, "conda_failed": 1}
    failures = [
        log.get("failure")
        for log in logs
        if isinstance(log.get("failure"), dict) and log.get("failure", {}).get("kind")
    ]
    if not failures:
        return None
    return sorted(failures, key=lambda item: -priority.get(str(item.get("kind")), 0))[0]


def _process_waiting_for_checkpoint(process: dict[str, Any]) -> str | None:
    cmdline = str(process.get("cmdline", ""))
    lower = cmdline.lower()
    if "waiting" in lower and "checkpoint" in lower:
        return _extract_checkpoint_path(cmdline) or "checkpoint"
    return None


def _run_resource_summary(process: dict[str, Any] | None) -> dict[str, Any]:
    if process is None:
        return {"process_rss_mb": None, "pid": None, "gpu_indices": []}
    return {
        "process_rss_mb": process.get("rss_mb"),
        "pid": process.get("pid"),
        "gpu_indices": process.get("gpu_indices", []),
    }


def _artifact_paths(artifacts: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for value in artifacts.values():
        if not isinstance(value, dict):
            continue
        if value.get("path"):
            paths.append(Path(value["path"]))
        for raw_path in value.get("paths", []) or []:
            paths.append(Path(raw_path))
    return paths


def _run_size_bytes(*, artifacts: dict[str, Any], checkpoints: dict[str, Any]) -> int:
    paths = set(_artifact_paths(artifacts))
    for item in checkpoints.get("items", []):
        raw = item.get("path")
        if raw:
            paths.add(Path(raw))
        sidecar = item.get("sidecar_path")
        if sidecar:
            paths.add(Path(sidecar))
    return sum(_path_size_bytes(path) for path in paths)


def _directory_contains_run_marker(path: Path, filenames: Iterable[str]) -> bool:
    names = set(filenames)
    if names & DISCOVERY_FILENAMES:
        return True
    checkpoint_dir = path / "checkpoints"
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return False
    try:
        return any(_is_checkpoint(item) for item in checkpoint_dir.iterdir() if item.is_file())
    except OSError:
        return False


def _is_canonical_scan_partition_name(name: str) -> bool:
    if name in {"analysis", "visual_analysis", "evaluations", "training"}:
        return True
    if re.fullmatch(r"scene\d+", name):
        return True
    return name.startswith("scenegroup_")


def _shallow_child_may_contain_run(path: Path) -> bool:
    try:
        filenames = [item.name for item in path.iterdir() if item.is_file()]
    except OSError:
        return False
    return _directory_contains_run_marker(path, filenames)


def _prune_default_outputs_children(
    outputs_root: Path,
    current_path: Path,
    dirnames: list[str],
    *,
    warnings: list[str] | None,
) -> None:
    try:
        rel_parts = current_path.relative_to(outputs_root).parts
    except ValueError:
        return
    if not rel_parts:
        return
    partition = rel_parts[0]
    for dirname in list(dirnames):
        child = current_path / dirname
        if dirname in {"best_checkpoints", "cache", "feature_cache", "features"}:
            _remove_pruned_dirname(dirnames, dirname, child, warnings=warnings)
            continue
        if partition in {"analysis", "visual_analysis", "training"} and len(rel_parts) == 1:
            if not _shallow_child_may_contain_run(child):
                _remove_pruned_dirname(dirnames, dirname, child, warnings=warnings)
        elif partition == "evaluations" and len(rel_parts) == 2:
            if not _shallow_child_may_contain_run(child):
                _remove_pruned_dirname(dirnames, dirname, child, warnings=warnings)


def _remove_pruned_dirname(dirnames: list[str], dirname: str, path: Path, *, warnings: list[str] | None) -> None:
    if warnings is not None:
        warnings.append(f"skipped non-run outputs subtree by default: {path}")
    dirnames.remove(dirname)


def _is_discovery_artifact(path: Path) -> bool:
    return path.name in DISCOVERY_FILENAMES


def _is_checkpoint(path: Path) -> bool:
    if path.suffix == ".json" and any(path.name.endswith(suffix + ".json") for suffix in CHECKPOINT_SUFFIXES):
        return False
    return path.suffix.lower() in CHECKPOINT_SUFFIXES


def _is_tensorboard_event(path: Path) -> bool:
    return path.name.startswith("events.out.tfevents")


def _tensorboard_event_paths(run_dir: Path) -> list[Path]:
    roots = [run_dir / "tensorboard", run_dir]
    events: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for path in root.glob("events.out.tfevents*"):
                if path.is_file():
                    events.add(path)
        except OSError:
            continue
    return sorted(events, key=lambda path: path.as_posix())


def _first_present_path(artifacts: dict[str, Any], keys: Iterable[str]) -> Path | None:
    for key in keys:
        raw = artifacts.get(key, {}).get("path")
        if raw:
            return Path(raw)
    return None


def _read_yaml(path: Path | None, *, warnings: list[str] | None = None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = safe_load_yaml(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        if warnings is not None:
            warnings.append(f"failed to read YAML {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _read_json(path: Path | None, *, warnings: list[str] | None = None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if warnings is not None:
            warnings.append(f"failed to read JSON {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def _read_text_tail(path: Path, *, warnings: list[str] | None = None) -> str:
    try:
        with path.open("rb") as f:
            try:
                f.seek(-LOG_TAIL_BYTES, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode("utf-8", errors="replace")
    except OSError as exc:
        if warnings is not None:
            warnings.append(f"failed to read log {path}: {exc}")
        return ""


def _nested_dict(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(key, {})
    return cursor if isinstance(cursor, dict) else {}


def _dataset_family(dataset_cfg: dict[str, Any], run_dir: Path) -> str | None:
    raw = dataset_cfg.get("type") or dataset_cfg.get("family")
    if raw:
        return str(raw)
    lowered = run_dir.as_posix().lower()
    for candidate in ("deepsense", "mmw"):
        if candidate in lowered:
            return candidate
    return None


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = value
        elif key == "topk" and isinstance(value, dict):
            for top_key, top_value in value.items():
                if isinstance(top_value, (int, float)) and not isinstance(top_value, bool):
                    result[f"top{top_key}"] = top_value
    return result


def _primary_metric_name(metrics: dict[str, Any]) -> str | None:
    objective = metrics.get("objective") if isinstance(metrics.get("objective"), dict) else {}
    configured = objective.get("primary_metric")
    if configured and _metric_value(metrics, str(configured)) is not None:
        return str(configured)
    for name in (
        "val_adba",
        "val_beam_dba",
        "val_acc",
        "val_beam_top1",
        "loss",
        "val_loss",
        "top1",
    ):
        if _metric_value(metrics, name) is not None:
            return name
    return None


def _metric_value(metrics: dict[str, Any], name: str | None) -> float | int | None:
    if not name:
        return None
    value = metrics.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if name == "top1" and isinstance(metrics.get("topk"), dict):
        top1 = metrics["topk"].get("1") or metrics["topk"].get(1)
        if isinstance(top1, (int, float)) and not isinstance(top1, bool):
            return top1
    return None


def _best_checkpoint(paths: list[Path]) -> Path | None:
    for name in ("best.pth", "best_top1.pth", "last.pth"):
        for path in paths:
            if path.name == name:
                return path
    return paths[0] if paths else None


def _checkpoint_item(path: Path, *, best: Path | None, warnings: list[str] | None = None) -> dict[str, Any]:
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar = _read_json(sidecar_path, warnings=warnings) if sidecar_path.exists() else None
    role = _checkpoint_role(path, best=best)
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": _path_size_bytes(path),
        "mtime": _format_dt(_mtime(path)),
        "source": "run_checkpoint_dir",
        "retention_role": role,
        "selection_metadata": _checkpoint_selection_metadata(sidecar),
        "sidecar_present": sidecar is not None,
        "sidecar_path": str(sidecar_path) if sidecar is not None else None,
        "sidecar_metadata": sidecar,
        "normalization_artifacts": _checkpoint_normalization_artifacts(sidecar),
        "registry_default_candidate": role in {"recoverable_last", "duplicate_probe"},
        "registry_protected": role in {"best_reproducible", "best_top1_reproducible"},
    }


def _checkpoint_role(path: Path, *, best: Path | None) -> str:
    if path.name == "best.pth":
        return "best_reproducible"
    if path.name == "best_top1.pth":
        return "best_top1_reproducible"
    if path.name == "last.pth" and best is not None and best.name != "last.pth":
        return "recoverable_last"
    if "probe" in path.stem.lower() and best is not None:
        return "duplicate_probe"
    return "temporary_or_unclassified"


def _checkpoint_retention_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    retention_items: list[dict[str, Any]] = []
    protected_paths: list[str] = []
    candidate_paths: list[str] = []
    for item in items:
        role = str(item.get("retention_role") or "temporary_or_unclassified")
        protected = bool(item.get("registry_protected"))
        candidate = bool(item.get("registry_default_candidate"))
        rule_id = None
        risk = "medium"
        reason = "Checkpoint is not classified as a default deletion candidate."
        if role == "best_reproducible":
            reason = "Default reproducibility checkpoint; keep with sidecar metadata."
        elif role == "best_top1_reproducible":
            reason = "Default top-1 selection checkpoint; keep with sidecar metadata."
        elif role == "recoverable_last":
            rule_id = "checkpoint.last_recoverable"
            reason = "Recoverable last checkpoint; not the default reproducibility checkpoint."
        elif role == "duplicate_probe":
            rule_id = "checkpoint.duplicate_probe"
            reason = "Probe checkpoint duplicates a run that has a primary checkpoint."
        record = {
            "path": item.get("path"),
            "name": item.get("name"),
            "retention_role": role,
            "registry_protected": protected,
            "registry_default_candidate": candidate,
            "rule_id": rule_id,
            "risk": risk,
            "retention_reason": reason,
            "source": item.get("source"),
            "selection_metadata": item.get("selection_metadata"),
            "sidecar_present": item.get("sidecar_present"),
            "sidecar_path": item.get("sidecar_path"),
            "normalization_artifacts": item.get("normalization_artifacts"),
            "size_bytes": item.get("size_bytes"),
            "mtime": item.get("mtime"),
        }
        retention_items.append(record)
        if protected and item.get("path"):
            protected_paths.append(str(item["path"]))
        if candidate and item.get("path"):
            candidate_paths.append(str(item["path"]))
    return {
        "items": retention_items,
        "protected_paths": protected_paths,
        "candidate_paths": candidate_paths,
    }


def _checkpoint_selection_metadata(sidecar: dict[str, Any] | None) -> dict[str, Any]:
    if not sidecar:
        return {"available": False, "missing": True}
    keys = (
        "selection_metric",
        "selected_metric",
        "primary_metric",
        "best_metric",
        "selected_epoch",
        "best_epoch",
        "best_top1_epoch",
        "epoch",
    )
    values = {key: sidecar.get(key) for key in keys if key in sidecar}
    objective = sidecar.get("objective") if isinstance(sidecar.get("objective"), dict) else None
    if objective:
        values["objective"] = objective
    return {"available": bool(values), "missing": not bool(values), "values": values}


def _checkpoint_normalization_artifacts(sidecar: dict[str, Any] | None) -> dict[str, Any]:
    if not sidecar:
        return {"available": False, "paths": []}
    raw = sidecar.get("normalization_artifacts")
    if raw is None and isinstance(sidecar.get("metadata"), dict):
        raw = sidecar["metadata"].get("normalization_artifacts")
    if not isinstance(raw, dict):
        return {"available": False, "paths": []}
    paths = [str(value) for value in raw.values() if isinstance(value, (str, Path))]
    return {"available": bool(paths), "paths": paths, "raw": raw}


def _cleanup_log_refs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for log in logs:
        refs.append(
            {
                "path": log.get("path"),
                "failure": log.get("failure"),
                "mtime": log.get("mtime"),
                "size_bytes": log.get("size_bytes"),
            }
        )
    return refs


def _path_size_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            total = 0
            for item in path.rglob("*"):
                try:
                    if item.is_file():
                        total += int(item.stat().st_size)
                except OSError:
                    continue
            return total
    except OSError:
        return 0
    return 0


def _looks_like_kd_process(cmdline: str) -> bool:
    lower = cmdline.lower()
    if "kd_sensing.cli.train" in lower or "kd_sensing.cli.evaluate" in lower:
        return True
    if "kd-sensing-train" in lower or "kd-sensing-evaluate" in lower:
        return True
    return False


def _process_kind(cmdline: str) -> str:
    lower = cmdline.lower()
    if "evaluate" in lower:
        return "evaluation"
    return "training"


def _read_proc_cmdline(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return " ".join(part for part in raw.decode("utf-8", errors="replace").split("\0") if part)


def _read_proc_cwd(proc_dir: Path) -> str | None:
    try:
        return str((proc_dir / "cwd").resolve())
    except OSError:
        return None


def _read_proc_rss_mb(path: Path) -> float | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024, 3)
    except (OSError, ValueError):
        return None
    return None


def _arg_after(cmdline: str, flag: str) -> str | None:
    parts = cmdline.split()
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(flag + "="):
            return part.split("=", 1)[1]
    return None


def _override_value(cmdline: str, key: str) -> str | None:
    for part in cmdline.split():
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return None


def _read_meminfo() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            amount = rest.strip().split()
            if amount:
                values[key] = int(amount[0])
    except (OSError, ValueError):
        return {}
    return values


def _collect_gpu_processes(executable: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    records = []
    for row in result.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            continue
        records.append({"pid": _int_or_none(parts[0]), "gpu_uuid": parts[1], "memory_used_mb": _int_or_none(parts[2])})
    return records


def _attach_gpu_usage(processes: list[dict[str, Any]], gpu_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if not processes or not gpu_snapshot.get("available"):
        return processes
    uuid_to_index = {
        device.get("uuid"): device.get("index")
        for device in gpu_snapshot.get("devices", [])
        if device.get("uuid") is not None
    }
    usage_by_pid: dict[int, list[dict[str, Any]]] = {}
    for item in gpu_snapshot.get("processes", []):
        pid = item.get("pid")
        if pid is None:
            continue
        usage = dict(item)
        usage["gpu_index"] = uuid_to_index.get(item.get("gpu_uuid"))
        usage_by_pid.setdefault(int(pid), []).append(usage)
    enriched = []
    for process in processes:
        copy = dict(process)
        usage = usage_by_pid.get(int(copy.get("pid", -1)), [])
        copy["gpu_usage"] = usage
        copy["gpu_indices"] = [item.get("gpu_index") for item in usage if item.get("gpu_index") is not None]
        enriched.append(copy)
    return enriched


def _kb_to_mb(value: int | None) -> float | None:
    return round(value / 1024, 3) if value is not None else None


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _empty_resources(processes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "memory": {"available": False, "reason": "resource snapshot disabled"},
        "gpus": {"available": False, "reason": "resource snapshot disabled", "devices": [], "processes": []},
        "processes": processes or [],
    }


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


def _format_dt(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def _parse_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _mtime(path: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    except OSError:
        return None


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _ensure_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _as_list(value: str | Path | Iterable[str | Path] | None) -> list[str | Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _resolve_existing_or_requested(path: str | Path) -> Path:
    resolved = resolve_path(path)
    return resolved if resolved is not None else Path(path).expanduser().resolve()


def _filters_to_dict(filters: RunIndexFilters) -> dict[str, Any]:
    return {
        "states": list(filters.states),
        "dataset_family": filters.dataset_family,
        "objective": filters.objective,
        "run_name": filters.run_name,
        "since": _format_dt(filters.since),
        "until": _format_dt(filters.until),
    }


def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()[:300]
    return None


def _extract_checkpoint_path(text: str) -> str | None:
    match = re.search(r"([^\s'\"]+\.(?:pth|pt|ckpt))", text)
    return match.group(1) if match else None
