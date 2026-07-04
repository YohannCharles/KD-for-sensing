import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.utils.runtime_output_layout import output_layout_summary

from kd_sensing.diagnostics.run_index_base import (
    DEFAULT_STALE_AFTER,
    RUN_ARTIFACT_NAMES,
    RUN_STATUS_FILENAME,
    _ensure_utc,
    _format_dt,
    _parse_dt,
    _mtime,
    _utc_now,
)
from kd_sensing.diagnostics.run_index_scanner import (
    _extract_checkpoint_path,
    _is_checkpoint,
    _tensorboard_event_paths,
    associate_logs,
)
from kd_sensing.diagnostics.run_index_resources import (
    _public_process,
    _run_resource_summary,
    match_run_process,
)


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
        "claim_harvester": summarize_claim_harvester_fields(
            path,
            config=config,
            artifacts=artifacts,
            metrics=metrics,
            checkpoints=checkpoints,
            process=process,
        ),
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
    eval_artifact_paths = sorted(
        path
        for path in run_dir.rglob("*_missing_patterns.*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    )
    eval_artifacts = [_artifact_ref(path, artifact_type="missing_patterns") for path in eval_artifact_paths]
    tensorboard_events = [str(path) for path in _tensorboard_event_paths(run_dir)]
    items["checkpoints"] = {"present": bool(checkpoint_files), "paths": checkpoint_files}
    items["checkpoint_sidecars"] = {"present": bool(checkpoint_sidecars), "paths": checkpoint_sidecars}
    items["eval_artifacts"] = {"present": bool(eval_artifacts), "items": eval_artifacts}
    items["tensorboard_events"] = {"present": bool(tensorboard_events), "paths": tensorboard_events}
    if items.get("metrics_csv", {}).get("present") and "metrics.json" in missing:
        missing.remove("metrics.json")
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
        "config_digest": _file_digest(config_path),
        "dataset_family": _dataset_family(dataset_cfg, run_dir),
        "experiment_name": experiment.get("name"),
        "task": experiment.get("task"),
        "objective": experiment.get("objective") or objective_meta.get("name"),
        "modalities": list(modalities) if isinstance(modalities, (list, tuple)) else ([] if modalities is None else [modalities]),
        "seed": experiment.get("seed"),
        "output_run_name": _nested_dict(cfg, "output").get("run_name"),
        "scene": runtime.get("scene"),
        "scene_scope": runtime.get("scene_scope") or runtime.get("output_scope"),
        "split": _first_value(runtime, dataset_cfg, cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}, keys=("split",)),
        "sample_count": _first_value(
            runtime,
            dataset_cfg,
            cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {},
            keys=("sample_count", "num_samples", "effective_num_samples"),
        ),
        "label_space": _first_value(runtime, dataset_cfg, objective_meta, keys=("label_space", "target_label_space")),
        "metric_profile": _first_value(
            runtime,
            objective_meta,
            cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {},
            keys=("metric_profile", "metrics_profile", "primary_metric"),
        ),
        "target_source": _first_value(
            runtime,
            dataset_cfg,
            objective_meta,
            keys=("target_source", "beam_target_source", "target_beam_source"),
        ),
        "difficulty_digest": _first_value(runtime, dataset_cfg, objective_meta, keys=("difficulty_digest",)),
    }

def summarize_metrics(run_dir: Path, *, artifacts: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    metrics_path = Path(artifacts["metrics"]["path"]) if artifacts.get("metrics", {}).get("path") else None
    metrics_csv_path = (
        Path(artifacts["metrics_csv"]["path"]) if artifacts.get("metrics_csv", {}).get("path") else None
    )
    report_path = Path(artifacts["test_report"]["path"]) if artifacts.get("test_report", {}).get("path") else None
    raw: dict[str, Any] = {}
    source = None
    if metrics_path is not None:
        raw = _read_json(metrics_path, warnings=warnings) or {}
        source = metrics_path
    elif metrics_csv_path is not None:
        raw = _read_metrics_csv(metrics_csv_path, warnings=warnings)
        source = metrics_csv_path
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

def summarize_claim_harvester_fields(
    run_dir: Path,
    *,
    config: dict[str, Any],
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    checkpoints: dict[str, Any],
    process: dict[str, Any] | None,
) -> dict[str, Any]:
    """Expose stable read-only fields consumed by the research claim harvester."""

    artifact_paths = {
        key: value.get("path")
        for key, value in artifacts.items()
        if isinstance(value, dict) and value.get("path")
    }
    artifact_paths["checkpoints"] = list(checkpoints.get("paths", []))
    checkpoint_items = checkpoints.get("items", [])
    primary_checkpoint = checkpoints.get("primary_checkpoint")
    primary_sidecar = next(
        (item for item in checkpoint_items if item.get("path") == primary_checkpoint and item.get("sidecar_path")),
        None,
    )
    return {
        "run_id": _stable_digest(str(run_dir)),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config_path": config.get("config_path"),
        "config_digest": config.get("config_digest"),
        "seed": config.get("seed"),
        "scene_scope": config.get("scene_scope") or config.get("scene"),
        "dataset_family": config.get("dataset_family"),
        "split": config.get("split"),
        "sample_count": config.get("sample_count"),
        "label_space": config.get("label_space"),
        "metric_profile": config.get("metric_profile") or metrics.get("primary", {}).get("name"),
        "target_source": config.get("target_source"),
        "difficulty_digest": config.get("difficulty_digest"),
        "artifact_paths": artifact_paths,
        "eval_artifacts": list(artifacts.get("eval_artifacts", {}).get("items", [])),
        "checkpoint_provenance": {
            "checkpoint_path": primary_checkpoint,
            "sidecar_path": primary_sidecar.get("sidecar_path") if primary_sidecar else None,
            "selection_metric": (primary_sidecar or {}).get("selection_metadata", {}).get("values", {}).get("selection_metric"),
            "source": "run_local_checkpoint" if primary_checkpoint else "unavailable",
        },
        "active_process": _public_process(process),
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

def _complete_artifacts_present(artifacts: dict[str, Any]) -> bool:
    required = ("train_log", "training_outputs", "final_config", "resolved_config")
    has_metrics = artifacts.get("metrics", {}).get("present") or artifacts.get("metrics_csv", {}).get("present")
    return bool(has_metrics) and all(artifacts.get(key, {}).get("present") for key in required)

def _started_without_metrics(artifacts: dict[str, Any]) -> bool:
    return (
        artifacts.get("startup_summary", {}).get("present")
        and artifacts.get("final_config", {}).get("present")
        and artifacts.get("resolved_config", {}).get("present")
        and not artifacts.get("metrics", {}).get("present")
        and not artifacts.get("metrics_csv", {}).get("present")
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

def _read_metrics_csv(path: Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        if warnings is not None:
            warnings.append(f"failed to read CSV {path}: {exc}")
        return {}
    if not rows:
        return {}
    row = rows[-1]
    return {
        key: _coerce_scalar(value)
        for key, value in row.items()
        if key is not None and value not in (None, "")
    }

def _nested_dict(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(key, {})
    return cursor if isinstance(cursor, dict) else {}

def _first_value(*dicts: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for data in dicts:
        if not isinstance(data, dict):
            continue
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value
    return None

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

def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        number = float(text)
    except ValueError:
        return value
    if number.is_integer() and not any(marker in text.lower() for marker in (".", "e")):
        return int(number)
    return number

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

def _artifact_ref(path: Path, *, artifact_type: str) -> dict[str, Any]:
    return {
        "type": artifact_type,
        "path": str(path),
        "mtime": _format_dt(_mtime(path)),
        "size_bytes": _path_size_bytes(path),
    }

def _file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()[:16]

def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
