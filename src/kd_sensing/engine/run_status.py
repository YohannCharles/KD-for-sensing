from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import traceback
from typing import Any

RUN_STATUS_FILENAME = "run_status.json"
_ACTIVE_RUN_DIRS: dict[int, Path] = {}


def register_active_run(cfg: dict, run_dir: str | Path) -> None:
    _ACTIVE_RUN_DIRS[id(cfg)] = Path(run_dir)


def active_run_dir(cfg: dict) -> Path | None:
    return _ACTIVE_RUN_DIRS.get(id(cfg))


def write_running_status(
    run_dir: str | Path,
    cfg: dict,
    *,
    kind: str,
    config_path: str | None = None,
    started_at: dt.datetime | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    register_active_run(cfg, run_path)
    now = _utc_now() if started_at is None else _ensure_utc(started_at)
    payload = _base_payload(cfg, run_path, kind=kind, state="running")
    payload.update(
        {
            "start_time": _format_dt(now),
            "updated_at": _format_dt(now),
            "config_path": config_path or _config_path_from_cfg(cfg),
        }
    )
    _write_status(run_path, payload)
    return payload


def write_complete_status(
    run_dir: str | Path,
    cfg: dict,
    *,
    kind: str,
    primary_metric: dict[str, Any] | None = None,
    metrics_path: str | Path | None = None,
    best_checkpoint: str | Path | None = None,
    completed_at: dt.datetime | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    now = _utc_now() if completed_at is None else _ensure_utc(completed_at)
    previous = _read_status(run_path)
    start_time = _parse_dt(previous.get("start_time")) if previous else None
    config_path = (previous.get("config_path") or _config_path_from_cfg(cfg)) if previous else _config_path_from_cfg(cfg)
    payload = _base_payload(cfg, run_path, kind=kind, state="complete")
    payload.update(
        {
            "start_time": previous.get("start_time") if previous else None,
            "end_time": _format_dt(now),
            "updated_at": _format_dt(now),
            "duration_seconds": round((now - start_time).total_seconds(), 3) if start_time is not None else None,
            "primary_metric": primary_metric,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
            "config_path": config_path,
        }
    )
    _write_status(run_path, payload)
    return payload


def write_failed_status(
    run_dir: str | Path,
    cfg: dict,
    exc: BaseException,
    *,
    kind: str,
    failed_at: dt.datetime | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    now = _utc_now() if failed_at is None else _ensure_utc(failed_at)
    previous = _read_status(run_path)
    start_time = _parse_dt(previous.get("start_time")) if previous else None
    config_path = (previous.get("config_path") or _config_path_from_cfg(cfg)) if previous else _config_path_from_cfg(cfg)
    payload = _base_payload(cfg, run_path, kind=kind, state="failed")
    payload.update(
        {
            "start_time": previous.get("start_time") if previous else None,
            "end_time": _format_dt(now),
            "updated_at": _format_dt(now),
            "duration_seconds": round((now - start_time).total_seconds(), 3) if start_time is not None else None,
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exception_only(type(exc), exc)[-1].strip(),
            },
            "log_path": str(log_path) if log_path is not None else None,
            "config_path": config_path,
        }
    )
    _write_status(run_path, payload)
    return payload


def write_failed_status_for_active_run(cfg: dict, exc: BaseException, *, kind: str) -> dict[str, Any] | None:
    run_dir = active_run_dir(cfg)
    if run_dir is None:
        return None
    return write_failed_status(run_dir, cfg, exc, kind=kind)


def _base_payload(cfg: dict, run_dir: Path, *, kind: str, state: str) -> dict[str, Any]:
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    model_cfg = _nested_dict(cfg, "model")
    model_primary = _nested_dict(cfg, "model", "primary")
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    objective_meta = runtime.get("prediction_objective") if isinstance(runtime.get("prediction_objective"), dict) else {}
    modalities = model_primary.get("modalities") or model_cfg.get("modalities")
    if modalities is None and experiment.get("task") in {"image", "radar", "gps", "lidar", "mmwave", "csi"}:
        modalities = [experiment["task"]]
    return {
        "schema_version": 1,
        "state": state,
        "kind": kind,
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "experiment_name": experiment.get("name"),
        "task": experiment.get("task"),
        "objective": experiment.get("objective") or objective_meta.get("name"),
        "enabled_modalities": _modalities_list(modalities),
        "seed": experiment.get("seed"),
    }


def _write_status(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / RUN_STATUS_FILENAME
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def _read_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_STATUS_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _config_path_from_cfg(cfg: dict) -> str | None:
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    value = runtime.get("cli_config_path") or runtime.get("config_path")
    return str(value) if value is not None else None


def _nested_dict(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(key, {})
    return cursor if isinstance(cursor, dict) else {}


def _modalities_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return [value]


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _ensure_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _format_dt(value: dt.datetime) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def _parse_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
