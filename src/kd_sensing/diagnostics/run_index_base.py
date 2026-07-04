from dataclasses import dataclass
import datetime as dt
from pathlib import Path
from typing import Any, Iterable

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
    "metrics_csv": "metrics.csv",
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
