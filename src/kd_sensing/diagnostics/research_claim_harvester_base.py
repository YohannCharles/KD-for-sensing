from dataclasses import asdict, dataclass, field
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from kd_sensing.config.parsing import safe_load_yaml


SCHEMA_VERSION = 1
DEFAULT_LEDGER_DIR = Path("outputs/analysis/research_ledger")
STRICT_FIELDS = (
    "split",
    "sample_count",
    "label_space",
    "metric_profile",
    "target_source",
    "difficulty_digest",
    "seed",
    "run_family",
)
CONSISTENCY_FIELDS = (
    "split",
    "sample_count",
    "label_space",
    "metric_profile",
    "target_source",
    "difficulty_digest",
    "run_family",
)
IDENTITY_FIELDS = {
    "run",
    "run_id",
    "run_name",
    "name",
    "method",
    "model",
    "group",
    "family",
    "run_family",
    "seed",
    "pattern",
    "missing_pattern",
    "condition",
    "split",
    "sample_count",
    "samples",
    "n",
    "label_space",
    "metric_profile",
    "metrics_profile",
    "target_source",
    "beam_target_source",
    "difficulty_digest",
    "config_path",
    "config_digest",
    "artifact_path",
}


@dataclass(frozen=True)
class ComparabilityWarning:
    field: str
    kind: str
    expected: Any = None
    actual: Any = None
    severity: str = "needs_review"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ClaimCandidate:
    candidate_id: str
    source_type: str
    run_id: str
    run_name: str | None
    method: str | None
    seed: Any
    pattern: str | None
    metrics: dict[str, Any]
    sample_count: Any
    split: Any
    label_space: Any
    metric_profile: Any
    target_source: Any
    difficulty_digest: Any
    run_family: Any
    artifact_paths: dict[str, Any]
    generated_at: str
    config_path: str | None = None
    config_digest: str | None = None
    scene_scope: str | None = None
    checkpoint_provenance: dict[str, Any] = field(default_factory=dict)
    comparability_status: str = "needs_review"
    claim_status: str = "draft"
    candidate_only: bool = True
    warnings: list[dict[str, Any]] = field(default_factory=list)
    next_action_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class LedgerRecord:
    run_id: str
    run_name: str | None
    config_path: str | None
    config_digest: str | None
    seed: Any
    scene_scope: str | None
    artifact_paths: dict[str, Any]
    metric_summary: dict[str, Any]
    claim_status: str
    comparability_status: str
    caveat: str
    generated_at: str
    candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class DashboardSummary:
    metadata: dict[str, Any]
    active_changes: list[dict[str, Any]]
    run_state_counts: dict[str, int]
    resources: dict[str, Any]
    claim_counts: dict[str, int]
    candidates: list[dict[str, Any]]
    upgradable_candidates: list[dict[str, Any]]
    next_action_hints: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _read_yaml(path: Path | None, *, warnings: list[str] | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = safe_load_yaml(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        if warnings is not None:
            warnings.append(f"failed to read YAML {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}

def _read_json(path: Path | None, *, warnings: list[str] | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if warnings is not None:
            warnings.append(f"failed to read JSON {path}: {exc}")
        return None

def _dict_at(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(key, {})
    return cursor if isinstance(cursor, dict) else {}

def _metric_fields(row: dict[str, Any]) -> dict[str, Any]:
    if "metric_name" in row and "metric_value" in row:
        return {str(row["metric_name"]): _coerce(row["metric_value"])}
    metrics = {}
    for key, value in row.items():
        if key.lower() in IDENTITY_FIELDS:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = value
    return metrics

def _numeric_items(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("metrics"), dict):
        data = data["metrics"]
    return {
        key: value
        for key, value in data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

def _primary_metric(metrics: dict[str, Any], run_status: dict[str, Any]) -> dict[str, Any]:
    primary = run_status.get("primary_metric") if isinstance(run_status.get("primary_metric"), dict) else {}
    if primary.get("name") and primary.get("value") is not None:
        return {"name": primary["name"], "value": primary["value"]}
    scalars = _numeric_items(metrics)
    for key in ("val_adba", "val_beam_dba", "val_acc", "val_beam_top1", "top1", "loss", "val_loss"):
        if key in scalars:
            return {"name": key, "value": scalars[key]}
    name, value = next(iter(scalars.items()), (None, None))
    return {"name": name, "value": value}

def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None

def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None

def _coerce(value: Any) -> Any:
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

def _missing(value: Any) -> bool:
    return value is None or value == "" or value == []

def _canonical_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)

def _candidate_id(*parts: Any) -> str:
    return "candidate-" + _stable_digest("|".join(json.dumps(part, sort_keys=True, default=str) for part in parts))

def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

def _file_digest(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None

def _dataset_family_from_path(path: Path) -> str | None:
    lowered = path.as_posix().lower()
    for candidate in ("deepsense", "mmw", "scene31"):
        if candidate in lowered:
            return "deepsense6g" if candidate == "scene31" else candidate
    return None

def _run_name_from_artifact(path: Path) -> str:
    return re.sub(r"_missing_patterns$", "", path.stem)

def _method_from_run_name(run_name: Any) -> str | None:
    if run_name is None:
        return None
    text = str(run_name)
    return re.sub(r"_?seed\d+.*$", "", text)

def _seed_from_text(text: str) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", text)
    return int(match.group(1)) if match else None

def _as_optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)

def _as_paths(value: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value).expanduser().resolve()]
    return [Path(item).expanduser().resolve() for item in value]

def _format_dt(value: dt.datetime) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")

def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _ensure_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)
