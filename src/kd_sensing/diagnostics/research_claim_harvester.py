from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.diagnostics.run_index import build_run_index


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


def harvest_research_claims(
    scan_roots: str | Path | Iterable[str | Path] = "outputs",
    *,
    run_index: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Read local artifacts and return draft claim candidates."""

    generated_at = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    roots = _as_paths(scan_roots)
    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []

    for path in _missing_pattern_artifacts(roots):
        candidates.extend(read_scene31_missing_pattern_artifact(path, generated_at=generated_at, warnings=warnings))

    index = run_index
    if index is None:
        index = build_run_index(outputs=roots, logs=None, include_resources=False, include_legacy_containers=True, now=now)
        warnings.extend(index.get("warnings", []))
    for record in run_index_records_for_harvester(index):
        run_dir = record.get("run_dir")
        if run_dir:
            candidate = training_run_claim_candidate(Path(run_dir), generated_at=generated_at, run_index_record=record, warnings=warnings)
            if candidate is not None:
                candidates.append(candidate)

    candidates = apply_strict_comparability_gate(candidates)
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "scan_roots": [str(path) for path in roots],
            "candidate_only": True,
            "sqlite_backend": {"status": "deferred", "reason": "JSONL/CSV are the first supported local ledgers."},
        },
        "candidates": candidates,
        "warnings": warnings,
    }


def read_scene31_missing_pattern_artifact(
    path: str | Path,
    *,
    generated_at: str | None = None,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    source = Path(path)
    generated = generated_at or _format_dt(_utc_now())
    rows = _read_missing_rows(source, warnings=warnings)
    candidates = [
        _candidate_from_missing_row(row, source=source, generated_at=generated)
        for row in rows
    ]
    return [candidate.to_dict() for candidate in candidates]


def read_training_run_artifact(run_dir: str | Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    path = Path(run_dir).expanduser().resolve()
    config_path = _first_existing(path / "final_config.yaml", path / "resolved_config.yaml")
    cfg = _read_yaml(config_path, warnings=warnings)
    metrics_path = _first_existing(path / "metrics.json", path / "metrics.csv")
    metrics_raw = _read_metrics(metrics_path, warnings=warnings)
    train_log = _read_json(path / "train_log.json", warnings=warnings) or {}
    run_status = _read_json(path / "run_status.json", warnings=warnings) or {}
    checkpoint_provenance = _checkpoint_provenance(path, run_status=run_status, warnings=warnings)
    config = _config_summary(cfg, run_dir=path, config_path=config_path)
    primary = _primary_metric(metrics_raw, run_status)
    return {
        "run_id": _stable_digest(str(path)),
        "run_name": path.name,
        "run_dir": str(path),
        "config": config,
        "metrics": {
            "path": str(metrics_path) if metrics_path else None,
            "raw": metrics_raw,
            "scalars": _numeric_items(metrics_raw),
            "primary": primary,
        },
        "train_log": {"path": str(path / "train_log.json") if (path / "train_log.json").exists() else None, "raw": train_log},
        "run_status": run_status,
        "checkpoint_provenance": checkpoint_provenance,
        "artifact_paths": _run_artifact_paths(path, metrics_path=metrics_path, config_path=config_path, checkpoint=checkpoint_provenance),
    }


def training_run_claim_candidate(
    run_dir: str | Path,
    *,
    generated_at: str,
    run_index_record: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any] | None:
    record = read_training_run_artifact(run_dir, warnings=warnings)
    metrics = record["metrics"]["scalars"]
    if not metrics:
        return None
    config = dict(record["config"])
    if run_index_record:
        config = {**config, **_claim_config_from_run_index(run_index_record)}
    provenance = record["checkpoint_provenance"]
    candidate_warnings: list[dict[str, Any]] = []
    if provenance.get("status") != "complete":
        candidate_warnings.append(
            ComparabilityWarning(
                field="checkpoint_provenance",
                kind="provenance_incomplete",
                severity="needs_review",
                message="Checkpoint sidecar or selected checkpoint provenance is incomplete.",
            ).to_dict()
        )
    method = config.get("output_run_name") or record.get("run_name")
    candidate = ClaimCandidate(
        candidate_id=_candidate_id(record.get("run_id"), method, None, record["artifact_paths"]),
        source_type="training_run",
        run_id=str(record.get("run_id")),
        run_name=record.get("run_name"),
        method=str(method) if method is not None else None,
        seed=config.get("seed"),
        pattern=None,
        metrics=metrics,
        sample_count=config.get("sample_count"),
        split=config.get("split"),
        label_space=config.get("label_space"),
        metric_profile=config.get("metric_profile") or record["metrics"]["primary"].get("name"),
        target_source=config.get("target_source"),
        difficulty_digest=config.get("difficulty_digest"),
        run_family=config.get("run_family") or config.get("dataset_family") or method,
        artifact_paths=record["artifact_paths"],
        generated_at=generated_at,
        config_path=config.get("config_path"),
        config_digest=config.get("config_digest"),
        scene_scope=config.get("scene_scope"),
        checkpoint_provenance=provenance,
    )
    candidate.warnings = [*_required_warnings(candidate.to_dict()), *candidate_warnings]
    candidate.next_action_hints = _candidate_next_actions(candidate.warnings)
    return candidate.to_dict()


def apply_strict_comparability_gate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    output = [dict(candidate, warnings=list(candidate.get("warnings", []))) for candidate in candidates]
    for candidate in output:
        grouped[(candidate.get("method"), candidate.get("pattern"), candidate.get("run_family"))].append(candidate)

    for group in grouped.values():
        for field_name in STRICT_FIELDS:
            for candidate in group:
                if _missing(candidate.get(field_name)):
                    _append_warning(
                        candidate,
                        ComparabilityWarning(
                            field=field_name,
                            kind="required_missing",
                            severity="needs_review",
                            message=f"Strict comparability requires {field_name}.",
                        ),
                    )
        for field_name in CONSISTENCY_FIELDS:
            observed = [_canonical_value(candidate.get(field_name)) for candidate in group if not _missing(candidate.get(field_name))]
            if len(set(observed)) <= 1:
                continue
            expected = observed[0]
            for candidate in group:
                actual = _canonical_value(candidate.get(field_name))
                _append_warning(
                    candidate,
                    ComparabilityWarning(
                        field=field_name,
                        kind="field_conflict",
                        expected=expected,
                        actual=actual,
                        severity="not_comparable",
                        message=f"{field_name} differs within the candidate group.",
                    ),
                )
        seeds = [str(candidate.get("seed")) for candidate in group if not _missing(candidate.get("seed"))]
        duplicate_seeds = {seed for seed, count in Counter(seeds).items() if count > 1}
        if duplicate_seeds and len(group) > 1:
            for candidate in group:
                if str(candidate.get("seed")) in duplicate_seeds:
                    _append_warning(
                        candidate,
                        ComparabilityWarning(
                            field="seed",
                            kind="duplicate_seed",
                            actual=candidate.get("seed"),
                            severity="needs_review",
                            message="Multiple candidates in this group share the same seed.",
                        ),
                    )
        for candidate in group:
            severities = {warning.get("severity") for warning in candidate.get("warnings", [])}
            if "not_comparable" in severities:
                candidate["comparability_status"] = "not_comparable"
            elif "needs_review" in severities:
                candidate["comparability_status"] = "needs_review"
            else:
                candidate["comparability_status"] = "strict"
            candidate["next_action_hints"] = _candidate_next_actions(candidate.get("warnings", []))
    return output


def run_index_records_for_harvester(index: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in index.get("runs", []):
        claim = run.get("claim_harvester") if isinstance(run.get("claim_harvester"), dict) else {}
        records.append(
            {
                **claim,
                "run_dir": claim.get("run_dir") or run.get("run_dir"),
                "run_name": claim.get("run_name") or run.get("run_name"),
                "state": run.get("state"),
                "eval_artifacts": claim.get("eval_artifacts", []),
                "active_process": claim.get("active_process") or run.get("process"),
            }
        )
    return records


def ledger_records_from_candidates(candidates: Iterable[dict[str, Any]], *, generated_at: str | None = None) -> list[dict[str, Any]]:
    generated = generated_at or _format_dt(_utc_now())
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        warnings = candidate.get("warnings", [])
        caveat = "; ".join(warning.get("message") or warning.get("kind", "") for warning in warnings if warning) or "draft candidate"
        records.append(
            LedgerRecord(
                run_id=str(candidate.get("run_id") or candidate.get("candidate_id")),
                run_name=candidate.get("run_name"),
                config_path=candidate.get("config_path"),
                config_digest=candidate.get("config_digest"),
                seed=candidate.get("seed"),
                scene_scope=candidate.get("scene_scope"),
                artifact_paths=dict(candidate.get("artifact_paths", {})),
                metric_summary=dict(candidate.get("metrics", {})),
                claim_status=str(candidate.get("claim_status") or "draft"),
                comparability_status=str(candidate.get("comparability_status") or "needs_review"),
                caveat=caveat,
                generated_at=generated,
                candidate_id=candidate.get("candidate_id"),
            ).to_dict()
        )
    return records


def write_jsonl_ledger(
    records: Iterable[dict[str, Any]],
    *,
    ledger_dir: str | Path = DEFAULT_LEDGER_DIR,
    now: dt.datetime | None = None,
) -> Path:
    generated = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    stamp = generated.replace("-", "").replace(":", "").replace("+00:00", "Z").replace(".", "_")
    target_dir = Path(ledger_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"research_ledger_{stamp}.jsonl"
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return target


def write_ledger_csv(records: Iterable[dict[str, Any]], *, output_path: str | Path) -> Path:
    rows = list(records)
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "config_path",
        "config_digest",
        "seed",
        "scene_scope",
        "metric_summary",
        "claim_status",
        "comparability_status",
        "caveat",
        "generated_at",
        "candidate_id",
    ]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            copy = dict(row)
            copy["metric_summary"] = json.dumps(copy.get("metric_summary", {}), sort_keys=True)
            writer.writerow({key: copy.get(key) for key in fieldnames})
    return target


def build_dashboard_summary(
    *,
    project_root: str | Path = ".",
    outputs: str | Path | Iterable[str | Path] = "outputs",
    logs: str | Path | Iterable[str | Path] | None = "logs",
    scan_roots: str | Path | Iterable[str | Path] | None = None,
    include_resources: bool = True,
    stale_after: dt.timedelta | None = None,
    now: dt.datetime | None = None,
    run_index: dict[str, Any] | None = None,
    active_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generated = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    index = run_index or build_run_index(
        outputs=outputs,
        logs=logs,
        include_resources=include_resources,
        stale_after=stale_after or dt.timedelta(hours=12),
        now=now,
    )
    harvest = harvest_research_claims(scan_roots or outputs, run_index=index, now=now)
    candidates = harvest["candidates"]
    state_counts = Counter(str(run.get("state") or "unknown") for run in index.get("runs", []))
    claim_counts = Counter(str(candidate.get("comparability_status") or "needs_review") for candidate in candidates)
    changes = active_changes if active_changes is not None else collect_active_openspec_changes(project_root)
    upgradable = [
        _candidate_brief(candidate)
        for candidate in candidates
        if candidate.get("comparability_status") == "strict"
    ][:10]
    hints = _dashboard_hints(index=index, candidates=candidates)
    summary = DashboardSummary(
        metadata={
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated,
            "project_root": str(Path(project_root).expanduser().resolve()),
            "candidate_only": True,
            "read_only": True,
            "does_not_update_claim_registry": True,
        },
        active_changes=changes,
        run_state_counts=dict(sorted(state_counts.items())),
        resources=_resource_summary(index.get("resources", {})),
        claim_counts=dict(sorted(claim_counts.items())),
        candidates=candidates,
        upgradable_candidates=upgradable,
        next_action_hints=hints,
        warnings=list(index.get("warnings", [])) + list(harvest.get("warnings", [])),
    )
    return summary.to_dict()


def render_dashboard_summary(summary: dict[str, Any]) -> str:
    active = summary.get("active_changes", [])
    active_text = ", ".join(str(item.get("name") or item.get("changeName")) for item in active) if active else "none"
    run_counts = ", ".join(f"{key}={value}" for key, value in summary.get("run_state_counts", {}).items()) or "none"
    claim_counts = ", ".join(f"{key}={value}" for key, value in summary.get("claim_counts", {}).items()) or "none"
    resources = summary.get("resources", {})
    lines = [
        "daily research dashboard",
        f"generated_at: {summary.get('metadata', {}).get('generated_at')}",
        f"active_changes: {active_text}",
        f"runs: {run_counts}",
        f"resources: gpus={resources.get('gpu_count', 0)} processes={resources.get('process_count', 0)}",
        f"claim_candidates: {claim_counts}",
        f"upgradable_candidates: {len(summary.get('upgradable_candidates', []))}",
        "candidate_only: true",
    ]
    hints = summary.get("next_action_hints", [])
    if hints:
        lines.append("next_actions:")
        lines.extend(f"- {hint}" for hint in hints[:8])
    return "\n".join(lines)


def collect_active_openspec_changes(project_root: str | Path = ".") -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["openspec", "list", "--json"],
            cwd=Path(project_root),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    raw_changes = payload.get("changes", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_changes, list):
        return []
    changes = []
    for item in raw_changes:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("changeName") or item.get("id")
        if not name:
            continue
        if item.get("status") in {"archived", "complete"}:
            continue
        changes.append({"name": name, "status": item.get("status") or item.get("state") or "active"})
    return changes


def _read_missing_rows(path: Path, *, warnings: list[str] | None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except OSError as exc:
            if warnings is not None:
                warnings.append(f"failed to read CSV {path}: {exc}")
            return []
    data = _read_json(path, warnings=warnings)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []
    metadata = {key: value for key, value in data.items() if key not in {"rows", "records", "results", "missing_patterns", "patterns"}}
    for key in ("rows", "records", "results", "missing_patterns", "patterns"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [{**metadata, **row} for row in rows if isinstance(row, dict)]
    return [data]


def _candidate_from_missing_row(row: dict[str, Any], *, source: Path, generated_at: str) -> ClaimCandidate:
    normalized = {str(key): _coerce(value) for key, value in row.items()}
    run_name = _first(normalized, "run_name", "run", "name") or _run_name_from_artifact(source)
    method = _first(normalized, "method", "model", "group") or _method_from_run_name(run_name)
    seed = _first(normalized, "seed") or _seed_from_text(str(run_name))
    pattern = _first(normalized, "pattern", "missing_pattern", "condition")
    metrics = _metric_fields(normalized)
    candidate_warnings: list[dict[str, Any]] = []
    candidate = ClaimCandidate(
        candidate_id=_candidate_id(str(source), run_name, pattern, {"source": str(source)}),
        source_type="scene31_missing_pattern",
        run_id=str(_first(normalized, "run_id") or _stable_digest(f"{source}:{run_name}")),
        run_name=str(run_name) if run_name is not None else None,
        method=str(method) if method is not None else None,
        seed=seed,
        pattern=str(pattern) if pattern is not None else None,
        metrics=metrics,
        sample_count=_first(normalized, "sample_count", "samples", "n"),
        split=_first(normalized, "split"),
        label_space=_first(normalized, "label_space"),
        metric_profile=_first(normalized, "metric_profile", "metrics_profile"),
        target_source=_first(normalized, "target_source", "beam_target_source"),
        difficulty_digest=_first(normalized, "difficulty_digest"),
        run_family=_first(normalized, "run_family", "family") or method,
        artifact_paths={"source": str(source)},
        generated_at=generated_at,
        config_path=_as_optional_str(_first(normalized, "config_path")),
        config_digest=_as_optional_str(_first(normalized, "config_digest")),
    )
    candidate.warnings = _required_warnings(candidate.to_dict())
    candidate.next_action_hints = _candidate_next_actions(candidate.warnings)
    return candidate


def _missing_pattern_artifacts(roots: Iterable[Path]) -> list[Path]:
    artifacts: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name.endswith(("_missing_patterns.csv", "_missing_patterns.json")):
            artifacts.add(root)
        elif root.exists() and root.is_dir():
            artifacts.update(path for path in root.rglob("*_missing_patterns.csv") if path.is_file())
            artifacts.update(path for path in root.rglob("*_missing_patterns.json") if path.is_file())
    return sorted(artifacts, key=lambda path: path.as_posix())


def _config_summary(cfg: dict[str, Any], *, run_dir: Path, config_path: Path | None) -> dict[str, Any]:
    experiment = _dict_at(cfg, "experiment")
    dataset = _dict_at(cfg, "data", "dataset")
    runtime = _dict_at(cfg, "runtime")
    objective = runtime.get("prediction_objective") if isinstance(runtime.get("prediction_objective"), dict) else {}
    output = _dict_at(cfg, "output")
    evaluation = _dict_at(cfg, "evaluation")
    return {
        "config_path": str(config_path) if config_path else None,
        "config_digest": _file_digest(config_path),
        "seed": _first(experiment, "seed"),
        "scene_scope": _first(runtime, "scene_scope", "output_scope", "scene"),
        "dataset_family": _first(dataset, "type", "family") or _dataset_family_from_path(run_dir),
        "split": _first(runtime, "split") or _first(dataset, "split") or _first(evaluation, "split"),
        "sample_count": _first(runtime, "sample_count", "num_samples", "effective_num_samples")
        or _first(dataset, "sample_count", "num_samples"),
        "label_space": _first(runtime, "label_space") or _first(dataset, "label_space") or _first(objective, "label_space"),
        "metric_profile": _first(runtime, "metric_profile") or _first(evaluation, "metric_profile", "primary_metric") or _first(objective, "primary_metric"),
        "target_source": _first(runtime, "target_source", "beam_target_source") or _first(dataset, "beam_target_source"),
        "difficulty_digest": _first(runtime, "difficulty_digest") or _first(dataset, "difficulty_digest"),
        "run_family": _first(runtime, "run_family") or _first(experiment, "objective") or _first(dataset, "type", "family"),
        "output_run_name": _first(output, "run_name") or _first(experiment, "name"),
    }


def _read_metrics(path: Path | None, *, warnings: list[str] | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError as exc:
            if warnings is not None:
                warnings.append(f"failed to read CSV {path}: {exc}")
            return {}
        return {key: _coerce(value) for key, value in (rows[-1] if rows else {}).items() if key}
    data = _read_json(path, warnings=warnings)
    return data if isinstance(data, dict) else {}


def _checkpoint_provenance(run_dir: Path, *, run_status: dict[str, Any], warnings: list[str] | None) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    sidecars = sorted(checkpoint_dir.glob("*.json")) if checkpoint_dir.exists() else []
    sidecar_records = []
    for sidecar_path in sidecars:
        metadata = _read_json(sidecar_path, warnings=warnings) or {}
        checkpoint_path = sidecar_path.with_suffix("")
        sidecar_records.append({"path": sidecar_path, "checkpoint_path": checkpoint_path, "metadata": metadata})
    explicit_checkpoint = _first(run_status, "best_checkpoint", "checkpoint_path", "weights_path")
    selected = None
    if explicit_checkpoint:
        explicit_name = Path(str(explicit_checkpoint)).name
        selected = next((item for item in sidecar_records if item["checkpoint_path"].name == explicit_name), None)
    if selected is None and sidecar_records:
        selected = sidecar_records[0]
    if explicit_checkpoint:
        checkpoint_path = str(explicit_checkpoint)
        source = "explicit_path"
    elif selected is not None:
        checkpoint_path = str(selected["checkpoint_path"])
        source = "run_local_checkpoint"
    else:
        best = _first_existing(checkpoint_dir / "best.pth", checkpoint_dir / "best_top1.pth", checkpoint_dir / "last.pth")
        checkpoint_path = str(best) if best else None
        source = "run_local_checkpoint" if best else "unavailable"
    metadata = selected["metadata"] if selected is not None else {}
    selection_metric = _first(metadata, "selection_metric", "selected_metric", "primary_metric", "best_metric")
    selected_epoch = _first(metadata, "selected_epoch", "best_epoch", "best_top1_epoch", "epoch")
    status = "complete" if checkpoint_path and selected is not None else "incomplete"
    return {
        "status": status,
        "source": source,
        "checkpoint_path": checkpoint_path,
        "sidecar_path": str(selected["path"]) if selected is not None else None,
        "selection_metric": selection_metric,
        "selected_epoch": selected_epoch,
        "run_dir": str(run_dir),
    }


def _run_artifact_paths(
    run_dir: Path,
    *,
    metrics_path: Path | None,
    config_path: Path | None,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "run_dir": str(run_dir),
        "config": str(config_path) if config_path else None,
        "metrics": str(metrics_path) if metrics_path else None,
        "train_log": str(run_dir / "train_log.json") if (run_dir / "train_log.json").exists() else None,
        "run_status": str(run_dir / "run_status.json") if (run_dir / "run_status.json").exists() else None,
        "checkpoint": checkpoint.get("checkpoint_path"),
        "checkpoint_sidecar": checkpoint.get("sidecar_path"),
    }
    return {key: value for key, value in result.items() if value}


def _claim_config_from_run_index(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "config_path",
            "config_digest",
            "seed",
            "scene_scope",
            "dataset_family",
            "split",
            "sample_count",
            "label_space",
            "metric_profile",
            "target_source",
            "difficulty_digest",
        )
        if record.get(key) not in (None, "")
    }


def _required_warnings(data: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    for field_name in STRICT_FIELDS:
        if _missing(data.get(field_name)):
            warnings.append(
                ComparabilityWarning(
                    field=field_name,
                    kind="required_missing",
                    severity="needs_review",
                    message=f"Strict comparability requires {field_name}.",
                ).to_dict()
            )
    return warnings


def _append_warning(candidate: dict[str, Any], warning: ComparabilityWarning) -> None:
    record = warning.to_dict()
    duplicate = any(
        existing.get("field") == record.get("field")
        and existing.get("kind") == record.get("kind")
        and existing.get("expected") == record.get("expected")
        and existing.get("actual") == record.get("actual")
        for existing in candidate.get("warnings", [])
    )
    if not duplicate:
        candidate.setdefault("warnings", []).append(record)


def _candidate_next_actions(warnings: list[dict[str, Any]]) -> list[str]:
    fields = {warning.get("field") for warning in warnings}
    actions = []
    if "checkpoint_provenance" in fields:
        actions.append("add checkpoint sidecar or selected checkpoint provenance")
    missing = sorted(str(field) for field in fields if field and field != "checkpoint_provenance")
    if missing:
        actions.append("fill strict comparability fields: " + ", ".join(missing))
    if any(warning.get("severity") == "not_comparable" for warning in warnings):
        actions.append("rerun or separate non-comparable candidates before claim review")
    return actions


def _dashboard_hints(*, index: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    for state in ("failed", "killed", "waiting", "stale", "partial"):
        count = sum(1 for run in index.get("runs", []) if run.get("state") == state)
        if count:
            hints.append(f"review {count} {state} run(s) in the run index")
    for candidate in candidates:
        for hint in candidate.get("next_action_hints", []):
            if hint not in hints:
                hints.append(hint)
    if not any(candidate.get("comparability_status") == "strict" for candidate in candidates) and candidates:
        hints.append("no strict candidates yet; review missing strict fields before promoting claims")
    return hints


def _candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics", {})
    primary_name, primary_value = next(iter(metrics.items()), (None, None)) if metrics else (None, None)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "run_name": candidate.get("run_name"),
        "method": candidate.get("method"),
        "seed": candidate.get("seed"),
        "pattern": candidate.get("pattern"),
        "comparability_status": candidate.get("comparability_status"),
        "claim_status": candidate.get("claim_status"),
        "candidate_only": candidate.get("candidate_only", True),
        "primary_metric": {"name": primary_name, "value": primary_value},
        "next_action_hints": list(candidate.get("next_action_hints", [])),
    }


def _resource_summary(resources: dict[str, Any]) -> dict[str, Any]:
    gpus = resources.get("gpus", {}) if isinstance(resources.get("gpus"), dict) else {}
    return {
        "gpu_available": bool(gpus.get("available")),
        "gpu_count": len(gpus.get("devices", []) or []),
        "process_count": len(resources.get("processes", []) or []),
        "memory": resources.get("memory", {}),
    }


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
