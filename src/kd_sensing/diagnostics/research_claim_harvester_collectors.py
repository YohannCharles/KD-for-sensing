import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

from kd_sensing.diagnostics.run_index import build_run_index
from kd_sensing.diagnostics.paper_artifact_export import load_input_rows
from kd_sensing.diagnostics.research_claim_harvester_base import (
    DOCTOR_REQUIRED_FIELDS,
    SCHEMA_VERSION,
    ClaimCandidate,
    ComparabilityWarning,
    _as_optional_str,
    _as_paths,
    _candidate_id,
    _coerce,
    _dataset_family_from_path,
    _dict_at,
    _ensure_utc,
    _file_digest,
    _first,
    _first_existing,
    _format_dt,
    _metric_fields,
    _missing,
    _numeric_items,
    _primary_metric,
    _read_json,
    _read_yaml,
    _run_name_from_artifact,
    _method_from_run_name,
    _seed_from_text,
    _stable_digest,
    _utc_now,
)
from kd_sensing.diagnostics.research_claim_harvester_gate import (
    _candidate_next_actions,
    _required_warnings,
    apply_strict_comparability_gate,
)


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

def build_claim_doctor_report(
    *,
    claim_registry_paths: Iterable[str | Path] = (),
    candidate_ledger_paths: Iterable[str | Path] = (),
    summary_paths: Iterable[str | Path] = (),
    run_index: dict[str, Any] | None = None,
    candidates: Iterable[dict[str, Any]] = (),
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Report missing evidence for claim review without mutating claim docs."""

    generated_at = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    rows: list[dict[str, Any]] = []
    input_files: dict[str, list[str]] = {
        "claim_registry": [],
        "candidate_ledger": [],
        "summary_artifacts": [],
    }
    for candidate in candidates:
        rows.append({**dict(candidate), "_doctor_source_type": "candidate"})
    for path in claim_registry_paths:
        source = Path(path)
        input_files["claim_registry"].append(str(source))
        rows.extend({**row, "_doctor_source_type": "claim_registry", "_source_file": str(source)} for row in _load_doctor_rows(source))
    for path in candidate_ledger_paths:
        source = Path(path)
        input_files["candidate_ledger"].append(str(source))
        rows.extend({**row, "_doctor_source_type": "candidate_ledger", "_source_file": str(source)} for row in _load_doctor_rows(source))
    for path in summary_paths:
        source = Path(path)
        input_files["summary_artifacts"].append(str(source))
        rows.extend({**row, "_doctor_source_type": "summary_artifact", "_source_file": str(source)} for row in _load_doctor_rows(source))
    if run_index:
        rows.extend({**record, "_doctor_source_type": "run_index"} for record in run_index_records_for_harvester(run_index))

    diagnostics = [_doctor_row(row) for row in rows]
    missing_counts: dict[str, int] = {}
    for item in diagnostics:
        for field_name in item["missing_fields"]:
            missing_counts[field_name] = missing_counts.get(field_name, 0) + 1
    upgradable = [
        item
        for item in diagnostics
        if not item["missing_fields"] and item["claim_status"] in {"draft", "pending", "unverified", "not_comparable"}
    ]
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "inputs": input_files,
            "does_not_update_claim_registry": True,
        },
        "required_fields": list(DOCTOR_REQUIRED_FIELDS),
        "diagnostics": diagnostics,
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "upgradable_candidates": upgradable,
        "next_action_hints": _unique_hint_list(diagnostics),
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

def _load_doctor_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        return rows
    return load_input_rows(path)

def _doctor_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_doctor_row(row)
    missing_fields = [field_name for field_name in DOCTOR_REQUIRED_FIELDS if _missing(normalized.get(field_name))]
    return {
        "claim_id": normalized.get("claim_id") or normalized.get("candidate_id") or normalized.get("run_id"),
        "source_type": normalized.get("source_type"),
        "run_name": normalized.get("run_name"),
        "method": normalized.get("method"),
        "claim_status": _doctor_status(normalized),
        "candidate_only": bool(normalized.get("candidate_only")),
        "comparability_status": normalized.get("comparability_status"),
        "missing_fields": missing_fields,
        "next_action_hints": _doctor_next_actions(missing_fields),
        "caveat": normalized.get("caveat"),
    }

def _normalize_doctor_row(row: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _doctor_pick(row, "checkpoint_provenance", "checkpoint provenance", "checkpoint", "checkpoint_path", "best_checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("checkpoint_path") or checkpoint.get("sidecar_path") or checkpoint.get("status")
    return {
        "claim_id": _doctor_pick(row, "claim_id", "claim id", "id", "claim"),
        "candidate_id": _doctor_pick(row, "candidate_id"),
        "run_id": _doctor_pick(row, "run_id"),
        "source_type": row.get("_doctor_source_type") or _doctor_pick(row, "source_type", "source"),
        "run_name": _doctor_pick(row, "run_name", "run", "name"),
        "method": _doctor_pick(row, "method", "model line", "model_line", "model", "line"),
        "claim_status": _doctor_pick(row, "claim_status", "claim status", "status", "comparability_status"),
        "candidate_only": _truthy(_doctor_pick(row, "candidate_only", "candidate only", "draft")),
        "comparability_status": _doctor_pick(row, "comparability_status", "comparability status"),
        "seed": _doctor_pick(row, "seed"),
        "split": _doctor_pick(row, "split", "dataset_split", "dataset / split"),
        "metric_profile": _doctor_pick(row, "metric_profile", "metric profile", "metrics_profile", "target / metric field", "metric"),
        "label_space": _doctor_pick(row, "label_space", "label space"),
        "checkpoint_provenance": checkpoint,
        "difficulty_digest": _doctor_pick(row, "difficulty_digest", "difficulty digest"),
        "stress_provenance": _doctor_pick(row, "stress_provenance", "stress provenance", "stress suite status", "stress_status"),
        "paired_baseline": _doctor_pick(row, "paired_baseline", "paired baseline", "baseline"),
        "statistical_evidence": _doctor_pick(row, "statistical_evidence", "statistical evidence", "seed_count", "mean/std", "ci"),
        "caveat": _doctor_pick(row, "caveat", "note", "notes", "warning", "warnings"),
    }

def _doctor_pick(row: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return None

def _doctor_status(row: dict[str, Any]) -> str:
    status = str(row.get("claim_status") or "draft").lower()
    if row.get("candidate_only") and "candidate" not in status:
        return "draft"
    return status

def _doctor_next_actions(missing_fields: list[str]) -> list[str]:
    actions = {
        "seed": "record seed or seed set",
        "split": "record dataset split/protocol",
        "metric_profile": "record metric profile",
        "label_space": "record label space",
        "checkpoint_provenance": "add checkpoint sidecar or selected checkpoint provenance",
        "difficulty_digest": "record difficulty digest",
        "stress_provenance": "attach stress-suite provenance",
        "paired_baseline": "attach paired baseline evidence",
        "statistical_evidence": "attach multi-seed mean/std or CI evidence",
    }
    return [actions[field_name] for field_name in missing_fields if field_name in actions]

def _unique_hint_list(diagnostics: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    for item in diagnostics:
        for hint in item.get("next_action_hints", []):
            if hint not in hints:
                hints.append(hint)
    return hints

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "draft", "candidate", "candidate_only"}
