import datetime as dt
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.utils.runtime_output_layout import (
    PARTITION_ARCHIVE,
    PARTITION_CACHE,
    PROTECTED_MAINLINE_PARTITIONS,
    output_layout_summary,
    runtime_output_scope_from_config,
)

from kd_sensing.diagnostics.runtime_artifact_cleanup_base import (
    CHECKPOINT_SUFFIXES,
    MANIFEST_SCHEMA_VERSION,
    ORGANIZE_RULES_VERSION,
    collect_git_tracked_paths,
    evaluate_protection,
    _aggregate_by,
    _ensure_utc,
    _format_dt,
    _manifest_state_compatible,
    _path_contains_or_equals,
    _path_mtime,
    _path_size_bytes,
    _relative_path,
    _skip,
    _utc_now,
)


def build_runtime_output_organize_manifest(
    *,
    project_root: str | Path = ".",
    outputs_root: str | Path = "outputs",
    now: dt.datetime | None = None,
    command_args: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a read-only move/archive/protect/review plan for outputs/."""

    root = Path(project_root).expanduser().resolve()
    outputs = Path(outputs_root).expanduser()
    if not outputs.is_absolute():
        outputs = root / outputs
    outputs = outputs.resolve()
    generated_at = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    tracked_paths = collect_git_tracked_paths(root)
    warnings: list[str] = []
    plans: list[dict[str, Any]] = []

    if not outputs.exists():
        warnings.append(f"outputs root does not exist: {outputs}")
    elif not outputs.is_dir():
        warnings.append(f"outputs root is not a directory: {outputs}")
    else:
        for child in sorted(outputs.iterdir(), key=lambda path: path.name):
            record = _organize_record_for_outputs_child(
                child,
                project_root=root,
                outputs_root=outputs,
                tracked_paths=tracked_paths,
            )
            if record is not None:
                plans.append(record)

    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "rules_version": ORGANIZE_RULES_VERSION,
        "generated_at": generated_at,
        "project_root": str(root),
        "outputs_root": str(outputs),
        "allowed_roots": [str(outputs)],
        "dry_run": True,
        "command_args": list(command_args),
    }
    return {
        "metadata": metadata,
        "summary": _organize_summary(plans),
        "plans": plans,
        "warnings": warnings,
    }

def apply_runtime_output_organize_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    confirm_organize: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if not confirm_organize:
        raise ValueError("Organization refused: inspect the manifest and pass confirm_organize=True.")

    source = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    root = Path(project_root or manifest.get("metadata", {}).get("project_root") or ".").expanduser().resolve()
    outputs_root = Path(manifest.get("metadata", {}).get("outputs_root") or root / "outputs").expanduser().resolve()
    tracked_paths = collect_git_tracked_paths(root)
    report: dict[str, Any] = {
        "metadata": {
            "generated_at": _format_dt(_utc_now()),
            "manifest_path": str(source),
            "project_root": str(root),
            "outputs_root": str(outputs_root),
            "rules_version": ORGANIZE_RULES_VERSION,
        },
        "moved": [],
        "skipped": [],
        "failed": [],
    }

    for record in manifest.get("plans", []):
        result = _apply_one_organize_record(
            record,
            project_root=root,
            outputs_root=outputs_root,
            tracked_paths=tracked_paths,
        )
        report[result["status"]].append(result["record"])

    report["summary"] = {
        "moved_count": len(report["moved"]),
        "skipped_count": len(report["skipped"]),
        "failed_count": len(report["failed"]),
        "moved_size_bytes": sum(int(item.get("size_bytes") or 0) for item in report["moved"]),
    }
    target = _organize_report_path(source, report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    report["metadata"]["report_path"] = str(target)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report

def _organize_record_for_outputs_child(
    path: Path,
    *,
    project_root: Path,
    outputs_root: Path,
    tracked_paths: set[str],
) -> dict[str, Any] | None:
    name = path.name
    layout = output_layout_summary(path, outputs_root=outputs_root)
    target: Path | None = None
    action = "review"
    artifact_type = str(layout.get("canonical_partition") or "unknown")
    rule_id = "output.review"
    reason = "Output path does not match a known current partition and needs review."
    risk = "medium"

    if name == PARTITION_CACHE:
        action = "protect"
        artifact_type = "cache"
        rule_id = "organize.cache_protected"
        reason = "Cache partition is protected and is not moved into training, evaluation, analysis, or archive."
        risk = "low"
    elif name in PROTECTED_MAINLINE_PARTITIONS or layout.get("canonical_partition") in {"scene", "scenegroup"}:
        action = "protect"
        artifact_type = str(layout.get("canonical_partition") or name)
        rule_id = "organize.current_partition_protected"
        reason = "Canonical runtime output partition is protected by default."
        risk = "low"
    elif name.isdigit():
        action = "archive"
        artifact_type = "legacy_numeric_scene"
        rule_id = "organize.legacy_numeric_scene"
        reason = "Numeric outputs scene root is legacy and should be archived or manually migrated."
        target = outputs_root / PARTITION_ARCHIVE / "legacy_numeric_scene" / name
    elif name == "best_checkpoints":
        action = "review"
        artifact_type = "legacy_registry"
        rule_id = "organize.legacy_registry"
        reason = "Root-level best checkpoint registry is legacy; sidecar metadata must be reviewed before migration."
        target = outputs_root / PARTITION_ARCHIVE / "legacy_best_checkpoints" / name
        risk = "high"
    elif name.startswith("eval_"):
        action = "archive"
        artifact_type = "legacy_evaluation"
        rule_id = "organize.legacy_evaluation"
        reason = "Root-level eval_* output is legacy; current grouped evaluations live under outputs/evaluations/."
        target = outputs_root / PARTITION_ARCHIVE / "legacy_eval_runs" / name
    elif _looks_like_run_dir(path):
        artifact_type = "legacy_root_run"
        rule_id = "organize.legacy_root_run"
        cfg = _run_config_for_organize(path)
        scope = runtime_output_scope_from_config(cfg) if cfg else None
        if scope is None:
            action = "review"
            reason = "Root-level training run has no reliable scene or scenegroup metadata."
            target = outputs_root / PARTITION_ARCHIVE / "legacy_root_runs" / name
            risk = "high"
        else:
            action = "move"
            reason = f"Root-level training run can be moved under canonical scope {scope.slug}."
            target = outputs_root / scope.slug / name
            layout["scope_kind"] = scope.kind
            layout["scope_slug"] = scope.slug
            layout["scene_ids"] = list(scope.scene_ids)
            layout["scene_slugs"] = list(scope.scene_slugs)
    elif path.is_dir():
        target = outputs_root / PARTITION_ARCHIVE / "manual_review" / name

    return _organize_record(
        path,
        project_root=project_root,
        target=target,
        action=action,
        artifact_type=artifact_type,
        rule_id=rule_id,
        reason=reason,
        risk=risk,
        tracked_paths=tracked_paths,
        layout=layout,
    )

def _organize_record(
    path: Path,
    *,
    project_root: Path,
    target: Path | None,
    action: str,
    artifact_type: str,
    rule_id: str,
    reason: str,
    risk: str,
    tracked_paths: set[str],
    layout: dict[str, Any],
) -> dict[str, Any]:
    source = path.resolve()
    target_path = target.resolve() if target is not None else None
    decision = evaluate_protection(source, project_root=project_root, tracked_paths=tracked_paths)
    conflict = _organize_conflict(source, target_path)
    requires_review = action == "review" or bool(conflict) or decision.tracked
    protected = action == "protect" or decision.protected
    if protected and "protected" not in action:
        requires_review = True
    return {
        "source_path": str(source),
        "relative_path": _relative_path(source, project_root),
        "target_path": str(target_path) if target_path is not None else None,
        "target_relative_path": _relative_path(target_path, project_root) if target_path is not None else None,
        "action": "protect" if protected and action == "protect" else action,
        "artifact_type": artifact_type,
        "size_bytes": _path_size_bytes(source),
        "mtime": _format_dt(_path_mtime(source)),
        "rule_id": rule_id,
        "matched_rules": [rule_id],
        "risk": risk,
        "reason": reason,
        "layout": layout,
        "tracked": decision.tracked,
        "protected": protected,
        "protection_reasons": list(decision.reasons),
        "conflict": conflict,
        "requires_manual_review": requires_review or protected,
    }

def _apply_one_organize_record(
    record: dict[str, Any],
    *,
    project_root: Path,
    outputs_root: Path,
    tracked_paths: set[str],
) -> dict[str, Any]:
    source = Path(record.get("source_path", "")).expanduser().resolve()
    target_raw = record.get("target_path")
    target = Path(target_raw).expanduser().resolve() if target_raw else None
    base = {
        "source_path": str(source),
        "target_path": str(target) if target is not None else None,
        "relative_path": _relative_path(source, project_root),
        "rule_id": record.get("rule_id"),
        "action": record.get("action"),
    }
    if record.get("action") not in {"move", "archive"}:
        return _skip(base, "no_execution_for_action")
    if record.get("protected"):
        return _skip(base, "manifest_record_is_protected")
    if record.get("requires_manual_review"):
        return _skip(base, "manual_review_required")
    if target is None:
        return _skip(base, "target_missing")
    if not source.exists():
        return _skip(base, "source_missing")
    if not _path_contains_or_equals(outputs_root, source):
        return _skip(base, "outside_outputs_root")
    decision = evaluate_protection(source, project_root=project_root, tracked_paths=tracked_paths)
    if decision.protected:
        base["protection_reasons"] = list(decision.reasons)
        return _skip(base, "source_now_protected")
    if not _manifest_state_compatible({"size_bytes": record.get("size_bytes"), "mtime": record.get("mtime")}, source):
        return _skip(base, "source_changed_since_manifest")
    if target.exists():
        return _skip(base, "target_exists")
    try:
        size = _path_size_bytes(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    except OSError as exc:
        failed = dict(base)
        failed["reason"] = str(exc)
        return {"status": "failed", "record": failed}
    moved = dict(base)
    moved["size_bytes"] = size
    return {"status": "moved", "record": moved}

def _organize_conflict(source: Path, target: Path | None) -> dict[str, Any]:
    if target is None:
        return {"status": "none", "target_exists": False}
    if source == target:
        return {"status": "same_path", "target_exists": True}
    if target.exists():
        return {"status": "target_exists", "target_exists": True}
    return {"status": "none", "target_exists": False}

def _looks_like_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = {
        "final_config.yaml",
        "resolved_config.yaml",
        "startup_summary.json",
        "metrics.json",
        "train_log.json",
        "training_outputs.npz",
        "test_report.json",
        "run_status.json",
    }
    if any((path / name).exists() for name in markers):
        return True
    checkpoint_dir = path / "checkpoints"
    return checkpoint_dir.exists() and any(item.suffix.lower() in CHECKPOINT_SUFFIXES for item in checkpoint_dir.iterdir())

def _run_config_for_organize(path: Path) -> dict[str, Any] | None:
    for name in ("final_config.yaml", "resolved_config.yaml"):
        candidate = path / name
        if not candidate.exists():
            continue
        try:
            data = safe_load_yaml(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        return data if isinstance(data, dict) else None
    return None

def _organize_summary(plans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "plan_count": len(plans),
        "total_size_bytes": sum(int(item.get("size_bytes") or 0) for item in plans),
        "manual_review_count": sum(1 for item in plans if item.get("requires_manual_review")),
        "conflict_count": sum(1 for item in plans if item.get("conflict", {}).get("status") != "none"),
        "by_action": _aggregate_by(plans, "action"),
        "by_artifact_type": _aggregate_by(plans, "artifact_type"),
    }

def _organize_report_path(manifest_path: Path, report_path: str | Path | None) -> Path:
    if report_path is not None:
        return Path(report_path).expanduser().resolve()
    return manifest_path.with_name(manifest_path.stem + ".organize_report.json")
