import datetime as dt
import os
from pathlib import Path
from typing import Any, Iterable

from kd_sensing.diagnostics.run_index import DEFAULT_STALE_AFTER, build_run_index

from kd_sensing.diagnostics.runtime_artifact_cleanup_base import (
    CHECKPOINT_SUFFIXES,
    DEFAULT_SCAN_ROOTS,
    RISK_ORDER,
    CleanupManifestMetadata,
    CleanupRecord,
    collect_git_tracked_paths,
    evaluate_protection,
    _aggregate_by,
    _format_dt,
    _ensure_utc,
    _is_current_mainline_output_path,
    _is_git_tracked,
    _matches_current_mainline_output_partition,
    _path_mtime,
    _path_size_bytes,
    _relative_path,
    _top_level_part,
    _utc_now,
)


def build_cleanup_manifest(
    *,
    project_root: str | Path = ".",
    scan_roots: Iterable[str | Path] | None = None,
    include_resources: bool = False,
    stale_after: dt.timedelta = DEFAULT_STALE_AFTER,
    now: dt.datetime | None = None,
    command_args: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a read-only cleanup candidate manifest."""

    root = Path(project_root).expanduser().resolve()
    generated_at = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    roots = _resolve_scan_roots(root, scan_roots)
    tracked_paths = collect_git_tracked_paths(root)
    warnings: list[str] = []
    matches: dict[Path, dict[str, Any]] = {}

    run_index = _build_run_index_for_cleanup(
        roots,
        project_root=root,
        include_resources=include_resources,
        stale_after=stale_after,
        now=now,
        include_legacy_containers=True,
    )
    warnings.extend(run_index.get("warnings", []))
    active_run_dirs = {
        Path(run["run_dir"]).resolve()
        for run in run_index.get("runs", [])
        if run.get("cleanup", {}).get("protected")
    }

    for scan_root in roots:
        if not scan_root.exists():
            warnings.append(f"scan root does not exist: {scan_root}")
            continue
        decision = evaluate_protection(scan_root, project_root=root, tracked_paths=tracked_paths)
        if decision.protected and _top_level_part(scan_root, root) in {"dataset", "All_models"}:
            _add_match(
                matches,
                scan_root,
                rule_id="protected.root",
                artifact_type="protected_root",
                risk="high",
                reason="Protected data or historical model root.",
                force_protected=True,
                protection_reasons=decision.reasons,
            )
            continue
        _collect_path_rule_matches(matches, scan_root, project_root=root)

    _collect_run_rule_matches(matches, run_index, project_root=root)

    candidates: list[CleanupRecord] = []
    protected: list[CleanupRecord] = []
    for match in sorted(matches.values(), key=lambda item: _relative_path(Path(item["path"]), root)):
        record = _materialize_record(
            match,
            project_root=root,
            tracked_paths=tracked_paths,
            active_run_dirs=active_run_dirs,
        )
        if record.protected:
            protected.append(record)
        else:
            candidates.append(record)

    metadata = CleanupManifestMetadata(
        generated_at=generated_at,
        project_root=str(root),
        scan_roots=tuple(str(path) for path in roots),
        allowed_roots=tuple(str(path) for path in roots),
        command_args=tuple(command_args),
    )
    candidate_dicts = [record.to_dict() for record in candidates]
    protected_dicts = [record.to_dict() for record in protected]
    return {
        "metadata": metadata.to_dict(),
        "summary": _manifest_summary(candidate_dicts, protected_dicts, roots),
        "candidates": candidate_dicts,
        "protected": protected_dicts,
        "run_index": {
            "run_count": len(run_index.get("runs", [])),
            "roots": run_index.get("roots", {}),
            "generated_at": run_index.get("generated_at"),
        },
        "warnings": warnings,
    }

def _collect_path_rule_matches(matches: dict[Path, dict[str, Any]], scan_root: Path, *, project_root: Path) -> None:
    if _matches_current_mainline_output_partition(scan_root, project_root):
        _add_current_mainline_output_match(matches, scan_root)
        return
    if _matches_python_cache(scan_root):
        _add_python_cache_match(matches, scan_root)
        return
    if _matches_pytest_cache(scan_root):
        _add_pytest_cache_match(matches, scan_root)
        return
    if _matches_debug_plan(scan_root):
        _add_debug_plan_match(matches, scan_root)
        return
    if _top_level_part(scan_root, project_root) == "logs":
        _add_log_matches(matches, scan_root, project_root=project_root)
    if _top_level_part(scan_root, project_root) == "cache":
        _add_match(
            matches,
            scan_root,
            rule_id="cache.generic",
            artifact_type="cache",
            risk="low",
            reason="Top-level local cache root.",
        )

    for current, dirnames, filenames in os.walk(scan_root):
        current_path = Path(current)
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for dirname in list(dirnames):
            path = current_path / dirname
            if _matches_current_mainline_output_partition(path, project_root):
                _add_current_mainline_output_match(matches, path)
                dirnames.remove(dirname)
            elif dirname == "__pycache__":
                _add_python_cache_match(matches, path)
                dirnames.remove(dirname)
            elif dirname == ".pytest_cache":
                _add_pytest_cache_match(matches, path)
                dirnames.remove(dirname)
            elif _matches_debug_plan(path):
                _add_debug_plan_match(matches, path)
                dirnames.remove(dirname)
            elif _matches_smoke_artifact(path, project_root):
                _add_smoke_match(matches, path)
                dirnames.remove(dirname)
        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() in {".pyc", ".pyo"}:
                _add_python_cache_match(matches, path)
            elif _is_personal_backup_archive(path):
                _add_match(
                    matches,
                    path,
                    rule_id="archive.personal_backup",
                    artifact_type="backup_archive",
                    risk="low",
                    reason="Personal backup archive under local artifact roots.",
                )

def _collect_run_rule_matches(matches: dict[Path, dict[str, Any]], run_index: dict[str, Any], *, project_root: Path) -> None:
    for run in run_index.get("runs", []):
        run_dir = Path(run["run_dir"]).resolve()
        run_summary = _public_run_summary(run)
        rel = _relative_path(run_dir, project_root)
        if run.get("cleanup", {}).get("protected"):
            _add_match(
                matches,
                run_dir,
                rule_id="run.active_protected",
                artifact_type="experiment_run",
                risk="high",
                reason="Run index reports an active or recent unfinished run.",
                force_protected=True,
                protection_reasons=tuple(run.get("cleanup", {}).get("protection_reasons", [])),
                run_summary=run_summary,
            )
        if rel.startswith("outputs/other/") or rel == "outputs/other":
            _add_match(
                matches,
                run_dir,
                rule_id="output.ambiguous_other",
                artifact_type="experiment_run",
                risk="medium",
                reason="Historical run is under semantically ambiguous outputs/other.",
                run_summary=run_summary,
            )
        state = str(run.get("state") or "")
        if state in {"failed", "killed", "stale", "partial"}:
            _add_match(
                matches,
                run_dir,
                rule_id=f"run.{state}",
                artifact_type="experiment_run",
                risk="high" if state in {"stale", "partial"} else "medium",
                reason=f"Run index classified this run as {state}.",
                run_summary=run_summary,
            )
        _add_checkpoint_matches(matches, run, run_summary=run_summary)

def _add_checkpoint_matches(matches: dict[Path, dict[str, Any]], run: dict[str, Any], *, run_summary: dict[str, Any]) -> None:
    state = str(run.get("state") or "")
    metrics_available = bool(run.get("metrics", {}).get("available"))
    has_primary = bool(run.get("checkpoints", {}).get("primary_checkpoint"))
    for item in run.get("checkpoints", {}).get("retention", {}).get("items", []):
        raw_path = item.get("path")
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        name = path.name
        if item.get("registry_protected"):
            _add_match(
                matches,
                path,
                rule_id="checkpoint.reproducible_protected",
                artifact_type="checkpoint",
                risk="high",
                reason=str(item.get("retention_reason") or "Protected checkpoint."),
                force_protected=True,
                protection_reasons=("checkpoint_retention_protected",),
                run_summary=run_summary,
                checkpoint_summary=item,
            )
            sidecar_path = item.get("sidecar_path")
            if sidecar_path:
                _add_match(
                    matches,
                    Path(sidecar_path).resolve(),
                    rule_id="checkpoint.sidecar_protected",
                    artifact_type="checkpoint_sidecar",
                    risk="high",
                    reason="Sidecar metadata for a protected checkpoint.",
                    force_protected=True,
                    protection_reasons=("checkpoint_sidecar_for_protected_checkpoint",),
                    run_summary=run_summary,
                    checkpoint_summary=item,
                )
            continue
        if name == "last.pth" and state == "complete" and metrics_available and has_primary:
            _add_match(
                matches,
                path,
                rule_id="checkpoint.last_recoverable",
                artifact_type="checkpoint",
                risk="medium",
                reason="last.pth is recoverable state, not the default reproducibility checkpoint.",
                run_summary=run_summary,
                checkpoint_summary=item,
            )
        elif "probe" in path.stem.lower() and has_primary:
            _add_match(
                matches,
                path,
                rule_id="checkpoint.duplicate_probe",
                artifact_type="checkpoint",
                risk="medium",
                reason="Probe checkpoint has a primary checkpoint in the same run.",
                run_summary=run_summary,
                checkpoint_summary=item,
            )
        elif state in {"failed", "killed"} and path.suffix.lower() in CHECKPOINT_SUFFIXES:
            _add_match(
                matches,
                path,
                rule_id="checkpoint.failed_run_temporary",
                artifact_type="checkpoint",
                risk="medium",
                reason="Temporary checkpoint belongs to a failed or killed run.",
                run_summary=run_summary,
                checkpoint_summary=item,
            )

def _materialize_record(
    match: dict[str, Any],
    *,
    project_root: Path,
    tracked_paths: set[str],
    active_run_dirs: Iterable[Path],
) -> CleanupRecord:
    path = Path(match["path"]).resolve()
    decision = evaluate_protection(path, project_root=project_root, tracked_paths=tracked_paths, active_run_dirs=active_run_dirs)
    forced_reasons = tuple(match.get("protection_reasons", ()))
    protection_reasons = tuple(dict.fromkeys((*decision.reasons, *forced_reasons)))
    protected = bool(decision.protected or match.get("force_protected"))
    return CleanupRecord(
        path=str(path),
        relative_path=_relative_path(path, project_root),
        artifact_type=str(match["artifact_type"]),
        size_bytes=_path_size_bytes(path),
        mtime=_format_dt(_path_mtime(path)),
        rule_id=str(match["rule_id"]),
        matched_rules=tuple(match.get("matched_rules", [match["rule_id"]])),
        risk=str(match["risk"]),
        reason="; ".join(match.get("reasons", [match["reason"]])),
        tracked=decision.tracked,
        protected=protected,
        protection_reasons=protection_reasons,
        action="protect" if protected else "delete",
        run_summary=match.get("run_summary"),
        checkpoint_summary=match.get("checkpoint_summary"),
    )

def _add_match(
    matches: dict[Path, dict[str, Any]],
    path: Path,
    *,
    rule_id: str,
    artifact_type: str,
    risk: str,
    reason: str,
    force_protected: bool = False,
    protection_reasons: Iterable[str] = (),
    run_summary: dict[str, Any] | None = None,
    checkpoint_summary: dict[str, Any] | None = None,
) -> None:
    key = path.expanduser().resolve()
    existing = matches.get(key)
    if existing is None:
        matches[key] = {
            "path": key,
            "rule_id": rule_id,
            "matched_rules": [rule_id],
            "artifact_type": artifact_type,
            "risk": risk,
            "reason": reason,
            "reasons": [reason],
            "force_protected": force_protected,
            "protection_reasons": tuple(protection_reasons),
            "run_summary": run_summary,
            "checkpoint_summary": checkpoint_summary,
        }
        return
    if rule_id not in existing["matched_rules"]:
        existing["matched_rules"].append(rule_id)
    if reason not in existing["reasons"]:
        existing["reasons"].append(reason)
    if RISK_ORDER.get(risk, 0) > RISK_ORDER.get(str(existing.get("risk")), 0):
        existing["rule_id"] = rule_id
        existing["risk"] = risk
        existing["artifact_type"] = artifact_type
        existing["reason"] = reason
    existing["force_protected"] = bool(existing.get("force_protected") or force_protected)
    existing["protection_reasons"] = tuple(dict.fromkeys((*existing.get("protection_reasons", ()), *protection_reasons)))
    existing["run_summary"] = existing.get("run_summary") or run_summary
    existing["checkpoint_summary"] = existing.get("checkpoint_summary") or checkpoint_summary

def _add_python_cache_match(matches: dict[Path, dict[str, Any]], path: Path) -> None:
    _add_match(
        matches,
        path,
        rule_id="cache.python_bytecode",
        artifact_type="python_cache",
        risk="low",
        reason="Python bytecode cache.",
    )

def _add_pytest_cache_match(matches: dict[Path, dict[str, Any]], path: Path) -> None:
    _add_match(
        matches,
        path,
        rule_id="cache.pytest",
        artifact_type="pytest_cache",
        risk="low",
        reason="pytest cache directory.",
    )

def _add_debug_plan_match(matches: dict[Path, dict[str, Any]], path: Path) -> None:
    name = path.name.lower()
    rule = "transient.debug" if name.startswith("_debug") or "_debug" in name else "transient.plan_check"
    kind = "debug_artifact" if rule == "transient.debug" else "plan_check_artifact"
    _add_match(matches, path, rule_id=rule, artifact_type=kind, risk="low", reason="Short-lived debug or plan-check artifact.")

def _add_smoke_match(matches: dict[Path, dict[str, Any]], path: Path) -> None:
    _add_match(
        matches,
        path,
        rule_id="transient.smoke",
        artifact_type="smoke_output",
        risk="low",
        reason="Short-lived smoke output under local artifact roots.",
    )

def _add_current_mainline_output_match(matches: dict[Path, dict[str, Any]], path: Path) -> None:
    _add_match(
        matches,
        path,
        rule_id="protected.current_mainline_output",
        artifact_type="current_mainline_output",
        risk="high",
        reason="Current mainline output partition is protected by default.",
        force_protected=True,
        protection_reasons=("current_mainline_output_partition",),
    )

def _add_log_matches(matches: dict[Path, dict[str, Any]], scan_root: Path, *, project_root: Path) -> None:
    rel = _relative_path(scan_root, project_root)
    if rel != "logs":
        _add_match(matches, scan_root, rule_id="logs.local", artifact_type="log", risk="low", reason="Local log path.")
        return
    for child in sorted(scan_root.iterdir()):
        _add_match(matches, child, rule_id="logs.local", artifact_type="log", risk="low", reason="Local log path.")

def _build_run_index_for_cleanup(
    roots: tuple[Path, ...],
    *,
    project_root: Path,
    include_resources: bool,
    stale_after: dt.timedelta,
    now: dt.datetime | None,
    include_legacy_containers: bool = True,
) -> dict[str, Any]:
    output_roots = [path for path in roots if _top_level_part(path, project_root) == "outputs" or path.name == "outputs"]
    log_roots = [path for path in roots if _top_level_part(path, project_root) == "logs" or path.name == "logs"]
    return build_run_index(
        outputs=output_roots,
        logs=log_roots if log_roots else None,
        include_resources=include_resources,
        stale_after=stale_after,
        now=now,
        include_legacy_containers=include_legacy_containers,
    )

def _resolve_scan_roots(project_root: Path, scan_roots: Iterable[str | Path] | None) -> tuple[Path, ...]:
    raw_roots = tuple(scan_roots) if scan_roots is not None else DEFAULT_SCAN_ROOTS
    resolved: list[Path] = []
    for raw in raw_roots:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = project_root / path
        resolved.append(path.resolve())
    return tuple(dict.fromkeys(resolved))

def _public_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    checkpoints = run.get("checkpoints", {})
    return {
        "run_dir": run.get("run_dir"),
        "run_name": run.get("run_name"),
        "state": run.get("state"),
        "state_reason": run.get("state_reason"),
        "size_bytes": run.get("size_bytes"),
        "timestamps": run.get("timestamps"),
        "missing_artifacts": run.get("artifacts", {}).get("missing", []),
        "checkpoint_count": checkpoints.get("count", 0),
        "checkpoint_total_size_bytes": checkpoints.get("total_size_bytes", 0),
        "primary_checkpoint": checkpoints.get("primary_checkpoint"),
        "logs": run.get("logs", []),
        "cleanup": run.get("cleanup", {}),
    }

def _manifest_summary(candidates: list[dict[str, Any]], protected: list[dict[str, Any]], roots: tuple[Path, ...]) -> dict[str, Any]:
    return {
        "scan_root_count": len(roots),
        "scan_roots": [str(path) for path in roots],
        "candidate_count": len(candidates),
        "candidate_total_size_bytes": sum(int(item.get("size_bytes") or 0) for item in candidates),
        "protected_count": len(protected),
        "protected_total_size_bytes": sum(int(item.get("size_bytes") or 0) for item in protected),
        "by_rule": _aggregate_by(candidates, "rule_id"),
        "by_artifact_type": _aggregate_by(candidates, "artifact_type"),
    }

def _matches_python_cache(path: Path) -> bool:
    return path.name == "__pycache__" or path.suffix.lower() in {".pyc", ".pyo"}

def _matches_pytest_cache(path: Path) -> bool:
    return path.name == ".pytest_cache"

def _matches_debug_plan(path: Path) -> bool:
    name = path.name.lower()
    return name in {"_debug", "_plan_check"} or name.startswith("_debug_") or "_plan_check" in name or name.endswith("_plan")

def _matches_smoke_artifact(path: Path, project_root: Path) -> bool:
    rel = _relative_path(path, project_root)
    if not rel.startswith("outputs/"):
        return False
    name = path.name.lower()
    return "smoke" in name

def _is_personal_backup_archive(path: Path) -> bool:
    lower = path.name.lower()
    if not lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        return False
    return any(token in lower for token in ("backup", "bak", "old", "copy", "personal"))
