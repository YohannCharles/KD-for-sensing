from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

from kd_sensing.diagnostics.run_index import DEFAULT_STALE_AFTER, build_run_index
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.utils.runtime_output_layout import (
    PARTITION_ARCHIVE,
    PARTITION_CACHE,
    PARTITION_CLEANUP_MANIFESTS,
    PROTECTED_MAINLINE_PARTITIONS,
    output_layout_summary,
    runtime_output_scope_from_config,
)


MANIFEST_SCHEMA_VERSION = 1
RULES_VERSION = "runtime-artifact-cleanup.v1"
ORGANIZE_RULES_VERSION = "runtime-output-organize.v1"
DEFAULT_SCAN_ROOTS = ("outputs", "logs", "cache", ".pytest_cache")
PROTECTED_ROOTS = ("dataset", "All_models", "src", "configs", "docs", "openspec", "tests")
CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class CleanupManifestMetadata:
    generated_at: str
    project_root: str
    scan_roots: tuple[str, ...]
    allowed_roots: tuple[str, ...]
    command_args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "scan_roots": list(self.scan_roots),
            "allowed_roots": list(self.allowed_roots),
            "dry_run": True,
            "command_args": list(self.command_args),
        }


@dataclass(frozen=True)
class ProtectionDecision:
    protected: bool
    tracked: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CleanupRecord:
    path: str
    relative_path: str
    artifact_type: str
    size_bytes: int
    mtime: str | None
    rule_id: str
    matched_rules: tuple[str, ...]
    risk: str
    reason: str
    tracked: bool
    protected: bool
    protection_reasons: tuple[str, ...]
    action: str
    run_summary: dict[str, Any] | None = None
    checkpoint_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": self.path,
            "relative_path": self.relative_path,
            "artifact_type": self.artifact_type,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "rule_id": self.rule_id,
            "matched_rules": list(self.matched_rules),
            "risk": self.risk,
            "reason": self.reason,
            "tracked": self.tracked,
            "protected": self.protected,
            "protection_reasons": list(self.protection_reasons),
            "action": self.action,
        }
        if self.run_summary is not None:
            record["run_summary"] = self.run_summary
        if self.checkpoint_summary is not None:
            record["checkpoint_summary"] = self.checkpoint_summary
        return record


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


def write_cleanup_manifest(manifest: dict[str, Any], *, output_path: str | Path | None = None) -> Path:
    project_root = Path(manifest.get("metadata", {}).get("project_root") or ".").resolve()
    target = Path(output_path).expanduser() if output_path is not None else default_manifest_path(project_root, manifest)
    if not target.is_absolute():
        target = project_root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return target


def default_manifest_path(project_root: str | Path, manifest: dict[str, Any] | None = None) -> Path:
    root = Path(project_root).expanduser().resolve()
    generated = (manifest or {}).get("metadata", {}).get("generated_at") or _format_dt(_utc_now())
    stamp = generated.replace("-", "").replace(":", "").replace("+00:00", "Z")
    stamp = stamp.replace(".", "_")
    return root / "outputs" / "cleanup_manifests" / f"runtime_cleanup_{stamp}.json"


def apply_cleanup_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    confirm_delete: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if not confirm_delete:
        raise ValueError("Deletion refused: inspect the manifest and pass confirm_delete=True.")

    source = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    root = Path(project_root or manifest.get("metadata", {}).get("project_root") or ".").expanduser().resolve()
    scan_roots = tuple(Path(path).expanduser().resolve() for path in manifest.get("metadata", {}).get("scan_roots", []))
    tracked_paths = collect_git_tracked_paths(root)
    protected_paths = tuple(
        Path(record.get("path", "")).expanduser().resolve()
        for record in manifest.get("protected", [])
        if record.get("protected") or record.get("action") == "protect"
    )
    report: dict[str, Any] = {
        "metadata": {
            "generated_at": _format_dt(_utc_now()),
            "manifest_path": str(source),
            "project_root": str(root),
            "rules_version": RULES_VERSION,
        },
        "deleted": [],
        "skipped": [],
        "failed": [],
    }

    for record in manifest.get("candidates", []):
        result = _apply_one_record(
            record,
            project_root=root,
            scan_roots=scan_roots,
            tracked_paths=tracked_paths,
            protected_paths=protected_paths,
        )
        report[result["status"]].append(result["record"])

    report["summary"] = {
        "deleted_count": len(report["deleted"]),
        "skipped_count": len(report["skipped"]),
        "failed_count": len(report["failed"]),
        "deleted_size_bytes": sum(int(item.get("size_bytes") or 0) for item in report["deleted"]),
    }
    target = _delete_report_path(source, report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    report["metadata"]["report_path"] = str(target)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


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


def write_runtime_output_organize_manifest(
    manifest: dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> Path:
    project_root = Path(manifest.get("metadata", {}).get("project_root") or ".").resolve()
    target = (
        Path(output_path).expanduser()
        if output_path is not None
        else default_organize_manifest_path(project_root, manifest)
    )
    if not target.is_absolute():
        target = project_root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return target


def default_organize_manifest_path(project_root: str | Path, manifest: dict[str, Any] | None = None) -> Path:
    root = Path(project_root).expanduser().resolve()
    generated = (manifest or {}).get("metadata", {}).get("generated_at") or _format_dt(_utc_now())
    stamp = generated.replace("-", "").replace(":", "").replace("+00:00", "Z")
    stamp = stamp.replace(".", "_")
    return root / "outputs" / PARTITION_CLEANUP_MANIFESTS / f"runtime_organize_{stamp}.json"


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


def render_organize_summary(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary", {})
    lines = [
        "runtime output organize manifest",
        f"rules: {manifest.get('metadata', {}).get('rules_version')}",
        f"plans: {summary.get('plan_count', 0)}",
        f"move: {summary.get('by_action', {}).get('move', {}).get('count', 0)}",
        f"archive: {summary.get('by_action', {}).get('archive', {}).get('count', 0)}",
        f"protect: {summary.get('by_action', {}).get('protect', {}).get('count', 0)}",
        f"review: {summary.get('by_action', {}).get('review', {}).get('count', 0)}",
    ]
    return "\n".join(lines)


def render_cleanup_summary(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary", {})
    lines = [
        "runtime cleanup manifest",
        f"rules: {manifest.get('metadata', {}).get('rules_version')}",
        f"candidates: {summary.get('candidate_count', 0)}",
        f"candidate_size_bytes: {summary.get('candidate_total_size_bytes', 0)}",
        f"protected: {summary.get('protected_count', 0)}",
    ]
    return "\n".join(lines)


def collect_git_tracked_paths(project_root: str | Path) -> set[str]:
    root = Path(project_root).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()
    return {path for path in result.stdout.decode("utf-8").split("\0") if path}


def evaluate_protection(
    path: str | Path,
    *,
    project_root: str | Path,
    tracked_paths: set[str],
    active_run_dirs: Iterable[Path] = (),
) -> ProtectionDecision:
    root = Path(project_root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    reasons: list[str] = []
    rel = _relative_path(resolved, root)
    tracked = _is_git_tracked(rel, resolved, tracked_paths)
    top = rel.split("/", 1)[0] if rel and not rel.startswith("..") else ""
    if rel.startswith(".."):
        reasons.append("outside_project_root")
    if top in PROTECTED_ROOTS:
        reasons.append(f"protected_root:{top}")
    if top == ".git":
        reasons.append("protected_root:.git")
    if _is_current_mainline_output_path(resolved, root):
        reasons.append("current_mainline_output_partition")
    if tracked:
        reasons.append("git_tracked")
    for run_dir in active_run_dirs:
        if _path_contains_or_equals(resolved, run_dir) or _path_contains_or_equals(run_dir, resolved):
            reasons.append("active_run")
            break
    return ProtectionDecision(protected=bool(reasons), tracked=tracked, reasons=tuple(dict.fromkeys(reasons)))


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


def _apply_one_record(
    record: dict[str, Any],
    *,
    project_root: Path,
    scan_roots: tuple[Path, ...],
    tracked_paths: set[str],
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    path = Path(record.get("path", "")).expanduser().resolve()
    base = {"path": str(path), "relative_path": _relative_path(path, project_root), "rule_id": record.get("rule_id")}
    if record.get("protected"):
        return _skip(base, "manifest_record_is_protected")
    if not path.exists():
        return _skip(base, "path_missing")
    if scan_roots and not any(_path_contains_or_equals(root, path) for root in scan_roots):
        return _skip(base, "outside_manifest_scan_roots")
    for protected_path in protected_paths:
        if _path_contains_or_equals(path, protected_path) or _path_contains_or_equals(protected_path, path):
            base["protected_path"] = str(protected_path)
            return _skip(base, "manifest_protected_path_overlap")
    decision = evaluate_protection(path, project_root=project_root, tracked_paths=tracked_paths)
    if decision.protected:
        base["protection_reasons"] = list(decision.reasons)
        return _skip(base, "path_now_protected")
    if not _manifest_state_compatible(record, path):
        return _skip(base, "path_changed_since_manifest")
    try:
        size = _path_size_bytes(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        failed = dict(base)
        failed["reason"] = str(exc)
        return {"status": "failed", "record": failed}
    deleted = dict(base)
    deleted["size_bytes"] = size
    return {"status": "deleted", "record": deleted}


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


def _aggregate_by(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        bucket = result.setdefault(value, {"count": 0, "size_bytes": 0})
        bucket["count"] += 1
        bucket["size_bytes"] += int(record.get("size_bytes") or 0)
    return result


def _is_git_tracked(rel_path: str, path: Path, tracked_paths: set[str]) -> bool:
    if rel_path in tracked_paths:
        return True
    if path.is_dir():
        prefix = rel_path.rstrip("/") + "/"
        return any(item.startswith(prefix) for item in tracked_paths)
    return False


def _manifest_state_compatible(record: dict[str, Any], path: Path) -> bool:
    if int(record.get("size_bytes") or -1) != _path_size_bytes(path):
        return False
    recorded_mtime = record.get("mtime")
    current_mtime = _format_dt(_path_mtime(path))
    return not recorded_mtime or recorded_mtime == current_mtime


def _skip(base: dict[str, Any], reason: str) -> dict[str, Any]:
    skipped = dict(base)
    skipped["reason"] = reason
    return {"status": "skipped", "record": skipped}


def _delete_report_path(manifest_path: Path, report_path: str | Path | None) -> Path:
    if report_path is not None:
        return Path(report_path).expanduser().resolve()
    return manifest_path.with_name(manifest_path.stem + ".delete_report.json")


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


def _matches_current_mainline_output_partition(path: Path, project_root: Path) -> bool:
    rel = _relative_path(path, project_root)
    return rel in {f"outputs/{partition}" for partition in PROTECTED_MAINLINE_PARTITIONS}


def _is_current_mainline_output_path(path: Path, project_root: Path) -> bool:
    rel = _relative_path(path, project_root)
    protected = tuple(f"outputs/{partition}" for partition in PROTECTED_MAINLINE_PARTITIONS)
    return any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in protected)


def _is_personal_backup_archive(path: Path) -> bool:
    lower = path.name.lower()
    if not lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        return False
    return any(token in lower for token in ("backup", "bak", "old", "copy", "personal"))


def _top_level_part(path: Path, project_root: Path) -> str:
    rel = _relative_path(path, project_root)
    return rel.split("/", 1)[0] if rel and not rel.startswith("..") else ""


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), project_root.resolve()).replace(os.sep, "/")


def _path_contains_or_equals(parent: Path, child: Path) -> bool:
    parent = parent.resolve()
    child = child.resolve()
    return child == parent or child.is_relative_to(parent)


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


def _path_mtime(path: Path) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    except OSError:
        return None


def _format_dt(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def _ensure_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
