import json
from pathlib import Path
import shutil
from typing import Any

from kd_sensing.diagnostics.runtime_artifact_cleanup_base import (
    RULES_VERSION,
    collect_git_tracked_paths,
    evaluate_protection,
    _format_dt,
    _manifest_state_compatible,
    _path_contains_or_equals,
    _path_size_bytes,
    _relative_path,
    _skip,
    _utc_now,
)


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

def _delete_report_path(manifest_path: Path, report_path: str | Path | None) -> Path:
    if report_path is not None:
        return Path(report_path).expanduser().resolve()
    return manifest_path.with_name(manifest_path.stem + ".delete_report.json")
