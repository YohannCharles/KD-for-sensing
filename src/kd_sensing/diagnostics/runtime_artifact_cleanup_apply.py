import json
import os
from pathlib import Path
import shutil
from typing import Any

from kd_sensing.diagnostics.runtime_artifact_cleanup_base import (
    MANIFEST_SCHEMA_VERSION,
    RULES_VERSION,
    collect_git_tracked_paths,
    evaluate_protection,
    _filesystem_type,
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
    metadata = _validate_manifest_metadata(manifest, project_root=project_root)
    root = Path(metadata["project_root"]).expanduser().resolve()
    scan_roots = tuple(Path(path).expanduser().resolve() for path in metadata["scan_roots"])
    allowed_roots = tuple(Path(path).expanduser().resolve() for path in metadata["allowed_roots"])
    _validate_manifest_records(manifest.get("candidates", []))
    tracked_paths = collect_git_tracked_paths(root, required=True)
    protected_paths = tuple(
        Path(str(record["path"])).expanduser().resolve()
        for record in manifest.get("protected", [])
        if (record.get("protected") or record.get("action") == "protect") and str(record.get("path") or "").strip()
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
            allowed_roots=allowed_roots,
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
    allowed_roots: tuple[Path, ...],
    tracked_paths: set[str],
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    raw_path = Path(os.path.abspath(Path(str(record["path"])).expanduser()))
    current_type = _filesystem_type(raw_path)
    base = {
        "path": str(raw_path),
        "relative_path": _relative_path(raw_path, project_root),
        "rule_id": record.get("rule_id"),
    }
    if record.get("protected"):
        return _skip(base, "manifest_record_is_protected")
    if current_type == "missing":
        return _skip(base, "path_missing")
    if current_type != record["filesystem_type"]:
        base["manifest_filesystem_type"] = record["filesystem_type"]
        base["current_filesystem_type"] = current_type
        return _skip(base, "filesystem_type_changed_since_manifest")
    path = raw_path.resolve()
    if path == project_root or path in scan_roots or path in allowed_roots:
        return _skip(base, "protected_root_candidate")
    if not any(raw_path.is_relative_to(root) and _path_contains_or_equals(root, path) for root in allowed_roots):
        return _skip(base, "outside_manifest_allowed_roots")
    if scan_roots and not any(
        raw_path.is_relative_to(root) and _path_contains_or_equals(root, path) for root in scan_roots
    ):
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
        if current_type == "directory":
            shutil.rmtree(raw_path)
        else:
            raw_path.unlink()
    except OSError as exc:
        failed = dict(base)
        failed["reason"] = str(exc)
        return {"status": "failed", "record": failed}
    deleted = dict(base)
    deleted["size_bytes"] = size
    return {"status": "deleted", "record": deleted}


def _validate_manifest_metadata(
    manifest: dict[str, Any],
    *,
    project_root: str | Path | None,
) -> dict[str, Any]:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Cleanup manifest metadata is required.")
    if metadata.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported cleanup manifest schema_version; generate a fresh manifest.")
    if metadata.get("rules_version") != RULES_VERSION:
        raise ValueError("Unsupported cleanup manifest rules_version; generate a fresh manifest.")
    raw_root = str(metadata.get("project_root") or "").strip()
    if not raw_root:
        raise ValueError("Cleanup manifest project_root is required.")
    manifest_root = Path(raw_root).expanduser().resolve()
    if project_root is not None and Path(project_root).expanduser().resolve() != manifest_root:
        raise ValueError("Cleanup manifest project_root does not match the requested project_root.")
    scan_roots = _validated_roots(metadata.get("scan_roots"), field="scan_roots", project_root=manifest_root)
    allowed_roots = _validated_roots(metadata.get("allowed_roots"), field="allowed_roots", project_root=manifest_root)
    for scan_root in scan_roots:
        if not any(_path_contains_or_equals(allowed_root, scan_root) for allowed_root in allowed_roots):
            raise ValueError(f"Cleanup scan root is outside allowed_roots: {scan_root}")
    return {**metadata, "project_root": str(manifest_root), "scan_roots": scan_roots, "allowed_roots": allowed_roots}


def _validated_roots(value: Any, *, field: str, project_root: Path) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Cleanup manifest {field} must be a non-empty list.")
    roots: list[Path] = []
    for raw in value:
        text = str(raw or "").strip()
        if not text:
            raise ValueError(f"Cleanup manifest {field} contains an empty path.")
        root = Path(text).expanduser().resolve()
        if root == project_root or not _path_contains_or_equals(project_root, root):
            raise ValueError(f"Cleanup manifest {field} contains an unsafe root: {root}")
        roots.append(root)
    return tuple(roots)


def _validate_manifest_records(records: Any) -> None:
    if not isinstance(records, list):
        raise ValueError("Cleanup manifest candidates must be a list.")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not str(record.get("path") or "").strip():
            raise ValueError(f"Cleanup manifest candidate {index} has an empty path.")
        filesystem_type = record.get("filesystem_type")
        if filesystem_type not in {"regular_file", "directory", "symlink", "other"}:
            raise ValueError(
                f"Cleanup manifest candidate {index} has an invalid filesystem_type; generate a fresh manifest."
            )

def _delete_report_path(manifest_path: Path, report_path: str | Path | None) -> Path:
    if report_path is not None:
        return Path(report_path).expanduser().resolve()
    return manifest_path.with_name(manifest_path.stem + ".delete_report.json")
