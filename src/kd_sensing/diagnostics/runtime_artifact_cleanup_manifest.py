import json
from pathlib import Path
from typing import Any

from kd_sensing.utils.runtime_output_layout import PARTITION_CLEANUP_MANIFESTS

from kd_sensing.diagnostics.runtime_artifact_cleanup_base import (
    _format_dt,
    _utc_now,
)


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
