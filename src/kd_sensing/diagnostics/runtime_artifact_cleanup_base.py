from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

from kd_sensing.utils.runtime_output_layout import PROTECTED_MAINLINE_PARTITIONS


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

def _matches_current_mainline_output_partition(path: Path, project_root: Path) -> bool:
    rel = _relative_path(path, project_root)
    return rel in {f"outputs/{partition}" for partition in PROTECTED_MAINLINE_PARTITIONS}

def _is_current_mainline_output_path(path: Path, project_root: Path) -> bool:
    rel = _relative_path(path, project_root)
    protected = tuple(f"outputs/{partition}" for partition in PROTECTED_MAINLINE_PARTITIONS)
    return any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in protected)

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
