import datetime as dt
import os
from pathlib import Path
import re
from typing import Any, Iterable

from kd_sensing.utils.runtime_output_layout import DEFAULT_NON_RUN_PARTITIONS, is_default_outputs_root

from kd_sensing.diagnostics.run_index_base import (
    CHECKPOINT_SUFFIXES,
    DISCOVERY_FILENAMES,
    LOG_TAIL_BYTES,
    RunIndexFilters,
    _format_dt,
    _mtime,
)


def discover_run_dirs(
    outputs_root: str | Path,
    *,
    warnings: list[str] | None = None,
    include_legacy_containers: bool = False,
) -> list[Path]:
    root = Path(outputs_root)
    candidates: set[Path] = set()
    skip_default_partitions = is_default_outputs_root(root)
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        if skip_default_partitions and current_path == root:
            for dirname in list(dirnames):
                if dirname in DEFAULT_NON_RUN_PARTITIONS:
                    skipped = current_path / dirname
                    if warnings is not None:
                        warnings.append(f"skipped non-run outputs partition by default: {skipped}")
                    dirnames.remove(dirname)
                elif not include_legacy_containers and not _is_canonical_scan_partition_name(dirname):
                    skipped = current_path / dirname
                    if warnings is not None:
                        warnings.append(f"skipped non-canonical outputs partition by default: {skipped}")
                    dirnames.remove(dirname)
        if skip_default_partitions and current_path != root:
            _prune_default_outputs_children(root, current_path, dirnames, warnings=warnings)
        if _directory_contains_run_marker(current_path, filenames):
            candidates.add(current_path)
            dirnames[:] = []
            continue
        if current_path.name == "checkpoints" and any(_is_checkpoint(current_path / name) for name in filenames):
            candidates.add(current_path.parent)
            dirnames[:] = []
            continue
        if current_path.name == "tensorboard" and any(_is_tensorboard_event(current_path / name) for name in filenames):
            candidates.add(current_path.parent)
            dirnames[:] = []
            continue
        for filename in filenames:
            path = current_path / filename
            if _is_discovery_artifact(path):
                candidates.add(path.parent)
            elif _is_checkpoint(path):
                candidates.add(path.parent.parent if path.parent.name == "checkpoints" else path.parent)
            elif _is_tensorboard_event(path):
                candidates.add(path.parent.parent if path.parent.name == "tensorboard" else path.parent)
    return sorted(candidates, key=lambda path: path.as_posix())

def collect_log_records(log_roots: Iterable[Path], *, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in log_roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in sorted(paths):
            if path.suffix not in {".log", ".out", ".err", ".txt", ".status", ""}:
                continue
            tail = _read_text_tail(path, warnings=warnings)
            failure = parse_failure_patterns(tail)
            records.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "mtime": _format_dt(_mtime(path)),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "failure": failure,
                    "tail": tail,
                }
            )
    return records

def associate_logs(run_dir: Path, *, config: dict[str, Any], logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = {
        run_dir.name.lower(),
        str(config.get("experiment_name") or "").lower(),
        str(config.get("output_run_name") or "").lower(),
    }
    tokens = {token for token in tokens if token and token != "none"}
    matches: list[tuple[int, dict[str, Any]]] = []
    for record in logs:
        haystack = f"{record.get('path', '')}\n{record.get('tail', '')}".lower()
        score = sum(1 for token in tokens if token in haystack)
        if str(run_dir).lower() in haystack:
            score += 4
        if score:
            public = {key: value for key, value in record.items() if key != "tail"}
            matches.append((score, public))
    return [record for _, record in sorted(matches, key=lambda item: (-item[0], str(item[1].get("path"))))[:5]]

def parse_failure_patterns(text: str) -> dict[str, Any]:
    if not text:
        return {"kind": None, "message": None, "waiting_for": None}
    lower = text.lower()
    if re.search(r"(^|\n)\s*(?:.+\s)?killed\s*$", text, flags=re.IGNORECASE):
        return {"kind": "killed", "message": "log contains Killed", "waiting_for": None}
    if "waiting" in lower and "checkpoint" in lower:
        wait_path = _extract_checkpoint_path(text)
        return {
            "kind": "waiting",
            "message": "log indicates waiting for checkpoint",
            "waiting_for": wait_path,
        }
    if "traceback (most recent call last)" in lower:
        return {"kind": "traceback", "message": _last_nonempty_line(text), "waiting_for": None}
    if "error conda.cli.main_run" in lower or "condaerror" in lower:
        return {"kind": "conda_failed", "message": _last_nonempty_line(text), "waiting_for": None}
    return {"kind": None, "message": None, "waiting_for": None}

def _directory_contains_run_marker(path: Path, filenames: Iterable[str]) -> bool:
    names = set(filenames)
    if names & DISCOVERY_FILENAMES:
        return True
    checkpoint_dir = path / "checkpoints"
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return False
    try:
        return any(_is_checkpoint(item) for item in checkpoint_dir.iterdir() if item.is_file())
    except OSError:
        return False

def _is_canonical_scan_partition_name(name: str) -> bool:
    if name in {"analysis", "visual_analysis", "evaluations", "training"}:
        return True
    if re.fullmatch(r"scene\d+", name):
        return True
    return name.startswith("scenegroup_")

def _shallow_child_may_contain_run(path: Path) -> bool:
    try:
        filenames = [item.name for item in path.iterdir() if item.is_file()]
    except OSError:
        return False
    return _directory_contains_run_marker(path, filenames)

def _prune_default_outputs_children(
    outputs_root: Path,
    current_path: Path,
    dirnames: list[str],
    *,
    warnings: list[str] | None,
) -> None:
    try:
        rel_parts = current_path.relative_to(outputs_root).parts
    except ValueError:
        return
    if not rel_parts:
        return
    partition = rel_parts[0]
    for dirname in list(dirnames):
        child = current_path / dirname
        if dirname in {"best_checkpoints", "cache", "feature_cache", "features"}:
            _remove_pruned_dirname(dirnames, dirname, child, warnings=warnings)
            continue
        if partition in {"analysis", "visual_analysis", "training"} and len(rel_parts) == 1:
            if not _shallow_child_may_contain_run(child):
                _remove_pruned_dirname(dirnames, dirname, child, warnings=warnings)
        elif partition == "evaluations" and len(rel_parts) == 2:
            if not _shallow_child_may_contain_run(child):
                _remove_pruned_dirname(dirnames, dirname, child, warnings=warnings)

def _remove_pruned_dirname(dirnames: list[str], dirname: str, path: Path, *, warnings: list[str] | None) -> None:
    if warnings is not None:
        warnings.append(f"skipped non-run outputs subtree by default: {path}")
    dirnames.remove(dirname)

def _is_discovery_artifact(path: Path) -> bool:
    return path.name in DISCOVERY_FILENAMES

def _is_checkpoint(path: Path) -> bool:
    if path.suffix == ".json" and any(path.name.endswith(suffix + ".json") for suffix in CHECKPOINT_SUFFIXES):
        return False
    return path.suffix.lower() in CHECKPOINT_SUFFIXES

def _is_tensorboard_event(path: Path) -> bool:
    return path.name.startswith("events.out.tfevents")

def _tensorboard_event_paths(run_dir: Path) -> list[Path]:
    roots = [run_dir / "tensorboard", run_dir]
    events: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for path in root.glob("events.out.tfevents*"):
                if path.is_file():
                    events.add(path)
        except OSError:
            continue
    return sorted(events, key=lambda path: path.as_posix())

def _read_text_tail(path: Path, *, warnings: list[str] | None = None) -> str:
    try:
        with path.open("rb") as f:
            try:
                f.seek(-LOG_TAIL_BYTES, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode("utf-8", errors="replace")
    except OSError as exc:
        if warnings is not None:
            warnings.append(f"failed to read log {path}: {exc}")
        return ""

def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()[:300]
    return None

def _extract_checkpoint_path(text: str) -> str | None:
    match = re.search(r"([^\s'\"]+\.(?:pth|pt|ckpt))", text)
    return match.group(1) if match else None
