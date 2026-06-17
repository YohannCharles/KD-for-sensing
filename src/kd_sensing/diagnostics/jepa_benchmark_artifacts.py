from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.difficulty import (
    DifficultyContext,
    apply_difficulty_pipeline,
    normalize_difficulty_profiles,
)
from kd_sensing.data.difficulty.presets import (
    PREDICTIVE_JEPA_CANONICAL_CONDITIONS,
    PREDICTIVE_JEPA_CONDITION_IDS,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    SCENARIO_D_CANONICAL_CONDITIONS,
    SCENARIO_D_CONDITION_IDS,
    SCENARIO_D_SUITE_TYPE,
    normalize_predictive_jepa_condition_id,
    normalize_predictive_jepa_operator_params,
    normalize_scenario_d_condition_id,
    normalize_scenario_d_operator_params,
    predictive_jepa_condition,
    scenario_d_condition,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path

from kd_sensing.diagnostics.jepa_benchmark_common import *


@dataclass
class OutputRegistry:
    root: Path
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def skipped_output(self, path: str | Path, *, reason: str, kind: str) -> None:
        self.skipped.append(
            {
                "path": _relative_to_root(Path(path), self.root),
                "kind": kind,
                "status": "skipped",
                "reason": str(reason),
            }
        )

    def list_outputs(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self.root.exists():
            for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
                records.append(
                    {
                        "path": _relative_to_root(path, self.root),
                        "kind": _output_kind(path),
                        "status": "generated",
                        "size_bytes": int(path.stat().st_size),
                    }
                )
        records.extend(self.skipped)
        return records


def _validate_existing_path(
    value: Any,
    *,
    field: str,
    manifest_path: str | Path | None,
    validate_paths: bool,
) -> None:
    if not value:
        raise BenchmarkManifestError(f"{field} is required in {manifest_path or '<memory>'}.")
    if not validate_paths:
        return
    path = resolve_path(str(value))
    if path is None or not path.exists():
        raise FileNotFoundError(f"{field} path does not exist for benchmark manifest: {value}")


def _command_uses_kd_env(command: Any) -> bool:
    if isinstance(command, str):
        parts = command.split()
    elif isinstance(command, (list, tuple)):
        parts = [str(item) for item in command]
    else:
        return False
    return len(parts) >= 4 and parts[0:4] == ["conda", "run", "-n", "kd_mm_beam"]


def _require_mapping(cfg: Mapping[str, Any], key: str, path: str | Path | None) -> None:
    if not isinstance(cfg.get(key), Mapping):
        raise BenchmarkManifestError(f"{key} must be a mapping in {path or '<memory>'}.")


def _require_list(cfg: Mapping[str, Any], key: str, path: str | Path | None) -> None:
    if not isinstance(cfg.get(key), list):
        raise BenchmarkManifestError(f"{key} must be a list in {path or '<memory>'}.")


def _load_mapping_text(text: str, *, path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
    else:
        parsed = safe_load_yaml(text) or {}
    if not isinstance(parsed, Mapping):
        raise BenchmarkManifestError(f"Benchmark manifest must be a mapping: {path}")
    return dict(parsed)


def _resolve_output_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if resolved is None:
        raise BenchmarkManifestError(f"Output directory could not be resolved: {path}")
    return resolved


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Benchmark output directory is not empty. Use --force to write into it: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_existing_user_path(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if resolved is None:
        raise FileNotFoundError(f"Path could not be resolved: {path}")
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def _resolve_artifact_path(
    *,
    explicit: str | Path | None,
    manifest: Mapping[str, Any],
    manifest_file: Path | None,
    filename: str,
    output_key: str,
) -> Path | None:
    if explicit:
        return resolve_path(explicit)
    output_dir = manifest.get("output_dir") or manifest.get("outputs", {}).get("output_dir")
    output_files = manifest.get("output_files", {}) if isinstance(manifest.get("output_files"), Mapping) else {}
    if output_dir and output_key in output_files:
        base = resolve_path(str(output_dir))
        return (base / str(output_files[output_key])).resolve() if base is not None else None
    if manifest_file is not None:
        candidate = manifest_file.parent / "tables" / filename
        if candidate.exists():
            return candidate
        sibling = manifest_file.parent / filename
        if sibling.exists():
            return sibling
    return None


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_scalar(row.get(key, "")) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _output_formats(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("figures", {}).get("formats", ["png"]) if isinstance(manifest.get("figures"), Mapping) else ["png"]
    if isinstance(raw, str):
        raw = [raw]
    formats: list[str] = []
    for item in raw:
        fmt = str(item).strip().lower().lstrip(".")
        if fmt and fmt not in formats:
            formats.append(fmt)
    return tuple(formats or ["png"])


def _git_status_short() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return f"unavailable:{exc}"
    if result.returncode != 0:
        return f"unavailable:{result.stderr.strip()}"
    return result.stdout.strip()


__all__ = [
    "OutputRegistry",
    "_command_uses_kd_env",
    "_git_status_short",
    "_load_mapping_text",
    "_output_formats",
    "_prepare_output_dir",
    "_read_csv",
    "_require_list",
    "_require_mapping",
    "_resolve_artifact_path",
    "_resolve_existing_user_path",
    "_resolve_output_dir",
    "_validate_existing_path",
    "_write_csv",
]
