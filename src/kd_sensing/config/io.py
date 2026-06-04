"""YAML config loading and command-line override parsing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    yaml = None

from kd_sensing.config.defaults import DEFAULT_CONFIG
from kd_sensing.config.migration_guards import (
    reject_removed_config_path,
    reject_removed_image_path_config,
    reject_removed_kd_config,
    reject_removed_override_key,
    reject_retired_hist_config,
)
from kd_sensing.config.normalization import normalize_loaded_config
from kd_sensing.config.parsing import parse_scalar, parse_simple_yaml, safe_load_yaml
from kd_sensing.config.source import load_config_source
from kd_sensing.config.validation import validate_loaded_config


def load_config(config_path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    file_cfg = {}
    if config_path:
        reject_removed_config_path(config_path)
        source = load_config_source(config_path)
        file_cfg = _resolve_base_config(source)
        cfg = deep_merge(cfg, file_cfg)
    override_cfg = parse_overrides(overrides) if overrides else {}
    if override_cfg:
        cfg = deep_merge(cfg, override_cfg)
    file_cfg_for_keys = file_cfg if config_path else {}
    override_changes_objective = _has_dotted_key(override_cfg, "experiment.objective")
    explicit_early_metric = _has_dotted_key(override_cfg, "training.early_stopping_metric") or (
        _has_dotted_key(file_cfg_for_keys, "training.early_stopping_metric") and not override_changes_objective
    )
    explicit_early_mode = _has_dotted_key(override_cfg, "training.early_stopping_mode") or (
        _has_dotted_key(file_cfg_for_keys, "training.early_stopping_mode") and not override_changes_objective
    )
    normalize_loaded_config(
        cfg,
        file_cfg=file_cfg_for_keys,
        override_cfg=override_cfg,
        explicit_early_stopping_metric=explicit_early_metric,
        explicit_early_stopping_mode=explicit_early_mode,
    )
    reject_removed_kd_config(cfg)
    reject_retired_hist_config(cfg)
    reject_removed_image_path_config(cfg)
    validate_loaded_config(cfg)
    return cfg


def _resolve_base_config(source: Any, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    file_cfg = copy.deepcopy(source.data)
    base_entries = file_cfg.pop("_base_", None)
    if not base_entries:
        return file_cfg
    source_path = Path(source.path)
    if source_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, source_path))
        raise ValueError(f"Circular config _base_ reference: {chain}")
    if isinstance(base_entries, (str, Path)):
        base_entries = [base_entries]
    if not isinstance(base_entries, list):
        raise ValueError("Config _base_ must be a path string or a list of path strings.")
    merged: dict[str, Any] = {}
    for entry in base_entries:
        base_path = Path(str(entry))
        if not base_path.is_absolute():
            base_path = source_path.parent / base_path
        reject_removed_config_path(base_path)
        base_source = load_config_source(base_path)
        merged = deep_merge(merged, _resolve_base_config(base_source, (*stack, source_path)))
    return deep_merge(merged, file_cfg)


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            json.dump(cfg, f, indent=2)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in overrides:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override must use key=value format, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        reject_removed_override_key(key)
        set_by_dotted_key(result, key, parse_scalar(raw_value.strip()))
    return result


def set_by_dotted_key(target: dict[str, Any], key: str, value: Any) -> None:
    reject_removed_override_key(key)
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid dotted override key: {key}")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set nested override through non-dict key: {key}")
    cursor[parts[-1]] = value


def _has_dotted_key(target: dict[str, Any], key: str) -> bool:
    cursor: Any = target
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True
