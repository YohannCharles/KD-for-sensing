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
from kd_sensing.config.migration_guards import reject_removed_image_path_config
from kd_sensing.config.normalization import normalize_loaded_config
from kd_sensing.config.parsing import parse_scalar, parse_simple_yaml, safe_load_yaml
from kd_sensing.config.source import load_config_source
from kd_sensing.config.validation import validate_loaded_config


def load_config(config_path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    file_cfg = {}
    if config_path:
        source = load_config_source(config_path)
        file_cfg = source.data
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
    reject_removed_image_path_config(cfg)
    validate_loaded_config(cfg)
    return cfg


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
        set_by_dotted_key(result, key.strip(), parse_scalar(raw_value.strip()))
    return result


def set_by_dotted_key(target: dict[str, Any], key: str, value: Any) -> None:
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

