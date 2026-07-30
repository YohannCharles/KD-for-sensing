"""YAML config loading and command-line override parsing."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from kd_sensing.config.normalization import normalize_loaded_config
from kd_sensing.config.parsing import parse_scalar, safe_load_yaml
from kd_sensing.config.validation import validate_loaded_config
from kd_sensing.utils.paths import resolve_path


@dataclass(frozen=True)
class LoadedConfigSource:
    path: Path
    data: dict[str, Any]
    source_type: str


def load_config(config_path: str | Path, overrides: Optional[Iterable[str]] = None) -> dict[str, Any]:
    cfg = _resolve_base_config(load_config_source(config_path))
    override_items = list(overrides or ())
    override_cfg = parse_overrides(override_items) if override_items else {}
    _validate_override_paths(cfg, override_items)
    if override_cfg:
        cfg = deep_merge(cfg, override_cfg)
    normalize_loaded_config(cfg)
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
        base_source = load_config_source(base_path)
        merged = deep_merge(merged, _resolve_base_config(base_source, (*stack, source_path)))
    return deep_merge(merged, file_cfg)


def load_config_source(config_path: str | Path) -> LoadedConfigSource:
    path = resolve_path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return LoadedConfigSource(path=path, data=safe_load_yaml(f.read()) or {}, source_type="file")
    raise FileNotFoundError(f"Config file not found: {path}")


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


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
        set_by_dotted_key(result, key, parse_scalar(raw_value.strip()))
    return result


def _validate_override_paths(cfg: dict[str, Any], overrides: Iterable[str]) -> None:
    for item in overrides:
        if not item:
            continue
        key = item.split("=", 1)[0].strip()
        cursor: Any = cfg
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"Unknown config override path: {key}")
            cursor = cursor[part]


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
