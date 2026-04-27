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
from kd_sensing.utils.paths import resolve_path


def load_config(config_path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = resolve_path(config_path)
        with path.open("r", encoding="utf-8") as f:
            file_cfg = safe_load_yaml(f.read()) or {}
        cfg = deep_merge(cfg, file_cfg)
    if overrides:
        cfg = deep_merge(cfg, parse_overrides(overrides))
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


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def safe_load_yaml(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text)
    return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple nested mapping subset used by this repo's configs."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line without ':': {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root
