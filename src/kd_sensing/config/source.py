from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kd_sensing.config.canonical import build_virtual_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.utils.paths import resolve_path


@dataclass(frozen=True)
class LoadedConfigSource:
    path: Path
    data: dict[str, Any]
    source_type: str


def load_config_source(config_path: str | Path) -> LoadedConfigSource:
    path = resolve_path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return LoadedConfigSource(path=path, data=safe_load_yaml(f.read()) or {}, source_type="file")
    file_cfg = build_virtual_config(path)
    if file_cfg is None:
        raise FileNotFoundError(f"Config file not found: {path}")
    return LoadedConfigSource(path=path, data=file_cfg, source_type="virtual")
