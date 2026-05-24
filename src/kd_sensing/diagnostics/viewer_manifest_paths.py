from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

def _raw_paths(row: pd.Series, data_root: Path) -> dict[str, str]:
    return {
        "image": _last_existing_path(row, "camera", data_root),
        "lidar": _last_existing_path(row, "lidar", data_root),
        "radar": _last_existing_path(row, "radar", data_root),
        "gps": _last_existing_path(row, "gps", data_root),
        "mmwave": _last_existing_path(row, "mmwave", data_root),
    }


def _all_source_paths(row: pd.Series, data_root: Path) -> list[str]:
    paths: list[str] = []
    for prefix in ("camera", "radar", "gps", "bs_gps", "lidar", "mmwave", "beam", "future_beam"):
        for value in _all_row_paths(row, prefix, data_root):
            paths.append(value)
            da_path = _radar_da_path(Path(value)) if value.endswith("_RA.npy") else None
            if da_path is not None:
                paths.append(str(da_path))
    seen: set[str] = set()
    unique = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique

def _radar_da_path(ra_path: Path) -> Path | None:
    text = str(ra_path)
    if "_RA" not in text:
        return None
    return Path(text.replace("_RA", "_DA"))

def _sorted_numbered_columns(columns: Iterable[str], prefix: str) -> list[str]:
    selected = []
    for col in columns:
        name = str(col)
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            selected.append(name)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


def _last_existing_path(row: pd.Series, prefix: str, data_root: Path) -> str:
    values = [str(row[col]) for col in _sorted_numbered_columns(row.index, prefix) if str(row[col]).strip() != "-99"]
    if not values:
        return ""
    return str(_resolve_row_path(values[-1], data_root))


def _all_row_paths(row: pd.Series, prefix: str, data_root: Path) -> list[str]:
    paths = []
    for col in _sorted_numbered_columns(row.index, prefix):
        value = str(row[col]).strip()
        if value == "-99" or not value:
            continue
        paths.append(str(_resolve_row_path(value, data_root)))
    return paths


def _resolve_row_path(value: str, data_root: Path) -> Path:
    text = str(value).strip()
    path = Path(text).expanduser()
    if path.is_absolute() and path.exists():
        return path
    relative = text.lstrip("/")
    return data_root / relative

__all__ = ["_all_row_paths", "_all_source_paths", "_last_existing_path", "_radar_da_path", "_raw_paths"]
