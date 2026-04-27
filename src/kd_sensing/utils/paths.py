"""Project and resource path helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike[str]]


def project_root(start: Optional[PathLike] = None) -> Path:
    """Resolve the repository root from any working directory."""

    env_root = os.environ.get("KD_SENSING_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
        if (candidate / "openspec").exists() and (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


def resolve_path(path: Optional[PathLike], base: Optional[PathLike] = None) -> Optional[Path]:
    """Resolve a user path relative to the project root unless already absolute."""

    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = Path(base).expanduser().resolve() if base is not None else project_root()
    return (root / candidate).resolve()


def data_dir(path: Optional[PathLike] = None) -> Path:
    return resolve_path(path or "dataset")


def weights_dir(path: Optional[PathLike] = None) -> Path:
    return resolve_path(path or "All_models")


def config_dir(path: Optional[PathLike] = None) -> Path:
    return resolve_path(path or "configs")


def output_dir(path: Optional[PathLike] = None) -> Path:
    target = resolve_path(path or "outputs")
    target.mkdir(parents=True, exist_ok=True)
    return target

