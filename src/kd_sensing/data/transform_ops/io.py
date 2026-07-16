import os
from pathlib import Path
import tempfile

import numpy as np


def joined_resource(data_root: str | Path, rel_path: str) -> Path:
    root = Path(data_root).resolve()
    value = str(rel_path).strip().replace("\\", "/")
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Resource path must be relative to data_root: {rel_path!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Resource path escapes data_root: {rel_path!r}")
    return resolved


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as f:
            tmp_name = f.name
            np.save(f, array)
        os.replace(tmp_name, target)
    finally:
        if tmp_name is not None:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink()


__all__ = ["atomic_save_npy", "joined_resource"]
