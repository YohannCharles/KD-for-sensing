from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.preprocessing.raymobtime_s008_common import (
    _load_np_arrays,
    _missing_required_paths,
    _nullable_float,
    _read_coord_csv,
    _required_paths,
    _value_counts,
    resolve_raymobtime_paths,
)

def audit_s008_files(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    missing = _missing_required_paths(paths.data_root)
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Raymobtime s008 audit missing required path(s): {missing_text}")
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    npz_summary = _summarize_npz_tree(paths.data_root / "baseline_data")
    csv_summary = _summarize_coord_csv(paths.csv_path)
    (paths.output_dir / "audit_summary.json").write_text(
        json.dumps(
            {
                "data_root": str(paths.data_root),
                "required_paths": {str(path.relative_to(paths.data_root)): path.exists() for path in _required_paths(paths.data_root)},
                "npz_files": len(npz_summary),
                "csv": csv_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(npz_summary).to_csv(paths.output_dir / "npz_shapes.csv", index=False)
    (paths.output_dir / "csv_summary.json").write_text(json.dumps(csv_summary, indent=2), encoding="utf-8")
    return {
        "audit_summary": str(paths.output_dir / "audit_summary.json"),
        "npz_shapes": str(paths.output_dir / "npz_shapes.csv"),
        "csv_summary": str(paths.output_dir / "csv_summary.json"),
    }

def _summarize_npz_tree(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".npz", ".npy"}:
            continue
        try:
            arrays = _load_np_arrays(path)
        except Exception as exc:
            rows.append({"path": str(path), "key": None, "error": str(exc)})
            continue
        for key, value in arrays.items():
            arr = np.asarray(value)
            row = {
                "path": str(path),
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            if arr.size and np.issubdtype(arr.dtype, np.number):
                row.update(
                    {
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "mean": float(np.nanmean(arr)),
                    }
                )
            rows.append(row)
    return rows


def _summarize_coord_csv(path: Path) -> dict[str, Any]:
    frame = _read_coord_csv(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    for column in ("Val", "LOS"):
        summary[f"{column}_distribution"] = _value_counts(frame[column])
    for column in ("EpisodeID", "SceneID", "VehicleArrayID"):
        summary[f"{column}_unique"] = int(frame[column].nunique(dropna=True))
    for column in ("x", "y", "z"):
        values = pd.to_numeric(frame[column], errors="coerce")
        summary[f"{column}_range"] = {
            "min": _nullable_float(values.min()),
            "max": _nullable_float(values.max()),
        }
    return summary




__all__ = ["audit_s008_files"]
