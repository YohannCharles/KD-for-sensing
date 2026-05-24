from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.layouts import raymobtime_s008_layout
from kd_sensing.utils.paths import resolve_path

INDEX_COLUMNS = (
    "sample_id",
    "split",
    "source_split",
    "source_split_index",
    "valid_index",
    "source_row_index",
    "EpisodeID",
    "SceneID",
    "VehicleArrayID",
    "VehicleName",
    "x",
    "y",
    "z",
    "LOS",
    "coord_row_in_split",
)
REQUIRED_CSV_COLUMNS = ("Val", "EpisodeID", "SceneID", "VehicleArrayID", "VehicleName", "x", "y", "z", "LOS")
RAY_FEATURE_NAMES = (
    "max_power_dbm",
    "mean_power_dbm_valid",
    "sum_power_linear",
    "num_valid_rays",
    "min_toa",
    "mean_toa",
    "strongest_ray_toa",
    "strongest_ray_elev_aod",
    "strongest_ray_az_aod",
    "strongest_ray_elev_aoa",
    "strongest_ray_az_aoa",
    "strongest_ray_phase",
    "power_spread_db",
    "toa_spread",
)
SOURCE_SPLIT_ORDER = ("train", "validation", "test")
OUTPUT_SPLIT_TO_INDEX_NAME = {"train": "train", "validation": "val", "test": "test"}
HDF5_SUFFIXES = {".hdf5", ".h5"}
FALLBACK_LINK_QUALITY_DBM = -120.0
RAYMOBTIME_S008_HDF5_FIELD_MAPPING = {
    "dataset": "allEpisodeData",
    "shape": ["SceneID", "VehicleArrayID", "path", "field"],
    "episode_source": "filename suffix _e<episode>",
    "fields": {
        "0": "power_dbm",
        "1": "toa",
        "2": "elev_aod",
        "3": "az_aod",
        "4": "elev_aoa",
        "5": "az_aoa",
        "8": "phase",
    },
}
RAYMOBTIME_HDF5_CONTEXT = "Raymobtime s008 HDF5 ray-tracing parser"

_RAY_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sample_id": ("sample_id", "sampleid", "sample", "snapshot_id"),
    "EpisodeID": ("episodeid", "episode_id", "episode", "episode_index", "e"),
    "SceneID": ("sceneid", "scene_id", "scene", "scene_index", "s"),
    "VehicleArrayID": (
        "vehiclearrayid",
        "vehicle_array_id",
        "vehicleid",
        "vehicle_id",
        "vehicle",
        "receiver_id",
        "rx_id",
        "rx",
        "v",
    ),
    "path_index": ("path_index", "pathid", "path_id", "path", "ray_id", "ray_index"),
    "power_dbm": (
        "power_dbm",
        "received_power_dbm",
        "path_power_dbm",
        "pr_dbm",
        "pr",
        "power",
    ),
    "toa": ("toa", "to_a", "time_of_arrival", "delay", "tau"),
    "elev_aod": ("elev_aod", "elevation_aod", "departure_elevation", "aod_elevation", "dod_theta"),
    "az_aod": ("az_aod", "azimuth_aod", "departure_azimuth", "aod_azimuth", "dod_phi"),
    "elev_aoa": ("elev_aoa", "elevation_aoa", "arrival_elevation", "aoa_elevation", "doa_theta"),
    "az_aoa": ("az_aoa", "azimuth_aoa", "arrival_azimuth", "aoa_azimuth", "doa_phi"),
    "phase": ("phase", "path_phase", "phase_rad", "phase_deg"),
}

@dataclass(frozen=True)
class RaymobtimePaths:
    data_root: Path
    output_dir: Path
    cache_dir: Path

    @property
    def csv_path(self) -> Path:
        return self.data_root / "raw_data" / "CoordVehiclesRxPerScene_s008.csv"

    @property
    def ray_zip_path(self) -> Path:
        return self.data_root / "raw_data" / "ray_tracing_data_s008_carrier60GHz.zip"


def resolve_raymobtime_paths(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> RaymobtimePaths:
    layout = raymobtime_s008_layout()
    root = resolve_path(data_root or layout.root)
    out = resolve_path(output_dir or "outputs/raymobtime_s008/audit")
    cache = resolve_path(cache_dir or root / "cache")
    return RaymobtimePaths(data_root=root, output_dir=out, cache_dir=cache)



def _read_coord_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raymobtime s008 coordinate CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Raymobtime s008 coordinate CSV {path} missing required columns: {missing}.")
    return frame


def _sample_id(episode: Any, scene: Any, vehicle: Any) -> str:
    return f"e{_int_token(episode)}_s{_int_token(scene)}_v{_int_token(vehicle)}"


def _int_token(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _split_index_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "train": cache_dir / "index_train.csv",
        "val": cache_dir / "index_val.csv",
        "test": cache_dir / "index_test.csv",
    }


def _load_beam_output(path: Path, *, key: str | None) -> np.ndarray:
    arrays = _load_split_arrays(path, key=key)
    if "all" in arrays:
        return arrays["all"]
    return np.concatenate([arrays[split] for split in SOURCE_SPLIT_ORDER if split in arrays], axis=0)


def _load_split_arrays(path: Path, *, key: str | None, allow_missing: bool = False) -> dict[str, np.ndarray]:
    files = sorted(item for item in path.iterdir() if item.suffix.lower() in {".npz", ".npy"}) if path.exists() else []
    if not files:
        if allow_missing:
            return {}
        raise FileNotFoundError(f"No .npz or .npy file found under {path}.")
    split_arrays: dict[str, np.ndarray] = {}
    unsplit_arrays: list[np.ndarray] = []
    for file in files:
        array = _select_np_array(_load_np_arrays(file), key=key, source=file)
        split = _split_from_filename(file)
        if split is None:
            unsplit_arrays.append(array)
            continue
        if split in split_arrays:
            raise ValueError(f"Multiple Raymobtime files for split '{split}' found under {path}.")
        split_arrays[split] = array
    if split_arrays:
        return split_arrays
    if len(unsplit_arrays) == 1:
        return {"all": unsplit_arrays[0]}
    raise ValueError(
        f"Raymobtime directory {path} contains multiple arrays but no train/validation/test split token in file names."
    )


def _select_np_array(arrays: dict[str, np.ndarray], *, key: str | None, source: Path) -> np.ndarray:
    if key is not None:
        if key not in arrays:
            raise ValueError(f"Raymobtime array {source} does not contain key '{key}'. Available keys: {list(arrays)}.")
        return np.asarray(arrays[key])
    numeric = [(name, np.asarray(value)) for name, value in arrays.items() if np.issubdtype(np.asarray(value).dtype, np.number)]
    if not numeric:
        raise ValueError(f"Raymobtime array {source} contains no numeric arrays.")
    numeric.sort(key=lambda item: (item[1].ndim not in {1, 2, 3}, item[0]))
    return numeric[0][1]


def _split_from_filename(path: Path) -> str | None:
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    if "train" in tokens or "training" in tokens:
        return "train"
    if "validation" in tokens or "val" in tokens:
        return "validation"
    if "test" in tokens:
        return "test"
    return None


def _load_np_arrays(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {key: loaded[key] for key in loaded.files}
    return {"array": loaded}


def _row_selector_for_array(array: np.ndarray, valid_index: pd.DataFrame):
    if len(array) >= int(valid_index["source_row_index"].max()) + 1 and len(array) != len(valid_index):
        return lambda frame: frame["source_row_index"].to_numpy(dtype=np.int64)
    if len(array) < len(valid_index):
        raise ValueError(
            f"Beam output has {len(array)} rows but Raymobtime valid receiver index needs {len(valid_index)} rows."
        )
    return lambda frame: frame["valid_index"].to_numpy(dtype=np.int64)



def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fingerprint(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _nullable_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _los_series_to_float(series: pd.Series) -> np.ndarray:
    return np.asarray([_los_value(value) for value in series], dtype=np.float32)


def _los_value(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower()
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    if text in {"1", "true", "los"}:
        return 1.0
    if text in {"0", "false", "nlos"}:
        return 0.0
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported Raymobtime LOS value {value!r}; expected 0/1 or LOS=0/LOS=1.") from exc



def _required_paths(root: Path) -> tuple[Path, ...]:
    return tuple(root / rel for rel in raymobtime_s008_layout().required_paths)


def _missing_required_paths(root: Path) -> list[Path]:
    return [path for path in _required_paths(root) if not path.exists()]




__all__ = [
    "FALLBACK_LINK_QUALITY_DBM",
    "HDF5_SUFFIXES",
    "INDEX_COLUMNS",
    "OUTPUT_SPLIT_TO_INDEX_NAME",
    "RAYMOBTIME_HDF5_CONTEXT",
    "RAYMOBTIME_S008_HDF5_FIELD_MAPPING",
    "RAY_FEATURE_NAMES",
    "REQUIRED_CSV_COLUMNS",
    "SOURCE_SPLIT_ORDER",
    "RaymobtimePaths",
    "resolve_raymobtime_paths",
]
