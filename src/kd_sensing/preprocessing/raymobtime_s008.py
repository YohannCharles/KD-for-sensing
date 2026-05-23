from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.layouts import raymobtime_s008_layout
from kd_sensing.registries import PREPROCESSORS
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


def build_s008_index(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    split_seed: int = 42,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    frame = _read_coord_csv(paths.csv_path)
    valid = frame.loc[frame["Val"].astype(str).str.upper() == "V"].copy()
    valid = valid.reset_index(names="source_row_index")
    valid["valid_index"] = np.arange(len(valid), dtype=np.int64)
    valid["sample_id"] = [
        _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
        for row in valid.itertuples(index=False)
    ]
    detected_splits = _detect_source_split_lengths(paths.data_root, total_count=len(valid))
    split_aliases: dict[str, str] = {}
    if detected_splits:
        split_names, source_split_names, source_split_indices = _assign_detected_source_splits(
            len(valid),
            detected_splits["lengths"],
        )
        split_protocol = "raymobtime_s008_official_baseline_split"
        split_source = detected_splits["source"]
        if "validation" in detected_splits["lengths"] and "test" not in detected_splits["lengths"]:
            split_aliases["test"] = "validation"
    else:
        split_names = _assign_splits(len(valid), split_ratios=split_ratios, seed=split_seed)
        source_split_names = np.full(len(valid), "all", dtype=object)
        source_split_indices = valid["valid_index"].to_numpy(dtype=np.int64)
        split_protocol = "raymobtime_s008_snapshot_random"
        split_source = None
    valid["split"] = split_names
    valid["source_split"] = source_split_names
    valid["source_split_index"] = source_split_indices
    valid["coord_row_in_split"] = valid.groupby("split").cumcount()
    index = valid.loc[:, list(INDEX_COLUMNS)].copy()
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    index.to_csv(paths.cache_dir / "index_all_valid.csv", index=False)
    for split, split_name in OUTPUT_SPLIT_TO_INDEX_NAME.items():
        split_frame = _index_frame_for_output_split(index, split, aliases=split_aliases)
        path = paths.cache_dir / f"index_{split_name}.csv"
        split_frame.to_csv(path, index=False)
        output_paths[f"index_{split_name}"] = str(path)
    metadata = _split_metadata(
        index,
        split_seed=split_seed,
        split_ratios=split_ratios,
        data_root=paths.data_root,
        split_protocol=split_protocol,
        split_source=split_source,
        split_aliases=split_aliases,
    )
    metadata_path = paths.cache_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_paths["index_all_valid"] = str(paths.cache_dir / "index_all_valid.csv")
    output_paths["split_metadata"] = str(metadata_path)
    return output_paths


def normalize_beam_labels(
    values: np.ndarray,
    *,
    num_tx_beams: int | None = None,
    num_rx_beams: int | None = None,
) -> dict[str, np.ndarray | int]:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        labels = array.astype(np.int64)
        rx_count = int(num_rx_beams or 1)
        tx_count = int(num_tx_beams or (int(labels.max(initial=0)) // max(rx_count, 1) + 1))
        beam_tx = labels // max(rx_count, 1)
        beam_rx = labels % max(rx_count, 1)
        num_classes = int(max(int(labels.max(initial=0)) + 1, tx_count * rx_count))
    elif array.ndim == 2 and int(array.shape[1]) == 2:
        pair = array.astype(np.int64)
        beam_tx = pair[:, 0]
        beam_rx = pair[:, 1]
        tx_count = int(num_tx_beams or int(beam_tx.max(initial=0)) + 1)
        rx_count = int(num_rx_beams or int(beam_rx.max(initial=0)) + 1)
        labels = beam_tx * max(rx_count, 1) + beam_rx
        num_classes = int(tx_count * rx_count)
    elif array.ndim == 3:
        scores = np.abs(array) if np.iscomplexobj(array) else array
        n, dim0, dim1 = scores.shape
        if num_tx_beams is not None and num_rx_beams is not None:
            expected = (int(num_tx_beams), int(num_rx_beams))
            reversed_expected = (int(num_rx_beams), int(num_tx_beams))
            if (dim0, dim1) == expected:
                tx_count, rx_count = expected
            elif (dim0, dim1) == reversed_expected:
                scores = np.swapaxes(scores, 1, 2)
                tx_count, rx_count = expected
            else:
                raise ValueError(
                    "Raymobtime beam score matrix shape does not match configured beam dimensions: "
                    f"got [N, {dim0}, {dim1}], expected [N, {expected[0]}, {expected[1]}] "
                    f"or [N, {reversed_expected[0]}, {reversed_expected[1]}]."
                )
        else:
            tx_count = int(num_tx_beams or dim0)
            rx_count = int(num_rx_beams or dim1)
        flat = scores.reshape(n, -1)
        labels = np.argmax(flat, axis=1).astype(np.int64)
        beam_tx = labels // max(rx_count, 1)
        beam_rx = labels % max(rx_count, 1)
        num_classes = int(tx_count * rx_count)
    else:
        raise ValueError(
            "Raymobtime beam_output must have shape [N], [N, 2], or [N, Tx, Rx]; "
            f"got {tuple(array.shape)}."
        )
    return {
        "beam_label": labels.astype(np.int64),
        "beam_tx": beam_tx.astype(np.int64),
        "beam_rx": beam_rx.astype(np.int64),
        "num_beam_classes": int(num_classes),
        "num_tx_beams": int(tx_count),
        "num_rx_beams": int(rx_count),
    }


def build_s008_labels(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    beam_key: str | None = None,
    num_tx_beams: int | None = 32,
    num_rx_beams: int | None = 8,
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    if not (paths.cache_dir / "index_all_valid.csv").exists():
        build_s008_index(data_root=paths.data_root, cache_dir=paths.cache_dir)
    valid_index = pd.read_csv(paths.cache_dir / "index_all_valid.csv")
    beam_arrays = _load_split_arrays(paths.data_root / "baseline_data" / "beam_output", key=beam_key)
    if "all" in beam_arrays:
        normalized_all = normalize_beam_labels(
            beam_arrays["all"],
            num_tx_beams=num_tx_beams,
            num_rx_beams=num_rx_beams,
        )
        normalized_by_source: dict[str, dict[str, np.ndarray | int]] | None = None
        labels = np.asarray(normalized_all["beam_label"])
        row_selector = _row_selector_for_array(labels, valid_index)
        metadata = _beam_metadata(normalized_all)
    else:
        normalized_all = None
        normalized_by_source = {
            split: normalize_beam_labels(array, num_tx_beams=num_tx_beams, num_rx_beams=num_rx_beams)
            for split, array in beam_arrays.items()
        }
        row_selector = None
        metadata = _merged_beam_metadata(normalized_by_source)
    output_paths: dict[str, Any] = {}
    for split, split_file in _split_index_files(paths.cache_dir).items():
        frame = pd.read_csv(split_file)
        if normalized_by_source is None:
            assert normalized_all is not None and row_selector is not None
            rows = row_selector(frame)
            beam_label = np.asarray(normalized_all["beam_label"])[rows]
            beam_tx = np.asarray(normalized_all["beam_tx"])[rows]
            beam_rx = np.asarray(normalized_all["beam_rx"])[rows]
        else:
            beam_label = _values_for_source_frame(frame, normalized_by_source, "beam_label")
            beam_tx = _values_for_source_frame(frame, normalized_by_source, "beam_tx")
            beam_rx = _values_for_source_frame(frame, normalized_by_source, "beam_rx")
        path = paths.cache_dir / f"labels_{split}.npz"
        np.savez(
            path,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            valid_index=frame["valid_index"].to_numpy(dtype=np.int64),
            beam_label=beam_label.astype(np.int64),
            beam_tx=beam_tx.astype(np.int64),
            beam_rx=beam_rx.astype(np.int64),
            los_label=_los_series_to_float(frame["LOS"]),
            **metadata,
        )
        output_paths[f"labels_{split}"] = str(path)
    output_paths["beam_metadata"] = metadata
    return output_paths


def extract_s008_ray_features(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    link_target_name: str = "link_power_max_dbm",
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    if not (paths.cache_dir / "index_all_valid.csv").exists():
        build_s008_index(data_root=paths.data_root, cache_dir=paths.cache_dir)
    ray_table = _load_ray_table(paths.ray_zip_path)
    output_paths = {}
    unmatched: list[dict[str, Any]] = []
    split_payloads: dict[str, dict[str, Any]] = {}
    split_quality: dict[str, dict[str, Any]] = {}
    for split, split_file in _split_index_files(paths.cache_dir).items():
        frame = pd.read_csv(split_file)
        no_los, with_los, link_quality, split_unmatched = _features_for_index(frame, ray_table, link_target_name)
        unmatched.extend(split_unmatched)
        quality = _ray_quality_summary(
            split,
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            link_quality=link_quality,
            unmatched=split_unmatched,
        )
        split_quality[split] = quality
        split_payloads[split] = {
            "frame": frame,
            "no_los": no_los,
            "with_los": with_los,
            "link_quality": link_quality,
        }
    quality_summary = _validate_ray_quality_summary(split_quality)
    for split, payload in split_payloads.items():
        frame = payload["frame"]
        no_los = payload["no_los"]
        with_los = payload["with_los"]
        link_quality = payload["link_quality"]
        path = paths.cache_dir / f"ray_features_{split}.npz"
        np.savez(
            path,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            ray_features_no_los=no_los.astype(np.float32),
            ray_features_with_los=with_los.astype(np.float32),
            link_quality=link_quality.astype(np.float32),
            feature_names=np.asarray(RAY_FEATURE_NAMES, dtype=object),
            with_los_feature_names=np.asarray((*RAY_FEATURE_NAMES, "LOS"), dtype=object),
            link_target_name=np.asarray(link_target_name),
            link_target_unit=np.asarray("dBm"),
        )
        output_paths[f"ray_features_{split}"] = str(path)
    report_path = paths.cache_dir / "ray_unmatched_report.json"
    report = {
        "unmatched": unmatched,
        "count": len(unmatched),
        "summary": quality_summary,
        "splits": split_quality,
        "hdf5_schema": RAYMOBTIME_S008_HDF5_FIELD_MAPPING,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_paths["unmatched_report"] = str(report_path)
    output_paths["ray_quality_summary"] = quality_summary
    return output_paths


def build_s008_cache(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    split_seed: int = 42,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
    beam_key: str | None = None,
    num_tx_beams: int | None = 32,
    num_rx_beams: int | None = 8,
    link_target_name: str = "link_power_max_dbm",
) -> dict[str, Any]:
    paths = resolve_raymobtime_paths(data_root=data_root, output_dir=output_dir, cache_dir=cache_dir)
    index_paths = build_s008_index(
        data_root=paths.data_root,
        cache_dir=paths.cache_dir,
        split_seed=split_seed,
        split_ratios=split_ratios,
    )
    label_paths = build_s008_labels(
        data_root=paths.data_root,
        cache_dir=paths.cache_dir,
        beam_key=beam_key,
        num_tx_beams=num_tx_beams,
        num_rx_beams=num_rx_beams,
    )
    ray_paths = extract_s008_ray_features(
        data_root=paths.data_root,
        cache_dir=paths.cache_dir,
        link_target_name=link_target_name,
    )
    output_paths: dict[str, Any] = {**index_paths, **label_paths, **ray_paths}
    for split, split_file in _split_index_files(paths.cache_dir).items():
        frame = pd.read_csv(split_file)
        labels = np.load(paths.cache_dir / f"labels_{split}.npz", allow_pickle=True)
        rays = np.load(paths.cache_dir / f"ray_features_{split}.npz", allow_pickle=True)
        coord = frame.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
        image = _optional_modality_array(paths.data_root / "baseline_data" / "image_v2_input", frame)
        lidar = _optional_modality_array(paths.data_root / "baseline_data" / "lidar_input", frame)
        cache_path = paths.cache_dir / f"cache_{split}.npz"
        np.savez(
            cache_path,
            sample_id=frame["sample_id"].astype(str).to_numpy(),
            coord=coord,
            image=image,
            lidar=lidar,
            ray=rays["ray_features_no_los"].astype(np.float32),
            ray_features_with_los=rays["ray_features_with_los"].astype(np.float32),
            target_beam=labels["beam_label"].astype(np.int64),
            beam_tx=labels["beam_tx"].astype(np.int64),
            beam_rx=labels["beam_rx"].astype(np.int64),
            los_label=labels["los_label"].astype(np.float32),
            link_quality=rays["link_quality"].astype(np.float32),
            valid_index=frame["valid_index"].to_numpy(dtype=np.int64),
        )
        output_paths[f"cache_{split}"] = str(cache_path)
    metadata_path = paths.cache_dir / "cache_metadata.json"
    metadata = _cache_metadata(paths.cache_dir, link_target_name=link_target_name)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_paths["cache_metadata"] = str(metadata_path)
    return output_paths


@PREPROCESSORS.register("raymobtime_s008_audit")
class RaymobtimeS008AuditPreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return audit_s008_files(**self.kwargs)


@PREPROCESSORS.register("raymobtime_s008_index")
class RaymobtimeS008IndexPreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return build_s008_index(**self.kwargs)


@PREPROCESSORS.register("raymobtime_s008_ray_features")
class RaymobtimeS008RayFeaturePreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return extract_s008_ray_features(**self.kwargs)


@PREPROCESSORS.register("raymobtime_s008_cache")
class RaymobtimeS008CachePreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return build_s008_cache(**self.kwargs)


def _required_paths(root: Path) -> tuple[Path, ...]:
    return tuple(root / rel for rel in raymobtime_s008_layout().required_paths)


def _missing_required_paths(root: Path) -> list[Path]:
    return [path for path in _required_paths(root) if not path.exists()]


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


def _assign_splits(
    count: int,
    *,
    split_ratios: list[float] | tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("split_ratios must contain three non-negative values for train/validation/test.")
    ratios = ratios / ratios.sum()
    indices = np.arange(count)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    train_end = int(math.floor(count * ratios[0]))
    val_end = train_end + int(math.floor(count * ratios[1]))
    if count >= 3:
        train_end = max(1, min(train_end, count - 2))
        val_end = max(train_end + 1, min(val_end, count - 1))
    names = np.full(count, "test", dtype=object)
    names[indices[:train_end]] = "train"
    names[indices[train_end:val_end]] = "validation"
    return names


def _detect_source_split_lengths(data_root: Path, *, total_count: int) -> dict[str, Any] | None:
    for rel in (
        "baseline_data/coord_input",
        "baseline_data/beam_output",
        "baseline_data/image_v2_input",
        "baseline_data/lidar_input",
    ):
        path = data_root / rel
        if not path.exists():
            continue
        arrays = _load_split_arrays(path, key=None, allow_missing=True)
        lengths = {split: int(len(array)) for split, array in arrays.items() if split != "all"}
        if lengths and sum(lengths.values()) == int(total_count):
            return {"source": rel, "lengths": lengths}
    return None


def _assign_detected_source_splits(
    count: int,
    split_lengths: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_names = np.empty(count, dtype=object)
    source_split_names = np.empty(count, dtype=object)
    source_split_indices = np.empty(count, dtype=np.int64)
    cursor = 0
    for source_split in SOURCE_SPLIT_ORDER:
        length = int(split_lengths.get(source_split, 0))
        if length <= 0:
            continue
        end = cursor + length
        if end > count:
            raise ValueError(
                f"Detected Raymobtime source split lengths exceed valid receiver count: {split_lengths} > {count}."
            )
        split_names[cursor:end] = source_split
        source_split_names[cursor:end] = source_split
        source_split_indices[cursor:end] = np.arange(length, dtype=np.int64)
        cursor = end
    if cursor != count:
        raise ValueError(
            f"Detected Raymobtime source split lengths cover {cursor} rows but valid receiver count is {count}."
        )
    return split_names, source_split_names, source_split_indices


def _index_frame_for_output_split(index: pd.DataFrame, split: str, *, aliases: dict[str, str]) -> pd.DataFrame:
    source = aliases.get(split, split)
    frame = index.loc[index["split"] == source].copy()
    if source != split:
        frame["split"] = split
        frame["coord_row_in_split"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _split_metadata(
    index: pd.DataFrame,
    *,
    split_seed: int,
    split_ratios: Any,
    data_root: Path,
    split_protocol: str,
    split_source: str | None,
    split_aliases: dict[str, str],
) -> dict[str, Any]:
    metadata = {
        "dataset": "raymobtime_s008",
        "task_semantics": "current_snapshot_beam_selection",
        "data_root": str(data_root),
        "split_seed": int(split_seed),
        "split_ratios": [float(value) for value in split_ratios],
        "split_protocol": split_protocol,
        "sample_count": int(len(index)),
        "fingerprint": _fingerprint(index["sample_id"].astype(str).tolist()),
        "splits": {},
    }
    if split_source is not None:
        metadata["split_source"] = split_source
    if split_aliases:
        metadata["split_aliases"] = split_aliases
    for split, frame in index.groupby("split", sort=True):
        metadata["splits"][str(split)] = {
            "num_samples": int(len(frame)),
            "beam_label_distribution": {},
            "los_distribution": _value_counts(frame["LOS"]),
        }
    return metadata


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


def _beam_metadata(normalized: dict[str, np.ndarray | int]) -> dict[str, int]:
    return {
        "num_beam_classes": int(normalized["num_beam_classes"]),
        "num_tx_beams": int(normalized["num_tx_beams"]),
        "num_rx_beams": int(normalized["num_rx_beams"]),
    }


def _merged_beam_metadata(normalized_by_source: dict[str, dict[str, np.ndarray | int]]) -> dict[str, int]:
    metadata_values = [_beam_metadata(value) for value in normalized_by_source.values()]
    if not metadata_values:
        raise ValueError("Raymobtime beam output contains no split arrays.")
    first = metadata_values[0]
    for metadata in metadata_values[1:]:
        if metadata != first:
            raise ValueError(f"Raymobtime beam split metadata is inconsistent: {metadata_values}.")
    return first


def _values_for_source_frame(
    frame: pd.DataFrame,
    normalized_by_source: dict[str, dict[str, np.ndarray | int]],
    key: str,
) -> np.ndarray:
    if frame.empty:
        dtype = np.asarray(next(iter(normalized_by_source.values()))[key]).dtype
        return np.asarray([], dtype=dtype)
    if not {"source_split", "source_split_index"} <= set(frame.columns):
        raise ValueError("Raymobtime split index is missing source_split/source_split_index columns.")
    result: np.ndarray | None = None
    for source_split, group in frame.groupby("source_split", sort=False):
        source_key = str(source_split)
        if source_key not in normalized_by_source:
            raise ValueError(
                f"Raymobtime split index references source_split={source_key!r}, "
                f"but available beam splits are {sorted(normalized_by_source)}."
            )
        values = np.asarray(normalized_by_source[source_key][key])
        positions = group["source_split_index"].to_numpy(dtype=np.int64)
        if positions.size and int(positions.max()) >= len(values):
            raise ValueError(
                f"Raymobtime source_split={source_key!r} references row {int(positions.max())}, "
                f"but beam split has only {len(values)} rows."
            )
        selected = values[positions]
        if result is None:
            result = np.empty((len(frame), *selected.shape[1:]), dtype=selected.dtype)
        result[frame.index.get_indexer(group.index)] = selected
    assert result is not None
    return result


def _load_ray_table(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            suffix = Path(name).suffix.lower()
            if suffix not in {".csv", ".txt", ".json", *HDF5_SUFFIXES}:
                continue
            raw = archive.read(name)
            frame = _ray_frame_from_bytes(raw, suffix=suffix, source=name)
            if frame is not None and not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table = _canonicalize_ray_columns(table)
    if "sample_id" not in table.columns and {"EpisodeID", "SceneID", "VehicleArrayID"} <= set(table.columns):
        table["sample_id"] = [
            _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
            for row in table.itertuples(index=False)
        ]
    return table


def _ray_frame_from_bytes(raw: bytes, *, suffix: str, source: str) -> pd.DataFrame | None:
    if suffix in HDF5_SUFFIXES:
        return _ray_frame_from_hdf5(raw, source=source)
    try:
        if suffix == ".json":
            data = json.loads(raw.decode("utf-8"))
            frame = pd.DataFrame(data if isinstance(data, list) else data.get("rows", data))
        else:
            text = raw.decode("utf-8")
            sep = "," if "," in text.splitlines()[0] else None
            frame = pd.read_csv(StringIO(text), sep=sep)
    except Exception:
        try:
            array = np.load(BytesIO(raw), allow_pickle=True)
        except Exception:
            return None
        if isinstance(array, np.ndarray) and array.ndim == 2:
            frame = pd.DataFrame(array)
        else:
            return None
    frame["_source"] = source
    return frame


def _ray_frame_from_hdf5(raw: bytes, *, source: str) -> pd.DataFrame:
    h5py = _require_h5py(source)
    with h5py.File(BytesIO(raw), "r") as handle:
        frames: list[pd.DataFrame] = []
        has_all_episode_data = "allEpisodeData" in handle
        if has_all_episode_data:
            frames.append(_raymobtime_all_episode_data_frame(handle["allEpisodeData"][()], source=source))
        frames.extend(_generic_hdf5_ray_frames(handle, source=source))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        if has_all_episode_data:
            return pd.DataFrame(
                columns=[
                    "sample_id",
                    "EpisodeID",
                    "SceneID",
                    "VehicleArrayID",
                    "path_index",
                    "power_dbm",
                    "toa",
                    "elev_aod",
                    "az_aod",
                    "elev_aoa",
                    "az_aoa",
                    "phase",
                    "_source",
                ]
            )
        raise ValueError(
            f"{RAYMOBTIME_HDF5_CONTEXT} could not extract canonical ray rows from {source}. "
            "Expected allEpisodeData or datasets containing power and sample alignment fields."
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["_source"] = source
    return _canonicalize_ray_columns(frame)


def _require_h5py(source: str):
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised with environments missing h5py
        raise ModuleNotFoundError(
            f"{RAYMOBTIME_HDF5_CONTEXT} requires the 'h5py' dependency to read {source}. "
            "Install it in the kd_mm_beam environment, for example: "
            "conda run -n kd_mm_beam python -m pip install h5py"
        ) from exc
    return h5py


def _raymobtime_all_episode_data_frame(array: np.ndarray, *, source: str) -> pd.DataFrame:
    values = np.asarray(array)
    if values.ndim != 4 or values.shape[-1] < 9:
        raise ValueError(
            f"{RAYMOBTIME_HDF5_CONTEXT} expected allEpisodeData shape [SceneID, VehicleArrayID, path, fields>=9] "
            f"for {source}; got {tuple(values.shape)}."
        )
    episode = _episode_id_from_hdf5_source(source)
    scene_count, vehicle_count, path_count, _ = values.shape
    scene_ids = np.repeat(np.arange(scene_count, dtype=np.int64), vehicle_count * path_count)
    vehicle_ids = np.tile(np.repeat(np.arange(vehicle_count, dtype=np.int64), path_count), scene_count)
    path_ids = np.tile(np.arange(path_count, dtype=np.int64), scene_count * vehicle_count)
    flat = values.reshape(-1, values.shape[-1])
    frame = pd.DataFrame(
        {
            "EpisodeID": np.full(flat.shape[0], episode, dtype=np.int64),
            "SceneID": scene_ids,
            "VehicleArrayID": vehicle_ids,
            "path_index": path_ids,
            "power_dbm": flat[:, 0],
            "toa": flat[:, 1],
            "elev_aod": flat[:, 2],
            "az_aod": flat[:, 3],
            "elev_aoa": flat[:, 4],
            "az_aoa": flat[:, 5],
            "phase": flat[:, 8],
            "raymobtime_path_flag": flat[:, 6],
        }
    )
    frame = frame.loc[np.isfinite(pd.to_numeric(frame["power_dbm"], errors="coerce"))].copy()
    frame["_source"] = source
    frame["sample_id"] = [
        _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
        for row in frame.itertuples(index=False)
    ]
    return frame


def _episode_id_from_hdf5_source(source: str) -> int:
    match = re.search(r"(?:^|[_/-])e(\d+)(?:\D|$)", source)
    if not match:
        raise ValueError(
            f"{RAYMOBTIME_HDF5_CONTEXT} could not infer EpisodeID from HDF5 entry name {source!r}; "
            "expected a filename token like '_e0.hdf5'."
        )
    return int(match.group(1))


def _generic_hdf5_ray_frames(group, *, source: str, group_path: str = "") -> list[pd.DataFrame]:
    h5py = _require_h5py(source)
    frames: list[pd.DataFrame] = []
    datasets: dict[str, np.ndarray] = {}
    for key, item in group.items():
        item_path = f"{group_path}/{key}" if group_path else str(key)
        if isinstance(item, h5py.Dataset):
            value = item[()]
            if getattr(value, "dtype", None) is not None and value.dtype.names:
                frame = _hdf5_structured_array_frame(value, source=source, group_path=item_path)
                if frame is not None and not frame.empty:
                    frames.append(frame)
            else:
                datasets[str(key)] = np.asarray(value)
        elif isinstance(item, h5py.Group):
            frames.extend(_generic_hdf5_ray_frames(item, source=source, group_path=item_path))
    frame = _hdf5_dataset_mapping_frame(datasets, source=source, group_path=group_path)
    if frame is not None and not frame.empty:
        frames.append(frame)
    return frames


def _hdf5_structured_array_frame(array: np.ndarray, *, source: str, group_path: str) -> pd.DataFrame | None:
    if not array.dtype.names:
        return None
    arrays = {name: np.asarray(array[name]) for name in array.dtype.names}
    return _hdf5_dataset_mapping_frame(arrays, source=source, group_path=group_path)


def _hdf5_dataset_mapping_frame(
    datasets: dict[str, np.ndarray],
    *,
    source: str,
    group_path: str,
) -> pd.DataFrame | None:
    if not datasets:
        return None
    canonical: dict[str, np.ndarray] = {}
    raw_names: dict[str, str] = {}
    for name, values in datasets.items():
        canonical_name = _canonical_ray_column_name(name)
        if canonical_name is None:
            continue
        canonical[canonical_name] = np.asarray(values)
        raw_names[canonical_name] = name
    if "power_dbm" not in canonical:
        return None
    power = np.asarray(canonical["power_dbm"])
    if power.ndim == 0:
        return None
    sample_shape = power.shape[:-1] if power.ndim >= 2 else power.shape
    path_count = int(power.shape[-1]) if power.ndim >= 2 else 1
    row_count = int(power.size)
    columns: dict[str, np.ndarray] = {}
    for name, values in canonical.items():
        broadcast = _broadcast_hdf5_values(
            np.asarray(values),
            power_shape=power.shape,
            sample_shape=sample_shape,
            path_count=path_count,
            row_count=row_count,
        )
        if broadcast is not None:
            columns[name] = broadcast
    if "power_dbm" not in columns:
        return None
    if "path_index" not in columns:
        columns["path_index"] = (
            np.tile(np.arange(path_count, dtype=np.int64), row_count // max(path_count, 1))
            if path_count > 1
            else np.arange(row_count, dtype=np.int64)
        )
    frame = pd.DataFrame(columns)
    if "sample_id" not in frame.columns:
        ids = _ids_from_hdf5_context(source=source, group_path=group_path)
        for key, value in ids.items():
            if key not in frame.columns:
                frame[key] = value
    if "sample_id" not in frame.columns and {"EpisodeID", "SceneID", "VehicleArrayID"} <= set(frame.columns):
        frame["sample_id"] = [
            _sample_id(row.EpisodeID, row.SceneID, row.VehicleArrayID)
            for row in frame.itertuples(index=False)
        ]
    frame["_source"] = source
    frame["_hdf5_group"] = group_path
    frame["_hdf5_power_field"] = raw_names.get("power_dbm")
    frame = frame.loc[np.isfinite(pd.to_numeric(frame["power_dbm"], errors="coerce"))].copy()
    return frame


def _broadcast_hdf5_values(
    values: np.ndarray,
    *,
    power_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    path_count: int,
    row_count: int,
) -> np.ndarray | None:
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O", "U"}:
        array = _decode_hdf5_strings(array)
    if array.shape == ():
        return np.repeat(array.reshape(1), row_count)
    if tuple(array.shape) == tuple(power_shape):
        return array.reshape(row_count)
    sample_count = int(np.prod(sample_shape)) if sample_shape else 1
    if tuple(array.shape) == tuple(sample_shape):
        return np.repeat(array.reshape(sample_count), max(path_count, 1))
    if array.size == row_count:
        return array.reshape(row_count)
    if sample_count > 0 and array.size == sample_count:
        return np.repeat(array.reshape(sample_count), max(path_count, 1))
    if sample_count == 1 and path_count > 1 and array.size == path_count:
        return array.reshape(path_count)
    if array.size == 1:
        return np.repeat(array.reshape(1), row_count)
    return None


def _decode_hdf5_strings(values: np.ndarray) -> np.ndarray:
    def decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    return np.vectorize(decode, otypes=[object])(values)


def _ids_from_hdf5_context(*, source: str, group_path: str) -> dict[str, int]:
    text = f"{source}/{group_path}"
    ids: dict[str, int] = {}
    episode = re.search(r"(?:episode|EpisodeID|episode_id|[_/-]e)(\d+)", text)
    scene = re.search(r"(?:scene|SceneID|scene_id|[_/-]s)(\d+)", text)
    vehicle = re.search(r"(?:vehicle|VehicleArrayID|vehicle_array_id|rx|[_/-]v)(\d+)", text)
    if episode:
        ids["EpisodeID"] = int(episode.group(1))
    if scene:
        ids["SceneID"] = int(scene.group(1))
    if vehicle:
        ids["VehicleArrayID"] = int(vehicle.group(1))
    return ids


def _canonicalize_ray_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in list(result.columns):
        canonical = _canonical_ray_column_name(column)
        if canonical is None or canonical in result.columns:
            continue
        result[canonical] = result[column]
    if "power_dbm" in result.columns:
        result["power_dbm"] = pd.to_numeric(result["power_dbm"], errors="coerce")
    return result


def _canonical_ray_column_name(name: Any) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())
    for canonical, aliases in _RAY_COLUMN_ALIASES.items():
        if token == re.sub(r"[^a-z0-9]+", "", canonical.lower()):
            return canonical
        for alias in aliases:
            if token == re.sub(r"[^a-z0-9]+", "", alias.lower()):
                return canonical
    return None


def _features_for_index(
    index: pd.DataFrame,
    ray_table: pd.DataFrame,
    link_target_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features = []
    with_los = []
    link_quality = []
    unmatched = []
    grouped = ray_table.groupby("sample_id") if "sample_id" in ray_table.columns else None
    for row in index.itertuples(index=False):
        sample_id = str(row.sample_id)
        if grouped is None or sample_id not in grouped.groups:
            vector, link = _empty_ray_feature()
            unmatched.append({"sample_id": sample_id, "reason": "missing_ray_paths"})
        else:
            vector, link = _ray_feature_from_rows(grouped.get_group(sample_id), link_target_name)
        features.append(vector)
        with_los.append(np.concatenate([vector, np.asarray([_los_value(row.LOS)], dtype=np.float32)]))
        link_quality.append(link)
    return (
        np.vstack(features).astype(np.float32),
        np.vstack(with_los).astype(np.float32),
        np.asarray(link_quality, dtype=np.float32),
        unmatched,
    )


def _ray_quality_summary(
    split: str,
    *,
    sample_ids: np.ndarray,
    link_quality: np.ndarray,
    unmatched: list[dict[str, Any]],
) -> dict[str, Any]:
    values = np.asarray(link_quality, dtype=np.float64)
    finite = values[np.isfinite(values)]
    sample_count = int(len(sample_ids))
    unmatched_count = int(len(unmatched))
    fallback_mask = np.isclose(values, FALLBACK_LINK_QUALITY_DBM, atol=1e-6)
    fallback_count = int(np.count_nonzero(fallback_mask))
    summary = {
        "split": split,
        "num_samples": sample_count,
        "matched_ray_paths": int(max(sample_count - unmatched_count, 0)),
        "unmatched_samples": unmatched_count,
        "fallback_link_targets": fallback_count,
        "fallback_ratio": float(fallback_count / max(sample_count, 1)),
        "fallback_value_dbm": float(FALLBACK_LINK_QUALITY_DBM),
        "link_target_distribution": _numeric_distribution(finite),
    }
    return summary


def _numeric_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def _validate_ray_quality_summary(split_quality: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "num_samples": int(sum(int(item.get("num_samples", 0)) for item in split_quality.values())),
        "matched_ray_paths": int(sum(int(item.get("matched_ray_paths", 0)) for item in split_quality.values())),
        "unmatched_samples": int(sum(int(item.get("unmatched_samples", 0)) for item in split_quality.values())),
        "fallback_link_targets": int(sum(int(item.get("fallback_link_targets", 0)) for item in split_quality.values())),
        "fallback_value_dbm": float(FALLBACK_LINK_QUALITY_DBM),
    }
    totals["fallback_ratio"] = float(totals["fallback_link_targets"] / max(totals["num_samples"], 1))
    train = split_quality.get("train", {})
    train_distribution = train.get("link_target_distribution", {}) if isinstance(train, dict) else {}
    train_std = train_distribution.get("std")
    issues: list[str] = []
    if totals["num_samples"] > 0 and totals["matched_ray_paths"] == 0:
        issues.append("all samples are missing ray paths")
    if totals["num_samples"] > 0 and totals["fallback_link_targets"] == totals["num_samples"]:
        issues.append(f"all link_quality targets equal fallback {FALLBACK_LINK_QUALITY_DBM:g} dBm")
    if train.get("num_samples", 0) and train_std is not None and float(train_std) <= 1e-8:
        issues.append("train split link_quality standard deviation is 0")
    summary = {
        **totals,
        "splits": split_quality,
        "status": "failed" if issues else "ok",
        "issues": issues,
    }
    if issues:
        raise ValueError(
            "Raymobtime s008 ray feature cache quality gate failed: "
            + "; ".join(issues)
            + ". Check ray-tracing HDF5 parsing, sample_id alignment, and data_root configuration."
        )
    return summary


def _empty_ray_feature() -> tuple[np.ndarray, float]:
    vector = np.zeros((len(RAY_FEATURE_NAMES),), dtype=np.float32)
    vector[0] = FALLBACK_LINK_QUALITY_DBM
    vector[1] = FALLBACK_LINK_QUALITY_DBM
    return vector, FALLBACK_LINK_QUALITY_DBM


def _ray_feature_from_rows(rows: pd.DataFrame, link_target_name: str) -> tuple[np.ndarray, float]:
    power = _numeric_column(rows, ("power_dbm", "received_power_dbm", "path_power_dbm", "Pr_dBm", "power"))
    toa = _numeric_column(rows, ("toa", "time_of_arrival", "delay", "ToA"))
    valid = np.isfinite(power)
    if power.size == 0 or not np.any(valid):
        return _empty_ray_feature()
    power_v = power[valid]
    strongest = int(np.nanargmax(power))
    toa_v = toa[np.isfinite(toa)] if toa.size else np.asarray([], dtype=np.float64)
    sum_linear = float(np.sum(np.power(10.0, power_v / 10.0)))
    vector = np.asarray(
        [
            float(np.nanmax(power_v)),
            float(np.nanmean(power_v)),
            sum_linear,
            float(power_v.size),
            _safe_min(toa_v),
            _safe_mean(toa_v),
            _row_value(rows, strongest, ("toa", "time_of_arrival", "delay", "ToA")),
            _row_value(rows, strongest, ("elev_aod", "elevation_aod", "departure_elevation")),
            _row_value(rows, strongest, ("az_aod", "azimuth_aod", "departure_azimuth")),
            _row_value(rows, strongest, ("elev_aoa", "elevation_aoa", "arrival_elevation")),
            _row_value(rows, strongest, ("az_aoa", "azimuth_aoa", "arrival_azimuth")),
            _row_value(rows, strongest, ("phase", "path_phase")),
            float(np.nanmax(power_v) - np.nanmin(power_v)) if power_v.size else 0.0,
            float(np.nanmax(toa_v) - np.nanmin(toa_v)) if toa_v.size else 0.0,
        ],
        dtype=np.float32,
    )
    link = float(np.nanmax(power_v)) if link_target_name == "link_power_max_dbm" else float(10.0 * np.log10(max(sum_linear, 1e-12)))
    return vector, link


def _numeric_column(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def _row_value(frame: pd.DataFrame, index: int, names: tuple[str, ...]) -> float:
    for name in names:
        if name in frame.columns:
            value = pd.to_numeric(pd.Series([frame.iloc[index][name]]), errors="coerce").iloc[0]
            return 0.0 if pd.isna(value) else float(value)
    return 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if values.size else 0.0


def _safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else 0.0


def _optional_modality_array(path: Path, frame: pd.DataFrame) -> np.ndarray:
    count = len(frame)
    arrays = _load_split_arrays(path, key=None, allow_missing=True)
    if not arrays:
        return np.zeros((count, 1), dtype=np.float32)
    if "all" in arrays:
        first = arrays["all"]
        rows = _row_selector_for_array(first, frame)(frame)
        return first[rows].astype(np.float32)
    return _array_for_source_frame(frame, arrays).astype(np.float32)


def _array_for_source_frame(frame: pd.DataFrame, arrays_by_source: dict[str, np.ndarray]) -> np.ndarray:
    if frame.empty:
        first = next(iter(arrays_by_source.values()))
        return np.empty((0, *first.shape[1:]), dtype=first.dtype)
    if not {"source_split", "source_split_index"} <= set(frame.columns):
        raise ValueError("Raymobtime split index is missing source_split/source_split_index columns.")
    result: np.ndarray | None = None
    for source_split, group in frame.groupby("source_split", sort=False):
        source_key = str(source_split)
        if source_key not in arrays_by_source:
            raise ValueError(
                f"Raymobtime split index references source_split={source_key!r}, "
                f"but available modality splits are {sorted(arrays_by_source)}."
            )
        values = np.asarray(arrays_by_source[source_key])
        positions = group["source_split_index"].to_numpy(dtype=np.int64)
        if positions.size and int(positions.max()) >= len(values):
            raise ValueError(
                f"Raymobtime source_split={source_key!r} references row {int(positions.max())}, "
                f"but modality split has only {len(values)} rows."
            )
        selected = values[positions]
        if result is None:
            result = np.empty((len(frame), *selected.shape[1:]), dtype=selected.dtype)
        result[frame.index.get_indexer(group.index)] = selected
    assert result is not None
    return result


def _cache_metadata(cache_dir: Path, *, link_target_name: str) -> dict[str, Any]:
    ray_report = _read_json_file(cache_dir / "ray_unmatched_report.json")
    ray_summary = ray_report.get("summary", {}) if isinstance(ray_report, dict) else {}
    ray_splits = ray_report.get("splits", {}) if isinstance(ray_report, dict) else {}
    metadata: dict[str, Any] = {
        "dataset": "raymobtime_s008",
        "task_semantics": "current_snapshot_beam_selection",
        "link_target_name": link_target_name,
        "link_target_unit": "dBm",
        "link_target_aggregation": link_target_name.replace("link_power_", "").replace("_dbm", ""),
        "ray_feature_version": 2,
        "ray_quality": ray_summary,
        "hdf5_schema": RAYMOBTIME_S008_HDF5_FIELD_MAPPING,
        "normalization": {},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        path = cache_dir / f"cache_{split}.npz"
        if not path.exists():
            continue
        split_quality = ray_splits.get(split, {}) if isinstance(ray_splits, dict) else {}
        cache = np.load(path, allow_pickle=True)
        metadata["splits"][split] = {
            "num_samples": int(len(cache["sample_id"])),
            "fingerprint": _fingerprint([str(item) for item in cache["sample_id"].tolist()]),
            "ray_quality": split_quality,
        }
    train_path = cache_dir / "cache_train.npz"
    if train_path.exists():
        train = np.load(train_path, allow_pickle=True)
        for key in ("coord", "ray", "link_quality"):
            values = np.asarray(train[key], dtype=np.float32)
            metadata["normalization"][key] = {
                "mean": np.asarray(values.mean(axis=0)).reshape(-1).astype(float).tolist(),
                "std": np.asarray(values.std(axis=0)).reshape(-1).clip(1e-6, None).astype(float).tolist(),
            }
    labels_path = cache_dir / "labels_train.npz"
    if labels_path.exists():
        labels = np.load(labels_path, allow_pickle=True)
        metadata["beam"] = {
            "num_beam_classes": int(np.asarray(labels["num_beam_classes"]).item()),
            "num_tx_beams": int(np.asarray(labels["num_tx_beams"]).item()),
            "num_rx_beams": int(np.asarray(labels["num_rx_beams"]).item()),
        }
    return metadata


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


__all__ = [
    "RAY_FEATURE_NAMES",
    "audit_s008_files",
    "build_s008_cache",
    "build_s008_index",
    "build_s008_labels",
    "extract_s008_ray_features",
    "normalize_beam_labels",
    "resolve_raymobtime_paths",
]
