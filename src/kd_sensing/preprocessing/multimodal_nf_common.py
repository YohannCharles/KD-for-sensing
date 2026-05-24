from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kd_sensing.data.dataset_runtime import SampleIndex, SampleRow, write_index_csv
from kd_sensing.data.layouts import MULTIMODAL_NF_FAMILY, multimodal_nf_layout
from kd_sensing.utils.paths import resolve_path


MULTIMODAL_NF_DATASET_TYPE = "multimodal_nf"
MULTIMODAL_NF_HDF5_KEYS = {
    "csi": ("H", "channel", "Channel"),
    "gps": ("Pos", "position", "Position"),
    "beam_idx": ("BeamIdx", "beam_idx", "beam_index"),
    "beam_power": ("BeamPower", "beam_power", "power"),
    "los": ("Has_LoS", "HasLOS", "LoS"),
    "nf": ("Is_NF", "IsNF", "near_field"),
    "image": ("image", "Image", "RGB", "rgb"),
    "lidar": ("lidar", "LiDAR", "points", "point_cloud"),
    "city": ("City", "city", "city_id"),
    "trajectory": ("Trajectory", "TrajIdx", "trajectory_id", "traj_id"),
    "frame": ("Frame", "FrameIdx", "frame_id"),
    "metadata": ("Metadata", "metadata"),
    "traj_nlos": ("Traj_Is_NLoS", "traj_is_nlos"),
    "mode": ("Mode_Idx", "mode_idx"),
}
REQUIRED_MULTIMODAL_NF_FIELDS = ("csi", "gps", "beam_idx", "beam_power", "los", "nf")
DEFAULT_DENSE_CODEBOOK_SHAPE = (90, 45, 16)
DEFAULT_SMALL_CODEBOOK_SHAPE = (20, 20, 10)
DEFAULT_FLATTEN_ORDER = "azimuth_elevation_range"


@dataclass(frozen=True)
class MultimodalNFPaths:
    data_root: Path
    raw_root: Path
    codebook_root: Path
    cache_dir: Path
    output_dir: Path


def resolve_multimodal_nf_paths(
    *,
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    codebook_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> MultimodalNFPaths:
    layout = multimodal_nf_layout()
    root = resolve_path(data_root or layout.root)
    raw = resolve_path(raw_root) if raw_root is not None else root / "raw"
    codebooks = resolve_path(codebook_root) if codebook_root is not None else root / "codebooks"
    cache = resolve_path(cache_dir) if cache_dir is not None else root / "cache"
    output = resolve_path(output_dir) if output_dir is not None else resolve_path(layout.audit_output_root)
    return MultimodalNFPaths(
        data_root=root,
        raw_root=raw,
        codebook_root=codebooks,
        cache_dir=cache,
        output_dir=output,
    )


def audit_multimodal_nf_files(
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    codebook_root: str | Path | None = None,
    channel_path: str | Path | None = None,
    image_path: str | Path | None = None,
    lidar_path: str | Path | None = None,
    codebook_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    paths = resolve_multimodal_nf_paths(
        data_root=data_root,
        raw_root=raw_root,
        codebook_root=codebook_root,
        output_dir=output_dir,
    )
    hdf5_files = _candidate_hdf5_files(paths, channel_path=channel_path, image_path=image_path, lidar_path=lidar_path)
    summaries = [_hdf5_file_summary(path) for path in hdf5_files]
    combined_keys = {field for summary in summaries for field in summary.get("resolved_fields", {})}
    missing = [field for field in REQUIRED_MULTIMODAL_NF_FIELDS if field not in combined_keys]
    codebooks = _candidate_codebook_files(paths, codebook_path=codebook_path)
    codebook_summaries = []
    for item in codebooks:
        try:
            codebook_summaries.append(parse_codebook_metadata(item))
        except Exception as exc:
            codebook_summaries.append({"path": str(item), "error": str(exc), "fingerprint": fingerprint_path(item)})
    cities = sorted({city for summary in summaries for city in summary.get("cities", [])})
    sample_count = int(sum(int(summary.get("sample_count", 0) or 0) for summary in summaries))
    report = {
        "dataset": MULTIMODAL_NF_DATASET_TYPE,
        "family": MULTIMODAL_NF_FAMILY,
        "data_root": str(paths.data_root),
        "raw_root": str(paths.raw_root),
        "hdf5_files": summaries,
        "codebooks": codebook_summaries,
        "city_ids": cities,
        "sample_count": sample_count,
        "missing_fields": missing,
        "fingerprint": _fingerprint([json.dumps(summary, sort_keys=True) for summary in summaries]),
    }
    if require_complete and missing:
        raise ValueError(
            "Multimodal-NF audit found missing required field(s): "
            f"{missing}. Files checked: {[str(path) for path in hdf5_files]}"
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.output_dir / "multimodal_nf_audit.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def build_multimodal_nf_index(
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    channel_path: str | Path | None = None,
    image_path: str | Path | None = None,
    lidar_path: str | Path | None = None,
    split_mode: str = "city",
    train_cities: list[str] | tuple[str, ...] | None = None,
    val_cities: list[str] | tuple[str, ...] | None = None,
    test_cities: list[str] | tuple[str, ...] | None = None,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_seed: int = 42,
    seq_len: int = 8,
    num_pred: int | None = None,
    pred_horizon: int | None = None,
) -> dict[str, Any]:
    horizon = _resolve_prediction_horizon(num_pred=num_pred, pred_horizon=pred_horizon)
    paths = resolve_multimodal_nf_paths(data_root=data_root, raw_root=raw_root, cache_dir=cache_dir)
    rows, metadata = build_multimodal_nf_rows(
        data_root=paths.data_root,
        raw_root=paths.raw_root,
        channel_path=channel_path,
        image_path=image_path,
        lidar_path=lidar_path,
        split_mode=split_mode,
        train_cities=train_cities,
        val_cities=val_cities,
        test_cities=test_cities,
        split_ratios=split_ratios,
        split_seed=split_seed,
        seq_len=seq_len,
        num_pred=horizon,
    )
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = _index_prefix(seq_len=seq_len, num_pred=horizon)
    all_index = write_index_csv(paths.cache_dir / f"{prefix}_all.csv", rows)
    split_paths: dict[str, str] = {"index_all": str(all_index)}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if _normalize_split(row.split) == split]
        output_name = "val" if split == "validation" else split
        split_paths[f"index_{output_name}"] = str(write_index_csv(paths.cache_dir / f"{prefix}_{output_name}.csv", split_rows))
    metadata_path = paths.cache_dir / f"{prefix}_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    split_paths["split_metadata"] = str(metadata_path)
    return split_paths


def build_multimodal_nf_rows(
    *,
    data_root: str | Path,
    raw_root: str | Path | None = None,
    channel_path: str | Path | None = None,
    image_path: str | Path | None = None,
    lidar_path: str | Path | None = None,
    split_mode: str = "city",
    train_cities: list[str] | tuple[str, ...] | None = None,
    val_cities: list[str] | tuple[str, ...] | None = None,
    test_cities: list[str] | tuple[str, ...] | None = None,
    split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_seed: int = 42,
    seq_len: int = 8,
    num_pred: int = 3,
) -> tuple[list[SampleRow], dict[str, Any]]:
    seq_len = int(seq_len)
    num_pred = int(num_pred)
    if seq_len <= 0 or num_pred <= 0:
        raise ValueError(f"Multimodal-NF sequence windows require positive seq_len and num_pred, got {seq_len}, {num_pred}.")
    paths = resolve_multimodal_nf_paths(data_root=data_root, raw_root=raw_root)
    channel_files = _resolve_channel_hdf5_files(paths, channel_path)
    image_files = _resolve_optional_hdf5_map(paths, image_path, suffix="_img")
    lidar_files = _resolve_optional_hdf5_map(paths, lidar_path, suffix="_lidar")
    h5py = _require_h5py("Multimodal-NF index builder")
    pending_rows: list[dict[str, Any]] = []
    beam_chunks = []
    los_chunks = []
    nf_chunks = []
    frame_items: list[dict[str, Any]] = []
    for channel_file in channel_files:
        with h5py.File(channel_file, "r") as handle:
            resolved = _resolve_hdf5_fields(handle)
            missing = [field for field in REQUIRED_MULTIMODAL_NF_FIELDS if field not in resolved]
            if missing:
                raise ValueError(
                    f"Multimodal-NF HDF5 file {channel_file} missing required fields {missing}. "
                    f"Available datasets: {_dataset_paths(handle)}."
                )
            count = int(handle[resolved["csi"]].shape[0])
            cities = _row_tokens(handle, resolved.get("city"), count, fallback=_city_from_path(channel_file))
            metadata_tokens = _metadata_row_tokens(handle, resolved.get("metadata"), count)
            trajectories = _row_tokens(
                handle,
                resolved.get("trajectory"),
                count,
                fallback_values=metadata_tokens,
                fallback_sequence=True,
                frames_per_traj=20,
            )
            frames = _row_tokens(
                handle,
                resolved.get("frame"),
                count,
                fallback_values=_frame_tokens_from_runs(trajectories) if metadata_tokens is not None else None,
                fallback_frames=True,
                frames_per_traj=20,
            )
            beam_idx = np.asarray(handle[resolved["beam_idx"]][:])
            beam_power = np.asarray(handle[resolved["beam_power"]][:])
            los = np.asarray(handle[resolved["los"]][:])
            nf = np.asarray(handle[resolved["nf"]][:])
        beam_chunks.append(beam_idx)
        los_chunks.append(los)
        nf_chunks.append(nf)
        file_city = _city_from_path(channel_file)
        image_file = image_files.get(file_city)
        lidar_file = lidar_files.get(file_city)
        for idx in range(count):
            city = str(cities[idx])
            trajectory = str(trajectories[idx])
            frame = str(frames[idx])
            frame_items.append(
                {
                    "channel_file": channel_file,
                    "channel_index": int(idx),
                    "global_index": len(frame_items),
                    "file_city": file_city,
                    "city": city,
                    "trajectory": trajectory,
                    "frame": frame,
                    "resolved": resolved,
                    "image_file": image_file,
                    "lidar_file": lidar_file,
                    "beam_top1_triplet": np.asarray(beam_idx[idx, 0]).astype(int).tolist(),
                    "beam_power_top1": float(np.asarray(beam_power[idx, 0]).reshape(())),
                    "los_label": int(np.asarray(los[idx]).reshape(())),
                    "nf_label": int(np.asarray(nf[idx]).reshape(())),
                }
            )
    window_items = _sequence_window_items(frame_items, seq_len=seq_len, num_pred=num_pred)
    splits = _assign_multimodal_nf_splits(
        [str(item["city"]) for item in window_items],
        split_mode=split_mode,
        train_cities=train_cities,
        val_cities=val_cities,
        test_cities=test_cities,
        split_ratios=split_ratios,
        seed=split_seed,
    )
    rows = []
    for row_idx, item in enumerate(window_items):
        city = item["city"]
        trajectory = item["trajectory"]
        frame = item["history_frames"][-1]
        channel_file = item["channel_file"]
        channel_index = int(item["history_indices"][-1])
        target_indices = [int(value) for value in item["target_indices"]]
        sample_id = (
            f"{city}:traj{trajectory}:hist{item['history_frames'][0]}-{item['history_frames'][-1]}"
            f":pred{item['target_frames'][0]}-{item['target_frames'][-1]}"
        )
        refs = {
            "channel_path": str(channel_file),
            "channel_index": channel_index,
            "history_indices": [int(value) for value in item["history_indices"]],
            "target_indices": target_indices,
            "global_index": int(item["target_global_indices"][0]),
            "hdf5_keys": item["resolved"],
        }
        if item["image_file"] is not None:
            refs["image_path"] = str(item["image_file"])
        if item["lidar_file"] is not None:
            refs["lidar_path"] = str(item["lidar_file"])
        rows.append(
            SampleRow(
                sample_id=sample_id,
                split=splits[row_idx],
                dataset_type=MULTIMODAL_NF_DATASET_TYPE,
                family=MULTIMODAL_NF_FAMILY,
                scene_or_city=city,
                trajectory_id=trajectory,
                frame_id=frame,
                resource_refs=refs,
                target_ref={
                    "channel_path": str(channel_file),
                    "channel_index": target_indices[0],
                    "target_indices": target_indices,
                    "beam_idx_key": item["resolved"]["beam_idx"],
                    "beam_power_key": item["resolved"]["beam_power"],
                },
                metadata={
                    "beam_top1_triplet": item["target_beam_top1_triplets"][0],
                    "beam_top1_triplets": item["target_beam_top1_triplets"],
                    "beam_power_top1": item["target_beam_power_top1"][0],
                    "beam_power_top1_horizon": item["target_beam_power_top1"],
                    "los_label": item["target_los_labels"][0],
                    "los_labels": item["target_los_labels"],
                    "nf_label": item["target_nf_labels"][0],
                    "nf_labels": item["target_nf_labels"],
                    "history_frame_ids": item["history_frames"],
                    "target_frame_ids": item["target_frames"],
                },
            )
        )
    beam_idx_all = np.concatenate(beam_chunks, axis=0) if beam_chunks else np.empty((0, 5, 3), dtype=np.int64)
    los_all = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in los_chunks], axis=0) if los_chunks else np.empty((0,))
    nf_all = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in nf_chunks], axis=0) if nf_chunks else np.empty((0,))
    metadata = _index_metadata(
        rows,
        split_mode=split_mode,
        split_ratios=split_ratios,
        split_seed=split_seed,
        data_root=paths.data_root,
        channel_paths=channel_files,
        beam_idx=beam_idx_all,
        los=los_all,
        nf=nf_all,
        seq_len=seq_len,
        num_pred=num_pred,
    )
    return rows, metadata


def load_multimodal_nf_index(path: str | Path, split: str | None = None) -> SampleIndex:
    index = SampleIndex.from_csv(path, storage_kind="hdf5_frame")
    if split is not None:
        return index.for_split(split)
    return index


def _sequence_window_items(frame_items: list[dict[str, Any]], *, seq_len: int, num_pred: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in frame_items:
        key = (str(item["channel_file"]), str(item["city"]), str(item["trajectory"]))
        grouped.setdefault(key, []).append(item)
    windows: list[dict[str, Any]] = []
    for _, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: (_frame_sort_key(item["frame"]), int(item["channel_index"])))
        if len(ordered) < seq_len + num_pred:
            continue
        for start in range(0, len(ordered) - seq_len - num_pred + 1):
            history = ordered[start : start + seq_len]
            future = ordered[start + seq_len : start + seq_len + num_pred]
            last_history = history[-1]
            windows.append(
                {
                    **last_history,
                    "history_indices": [int(item["channel_index"]) for item in history],
                    "history_global_indices": [int(item["global_index"]) for item in history],
                    "history_frames": [str(item["frame"]) for item in history],
                    "target_indices": [int(item["channel_index"]) for item in future],
                    "target_global_indices": [int(item["global_index"]) for item in future],
                    "target_frames": [str(item["frame"]) for item in future],
                    "target_beam_top1_triplets": [item["beam_top1_triplet"] for item in future],
                    "target_beam_power_top1": [float(item["beam_power_top1"]) for item in future],
                    "target_los_labels": [int(item["los_label"]) for item in future],
                    "target_nf_labels": [int(item["nf_label"]) for item in future],
                }
            )
    return windows


def _frame_sort_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), str(value)
    except (TypeError, ValueError):
        return 0, str(value)


def parse_codebook_metadata(
    codebook_path: str | Path | None = None,
    *,
    codebook_shape: list[int] | tuple[int, int, int] | None = None,
    profile: str | None = None,
    flatten_order: str = DEFAULT_FLATTEN_ORDER,
) -> dict[str, Any]:
    path = resolve_path(codebook_path) if codebook_path is not None else None
    raw: Any = {}
    if path is not None and path.exists() and codebook_shape is None and profile is None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".npz", ".npy"}:
            raw = _load_numpy_codebook(path)
        elif suffix in {".pkl", ".pickle"}:
            with path.open("rb") as handle:
                raw = pickle.load(handle)
        else:
            raise ValueError(f"Unsupported Multimodal-NF codebook file extension '{path.suffix}' for {path}.")
    shape = _codebook_shape_from_inputs(raw, codebook_shape=codebook_shape, profile=profile)
    num_classes = int(math.prod(shape))
    metadata = {
        "path": str(path) if path is not None else None,
        "fingerprint": fingerprint_path(path) if path is not None and path.exists() else None,
        "shape": list(shape),
        "flatten_order": str(flatten_order),
        "num_beam_classes": num_classes,
        "profile": profile or _profile_for_shape(shape),
    }
    return metadata


def flatten_beam_triplet(
    triplet: list[int] | tuple[int, int, int] | np.ndarray,
    codebook_shape: list[int] | tuple[int, int, int],
    *,
    flatten_order: str = DEFAULT_FLATTEN_ORDER,
) -> int:
    if str(flatten_order) != DEFAULT_FLATTEN_ORDER:
        raise ValueError(f"Unsupported Multimodal-NF flatten_order '{flatten_order}'.")
    values = np.asarray(triplet, dtype=np.int64).reshape(-1)
    if values.shape[0] != 3:
        raise ValueError(f"Beam triplet must contain 3 indices, got {values.tolist()}.")
    shape = tuple(int(value) for value in codebook_shape)
    if np.any(values < 0) or any(int(values[idx]) >= shape[idx] for idx in range(3)):
        raise ValueError(f"Beam triplet {values.tolist()} is outside codebook shape {list(shape)}.")
    az, el, rg = (int(value) for value in values)
    return int((az * shape[1] + el) * shape[2] + rg)


def unflatten_beam_class(
    class_id: int,
    codebook_shape: list[int] | tuple[int, int, int],
    *,
    flatten_order: str = DEFAULT_FLATTEN_ORDER,
) -> tuple[int, int, int]:
    if str(flatten_order) != DEFAULT_FLATTEN_ORDER:
        raise ValueError(f"Unsupported Multimodal-NF flatten_order '{flatten_order}'.")
    shape = tuple(int(value) for value in codebook_shape)
    total = int(math.prod(shape))
    value = int(class_id)
    if value < 0 or value >= total:
        raise ValueError(f"Beam class {value} is outside codebook size {total}.")
    az, rem = divmod(value, shape[1] * shape[2])
    el, rg = divmod(rem, shape[2])
    return int(az), int(el), int(rg)


def fingerprint_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hdf5_file_summary(path: Path) -> dict[str, Any]:
    h5py = _require_h5py("Multimodal-NF audit")
    with h5py.File(path, "r") as handle:
        resolved = _resolve_hdf5_fields(handle)
        datasets = {
            dataset_path: {
                "shape": [int(value) for value in item.shape],
                "dtype": str(item.dtype),
            }
            for dataset_path, item in _iter_hdf5_datasets(handle)
        }
        sample_count = 0
        if "csi" in resolved:
            sample_count = int(handle[resolved["csi"]].shape[0])
        cities = []
        if "city" in resolved:
            cities = sorted({str(item) for item in _decode_hdf5_values(np.asarray(handle[resolved["city"]][:]))})
        elif sample_count:
            cities = [_city_from_path(path)]
        return {
            "path": str(path),
            "fingerprint": fingerprint_path(path),
            "datasets": datasets,
            "resolved_fields": resolved,
            "cities": cities,
            "sample_count": sample_count,
            "missing_fields": [field for field in REQUIRED_MULTIMODAL_NF_FIELDS if field not in resolved],
        }


def _resolve_hdf5_fields(handle) -> dict[str, str]:
    paths = _dataset_paths(handle)
    by_leaf = {Path(path).name.lower(): path for path in paths}
    resolved = {}
    for field, aliases in MULTIMODAL_NF_HDF5_KEYS.items():
        for alias in aliases:
            alias_key = alias.lower()
            if alias in paths:
                resolved[field] = alias
                break
            if alias_key in by_leaf:
                resolved[field] = by_leaf[alias_key]
                break
    return resolved


def _candidate_hdf5_files(
    paths: MultimodalNFPaths,
    *,
    channel_path: str | Path | None,
    image_path: str | Path | None,
    lidar_path: str | Path | None,
) -> list[Path]:
    explicit = [
        resolve_path(item)
        for item in (channel_path, image_path, lidar_path)
        if item is not None
    ]
    if explicit:
        return list(dict.fromkeys(explicit))
    candidates = []
    for root in (paths.raw_root, paths.data_root):
        if root.exists():
            candidates.extend(sorted(root.rglob("*.h5")))
            candidates.extend(sorted(root.rglob("*.hdf5")))
    return list(dict.fromkeys(candidates))


def _candidate_codebook_files(paths: MultimodalNFPaths, *, codebook_path: str | Path | None) -> list[Path]:
    if codebook_path is not None:
        return [resolve_path(codebook_path)]
    candidates = []
    for root in (paths.codebook_root, paths.data_root):
        if root.exists():
            for suffix in ("*.pkl", "*.pickle", "*.json", "*.npz", "*.npy"):
                candidates.extend(sorted(root.rglob(suffix)))
    return list(dict.fromkeys(candidates))


def _resolve_channel_hdf5(paths: MultimodalNFPaths, channel_path: str | Path | None) -> Path:
    return _resolve_channel_hdf5_files(paths, channel_path)[0]


def _resolve_channel_hdf5_files(paths: MultimodalNFPaths, channel_path: str | Path | None) -> list[Path]:
    if channel_path is not None:
        path = resolve_path(channel_path)
        if not path.exists():
            raise FileNotFoundError(f"Multimodal-NF channel HDF5 not found: {path}")
        return [path]
    matches = []
    for candidate in _candidate_hdf5_files(paths, channel_path=None, image_path=None, lidar_path=None):
        stem = candidate.stem.lower()
        if "_img" in stem or "_lidar" in stem:
            continue
        try:
            summary = _hdf5_file_summary(candidate)
        except OSError:
            continue
        if not summary["missing_fields"]:
            matches.append(candidate)
    if matches:
        return sorted(matches)
    raise FileNotFoundError(
        "Could not find a Multimodal-NF channel HDF5 file with required fields "
        f"{list(REQUIRED_MULTIMODAL_NF_FIELDS)} under {paths.raw_root} or {paths.data_root}."
    )


def _resolve_optional_hdf5_path(paths: MultimodalNFPaths, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configured Multimodal-NF HDF5 file not found: {resolved}")
    return resolved


def _resolve_optional_hdf5_map(
    paths: MultimodalNFPaths,
    path: str | Path | None,
    *,
    suffix: str,
) -> dict[str, Path]:
    if path is not None:
        resolved = _resolve_optional_hdf5_path(paths, path)
        return {_city_from_path(resolved): resolved} if resolved is not None else {}
    matches: dict[str, Path] = {}
    for candidate in _candidate_hdf5_files(paths, channel_path=None, image_path=None, lidar_path=None):
        stem = candidate.stem.lower()
        if suffix.lower() not in stem:
            continue
        matches.setdefault(_city_from_path(candidate), candidate)
    return dict(sorted(matches.items()))


def _assign_multimodal_nf_splits(
    cities: list[str],
    *,
    split_mode: str,
    train_cities: list[str] | tuple[str, ...] | None,
    val_cities: list[str] | tuple[str, ...] | None,
    test_cities: list[str] | tuple[str, ...] | None,
    split_ratios: list[float] | tuple[float, float, float],
    seed: int,
) -> list[str]:
    mode = str(split_mode or "city").lower()
    if mode in {"frame", "frame_debug", "debug"}:
        return _ratio_splits(len(cities), split_ratios=split_ratios, seed=seed)
    if mode != "city":
        raise ValueError("Multimodal-NF split_mode must be 'city' or 'frame_debug'.")
    explicit = {
        "train": set(str(item) for item in (train_cities or ())),
        "validation": set(str(item) for item in (val_cities or ())),
        "test": set(str(item) for item in (test_cities or ())),
    }
    if any(explicit.values()):
        assigned = []
        for city in cities:
            matches = [split for split, values in explicit.items() if str(city) in values]
            assigned.append(matches[0] if matches else "train")
        return assigned
    unique_cities = sorted(set(str(city) for city in cities))
    city_splits = dict(zip(unique_cities, _ratio_splits(len(unique_cities), split_ratios=split_ratios, seed=seed)))
    return [city_splits[str(city)] for city in cities]


def _ratio_splits(
    count: int,
    *,
    split_ratios: list[float] | tuple[float, float, float],
    seed: int,
) -> list[str]:
    if count <= 0:
        return []
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("split_ratios must contain three non-negative values for train/validation/test.")
    ratios = ratios / ratios.sum()
    indices = np.arange(count)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    if count == 1:
        train_end, val_end = 1, 1
    elif count == 2:
        train_end, val_end = 1, 1
    else:
        train_end = max(1, min(int(math.floor(count * ratios[0])), count - 2))
        val_end = max(train_end + 1, min(train_end + int(math.floor(count * ratios[1])), count - 1))
    names = np.full(count, "test", dtype=object)
    names[indices[:train_end]] = "train"
    names[indices[train_end:val_end]] = "validation"
    return [str(item) for item in names.tolist()]


def _index_metadata(
    rows: list[SampleRow],
    *,
    split_mode: str,
    split_ratios: Any,
    split_seed: int,
    data_root: Path,
    channel_paths: list[Path],
    beam_idx: np.ndarray,
    los: np.ndarray,
    nf: np.ndarray,
    seq_len: int,
    num_pred: int,
) -> dict[str, Any]:
    by_split: dict[str, list[SampleRow]] = {}
    for row in rows:
        by_split.setdefault(_normalize_split(row.split), []).append(row)
    metadata = {
        "dataset": MULTIMODAL_NF_DATASET_TYPE,
        "family": MULTIMODAL_NF_FAMILY,
        "data_root": str(data_root),
        "split_protocol": "city_level" if str(split_mode).lower() == "city" else "frame_level_debug",
        "task_semantics": "future_near_field_beam_prediction",
        "seq_len": int(seq_len),
        "num_pred": int(num_pred),
        "pred_horizon": int(num_pred),
        "split_mode": split_mode,
        "split_seed": int(split_seed),
        "split_ratios": [float(value) for value in split_ratios],
        "sample_count": int(len(rows)),
        "fingerprint": _fingerprint([row.sample_id for row in rows]),
        "channel_paths": [str(path) for path in channel_paths],
        "channel_fingerprints": {str(path): fingerprint_path(path) for path in channel_paths},
        "splits": {},
    }
    if channel_paths:
        metadata["channel_path"] = str(channel_paths[0])
        metadata["channel_fingerprint"] = fingerprint_path(channel_paths[0])
    for split, split_rows in sorted(by_split.items()):
        indices = [int(row.resource_refs.get("global_index", row.resource_refs["channel_index"])) for row in split_rows]
        metadata["splits"][split] = {
            "num_samples": len(split_rows),
            "cities": sorted({str(row.scene_or_city) for row in split_rows}),
            "beam_label_distribution": _beam_distribution(beam_idx[indices]),
            "los_distribution": _value_counts(np.asarray(los)[indices]),
            "nf_distribution": _value_counts(np.asarray(nf)[indices]),
        }
    return metadata


def _resolve_prediction_horizon(*, num_pred: int | None, pred_horizon: int | None) -> int:
    if pred_horizon is not None:
        return int(pred_horizon)
    if num_pred is not None:
        return int(num_pred)
    return 3


def _index_prefix(*, seq_len: int, num_pred: int) -> str:
    return f"multimodal_nf_seq{int(seq_len)}_pred{int(num_pred)}"


def _beam_distribution(beam_idx: np.ndarray) -> dict[str, int]:
    if beam_idx.size == 0:
        return {}
    top1 = np.asarray(beam_idx)[:, 0, :].reshape(-1, 3)
    counts: dict[str, int] = {}
    for triplet in top1:
        key = ",".join(str(int(value)) for value in triplet)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _value_counts(values: np.ndarray) -> dict[str, int]:
    flat = np.asarray(values).reshape(-1)
    counts: dict[str, int] = {}
    for value in flat:
        key = str(int(value)) if np.issubdtype(np.asarray(value).dtype, np.number) else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _row_tokens(
    handle,
    dataset_path: str | None,
    count: int,
    *,
    fallback_values: list[str] | None = None,
    fallback: str | None = None,
    fallback_sequence: bool = False,
    fallback_frames: bool = False,
    frames_per_traj: int = 20,
) -> list[str]:
    if dataset_path is not None:
        values = _decode_hdf5_values(np.asarray(handle[dataset_path][:]))
        if len(values) >= count:
            return [str(item) for item in values[:count]]
    if fallback_values is not None and len(fallback_values) >= count:
        return [str(item) for item in fallback_values[:count]]
    if fallback is not None:
        return [str(fallback)] * count
    if fallback_sequence:
        return [str(idx // int(frames_per_traj)) for idx in range(count)]
    if fallback_frames:
        return [str(idx % int(frames_per_traj)) for idx in range(count)]
    return [str(idx) for idx in range(count)]


def _metadata_row_tokens(handle, dataset_path: str | None, count: int) -> list[str] | None:
    if dataset_path is None or dataset_path not in handle:
        return None
    values = np.asarray(handle[dataset_path][:count])
    if values.shape[0] < count:
        return None
    rows = values.reshape(count, -1)
    return ["_".join(_decode_token_value(value) for value in row.tolist()) for row in rows]


def _frame_tokens_from_runs(trajectory_tokens: list[str]) -> list[str]:
    frames = []
    previous = None
    frame_idx = 0
    for token in trajectory_tokens:
        if token != previous:
            frame_idx = 0
            previous = token
        frames.append(str(frame_idx))
        frame_idx += 1
    return frames


def _decode_hdf5_values(values: np.ndarray) -> list[Any]:
    decoded = []
    for item in values.reshape(-1).tolist():
        decoded.append(_decode_token_value(item))
    return decoded


def _decode_token_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _dataset_paths(handle) -> list[str]:
    return [path for path, _ in _iter_hdf5_datasets(handle)]


def _iter_hdf5_datasets(handle) -> list[tuple[str, Any]]:
    h5py = _require_h5py("Multimodal-NF HDF5 traversal")
    datasets = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            datasets.append((name, item))

    handle.visititems(visitor)
    return datasets


def _city_from_path(path: Path) -> str:
    match = re.search(r"City[_-]?([A-Za-z0-9]+)", path.stem, flags=re.IGNORECASE)
    if match:
        return f"City_{match.group(1)}"
    return path.stem


def _load_numpy_codebook(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {key: loaded[key] for key in loaded.files}
    return {"array": loaded}


def _codebook_shape_from_inputs(
    raw: Any,
    *,
    codebook_shape: list[int] | tuple[int, int, int] | None,
    profile: str | None,
) -> tuple[int, int, int]:
    if codebook_shape is not None:
        return _normalize_codebook_shape(codebook_shape)
    if profile:
        normalized = str(profile).strip().lower()
        if normalized == "dense":
            return DEFAULT_DENSE_CODEBOOK_SHAPE
        if normalized == "small":
            return DEFAULT_SMALL_CODEBOOK_SHAPE
    if isinstance(raw, dict):
        for key in ("shape", "codebook_shape", "beam_codebook_shape"):
            if key in raw:
                return _normalize_codebook_shape(raw[key])
        if all(key in raw for key in ("num_azimuth", "num_elevation", "num_range")):
            return _normalize_codebook_shape((raw["num_azimuth"], raw["num_elevation"], raw["num_range"]))
        for key in ("codebook", "array", "beam_codebook"):
            if key in raw:
                array = np.asarray(raw[key])
                if array.ndim >= 3:
                    return _normalize_codebook_shape(array.shape[:3])
    if isinstance(raw, (list, tuple, np.ndarray)):
        array = np.asarray(raw)
        if array.ndim >= 3:
            return _normalize_codebook_shape(array.shape[:3])
        if array.size == 3:
            return _normalize_codebook_shape(array.reshape(-1).tolist())
    raise ValueError(
        "Could not parse Multimodal-NF codebook shape. Configure codebook_shape, "
        "codebook_profile ('dense' or 'small'), or provide metadata with shape/codebook_shape."
    )


def _normalize_codebook_shape(value: Any) -> tuple[int, int, int]:
    values = [int(item) for item in np.asarray(value).reshape(-1).tolist()]
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f"Multimodal-NF codebook shape must contain three positive integers, got {value}.")
    return int(values[0]), int(values[1]), int(values[2])


def _profile_for_shape(shape: tuple[int, int, int]) -> str | None:
    if tuple(shape) == DEFAULT_DENSE_CODEBOOK_SHAPE:
        return "dense"
    if tuple(shape) == DEFAULT_SMALL_CODEBOOK_SHAPE:
        return "small"
    return None


def _fingerprint(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_split(split: str) -> str:
    key = str(split).strip().lower()
    return {"val": "validation", "valid": "validation"}.get(key, key)


def _require_h5py(context: str):
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{context} requires the 'h5py' dependency. Install project dependencies in kd_mm_beam."
        ) from exc
    return h5py


__all__ = [
    "DEFAULT_DENSE_CODEBOOK_SHAPE",
    "DEFAULT_FLATTEN_ORDER",
    "DEFAULT_SMALL_CODEBOOK_SHAPE",
    "MULTIMODAL_NF_DATASET_TYPE",
    "MULTIMODAL_NF_HDF5_KEYS",
    "REQUIRED_MULTIMODAL_NF_FIELDS",
    "MultimodalNFPaths",
    "audit_multimodal_nf_files",
    "build_multimodal_nf_index",
    "build_multimodal_nf_rows",
    "fingerprint_path",
    "flatten_beam_triplet",
    "load_multimodal_nf_index",
    "parse_codebook_metadata",
    "resolve_multimodal_nf_paths",
    "unflatten_beam_class",
]
