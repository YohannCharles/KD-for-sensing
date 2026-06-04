from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pandas as pd
import torch

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.data.transform_ops.image import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from kd_sensing.data.transform_ops.lidar import filter_lidar_points, read_lidar_point_cloud
from kd_sensing.diagnostics.viewer_manifest_paths import _all_row_paths, _all_source_paths, _last_existing_path, _radar_da_path
from kd_sensing.diagnostics.viewer_manifest_schema import _json_ready
from kd_sensing.diagnostics.visualization.sampling import SampleCandidate
from kd_sensing.diagnostics.visualization.stats import modality_statistics
from kd_sensing.modalities import DEFAULT_IMAGE_PROFILE

def _manifest_record(
    dataset: Any,
    *,
    split: str,
    row: pd.Series,
    candidate: SampleCandidate,
    sample: dict[str, Any],
    sample_id: str,
    enabled_modalities: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    data_root = Path(str(getattr(dataset, "data_root", ""))).expanduser()
    asset_dir = output_dir / "viewer_assets" / sample_id
    data_spaces: dict[str, dict[str, str]] = {"raw": {}, "processed": {}}
    raw = _write_raw_assets(row, sample, asset_dir, data_root, dataset=dataset, data_spaces=data_spaces)
    processed = _write_processed_assets(sample, asset_dir, dataset=dataset, data_spaces=data_spaces)
    current_beam = _tensor_int_list(sample.get("input_beam"))
    future_beams = _tensor_int_list(sample.get("target_beam"))
    label_metadata = _beam_label_metadata(dataset, sample)
    extra = {
        "dataset_index": int(candidate.dataset_index),
        "csv_row_index": int(candidate.csv_row_index),
        "csv_path": str(getattr(dataset, "root_csv", "")),
        "enabled_modalities": list(enabled_modalities),
        "image_profile": getattr(dataset, "image_profile", None),
        "statistics": modality_statistics(sample),
        "source_paths": _all_source_paths(row, data_root),
        "data_spaces": data_spaces,
        **label_metadata,
    }
    return {
        "sample_id": sample_id,
        "scene_id": getattr(dataset, "scene_id", None),
        "scene_slug": getattr(dataset, "scene_slug", None),
        "split": split,
        "sequence_id": candidate.seq_index,
        "time_index": int(candidate.dataset_index),
        "timestamp": str(candidate.csv_row_index),
        "raw": raw,
        "processed": processed,
        "label": {
            "current_beam": current_beam[-1] if current_beam else None,
            "future_beams": future_beams or ([candidate.future_label] if candidate.future_label is not None else []),
            **label_metadata,
        },
        "extra": extra,
    }


def _beam_label_metadata(dataset: Any, sample: dict[str, Any]) -> dict[str, Any]:
    sample_metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    mapping = getattr(dataset, "beam_label_mapping", None)
    if mapping is not None and hasattr(mapping, "metadata"):
        payload = mapping.metadata()
    else:
        payload = resolve_beam_label_mapping(None).metadata()
    for key in (
        "raw_input_beam",
        "raw_target_beam",
        "calibrated_input_beam",
        "calibrated_target_beam",
        "input_beam_label_source",
        "target_beam_label_source",
    ):
        if key in sample_metadata:
            payload[key] = sample_metadata[key]
    return payload


def _write_raw_assets(
    row: pd.Series,
    sample: dict[str, Any],
    asset_dir: Path,
    data_root: Path,
    *,
    dataset: Any,
    data_spaces: dict[str, dict[str, str]],
) -> dict[str, str]:
    raw: dict[str, str] = {}
    asset_dir.mkdir(parents=True, exist_ok=True)

    raw["image"] = _last_existing_path(row, "camera", data_root)
    if raw["image"]:
        data_spaces["raw"]["image"] = "source_image"

    lidar_source_path = _last_existing_path(row, "lidar", data_root)
    if lidar_source_path:
        path = asset_dir / "raw_lidar_points.png"
        lidar_space = _save_raw_lidar_preview(
            lidar_source_path,
            path,
            roi=getattr(dataset, "lidar_roi", None),
            bev_size=getattr(dataset, "lidar_bev_size", (224, 224)),
        )
        if lidar_space is not None:
            raw["lidar"] = str(path)
            data_spaces["raw"]["lidar"] = lidar_space
        else:
            raw["lidar"] = lidar_source_path
            data_spaces["raw"]["lidar"] = "source_lidar_unrendered"
    else:
        raw["lidar"] = ""

    radar_path = _last_existing_path(row, "radar", data_root)
    if radar_path:
        path = asset_dir / "raw_radar.png"
        if _save_raw_radar_preview(radar_path, path):
            raw["radar"] = str(path)
            data_spaces["raw"]["radar"] = _radar_source_space(radar_path)
        else:
            raw["radar"] = radar_path
            data_spaces["raw"]["radar"] = "source_radar_unrendered"
    else:
        raw["radar"] = ""

    gps_payload = _raw_gps_payload(_all_row_paths(row, "gps", data_root))
    if gps_payload is not None:
        path = asset_dir / "raw_gps.json"
        path.write_text(json.dumps(_json_ready(gps_payload), ensure_ascii=False), encoding="utf-8")
        raw["gps"] = str(path)
        data_spaces["raw"]["gps"] = "lat_lon"
    else:
        raw["gps"] = _last_existing_path(row, "gps", data_root)

    mmwave_payload = _raw_mmwave_payload(_all_row_paths(row, "mmwave", data_root))
    if mmwave_payload is not None:
        path = asset_dir / "raw_mmwave.json"
        path.write_text(json.dumps(_json_ready(mmwave_payload), ensure_ascii=False), encoding="utf-8")
        raw["mmwave"] = str(path)
        data_spaces["raw"]["mmwave"] = "linear_power"
    else:
        raw["mmwave"] = _last_existing_path(row, "mmwave", data_root)

    return raw


def _write_processed_assets(
    sample: dict[str, Any],
    asset_dir: Path,
    *,
    dataset: Any,
    data_spaces: dict[str, dict[str, str]],
) -> dict[str, str]:
    processed: dict[str, str] = {}
    asset_dir.mkdir(parents=True, exist_ok=True)
    if "image" in sample:
        path = asset_dir / "processed_image.png"
        profile = str(getattr(dataset, "image_profile", DEFAULT_IMAGE_PROFILE) or DEFAULT_IMAGE_PROFILE)
        _save_processed_image(_last_time_frame(sample["image"]), path, profile=profile)
        processed["image"] = str(path)
        data_spaces["processed"]["image"] = profile
    if "lidar" in sample:
        path = asset_dir / "processed_lidar.png"
        _save_lidar_image(sample["lidar"], path)
        processed["lidar"] = str(path)
        data_spaces["processed"]["lidar"] = "dataset_bev"
    if "radar_ra" in sample or "radar_da" in sample:
        path = asset_dir / "processed_radar.png"
        _save_radar_image(sample.get("radar_ra"), sample.get("radar_da"), path)
        processed["radar"] = str(path)
        data_spaces["processed"]["radar"] = "dataset_ra_da"
    if "gps" in sample:
        path = asset_dir / "processed_gps.json"
        gps_payload = _processed_gps_payload(sample["gps"], dataset)
        path.write_text(json.dumps(_json_ready(gps_payload), ensure_ascii=False), encoding="utf-8")
        processed["gps"] = str(path)
        data_spaces["processed"]["gps"] = str(gps_payload.get("feature_space", "gps_features"))
    if "mmwave" in sample:
        path = asset_dir / "processed_mmwave.json"
        payload = _processed_mmwave_payload(sample["mmwave"], dataset)
        path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False), encoding="utf-8")
        processed["mmwave"] = str(path)
        data_spaces["processed"]["mmwave"] = str(payload.get("scale", "mmwave_features"))
    return processed


def _save_raw_lidar_preview(
    lidar_path: str,
    output_path: Path,
    *,
    roi: list[float] | tuple[float, ...] | None,
    bev_size: list[int] | tuple[int, int],
) -> str | None:
    source = Path(lidar_path).expanduser()
    if not source.exists():
        return None
    try:
        if source.suffix.lower() == ".npy":
            array = np.load(source)
            if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
                _save_lidar_image(array, output_path)
                return "precomputed_bev"
        points = read_lidar_point_cloud(Path("/"), str(source))
        _save_lidar_points_topdown(points, output_path, roi=roi, bev_size=bev_size)
        return "point_cloud_topdown"
    except Exception:
        return None


def _save_lidar_points_topdown(
    points: np.ndarray,
    path: Path,
    *,
    roi: list[float] | tuple[float, ...] | None,
    bev_size: list[int] | tuple[int, int],
) -> None:
    height, width = int(bev_size[0]), int(bev_size[1])
    if roi is None:
        finite = np.asarray(points, dtype=np.float32)
        finite = finite[np.isfinite(finite).all(axis=1)] if finite.ndim == 2 else np.empty((0, 4), dtype=np.float32)
        if finite.size:
            x_min, x_max = float(np.min(finite[:, 0])), float(np.max(finite[:, 0]))
            y_min, y_max = float(np.min(finite[:, 1])), float(np.max(finite[:, 1]))
        else:
            x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0
        roi_values = (x_min, x_max, y_min, y_max)
        filtered = finite
    else:
        roi_values = tuple(float(value) for value in roi)
        filtered = filter_lidar_points(points, roi=roi_values)
    image = np.zeros((height, width), dtype=np.float32)
    if filtered.size:
        x_min, x_max, y_min, y_max = roi_values[:4]
        x_span = max(float(x_max) - float(x_min), 1e-6)
        y_span = max(float(y_max) - float(y_min), 1e-6)
        cols = np.floor((filtered[:, 0] - float(x_min)) / x_span * width).astype(np.int64)
        rows = height - 1 - np.floor((filtered[:, 1] - float(y_min)) / y_span * height).astype(np.int64)
        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)
        np.add.at(image, (rows, cols), 1.0)
        image = np.log1p(image)
    Image.fromarray(_normalize_to_uint8(image)).convert("RGB").save(path)


def _processed_gps_payload(value: Any, dataset: Any) -> dict[str, Any]:
    array = _as_numpy(value)
    feature_space = str(getattr(dataset, "gps_feature_mode", "gps_features") or "gps_features")
    normalized = getattr(dataset, "gps_scaler", None) is not None
    names = _gps_feature_names(feature_space, array)
    return {
        "features": array.tolist(),
        "feature_names": names,
        "feature_space": f"{feature_space}_zscore" if normalized else feature_space,
        "normalized": bool(normalized),
    }


def _gps_feature_names(feature_space: str, array: np.ndarray) -> list[str]:
    last_dim = int(array.shape[-1]) if array.ndim else 1
    if feature_space == "relative_polar" and last_dim == 3:
        return ["distance", "sin_theta", "cos_theta"]
    return [f"feature_{idx}" for idx in range(last_dim)]


def _processed_mmwave_payload(value: Any, dataset: Any) -> dict[str, Any]:
    array = _as_numpy(value)
    normalized = getattr(dataset, "mmwave_scaler", None) is not None and bool(getattr(dataset, "mmwave_normalize", False))
    payload = {"beam_power_seq": array.tolist()} if array.ndim >= 2 else {"beam_power": array.reshape(-1).tolist()}
    payload.update(
        {
            "scale": "z_score" if normalized else "db",
            "units": "standard_deviation" if normalized else "dB",
            "normalized": bool(normalized),
        }
    )
    return payload


def _save_raw_radar_preview(radar_path: str, output_path: Path) -> bool:
    ra_path = Path(radar_path).expanduser()
    if not ra_path.exists():
        return False
    da_path = _radar_da_path(ra_path)
    try:
        ra = np.load(ra_path)
        da = np.load(da_path) if da_path is not None and da_path.exists() else None
    except Exception:
        return False
    _save_radar_image(ra, da, output_path)
    return output_path.exists()


def _radar_source_space(radar_path: str) -> str:
    name = Path(radar_path).name
    if "_RA" in name or "_DA" in name:
        return "precomputed_ra_da"
    return "source_radar_array"



def _raw_gps_payload(paths: list[str]) -> dict[str, list[float]] | None:
    points: list[list[float]] = []
    for path in paths:
        try:
            values = np.loadtxt(path, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if values.size >= 2 and np.all(np.isfinite(values[:2])):
            points.append([float(values[0]), float(values[1])])
    if not points:
        return None
    return {
        "x": [point[0] for point in points],
        "y": [point[1] for point in points],
        "coordinate_space": "lat_lon",
    }


def _raw_mmwave_payload(paths: list[str]) -> dict[str, Any] | None:
    rows: list[list[float]] = []
    for path in paths:
        try:
            values = np.loadtxt(path, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if values.size > 0 and np.all(np.isfinite(values)):
            rows.append(values.astype(float).tolist())
    if not rows:
        return None
    if len(rows) == 1:
        return {"beam_power": rows[0], "scale": "linear_power", "normalized": False}
    return {"beam_power_seq": rows, "scale": "linear_power", "normalized": False}


def _tensor_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return _as_numpy(value).reshape(-1).astype(int).tolist()


def _last_time_frame(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim >= 3:
        return array[-1]
    return array


def _save_scalar_image(value: Any, path: Path) -> None:
    array = _as_numpy(value)
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        image = _normalize_to_uint8(array[..., :3])
        Image.fromarray(image).save(path)
        return
    if array.ndim == 3:
        array = array[0]
    Image.fromarray(_normalize_to_uint8(array)).save(path)


def _save_processed_image(value: Any, path: Path, *, profile: str) -> None:
    array = _as_numpy(value)
    if profile == "rgb_imagenet" and array.ndim == 3 and array.shape[0] == 3:
        mean = np.asarray(IMAGENET_RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(IMAGENET_RGB_STD, dtype=np.float32).reshape(3, 1, 1)
        rgb = np.clip(array.astype(np.float32) * std + mean, 0.0, 1.0)
        Image.fromarray((np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)).save(path)
        return
    _save_scalar_image(array, path)


def _save_lidar_image(value: Any, path: Path) -> None:
    array = _as_numpy(value)
    if array.ndim == 4:
        array = array[-1]
    if array.ndim == 3:
        channels = [_normalize_float(array[idx]) for idx in range(min(3, array.shape[0]))]
        while len(channels) < 3:
            channels.append(np.zeros_like(channels[0] if channels else array[0], dtype=np.float32))
        image = (np.stack(channels, axis=-1) * 255).astype(np.uint8)
        Image.fromarray(image).save(path)
        return
    Image.fromarray(_normalize_to_uint8(array)).save(path)


def _save_radar_image(ra: Any, da: Any, path: Path) -> None:
    panels = []
    for value in (ra, da):
        if value is None:
            continue
        array = _as_numpy(value)
        if array.ndim >= 3:
            array = array[-1]
        panels.append(_normalize_to_uint8(array))
    if not panels:
        panels = [np.zeros((16, 16), dtype=np.uint8)]
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.zeros((height - panel.shape[0], panel.shape[1]), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    Image.fromarray(np.hstack(padded)).save(path)


def _normalize_to_uint8(value: Any) -> np.ndarray:
    array = _normalize_float(value)
    return (array * 255).astype(np.uint8)


def _normalize_float(value: Any) -> np.ndarray:
    array = _as_numpy(value).astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - min_value) / (max_value - min_value), 0.0, 1.0)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = ["_manifest_record"]
