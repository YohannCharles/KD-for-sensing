import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kd_sensing.data.transform_ops.io import atomic_save_npy
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


MMW_RADAR_PREPARATION_COMMAND = (
    "conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/mmw_radar_maps.yaml"
)


def generate_mmw_radar_maps(
    data_root: str | Path = "dataset/MMW/sunny",
    scenes: list[str] | tuple[str, ...] | None = None,
    prepared_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    range_bins: int = 128,
    angle_bins: int = 64,
    doppler_bins: int = 128,
    depth_range: list[float] | tuple[float, float] = (0.0, 160.0),
    azimuth_range: list[float] | tuple[float, float] = (-0.96, 0.96),
    velocity_range: list[float] | tuple[float, float] = (-10.0, 10.0),
    overwrite: bool = False,
    progress: bool = True,
    materialize_split_columns: bool = True,
) -> dict[str, Any]:
    root = resolve_path(data_root)
    prepared = _resolve_prepared_roots(root, scenes=scenes, prepared_roots=prepared_roots)
    reports = []
    total_frames = 0
    total_generated = 0
    total_skipped = 0
    total_failed = 0
    for prepared_root in prepared:
        report = _generate_scene_radar_maps(
            root,
            prepared_root,
            range_bins=int(range_bins),
            angle_bins=int(angle_bins),
            doppler_bins=int(doppler_bins),
            depth_range=tuple(float(value) for value in depth_range),
            azimuth_range=tuple(float(value) for value in azimuth_range),
            velocity_range=tuple(float(value) for value in velocity_range),
            overwrite=bool(overwrite),
            progress=bool(progress),
            materialize_split_columns=bool(materialize_split_columns),
        )
        reports.append(report)
        total_frames += int(report["unique_radar_frames"])
        total_generated += int(report["generated"])
        total_skipped += int(report["skipped"])
        total_failed += int(report["failed"])
    payload = {
        "type": "mmw_radar_maps",
        "data_root": str(root),
        "scene_count": len(reports),
        "unique_radar_frames": total_frames,
        "generated": total_generated,
        "skipped": total_skipped,
        "failed": total_failed,
        "scenes": reports,
    }
    report_path = root / "Prepared" / "mmw_radar_maps_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


def materialize_mmw_radar_split_csv(
    data_root: str | Path,
    csv_path: str | Path,
    scene: str,
    *,
    output_path: str | Path | None = None,
    require_maps: bool = True,
) -> dict[str, Any]:
    root = resolve_path(data_root)
    source = Path(csv_path)
    if not source.is_absolute():
        source = root / source
    if not source.exists():
        raise FileNotFoundError(
            f"MMW split CSV is missing: {source}. Prepare sequence splits with "
            "conda run -n kd_mm_beam kd-sensing-preprocess --action mmw_sequence_splits_from_manifest."
        )
    target = Path(output_path) if output_path is not None else source.with_name(f"{source.stem}_with_radar{source.suffix}")
    if not target.is_absolute():
        target = root / target
    frame = pd.read_csv(source, na_values="").fillna("")
    radar_cols = _numbered_columns(frame.columns, "radar")
    beam_cols = _numbered_columns(frame.columns, "beam")
    if radar_cols and output_path is None:
        target = source
        output = frame
        created = False
    else:
        existing = _valid_materialized_radar_csv(target)
        if existing is not None:
            output = existing
            radar_cols = _numbered_columns(output.columns, "radar")
            created = False
        else:
            if not beam_cols:
                raise ValueError(f"Cannot materialize radar columns for {source}; no beamN columns were found.")
            output = frame.copy()
            missing = []
            for beam_col in beam_cols:
                suffix = beam_col[len("beam") :]
                values = []
                for value in output[beam_col].tolist():
                    rel_path = _radar_rel_path_for_beam(value, str(scene))
                    values.append(rel_path)
                    ra_path = root / rel_path
                    da_path = root / rel_path.replace("_RA", "_DA")
                    if not ra_path.exists():
                        missing.append(str(ra_path))
                    if not da_path.exists():
                        missing.append(str(da_path))
                output[f"radar{suffix}"] = values
            if missing and require_maps:
                examples = ", ".join(missing[:4])
                raise FileNotFoundError(
                    "MMW radar split CSV cannot be materialized because prepared radar map artifacts are missing. "
                    f"scene={scene}; source_csv={source}; target_output={target}; missing_count={len(missing)}; "
                    f"examples={examples}. Run: {MMW_RADAR_PREPARATION_COMMAND} "
                    f"(ensure preprocessing.scenes contains {scene} and data_root={root})."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_csv(output, target)
            created = True
            radar_cols = _numbered_columns(output.columns, "radar")
    metadata_path = target.with_name(f"{target.stem}_metadata.json")
    split_metadata_path = source.parent / "split_metadata.json"
    metadata = {
        "type": "mmw_radar_split_csv_materialization",
        "public_utility": "kd_sensing.preprocessing.mmw_radar.materialize_mmw_radar_split_csv",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(root),
        "condition": root.name,
        "scene": str(scene),
        "scenario": str(scene),
        "input_manifest": str(root / "Prepared" / str(scene) / "manifests" / "frame_manifest.csv"),
        "source_csv": str(source),
        "output_csv": str(target),
        "metadata_path": str(metadata_path),
        "split_metadata_path": str(split_metadata_path) if split_metadata_path.exists() else None,
        "split_config": _load_split_metadata_summary(split_metadata_path),
        "seq_len": len(beam_cols),
        "num_pred": len(_numbered_columns(output.columns, "future_beam")),
        "sample_count": int(len(output)),
        "radar_columns": len(radar_cols),
        "created_output": created,
        "repair_command": f"{MMW_RADAR_PREPARATION_COMMAND} # ensure preprocessing.scenes contains {scene}",
    }
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(target),
        "metadata_path": str(metadata_path),
        "created": created,
        "metadata": metadata,
    }


def _valid_materialized_radar_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path, na_values="").fillna("")
    except (pd.errors.EmptyDataError, OSError):
        return None
    if _numbered_columns(frame.columns, "beam") and _numbered_columns(frame.columns, "radar"):
        return frame
    return None


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = _atomic_temp_path(path)
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(path)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_temp_path(path: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).timestamp()
    return path.with_name(f".{path.name}.{os.getpid()}.{stamp:.6f}.tmp")


def _generate_scene_radar_maps(
    data_root: Path,
    prepared_root: Path,
    *,
    range_bins: int,
    angle_bins: int,
    doppler_bins: int,
    depth_range: tuple[float, float],
    azimuth_range: tuple[float, float],
    velocity_range: tuple[float, float],
    overwrite: bool,
    progress: bool,
    materialize_split_columns: bool,
) -> dict[str, Any]:
    scene = prepared_root.name
    manifest_path = prepared_root / "manifests" / "frame_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MMW frame manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, na_values="").fillna("")
    radar_by_frame = _radar_paths_from_manifest(data_root, scene, manifest)
    output_dir = prepared_root / "derived" / "radar_maps" / "rsu_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    items = sorted(radar_by_frame.items())
    iterator = tqdm(items, desc=f"MMW radar maps {scene}", unit="frame") if progress else items
    for frame_id, radar_path in iterator:
        ra_path = output_dir / f"{frame_id}_RA.npy"
        da_path = output_dir / f"{frame_id}_DA.npy"
        if ra_path.exists() and da_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            detections = _load_radar_detections(radar_path)
            ra, da = _detections_to_maps(
                detections,
                range_bins=range_bins,
                angle_bins=angle_bins,
                doppler_bins=doppler_bins,
                depth_range=depth_range,
                azimuth_range=azimuth_range,
                velocity_range=velocity_range,
            )
            atomic_save_npy(ra_path, ra)
            atomic_save_npy(da_path, da)
            generated += 1
        except Exception as exc:  # noqa: BLE001 - report per-frame preprocessing failures.
            failed += 1
            if len(failures) < 20:
                failures.append({"frame_id": str(frame_id), "path": str(radar_path), "reason": str(exc)})
    split_reports = []
    if materialize_split_columns:
        split_reports = _materialize_split_columns(prepared_root, scene)
    report = {
        "scene": scene,
        "manifest_path": str(manifest_path),
        "radar_map_dir": str(output_dir),
        "unique_radar_frames": len(radar_by_frame),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "parameters": {
            "range_bins": range_bins,
            "angle_bins": angle_bins,
            "doppler_bins": doppler_bins,
            "depth_range": list(depth_range),
            "azimuth_range": list(azimuth_range),
            "velocity_range": list(velocity_range),
        },
        "split_columns": split_reports,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (prepared_root / "radar_maps_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _resolve_prepared_roots(
    data_root: Path,
    *,
    scenes: list[str] | tuple[str, ...] | None,
    prepared_roots: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if prepared_roots:
        return [resolve_path(path) for path in prepared_roots]
    prepared_dir = data_root / "Prepared"
    if scenes:
        return [prepared_dir / str(scene) for scene in scenes]
    return sorted(path for path in prepared_dir.iterdir() if path.is_dir())


def _radar_paths_from_manifest(data_root: Path, scene: str, manifest: pd.DataFrame) -> dict[str, Path]:
    radar_by_frame: dict[str, Path] = {}
    for _, row in manifest.iterrows():
        frame_id = _frame_id_text(row.get("frame_id", ""))
        if not frame_id:
            continue
        radar_path = _radar_path_from_row(data_root, scene, row)
        if radar_path is None:
            continue
        radar_by_frame.setdefault(frame_id, radar_path)
    return radar_by_frame


def _radar_path_from_row(data_root: Path, scene: str, row: pd.Series) -> Path | None:
    direct = str(row.get("radar", "")).strip()
    if direct and direct != "-99":
        return data_root / direct
    rsu_json = str(row.get("rsu_json", "")).strip()
    if rsu_json:
        try:
            payload = json.loads(rsu_json)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            agents = payload.get("agents")
            if isinstance(agents, dict):
                rsu_1 = agents.get("rsu_1")
                if isinstance(rsu_1, dict):
                    radar = str(rsu_1.get("radar", "")).strip()
                    if radar:
                        return data_root / radar
    frame_id = _frame_id_text(row.get("frame_id", ""))
    if frame_id:
        return data_root / "Sensor_Data" / scene / "rsu_1" / f"{frame_id}.json"
    return None


def _load_radar_detections(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"radar JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"radar JSON must contain a list of detections: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _detections_to_maps(
    detections: list[dict[str, Any]],
    *,
    range_bins: int,
    angle_bins: int,
    doppler_bins: int,
    depth_range: tuple[float, float],
    azimuth_range: tuple[float, float],
    velocity_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    ra = np.zeros((range_bins, angle_bins), dtype=np.float32)
    da = np.zeros((doppler_bins, angle_bins), dtype=np.float32)
    if not detections:
        return ra, da
    depth = _numeric_detection_array(detections, "depth")
    azimuth = _numeric_detection_array(detections, "azimuth")
    velocity = _numeric_detection_array(detections, "velocity")
    valid = np.isfinite(depth) & np.isfinite(azimuth) & np.isfinite(velocity)
    if not np.any(valid):
        return ra, da
    depth = depth[valid]
    azimuth = azimuth[valid]
    velocity = velocity[valid]
    angle_idx = _bin_indices(azimuth, azimuth_range, angle_bins)
    range_idx = _bin_indices(depth, depth_range, range_bins)
    doppler_idx = _bin_indices(velocity, velocity_range, doppler_bins)
    np.add.at(ra, (range_idx, angle_idx), 1.0)
    np.add.at(da, (doppler_idx, angle_idx), 1.0)
    return _log_normalize(ra), _log_normalize(da)


def _numeric_detection_array(detections: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for item in detections:
        try:
            values.append(float(item.get(key, np.nan)))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)


def _bin_indices(values: np.ndarray, value_range: tuple[float, float], bins: int) -> np.ndarray:
    low, high = value_range
    span = max(float(high) - float(low), 1e-6)
    clipped = np.clip(values, low, high)
    indices = np.floor((clipped - low) / span * int(bins)).astype(np.int64)
    return np.clip(indices, 0, int(bins) - 1)


def _log_normalize(values: np.ndarray) -> np.ndarray:
    mapped = np.log1p(values.astype(np.float32))
    maximum = float(mapped.max())
    if maximum > 0.0:
        mapped = mapped / maximum
    return mapped.astype(np.float32)


def _materialize_split_columns(prepared_root: Path, scene: str) -> list[dict[str, Any]]:
    reports = []
    split_dir = prepared_root / "splits"
    for split in ("train", "test", "val"):
        source = split_dir / f"{split}.csv"
        if not source.exists():
            continue
        frame = pd.read_csv(source, na_values="").fillna("")
        reports.append(
            materialize_mmw_radar_split_csv(
                prepared_root.parents[1],
                source,
                scene,
                require_maps=False,
            )
        )
        reports.append(_write_split_with_columns(frame, source, scene, include_radar=False, include_bs_gps=True))
        reports.append(_write_split_with_columns(frame, source, scene, include_radar=True, include_bs_gps=True))
    return reports


def _write_split_with_columns(
    frame: pd.DataFrame,
    source: Path,
    scene: str,
    *,
    include_radar: bool,
    include_bs_gps: bool,
) -> dict[str, Any]:
    output = frame.copy()
    suffix_parts = []
    if include_radar:
        suffix_parts.append("radar")
        for beam_col in _numbered_columns(output.columns, "beam"):
            suffix = beam_col[len("beam") :]
            output[f"radar{suffix}"] = [_radar_rel_path_for_beam(value, scene) for value in output[beam_col].tolist()]
    if include_bs_gps:
        suffix_parts.append("bs_gps")
        for gps_col in _numbered_columns(output.columns, "gps"):
            suffix = gps_col[len("gps") :]
            output[f"bs_gps{suffix}"] = [_rsu_gps_rel_path_for_gps(value, scene) for value in output[gps_col].tolist()]
    target = source.with_name(f"{source.stem}_with_{'_with_'.join(suffix_parts)}{source.suffix}")
    output.to_csv(target, index=False)
    return {
        "source": str(source),
        "path": str(target),
        "rows": int(len(output)),
        "radar_columns": len(_numbered_columns(output.columns, "radar")),
        "bs_gps_columns": len(_numbered_columns(output.columns, "bs_gps")),
    }


def _numbered_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        text = str(col)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            selected.append(text)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


def _frame_id_text(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        numeric = int(float(text))
    except ValueError:
        return text
    return f"{numeric:06d}"


def _radar_rel_path_for_beam(value: object, scene: str) -> str:
    frame_id = Path(str(value).replace("\\", "/")).stem
    if not frame_id:
        return "-99"
    return (Path("Prepared") / scene / "derived" / "radar_maps" / "rsu_1" / f"{frame_id}_RA.npy").as_posix()


def _rsu_gps_rel_path_for_gps(value: object, scene: str) -> str:
    frame_id = Path(str(value).replace("\\", "/")).stem
    if not frame_id:
        return "-99"
    return (Path("Sensor_Data") / scene / "rsu_1" / f"{frame_id}.yaml").as_posix()


def _load_split_metadata_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"path": str(path), "status": "invalid_json"}
    return {
        key: payload.get(key)
        for key in (
            "seed",
            "split_seed",
            "train_ratio",
            "split_tag",
            "seq_len",
            "num_pred",
            "pred_len",
            "train_window_count",
            "test_window_count",
        )
        if key in payload
    } | {"path": str(path)}


@PREPROCESSORS.register("mmw_radar_maps")
class MMWRadarMapsPreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return generate_mmw_radar_maps(**self.kwargs)


__all__ = [
    "MMWRadarMapsPreprocessor",
    "generate_mmw_radar_maps",
    "materialize_mmw_radar_split_csv",
]
