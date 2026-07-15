from contextlib import ExitStack
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid
import warnings

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
    max_failure_rate: float = 0.0,
) -> dict[str, Any]:
    root = resolve_path(data_root)
    prepared = _resolve_prepared_roots(root, scenes=scenes, prepared_roots=prepared_roots)
    if not prepared:
        raise RuntimeError(f"MMW radar preprocessing found no prepared scene roots under {root / 'Prepared'}.")
    reports = []
    publish_pairs: list[tuple[Path, Path]] = []
    total_frames = 0
    total_generated = 0
    total_skipped = 0
    total_failed = 0
    with ExitStack() as stack:
        for prepared_root in prepared:
            report, scene_pairs = _stage_scene_radar_maps(
                stack,
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
                max_failure_rate=float(max_failure_rate),
            )
            publish_pairs.extend(scene_pairs)
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
        report_stage = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix=f".{report_path.name}.stage-", dir=report_path.parent)
            )
        ) / report_path.name
        report_stage.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        publish_pairs.append((report_stage, report_path))
        _publish_paths(publish_pairs)
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
    scene = _validated_scene_name(scene)
    source = Path(csv_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"MMW split CSV must be under data_root {root}: {source}")
    if not source.exists():
        raise FileNotFoundError(
            f"MMW split CSV is missing: {source}. Prepare sequence splits with "
            "conda run -n kd_mm_beam kd-sensing-preprocess --action mmw_sequence_splits_from_manifest."
        )
    target = Path(output_path) if output_path is not None else source.with_name(f"{source.stem}_with_radar{source.suffix}")
    if not target.is_absolute():
        target = root / target
    if target.is_symlink():
        raise ValueError(f"MMW radar output CSV must not be a symbolic link: {target}")
    target = target.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"MMW radar output CSV must be under data_root {root}: {target}")
    if output_path is not None:
        _ensure_disjoint(source, target, "source CSV", "output CSV")
    frame = pd.read_csv(source, na_values="").fillna("")
    radar_cols = _numbered_columns(frame.columns, "radar")
    beam_cols = _numbered_columns(frame.columns, "beam")
    write_output = False
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
            output = _materialize_radar_columns(
                root,
                frame,
                beam_cols,
                scene,
                source=source,
                target=target,
                require_maps=require_maps,
            )
            write_output = True
            created = True
            radar_cols = _numbered_columns(output.columns, "radar")
    metadata_path = target.with_name(f"{target.stem}_metadata.json")
    split_metadata_path = source.parent / "split_metadata.json"
    metadata = _radar_split_metadata(
        root,
        source,
        target,
        metadata_path,
        split_metadata_path,
        output,
        scene,
        created=created,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.stage-", dir=target.parent) as temporary:
        stage_root = Path(temporary)
        pairs: list[tuple[Path, Path]] = []
        if write_output:
            staged_target = stage_root / target.name
            output.to_csv(staged_target, index=False)
            pairs.append((staged_target, target))
        staged_metadata = stage_root / metadata_path.name
        staged_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pairs.append((staged_metadata, metadata_path))
        _publish_paths(pairs)
    return {
        "path": str(target),
        "metadata_path": str(metadata_path),
        "created": created,
        "metadata": metadata,
    }


def _radar_split_metadata(
    root: Path,
    source: Path,
    target: Path,
    metadata_path: Path,
    split_metadata_path: Path,
    output: pd.DataFrame,
    scene: str,
    *,
    created: bool,
) -> dict[str, Any]:
    return {
        "type": "mmw_radar_split_csv_materialization",
        "public_utility": "kd_sensing.preprocessing.mmw_radar.materialize_mmw_radar_split_csv",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(root),
        "condition": root.name,
        "scene": scene,
        "scenario": scene,
        "input_manifest": str(root / "Prepared" / scene / "manifests" / "frame_manifest.csv"),
        "source_csv": str(source),
        "output_csv": str(target),
        "metadata_path": str(metadata_path),
        "split_metadata_path": str(split_metadata_path) if split_metadata_path.exists() else None,
        "split_config": _load_split_metadata_summary(split_metadata_path),
        "seq_len": len(_numbered_columns(output.columns, "beam")),
        "num_pred": len(_numbered_columns(output.columns, "future_beam")),
        "sample_count": int(len(output)),
        "radar_columns": len(_numbered_columns(output.columns, "radar")),
        "created_output": created,
        "repair_command": f"{MMW_RADAR_PREPARATION_COMMAND} # ensure preprocessing.scenes contains {scene}",
    }


def _materialize_radar_columns(
    root: Path,
    frame: pd.DataFrame,
    beam_cols: list[str],
    scene: str,
    *,
    source: Path,
    target: Path,
    require_maps: bool,
) -> pd.DataFrame:
    output = frame.copy()
    missing = []
    for beam_col in beam_cols:
        suffix = beam_col[len("beam") :]
        values = []
        for value in output[beam_col].tolist():
            rel_path = _radar_rel_path_for_beam(value, scene)
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
    return output


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


def _stage_scene_radar_maps(
    stack: ExitStack,
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
    max_failure_rate: float,
) -> tuple[dict[str, Any], list[tuple[Path, Path]]]:
    scene = prepared_root.name
    manifest_path = prepared_root / "manifests" / "frame_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MMW frame manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, na_values="").fillna("")
    radar_by_frame = _radar_paths_from_manifest(data_root, scene, manifest)
    output_dir = prepared_root / "derived" / "radar_maps" / "rsu_1"
    if output_dir.is_symlink():
        raise ValueError(f"MMW radar output directory must not be a symbolic link: {output_dir}")
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(prepared_root):
        raise ValueError(f"MMW radar output directory escapes prepared root {prepared_root}: {output_dir}")
    for radar_path in radar_by_frame.values():
        _ensure_disjoint(radar_path, output_dir, "radar input", "radar map output directory")
    stage_root = Path(
        stack.enter_context(
            tempfile.TemporaryDirectory(prefix=f".{prepared_root.name}.radar-stage-", dir=prepared_root.parent)
        )
    )
    staged_output_dir = stage_root / "radar_maps"
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"MMW radar output must be a directory: {output_dir}")
        shutil.copytree(output_dir, staged_output_dir, symlinks=True)
    else:
        staged_output_dir.mkdir()
    generated = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    items = sorted(radar_by_frame.items())
    iterator = tqdm(items, desc=f"MMW radar maps {scene}", unit="frame") if progress else items
    for frame_id, radar_path in iterator:
        ra_path = staged_output_dir / f"{frame_id}_RA.npy"
        da_path = staged_output_dir / f"{frame_id}_DA.npy"
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
    _validate_batch_outcome(
        f"MMW radar preprocessing for scene {scene}",
        attempted=len(items),
        succeeded=generated + skipped,
        failed=failed,
        failures=failures,
        max_failure_rate=max_failure_rate,
    )
    split_reports = []
    publish_pairs: list[tuple[Path, Path]] = [(staged_output_dir, output_dir)]
    if materialize_split_columns:
        split_reports, split_pairs = _stage_split_columns(data_root, prepared_root, scene, stage_root / "files")
        publish_pairs.extend(split_pairs)
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
    report_path = prepared_root / "radar_maps_report.json"
    staged_report = _staged_file_path(stage_root / "files", report_path, len(publish_pairs))
    staged_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    publish_pairs.append((staged_report, report_path))
    return report, publish_pairs


def _resolve_prepared_roots(
    data_root: Path,
    *,
    scenes: list[str] | tuple[str, ...] | None,
    prepared_roots: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    prepared_dir = (data_root / "Prepared").resolve()
    if prepared_roots:
        candidates = [resolve_path(path) for path in prepared_roots]
    elif scenes:
        candidates = [prepared_dir / str(scene) for scene in scenes]
    else:
        if not prepared_dir.exists():
            return []
        candidates = sorted(path for path in prepared_dir.iterdir() if path.is_dir())
    resolved: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"MMW prepared root must not be a symbolic link: {candidate}")
        path = candidate.resolve()
        if path.parent != prepared_dir:
            raise ValueError(f"MMW prepared root must be a direct child of {prepared_dir}: {path}")
        resolved.append(path)
    return sorted(set(resolved))


def _radar_paths_from_manifest(data_root: Path, scene: str, manifest: pd.DataFrame) -> dict[str, Path]:
    radar_by_frame: dict[str, Path] = {}
    for _, row in manifest.iterrows():
        frame_id = _frame_id_text(row.get("frame_id", ""))
        if not frame_id:
            continue
        radar_path = _radar_path_from_row(data_root, scene, row)
        if radar_path is None:
            continue
        radar_path = radar_path.resolve()
        if not radar_path.is_relative_to(data_root.resolve()):
            raise ValueError(f"MMW radar input escapes data_root: {radar_path}")
        previous = radar_by_frame.get(frame_id)
        if previous is not None and previous.resolve() != radar_path.resolve():
            raise ValueError(
                "MMW radar resource identity collision: "
                f"scene={scene}, frame_id={frame_id}, paths={[str(previous), str(radar_path)]}"
            )
        radar_by_frame[frame_id] = radar_path
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


def _stage_split_columns(
    data_root: Path,
    prepared_root: Path,
    scene: str,
    stage_root: Path,
) -> tuple[list[dict[str, Any]], list[tuple[Path, Path]]]:
    reports: list[dict[str, Any]] = []
    pairs: list[tuple[Path, Path]] = []
    split_dir = prepared_root / "splits"
    sources = sorted(
        path
        for split in ("train", "test", "val")
        for path in split_dir.rglob(f"{split}.csv")
    )
    for source in sources:
        source = source.resolve()
        if not source.is_relative_to(prepared_root):
            raise ValueError(f"MMW split CSV escapes prepared root {prepared_root}: {source}")
        frame = pd.read_csv(source, na_values="").fillna("")
        beam_cols = _numbered_columns(frame.columns, "beam")
        radar_cols = _numbered_columns(frame.columns, "radar")
        target = source if radar_cols else source.with_name(f"{source.stem}_with_radar{source.suffix}")
        existing = frame if target == source else _valid_materialized_radar_csv(target)
        if existing is not None:
            output = existing
            created = False
        else:
            if not beam_cols:
                raise ValueError(f"Cannot materialize radar columns for {source}; no beamN columns were found.")
            output = _materialize_radar_columns(
                data_root,
                frame,
                beam_cols,
                scene,
                source=source,
                target=target,
                require_maps=False,
            )
            staged_target = _staged_file_path(stage_root, target, len(pairs))
            output.to_csv(staged_target, index=False)
            pairs.append((staged_target, target))
            created = True
        metadata_path = target.with_name(f"{target.stem}_metadata.json")
        split_metadata_path = source.parent / "split_metadata.json"
        metadata = _radar_split_metadata(
            data_root,
            source,
            target,
            metadata_path,
            split_metadata_path,
            output,
            scene,
            created=created,
        )
        staged_metadata = _staged_file_path(stage_root, metadata_path, len(pairs))
        staged_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pairs.append((staged_metadata, metadata_path))
        reports.append(
            {
                "path": str(target),
                "metadata_path": str(metadata_path),
                "created": created,
                "metadata": metadata,
            }
        )
        for include_radar, include_bs_gps in ((False, True), (True, True)):
            report, pair = _stage_split_with_columns(
                frame,
                source,
                scene,
                stage_root,
                len(pairs),
                include_radar=include_radar,
                include_bs_gps=include_bs_gps,
            )
            reports.append(report)
            pairs.append(pair)
    return reports, pairs


def _stage_split_with_columns(
    frame: pd.DataFrame,
    source: Path,
    scene: str,
    stage_root: Path,
    index: int,
    *,
    include_radar: bool,
    include_bs_gps: bool,
) -> tuple[dict[str, Any], tuple[Path, Path]]:
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
    staged_target = _staged_file_path(stage_root, target, index)
    output.to_csv(staged_target, index=False)
    report = {
        "source": str(source),
        "path": str(target),
        "rows": int(len(output)),
        "radar_columns": len(_numbered_columns(output.columns, "radar")),
        "bs_gps_columns": len(_numbered_columns(output.columns, "bs_gps")),
    }
    return report, (staged_target, target)


def _staged_file_path(stage_root: Path, target: Path, index: int) -> Path:
    stage_root.mkdir(parents=True, exist_ok=True)
    return stage_root / f"{index:04d}_{target.name}"


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


def _validated_scene_name(value: object) -> str:
    scene = str(value).strip()
    if not scene or scene in {".", ".."} or "/" in scene or "\\" in scene:
        raise ValueError(f"MMW scene must be a single directory name: {value!r}")
    return scene


def _ensure_disjoint(left: Path, right: Path, left_name: str, right_name: str) -> None:
    left = left.resolve()
    right = right.resolve()
    if left == right or left.is_relative_to(right) or right.is_relative_to(left):
        raise ValueError(f"{left_name} and {right_name} must be disjoint: {left} vs {right}")


def _publish_paths(pairs: list[tuple[Path, Path]]) -> None:
    for _, target in pairs:
        if target.is_symlink():
            raise ValueError(f"MMW radar output must not be a symbolic link: {target}")
    targets = [target.resolve() for _, target in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError("MMW radar publish plan contains duplicate output paths.")
    for index, target in enumerate(targets):
        for other in targets[index + 1 :]:
            if target.is_relative_to(other) or other.is_relative_to(target):
                raise ValueError(f"MMW radar publish outputs must be disjoint: {target} vs {other}")

    token = uuid.uuid4().hex
    published: list[tuple[Path, Path | None]] = []
    try:
        for staged, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if target.exists():
                backup = target.with_name(f".{target.name}.{token}.backup")
                os.replace(target, backup)
            try:
                os.replace(staged, target)
            except Exception:
                if backup is not None:
                    os.replace(backup, target)
                raise
            published.append((target, backup))
    except Exception:
        for target, backup in reversed(published):
            _remove_path(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    for _, backup in published:
        if backup is None:
            continue
        try:
            _remove_path(backup)
        except OSError as exc:
            warnings.warn(f"Could not remove successful MMW radar publish backup {backup}: {exc}", RuntimeWarning)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _validate_batch_outcome(
    name: str,
    *,
    attempted: int,
    succeeded: int,
    failed: int,
    failures: list[dict[str, str]],
    max_failure_rate: float,
) -> None:
    threshold = float(max_failure_rate)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("max_failure_rate must be between 0 and 1.")
    detail = f"attempted={attempted}, succeeded={succeeded}, failed={failed}, examples={failures[:20]}"
    if attempted <= 0 or succeeded <= 0:
        raise RuntimeError(f"{name} produced zero successful resources; {detail}")
    if failed / attempted > threshold:
        raise RuntimeError(f"{name} exceeded max_failure_rate={threshold}; {detail}")
    if failed:
        warnings.warn(f"{name} completed with allowed failures; {detail}", RuntimeWarning, stacklevel=2)


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
