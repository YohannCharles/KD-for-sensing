from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.data.beam_label_space import label_space_metadata, validate_label_space_rows
from kd_sensing.data.deepsense6g_gps_lidar_bgam_manifest import (
    PSEUDO_HISTORY_SUMMARY_NAME,
    attach_pseudo_history_fields,
    enrich_gps_lidar_bgam_row,
)
from kd_sensing.data.mmw_town_topk_candidate_manifest import (
    MANIFEST_NAME as TOPK_MANIFEST_NAME,
    build_mmw_town_topk_candidate_manifest,
)
from kd_sensing.data.transform_ops.lidar import lidar_bev_grid_metadata
from kd_sensing.utils.geometry import load_beam_angle_table


MANIFEST_NAME = "gps_lidar_bgam_manifest.csv"
METADATA_NAME = "gps_lidar_bgam_manifest_metadata.json"


def build_mmw_town_gps_lidar_bgam_manifest(
    cfg: Mapping[str, Any],
    *,
    label_space: str | None = None,
    topk: int | None = None,
    output_dir: str | Path | None = None,
    input_manifest: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    candidate_cfg = _mapping(cfg.get("candidate"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    geometry_cfg = _mapping(cfg.get("geometry"))
    lidar_cfg = _mapping(cfg.get("lidar"))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_enabled"))
    selected_topk = int(topk if topk is not None else candidate_cfg.get("topk", data_cfg.get("topk", 8)))
    num_beams = int(data_cfg.get("num_beams", candidate_cfg.get("num_beams", 64)))
    scenes = _scene_slugs(data_cfg)
    label_meta_by_scene = {
        scene: label_space_metadata(data_cfg, selected_label_space, num_beams=num_beams, scene=scene)
        for scene in scenes
    }
    result_dir = _result_dir(outputs_cfg, output_dir, selected_label_space)
    manifest_dir = result_dir / str(outputs_cfg.get("manifest_dir", "manifest"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / str(outputs_cfg.get("manifest_name", MANIFEST_NAME))
    metadata_path = manifest_dir / str(outputs_cfg.get("metadata_name", METADATA_NAME))
    pseudo_history_summary_path = manifest_dir / PSEUDO_HISTORY_SUMMARY_NAME

    source_manifest = _resolve_source_manifest(
        cfg,
        label_space=selected_label_space,
        topk=selected_topk,
        output_dir=output_dir,
        input_manifest=input_manifest,
    )
    rows = _read_csv(source_manifest) if source_manifest is not None else []
    _validate_rows_by_scene(
        rows,
        expected_by_scene=label_meta_by_scene,
        source_path=source_manifest or "<missing_mmw_top8_manifest>",
        artifact_name="MMW Top8 candidate manifest",
        strict=selected_label_space != "mapping_disabled",
    )
    beam_table = load_beam_angle_table(geometry_cfg, num_beams=num_beams)
    bev_metadata = lidar_bev_grid_metadata(
        bev_size=tuple(lidar_cfg.get("bev_size", (224, 224))),
        roi=tuple(lidar_cfg.get("roi", (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0))),
        fov_degrees=lidar_cfg.get("fov_degrees"),
        remove_ground=bool(lidar_cfg.get("remove_ground", False)),
        ground_z_threshold=float(lidar_cfg.get("ground_z_threshold", 0.1)),
        background_path=lidar_cfg.get("background_path"),
        background_distance_threshold=float(lidar_cfg.get("background_distance_threshold", 0.2)),
        cell_center_convention=str(lidar_cfg.get("cell_center_convention", "center")),
        cache_version=str(lidar_cfg.get("cache_version", "bgam_bev_v1")),
    )
    enriched: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_fields: dict[str, int] = {}
    for row in rows:
        try:
            item = enrich_gps_lidar_bgam_row(
                row,
                cfg,
                topk=selected_topk,
                num_beams=num_beams,
                beam_angle_source=beam_table.beam_angle_source,
                beam_angle_convention=str(beam_table.metadata.get("beam_angle_convention", "")),
                bev_metadata=bev_metadata,
            )
            item["dataset_family"] = "MMW"
            item["mmw_condition"] = str(item.get("condition") or data_cfg.get("condition", "sunny"))
            item["mmw_prediction_horizon"] = int(_mapping(cfg.get("history")).get("prediction_horizon", data_cfg.get("prediction_horizon", 3)) or 3)
            enriched.append(item)
        except ValueError as exc:
            warnings.append(f"{row.get('sample_id', '<unknown>')}: {exc}")
            for field in ("lidar_path", "lidar_bev_cache_path", "theta_gps"):
                if field in str(exc):
                    missing_fields[field] = missing_fields.get(field, 0) + 1

    history_payload = attach_pseudo_history_fields(
        enriched,
        cfg,
        num_beams=num_beams,
        label_metadata={"label_space": selected_label_space},
    )
    _write_csv(pseudo_history_summary_path, history_payload["summary_rows"])
    _write_csv(manifest_path, enriched, fieldnames=_manifest_fieldnames(selected_topk))
    lidar_available_count = sum(1 for row in enriched if _bool(row.get("lidar_available")))
    metadata = {
        "workflow": "mmw_town_gps_lidar_bgam_manifest",
        "dataset_family": "MMW",
        "input_manifest": str(source_manifest or ""),
        "gps_v2_artifact": str(data_cfg.get("gps_v2_artifact_root", data_cfg.get("gps_output_root", ""))),
        "top8_candidate_source": str(data_cfg.get("top8_manifest_path", "")),
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "label_space": selected_label_space,
        "mapping_by_scene": label_meta_by_scene,
        "topk": selected_topk,
        "num_beams": num_beams,
        "row_count": len(enriched),
        "support_count": sum(1 for row in enriched if _is_support_role(row.get("support_query_role") or row.get("split_role"))),
        "query_count": sum(1 for row in enriched if _is_query_role(row.get("support_query_role") or row.get("split_role"))),
        "lidar_availability": {
            "available_count": lidar_available_count,
            "missing_count": len(enriched) - lidar_available_count,
            "missing_fields": missing_fields,
            "preferred_source": str(lidar_cfg.get("source", "rsu")),
        },
        "beam_angle": beam_table.metadata,
        "bev_grid_metadata": bev_metadata,
        "pseudo_history": history_payload["metadata"],
        "pseudo_history_summary_path": str(pseudo_history_summary_path),
        "query_label_used_for_training": False,
        "warnings": warnings,
    }
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "pseudo_history_summary_path": str(pseudo_history_summary_path),
        "row_count": len(enriched),
        "support_count": metadata["support_count"],
        "query_count": metadata["query_count"],
        "lidar_available_count": lidar_available_count,
        "warnings": warnings,
    }


def _resolve_source_manifest(
    cfg: Mapping[str, Any],
    *,
    label_space: str,
    topk: int,
    output_dir: str | Path | None,
    input_manifest: str | Path | None,
) -> Path | None:
    data_cfg = _mapping(cfg.get("data"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    if input_manifest:
        candidate = Path(input_manifest)
        return candidate if candidate.exists() else None
    configured = str(data_cfg.get("top8_manifest_path") or "").strip()
    if configured and Path(configured).exists():
        return Path(configured)
    topk_root = Path(str(data_cfg.get("topk_candidate_source") or outputs_cfg.get("topk_root") or "outputs/analysis/mmw_town_top8_selector"))
    candidate = topk_root / label_space / str(outputs_cfg.get("manifest_dir", "manifest")) / TOPK_MANIFEST_NAME
    if candidate.exists():
        return candidate
    try:
        result = build_mmw_town_topk_candidate_manifest(
            cfg,
            label_space=label_space,
            topk=topk,
            output_dir=topk_root,
        )
        candidate = Path(str(result.get("manifest_path", "")))
        return candidate if candidate.exists() else None
    except Exception:
        return None


def _result_dir(outputs_cfg: Mapping[str, Any], output_dir: str | Path | None, label_space: str) -> Path:
    root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/mmw_town_gps_lidar_bgam"))
    if root.name == str(label_space):
        return root
    return root / str(label_space)


def _scene_slugs(data_cfg: Mapping[str, Any]) -> list[str]:
    values = []
    for item in data_cfg.get("scenes") or []:
        if isinstance(item, Mapping):
            values.append(str(item.get("slug") or item.get("scene") or item.get("name") or ""))
        else:
            values.append(str(item))
    return [value for value in values if value]


def _validate_rows_by_scene(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_by_scene: Mapping[str, Mapping[str, Any]],
    source_path: str | Path,
    artifact_name: str,
    strict: bool,
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scene = str(row.get("scene") or row.get("target_scene") or "")
        grouped.setdefault(scene, []).append(row)
    for scene, scene_rows in grouped.items():
        expected = expected_by_scene.get(scene)
        if expected is None:
            continue
        validate_label_space_rows(
            scene_rows,
            expected=expected,
            source_path=source_path,
            artifact_name=f"{artifact_name} scene={scene}",
            require_fields=strict,
        )


def _manifest_fieldnames(topk: int) -> list[str]:
    base = [
        "dataset_family",
        "mmw_condition",
        "mmw_prediction_horizon",
        "scene",
        "scene_name",
        "scene_id",
        "agent",
        "sample_id",
        "current_sample_id",
        "timestamp",
        "frame_id",
        "split",
        "support_query_role",
        "split_role",
        "target_label",
        "gt_beam",
        "target_label_raw",
        "gps_top1",
        "gps_top1_prob",
        "gps_top2_prob",
        "gps_top1_top2_margin",
        "gps_entropy",
        "gps_normalized_gain",
        "theta_gps",
        "theta_degrees",
        "distance_to_rsu",
        "gps_dx",
        "gps_dy",
        "coordinate_frame",
        "rsu_x",
        "rsu_y",
        "rsu_yaw",
        "rsu_yaw_source",
        "lidar_path",
        "lidar_source",
        "lidar_bev_cache_path",
        "lidar_available",
        "lidar_missing_reason",
        "beam_power_path",
        "target_in_top8",
        "target_candidate_index",
        "nearest_candidate_index",
        "nearest_candidate_error",
        "top8_oracle_error",
        "top8_oracle_beam",
        "top8_oracle_normalized_gain",
        "beam_angle_source",
        "beam_angle_convention",
        "bev_grid_metadata_json",
        "query_label_used_for_training",
        "label_space",
        "beam_label_space",
        "beam_label_mapping_fingerprint",
        "raw_to_mapped_mapping_source",
        "history_pseudo_beams",
        "history_pseudo_probs",
        "history_pseudo_entropy",
        "history_pseudo_confidence",
        "history_valid_mask",
        "history_timestamps",
        "history_source_row_indices",
        "history_pseudo_label_source",
        "history_alignment_policy",
        "history_len",
        "history_missing_count",
        "history_sensor_period",
        "history_beam_period",
        "history_replication_ratio",
        "query_label_used_for_pseudo_history",
        "history_label_space",
        "history_beam_label_space",
        "history_beam_label_mapping_fingerprint",
    ]
    for idx in range(int(topk)):
        base.extend(
            [
                f"cand{idx}_beam",
                f"cand{idx}_logit",
                f"cand{idx}_prob",
                f"cand{idx}_rank",
                f"cand{idx}_dist_to_gps_top1",
                f"cand{idx}_normalized_gain",
            ]
        )
    return base


def _is_support_role(value: Any) -> bool:
    return str(value or "").startswith("support")


def _is_query_role(value: Any) -> bool:
    return str(value or "").startswith("query")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return []
    with candidate.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys([*(fieldnames or ()), *_fieldnames(rows)]))
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in names})


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    return names


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PSEUDO_HISTORY_SUMMARY_NAME",
    "build_mmw_town_gps_lidar_bgam_manifest",
]
