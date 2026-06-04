from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.data.deepsense6g_topk_candidate_manifest import (
    MANIFEST_NAME as TOP8_MANIFEST_NAME,
    build_topk_candidate_manifest,
    circular_distance,
    ratio_tag,
)
from kd_sensing.data.beam_label_space import (
    attach_label_space_metadata,
    label_space_metadata,
    validate_label_space_rows,
)
from kd_sensing.data.transform_ops.lidar import lidar_bev_grid_metadata
from kd_sensing.utils.geometry import gps_to_rsu_aod, load_beam_angle_table


MANIFEST_NAME = "gps_lidar_bgam_manifest.csv"
METADATA_NAME = "gps_lidar_bgam_manifest_metadata.json"
PSEUDO_HISTORY_SUMMARY_NAME = "pseudo_history_summary.csv"


def build_gps_lidar_bgam_manifest(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
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
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    selected_topk = int(topk if topk is not None else candidate_cfg.get("topk", data_cfg.get("topk", 8)))
    num_beams = int(data_cfg.get("num_beams", candidate_cfg.get("num_beams", 64)))
    label_meta = label_space_metadata(data_cfg, selected_label_space, num_beams=num_beams)
    tag = ratio_tag(ratio)
    out_root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/deepsense6g_gps_lidar_bgam"))
    result_dir = out_root / tag / selected_label_space
    manifest_dir = result_dir / str(outputs_cfg.get("manifest_dir", "manifest"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / str(outputs_cfg.get("manifest_name", MANIFEST_NAME))
    metadata_path = manifest_dir / str(outputs_cfg.get("metadata_name", METADATA_NAME))

    source_manifest = _resolve_source_manifest(
        cfg,
        ratio=ratio,
        label_space=selected_label_space,
        topk=selected_topk,
        output_root=out_root,
        input_manifest=input_manifest,
    )
    rows = _read_csv(source_manifest) if source_manifest is not None else []
    validate_label_space_rows(
        rows,
        expected=label_meta,
        source_path=source_manifest or "<missing_top8_manifest>",
        artifact_name="Top8 candidate manifest",
        require_fields=selected_label_space != "mapping_disabled",
    )
    beam_table = load_beam_angle_table(geometry_cfg, num_beams=num_beams)
    bev_metadata = lidar_bev_grid_metadata(
        bev_size=tuple(lidar_cfg.get("bev_size", (64, 64))),
        roi=tuple(lidar_cfg.get("roi", (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0))),
        fov_degrees=lidar_cfg.get("fov_degrees"),
        remove_ground=bool(lidar_cfg.get("remove_ground", False)),
        ground_z_threshold=float(lidar_cfg.get("ground_z_threshold", 0.1)),
        cell_center_convention=str(lidar_cfg.get("cell_center_convention", "center")),
        cache_version=str(lidar_cfg.get("cache_version", "bgam_bev_v1")),
    )
    enriched: list[dict[str, Any]] = []
    missing_fields: dict[str, int] = {}
    warnings: list[str] = []
    for row in rows:
        try:
            enriched_row = enrich_gps_lidar_bgam_row(
                row,
                cfg,
                topk=selected_topk,
                num_beams=num_beams,
                beam_angle_source=beam_table.beam_angle_source,
                beam_angle_convention=str(beam_table.metadata.get("beam_angle_convention", "")),
                bev_metadata=bev_metadata,
            )
            enriched.append(attach_label_space_metadata(enriched_row, label_meta))
        except ValueError as exc:
            warnings.append(f"{row.get('sample_id', '<unknown>')}: {exc}")
            for field in _missing_field_names(str(exc)):
                missing_fields[field] = missing_fields.get(field, 0) + 1

    history_payload = attach_pseudo_history_fields(enriched, cfg, num_beams=num_beams, label_metadata=label_meta)
    pseudo_history_summary_path = manifest_dir / PSEUDO_HISTORY_SUMMARY_NAME
    _write_csv(pseudo_history_summary_path, history_payload["summary_rows"], fieldnames=_pseudo_history_summary_fieldnames())
    _write_csv(manifest_path, enriched, fieldnames=_manifest_fieldnames(selected_topk))
    lidar_available_count = sum(1 for row in enriched if _bool(row.get("lidar_available")))
    metadata = {
        "workflow": "deepsense6g_gps_lidar_bgam_manifest",
        "input_manifest": str(source_manifest or ""),
        "gps_v2_artifact": str(data_cfg.get("gps_v2_artifact_root", "")),
        "top8_candidate_source": str(data_cfg.get("topk_candidate_source", "")),
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "support_ratio": ratio,
        "ratio_tag": tag,
        "label_space": selected_label_space,
        "beam_label_space": label_meta.get("beam_label_space", ""),
        "beam_label_mapping_fingerprint": label_meta.get("beam_label_mapping_fingerprint", ""),
        "beam_label_mapping": label_meta.get("beam_label_mapping", {}),
        "raw_to_mapped_mapping_source": label_meta.get("raw_to_mapped_mapping_source", ""),
        "topk": selected_topk,
        "num_beams": num_beams,
        "row_count": len(enriched),
        "support_count": sum(1 for row in enriched if _is_support_role(row.get("support_query_role") or row.get("split_role"))),
        "query_count": sum(1 for row in enriched if _is_query_role(row.get("support_query_role") or row.get("split_role"))),
        "lidar_availability": {
            "available_count": lidar_available_count,
            "missing_count": len(enriched) - lidar_available_count,
            "missing_fields": missing_fields,
        },
        "rsu_pose_source": _rsu_pose_source(enriched),
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


def enrich_gps_lidar_bgam_row(
    row: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    topk: int,
    num_beams: int,
    beam_angle_source: str,
    beam_angle_convention: str,
    bev_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    geometry_cfg = _mapping(cfg.get("geometry"))
    lidar_cfg = _mapping(cfg.get("lidar"))
    mapping = _mapping(data_cfg.get("column_mapping"))
    normalized = dict(row)
    _apply_column_mapping(normalized, row, mapping)
    _ensure_candidates_from_gps_prior(normalized, topk=topk, num_beams=num_beams)

    if not _has_number(normalized.get("theta_gps")):
        aod = gps_to_rsu_aod(normalized, geometry_cfg)
        normalized["theta_gps"] = aod.theta_gps
        normalized["distance_to_rsu"] = aod.distance_to_rsu
        normalized["gps_dx"] = aod.dx
        normalized["gps_dy"] = aod.dy
        normalized["coordinate_frame"] = aod.coordinate_source
        normalized["rsu_yaw"] = aod.metadata.get("rsu_yaw", "")
        normalized["rsu_yaw_source"] = aod.metadata.get("rsu_yaw_source", "")
    else:
        normalized["theta_gps"] = _float(normalized.get("theta_gps"), 0.0)
        normalized["distance_to_rsu"] = _float(normalized.get("distance_to_rsu"), _float(normalized.get("range"), 0.0))
        normalized.setdefault("coordinate_frame", str(geometry_cfg.get("coordinate_frame", "manifest")))
        normalized.setdefault("rsu_yaw", geometry_cfg.get("default_rsu_yaw", 0.0))
        normalized.setdefault("rsu_yaw_source", "manifest_or_existing")

    lidar_path = str(normalized.get("lidar_path") or "").strip()
    bev_path = str(normalized.get("lidar_bev_cache_path") or "").strip()
    if not bev_path and lidar_path and str(lidar_cfg.get("cache_dir") or "").strip():
        from kd_sensing.data.transform_ops.lidar import lidar_cache_path, parameterized_lidar_cache_dir

        cache_dir = parameterized_lidar_cache_dir(
            lidar_cfg.get("cache_dir", "outputs/cache/deepsense6g_lidar_bev"),
            bev_size=tuple(lidar_cfg.get("bev_size", (64, 64))),
            roi=tuple(lidar_cfg.get("roi", (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0))),
            fov_degrees=lidar_cfg.get("fov_degrees"),
            remove_ground=bool(lidar_cfg.get("remove_ground", False)),
            ground_z_threshold=float(lidar_cfg.get("ground_z_threshold", 0.1)),
        )
        bev_path = str(lidar_cache_path(cache_dir, lidar_path))
    normalized["lidar_path"] = lidar_path
    normalized["lidar_bev_cache_path"] = bev_path
    raw_exists = _path_exists(lidar_path, data_root=str(lidar_cfg.get("data_root", data_cfg.get("data_root", ""))), scene=str(normalized.get("scene", "")))
    bev_exists = _path_exists(bev_path, data_root="", scene="")
    normalized["lidar_available"] = bool(raw_exists or bev_exists)
    normalized["lidar_missing_reason"] = "" if normalized["lidar_available"] else _lidar_missing_reason(lidar_path, bev_path)
    normalized["beam_angle_source"] = beam_angle_source
    normalized["beam_angle_convention"] = beam_angle_convention
    normalized["bev_grid_metadata_json"] = json.dumps(_json_ready(dict(bev_metadata or {})), sort_keys=True)
    normalized["query_label_used_for_training"] = False
    normalized.setdefault("support_query_role", normalized.get("split_role", normalized.get("role", "")))
    normalized.setdefault("split_role", normalized.get("support_query_role", ""))
    normalized.setdefault("gt_beam", normalized.get("target_label", ""))
    return normalized


def attach_pseudo_history_fields(
    rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    *,
    num_beams: int,
    label_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    history_cfg = _mapping(cfg.get("history"))
    history_len = int(history_cfg.get("history_len", history_cfg.get("length", 8)) or 8)
    history_len = max(1, history_len)
    alignment_policy = str(history_cfg.get("alignment_policy", "nearest_past"))
    configured_source = str(history_cfg.get("pseudo_label_source", "gps_v2_logits"))
    allow_geometry = bool(history_cfg.get("allow_geometry_fallback", history_cfg.get("geometry_fallback", False)))
    sensor_period = float(history_cfg.get("sensor_period", history_cfg.get("gps_period", 1.0)) or 1.0)
    beam_period = float(history_cfg.get("beam_period", 1.0) or 1.0)
    group_keys = tuple(str(item) for item in history_cfg.get("group_keys", ("scene", "split")) if str(item))
    groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(rows):
        if group_keys:
            key = tuple(
                str(row.get(field) or row.get("support_query_role") or "")
                if field == "split"
                else str(row.get(field) or "")
                for field in group_keys
            )
        else:
            key = (str(row.get("scene") or ""), str(row.get("split") or row.get("support_query_role") or ""))
        groups.setdefault(key, []).append((idx, row))
    sorted_groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for key, values in groups.items():
        sorted_groups[key] = sorted(values, key=lambda item: (_history_order_key(item[1]), item[0]))

    history_source_counts: dict[str, int] = {}
    total_missing = 0
    for values in sorted_groups.values():
        for position, (row_index, row) in enumerate(values):
            window = values[max(0, position - history_len + 1) : position + 1]
            pad = history_len - len(window)
            beams = [-1 for _ in range(pad)]
            probs = [0.0 for _ in range(pad)]
            entropy = [0.0 for _ in range(pad)]
            confidence = [0.0 for _ in range(pad)]
            valid = [False for _ in range(pad)]
            timestamps = ["" for _ in range(pad)]
            source_indices = [-1 for _ in range(pad)]
            sources = [configured_source for _ in range(pad)]
            for source_index, source_row in window:
                pseudo = _pseudo_history_value(
                    source_row,
                    cfg,
                    num_beams=num_beams,
                    configured_source=configured_source,
                    allow_geometry_fallback=allow_geometry,
                )
                beams.append(int(pseudo["beam"]))
                probs.append(float(pseudo["prob"]))
                entropy.append(float(pseudo["entropy"]))
                confidence.append(float(pseudo["confidence"]))
                is_valid = int(pseudo["beam"]) >= 0
                valid.append(is_valid)
                timestamps.append(str(source_row.get("timestamp") or source_row.get("frame_id") or _history_order_key(source_row)))
                source_indices.append(int(source_index) if is_valid else -1)
                sources.append(str(pseudo["source"]))
                history_source_counts[str(pseudo["source"])] = history_source_counts.get(str(pseudo["source"]), 0) + int(is_valid)
            missing = int(sum(1 for item in valid if not item))
            total_missing += missing
            row["history_pseudo_beams"] = json.dumps(beams)
            row["history_pseudo_probs"] = json.dumps(probs)
            row["history_pseudo_entropy"] = json.dumps(entropy)
            row["history_pseudo_confidence"] = json.dumps(confidence)
            row["history_valid_mask"] = json.dumps(valid)
            row["history_timestamps"] = json.dumps(timestamps)
            row["history_source_row_indices"] = json.dumps(source_indices)
            row["history_pseudo_label_source"] = _dominant_string(sources, default=configured_source)
            row["history_alignment_policy"] = alignment_policy
            row["history_len"] = int(history_len)
            row["history_missing_count"] = missing
            row["history_sensor_period"] = sensor_period
            row["history_beam_period"] = beam_period
            row["history_replication_ratio"] = sensor_period / max(beam_period, 1e-12)
            row["query_label_used_for_pseudo_history"] = False
            row["history_label_space"] = str(row.get("label_space") or label_metadata.get("label_space") or "")
            row["history_beam_label_space"] = str(row.get("beam_label_space") or label_metadata.get("beam_label_space") or "")
            row["history_beam_label_mapping_fingerprint"] = str(
                row.get("beam_label_mapping_fingerprint") or label_metadata.get("beam_label_mapping_fingerprint") or ""
            )

    return {
        "summary_rows": _pseudo_history_summary_rows(rows, num_beams=num_beams),
        "metadata": {
            "history_len": int(history_len),
            "history_alignment_policy": alignment_policy,
            "pseudo_label_source": configured_source,
            "history_group_keys": list(group_keys),
            "history_source_counts": history_source_counts,
            "sensor_period": sensor_period,
            "beam_period": beam_period,
            "replication_ratio": sensor_period / max(beam_period, 1e-12),
            "missing_history_count": int(total_missing),
            "query_label_used_for_pseudo_history": False,
            "label_space": str(label_metadata.get("label_space") or ""),
            "beam_label_space": str(label_metadata.get("beam_label_space") or ""),
            "beam_label_mapping_fingerprint": str(label_metadata.get("beam_label_mapping_fingerprint") or ""),
        },
    }


def _resolve_source_manifest(
    cfg: Mapping[str, Any],
    *,
    ratio: float,
    label_space: str,
    topk: int,
    output_root: Path,
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
    candidate = output_root / ratio_tag(ratio) / label_space / str(outputs_cfg.get("manifest_dir", "manifest")) / TOP8_MANIFEST_NAME
    if candidate.exists():
        return candidate
    try:
        result = build_topk_candidate_manifest(cfg, support_ratio=ratio, label_space=label_space, topk=topk)
        candidate = Path(str(result.get("manifest_path", "")))
        return candidate if candidate.exists() else None
    except Exception:
        return None


def _apply_column_mapping(target: dict[str, Any], row: Mapping[str, Any], mapping: Mapping[str, Any]) -> None:
    for internal, source in mapping.items():
        if not source or internal in target and target.get(internal) not in {None, ""}:
            continue
        if isinstance(source, str) and source in row:
            target[str(internal)] = row[source]
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            for name in source:
                if str(name) in row and row.get(str(name)) not in {None, ""}:
                    target[str(internal)] = row[str(name)]
                    break


def _ensure_candidates_from_gps_prior(row: dict[str, Any], *, topk: int, num_beams: int) -> None:
    if f"cand0_beam" in row and str(row.get("cand0_beam", "")) != "":
        return
    probs = [_float(row.get(f"gps_prob_{idx}"), math.nan) for idx in range(num_beams)]
    if not any(math.isfinite(value) for value in probs):
        logits = [_float(row.get(f"gps_logit_{idx}"), math.nan) for idx in range(num_beams)]
        if not any(math.isfinite(value) for value in logits):
            return
        logits_np = np.asarray([value if math.isfinite(value) else -1e9 for value in logits], dtype=np.float64)
        shifted = logits_np - float(np.max(logits_np))
        probs_np = np.exp(shifted) / np.exp(shifted).sum()
    else:
        probs_np = np.asarray([value if math.isfinite(value) else 0.0 for value in probs], dtype=np.float64)
        probs_np = probs_np / np.clip(probs_np.sum(), 1e-12, None)
        logits_np = np.log(np.clip(probs_np, 1e-12, None))
    order = np.argsort(probs_np)[::-1][: int(topk)]
    gps_top1 = int(order[0]) if order.size else -1
    row["gps_top1"] = row.get("gps_top1", gps_top1)
    target = _int(row.get("target_label"), _int(row.get("gt_beam"), -100))
    for idx, beam in enumerate(order.tolist()):
        row[f"cand{idx}_beam"] = int(beam)
        row[f"cand{idx}_prob"] = float(probs_np[beam])
        row[f"cand{idx}_logit"] = float(logits_np[beam])
        row[f"cand{idx}_rank"] = idx + 1
        row[f"cand{idx}_dist_to_gps_top1"] = circular_distance(beam, gps_top1, num_beams=num_beams)
    target_in = bool(target in set(int(item) for item in order.tolist())) if target >= 0 else False
    row["target_in_top8"] = target_in
    row["target_candidate_index"] = int(np.where(order == target)[0][0]) if target_in else -1
    if target >= 0 and order.size:
        distances = [circular_distance(int(beam), target, num_beams=num_beams) for beam in order.tolist()]
        nearest = int(np.argmin(np.asarray(distances)))
        row["nearest_candidate_index"] = nearest
        row["nearest_candidate_error"] = int(distances[nearest])
        row["top8_oracle_error"] = int(distances[nearest])
        row["top8_oracle_beam"] = int(order[nearest])
    else:
        row["nearest_candidate_index"] = -1
        row["nearest_candidate_error"] = ""
        row["top8_oracle_error"] = ""
        row["top8_oracle_beam"] = -1


def _pseudo_history_value(
    row: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    num_beams: int,
    configured_source: str,
    allow_geometry_fallback: bool,
) -> dict[str, Any]:
    beam = _int(row.get("gps_top1"), _int(row.get("gps_pred_top1"), -1))
    if 0 <= beam < int(num_beams):
        prob = _float(row.get("gps_top1_prob"), 0.0)
        entropy = _float(row.get("gps_entropy"), 0.0)
        return {
            "beam": int(beam),
            "prob": float(prob if prob > 0 else 1.0),
            "entropy": float(entropy),
            "confidence": float(prob if prob > 0 else math.exp(-max(entropy, 0.0))),
            "source": configured_source or "gps_v2_logits",
        }
    if allow_geometry_fallback and _has_number(row.get("theta_gps")):
        theta = _float(row.get("theta_gps"), 0.0)
        geometry_beam = int(round(((theta + math.pi) / (2.0 * math.pi)) * int(num_beams))) % int(num_beams)
        return {
            "beam": geometry_beam,
            "prob": 1.0,
            "entropy": 0.0,
            "confidence": 1.0,
            "source": "geometry_fallback",
        }
    return {"beam": -1, "prob": 0.0, "entropy": 0.0, "confidence": 0.0, "source": "missing"}


def _history_order_key(row: Mapping[str, Any]) -> float:
    for key in ("timestamp", "frame_id", "seq_index", "top8_manifest_row_index", "gps_logits_row_index"):
        value = row.get(key)
        if value in {None, ""}:
            continue
        parsed = _numeric_suffix(value)
        if parsed is not None:
            return parsed
    sample_id = str(row.get("sample_id") or "")
    parsed = _numeric_suffix(sample_id)
    return float(parsed) if parsed is not None else 0.0


def _numeric_suffix(value: Any) -> float | None:
    raw = str(value)
    token = ""
    for char in reversed(raw):
        if char.isdigit() or char in {".", "-"}:
            token = char + token
        elif token:
            break
    if not token or token in {"-", ".", "-."}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _pseudo_history_summary_rows(rows: Sequence[Mapping[str, Any]], *, num_beams: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in rows:
        beams = _json_list(row.get("history_pseudo_beams"))
        probs = _json_float_list(row.get("history_pseudo_confidence"))
        entropy = _json_float_list(row.get("history_pseudo_entropy"))
        valid = _json_bool_list(row.get("history_valid_mask"))
        source_indices = _json_list(row.get("history_source_row_indices"))
        for step, beam in enumerate(beams):
            confidence = probs[step] if step < len(probs) else 0.0
            bucket = _confidence_bucket(confidence, valid=bool(valid[step]) if step < len(valid) else False)
            key = (str(row.get("scene") or ""), int(step), bucket, str(row.get("label_space") or ""))
            payload = buckets.setdefault(
                key,
                {
                    "scene": key[0],
                    "history_step": key[1],
                    "confidence_bucket": key[2],
                    "label_space": key[3],
                    "sample_count": 0,
                    "valid_count": 0,
                    "entropy_values": [],
                    "error_values": [],
                },
            )
            payload["sample_count"] += 1
            if step < len(valid) and valid[step]:
                payload["valid_count"] += 1
                if step < len(entropy):
                    payload["entropy_values"].append(float(entropy[step]))
                source_index = source_indices[step] if step < len(source_indices) else -1
                if 0 <= int(source_index) < len(rows):
                    target = _int(rows[int(source_index)].get("target_label"), _int(rows[int(source_index)].get("gt_beam"), -100))
                    if int(beam) >= 0 and target >= 0:
                        payload["error_values"].append(circular_distance(int(beam), target, num_beams=num_beams))
    result = []
    for payload in buckets.values():
        entropy_values = np.asarray(payload.pop("entropy_values"), dtype=np.float64)
        error_values = np.asarray(payload.pop("error_values"), dtype=np.float64)
        sample_count = int(payload["sample_count"])
        valid_count = int(payload["valid_count"])
        result.append(
            {
                **payload,
                "coverage": valid_count / max(sample_count, 1),
                "pseudo_entropy_mean": float(entropy_values.mean()) if entropy_values.size else 0.0,
                "evaluation_only_pseudo_error_mean": float(error_values.mean()) if error_values.size else "",
                "evaluation_only_pseudo_accuracy": float(np.mean(error_values == 0)) if error_values.size else "",
                "target_query_label_used_for_pseudo_history": False,
            }
        )
    return sorted(result, key=lambda item: (str(item["scene"]), int(item["history_step"]), str(item["confidence_bucket"])))


def _pseudo_history_summary_fieldnames() -> list[str]:
    return [
        "scene",
        "history_step",
        "confidence_bucket",
        "label_space",
        "sample_count",
        "valid_count",
        "coverage",
        "pseudo_entropy_mean",
        "evaluation_only_pseudo_error_mean",
        "evaluation_only_pseudo_accuracy",
        "target_query_label_used_for_pseudo_history",
    ]


def _confidence_bucket(value: float, *, valid: bool) -> str:
    if not valid:
        return "missing"
    confidence = float(value)
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"


def _json_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    try:
        return [int(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _json_float_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    try:
        return [float(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _json_bool_list(value: Any) -> list[bool]:
    if isinstance(value, list):
        return [bool(item) for item in value]
    try:
        return [bool(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _dominant_string(values: Sequence[Any], *, default: str = "") -> str:
    counts: dict[str, int] = {}
    for value in values:
        item = str(value or "")
        if item:
            counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get) if counts else str(default)


def _manifest_fieldnames(topk: int) -> list[str]:
    base = [
        "scene",
        "sample_id",
        "timestamp",
        "frame_id",
        "split",
        "support_query_role",
        "split_role",
        "target_label",
        "gt_beam",
        "gps_top1",
        "gps_top1_prob",
        "gps_top2_prob",
        "gps_top1_top2_margin",
        "gps_entropy",
        "theta_gps",
        "distance_to_rsu",
        "gps_dx",
        "gps_dy",
        "coordinate_frame",
        "rsu_x",
        "rsu_y",
        "rsu_yaw",
        "rsu_yaw_source",
        "lidar_path",
        "lidar_bev_cache_path",
        "lidar_available",
        "lidar_missing_reason",
        "target_in_top8",
        "target_candidate_index",
        "nearest_candidate_index",
        "nearest_candidate_error",
        "top8_oracle_error",
        "top8_oracle_beam",
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
        base.extend([f"cand{idx}_beam", f"cand{idx}_logit", f"cand{idx}_prob", f"cand{idx}_rank", f"cand{idx}_dist_to_gps_top1"])
    return base


def _path_exists(value: str, *, data_root: str, scene: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    path = Path(raw)
    if path.exists():
        return True
    if not path.is_absolute() and data_root:
        base = Path(data_root)
        return (base / raw).exists() or (base / scene / raw).exists()
    return False


def _lidar_missing_reason(lidar_path: str, bev_path: str) -> str:
    missing = []
    if not lidar_path:
        missing.append("lidar_path")
    if not bev_path:
        missing.append("lidar_bev_cache_path")
    if not missing:
        missing.append("lidar_file")
    return "missing_" + "+".join(missing)


def _rsu_pose_source(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("rsu_yaw_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _missing_field_names(message: str) -> list[str]:
    fields = []
    for field in ("user_x", "user_y", "user_lat", "user_lon", "rsu_lat", "rsu_lon", "lidar_path", "lidar_bev_cache_path"):
        if field in message:
            fields.append(field)
    return fields


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys([*fieldnames, *_fieldnames(rows)]))
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


def _has_number(value: Any) -> bool:
    try:
        if value in {None, ""}:
            return False
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_query_role(value: Any) -> bool:
    return str(value or "").startswith("query")


def _is_support_role(value: Any) -> bool:
    role = str(value or "")
    return bool(role) and not role.startswith("query")


__all__ = [
    "MANIFEST_NAME",
    "METADATA_NAME",
    "build_gps_lidar_bgam_manifest",
    "enrich_gps_lidar_bgam_row",
]
