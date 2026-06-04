from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.data.beam_label_calibration import BeamLabelMapping
from kd_sensing.data.beam_label_space import (
    attach_label_space_metadata,
    label_space_metadata,
    resolve_label_space_mapping,
    validate_label_space_rows,
)
from kd_sensing.data.beam_soft_targets import read_beam_power_vector
from kd_sensing.data.deepsense6g_topk_candidate_manifest import (
    PREDICTION_LOGIT_INDEX_NAMES,
    PREDICTION_LOGIT_NAMES,
    StrictGpsLogits,
    circular_distance,
    load_strict_gps_logits,
    signed_circular_residual,
)
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.lidar import lidar_cache_path, parameterized_lidar_cache_dir


DEFAULT_SCENES = (
    "Town10_crossroad_seed24",
    "Town10_skybridge_seed24",
    "Town10_curvyroad_seed42",
    "Town10_Hroad_seed42",
)
MANIFEST_NAME = "top8_candidate_manifest.csv"
METADATA_NAME = "top8_candidate_manifest_metadata.json"
RECALL_SUMMARY_NAME = "top8_recall_summary.csv"
RANK_DISTRIBUTION_NAME = "candidate_rank_distribution.csv"


def build_mmw_town_topk_candidate_manifest(
    cfg: Mapping[str, Any],
    *,
    label_space: str | None = None,
    topk: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    candidate_cfg = _mapping(cfg.get("candidate"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_enabled"))
    selected_topk = int(topk if topk is not None else candidate_cfg.get("topk", data_cfg.get("topk", 8)))
    num_beams = int(data_cfg.get("num_beams", candidate_cfg.get("num_beams", 64)))
    if selected_topk <= 0 or selected_topk > num_beams:
        raise ValueError(f"topk must be in [1, {num_beams}], got {selected_topk}.")

    scenes = _scene_slugs(data_cfg)
    label_meta_by_scene = {
        scene: label_space_metadata(data_cfg, selected_label_space, num_beams=num_beams, scene=scene)
        for scene in scenes
    }
    mapping_by_scene = {
        scene: resolve_label_space_mapping(data_cfg, selected_label_space, num_beams=num_beams, scene=scene)
        for scene in scenes
    }
    gps_dir = _resolve_gps_dir(data_cfg, selected_label_space)
    strict = load_strict_gps_logits(
        gps_dir,
        num_beams=num_beams,
        logits_names=tuple(candidate_cfg.get("logits_names") or PREDICTION_LOGIT_NAMES),
        index_names=tuple(candidate_cfg.get("logits_index_names") or PREDICTION_LOGIT_INDEX_NAMES),
    )
    _validate_rows_by_scene(
        strict.index_rows,
        expected_by_scene=label_meta_by_scene,
        source_path=strict.index_path,
        artifact_name="MMW GPS logits index",
        strict=selected_label_space != "mapping_disabled",
    )

    predictions_path = gps_dir / "predictions.csv"
    predictions = _read_csv(predictions_path)
    _validate_rows_by_scene(
        predictions,
        expected_by_scene=label_meta_by_scene,
        source_path=predictions_path,
        artifact_name="MMW GPS predictions",
        strict=selected_label_space != "mapping_disabled",
    )
    support_manifest_path = gps_dir / "support_manifest.csv"
    support_manifest = _read_csv(support_manifest_path)
    _validate_rows_by_scene(
        support_manifest,
        expected_by_scene=label_meta_by_scene,
        source_path=support_manifest_path,
        artifact_name="MMW GPS support manifest",
        strict=False,
    )

    protocol = str(candidate_cfg.get("gps_protocol", "target_adapt_beambench"))
    ablation_by_scene = _resolve_scene_ablation(
        gps_dir,
        protocol=protocol,
        requested=str(candidate_cfg.get("gps_ablation", "best_by_scene")),
        scenes=scenes,
    )
    index = _LogitsIndex(strict)
    raw_index = _MMWTownRawIndex(data_cfg)
    frame_index = _MMWTownFrameManifestIndex(data_cfg)
    data_root = Path(str(data_cfg.get("data_root", "dataset/MMW/sunny")))

    selected_predictions = _selected_prediction_rows(
        predictions,
        protocol=protocol,
        label_space=selected_label_space,
        scenes=scenes,
        ablation_by_scene=ablation_by_scene,
    )
    support_rows = [
        row
        for row in support_manifest
        if str(row.get("protocol") or "") == protocol
        and str(row.get("label_space") or selected_label_space) == selected_label_space
        and str(row.get("role") or "") == "support"
        and str(row.get("scene") or row.get("target_scene") or "") in scenes
    ]

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    manifest_sources = [
        *(dict(row=row, role="support") for row in support_rows),
        *(dict(row=row, role=str(row.get("support_query_role") or "query_test")) for row in selected_predictions),
    ]
    for source in manifest_sources:
        row = source["row"]
        role_value = str(source["role"])
        scene = str(row.get("scene") or row.get("target_scene") or "")
        sample_id = str(row.get("sample_id") or "")
        ablation = str(row.get("ablation") or ablation_by_scene.get(scene, ""))
        if not ablation:
            warnings.append(f"{sample_id}: missing GPS ablation for scene={scene}.")
            continue
        try:
            row_index = index.lookup(scene, sample_id, protocol, ablation, role=role_value)
        except KeyError as exc:
            warnings.append(str(exc))
            continue
        raw = raw_index.lookup(scene, sample_id, split=str(row.get("split") or ""))
        if raw is None:
            warnings.append(f"{sample_id}: missing MMW prepared CSV row for scene={scene}.")
        label_meta = label_meta_by_scene.get(scene) or label_space_metadata(
            data_cfg, selected_label_space, num_beams=num_beams, scene=scene
        )
        mapping = mapping_by_scene.get(scene) or resolve_label_space_mapping(
            data_cfg, selected_label_space, num_beams=num_beams, scene=scene
        )
        manifest_row = _manifest_row(
            source_row=row,
            raw_row=raw,
            frame_index=frame_index,
            data_root=data_root,
            cfg=cfg,
            mapping=mapping,
            logits=strict.logits[row_index],
            logits_row_index=row_index,
            logits_source=str(strict.logits_path.name),
            protocol=protocol,
            ablation=ablation,
            role=role_value,
            topk=selected_topk,
            num_beams=num_beams,
        )
        rows.append(attach_label_space_metadata(manifest_row, label_meta))

    for manifest_row_index, row in enumerate(rows):
        row["top8_manifest_row_index"] = int(manifest_row_index)

    result_dir = _result_dir(outputs_cfg, output_dir, selected_label_space)
    manifest_dir = result_dir / str(outputs_cfg.get("manifest_dir", "manifest"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / str(outputs_cfg.get("manifest_name", MANIFEST_NAME))
    metadata_path = manifest_dir / str(outputs_cfg.get("metadata_name", METADATA_NAME))
    recall_path = manifest_dir / RECALL_SUMMARY_NAME
    rank_path = manifest_dir / RANK_DISTRIBUTION_NAME
    _write_csv(manifest_path, rows, fieldnames=_manifest_fieldnames(selected_topk))
    recall_rows = _topk_recall_rows(rows, topk=selected_topk)
    _write_csv(recall_path, recall_rows, fieldnames=("scene", "sample_count", "topk_recall", "topk_miss_rate"))
    _write_csv(rank_path, _candidate_rank_distribution(rows), fieldnames=("scene", "rank", "count"))

    metadata = {
        "workflow": "mmw_town_gps_top8_candidate_manifest",
        "dataset_family": "MMW",
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "gps_dir": str(gps_dir),
        "gps_logits_path": str(strict.logits_path),
        "gps_logits_index_path": str(strict.index_path),
        "label_space": selected_label_space,
        "mapping_by_scene": {scene: meta for scene, meta in label_meta_by_scene.items()},
        "num_beams": num_beams,
        "topk": selected_topk,
        "gps_protocol": protocol,
        "gps_ablation_by_scene": ablation_by_scene,
        "row_count": len(rows),
        "support_count": sum(1 for row in rows if _is_support_role(row.get("support_query_role"))),
        "query_count": sum(1 for row in rows if _is_query_role(row.get("support_query_role"))),
        "topk_recall_overall": _mean_bool(row.get("target_in_top8") for row in rows if _is_query_role(row.get("support_query_role"))),
        "normalized_gain_available_count": sum(1 for row in rows if str(row.get("gps_normalized_gain") or "") != ""),
        "query_label_used_for_training": False,
        "warnings": warnings,
    }
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "top8_recall_summary_path": str(recall_path),
        "candidate_rank_distribution_path": str(rank_path),
        "row_count": len(rows),
        "support_count": metadata["support_count"],
        "query_count": metadata["query_count"],
        "top8_recall_overall": metadata["topk_recall_overall"],
        "warnings": warnings,
    }


class _LogitsIndex:
    def __init__(self, strict: StrictGpsLogits) -> None:
        self.by_key: dict[tuple[str, str, str, str, str], int] = {}
        self.by_key_no_role: dict[tuple[str, str, str, str], int] = {}
        for row in strict.index_rows:
            row_index = _int(row.get("row_index"), -1)
            scene = str(row.get("scene") or row.get("target_scene") or "")
            sample_id = str(row.get("sample_id") or "")
            protocol = str(row.get("protocol") or "")
            ablation = str(row.get("ablation") or "")
            role = str(row.get("support_query_role") or row.get("role") or "")
            key = (scene, sample_id, protocol, ablation, role)
            if key in self.by_key:
                raise ValueError(f"Duplicate GPS logits mapping for {key}.")
            self.by_key[key] = row_index
            self.by_key_no_role.setdefault((scene, sample_id, protocol, ablation), row_index)

    def lookup(self, scene: str, sample_id: str, protocol: str, ablation: str, *, role: str = "") -> int:
        role_values = [role]
        if str(role).startswith("query"):
            role_values.append("query_test")
        if str(role) == "query":
            role_values.append("query_test")
        if str(role) == "support":
            role_values.append("support")
        for role_value in role_values:
            key = (scene, sample_id, protocol, ablation, role_value)
            if key in self.by_key:
                return self.by_key[key]
        key_no_role = (scene, sample_id, protocol, ablation)
        if key_no_role in self.by_key_no_role:
            return self.by_key_no_role[key_no_role]
        raise KeyError(f"Missing MMW GPS logits mapping for {(scene, sample_id, protocol, ablation, role)}.")


class _MMWTownRawIndex:
    def __init__(self, data_cfg: Mapping[str, Any]) -> None:
        self.data_root = Path(str(data_cfg.get("data_root", "dataset/MMW/sunny")))
        self.split_tag = str(data_cfg.get("split_tag", "l5p3_group_safe"))
        self.train_split = str(data_cfg.get("train_split", "train"))
        self.test_split = str(data_cfg.get("test_split", "test"))
        self.cache: dict[tuple[str, str], dict[str, dict[str, str]]] = {}

    def lookup(self, scene: str, sample_id: str, *, split: str = "") -> dict[str, str] | None:
        split_candidates = [str(split or ""), self.train_split, self.test_split]
        seen: set[str] = set()
        for split_name in split_candidates:
            if not split_name or split_name in seen:
                continue
            seen.add(split_name)
            index = self._load(scene, split_name)
            if sample_id in index:
                return index[sample_id]
        return None

    def _load(self, scene: str, split: str) -> dict[str, dict[str, str]]:
        key = (str(scene), str(split))
        if key in self.cache:
            return self.cache[key]
        path = self.data_root / "Prepared" / str(scene) / "splits" / self.split_tag / f"{split}.csv"
        rows = _read_csv(path)
        index: dict[str, dict[str, str]] = {}
        for row in rows:
            for field in ("target_sample_id", "sample_id"):
                value = str(row.get(field) or "")
                if value:
                    index[value] = row
        self.cache[key] = index
        return index


class _MMWTownFrameManifestIndex:
    def __init__(self, data_cfg: Mapping[str, Any]) -> None:
        self.data_root = Path(str(data_cfg.get("data_root", "dataset/MMW/sunny")))
        self.cache: dict[str, dict[str, dict[str, str]]] = {}

    def lookup(self, scene: str, sample_id: str = "", frame_id: str = "") -> dict[str, str] | None:
        index = self._load(scene)
        if sample_id and sample_id in index:
            return index[sample_id]
        if frame_id and frame_id in index:
            return index[frame_id]
        return None

    def _load(self, scene: str) -> dict[str, dict[str, str]]:
        scene_key = str(scene)
        if scene_key in self.cache:
            return self.cache[scene_key]
        path = self.data_root / "Prepared" / scene_key / "manifests" / "frame_manifest.csv"
        rows = _read_csv(path)
        index: dict[str, dict[str, str]] = {}
        for row in rows:
            for field in ("sample_id", "frame_id"):
                value = str(row.get(field) or "")
                if value:
                    index[value] = row
        self.cache[scene_key] = index
        return index


def _manifest_row(
    *,
    source_row: Mapping[str, Any],
    raw_row: Mapping[str, Any] | None,
    frame_index: _MMWTownFrameManifestIndex,
    data_root: Path,
    cfg: Mapping[str, Any],
    mapping: BeamLabelMapping,
    logits: np.ndarray,
    logits_row_index: int,
    logits_source: str,
    protocol: str,
    ablation: str,
    role: str,
    topk: int,
    num_beams: int,
) -> dict[str, Any]:
    logits_np = np.asarray(logits, dtype=np.float64)
    probs = _softmax_np(logits_np)
    order = np.argsort(logits_np)[::-1][:topk]
    top_probs = probs[order]
    top_logits = logits_np[order]
    gps_top1 = int(order[0]) if order.size else -1
    target = _int(source_row.get("true_beam"), _int(source_row.get("target_label"), -100))
    target_raw = _int(source_row.get("true_beam_raw"), _int(raw_row.get("future_beam_label1") if raw_row else "", -100))
    scene = str(source_row.get("scene") or source_row.get("target_scene") or (raw_row or {}).get("scene_slug") or "")
    sample_id = str(source_row.get("sample_id") or "")
    split = str(source_row.get("split") or (raw_row or {}).get("split") or "")
    raw = raw_row or {}
    geometry = _geometry_from_raw(raw)
    theta_degrees = _float(source_row.get("theta_degrees"), _float(geometry.get("relative_azimuth"), 0.0))
    rel_x = _float(source_row.get("E"), _float(geometry.get("relative_x"), _float(geometry.get("local_x"), 0.0)))
    rel_y = _float(source_row.get("N"), _float(geometry.get("relative_y"), _float(geometry.get("local_y"), 0.0)))
    rel_range = _float(geometry.get("relative_range"), math.sqrt(rel_x * rel_x + rel_y * rel_y))
    frame_id = _frame_id(raw, sample_id)
    current_sample_id = str(raw.get("sample_id") or "")
    frame_row = frame_index.lookup(scene, sample_id=current_sample_id, frame_id=frame_id)
    lidar_path, lidar_source = _lidar_path(raw, frame_row, cfg)
    lidar_bev_cache_path = _lidar_bev_cache_path(lidar_path, cfg)
    beam_power_path = str(raw.get("future_beam1") or raw.get("beam_power_path") or "")
    power = read_beam_power_vector(joined_resource(data_root, beam_power_path), num_classes=num_beams) if beam_power_path else None
    gps_gain = _normalized_gain(power, gps_top1, mapping=mapping)
    target_in = bool(target in {int(item) for item in order.tolist()}) if target >= 0 else False
    target_index = int(np.where(order == target)[0][0]) if target_in else -1
    nearest_idx, nearest_error = _nearest_candidate(order, target, num_beams=num_beams)
    nearest_beam = int(order[nearest_idx]) if nearest_idx >= 0 else -1
    row: dict[str, Any] = {
        "scene": scene,
        "scene_name": str(source_row.get("scene_name") or scene),
        "scene_id": _int(source_row.get("scene_id"), -1),
        "agent": str(raw.get("agent") or _agent_from_sample_id(sample_id)),
        "condition": str(raw.get("condition") or "sunny"),
        "town": str(raw.get("town") or "Town10"),
        "sample_id": sample_id,
        "current_sample_id": current_sample_id,
        "timestamp": frame_id,
        "frame_id": frame_id,
        "window_end_frame": str(raw.get("window_end_frame") or frame_id),
        "target_sample_id": str(raw.get("target_sample_id") or sample_id),
        "split": split,
        "support_query_role": role,
        "split_role": role,
        "target_label": target,
        "gt_beam": target,
        "target_label_raw": target_raw,
        "gps_top1": gps_top1,
        "gps_pred_top1": gps_top1,
        "gps_top1_prob": float(top_probs[0]) if top_probs.size else 0.0,
        "gps_top2_prob": float(top_probs[1]) if top_probs.size > 1 else 0.0,
        "gps_top1_top2_margin": float(top_probs[0] - top_probs[1]) if top_probs.size > 1 else float(top_probs[0]) if top_probs.size else 0.0,
        "gps_entropy": float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()) if probs.size else 0.0,
        "gps_circular_error": circular_distance(gps_top1, target, num_beams=num_beams) if gps_top1 >= 0 and target >= 0 else "",
        "gps_signed_residual": signed_circular_residual(target, gps_top1, num_beams=num_beams) if gps_top1 >= 0 and target >= 0 else "",
        "gps_normalized_gain": "" if gps_gain is None else gps_gain,
        "theta_degrees": theta_degrees,
        "theta": theta_degrees,
        "theta_gps": math.radians(theta_degrees),
        "range": rel_range,
        "distance_to_rsu": rel_range,
        "E": rel_x,
        "N": rel_y,
        "gps_dx": rel_x,
        "gps_dy": rel_y,
        "heading_degrees": _float(geometry.get("heading_difference"), 0.0),
        "heading": _float(geometry.get("heading_difference"), 0.0),
        "speed": _float(geometry.get("relative_velocity"), 0.0),
        "coordinate_frame": "mmw_relative_geometry",
        "rsu_x": 0.0,
        "rsu_y": 0.0,
        "rsu_yaw": 0.0,
        "rsu_yaw_source": "mmw_relative_geometry",
        "lidar_path": lidar_path,
        "lidar_source": lidar_source,
        "lidar_bev_cache_path": lidar_bev_cache_path,
        "beam_power_path": beam_power_path,
        "gps_logits_row_index": int(logits_row_index),
        "gps_logits_source": logits_source,
        "gps_protocol": protocol,
        "gps_ablation": ablation,
        "target_in_top8": target_in,
        "target_candidate_index": target_index,
        "nearest_candidate_index": nearest_idx,
        "nearest_candidate_error": nearest_error if nearest_error >= 0 else "",
        "top8_oracle_error": nearest_error if nearest_error >= 0 else "",
        "top8_oracle_beam": nearest_beam,
        "top8_oracle_normalized_gain": _empty_if_none(_normalized_gain(power, nearest_beam, mapping=mapping)),
        "top8_miss": not target_in,
        "query_label_used_for_training": False,
    }
    for cand_idx, beam in enumerate(order.tolist()):
        gain = _normalized_gain(power, int(beam), mapping=mapping)
        row[f"cand{cand_idx}_beam"] = int(beam)
        row[f"cand{cand_idx}_logit"] = float(top_logits[cand_idx])
        row[f"cand{cand_idx}_prob"] = float(top_probs[cand_idx])
        row[f"cand{cand_idx}_rank"] = int(cand_idx + 1)
        row[f"cand{cand_idx}_dist_to_gps_top1"] = int(circular_distance(beam, gps_top1, num_beams=num_beams))
        row[f"cand{cand_idx}_normalized_gain"] = _empty_if_none(gain)
    for cand_idx in range(len(order), topk):
        row[f"cand{cand_idx}_beam"] = -1
        row[f"cand{cand_idx}_logit"] = ""
        row[f"cand{cand_idx}_prob"] = ""
        row[f"cand{cand_idx}_rank"] = ""
        row[f"cand{cand_idx}_dist_to_gps_top1"] = ""
        row[f"cand{cand_idx}_normalized_gain"] = ""
    return row


def _resolve_scene_ablation(
    gps_dir: Path,
    *,
    protocol: str,
    requested: str,
    scenes: Sequence[str],
) -> dict[str, str]:
    if requested not in {"best", "best_by_scene"}:
        return {scene: requested for scene in scenes}
    result: dict[str, str] = {}
    rows = [row for row in _read_csv(gps_dir / "summary_by_scene.csv") if str(row.get("protocol") or "") == protocol]
    for scene in scenes:
        candidates = [row for row in rows if str(row.get("scene") or "") == scene]
        if candidates:
            result[scene] = str(max(candidates, key=lambda row: _float(row.get("DBA"), -1.0)).get("ablation") or "")
    summary = [row for row in _read_csv(gps_dir / "summary_overall.csv") if str(row.get("protocol") or "") == protocol]
    fallback = str(max(summary, key=lambda row: _float(row.get("DBA"), -1.0)).get("ablation") or "") if summary else ""
    return {scene: result.get(scene) or fallback for scene in scenes}


def _selected_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    label_space: str,
    scenes: Sequence[str],
    ablation_by_scene: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("protocol") or "") == protocol
        and str(row.get("label_space") or label_space) == label_space
        and str(row.get("scene") or "") in scenes
        and str(row.get("ablation") or "") == str(ablation_by_scene.get(str(row.get("scene") or ""), ""))
        and _is_query_role(row.get("support_query_role"))
    ]
    if selected:
        return selected
    return [
        row
        for row in rows
        if str(row.get("protocol") or "") == protocol
        and str(row.get("label_space") or label_space) == label_space
        and str(row.get("scene") or "") in scenes
        and _is_query_role(row.get("support_query_role"))
    ]


def _resolve_gps_dir(data_cfg: Mapping[str, Any], label_space: str) -> Path:
    root = Path(str(data_cfg.get("gps_v2_artifact_root") or data_cfg.get("gps_output_root") or "outputs/analysis/mmw_town_gps_adapter_v2"))
    if root.name == str(label_space) or (root / "predictions.csv").exists():
        return root
    return root / str(label_space)


def _result_dir(outputs_cfg: Mapping[str, Any], output_dir: str | Path | None, label_space: str) -> Path:
    root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/mmw_town_top8_selector"))
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
    return [value for value in values if value] or list(DEFAULT_SCENES)


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


def _geometry_from_raw(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("geometry5", "geometry4", "geometry3", "geometry2", "geometry1", "relative_geometry_json"):
        payload = _json_dict(row.get(key))
        if payload:
            return payload
    return {}


def _lidar_path(
    raw: Mapping[str, Any],
    frame_row: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> tuple[str, str]:
    lidar_cfg = _mapping(cfg.get("lidar"))
    source = str(lidar_cfg.get("source", "rsu")).strip().lower()
    if source == "rsu":
        rsu_lidar = _rsu_lidar_path(frame_row)
        if rsu_lidar:
            return rsu_lidar, "rsu"
    for key in ("lidar5", "lidar_path", "lidar", "lidar4", "lidar3", "lidar2", "lidar1"):
        value = str(raw.get(key) or "").strip()
        if value:
            return value, "cav_window"
    return "", "missing"


def _rsu_lidar_path(frame_row: Mapping[str, Any] | None) -> str:
    if not frame_row:
        return ""
    payload = _json_dict(frame_row.get("rsu_json"))
    agents = payload.get("agents") if isinstance(payload.get("agents"), Mapping) else {}
    for agent_payload in agents.values():
        if isinstance(agent_payload, Mapping):
            value = str(agent_payload.get("lidar") or "").strip()
            if value:
                return value
    return ""


def _lidar_bev_cache_path(lidar_path: str, cfg: Mapping[str, Any]) -> str:
    raw = str(lidar_path or "").strip()
    if not raw:
        return ""
    lidar_cfg = _mapping(cfg.get("lidar"))
    cache_dir = str(lidar_cfg.get("cache_dir") or "").strip()
    if not cache_dir:
        return ""
    path = parameterized_lidar_cache_dir(
        cache_dir,
        bev_size=tuple(lidar_cfg.get("bev_size", (224, 224))),
        roi=tuple(lidar_cfg.get("roi", (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0))),
        fov_degrees=lidar_cfg.get("fov_degrees"),
        remove_ground=bool(lidar_cfg.get("remove_ground", False)),
        ground_z_threshold=float(lidar_cfg.get("ground_z_threshold", 0.1)),
        background_path=lidar_cfg.get("background_path"),
        background_distance_threshold=float(lidar_cfg.get("background_distance_threshold", 0.2)),
    )
    return str(lidar_cache_path(path, raw))


def _frame_id(raw: Mapping[str, Any], sample_id: str) -> str:
    value = str(raw.get("window_end_frame") or raw.get("end_frame") or "").strip()
    if value:
        return value
    parts = str(sample_id).split(":")
    return parts[-1] if parts else ""


def _agent_from_sample_id(sample_id: str) -> str:
    parts = str(sample_id).split(":")
    return parts[-2] if len(parts) >= 2 else ""


def _normalized_gain(power: np.ndarray | None, mapped_beam: int, *, mapping: BeamLabelMapping) -> float | None:
    if power is None or int(mapped_beam) < 0:
        return None
    values = np.asarray(power, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    try:
        raw_beam = int(mapping.inverse_label(int(mapped_beam)))
    except Exception:
        raw_beam = int(mapped_beam)
    if raw_beam < 0 or raw_beam >= values.size:
        return None
    denominator = float(np.max(values))
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return None
    return float(values[raw_beam] / denominator)


def _empty_if_none(value: Any) -> Any:
    return "" if value is None else value


def _nearest_candidate(order: np.ndarray, target: int, *, num_beams: int) -> tuple[int, int]:
    if int(target) < 0 or order.size == 0:
        return -1, -1
    distances = [circular_distance(int(beam), int(target), num_beams=num_beams) for beam in order.tolist()]
    idx = int(np.argmin(np.asarray(distances, dtype=np.int64)))
    return idx, int(distances[idx])


def _topk_recall_rows(rows: Sequence[Mapping[str, Any]], *, topk: int) -> list[dict[str, Any]]:
    query_rows = [row for row in rows if _is_query_role(row.get("support_query_role"))]
    result = [
        {
            "scene": "overall",
            "sample_count": len(query_rows),
            "topk_recall": _mean_bool(row.get("target_in_top8") for row in query_rows),
            "topk_miss_rate": 1.0 - _mean_bool(row.get("target_in_top8") for row in query_rows),
        }
    ]
    for scene in sorted({str(row.get("scene") or "") for row in query_rows}):
        scene_rows = [row for row in query_rows if str(row.get("scene") or "") == scene]
        result.append(
            {
                "scene": scene,
                "sample_count": len(scene_rows),
                "topk_recall": _mean_bool(row.get("target_in_top8") for row in scene_rows),
                "topk_miss_rate": 1.0 - _mean_bool(row.get("target_in_top8") for row in scene_rows),
            }
        )
    return result


def _candidate_rank_distribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if not _is_query_role(row.get("support_query_role")):
            continue
        rank = "miss" if not _bool(row.get("target_in_top8")) else str(int(_int(row.get("target_candidate_index"), -1)) + 1)
        key = (str(row.get("scene") or ""), rank)
        counts[key] = counts.get(key, 0) + 1
    return [{"scene": scene, "rank": rank, "count": count} for (scene, rank), count in sorted(counts.items())]


def _manifest_fieldnames(topk: int) -> list[str]:
    base = [
        "scene",
        "scene_name",
        "scene_id",
        "agent",
        "condition",
        "town",
        "sample_id",
        "current_sample_id",
        "timestamp",
        "frame_id",
        "window_end_frame",
        "target_sample_id",
        "split",
        "support_query_role",
        "split_role",
        "target_label",
        "gt_beam",
        "target_label_raw",
        "gps_top1",
        "gps_pred_top1",
        "gps_top1_prob",
        "gps_top2_prob",
        "gps_top1_top2_margin",
        "gps_entropy",
        "gps_circular_error",
        "gps_signed_residual",
        "gps_normalized_gain",
        "theta_degrees",
        "theta",
        "theta_gps",
        "range",
        "distance_to_rsu",
        "E",
        "N",
        "gps_dx",
        "gps_dy",
        "heading_degrees",
        "heading",
        "speed",
        "coordinate_frame",
        "rsu_x",
        "rsu_y",
        "rsu_yaw",
        "rsu_yaw_source",
        "lidar_path",
        "lidar_source",
        "lidar_bev_cache_path",
        "beam_power_path",
        "gps_logits_row_index",
        "gps_logits_source",
        "gps_protocol",
        "gps_ablation",
        "target_in_top8",
        "target_candidate_index",
        "nearest_candidate_index",
        "nearest_candidate_error",
        "top8_oracle_error",
        "top8_oracle_beam",
        "top8_oracle_normalized_gain",
        "top8_miss",
        "query_label_used_for_training",
        "label_space",
        "beam_label_space",
        "beam_label_mapping_fingerprint",
        "raw_to_mapped_mapping_source",
        "top8_manifest_row_index",
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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(), 1e-12, None)


def _is_support_role(value: Any) -> bool:
    return str(value or "").startswith("support")


def _is_query_role(value: Any) -> bool:
    return str(value or "").startswith("query")


def _mean_bool(values: Sequence[Any] | Any) -> float:
    items = [_bool(value) for value in values]
    return float(np.mean(items)) if items else 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value in {None, ""}:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


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
    fieldnames: Sequence[str],
) -> None:
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


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = [
    "MANIFEST_NAME",
    "METADATA_NAME",
    "RANK_DISTRIBUTION_NAME",
    "RECALL_SUMMARY_NAME",
    "build_mmw_town_topk_candidate_manifest",
]
