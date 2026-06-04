from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.data.beam_label_space import (
    attach_label_space_metadata,
    label_space_metadata,
    validate_label_space_rows,
)


DEFAULT_SCENES = ("scenario31", "scenario32", "scenario33", "scenario34")
PREDICTION_LOGIT_NAMES = ("gps_logits.npy", "logits.npy", "pred_logits.npy")
PREDICTION_LOGIT_INDEX_NAMES = ("gps_logits_index.csv", "logits_index.csv", "pred_logits_index.csv")
MANIFEST_NAME = "top8_candidate_manifest.csv"
METADATA_NAME = "top8_candidate_manifest_metadata.json"
RECALL_SUMMARY_NAME = "top8_recall_summary.csv"
RANK_DISTRIBUTION_NAME = "candidate_rank_distribution.csv"
STRICT_LOGITS_ERROR = (
    "DeepSense6G Top8 candidate selector requires saved GPS v2 logits and a logits index. "
    "Rerun GPS v2 with --save-logits before building the Top8 candidate manifest."
)


@dataclass(frozen=True)
class StrictGpsLogits:
    logits_path: Path
    index_path: Path
    logits: np.ndarray
    index_rows: list[dict[str, str]]


def ratio_tag(value: float | str) -> str:
    number = float(value)
    return f"r{int(round(number * 100)):02d}"


def load_strict_gps_logits(
    gps_dir: str | Path,
    *,
    num_beams: int = 64,
    logits_names: Sequence[str] = PREDICTION_LOGIT_NAMES,
    index_names: Sequence[str] = PREDICTION_LOGIT_INDEX_NAMES,
) -> StrictGpsLogits:
    base = Path(gps_dir)
    logits_path = _first_existing(base / str(name) for name in logits_names)
    index_path = _first_existing(base / str(name) for name in index_names)
    if logits_path is None or index_path is None:
        missing = []
        if logits_path is None:
            missing.append("logits")
        if index_path is None:
            missing.append("logits index")
        raise FileNotFoundError(f"{STRICT_LOGITS_ERROR} Missing: {', '.join(missing)} in {base}.")
    logits = np.load(logits_path)
    if logits.ndim != 2 or int(logits.shape[-1]) != int(num_beams):
        raise ValueError(f"{logits_path} must have shape [N, {num_beams}], got {tuple(logits.shape)}.")
    index_rows = _read_csv(index_path)
    if not index_rows:
        raise ValueError(f"{index_path} is empty; {STRICT_LOGITS_ERROR}")
    for row in index_rows:
        row_index = _int(row.get("row_index"), -1)
        if row_index < 0 or row_index >= int(logits.shape[0]):
            raise ValueError(f"{index_path} contains invalid row_index={row_index}.")
    return StrictGpsLogits(logits_path=logits_path, index_path=index_path, logits=logits, index_rows=index_rows)


def build_topk_candidate_manifest(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    topk: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    candidate_cfg = _mapping(cfg.get("candidate"))
    image_cfg = _mapping(cfg.get("image"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    tag = ratio_tag(ratio)
    selected_topk = int(topk if topk is not None else candidate_cfg.get("topk", 8))
    num_beams = int(data_cfg.get("num_beams", 64))
    if selected_topk <= 0 or selected_topk > num_beams:
        raise ValueError(f"topk must be in [1, {num_beams}], got {selected_topk}.")
    label_meta = label_space_metadata(data_cfg, selected_label_space, num_beams=num_beams)

    gps_root = Path(str(data_cfg.get("gps_sweep_root", "outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep")))
    gps_dir = gps_root / tag / selected_label_space
    out_root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/deepsense6g_top8_selector"))
    result_dir = out_root / tag / selected_label_space
    manifest_dir = result_dir / str(outputs_cfg.get("manifest_dir", "manifest"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / MANIFEST_NAME
    metadata_path = manifest_dir / METADATA_NAME
    recall_path = manifest_dir / RECALL_SUMMARY_NAME
    rank_path = manifest_dir / RANK_DISTRIBUTION_NAME

    protocol = str(candidate_cfg.get("gps_protocol", "target_adapt_beambench"))
    ablation_request = str(candidate_cfg.get("gps_ablation", "best_by_scene"))
    scenes = tuple(str(item) for item in data_cfg.get("scenes", DEFAULT_SCENES))
    strict = load_strict_gps_logits(
        gps_dir,
        num_beams=num_beams,
        logits_names=tuple(candidate_cfg.get("logits_names") or PREDICTION_LOGIT_NAMES),
        index_names=tuple(candidate_cfg.get("logits_index_names") or PREDICTION_LOGIT_INDEX_NAMES),
    )
    validate_label_space_rows(
        strict.index_rows,
        expected=label_meta,
        source_path=strict.index_path,
        artifact_name="GPS logits index",
        require_fields=selected_label_space != "mapping_disabled",
    )
    scene_ablation = _resolve_scene_ablation(
        gps_dir,
        gps_root=gps_root,
        topk_analysis_dir=Path(str(data_cfg.get("topk_analysis_dir", gps_root / "topk_analysis"))),
        protocol=protocol,
        requested=ablation_request,
        scenes=scenes,
        support_ratio=ratio,
    )
    index = _LogitsIndex(strict)
    predictions = _read_csv(gps_dir / "predictions.csv")
    validate_label_space_rows(
        predictions,
        expected=label_meta,
        source_path=gps_dir / "predictions.csv",
        artifact_name="GPS predictions",
        require_fields=selected_label_space != "mapping_disabled",
    )
    selected_predictions = [
        row
        for row in predictions
        if str(row.get("protocol")) == protocol
        and str(row.get("label_space") or selected_label_space) == selected_label_space
        and str(row.get("scene") or "") in scenes
        and str(row.get("ablation") or "") == scene_ablation.get(str(row.get("scene") or ""), str(row.get("ablation") or ""))
        and str(row.get("support_query_role") or "").startswith("query")
    ]
    if not selected_predictions:
        selected_predictions = [
            row
            for row in predictions
            if str(row.get("protocol")) == protocol
            and str(row.get("label_space") or selected_label_space) == selected_label_space
            and str(row.get("scene") or "") in scenes
            and str(row.get("support_query_role") or "").startswith("query")
        ]

    support_rows = [
        row
        for row in _read_csv(gps_dir / "support_manifest.csv")
        if str(row.get("protocol")) == protocol
        and str(row.get("label_space") or selected_label_space) == selected_label_space
        and str(row.get("role") or "support") == "support"
        and str(row.get("scene") or row.get("target_scene") or "") in scenes
    ]

    data_root = Path(str(data_cfg.get("data_root", "dataset/DeepSense6G")))
    raw_index = _DeepSenseRawIndex(data_root, data_cfg)
    ae_index = _AeFeatureIndex(image_cfg)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = list(ae_index.warnings)
    for support in support_rows:
        scene = str(support.get("scene") or support.get("target_scene") or "")
        sample_id = str(support.get("sample_id") or "")
        ablation = scene_ablation.get(scene, ablation_request)
        if ablation in {"best", "best_by_scene"}:
            ablation = _fallback_ablation_for_scene(predictions, protocol=protocol, scene=scene)
        try:
            row_index = index.lookup(scene, sample_id, protocol, ablation, role="support")
        except KeyError as exc:
            warnings.append(str(exc))
            continue
        raw = raw_index.lookup(sample_id)
        manifest_row = _manifest_row(
            scene=scene,
            sample_id=sample_id,
            role="support",
            split=str(support.get("split") or "train"),
            split_role="support",
            target_label=_int(support.get("target_label"), -100),
            source_row=support,
            raw_row=raw,
            ae_feature=ae_index.lookup(scene, sample_id),
            data_root=data_root,
            image_cfg=image_cfg,
            logits=strict.logits[row_index],
            logits_row_index=row_index,
            logits_source=str(strict.logits_path.name),
            protocol=protocol,
            ablation=ablation,
            topk=selected_topk,
            num_beams=num_beams,
        )
        rows.append(attach_label_space_metadata(manifest_row, label_meta))

    seen_prediction_keys: set[tuple[str, str]] = set()
    for pred in selected_predictions:
        scene = str(pred.get("scene") or "")
        sample_id = str(pred.get("sample_id") or "")
        key = (scene, sample_id)
        if key in seen_prediction_keys:
            continue
        seen_prediction_keys.add(key)
        ablation = str(pred.get("ablation") or scene_ablation.get(scene, ""))
        try:
            row_index = index.lookup(scene, sample_id, protocol, ablation, role=str(pred.get("support_query_role") or "query_test"))
        except KeyError as exc:
            warnings.append(str(exc))
            continue
        raw = raw_index.lookup(sample_id)
        manifest_row = _manifest_row(
            scene=scene,
            sample_id=sample_id,
            role=str(pred.get("support_query_role") or "query_test"),
            split=str(pred.get("split") or "test"),
            split_role=str(pred.get("support_query_role") or "query_test"),
            target_label=_int(pred.get("true_beam"), _int(pred.get("target_label"), -100)),
            source_row=pred,
            raw_row=raw,
            ae_feature=ae_index.lookup(scene, sample_id),
            data_root=data_root,
            image_cfg=image_cfg,
            logits=strict.logits[row_index],
            logits_row_index=row_index,
            logits_source=str(strict.logits_path.name),
            protocol=protocol,
            ablation=ablation,
            topk=selected_topk,
            num_beams=num_beams,
        )
        rows.append(attach_label_space_metadata(manifest_row, label_meta))

    for manifest_row_index, row in enumerate(rows):
        row["top8_manifest_row_index"] = int(manifest_row_index)

    _write_csv(manifest_path, rows)
    recall_rows = _topk_recall_rows(rows)
    _write_csv(recall_path, recall_rows)
    _write_csv(rank_path, _candidate_rank_distribution(rows))
    alignment = _topk_alignment(
        rows,
        gps_root=gps_root,
        topk_analysis_dir=Path(str(data_cfg.get("topk_analysis_dir", gps_root / "topk_analysis"))),
        support_ratio=ratio,
        protocol=protocol,
        topk=selected_topk,
        tolerance=float(_mapping(cfg.get("metrics")).get("top8_recall_tolerance", 0.03)),
    )
    warnings.extend(alignment.get("warnings", []))
    metadata = {
        "workflow": "deepsense6g_gps_top8_candidate_manifest",
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "gps_dir": str(gps_dir),
        "gps_logits_path": str(strict.logits_path),
        "gps_logits_index_path": str(strict.index_path),
        "support_ratio": ratio,
        "ratio_tag": tag,
        "label_space": selected_label_space,
        "beam_label_space": label_meta.get("beam_label_space", ""),
        "beam_label_mapping_fingerprint": label_meta.get("beam_label_mapping_fingerprint", ""),
        "beam_label_mapping": label_meta.get("beam_label_mapping", {}),
        "raw_to_mapped_mapping_source": label_meta.get("raw_to_mapped_mapping_source", ""),
        "num_beams": num_beams,
        "topk": selected_topk,
        "gps_protocol": protocol,
        "gps_ablation_by_scene": scene_ablation,
        "camera_ae_feature_path": str(ae_index.features_path) if ae_index.features_path is not None else "",
        "camera_ae_feature_index_path": str(ae_index.index_path) if ae_index.index_path is not None else "",
        "camera_ae_feature_available_count": sum(1 for row in rows if _bool(row.get("camera_ae_feature_available"))),
        "row_count": len(rows),
        "support_count": sum(1 for row in rows if str(row.get("support_query_role")) == "support"),
        "query_count": sum(1 for row in rows if str(row.get("support_query_role")).startswith("query")),
        "top8_recall_overall": _mean_bool(row.get("target_in_top8") for row in rows if str(row.get("support_query_role")).startswith("query")),
        "topk_alignment": alignment,
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
        "top8_recall_overall": metadata["top8_recall_overall"],
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
        role_values = [role, "query_test" if str(role).startswith("query") else role, "support" if role == "support" else role]
        for role_value in role_values:
            key = (scene, sample_id, protocol, ablation, role_value)
            if key in self.by_key:
                return self.by_key[key]
        key_no_role = (scene, sample_id, protocol, ablation)
        if key_no_role in self.by_key_no_role:
            return self.by_key_no_role[key_no_role]
        raise KeyError(f"Missing GPS logits mapping for {(scene, sample_id, protocol, ablation, role)}.")


class _AeFeatureIndex:
    def __init__(self, image_cfg: Mapping[str, Any]) -> None:
        features_value = str(image_cfg.get("ae_feature_path") or "").strip()
        index_value = str(image_cfg.get("ae_feature_index_path") or "").strip()
        self.features_path = Path(features_value) if features_value else None
        self.index_path = Path(index_value) if index_value else None
        self.by_scene_sample: dict[tuple[str, str], int] = {}
        self.by_sample: dict[str, int] = {}
        self.warnings: list[str] = []
        if self.features_path is None and self.index_path is None:
            return
        if self.features_path is None or self.index_path is None:
            self.warnings.append("Both image.ae_feature_path and image.ae_feature_index_path are required to attach camera AE features.")
            return
        if not self.features_path.exists():
            self.warnings.append(f"Configured camera AE feature file is missing: {self.features_path}.")
        if not self.index_path.exists():
            self.warnings.append(f"Configured camera AE feature index is missing: {self.index_path}.")
            return
        for row in _read_csv(self.index_path):
            row_index = _int(row.get("row_index"), -1)
            scene = str(row.get("scene") or "")
            sample_id = str(row.get("sample_id") or "")
            if row_index < 0 or not sample_id:
                continue
            if scene:
                self.by_scene_sample[(scene, sample_id)] = row_index
            self.by_sample.setdefault(sample_id, row_index)

    def lookup(self, scene: str, sample_id: str) -> dict[str, Any]:
        if self.features_path is None:
            return {}
        row_index = self.by_scene_sample.get((str(scene), str(sample_id)), self.by_sample.get(str(sample_id), -1))
        if row_index < 0:
            return {}
        return {
            "camera_ae_feature_path": str(self.features_path),
            "camera_ae_feature_row_index": int(row_index),
            "camera_ae_feature_available": self.features_path.exists(),
        }


class _DeepSenseRawIndex:
    def __init__(self, data_root: Path, data_cfg: Mapping[str, Any]) -> None:
        self.data_root = data_root
        self.data_cfg = data_cfg
        self.cache: dict[tuple[str, str], list[dict[str, str]]] = {}

    def lookup(self, sample_id: str) -> Mapping[str, Any] | None:
        parsed = _parse_sample_id(sample_id)
        if parsed is None:
            return None
        scene, split, row_idx = parsed
        key = (scene, split)
        if key not in self.cache:
            csv_name = str(
                self.data_cfg.get(
                    f"{split}_csv_name",
                    "train_seqs_RA_GPS_LIDAR.csv" if split == "train" else "test_seqs_RA_GPS_LIDAR.csv",
                )
            )
            self.cache[key] = _read_csv(self.data_root / scene / csv_name)
        rows = self.cache.get(key, [])
        if row_idx < 0 or row_idx >= len(rows):
            return None
        return rows[row_idx]


def _manifest_row(
    *,
    scene: str,
    sample_id: str,
    role: str,
    split: str,
    split_role: str,
    target_label: int,
    source_row: Mapping[str, Any],
    raw_row: Mapping[str, Any] | None,
    ae_feature: Mapping[str, Any] | None,
    data_root: Path,
    image_cfg: Mapping[str, Any],
    logits: np.ndarray,
    logits_row_index: int,
    logits_source: str,
    protocol: str,
    ablation: str,
    topk: int,
    num_beams: int,
) -> dict[str, Any]:
    logits_np = np.asarray(logits, dtype=np.float64)
    probs = _softmax_np(logits_np)
    order = np.argsort(logits_np)[::-1][:topk]
    gps_top1 = int(order[0]) if order.size else -1
    top_probs = probs[order]
    top_logits = logits_np[order]
    top2_prob = float(top_probs[1]) if top_probs.size > 1 else 0.0
    top1_prob = float(top_probs[0]) if top_probs.size else 0.0
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()) if probs.size else 0.0
    target = int(target_label)
    gps_error = circular_distance(gps_top1, target, num_beams=num_beams) if gps_top1 >= 0 and target >= 0 else -1
    signed = signed_circular_residual(target, gps_top1, num_beams=num_beams) if gps_top1 >= 0 and target >= 0 else ""
    target_in = bool(target in {int(item) for item in order.tolist()}) if target >= 0 else False
    target_index = int(np.where(order == target)[0][0]) if target_in else -1
    nearest_idx, nearest_error = _nearest_candidate(order, target, num_beams=num_beams)
    nearest_beam = int(order[nearest_idx]) if nearest_idx >= 0 else -1
    image_path = _discover_path(raw_row, data_root=data_root, scene=scene, image_cfg=image_cfg, columns=("camera8", "camera", "rgb", "image", "image_path"))
    lidar_path = _discover_path(raw_row, data_root=data_root, scene=scene, image_cfg={}, columns=("lidar8", "lidar", "lidar_path"))
    radar_path = _discover_path(raw_row, data_root=data_root, scene=scene, image_cfg={}, columns=("radar8", "radar", "radar_path"))
    frame_id = _frame_id_from_sample_id(sample_id)
    theta = _float(source_row.get("theta_degrees"), _float(source_row.get("theta"), 0.0))
    easting = _float(source_row.get("E"), _float(source_row.get("easting"), 0.0))
    northing = _float(source_row.get("N"), _float(source_row.get("northing"), 0.0))
    heading = _float(source_row.get("heading_degrees"), _float(source_row.get("heading"), 0.0))
    speed = _float(source_row.get("speed"), 0.0)
    rel_range = _float(source_row.get("range"), math.sqrt(easting * easting + northing * northing))
    ae_feature = ae_feature or {}
    ae_feature_path = str(
        ae_feature.get("camera_ae_feature_path")
        or source_row.get("camera_ae_feature_path")
        or source_row.get("ae_feature_path")
        or ""
    )
    ae_row_index = _int(
        ae_feature.get("camera_ae_feature_row_index"),
        _int(source_row.get("camera_ae_feature_row_index"), _int(source_row.get("ae_feature_row_index"), -1)),
    )
    ae_available = (
        bool(ae_feature_path)
        and ae_row_index >= 0
        and (Path(ae_feature_path).exists() or _bool(source_row.get("camera_ae_feature_available") or source_row.get("ae_feature_available")))
    )
    row: dict[str, Any] = {
        "scene": scene,
        "sample_id": sample_id,
        "timestamp": frame_id,
        "frame_id": frame_id,
        "split": split,
        "support_query_role": role,
        "split_role": split_role,
        "target_label": target,
        "gps_top1": gps_top1,
        "gps_pred_top1": gps_top1,
        "gps_top1_prob": top1_prob,
        "gps_top2_prob": top2_prob,
        "gps_top1_top2_margin": float(top1_prob - top2_prob),
        "gps_entropy": entropy,
        "gps_circular_error": gps_error if gps_error >= 0 else "",
        "gps_signed_residual": signed,
        "theta_degrees": theta,
        "theta": theta,
        "range": rel_range,
        "E": easting,
        "N": northing,
        "heading_degrees": heading,
        "heading": heading,
        "speed": speed,
        "image_path": image_path,
        "image_exists": bool(image_path) and Path(image_path).exists(),
        "camera_ae_feature_row_index": ae_row_index,
        "camera_ae_feature_path": ae_feature_path,
        "camera_ae_feature_available": ae_available,
        "lidar_feature_path": _feature_path(lidar_path),
        "lidar_path": lidar_path,
        "lidar_feature_available": bool(_feature_path(lidar_path)) and Path(_feature_path(lidar_path)).exists(),
        "radar_feature_path": _feature_path(radar_path),
        "radar_path": radar_path,
        "radar_feature_available": bool(_feature_path(radar_path)) and Path(_feature_path(radar_path)).exists(),
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
        "top8_miss": not target_in,
    }
    for cand_idx, beam in enumerate(order.tolist()):
        row[f"cand{cand_idx}_beam"] = int(beam)
        row[f"cand{cand_idx}_logit"] = float(top_logits[cand_idx])
        row[f"cand{cand_idx}_prob"] = float(top_probs[cand_idx])
        row[f"cand{cand_idx}_rank"] = int(cand_idx + 1)
        row[f"cand{cand_idx}_dist_to_gps_top1"] = int(circular_distance(beam, gps_top1, num_beams=num_beams))
    for cand_idx in range(len(order), topk):
        row[f"cand{cand_idx}_beam"] = -1
        row[f"cand{cand_idx}_logit"] = ""
        row[f"cand{cand_idx}_prob"] = ""
        row[f"cand{cand_idx}_rank"] = ""
        row[f"cand{cand_idx}_dist_to_gps_top1"] = ""
    return row


def circular_distance(prediction: int, target: int, *, num_beams: int = 64) -> int:
    pred = int(prediction) % int(num_beams)
    truth = int(target) % int(num_beams)
    absolute = abs(pred - truth)
    return int(min(absolute, int(num_beams) - absolute))


def signed_circular_residual(target: int, prediction: int, *, num_beams: int = 64) -> int:
    diff = (int(target) - int(prediction)) % int(num_beams)
    half = int(num_beams) // 2
    return int(diff - int(num_beams) if diff > half else diff)


def _resolve_scene_ablation(
    gps_dir: Path,
    *,
    gps_root: Path,
    topk_analysis_dir: Path,
    protocol: str,
    requested: str,
    scenes: Sequence[str],
    support_ratio: float,
) -> dict[str, str]:
    if requested not in {"best", "best_by_scene", "best_by_top8"}:
        return {scene: requested for scene in scenes}
    by_scene_path = topk_analysis_dir / "deepsense6g_gps_v2_topk_by_scene.csv"
    support = f"{int(round(float(support_ratio) * 100))}%"
    result: dict[str, str] = {}
    for row in _read_csv(by_scene_path):
        if str(row.get("protocol")) != protocol:
            continue
        if str(row.get("support_ratio")) != support:
            continue
        if str(row.get("stage")) != "after_target_adapter":
            continue
        scene = str(row.get("scene") or "")
        if scene in scenes:
            result[scene] = str(row.get("ablation") or "")
    if set(result) >= set(scenes):
        return result
    summary_rows = [row for row in _read_csv(gps_dir / "summary_by_scene.csv") if str(row.get("protocol")) == protocol]
    for scene in scenes:
        candidates = [row for row in summary_rows if str(row.get("scene")) == scene]
        if candidates:
            result[scene] = str(max(candidates, key=lambda item: _float(item.get("DBA"), -1.0)).get("ablation") or "")
    if set(result) >= set(scenes):
        return result
    summary_overall = [row for row in _read_csv(gps_dir / "summary_overall.csv") if str(row.get("protocol")) == protocol]
    fallback = str(max(summary_overall, key=lambda item: _float(item.get("DBA"), -1.0)).get("ablation") or "") if summary_overall else ""
    return {scene: result.get(scene) or fallback for scene in scenes}


def _fallback_ablation_for_scene(rows: Sequence[Mapping[str, Any]], *, protocol: str, scene: str) -> str:
    for row in rows:
        if str(row.get("protocol")) == protocol and str(row.get("scene")) == scene:
            return str(row.get("ablation") or "")
    return ""


def _topk_alignment(
    rows: Sequence[Mapping[str, Any]],
    *,
    gps_root: Path,
    topk_analysis_dir: Path,
    support_ratio: float,
    protocol: str,
    topk: int,
    tolerance: float,
) -> dict[str, Any]:
    query_rows = [row for row in rows if str(row.get("support_query_role")).startswith("query")]
    actual = _mean_bool(row.get("target_in_top8") for row in query_rows)
    support = f"{int(round(float(support_ratio) * 100))}%"
    expected: float | None = None
    for row in _read_csv(topk_analysis_dir / "deepsense6g_gps_v2_topk_overall.csv"):
        if (
            str(row.get("protocol")) == protocol
            and str(row.get("support_ratio")) == support
            and str(row.get("stage")) == "after_target_adapter"
        ):
            expected = _float(row.get(f"top{topk}"), math.nan)
            break
    warnings: list[str] = []
    if expected is not None and not math.isnan(expected) and abs(actual - expected) > float(tolerance):
        warnings.append(
            f"Top{topk} recall alignment differs from GPS v2 TopK analysis: manifest={actual:.6f}, analysis={expected:.6f}."
        )
    return {
        "topk_analysis_dir": str(topk_analysis_dir),
        "actual_topk_recall": actual,
        "expected_topk_recall": expected,
        "absolute_diff": None if expected is None or math.isnan(expected) else abs(actual - expected),
        "warnings": warnings,
    }


def _topk_recall_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    query_rows = [row for row in rows if str(row.get("support_query_role")).startswith("query")]
    result = [
        {
            "scene": "overall",
            "sample_count": len(query_rows),
            "top8_recall": _mean_bool(row.get("target_in_top8") for row in query_rows),
            "top8_miss_rate": 1.0 - _mean_bool(row.get("target_in_top8") for row in query_rows),
        }
    ]
    for scene in sorted({str(row.get("scene")) for row in query_rows}):
        scene_rows = [row for row in query_rows if str(row.get("scene")) == scene]
        result.append(
            {
                "scene": scene,
                "sample_count": len(scene_rows),
                "top8_recall": _mean_bool(row.get("target_in_top8") for row in scene_rows),
                "top8_miss_rate": 1.0 - _mean_bool(row.get("target_in_top8") for row in scene_rows),
            }
        )
    return result


def _candidate_rank_distribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if not str(row.get("support_query_role")).startswith("query"):
            continue
        rank = "miss" if not _bool(row.get("target_in_top8")) else str(int(_int(row.get("target_candidate_index"), -1)) + 1)
        key = (str(row.get("scene") or ""), rank)
        counts[key] = counts.get(key, 0) + 1
    return [{"scene": scene, "rank": rank, "count": count} for (scene, rank), count in sorted(counts.items())]


def _nearest_candidate(order: np.ndarray, target: int, *, num_beams: int) -> tuple[int, int]:
    if int(target) < 0 or order.size == 0:
        return -1, -1
    distances = [circular_distance(int(beam), int(target), num_beams=num_beams) for beam in order.tolist()]
    idx = int(np.argmin(np.asarray(distances, dtype=np.int64)))
    return idx, int(distances[idx])


def _discover_path(
    raw: Mapping[str, Any] | None,
    *,
    data_root: Path,
    scene: str,
    image_cfg: Mapping[str, Any],
    columns: Sequence[str],
) -> str:
    if raw is None:
        return ""
    scene_root = data_root / scene
    configured = tuple(str(item) for item in image_cfg.get("path_columns", ()) if item)
    for column in (*columns, *configured):
        value = str(raw.get(column) or "").strip()
        if value:
            return str(_joined_resource(scene_root, value))
    return ""


def _feature_path(path_value: str) -> str:
    suffix = Path(str(path_value)).suffix.lower()
    return str(path_value) if suffix in {".npy", ".npz", ".pt", ".pth", ".csv", ".parquet"} else ""


def _joined_resource(scene_root: Path, value: str) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        raw = raw.lstrip("/")
        path = Path(raw)
    if raw.startswith("./"):
        raw = raw[2:]
        path = Path(raw)
    return scene_root / path


def _parse_sample_id(sample_id: str) -> tuple[str, str, int] | None:
    parts = str(sample_id).split(":")
    if len(parts) < 3:
        return None
    try:
        return parts[0], parts[1], int(parts[2])
    except ValueError:
        return None


def _frame_id_from_sample_id(sample_id: str) -> str:
    parts = str(sample_id).split(":")
    if len(parts) >= 4:
        return Path(parts[3]).stem
    return parts[-1] if parts else ""


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - float(np.max(logits))
    probs = np.exp(shifted)
    return probs / np.clip(probs.sum(), 1e-12, None)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_existing(paths: Any) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or _fieldnames(rows))
    with candidate.open("w", encoding="utf-8", newline="") as f:
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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean_bool(values: Any) -> float:
    items = [_bool(value) for value in values]
    return float(np.mean(items)) if items else 0.0


__all__ = [
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PREDICTION_LOGIT_NAMES",
    "STRICT_LOGITS_ERROR",
    "StrictGpsLogits",
    "build_topk_candidate_manifest",
    "circular_distance",
    "load_strict_gps_logits",
    "ratio_tag",
    "signed_circular_residual",
]
