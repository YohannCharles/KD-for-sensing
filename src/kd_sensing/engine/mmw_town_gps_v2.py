import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from kd_sensing.config.io import dump_config
from kd_sensing.data.beam_label_calibration import BeamLabelMapping, resolve_beam_label_mapping
from kd_sensing.data.beam_soft_targets import read_beam_power_vector
from kd_sensing.data.mmw.support_selection import angle_coverage_indices
from kd_sensing.data.transform_ops.gps import latlon_to_utm_xy, read_gps_latlon
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.evaluation.metrics import (
    circular_beam_distance,
    dba_from_circular_distances,
    dba_zero_ratio,
    signed_circular_beam_residual,
)
from kd_sensing.losses.circular import class_balanced_weights


FEATURE_NAMES = (
    "E_norm",
    "N_norm",
    "sin_theta",
    "cos_theta",
    "log_range_norm",
    "sin_heading",
    "cos_heading",
    "speed_norm",
)
SCALER_FIELDS = ("E", "N", "log_range", "speed")
PROTOCOLS = ("source_other_three", "target_adapt_beambench", "within_scene_train")


@dataclass
class SceneSpec:
    name: str
    slug: str
    scene_id: int


@dataclass
class MMWTownGpsV2Sample:
    sample_id: str
    scene: str
    scene_name: str
    scene_id: int
    split: str
    label_raw: int
    label: int
    order_key: float
    theta_degrees: float
    easting: float
    northing: float
    log_range: float
    heading_degrees: float
    speed: float
    branch_source: str = "pseudo"
    branch_key: str = ""
    branch_id: int = 0
    heading_source: str = "heading_difference"
    speed_source: str = "relative_velocity"
    mapping_fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureScaler:
    mean: dict[str, float]
    scale: dict[str, float]
    metadata: dict[str, Any]

    @classmethod
    def fit(cls, samples: list[MMWTownGpsV2Sample], *, fit_split: str) -> "FeatureScaler":
        values = {
            "E": np.asarray([item.easting for item in samples], dtype=np.float64),
            "N": np.asarray([item.northing for item in samples], dtype=np.float64),
            "log_range": np.asarray([item.log_range for item in samples], dtype=np.float64),
            "speed": np.asarray([item.speed for item in samples], dtype=np.float64),
        }
        mean: dict[str, float] = {}
        scale: dict[str, float] = {}
        for key, array in values.items():
            if array.size == 0:
                mean[key] = 0.0
                scale[key] = 1.0
            else:
                mean[key] = float(array.mean())
                std = float(array.std())
                scale[key] = std if std > 1e-8 else 1.0
        metadata = {
            "fit_split": str(fit_split),
            "fit_sample_count": int(len(samples)),
            "fields": list(SCALER_FIELDS),
            "mean": dict(mean),
            "scale": dict(scale),
            "feature_names": list(FEATURE_NAMES),
            "raw_lat_lon_in_tensor": False,
            "heading_coverage": float(
                sum(1 for item in samples if item.heading_source != "default_zero") / max(len(samples), 1)
            ),
            "speed_coverage": float(
                sum(1 for item in samples if item.speed_source != "default_zero") / max(len(samples), 1)
            ),
        }
        return cls(mean=mean, scale=scale, metadata=metadata)

    def transform(self, samples: list[MMWTownGpsV2Sample]) -> np.ndarray:
        rows = []
        for sample in samples:
            theta = math.radians(float(sample.theta_degrees))
            heading = math.radians(float(sample.heading_degrees))
            rows.append(
                [
                    _standardize(sample.easting, self.mean["E"], self.scale["E"]),
                    _standardize(sample.northing, self.mean["N"], self.scale["N"]),
                    math.sin(theta),
                    math.cos(theta),
                    _standardize(sample.log_range, self.mean["log_range"], self.scale["log_range"]),
                    math.sin(heading),
                    math.cos(heading),
                    _standardize(sample.speed, self.mean["speed"], self.scale["speed"]),
                ]
            )
        return np.asarray(rows, dtype=np.float32)


@dataclass
class AdapterFit:
    psi_degrees: float = 0.0
    delta_beams: float = 0.0
    scale: float = 1.0
    flip: str = "forward"
    sigma: float = 2.0
    tau: float = 1.0
    spline_bins: list[float] = field(default_factory=list)
    branch_params: dict[int, "AdapterFit"] = field(default_factory=dict)
    branch_fallback: dict[int, str] = field(default_factory=dict)
    criterion: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["branch_params"] = {str(key): value.to_dict() for key, value in self.branch_params.items()}
        return payload


def run_mmw_town_gps_v2(
    cfg: Mapping[str, Any],
    *,
    label_space: str | None = None,
    target_scene: str | None = None,
    support_ratio: float | None = None,
    support_num: int | None = None,
    support_mode: str | None = None,
    output_dir: str | Path | None = None,
    save_logits: bool | None = None,
    save_prior_probs: bool | None = None,
) -> dict[str, Any]:
    cfg_dict = _json_ready(dict(cfg))
    data_cfg = _mapping(cfg_dict.get("data"))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_enabled"))
    scenes = _scene_specs(data_cfg)
    if target_scene:
        requested = {item.strip() for item in str(target_scene).split(",") if item.strip()}
        scenes = [scene for scene in scenes if scene.name in requested or scene.slug in requested]
        if not scenes:
            raise ValueError(f"--target-scene did not match any configured scene: {target_scene}")
    num_beams = int(data_cfg.get("num_beams", 64))
    mapping_cfg = _resolve_label_space_config(data_cfg, selected_label_space)
    scene_mappings = {
        scene.slug: resolve_beam_label_mapping(mapping_cfg, scene=scene.slug, default_num_classes=num_beams)
        for scene in scenes
    }
    all_scenes_for_sources = _scene_specs(data_cfg)
    all_scene_mappings = {
        scene.slug: resolve_beam_label_mapping(mapping_cfg, scene=scene.slug, default_num_classes=num_beams)
        for scene in all_scenes_for_sources
    }
    dataset_type = str(data_cfg.get("dataset_type", "mmw_town")).strip().lower()
    data_root = Path(str(data_cfg.get("data_root", "dataset/MMW/sunny")))
    split_tag = str(data_cfg.get("split_tag", "l5p3_group_safe"))
    train_split = str(data_cfg.get("train_split", "train"))
    test_split = str(data_cfg.get("test_split", "test"))
    out_dir = Path(output_dir or data_cfg.get("output_root", "outputs/analysis/mmw_town_gps_adapter_v2")) / selected_label_space
    out_dir.mkdir(parents=True, exist_ok=True)
    adapt_cfg = _mapping(cfg_dict.get("adapt"))
    if support_ratio is not None:
        adapt_cfg["support_ratio"] = float(support_ratio)
    if support_num is not None:
        adapt_cfg["support_num"] = int(support_num)
    if support_mode is not None:
        adapt_cfg["support_mode"] = str(support_mode)
    metrics_cfg = _mapping(cfg_dict.get("metrics"))
    ablations = list(_mapping(cfg_dict.get("ablation")).get("enabled") or [])
    if not ablations:
        ablations = ["geo_plus_backbone"]
    protocols = tuple(str(item) for item in cfg_dict.get("protocols", PROTOCOLS))
    max_samples = _optional_int(_mapping(cfg_dict.get("train")).get("max_samples_per_split"))
    loaded = _load_all_samples(
        scenes=all_scenes_for_sources,
        scene_mappings=all_scene_mappings,
        data_root=data_root,
        dataset_type=dataset_type,
        data_cfg=data_cfg,
        split_tag=split_tag,
        train_split=train_split,
        test_split=test_split,
        max_samples=max_samples,
    )
    branch_metadata = _assign_branch_ids(
        [sample for splits in loaded.values() for rows in splits.values() for sample in rows],
        max_k=int(_mapping(adapt_cfg.get("branch")).get("max_k", 4)),
        min_samples=int(_mapping(adapt_cfg.get("branch")).get("min_samples", 8)),
    )
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    theta_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    gps_logits_rows: list[np.ndarray] = []
    gps_logits_index_rows: list[dict[str, Any]] = []
    run_notes: list[dict[str, Any]] = []
    loss_metadata: dict[str, Any] = {}
    output_cfg = _mapping(cfg_dict.get("output"))
    export_logits = bool(output_cfg.get("save_logits", False) if save_logits is None else save_logits)
    export_probs = bool(output_cfg.get("save_prior_probs", False) if save_prior_probs is None else save_prior_probs)
    for protocol in protocols:
        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol}.")
        for target in scenes:
            train_target = loaded[target.slug][train_split]
            test_target = loaded[target.slug][test_split]
            source_train = [
                sample
                for source in all_scenes_for_sources
                if source.slug != target.slug
                for sample in loaded[source.slug][train_split]
            ]
            if protocol == "within_scene_train":
                fit_samples = list(train_target)
                support_samples = list(train_target)
                query_samples: list[MMWTownGpsV2Sample] = []
                protocol_note = "sanity_upper_bound"
                support_info = {"selection_mode": "all_train", "support_count": len(support_samples), "query_count": 0}
            elif protocol == "target_adapt_beambench":
                fit_samples = list(source_train)
                support_samples, query_samples, support_info = select_support_samples(train_target, adapt_cfg)
                protocol_note = "few_shot_target_adapter"
            else:
                fit_samples = list(source_train)
                support_samples = []
                query_samples = []
                protocol_note = "source_other_three_target_labels_eval_only"
                support_info = {"selection_mode": "none", "support_count": 0, "query_count": 0}
            support_rows.extend(
                _support_manifest_rows(
                    support_samples,
                    query_samples,
                    protocol=protocol,
                    target_scene=target.slug,
                    label_space=selected_label_space,
                    support_info=support_info,
                )
            )
            scaler_fit = fit_samples if fit_samples else train_target
            scaler = FeatureScaler.fit(scaler_fit, fit_split="source" if protocol != "within_scene_train" else "train")
            if protocol == "target_adapt_beambench":
                adapter_fit_pool = support_samples
                adapter_fit_split = "target_support"
            else:
                adapter_fit_pool = fit_samples
                adapter_fit_split = "source" if protocol == "source_other_three" else "train"
            if not adapter_fit_pool:
                adapter_fit_pool = fit_samples or train_target
            for ablation in ablations:
                weighted = str(ablation).endswith("_weighted")
                class_weight_mode = "effective_num" if weighted else str(_mapping(cfg_dict.get("loss")).get("class_weight", "none"))
                if weighted or class_weight_mode != "none":
                    weights, meta = class_balanced_weights(
                        [sample.label for sample in fit_samples or adapter_fit_pool],
                        num_classes=num_beams,
                        mode=class_weight_mode,
                        beta=float(_mapping(cfg_dict.get("loss")).get("effective_num_beta", 0.999)),
                    )
                    loss_metadata[f"{protocol}:{target.slug}:{ablation}"] = meta
                    class_prior = np.asarray(weights.tolist(), dtype=np.float64)
                else:
                    class_prior = None
                fit = fit_adapter(
                    adapter_fit_pool,
                    ablation=str(ablation),
                    cfg=cfg_dict,
                    num_beams=num_beams,
                    class_prior=class_prior,
                )
                if export_logits and protocol == "target_adapt_beambench" and support_samples:
                    score_samples(
                        support_samples,
                        fit_samples=fit_samples or adapter_fit_pool,
                        adapter_fit=fit,
                        scaler=scaler,
                        ablation=str(ablation),
                        protocol=protocol,
                        target=target,
                        label_space=selected_label_space,
                        mapping=scene_mappings[target.slug],
                        num_beams=num_beams,
                        dba_delta=float(metrics_cfg.get("dba_delta", 5.0)),
                        logits_sink=gps_logits_rows,
                        logits_index_sink=gps_logits_index_rows,
                        support_query_role="support",
                    )
                rows = score_samples(
                    test_target,
                    fit_samples=fit_samples or adapter_fit_pool,
                    adapter_fit=fit,
                    scaler=scaler,
                    ablation=str(ablation),
                    protocol=protocol,
                    target=target,
                    label_space=selected_label_space,
                    mapping=scene_mappings[target.slug],
                    num_beams=num_beams,
                    dba_delta=float(metrics_cfg.get("dba_delta", 5.0)),
                    logits_sink=gps_logits_rows if export_logits else None,
                    logits_index_sink=gps_logits_index_rows if export_logits else None,
                )
                prediction_rows.extend(rows)
                summary = _summary_from_prediction_rows(
                    rows,
                    protocol=protocol,
                    ablation=str(ablation),
                    target=target,
                    source_scenes=[scene.slug for scene in all_scenes_for_sources if scene.slug != target.slug],
                    label_space=selected_label_space,
                    mapping=scene_mappings[target.slug],
                    protocol_note=protocol_note,
                    support_info=support_info,
                    scaler_metadata=scaler.metadata,
                    adapter_fit=fit,
                    dba_delta=float(metrics_cfg.get("dba_delta", 5.0)),
                    num_beams=num_beams,
                )
                summary_rows.append(summary)
                theta_rows.extend(_residual_by_theta_rows(rows, bins=int(metrics_cfg.get("theta_bins", 12))))
                branch_rows.extend(_residual_by_branch_rows(rows))
                run_notes.append(
                    {
                        "protocol": protocol,
                        "target_scene": target.slug,
                        "ablation": str(ablation),
                        "adapter_fit_split": adapter_fit_split,
                        "scaler_fit_split": scaler.metadata["fit_split"],
                        "support": support_info,
                    }
                )
    overall_rows = _overall_rows(summary_rows)
    _write_csv(out_dir / "predictions.csv", prediction_rows)
    _write_csv(out_dir / "summary_by_scene.csv", summary_rows)
    _write_csv(out_dir / "summary_overall.csv", overall_rows)
    _write_csv(out_dir / "residual_by_theta_bin.csv", theta_rows)
    _write_csv(out_dir / "residual_by_branch.csv", branch_rows)
    _write_csv(out_dir / "support_manifest.csv", support_rows)
    if export_logits:
        logits_array = (
            np.stack(gps_logits_rows, axis=0).astype(np.float32)
            if gps_logits_rows
            else np.empty((0, num_beams), dtype=np.float32)
        )
        np.save(out_dir / "gps_logits.npy", logits_array)
        _write_csv(out_dir / "gps_logits_index.csv", gps_logits_index_rows)
        if export_probs:
            probs = np.exp(logits_array - logits_array.max(axis=-1, keepdims=True)) if logits_array.size else logits_array
            if probs.size:
                probs = probs / probs.sum(axis=-1, keepdims=True).clip(min=1e-12)
            np.save(out_dir / "gps_prior_probs.npy", probs.astype(np.float32))
    if bool(output_cfg.get("write_config_snapshot", True)):
        dump_config(cfg_dict, out_dir / "resolved_config.yaml")
    metadata = {
        "workflow": "mmw_town_gps_adapter_v2",
        "dataset_type": dataset_type,
        "label_space": selected_label_space,
        "beam_label_space": _dominant_value(mapping.label_space for mapping in scene_mappings.values()),
        "beam_label_mapping_fingerprints": {
            scene: mapping.fingerprint for scene, mapping in scene_mappings.items()
        },
        "output_dir": str(out_dir),
        "num_beams": num_beams,
        "scenes": [asdict(scene) for scene in scenes],
        "protocols": list(protocols),
        "ablations": ablations,
        "mapping": {scene: mapping.metadata() for scene, mapping in scene_mappings.items()},
        "branch_metadata": branch_metadata,
        "loss_metadata": loss_metadata,
        "run_notes": run_notes,
        "standard_artifacts": [
            "summary_overall.csv",
            "summary_by_scene.csv",
            "predictions.csv",
            "residual_by_theta_bin.csv",
            "residual_by_branch.csv",
            "run_metadata.json",
            "resolved_config.yaml",
        ],
        "gps_prior_export": {
            "save_logits": export_logits,
            "save_prior_probs": export_probs,
            "logits_path": str(out_dir / "gps_logits.npy") if export_logits else "",
            "logits_index_path": str(out_dir / "gps_logits_index.csv") if export_logits else "",
        },
    }
    if export_logits:
        metadata["standard_artifacts"].extend(["gps_logits.npy", "gps_logits_index.csv"])
        if export_probs:
            metadata["standard_artifacts"].append("gps_prior_probs.npy")
    _write_json(out_dir / "run_metadata.json", metadata)
    return {
        "output_dir": str(out_dir),
        "label_space": selected_label_space,
        "summary_by_scene_rows": len(summary_rows),
        "prediction_rows": len(prediction_rows),
        "standard_artifacts": metadata["standard_artifacts"],
    }


def select_support_samples(
    samples: list[MMWTownGpsV2Sample],
    adapt_cfg: Mapping[str, Any],
) -> tuple[list[MMWTownGpsV2Sample], list[MMWTownGpsV2Sample], dict[str, Any]]:
    mode = str(adapt_cfg.get("support_mode", "temporal_first"))
    count = _support_count(samples, support_ratio=adapt_cfg.get("support_ratio"), support_num=adapt_cfg.get("support_num"))
    if count <= 0:
        return [], list(samples), {"selection_mode": mode, "support_count": 0, "query_count": len(samples)}
    if mode == "random":
        rng = np.random.default_rng(int(adapt_cfg.get("seed", 42)))
        indices = sorted(rng.choice(len(samples), size=count, replace=False).tolist())
        support_set = set(indices)
    elif mode == "trajectory":
        ordered_groups: dict[str, list[int]] = {}
        for idx, sample in enumerate(samples):
            key = sample.branch_key or sample.metadata.get("contiguous_segment_id") or sample.sample_id
            ordered_groups.setdefault(str(key), []).append(idx)
        selected: list[int] = []
        for indices in ordered_groups.values():
            selected.extend(indices)
            if len(selected) >= count:
                break
        support_set = set(sorted(selected)[:count])
    elif mode == "temporal_first":
        ordered = sorted(range(len(samples)), key=lambda idx: (samples[idx].order_key, samples[idx].sample_id))
        support_set = set(ordered[:count])
    elif mode == "angle_coverage":
        support_set = angle_coverage_indices(
            samples,
            count,
            angle_getter=_theta_angle_degrees,
            include_extrema=True,
        )
    else:
        raise ValueError("adapt.support_mode must be one of temporal_first, angle_coverage, random, or trajectory.")
    support = [sample for idx, sample in enumerate(samples) if idx in support_set]
    query = [sample for idx, sample in enumerate(samples) if idx not in support_set]
    return support, query, {
        "selection_mode": mode,
        "seed": int(adapt_cfg.get("seed", 42)),
        "support_count": len(support),
        "query_count": len(query),
        "support_ratio": adapt_cfg.get("support_ratio"),
        "support_num": adapt_cfg.get("support_num"),
        "support_num_overrides_ratio": adapt_cfg.get("support_num") not in {None, ""},
        "support_angle_range_degrees": _theta_range(support),
        "query_angle_range_degrees": _theta_range(query),
    }


def _theta_angle_degrees(sample: MMWTownGpsV2Sample) -> float | None:
    value = float(sample.theta_degrees)
    return value if math.isfinite(value) else None


def _theta_range(samples: list[MMWTownGpsV2Sample]) -> list[float] | None:
    values = [float(sample.theta_degrees) for sample in samples if math.isfinite(float(sample.theta_degrees))]
    if not values:
        return None
    return [float(min(values)), float(max(values))]


def fit_adapter(
    samples: list[MMWTownGpsV2Sample],
    *,
    ablation: str,
    cfg: Mapping[str, Any],
    num_beams: int,
    class_prior: np.ndarray | None = None,
) -> AdapterFit:
    if not samples:
        return AdapterFit(sigma=float(_mapping(_mapping(cfg.get("model")).get("adapter")).get("sigma", 2.0)))
    if ablation == "backbone_only":
        return AdapterFit(criterion=0.0)
    adapt_cfg = _mapping(cfg.get("adapt"))
    grid_cfg = _mapping(adapt_cfg.get("grid"))
    psi_candidates = [float(item) for item in grid_cfg.get("psi_degrees", [0.0])]
    delta_candidates = [float(item) for item in grid_cfg.get("delta_beams", [0.0])]
    scale_candidates = [float(item) for item in grid_cfg.get("scale", [1.0])]
    flip_candidates = [str(item) for item in grid_cfg.get("flip", ["forward"])]
    if ablation == "adapter_v1":
        psi_candidates = [0.0]
        scale_candidates = [1.0]
    best = _grid_search(samples, psi_candidates, delta_candidates, scale_candidates, flip_candidates, num_beams=num_beams)
    if ablation in {"circular_affine_spline", "geo_plus_backbone", "branch_mixture_circular", "branch_mixture_circular_weighted"}:
        bins = int(_mapping(_mapping(cfg.get("model")).get("adapter")).get("num_bins", 16))
        best.spline_bins = _fit_spline_bins(samples, best, num_beams=num_beams, bins=bins)
    if ablation in {"branch_mixture_circular", "branch_mixture_circular_weighted"}:
        branch_cfg = _mapping(adapt_cfg.get("branch"))
        min_branch_support = int(branch_cfg.get("min_branch_support", 5))
        for branch_id in sorted({int(sample.branch_id) for sample in samples}):
            branch_samples = [sample for sample in samples if int(sample.branch_id) == branch_id]
            if len(branch_samples) < min_branch_support:
                best.branch_fallback[branch_id] = f"support_count={len(branch_samples)} < min_branch_support={min_branch_support}"
                continue
            branch_fit = _grid_search(
                branch_samples,
                [best.psi_degrees],
                delta_candidates,
                [best.scale],
                [best.flip],
                num_beams=num_beams,
            )
            branch_fit.spline_bins = _fit_spline_bins(branch_samples, branch_fit, num_beams=num_beams, bins=len(best.spline_bins) or 16)
            best.branch_params[branch_id] = branch_fit
    if class_prior is not None and class_prior.size:
        best.tau = float(max(0.5, min(2.0, np.mean(class_prior[class_prior > 0]) if np.any(class_prior > 0) else 1.0)))
    optimizer_cfg = _mapping(_mapping(cfg.get("adapt")).get("optimizer"))
    if bool(optimizer_cfg.get("enabled", False)):
        best = _optimize_adapter_fit(
            best,
            samples,
            num_beams=num_beams,
            steps=int(optimizer_cfg.get("steps", 20)),
            lr=float(optimizer_cfg.get("lr", 0.02)),
            smoothness_weight=float(_mapping(_mapping(cfg.get("model")).get("adapter")).get("smoothness_weight", 0.001)),
        )
    return best


def score_samples(
    samples: list[MMWTownGpsV2Sample],
    *,
    fit_samples: list[MMWTownGpsV2Sample],
    adapter_fit: AdapterFit,
    scaler: FeatureScaler,
    ablation: str,
    protocol: str,
    target: SceneSpec,
    label_space: str,
    mapping: BeamLabelMapping,
    num_beams: int,
    dba_delta: float,
    logits_sink: list[np.ndarray] | None = None,
    logits_index_sink: list[dict[str, Any]] | None = None,
    support_query_role: str = "query_test",
) -> list[dict[str, Any]]:
    _ = scaler.transform(samples)
    logits = []
    predicted = []
    topk_rows = []
    majority_logits = _majority_logits(fit_samples, num_beams=num_beams)
    for sample in samples:
        if ablation == "backbone_only":
            row_logits = majority_logits.copy()
        else:
            fit = adapter_fit.branch_params.get(int(sample.branch_id), adapter_fit)
            row_logits = _adapter_logits(sample, fit, num_beams=num_beams)
            if ablation == "geo_plus_backbone":
                row_logits = row_logits + 0.1 * majority_logits
        pred = int(np.argmax(row_logits))
        order = np.argsort(row_logits)[::-1][: min(5, num_beams)]
        logits.append(row_logits)
        predicted.append(pred)
        topk_rows.append([int(item) for item in order.tolist()])
    prediction_distances = [
        int(circular_beam_distance(int(pred), int(sample.label), num_beams=num_beams))
        for pred, sample in zip(predicted, samples)
    ]
    summary_dba = dba_from_circular_distances(prediction_distances, delta=dba_delta)
    rows = []
    for idx, sample in enumerate(samples):
        pred = predicted[idx]
        dist = int(prediction_distances[idx])
        signed = int(signed_circular_beam_residual(pred, int(sample.label), num_beams=num_beams))
        if logits_sink is not None:
            row_index = len(logits_sink)
            logits_sink.append(np.asarray(logits[idx], dtype=np.float32))
            if logits_index_sink is not None:
                logits_index_sink.append(
                    {
                        "row_index": row_index,
                        "sample_id": sample.sample_id,
                        "scene": target.slug,
                        "scene_name": target.name,
                        "scene_id": target.scene_id,
                        "split": sample.split,
                        "protocol": protocol,
                        "ablation": ablation,
                        "support_query_role": support_query_role,
                        "label_space": label_space,
                        "beam_label_space": mapping.label_space,
                        "beam_label_mapping_fingerprint": mapping.fingerprint,
                    }
                )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "scene": target.slug,
                "scene_name": target.name,
                "scene_id": target.scene_id,
                "split": sample.split,
                "protocol": protocol,
                "ablation": ablation,
                "label_space": label_space,
                "beam_label_space": mapping.label_space,
                "beam_label_mapping_fingerprint": mapping.fingerprint,
                "true_beam_raw": sample.label_raw,
                "true_beam": sample.label,
                "pred_beam": pred,
                "predicted_beam": pred,
                "final_predicted_beam": pred,
                "topk_predictions": json.dumps(topk_rows[idx]),
                "circular_error": dist,
                "signed_residual": signed,
                "theta_degrees": float(sample.theta_degrees),
                "E": float(sample.easting),
                "N": float(sample.northing),
                "branch_id": int(sample.branch_id),
                "branch_source": sample.branch_source,
                "support_query_role": support_query_role,
                "DBA_reference_delta": float(dba_delta),
                "summary_DBA": summary_dba,
            }
        )
    return rows


def _load_all_samples(
    *,
    scenes: list[SceneSpec],
    scene_mappings: dict[str, BeamLabelMapping],
    data_root: Path,
    dataset_type: str,
    data_cfg: Mapping[str, Any],
    split_tag: str,
    train_split: str,
    test_split: str,
    max_samples: int | None,
) -> dict[str, dict[str, list[MMWTownGpsV2Sample]]]:
    loaded: dict[str, dict[str, list[MMWTownGpsV2Sample]]] = {}
    for scene in scenes:
        loaded[scene.slug] = {}
        for split in (train_split, test_split):
            if dataset_type == "deepsense6g":
                scene_root = data_root / scene.slug
                csv_name = str(
                    data_cfg.get(
                        f"{split}_csv_name",
                        "train_seqs_RA_GPS_LIDAR.csv" if split == train_split else "test_seqs_RA_GPS_LIDAR.csv",
                    )
                )
                loaded[scene.slug][split] = load_deepsense_scene_samples(
                    scene_root / csv_name,
                    scene_root=scene_root,
                    scene=scene,
                    split=split,
                    mapping=scene_mappings[scene.slug],
                    num_beams=int(data_cfg.get("num_beams", 64)),
                    max_samples=max_samples,
                )
            else:
                path = data_root / "Prepared" / scene.slug / "splits" / split_tag / f"{split}.csv"
                loaded[scene.slug][split] = load_scene_samples(
                    path,
                    scene=scene,
                    split=split,
                    mapping=scene_mappings[scene.slug],
                    max_samples=max_samples,
                )
    return loaded


def load_scene_samples(
    path: str | Path,
    *,
    scene: SceneSpec,
    split: str,
    mapping: BeamLabelMapping,
    max_samples: int | None = None,
) -> list[MMWTownGpsV2Sample]:
    source = Path(path)
    if not source.exists():
        return []
    samples: list[MMWTownGpsV2Sample] = []
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = _sample_from_row(row, scene=scene, split=split, mapping=mapping)
            samples.append(sample)
            if max_samples is not None and len(samples) >= int(max_samples):
                break
    return samples


def load_deepsense_scene_samples(
    path: str | Path,
    *,
    scene_root: Path,
    scene: SceneSpec,
    split: str,
    mapping: BeamLabelMapping,
    num_beams: int,
    max_samples: int | None = None,
) -> list[MMWTownGpsV2Sample]:
    source = Path(path)
    if not source.exists():
        return []
    samples: list[MMWTownGpsV2Sample] = []
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            sample = _deepsense_sample_from_row(
                row,
                row_idx=row_idx,
                scene_root=scene_root,
                scene=scene,
                split=split,
                mapping=mapping,
                num_beams=int(num_beams),
            )
            if sample is None:
                continue
            samples.append(sample)
            if max_samples is not None and len(samples) >= int(max_samples):
                break
    return samples


def _deepsense_sample_from_row(
    row: Mapping[str, Any],
    *,
    row_idx: int,
    scene_root: Path,
    scene: SceneSpec,
    split: str,
    mapping: BeamLabelMapping,
    num_beams: int,
) -> MMWTownGpsV2Sample | None:
    raw_label = _deepsense_raw_label(row, scene_root=scene_root, num_beams=int(num_beams))
    if raw_label < 0:
        return None
    mapped_label = mapping.map_label(raw_label)
    current = _deepsense_relative_xy(row, scene_root=scene_root, gps_index=8)
    if current is None:
        return None
    previous = _deepsense_relative_xy(row, scene_root=scene_root, gps_index=7)
    easting, northing = current
    theta = math.degrees(math.atan2(northing, easting))
    rel_range = math.sqrt(easting * easting + northing * northing)
    heading = 0.0
    speed = 0.0
    heading_source = "default_zero"
    speed_source = "default_zero"
    if previous is not None:
        dx = easting - previous[0]
        dy = northing - previous[1]
        speed = math.sqrt(dx * dx + dy * dy)
        if speed > 1e-8:
            heading = math.degrees(math.atan2(dy, dx))
            heading_source = "gps_delta"
        speed_source = "gps_delta"
    future_beam_path = str(row.get("future_beam1") or "")
    seq_index = _float_value(row.get("seq_index"), row_idx)
    sample_id = f"{scene.slug}:{split}:{row_idx}:{Path(future_beam_path).stem or row_idx}"
    branch_key = str(row.get("seq_index") or "")
    return MMWTownGpsV2Sample(
        sample_id=sample_id,
        scene=scene.slug,
        scene_name=scene.name,
        scene_id=int(scene.scene_id),
        split=str(split),
        label_raw=int(raw_label),
        label=int(mapped_label),
        order_key=float(seq_index) + float(row_idx) * 1e-6,
        theta_degrees=float(theta),
        easting=float(easting),
        northing=float(northing),
        log_range=float(math.log1p(max(rel_range, 0.0))),
        heading_degrees=float(heading),
        speed=float(speed),
        branch_source="trajectory_id" if branch_key else "pseudo",
        branch_key=branch_key,
        heading_source=heading_source,
        speed_source=speed_source,
        mapping_fingerprint=mapping.fingerprint,
        metadata={
            "dataset_type": "deepsense6g",
            "row_index": int(row_idx),
            "seq_index": row.get("seq_index", ""),
            "target_beam_path": future_beam_path,
            "gps_path": row.get("gps8", ""),
            "bs_gps_path": row.get("bs_gps8", ""),
            "source_geometry_available": True,
            "beam_label_space": mapping.label_space,
        },
    )


def _deepsense_raw_label(row: Mapping[str, Any], *, scene_root: Path, num_beams: int) -> int:
    direct = _int_value(
        row.get("future_beam_label1"),
        _int_value(row.get("target_beam"), _int_value(row.get("beam_label"), -100)),
    )
    if direct >= 0:
        return int(direct)
    rel_path = str(row.get("future_beam1") or "")
    if not rel_path:
        return -100
    power = read_beam_power_vector(joined_resource(scene_root, rel_path), num_classes=int(num_beams))
    if power is None:
        return -100
    return int(np.argmax(power))


def _deepsense_relative_xy(row: Mapping[str, Any], *, scene_root: Path, gps_index: int) -> tuple[float, float] | None:
    gps_path = str(row.get(f"gps{int(gps_index)}") or "")
    bs_path = str(row.get(f"bs_gps{int(gps_index)}") or "")
    if not gps_path or not bs_path:
        return None
    try:
        ue_latlon = read_gps_latlon(scene_root, gps_path)
        bs_latlon = read_gps_latlon(scene_root, bs_path)
        ue_x, ue_y = latlon_to_utm_xy(float(ue_latlon[0]), float(ue_latlon[1]))
        bs_x, bs_y = latlon_to_utm_xy(float(bs_latlon[0]), float(bs_latlon[1]))
    except Exception:
        return None
    return float(ue_x - bs_x), float(ue_y - bs_y)


def _sample_from_row(
    row: Mapping[str, Any],
    *,
    scene: SceneSpec,
    split: str,
    mapping: BeamLabelMapping,
) -> MMWTownGpsV2Sample:
    geometry = _last_geometry(row)
    raw_label = _int_value(row.get("future_beam_label1"), _int_value(row.get("beam_label"), -100))
    mapped_label = mapping.map_label(raw_label) if raw_label >= 0 else raw_label
    easting = _float_value(geometry.get("relative_x"), _float_value(geometry.get("local_x"), 0.0))
    northing = _float_value(geometry.get("relative_y"), _float_value(geometry.get("local_y"), 0.0))
    theta = _float_value(geometry.get("relative_azimuth"), math.degrees(math.atan2(northing, easting)))
    rel_range = _float_value(geometry.get("relative_range"), math.sqrt(easting * easting + northing * northing))
    heading_value = geometry.get("heading_difference")
    heading_source = "heading_difference"
    if heading_value in {None, ""}:
        heading = 0.0
        heading_source = "default_zero"
    else:
        heading = _float_value(heading_value, 0.0)
    speed_value = geometry.get("relative_velocity")
    speed_source = "relative_velocity"
    if speed_value in {None, ""}:
        speed = 0.0
        speed_source = "default_zero"
    else:
        speed = _float_value(speed_value, 0.0)
    branch_source = "pseudo"
    branch_key = ""
    for key in ("branch_id", "trajectory_id"):
        if row.get(key) not in {None, ""}:
            branch_key = str(row.get(key))
            branch_source = key
            break
    if not branch_key and row.get("contiguous_segment_id") not in {None, ""}:
        branch_key = str(row.get("contiguous_segment_id"))
    order_key = _float_value(row.get("seq_index"), _float_value(row.get("window_start_frame"), len(str(row.get("sample_id", "")))))
    return MMWTownGpsV2Sample(
        sample_id=str(row.get("target_sample_id") or row.get("sample_id") or ""),
        scene=scene.slug,
        scene_name=scene.name,
        scene_id=int(scene.scene_id),
        split=str(split),
        label_raw=int(raw_label),
        label=int(mapped_label),
        order_key=float(order_key),
        theta_degrees=float(theta),
        easting=float(easting),
        northing=float(northing),
        log_range=float(math.log1p(max(rel_range, 0.0))),
        heading_degrees=float(heading),
        speed=float(speed),
        branch_source=branch_source,
        branch_key=branch_key,
        heading_source=heading_source,
        speed_source=speed_source,
        mapping_fingerprint=mapping.fingerprint,
        metadata={
            "agent": row.get("agent", ""),
            "contiguous_segment_id": row.get("contiguous_segment_id", ""),
            "source_geometry_available": bool(geometry.get("available", True)),
            "beam_label_space": mapping.label_space,
        },
    )


def _assign_branch_ids(
    samples: list[MMWTownGpsV2Sample],
    *,
    max_k: int,
    min_samples: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    by_scene: dict[str, list[MMWTownGpsV2Sample]] = {}
    for sample in samples:
        by_scene.setdefault(sample.scene, []).append(sample)
    for scene, rows in by_scene.items():
        explicit_keys = sorted({sample.branch_key for sample in rows if sample.branch_source in {"branch_id", "trajectory_id"}})
        if explicit_keys:
            mapping = {key: idx for idx, key in enumerate(explicit_keys)}
            for sample in rows:
                sample.branch_id = int(mapping.get(sample.branch_key, 0))
            metadata[scene] = {"source": "explicit", "branch_count": len(mapping), "field": rows[0].branch_source}
            continue
        if len(rows) < int(min_samples):
            for sample in rows:
                sample.branch_id = 0
            metadata[scene] = {"source": "pseudo", "branch_count": 1, "fallback_reason": "sample_count_below_min"}
            continue
        features = np.asarray(
            [
                [
                    sample.easting,
                    sample.northing,
                    math.sin(math.radians(sample.heading_degrees)),
                    math.cos(math.radians(sample.heading_degrees)),
                    sample.log_range,
                ]
                for sample in rows
            ],
            dtype=np.float64,
        )
        labels, info = _deterministic_kmeans_with_silhouette(features, max_k=max(1, int(max_k)))
        for sample, label in zip(rows, labels.tolist()):
            sample.branch_id = int(label)
        metadata[scene] = {"source": "pseudo", **info}
    return metadata


def _deterministic_kmeans_with_silhouette(features: np.ndarray, *, max_k: int) -> tuple[np.ndarray, dict[str, Any]]:
    if features.shape[0] < 2:
        return np.zeros(features.shape[0], dtype=np.int64), {"branch_count": 1, "fallback_reason": "insufficient_samples"}
    normalized = (features - features.mean(axis=0, keepdims=True)) / features.std(axis=0, keepdims=True).clip(min=1e-6)
    best_labels = np.zeros(features.shape[0], dtype=np.int64)
    best_score = -1.0
    best_k = 1
    upper = min(int(max_k), int(features.shape[0]))
    for k in range(1, upper + 1):
        labels = _kmeans(normalized, k=k)
        score = -0.01 if k == 1 else _silhouette(normalized, labels)
        if score > best_score:
            best_score = float(score)
            best_labels = labels
            best_k = k
    return best_labels, {"branch_count": int(best_k), "silhouette": float(best_score), "algorithm": "numpy_kmeans_v1"}


def _kmeans(features: np.ndarray, *, k: int, iterations: int = 30) -> np.ndarray:
    if k <= 1:
        return np.zeros(features.shape[0], dtype=np.int64)
    order = np.argsort(features[:, 0])
    init_idx = np.linspace(0, len(order) - 1, num=k).round().astype(int)
    centers = features[order[init_idx]].copy()
    labels = np.zeros(features.shape[0], dtype=np.int64)
    for _ in range(iterations):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1).astype(np.int64)
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centers[idx] = features[mask].mean(axis=0)
    return labels


def _silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(int(item) for item in labels.tolist()))
    if len(unique) <= 1:
        return -1.0
    dist = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=-1))
    scores = []
    for idx in range(features.shape[0]):
        same = labels == labels[idx]
        other_scores = [dist[idx, labels == other].mean() for other in unique if other != int(labels[idx]) and np.any(labels == other)]
        a = float(dist[idx, same].mean()) if int(same.sum()) > 1 else 0.0
        b = float(min(other_scores)) if other_scores else 0.0
        denom = max(a, b, 1e-8)
        scores.append((b - a) / denom)
    return float(np.mean(scores))


def _grid_search(
    samples: list[MMWTownGpsV2Sample],
    psi_candidates: list[float],
    delta_candidates: list[float],
    scale_candidates: list[float],
    flip_candidates: list[str],
    *,
    num_beams: int,
) -> AdapterFit:
    best = AdapterFit(criterion=float("inf"))
    for psi in psi_candidates:
        for delta in delta_candidates:
            for scale in scale_candidates:
                for flip in flip_candidates:
                    fit = AdapterFit(psi_degrees=psi, delta_beams=delta, scale=scale, flip=flip)
                    preds = [_adapter_center(sample, fit, num_beams=num_beams) for sample in samples]
                    errors = [circular_beam_distance(pred, sample.label, num_beams=num_beams) for pred, sample in zip(preds, samples)]
                    score = float(np.mean(errors)) if errors else float("inf")
                    if score < best.criterion:
                        fit.criterion = score
                        best = fit
    return best


def _fit_spline_bins(samples: list[MMWTownGpsV2Sample], fit: AdapterFit, *, num_beams: int, bins: int) -> list[float]:
    residuals: list[list[int]] = [[] for _ in range(int(bins))]
    for sample in samples:
        pred = _adapter_center(sample, fit, num_beams=num_beams, use_spline=False)
        residual = signed_circular_beam_residual(pred, sample.label, num_beams=num_beams)
        idx = int(math.floor((sample.theta_degrees % 360.0) / 360.0 * int(bins))) % int(bins)
        residuals[idx].append(int(residual))
    values = []
    for bucket in residuals:
        values.append(float(np.median(bucket)) if bucket else 0.0)
    return values


def _optimize_adapter_fit(
    fit: AdapterFit,
    samples: list[MMWTownGpsV2Sample],
    *,
    num_beams: int,
    steps: int,
    lr: float,
    smoothness_weight: float,
) -> AdapterFit:
    if not samples or int(steps) <= 0:
        return fit
    theta = torch.tensor([sample.theta_degrees for sample in samples], dtype=torch.float32)
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)
    psi = torch.tensor(float(fit.psi_degrees), dtype=torch.float32, requires_grad=True)
    delta = torch.tensor(float(fit.delta_beams), dtype=torch.float32, requires_grad=True)
    log_scale = torch.tensor(math.log(max(float(fit.scale), 1e-6)), dtype=torch.float32, requires_grad=True)
    flip_logit = torch.tensor(6.0 if fit.flip == "reverse" else -6.0, dtype=torch.float32, requires_grad=True)
    if fit.spline_bins:
        spline = torch.tensor(fit.spline_bins, dtype=torch.float32, requires_grad=True)
    else:
        spline = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([psi, delta, log_scale, flip_logit, spline], lr=float(lr))
    for _ in range(int(steps)):
        optimizer.zero_grad()
        residual = _torch_spline_residual(theta, spline)
        forward_center = (((theta + psi) % 360.0) / 360.0) * int(num_beams) * log_scale.exp() + delta + residual
        reverse_center = (((-theta + psi) % 360.0) / 360.0) * int(num_beams) * log_scale.exp() + delta + residual
        forward_logits = _torch_circular_logits(forward_center, num_beams=num_beams, sigma=fit.sigma, tau=fit.tau)
        reverse_logits = _torch_circular_logits(reverse_center, num_beams=num_beams, sigma=fit.sigma, tau=fit.tau)
        alpha = torch.sigmoid(flip_logit)
        probability = (1.0 - alpha) * F.softmax(forward_logits, dim=-1) + alpha * F.softmax(reverse_logits, dim=-1)
        loss = F.nll_loss(torch.log(probability.clamp_min(1e-12)), labels)
        if spline.numel() > 1:
            loss = loss + float(smoothness_weight) * (spline - spline.roll(shifts=-1)).pow(2).mean()
        loss.backward()
        optimizer.step()
    optimized = AdapterFit(
        psi_degrees=float(psi.detach().item()),
        delta_beams=float(delta.detach().item()),
        scale=float(log_scale.detach().exp().item()),
        flip="reverse" if float(torch.sigmoid(flip_logit).detach().item()) >= 0.5 else "forward",
        sigma=float(fit.sigma),
        tau=float(fit.tau),
        spline_bins=[float(item) for item in spline.detach().tolist()] if fit.spline_bins else list(fit.spline_bins),
        branch_params=fit.branch_params,
        branch_fallback=fit.branch_fallback,
        criterion=float(fit.criterion),
    )
    return optimized


def _torch_spline_residual(theta: torch.Tensor, spline: torch.Tensor) -> torch.Tensor:
    if spline.numel() <= 1:
        return torch.zeros_like(theta)
    bins = int(spline.numel())
    pos = (theta.remainder(360.0) / 360.0) * bins
    left = torch.floor(pos).to(torch.long).remainder(bins)
    right = (left + 1).remainder(bins)
    frac = pos - torch.floor(pos)
    return spline[left] * (1.0 - frac) + spline[right] * frac


def _torch_circular_logits(center: torch.Tensor, *, num_beams: int, sigma: float, tau: float) -> torch.Tensor:
    classes = torch.arange(int(num_beams), dtype=torch.float32).view(1, -1)
    center = center.view(-1, 1).remainder(float(num_beams))
    diff = torch.abs(classes - center)
    dist = torch.minimum(diff, torch.as_tensor(float(num_beams)) - diff)
    return -(dist**2) / (2.0 * max(float(sigma), 1e-6) ** 2 * max(float(tau), 1e-6))


def _adapter_logits(sample: MMWTownGpsV2Sample, fit: AdapterFit, *, num_beams: int) -> np.ndarray:
    center = _adapter_center(sample, fit, num_beams=num_beams)
    beams = np.arange(int(num_beams), dtype=np.float64)
    diff = np.abs(beams - float(center))
    dist = np.minimum(diff, int(num_beams) - diff)
    sigma = max(float(fit.sigma), 1e-6)
    tau = max(float(fit.tau), 1e-6)
    return (-(dist**2) / (2.0 * sigma * sigma * tau)).astype(np.float32)


def _adapter_center(
    sample: MMWTownGpsV2Sample,
    fit: AdapterFit,
    *,
    num_beams: int,
    use_spline: bool = True,
) -> int:
    theta = -sample.theta_degrees if fit.flip == "reverse" else sample.theta_degrees
    center = (((theta + fit.psi_degrees) % 360.0) / 360.0) * int(num_beams) * float(fit.scale)
    center = (center + float(fit.delta_beams)) % int(num_beams)
    if use_spline and fit.spline_bins:
        pos = ((sample.theta_degrees % 360.0) / 360.0) * len(fit.spline_bins)
        left = int(math.floor(pos)) % len(fit.spline_bins)
        right = (left + 1) % len(fit.spline_bins)
        frac = pos - math.floor(pos)
        center += float(fit.spline_bins[left]) * (1.0 - frac) + float(fit.spline_bins[right]) * frac
    return int(round(center)) % int(num_beams)


def _majority_logits(samples: list[MMWTownGpsV2Sample], *, num_beams: int) -> np.ndarray:
    hist = np.ones(int(num_beams), dtype=np.float64) * 1e-3
    for sample in samples:
        if 0 <= int(sample.label) < int(num_beams):
            hist[int(sample.label)] += 1.0
    return np.log(hist / hist.sum()).astype(np.float32)


def _summary_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    protocol: str,
    ablation: str,
    target: SceneSpec,
    source_scenes: list[str],
    label_space: str,
    mapping: BeamLabelMapping,
    protocol_note: str,
    support_info: Mapping[str, Any],
    scaler_metadata: Mapping[str, Any],
    adapter_fit: AdapterFit,
    dba_delta: float,
    num_beams: int,
) -> dict[str, Any]:
    metrics = _metrics_from_prediction_rows(rows, num_beams=num_beams, dba_delta=dba_delta)
    return {
        "protocol": protocol,
        "ablation": ablation,
        "scene": target.slug,
        "scene_name": target.name,
        "target_scene": target.slug,
        "source_scenes": json.dumps(source_scenes),
        "label_space": label_space,
        "beam_label_space": mapping.label_space,
        "beam_label_mapping_fingerprint": mapping.fingerprint,
        "protocol_note": protocol_note,
        "support_count": int(support_info.get("support_count", 0)),
        "query_count": int(support_info.get("query_count", 0)),
        "support_mode": support_info.get("selection_mode", "none"),
        "strict_eligibility": protocol != "within_scene_train",
        "upper_bound_protocol": protocol == "within_scene_train",
        "adapter_fit": json.dumps(adapter_fit.to_dict()),
        "scaler_metadata": json.dumps(dict(scaler_metadata)),
        **metrics,
    }


def _metrics_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    num_beams: int,
    dba_delta: float,
) -> dict[str, float | int]:
    sample_count = len(rows)
    distances = np.asarray([float(row.get("circular_error", 0.0)) for row in rows], dtype=np.float64)
    if distances.size == 0:
        return {
            "sample_count": 0,
            "valid_label_count": 0,
            "DBA": 0.0,
            "DBA_zero_ratio": 0.0,
            "mean_circular_error": 0.0,
            "median_circular_error": 0.0,
            "exact_acc": 0.0,
            "pm1_acc": 0.0,
            "pm2_acc": 0.0,
            "pm4_acc": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
        }
    target = [int(row["true_beam"]) for row in rows]
    topk = [_json_list_int(row.get("topk_predictions")) for row in rows]
    def top_hit(k: int) -> float:
        hits = 0
        for truth, preds in zip(target, topk):
            if int(truth) in [int(item) % int(num_beams) for item in preds[:k]]:
                hits += 1
        return float(hits / max(len(target), 1))

    return {
        "sample_count": sample_count,
        "valid_label_count": sample_count,
        "DBA": dba_from_circular_distances(distances, delta=dba_delta),
        "DBA_zero_ratio": dba_zero_ratio(distances),
        "mean_circular_error": float(np.mean(distances)),
        "median_circular_error": float(np.median(distances)),
        "exact_acc": float(np.mean(distances == 0)),
        "pm1_acc": float(np.mean(distances <= 1)),
        "pm2_acc": float(np.mean(distances <= 2)),
        "pm4_acc": float(np.mean(distances <= 4)),
        "top1": top_hit(1),
        "top3": top_hit(3),
        "top5": top_hit(5),
    }


def _overall_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["protocol"]), str(row["ablation"]), str(row["label_space"])), []).append(row)
    result = []
    metric_keys = [
        "DBA",
        "DBA_zero_ratio",
        "mean_circular_error",
        "median_circular_error",
        "exact_acc",
        "pm1_acc",
        "pm2_acc",
        "pm4_acc",
        "top1",
        "top3",
        "top5",
    ]
    for (protocol, ablation, label_space), rows in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "protocol": protocol,
            "ablation": ablation,
            "label_space": label_space,
            "scene_count": len(rows),
            "valid_label_count": sum(int(row.get("valid_label_count", 0)) for row in rows),
        }
        for key in metric_keys:
            payload[key] = float(np.mean([float(row.get(key, 0.0)) for row in rows])) if rows else 0.0
        result.append(payload)
    return result


def _residual_by_theta_rows(rows: list[dict[str, Any]], *, bins: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        bin_id = int(math.floor((float(row["theta_degrees"]) % 360.0) / 360.0 * int(bins))) % int(bins)
        key = (row["protocol"], row["ablation"], row["scene"], row["label_space"], bin_id)
        grouped.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        result.append(
            {
                "protocol": key[0],
                "ablation": key[1],
                "scene": key[2],
                "label_space": key[3],
                "theta_bin": key[4],
                "count": len(values),
                "mean_circular_error": float(np.mean([float(row["circular_error"]) for row in values])),
                "mean_signed_residual": float(np.mean([float(row["signed_residual"]) for row in values])),
            }
        )
    return result


def _residual_by_branch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["protocol"], row["ablation"], row["scene"], row["label_space"], row["branch_id"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        result.append(
            {
                "protocol": key[0],
                "ablation": key[1],
                "scene": key[2],
                "label_space": key[3],
                "branch_id": key[4],
                "count": len(values),
                "mean_circular_error": float(np.mean([float(row["circular_error"]) for row in values])),
                "mean_signed_residual": float(np.mean([float(row["signed_residual"]) for row in values])),
                "branch_source": values[0].get("branch_source", ""),
            }
        )
    return result


def _support_manifest_rows(
    support: list[MMWTownGpsV2Sample],
    query: list[MMWTownGpsV2Sample],
    *,
    protocol: str,
    target_scene: str,
    label_space: str,
    support_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for role, samples in (("support", support), ("query", query)):
        for sample in samples:
            rows.append(
                {
                    "protocol": protocol,
                    "target_scene": target_scene,
                    "label_space": label_space,
                    "beam_label_space": sample.metadata.get("beam_label_space", ""),
                    "beam_label_mapping_fingerprint": sample.mapping_fingerprint,
                    "role": role,
                    "sample_id": sample.sample_id,
                    "scene": sample.scene,
                    "split": sample.split,
                    "target_label": sample.label,
                    "order_key": sample.order_key,
                    "selection_mode": support_info.get("selection_mode", "none"),
                    "seed": support_info.get("seed", ""),
                }
            )
    return rows


def _scene_specs(data_cfg: Mapping[str, Any]) -> list[SceneSpec]:
    scenes_raw = data_cfg.get("scenes") or []
    result = []
    for idx, item in enumerate(scenes_raw):
        raw = dict(item)
        slug = str(raw.get("slug") or raw.get("scene") or raw.get("name"))
        result.append(SceneSpec(name=str(raw.get("name", slug)), slug=slug, scene_id=int(raw.get("scene_id", idx))))
    return result


def _resolve_label_space_config(data_cfg: Mapping[str, Any], label_space: str) -> dict[str, Any]:
    label_spaces = _mapping(data_cfg.get("label_spaces"))
    if label_space not in label_spaces:
        if label_space == "mapping_disabled":
            return {"enabled": False, "label_space": "raw", "num_classes": int(data_cfg.get("num_beams", 64))}
        raise ValueError(f"data.label_space must be one of {sorted(label_spaces)}, got {label_space}.")
    spec = _mapping(label_spaces[label_space])
    if not bool(spec.get("enabled", False)):
        return {"enabled": False, "label_space": "raw", "num_classes": int(data_cfg.get("num_beams", 64))}
    for key in ("mapping_file", "fallback_mapping_file"):
        path = spec.get(key)
        if path and Path(str(path)).exists():
            payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
            payload["enabled"] = True
            payload.setdefault("fit_source", str(path))
            return payload
    raise FileNotFoundError(
        "mapping_enabled requires an existing mapping_file or fallback_mapping_file. "
        f"Checked: {spec.get('mapping_file')}, {spec.get('fallback_mapping_file')}"
    )


def _last_geometry(row: Mapping[str, Any]) -> dict[str, Any]:
    geometry_items = []
    for key, value in row.items():
        if str(key).startswith("geometry") and value not in {None, ""}:
            suffix = str(key).replace("geometry", "")
            try:
                index = int(suffix)
            except ValueError:
                index = 0
            payload = _json_dict(value)
            if payload:
                geometry_items.append((index, payload))
    if geometry_items:
        geometry_items.sort(key=lambda item: item[0])
        return dict(geometry_items[-1][1])
    return _json_dict(row.get("relative_geometry_json"))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _support_count(samples: list[Any], *, support_ratio: Any, support_num: Any) -> int:
    total = len(samples)
    explicit = _optional_int(support_num)
    if explicit is not None:
        return max(0, min(explicit, total))
    ratio = 0.0 if support_ratio in {None, ""} else float(support_ratio)
    return max(0, min(int(math.ceil(total * ratio)), total))


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in {None, ""}:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _json_list_int(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value in {None, ""}:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [int(item) for item in payload] if isinstance(payload, list) else []


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _int_value(value: Any, default: int) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _float_value(value: Any, default: float) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _standardize(value: float, mean: float, scale: float) -> float:
    return float((float(value) - float(mean)) / max(float(scale), 1e-8))


def _dominant_value(values: Iterable[Any]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        item = str(value or "")
        if item:
            counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get) if counts else ""


__all__ = [
    "FEATURE_NAMES",
    "FeatureScaler",
    "MMWTownGpsV2Sample",
    "SceneSpec",
    "fit_adapter",
    "load_scene_samples",
    "run_mmw_town_gps_v2",
    "select_support_samples",
]
