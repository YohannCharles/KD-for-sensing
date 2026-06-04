from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.evaluation.metrics import (
    circular_beam_distance,
    gps_good_bad_label,
    signed_circular_residual,
)


DEFAULT_RATIOS = (0.05, 0.10, 0.15, 0.20)
DEFAULT_SCENES = ("scenario31", "scenario32", "scenario33", "scenario34")
FALLBACK_PRIOR_SOURCE = "fallback_gaussian_from_top1"
UNAVAILABLE_PRIOR_SOURCE = "unavailable_support_prior"
PREDICTION_LOGIT_NAMES = ("gps_logits.npy", "logits.npy", "pred_logits.npy")


@dataclass(frozen=True)
class GpsPriorRecord:
    scene: str
    sample_id: str
    row_index: int
    logits: np.ndarray
    probs: np.ndarray
    source: str


def ratio_tag(value: float | str) -> str:
    number = float(value)
    return f"r{int(round(number * 100)):02d}"


def circular_gaussian_prior_from_top1(
    top1,
    *,
    num_beams: int = 64,
    sigma: float = 2.0,
    logit_floor: float = -80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build circular Gaussian logits/probabilities from GPS top1 only."""
    beams = _positive_num_beams(num_beams)
    top = np.asarray(top1, dtype=np.int64) % beams
    classes = np.arange(beams, dtype=np.int64)
    diff = np.abs(np.expand_dims(top, axis=-1) - classes)
    dist = np.minimum(diff, beams - diff).astype(np.float64)
    sigma_value = max(float(sigma), 1e-6)
    logits = -(dist**2) / (2.0 * sigma_value * sigma_value)
    logits = np.maximum(logits, float(logit_floor))
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=-1, keepdims=True).clip(min=1e-12)
    if np.asarray(top1).ndim == 0:
        return logits.reshape(beams), probs.reshape(beams)
    return logits, probs


def load_gps_prior_artifacts(
    result_dir: str | Path,
    *,
    protocol: str | None = None,
    ablation: str | None = None,
    num_beams: int = 64,
    fallback_sigma: float = 2.0,
) -> dict[tuple[str, str], GpsPriorRecord]:
    """Load GPS logits/probs aligned to predictions, falling back to top1 priors."""
    base = Path(result_dir)
    predictions = _read_csv(base / "predictions.csv")
    selected = _select_prediction_rows(predictions, base, protocol=protocol, ablation=ablation)
    logits_path = _first_existing(base / name for name in PREDICTION_LOGIT_NAMES)
    index_path = _first_existing([base / "gps_logits_index.csv", base / "logits_index.csv"])
    probs_path = _first_existing([base / "gps_prior_probs.npy", base / "prior_probs.npy"])
    records: dict[tuple[str, str], GpsPriorRecord] = {}

    if logits_path is not None and index_path is not None:
        logits = np.load(logits_path)
        if logits.ndim != 2 or int(logits.shape[-1]) != int(num_beams):
            raise ValueError(f"{logits_path} must have shape [N, {num_beams}], got {tuple(logits.shape)}.")
        probs = np.load(probs_path) if probs_path is not None else _softmax_np(logits)
        index_rows = _read_csv(index_path)
        index_by_key: dict[tuple[str, str, str, str], int] = {}
        for index_row in index_rows:
            if protocol is not None and str(index_row.get("protocol") or "") != str(protocol):
                continue
            if ablation is not None and str(index_row.get("ablation") or "") != str(ablation):
                continue
            row_index = _int(index_row.get("row_index"), -1)
            if row_index < 0 or row_index >= int(logits.shape[0]):
                raise ValueError(f"{index_path} contains invalid row_index={row_index}.")
            key = (
                str(index_row.get("scene") or index_row.get("target_scene") or ""),
                str(index_row.get("sample_id") or ""),
                str(index_row.get("protocol") or ""),
                str(index_row.get("ablation") or ""),
            )
            if key in index_by_key:
                raise ValueError(f"{index_path} contains duplicate logits mapping for {key}.")
            index_by_key[key] = row_index
            simple_key = (key[0], key[1])
            if simple_key in records:
                raise ValueError(f"Duplicate selected GPS prior row for {simple_key}.")
            records[simple_key] = GpsPriorRecord(
                scene=key[0],
                sample_id=key[1],
                row_index=row_index,
                logits=np.asarray(logits[row_index], dtype=np.float32),
                probs=np.asarray(probs[row_index], dtype=np.float32),
                source=str(logits_path.name),
            )
        for row in selected:
            key = (
                str(row.get("scene") or row.get("target_scene") or ""),
                str(row.get("sample_id") or ""),
                str(row.get("protocol") or ""),
                str(row.get("ablation") or ""),
            )
            if key not in index_by_key:
                raise ValueError(f"Missing GPS logits mapping for {key}.")
        return records

    for row_index, row in enumerate(selected):
        scene = str(row.get("scene") or row.get("target_scene") or "")
        sample_id = str(row.get("sample_id") or "")
        key = (scene, sample_id)
        if key in records:
            raise ValueError(f"Duplicate selected GPS prior row for {key}.")
        pred = _int(row.get("final_predicted_beam"), _int(row.get("predicted_beam"), _int(row.get("pred_beam"), -1)))
        if pred < 0:
            continue
        logits, probs = circular_gaussian_prior_from_top1(pred, num_beams=num_beams, sigma=fallback_sigma)
        records[key] = GpsPriorRecord(
            scene=scene,
            sample_id=sample_id,
            row_index=row_index,
            logits=np.asarray(logits, dtype=np.float32),
            probs=np.asarray(probs, dtype=np.float32),
            source=FALLBACK_PRIOR_SOURCE,
        )
    return records


def inspect_residual_inputs(
    gps_sweep_root: str | Path,
    *,
    label_space: str = "mapping_disabled",
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> dict[str, Any]:
    root = Path(gps_sweep_root)
    ratio_reports: list[dict[str, Any]] = []
    for ratio in ratios:
        tag = ratio_tag(ratio)
        base = root / tag / label_space
        predictions_path = base / "predictions.csv"
        support_path = base / "support_manifest.csv"
        summary_path = base / "summary_overall.csv"
        scene_summary_path = base / "summary_by_scene.csv"
        residual_prob_path = base / "residual_probability"
        figures_path = base / "gps_prediction_trajectory"
        logits_path = _first_existing(base / name for name in PREDICTION_LOGIT_NAMES)
        index_path = _first_existing([base / "gps_logits_index.csv", base / "logits_index.csv"])
        fields = {
            "predictions": _csv_fields(predictions_path),
            "support_manifest": _csv_fields(support_path),
            "summary_overall": _csv_fields(summary_path),
            "summary_by_scene": _csv_fields(scene_summary_path),
        }
        required = {
            "scene",
            "sample_id",
            "true_beam",
            "predicted_beam",
            "topk_predictions",
            "circular_error",
            "theta_degrees",
            "E",
            "N",
            "support_query_role",
            "protocol",
            "ablation",
        }
        present = set(fields["predictions"])
        ratio_reports.append(
            {
                "ratio": float(ratio),
                "tag": tag,
                "base_dir": str(base),
                "summary_exists": summary_path.exists(),
                "predictions_exists": predictions_path.exists(),
                "support_manifest_exists": support_path.exists(),
                "per_scene_metrics_exists": scene_summary_path.exists(),
                "residual_probability_exists": residual_prob_path.exists(),
                "figures_exists": figures_path.exists(),
                "fields": fields,
                "missing_prediction_fields": sorted(required - present),
                "logits_path": str(logits_path) if logits_path is not None else "",
                "logits_index_path": str(index_path) if index_path is not None else "",
                "gps_prior_source_recommendation": "exported_logits" if logits_path and index_path else FALLBACK_PRIOR_SOURCE,
            }
        )
    return {
        "gps_sweep_root": str(root),
        "label_space": str(label_space),
        "ratios": ratio_reports,
    }


def build_residual_manifest(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    residual_cfg = _mapping(cfg.get("residual"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    tag = ratio_tag(ratio)
    gps_root = Path(str(data_cfg.get("gps_sweep_root", "outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep")))
    gps_dir = gps_root / tag / selected_label_space
    out_root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/deepsense6g_residual_fusion"))
    result_dir = out_root / tag / selected_label_space
    manifest_dir = result_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "residual_manifest.csv"
    metadata_path = manifest_dir / "residual_manifest_metadata.json"
    num_beams = int(data_cfg.get("num_beams", 64))
    protocol = str(residual_cfg.get("gps_protocol", "target_adapt_beambench"))
    ablation = _resolve_best_ablation(gps_dir, residual_cfg.get("gps_ablation", "best"))
    fallback_sigma = float(residual_cfg.get("gps_prior_fallback_sigma", 2.0))
    prediction_rows = _select_prediction_rows(
        _read_csv(gps_dir / "predictions.csv"),
        gps_dir,
        protocol=protocol,
        ablation=ablation,
    )
    prior_records = load_gps_prior_artifacts(
        gps_dir,
        protocol=protocol,
        ablation=ablation,
        num_beams=num_beams,
        fallback_sigma=fallback_sigma,
    )
    data_root = Path(str(data_cfg.get("data_root", "dataset/DeepSense6G")))
    csv_index = _DeepSenseCsvIndex(data_root, data_cfg)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    support_rows = [
        row
        for row in _read_csv(gps_dir / "support_manifest.csv")
        if str(row.get("protocol")) == protocol and str(row.get("label_space", selected_label_space)) == selected_label_space
        and str(row.get("role") or "support") == "support"
    ]
    for row in support_rows:
        scene = str(row.get("scene") or row.get("target_scene") or "")
        sample_id = str(row.get("sample_id") or "")
        raw = csv_index.lookup(sample_id)
        modality = _discover_modalities(raw, data_root=data_root, scene=scene)
        prior = prior_records.get((scene, sample_id))
        rows.append(
            _manifest_row(
                scene=scene,
                sample_id=sample_id,
                split=str(row.get("split") or "train"),
                support_query_role=str(row.get("role") or "support"),
                target_label=_int(row.get("target_label"), -100),
                pred_row=None,
                prior=prior,
                num_beams=num_beams,
                fallback_sigma=fallback_sigma,
                modality=modality,
            )
        )

    for pred_row in prediction_rows:
        scene = str(pred_row.get("scene") or "")
        sample_id = str(pred_row.get("sample_id") or "")
        raw = csv_index.lookup(sample_id)
        modality = _discover_modalities(raw, data_root=data_root, scene=scene)
        prior = prior_records.get((scene, sample_id))
        if prior is None:
            warnings.append(f"missing prior for {scene}:{sample_id}")
        rows.append(
            _manifest_row(
                scene=scene,
                sample_id=sample_id,
                split=str(pred_row.get("split") or "test"),
                support_query_role=str(pred_row.get("support_query_role") or "query_test"),
                target_label=_int(pred_row.get("true_beam"), _int(pred_row.get("target_label"), -100)),
                pred_row=pred_row,
                prior=prior,
                num_beams=num_beams,
                fallback_sigma=fallback_sigma,
                modality=modality,
            )
        )

    _write_csv(manifest_path, rows)
    availability = _modality_availability(rows)
    metadata = {
        "workflow": "deepsense6g_gps_residual_fusion_manifest",
        "gps_dir": str(gps_dir),
        "manifest_path": str(manifest_path),
        "label_space": selected_label_space,
        "support_ratio": ratio,
        "ratio_tag": tag,
        "num_beams": num_beams,
        "gps_protocol": protocol,
        "gps_ablation": ablation,
        "gps_prior_source": _dominant_prior_source(rows),
        "warnings": warnings,
        "modality_availability": availability,
        "support_count": sum(1 for row in rows if str(row.get("support_query_role")) == "support"),
        "query_count": sum(1 for row in rows if str(row.get("support_query_role")).startswith("query")),
        "query_label_used_for_training": False,
    }
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "row_count": len(rows),
        "support_count": metadata["support_count"],
        "query_count": metadata["query_count"],
        "modality_availability": availability,
        "warnings": warnings,
    }


class ResidualManifestDataset(Dataset):
    """Dataset backed by residual_manifest.csv, reading only enabled modalities."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        enabled_modalities: Sequence[str] | None = None,
        num_beams: int = 64,
        fallback_sigma: float = 2.0,
        include_query_labels: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = _read_csv(self.manifest_path)
        self.enabled_modalities = tuple(str(item) for item in (enabled_modalities or ("gps_context",)))
        self.num_beams = int(num_beams)
        self.fallback_sigma = float(fallback_sigma)
        self.include_query_labels = bool(include_query_labels)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        target_label = _int(row.get("target_label"), -100)
        role = str(row.get("support_query_role") or "")
        if role.startswith("query") and not self.include_query_labels:
            target_label = -100
        pred = _int(row.get("gps_pred_top1"), -1)
        if pred >= 0:
            prior_logits, prior_probs = circular_gaussian_prior_from_top1(
                pred,
                num_beams=self.num_beams,
                sigma=self.fallback_sigma,
            )
        else:
            prior_probs = np.full((self.num_beams,), 1.0 / float(self.num_beams), dtype=np.float32)
            prior_logits = np.log(prior_probs)
        item: dict[str, Any] = {
            "sample_id": row.get("sample_id", ""),
            "scene": row.get("scene", ""),
            "support_query_role": role,
            "target_label": torch.tensor(target_label, dtype=torch.long),
            "gps_error": torch.tensor(_float(row.get("gps_circular_error"), math.inf), dtype=torch.float32),
            "gps_prior_logits": torch.tensor(prior_logits, dtype=torch.float32),
            "gps_prior_probs": torch.tensor(prior_probs, dtype=torch.float32),
            "gps_context_features": torch.tensor(_context_features(row, self.num_beams), dtype=torch.float32),
        }
        if "image" in self.enabled_modalities:
            item["image"] = _load_optional_array(row.get("image_feature_path") or row.get("image_path"), modality="image")
        if "lidar" in self.enabled_modalities:
            item["lidar"] = _load_optional_array(row.get("lidar_feature_path") or row.get("lidar_path"), modality="lidar")
        if "radar" in self.enabled_modalities:
            item["radar"] = _load_optional_array(row.get("radar_feature_path") or row.get("radar_path"), modality="radar")
        return item


def collate_residual_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty residual batch.")
    batch: dict[str, Any] = {}
    tensor_keys = {
        key
        for item in items
        for key, value in item.items()
        if torch.is_tensor(value)
    }
    for key in sorted(tensor_keys):
        batch[key] = torch.stack([item[key] for item in items])  # type: ignore[index]
    for key in ("sample_id", "scene", "support_query_role"):
        batch[key] = [str(item.get(key, "")) for item in items]
    return batch


def _manifest_row(
    *,
    scene: str,
    sample_id: str,
    split: str,
    support_query_role: str,
    target_label: int,
    pred_row: Mapping[str, Any] | None,
    prior: GpsPriorRecord | None,
    num_beams: int,
    fallback_sigma: float,
    modality: Mapping[str, str],
) -> dict[str, Any]:
    topk = _parse_topk(pred_row.get("topk_predictions") if pred_row else "")
    pred = _int(pred_row.get("final_predicted_beam") if pred_row else None, _int(pred_row.get("predicted_beam") if pred_row else None, -1))
    if pred < 0 and prior is not None:
        pred = int(np.argmax(prior.logits))
        topk = [int(item) for item in np.argsort(prior.logits)[::-1][:5].tolist()]
    error = _int(pred_row.get("circular_error") if pred_row else None, -1)
    signed = _int(pred_row.get("signed_residual") if pred_row else None, -999)
    if pred >= 0 and target_label >= 0 and (error < 0 or signed == -999):
        error = int(circular_beam_distance(pred, target_label, num_beams=num_beams))
        signed = int(signed_circular_residual(target_label, pred, num_beams=num_beams))
    good, bad = gps_good_bad_label(error if error >= 0 else math.inf, threshold=4.0)
    if prior is not None:
        probs = prior.probs
        prior_source = prior.source
        prior_available = True
        prior_row_index = prior.row_index
    elif pred >= 0:
        _, probs = circular_gaussian_prior_from_top1(pred, num_beams=num_beams, sigma=fallback_sigma)
        prior_source = FALLBACK_PRIOR_SOURCE
        prior_available = True
        prior_row_index = -1
    else:
        probs = np.full((num_beams,), 1.0 / float(num_beams), dtype=np.float32)
        prior_source = UNAVAILABLE_PRIOR_SOURCE
        prior_available = False
        prior_row_index = -1
    peak = float(np.max(probs)) if probs.size else 0.0
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()) if probs.size else 0.0
    mean = float((np.arange(num_beams, dtype=np.float64) * probs).sum()) if probs.size else 0.0
    frame_id = _frame_id_from_sample_id(sample_id)
    theta = _float(pred_row.get("theta_degrees") if pred_row else None, 0.0)
    easting = _float(pred_row.get("E") if pred_row else None, 0.0)
    northing = _float(pred_row.get("N") if pred_row else None, 0.0)
    rel_range = float(math.sqrt(easting * easting + northing * northing))
    return {
        "scene": scene,
        "sample_id": sample_id,
        "timestamp": frame_id,
        "frame_id": frame_id,
        "split": split,
        "support_query_role": support_query_role,
        "target_label": int(target_label),
        "gps_pred_top1": pred,
        "gps_pred_top3": json.dumps(topk[:3]),
        "gps_pred_top5": json.dumps(topk[:5]),
        "gps_topk_predictions": json.dumps(topk),
        "gps_circular_error": error if error >= 0 else "",
        "gps_signed_residual": signed if signed != -999 else "",
        "gps_is_good_error_lt4": bool(good),
        "gps_is_bad_error_ge4": bool(bad),
        "theta_degrees": theta,
        "range": rel_range,
        "E": easting,
        "N": northing,
        "heading_degrees": _float(pred_row.get("heading_degrees") if pred_row else None, 0.0),
        "gps_context_features": json.dumps(_context_features_dict(theta, easting, northing, pred, peak, entropy)),
        "gps_prior_row_index": prior_row_index,
        "gps_prior_peak_prob": peak,
        "gps_prior_entropy": entropy,
        "gps_prior_mean_beam": mean,
        "gps_prior_available": bool(prior_available),
        "gps_prior_source": prior_source,
        "gps_protocol": pred_row.get("protocol", "") if pred_row else "",
        "gps_ablation": pred_row.get("ablation", "") if pred_row else "",
        "image_path": modality.get("image_path", ""),
        "lidar_path": modality.get("lidar_path", ""),
        "radar_path": modality.get("radar_path", ""),
        "image_feature_path": modality.get("image_feature_path", ""),
        "lidar_feature_path": modality.get("lidar_feature_path", ""),
        "radar_feature_path": modality.get("radar_feature_path", ""),
    }


class _DeepSenseCsvIndex:
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
            path = self.data_root / scene / csv_name
            self.cache[key] = _read_csv(path)
        rows = self.cache.get(key, [])
        if row_idx < 0 or row_idx >= len(rows):
            return None
        return rows[row_idx]


def _discover_modalities(raw: Mapping[str, Any] | None, *, data_root: Path, scene: str) -> dict[str, str]:
    if raw is None:
        return {
            "image_path": "",
            "lidar_path": "",
            "radar_path": "",
            "image_feature_path": "",
            "lidar_feature_path": "",
            "radar_feature_path": "",
        }
    scene_root = data_root / scene
    image = _resolve_resource(scene_root, raw.get("camera8") or raw.get("image_path") or "")
    lidar = _resolve_resource(scene_root, raw.get("lidar8") or raw.get("lidar_path") or "")
    radar = _resolve_resource(scene_root, raw.get("radar8") or raw.get("radar_path") or "")
    return {
        "image_path": image if _exists(image) else "",
        "lidar_path": lidar if _exists(lidar) else "",
        "radar_path": radar if _exists(radar) else "",
        "image_feature_path": image if Path(image).suffix.lower() in {".npy", ".npz", ".pt", ".csv", ".parquet"} else "",
        "lidar_feature_path": lidar if Path(lidar).suffix.lower() in {".npy", ".npz", ".pt", ".csv", ".parquet"} else "",
        "radar_feature_path": radar if Path(radar).suffix.lower() in {".npy", ".npz", ".pt", ".csv", ".parquet"} else "",
    }


def _resolve_resource(scene_root: Path, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = joined_resource(scene_root, raw)
    return str(path)


def _exists(value: str) -> bool:
    return bool(value) and Path(value).exists()


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


def _context_features(row: Mapping[str, Any], num_beams: int) -> list[float]:
    pred = _int(row.get("gps_pred_top1"), -1)
    return list(
        _context_features_dict(
            _float(row.get("theta_degrees"), 0.0),
            _float(row.get("E"), 0.0),
            _float(row.get("N"), 0.0),
            pred,
            _float(row.get("gps_prior_peak_prob"), 0.0),
            _float(row.get("gps_prior_entropy"), 0.0),
            num_beams=num_beams,
        ).values()
    )


def _context_features_dict(
    theta_degrees: float,
    easting: float,
    northing: float,
    pred: int,
    prior_peak: float,
    prior_entropy: float,
    *,
    num_beams: int = 64,
) -> dict[str, float]:
    theta = math.radians(float(theta_degrees))
    pred_angle = (float(pred if pred >= 0 else 0) / float(num_beams)) * 2.0 * math.pi
    rel_range = math.sqrt(float(easting) * float(easting) + float(northing) * float(northing))
    return {
        "E": float(easting),
        "N": float(northing),
        "sin_theta": math.sin(theta),
        "cos_theta": math.cos(theta),
        "log_range": math.log1p(max(rel_range, 0.0)),
        "pred_beam_sin": math.sin(pred_angle),
        "pred_beam_cos": math.cos(pred_angle),
        "prior_peak": float(prior_peak),
        "prior_entropy": float(prior_entropy),
    }


def _load_optional_array(path_value: Any, *, modality: str) -> torch.Tensor:
    path = Path(str(path_value or ""))
    if not path.exists():
        raise FileNotFoundError(f"{modality} modality is enabled but path is missing: {path_value}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        first_key = sorted(payload.files)[0]
        array = payload[first_key]
    elif suffix == ".pt":
        loaded = torch.load(path, map_location="cpu")
        return loaded if torch.is_tensor(loaded) else torch.as_tensor(loaded, dtype=torch.float32)
    else:
        raise ValueError(f"{modality} modality path must be a precomputed array for residual Dataset: {path}")
    return torch.as_tensor(array, dtype=torch.float32)


def _select_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    base: Path,
    *,
    protocol: str | None,
    ablation: str | None,
) -> list[dict[str, Any]]:
    selected_protocol = str(protocol or "target_adapt_beambench")
    selected_ablation = _resolve_best_ablation(base, ablation or "best")
    selected = [
        dict(row)
        for row in rows
        if str(row.get("protocol")) == selected_protocol and str(row.get("ablation")) == selected_ablation
    ]
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for row in selected:
        key = (str(row.get("scene") or ""), str(row.get("sample_id") or ""))
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate selected prediction mappings: {duplicates[:5]}")
    return selected


def _resolve_best_ablation(base: Path, requested: Any) -> str:
    value = str(requested or "best")
    if value != "best":
        return value
    rows = [
        row
        for row in _read_csv(base / "summary_overall.csv")
        if str(row.get("protocol")) == "target_adapt_beambench"
    ]
    if not rows:
        return "branch_mixture_circular"
    best = max(rows, key=lambda row: _float(row.get("DBA"), -1.0))
    return str(best.get("ablation") or "branch_mixture_circular")


def _modality_availability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality in ("image", "lidar", "radar"):
        path_key = f"{modality}_path"
        feature_key = f"{modality}_feature_path"
        path_count = sum(1 for row in rows if str(row.get(path_key) or ""))
        feature_count = sum(1 for row in rows if str(row.get(feature_key) or ""))
        result[modality] = {
            "path_count": path_count,
            "feature_count": feature_count,
            "available": path_count > 0 or feature_count > 0,
        }
    return result


def _dominant_prior_source(rows: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("gps_prior_source") or "")
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


def _parse_topk(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed]


def _csv_fields(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=-1, keepdims=True).clip(min=1e-12)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int(value: Any, default: int) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _positive_num_beams(num_beams: int) -> int:
    beams = int(num_beams)
    if beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}.")
    return beams


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value
