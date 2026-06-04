from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.baselines.gps_window.adapter import (
    ALLOWED_PREDICTION_FIELDS,
    guard_no_target_oracle,
    load_beam_power_vectors,
    load_samples_from_csv,
    split_csv_path,
)
from kd_sensing.baselines.gps_window.geometry import (
    angle_to_beam,
    beam_score_kernel,
    circular_beam_distance,
    topk_neighbors,
)
from kd_sensing.baselines.gps_window.predictors import build_calibration_state, predict_sample
from kd_sensing.baselines.gps_window.support_split import split_calibration_support
from kd_sensing.baselines.gps_window.types import GpsWindowBaselineConfig, GpsWindowSample, normalize_scenarios
from kd_sensing.config.io import deep_merge
from kd_sensing.evaluation.metrics import beam_power_metrics, calculate_dba_score

ANCHOR_SOURCES = {"geometry_calibrated", "gps_neural_coarse"}
GPS_ANCHOR_REQUIRED_KEYS = (
    "gps_anchor_coarse_logits",
    "gps_anchor_center_beam",
    "gps_anchor_confidence",
    "gps_anchor_residual_anchor_beam",
)


@dataclass(frozen=True)
class GpsCoarseAnchorConfig:
    enabled: bool = False
    anchor_source: str = "geometry_calibrated"
    algorithm: str = "geometry_last"
    num_classes: int = 64
    group_size: int = 8
    horizon: int = 1
    beam_start_degrees: float = 0.0
    beam_direction: int = 1
    beam_offset: int = 0
    boresight_angle_degrees: float = 0.0
    auto_calibrate_boresight_angle: bool = False
    auto_calibrate_beam_mapping: bool = False
    auto_calibrate_beam_direction: bool = True
    calibration_mode: str = "source"
    calibration_split: str = "source"
    selection_split: str | None = None
    evaluation_split: str | None = None
    support_samples: int = 0
    calibration_holdout_fraction: float = 0.0
    calibration_holdout_min_samples: int = 0
    calibration_holdout_strategy: str = "tail"
    score_width: float = 2.0
    score_temperature: float = 1.0
    neighbor_top_k: int = 5
    neighbor_group_top_k: int = 3
    confidence: float = 1.0
    confidence_floor: float = 0.05
    confidence_from_score_margin: bool = True
    confidence_temperature: float = 1.0
    low_coverage_confidence: float = 0.25
    emit_beam_scores: bool = True
    write_predictions: bool = True
    write_metrics: bool = True
    output_dir: str = "outputs/gps_coarse_anchor"
    hidden_size: int = 64
    dropout: float = 0.1
    loss_weight: float = 1.0
    beam_auxiliary: bool = False
    beam_auxiliary_weight: float = 0.0
    label_smoothing: float = 0.0
    angle_lookup_k: int = 1
    used_fields: tuple[str, ...] = ALLOWED_PREDICTION_FIELDS
    artifact_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GpsCoarseAnchorConfig":
        raw = dict(payload or {})
        if "source" in raw and "anchor_source" not in raw:
            raw["anchor_source"] = raw.pop("source")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {key: value for key, value in raw.items() if key in allowed}
        if "used_fields" in values:
            values["used_fields"] = tuple(str(item) for item in values["used_fields"])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    @property
    def num_groups(self) -> int:
        return int(self.num_classes) // int(self.group_size)

    def validate(self) -> None:
        source = str(self.anchor_source).strip().lower()
        if source not in ANCHOR_SOURCES:
            raise ValueError(f"GPS coarse anchor anchor_source must be one of {sorted(ANCHOR_SOURCES)}, got {self.anchor_source}.")
        if int(self.num_classes) <= 0:
            raise ValueError(f"GPS coarse anchor num_classes must be positive, got {self.num_classes}.")
        if int(self.group_size) <= 0:
            raise ValueError(f"GPS coarse anchor group_size must be positive, got {self.group_size}.")
        if int(self.num_classes) % int(self.group_size) != 0:
            raise ValueError(
                f"GPS coarse anchor num_classes ({self.num_classes}) must be divisible by group_size ({self.group_size})."
            )
        if int(self.horizon) <= 0:
            raise ValueError(f"GPS coarse anchor horizon must be positive, got {self.horizon}.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["used_fields"] = list(self.used_fields)
        return payload


@dataclass(frozen=True)
class GpsCoarseAnchor:
    coarse_logits: torch.Tensor
    center_beam: torch.Tensor
    confidence: torch.Tensor
    residual_anchor_beam: torch.Tensor
    beam_scores: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, cfg: GpsCoarseAnchorConfig) -> None:
        expected_groups = int(cfg.num_groups)
        if self.coarse_logits.ndim != 3 or int(self.coarse_logits.shape[-1]) != expected_groups:
            raise ValueError(
                "coarse_logits must have shape [B, H, G] with "
                f"G=num_classes//group_size={expected_groups}, got {tuple(self.coarse_logits.shape)}."
            )
        shape = tuple(self.coarse_logits.shape[:2])
        for name, tensor in (
            ("center_beam", self.center_beam),
            ("confidence", self.confidence),
            ("residual_anchor_beam", self.residual_anchor_beam),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")
        if self.beam_scores is not None and tuple(self.beam_scores.shape) != (*shape, int(cfg.num_classes)):
            raise ValueError(
                f"beam_scores must have shape [B, H, C]={(*shape, int(cfg.num_classes))}, "
                f"got {tuple(self.beam_scores.shape)}."
            )

    def to_model_output(self) -> dict[str, Any]:
        return {
            "gps_anchor": self,
            "gps_anchor_coarse_logits": self.coarse_logits,
            "gps_anchor_center_beam": self.center_beam,
            "gps_anchor_beam_scores": self.beam_scores,
            "gps_anchor_confidence": self.confidence,
            "gps_anchor_residual_anchor_beam": self.residual_anchor_beam,
            "gps_anchor_metadata": self.metadata,
        }


class GpsCoarseHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        *,
        num_classes: int = 64,
        group_size: int = 8,
        hidden_size: int = 64,
        dropout: float = 0.1,
        beam_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        cfg = GpsCoarseAnchorConfig(
            enabled=True,
            anchor_source="gps_neural_coarse",
            num_classes=num_classes,
            group_size=group_size,
        )
        self.num_classes = int(cfg.num_classes)
        self.group_size = int(cfg.group_size)
        self.num_groups = int(cfg.num_groups)
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_size)),
            nn.Linear(int(input_size), int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.coarse_head = nn.Linear(int(hidden_size), self.num_groups)
        self.confidence_head = nn.Linear(int(hidden_size), 1)
        self.beam_head = nn.Linear(int(hidden_size), self.num_classes) if beam_auxiliary else None

    def forward(self, features: torch.Tensor) -> GpsCoarseAnchor:
        if features.ndim != 3:
            raise ValueError(f"GPS coarse head features must have shape [B, H, D], got {tuple(features.shape)}.")
        hidden = self.net(features)
        coarse_logits = self.coarse_head(hidden)
        coarse_group = coarse_logits.argmax(dim=-1)
        center = coarse_group * self.group_size + self.group_size // 2
        center = center.remainder(self.num_classes).to(torch.long)
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)
        beam_scores = self.beam_head(hidden) if self.beam_head is not None else None
        metadata = {
            "anchor_source": "gps_neural_coarse",
            "uses_neural_network": True,
            "num_classes": self.num_classes,
            "group_size": self.group_size,
        }
        return GpsCoarseAnchor(
            coarse_logits=coarse_logits,
            center_beam=center,
            confidence=confidence,
            residual_anchor_beam=center,
            beam_scores=beam_scores,
            metadata=metadata,
        )


def build_geometry_anchor(
    samples: list[GpsWindowSample],
    cfg: GpsCoarseAnchorConfig,
    *,
    calibration_samples: list[GpsWindowSample] | None = None,
    calibration_split: str | None = None,
    selection_split: str | None = None,
    evaluation_split: str | None = None,
) -> GpsCoarseAnchor:
    window_cfg = _window_cfg_from_anchor_cfg(cfg)
    calibration = build_calibration_state(list(calibration_samples or []), window_cfg)
    predictions = [predict_sample(sample, window_cfg, calibration) for sample in samples]
    if predictions:
        beam_scores = torch.stack([item.scores for item in predictions], dim=0)
    else:
        beam_scores = torch.empty(0, int(cfg.horizon), int(cfg.num_classes), dtype=torch.float32)
    center = torch.tensor([list(item.center_beams[: int(cfg.horizon)]) for item in predictions], dtype=torch.long)
    if center.ndim == 1:
        center = center.unsqueeze(1)
    coarse_logits = beam_scores_to_coarse_logits(beam_scores, group_size=cfg.group_size)
    confidence = anchor_confidence_from_scores(
        beam_scores,
        gps_coverage=[item.gps_coverage for item in predictions],
        cfg=cfg,
    )
    metadata = geometry_anchor_metadata(
        samples,
        predictions,
        cfg=cfg,
        calibration_state=calibration.to_dict(),
        calibration_split=calibration_split,
        selection_split=selection_split,
        evaluation_split=evaluation_split,
        calibration_sample_count=len(calibration_samples or []),
    )
    anchor = GpsCoarseAnchor(
        coarse_logits=coarse_logits,
        center_beam=center,
        confidence=confidence,
        residual_anchor_beam=center,
        beam_scores=beam_scores if cfg.emit_beam_scores else None,
        metadata=metadata,
    )
    anchor.validate(cfg)
    return anchor


def build_anchor_from_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    cfg: GpsCoarseAnchorConfig,
    *,
    scenario: str = "",
    split: str = "target_test",
) -> GpsCoarseAnchor:
    samples = [
        _sample_from_prediction_row(row, scenario=scenario, split=split, cfg=cfg)
        for row in rows
    ]
    return build_geometry_anchor(samples, cfg, calibration_samples=[])


def gps_anchor_tensors_from_batch(
    batch: Mapping[str, Any],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    missing = [key for key in GPS_ANCHOR_REQUIRED_KEYS if key not in batch]
    if missing:
        raise ValueError(
            "gps_anchor.enabled=true requires GPS anchor batch fields: "
            f"missing {missing}."
        )
    result: dict[str, torch.Tensor] = {}
    for source_key, target_key, dtype in (
        ("gps_anchor_coarse_logits", "gps_anchor_coarse_logits", torch.float32),
        ("gps_anchor_center_beam", "gps_anchor_center_beam", torch.long),
        ("gps_anchor_confidence", "gps_anchor_confidence", torch.float32),
        ("gps_anchor_residual_anchor_beam", "gps_anchor_residual_anchor_beam", torch.long),
    ):
        tensor = batch[source_key]
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if tensor.ndim == 1 and target_key != "gps_anchor_coarse_logits":
            tensor = tensor.unsqueeze(1)
        if target_key == "gps_anchor_coarse_logits" and tensor.ndim != 3:
            raise ValueError(f"{source_key} must have shape [B, H, G], got {tuple(tensor.shape)}.")
        if target_key != "gps_anchor_coarse_logits" and tensor.ndim != 2:
            raise ValueError(f"{source_key} must have shape [B, H], got {tuple(tensor.shape)}.")
        result[target_key] = tensor[:, :num_pred] if tensor.ndim >= 2 else tensor
    return result


def beam_scores_to_coarse_logits(beam_scores: torch.Tensor, *, group_size: int) -> torch.Tensor:
    if beam_scores.ndim != 3:
        raise ValueError(f"beam_scores must have shape [B, H, C], got {tuple(beam_scores.shape)}.")
    group = int(group_size)
    classes = int(beam_scores.shape[-1])
    if classes % group != 0:
        raise ValueError(f"beam_scores class count ({classes}) must be divisible by group_size ({group}).")
    return torch.logsumexp(beam_scores.view(*beam_scores.shape[:2], classes // group, group), dim=-1)


def coarse_labels_from_beam(
    labels: torch.Tensor,
    *,
    num_classes: int,
    group_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    if int(num_classes) % int(group_size) != 0:
        raise ValueError(f"num_classes ({num_classes}) must be divisible by group_size ({group_size}).")
    valid = labels.ge(0) & labels.lt(int(num_classes))
    coarse = torch.div(labels.clamp_min(0), int(group_size), rounding_mode="floor")
    return torch.where(valid, coarse, torch.full_like(coarse, int(ignore_index)))


def compute_gps_coarse_anchor_loss(
    anchor: GpsCoarseAnchor | Mapping[str, Any],
    labels: torch.Tensor,
    cfg: GpsCoarseAnchorConfig,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    coarse_logits = anchor.coarse_logits if isinstance(anchor, GpsCoarseAnchor) else anchor["coarse_logits"]
    beam_scores = anchor.beam_scores if isinstance(anchor, GpsCoarseAnchor) else anchor.get("beam_scores")
    coarse_target = coarse_labels_from_beam(
        labels.to(device=coarse_logits.device),
        num_classes=cfg.num_classes,
        group_size=cfg.group_size,
        ignore_index=ignore_index,
    )
    coarse_loss = F.cross_entropy(
        coarse_logits.reshape(-1, coarse_logits.shape[-1]),
        coarse_target.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=float(cfg.label_smoothing),
    )
    total = float(cfg.loss_weight) * coarse_loss
    diagnostics = {
        "gps_anchor/loss_coarse": float(coarse_loss.detach().cpu().item()),
        "gps_anchor/loss_weight": float(cfg.loss_weight),
    }
    if beam_scores is not None and float(cfg.beam_auxiliary_weight) > 0.0:
        beam_loss = F.cross_entropy(
            beam_scores.reshape(-1, beam_scores.shape[-1]),
            labels.to(device=beam_scores.device).reshape(-1),
            ignore_index=ignore_index,
        )
        total = total + float(cfg.beam_auxiliary_weight) * beam_loss
        diagnostics["gps_anchor/loss_beam_auxiliary"] = float(beam_loss.detach().cpu().item())
        diagnostics["gps_anchor/beam_auxiliary_weight"] = float(cfg.beam_auxiliary_weight)
    diagnostics["gps_anchor/loss_total"] = float(total.detach().cpu().item())
    return total, diagnostics


def geometry_anchor_metrics(
    anchor: GpsCoarseAnchor,
    labels: torch.Tensor,
    *,
    cfg: GpsCoarseAnchorConfig,
    beam_power_vectors: torch.Tensor | None = None,
) -> dict[str, Any]:
    labels = labels.to(dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    center = anchor.center_beam.detach().cpu().to(torch.long)
    coarse_pred = anchor.coarse_logits.detach().cpu().argmax(dim=-1)
    coarse_true = coarse_labels_from_beam(labels.cpu(), num_classes=cfg.num_classes, group_size=cfg.group_size)
    valid = labels.ge(0) & labels.lt(int(cfg.num_classes))
    center_top1 = center.eq(labels.cpu()) & valid.cpu()
    beam_scores = anchor.beam_scores.detach().cpu() if anchor.beam_scores is not None else None
    metric_scores = beam_scores if beam_scores is not None else _center_beam_scores(center, cfg=cfg)
    predicted = metric_scores.argmax(dim=-1).to(torch.long)
    final_top1 = predicted.eq(labels.cpu()) & valid.cpu()
    final_top3 = metric_scores.topk(k=min(3, int(cfg.num_classes)), dim=-1).indices
    final_top3_hit = final_top3.eq(labels.cpu().clamp_min(0).unsqueeze(-1)).any(dim=-1) & valid.cpu()
    coarse_valid = coarse_true.ne(-100)
    distances = []
    residuals = []
    for pred, truth, ok in zip(predicted.reshape(-1).tolist(), labels.cpu().reshape(-1).tolist(), valid.cpu().reshape(-1).tolist()):
        if not ok:
            continue
        distances.append(circular_beam_distance(int(pred), int(truth), num_classes=cfg.num_classes))
        residuals.append((int(truth) - int(pred)) % int(cfg.num_classes))
    confidence = anchor.confidence.detach().cpu()
    dba = calculate_dba_score(metric_scores, labels.cpu())
    metrics = {
        "sample_count": int(valid.any(dim=1).sum().item()) if valid.ndim == 2 else int(valid.sum().item()),
        "valid_label_count": int(valid.sum().item()),
        "coarse_accuracy": _masked_mean(coarse_pred.eq(coarse_true), coarse_valid),
        "center_beam_top1": _masked_mean(center_top1, valid.cpu()),
        "center_beam_top3": _masked_mean(_topk_center_hit(center, labels.cpu(), cfg=cfg, k=3) & valid.cpu(), valid.cpu()),
        "final_predicted_beam_top1": _masked_mean(final_top1, valid.cpu()),
        "final_predicted_beam_top3": _masked_mean(final_top3_hit, valid.cpu()),
        "dba_by_horizon": [float(item) for item in list(dba)],
        "dba_avg": float(sum(float(item) for item in list(dba)) / max(len(dba), 1)),
        "circular_beam_error_mean": float(sum(distances) / max(len(distances), 1)),
        "circular_beam_error_median": _median(distances),
        "confidence_mean": float(confidence.mean().item()) if confidence.numel() else 0.0,
        "confidence_min": float(confidence.min().item()) if confidence.numel() else 0.0,
        "confidence_max": float(confidence.max().item()) if confidence.numel() else 0.0,
        "residual_preview": residual_preview(residuals, cfg=cfg),
    }
    metrics.update(beam_power_metrics(predicted[:, 0], labels.cpu()[:, 0], beam_power_vectors))
    metrics["beam_power_available"] = bool(metrics.get("power_metrics_available", False))
    metrics["beam_power_unavailable_reason"] = metrics.get("power_metrics_unavailable_reason")
    return metrics


def _center_beam_scores(center: torch.Tensor, *, cfg: GpsCoarseAnchorConfig) -> torch.Tensor:
    flat = center.detach().cpu().reshape(-1).tolist()
    scores = beam_score_kernel(
        [int(item) for item in flat],
        num_classes=cfg.num_classes,
        width=cfg.score_width,
        temperature=cfg.score_temperature,
        neighbor_top_k=cfg.neighbor_top_k,
    )
    return scores.reshape(*center.shape, int(cfg.num_classes))


def residual_preview(residuals: Iterable[int], *, cfg: GpsCoarseAnchorConfig) -> dict[str, Any]:
    values = [int(item) % int(cfg.num_classes) for item in residuals]
    hist = [0] * int(cfg.num_classes)
    for value in values:
        hist[value] += 1
    total = max(len(values), 1)
    entropy = 0.0
    for count in hist:
        if count:
            p = count / total
            entropy -= p * math.log(p)
    topk = max(1, int(cfg.neighbor_top_k))
    half = topk // 2
    in_neighborhood = 0
    for value in values:
        distance = min(value, int(cfg.num_classes) - value)
        if distance <= half:
            in_neighborhood += 1
    return {
        "diagnostic_only": True,
        "residual_histogram": hist,
        "residual_entropy": float(entropy),
        "topk_neighborhood_coverage": float(in_neighborhood / total),
        "count": len(values),
    }


def write_anchor_predictions_csv(
    path: str | Path,
    samples: list[GpsWindowSample],
    anchor: GpsCoarseAnchor,
    labels: torch.Tensor,
    *,
    cfg: GpsCoarseAnchorConfig,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    coarse_topk = torch.topk(anchor.coarse_logits.detach().cpu(), k=min(3, int(cfg.num_groups)), dim=-1).indices
    center = anchor.center_beam.detach().cpu()
    if anchor.beam_scores is not None:
        beam_scores = anchor.beam_scores.detach().cpu()
        predicted = beam_scores.argmax(dim=-1).to(torch.long)
        beam_topk = torch.topk(beam_scores, k=min(int(cfg.neighbor_top_k), int(cfg.num_classes)), dim=-1).indices
    else:
        predicted = center.to(torch.long)
        beam_topk = None
    confidence = anchor.confidence.detach().cpu()
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "scene",
                "split",
                "horizon",
                "true_beam",
                "pred_beam",
                "predicted_beam",
                "final_predicted_beam",
                "topk_predictions",
                "final_topk",
                "anchor_center_beam",
                "anchor_coarse_topk",
                "anchor_confidence",
                "gps_coverage",
                "anchor_source",
            ],
        )
        writer.writeheader()
        for row_idx, sample in enumerate(samples):
            for h_idx, truth in enumerate(labels.detach().cpu()[row_idx].tolist()):
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "scene": sample.scenario,
                        "split": sample.split,
                        "horizon": h_idx + 1,
                        "true_beam": int(truth),
                        "pred_beam": int(predicted[row_idx, h_idx].item()),
                        "predicted_beam": int(predicted[row_idx, h_idx].item()),
                        "final_predicted_beam": int(predicted[row_idx, h_idx].item()),
                        "topk_predictions": json.dumps(
                            [int(item) for item in beam_topk[row_idx, h_idx].tolist()]
                            if beam_topk is not None
                            else [int(predicted[row_idx, h_idx].item())]
                        ),
                        "final_topk": json.dumps(
                            [int(item) for item in beam_topk[row_idx, h_idx].tolist()]
                            if beam_topk is not None
                            else [int(predicted[row_idx, h_idx].item())]
                        ),
                        "anchor_center_beam": int(center[row_idx, h_idx].item()),
                        "anchor_coarse_topk": json.dumps([int(item) for item in coarse_topk[row_idx, h_idx].tolist()]),
                        "anchor_confidence": float(confidence[row_idx, h_idx].item()),
                        "gps_coverage": float(sample.gps_coverage),
                        "anchor_source": anchor.metadata.get("anchor_source", cfg.anchor_source),
                    }
                )
    return target


def run_gps_coarse_anchor_evaluation(
    cfg: dict[str, Any],
    *,
    scenes: list[str] | None = None,
    source_scenes: list[str] | None = None,
    target_scenes: list[str] | None = None,
    execute: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    anchor_cfg = GpsCoarseAnchorConfig.from_mapping(cfg.get("coarse_anchor") or cfg.get("gps_anchor") or {})
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    data_root = Path(data_cfg.get("data_root", "dataset/MMW/sunny"))
    split_tag = str(data_cfg.get("split_tag", "l5p3_group_safe"))
    targets = list(target_scenes or scenes or normalize_scenarios(data_cfg.get("target_scenes")) or normalize_scenarios(data_cfg.get("scenes")))
    sources = list(source_scenes or normalize_scenarios(data_cfg.get("source_scenes")))
    out_dir = Path(output_dir or cfg.get("output", {}).get("dir", anchor_cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "anchor_source": anchor_cfg.anchor_source,
        "anchor_algorithm": anchor_cfg.algorithm,
        "target_scenes": targets,
        "source_scenes": sources,
        "split_tag": split_tag,
        "calibration_mode": anchor_cfg.calibration_mode,
        "support_samples": int(anchor_cfg.support_samples),
        "calibration_holdout_fraction": float(anchor_cfg.calibration_holdout_fraction),
        "calibration_holdout_strategy": anchor_cfg.calibration_holdout_strategy,
        "output_dir": str(out_dir),
        "execute": bool(execute),
    }
    _write_json(out_dir / "gps_coarse_anchor_plan.json", plan)
    if not execute:
        return {"mode": "plan_only", "plan_path": str(out_dir / "gps_coarse_anchor_plan.json"), **plan}
    scene_results = []
    for target_scene in targets:
        scene_sources = [item for item in sources if item != target_scene] if sources else [item for item in targets if item != target_scene]
        window_cfg = _window_cfg_from_anchor_cfg(anchor_cfg)
        support_samples = _calibration_samples_for_anchor(
            data_root=data_root,
            target_scene=target_scene,
            source_scenes=scene_sources,
            split_tag=split_tag,
            cfg=anchor_cfg,
            window_cfg=window_cfg,
        )
        calibration_samples, selection_samples, calibration_info = _split_anchor_calibration_support(support_samples, anchor_cfg)
        eval_samples = load_samples_from_csv(
            split_csv_path(data_root, target_scene, "test", split_tag=split_tag),
            scenario=target_scene,
            split="target_test",
            cfg=window_cfg,
        )
        scene_dir = out_dir / str(target_scene)
        scene_dir.mkdir(parents=True, exist_ok=True)
        anchor = build_geometry_anchor(
            eval_samples,
            anchor_cfg,
            calibration_samples=calibration_samples,
            calibration_split=calibration_info["fit_split"],
            selection_split=calibration_info["selection_split"],
            evaluation_split="target_test",
        )
        labels = torch.tensor([list(sample.target_beams[: int(anchor_cfg.horizon)]) for sample in eval_samples], dtype=torch.long)
        metrics = geometry_anchor_metrics(
            anchor,
            labels,
            cfg=anchor_cfg,
            beam_power_vectors=_beam_power_tensor(eval_samples, data_root=data_root, cfg=anchor_cfg),
        )
        summary = {
            "target_scene": target_scene,
            "source_scenes": scene_sources,
            "target_seen_during_calibration": bool(anchor_cfg.calibration_mode == "target_adapt"),
            "split_protocol": "target_adapt_to_target_test"
            if anchor_cfg.calibration_mode == "target_adapt"
            else "source_to_target_test",
            "calibration_split": calibration_info["fit_split"],
            "selection_split": calibration_info["selection_split"],
            "calibration_holdout": calibration_info,
            "support_fit_sample_count": len(calibration_samples),
            "support_selection_sample_count": len(selection_samples),
            "evaluation_split": "target_test",
            "metrics": metrics,
            "metadata": anchor.metadata,
        }
        if anchor_cfg.write_metrics:
            _write_json(scene_dir / "metrics.json", summary)
        if anchor_cfg.write_predictions:
            write_anchor_predictions_csv(scene_dir / "predictions.csv", eval_samples, anchor, labels, cfg=anchor_cfg)
        scene_results.append(summary)
    overall = {"mode": "execute", "scene_results": scene_results, "summary_path": str(out_dir / "summary.json")}
    _write_json(out_dir / "summary.json", overall)
    return overall


def _beam_power_tensor(
    samples: list[GpsWindowSample],
    *,
    data_root: Path,
    cfg: GpsCoarseAnchorConfig,
) -> torch.Tensor | None:
    power_np = load_beam_power_vectors(
        samples,
        data_root=data_root,
        horizon=cfg.horizon,
        num_classes=cfg.num_classes,
    )
    return torch.from_numpy(power_np) if power_np is not None else None


def _calibration_samples_for_anchor(
    *,
    data_root: Path,
    target_scene: str,
    source_scenes: list[str],
    split_tag: str,
    cfg: GpsCoarseAnchorConfig,
    window_cfg: GpsWindowBaselineConfig,
) -> list[GpsWindowSample]:
    if str(cfg.calibration_mode).strip().lower() == "target_adapt":
        samples = load_samples_from_csv(
            split_csv_path(data_root, target_scene, "train", split_tag=split_tag),
            scenario=target_scene,
            split="target_adapt_support",
            cfg=window_cfg,
        )
        return samples[: int(cfg.support_samples)] if int(cfg.support_samples) > 0 else samples
    result: list[GpsWindowSample] = []
    for source in source_scenes:
        result.extend(
            load_samples_from_csv(
                split_csv_path(data_root, source, "train", split_tag=split_tag),
                scenario=source,
                split="source",
                cfg=window_cfg,
            )
        )
    return result


def _split_anchor_calibration_support(
    samples: list[GpsWindowSample],
    cfg: GpsCoarseAnchorConfig,
) -> tuple[list[GpsWindowSample], list[GpsWindowSample], dict[str, Any]]:
    return split_calibration_support(
        samples,
        calibration_mode=cfg.calibration_mode,
        holdout_fraction=cfg.calibration_holdout_fraction,
        holdout_min_samples=cfg.calibration_holdout_min_samples,
        holdout_strategy=cfg.calibration_holdout_strategy,
    )


def geometry_anchor_metadata(
    samples: list[GpsWindowSample],
    predictions: list[Any],
    *,
    cfg: GpsCoarseAnchorConfig,
    calibration_state: dict[str, Any],
    calibration_split: str | None,
    selection_split: str | None,
    evaluation_split: str | None,
    calibration_sample_count: int,
) -> dict[str, Any]:
    used_fields = tuple(cfg.used_fields or ALLOWED_PREDICTION_FIELDS)
    guard = guard_no_target_oracle(
        split=evaluation_split or (samples[0].split if samples else "target_test"),
        phase="gps_coarse_anchor",
        used_fields=used_fields,
        calibration_split=calibration_split,
    )
    fallback_count = sum(1 for item in predictions if getattr(item, "fallback_status", "none") != "none")
    coverage = [float(sample.gps_coverage) for sample in samples]
    return {
        "anchor_source": cfg.anchor_source,
        "anchor_algorithm": cfg.algorithm,
        "uses_neural_network": False,
        "used_fields": list(used_fields),
        "oracle_guard": guard,
        "used_target_test_for_calibration": bool(guard["used_target_test_for_calibration"]),
        "used_target_oracle_fields": list(guard["used_target_oracle_fields"]),
        "eligible_for_main_claim": bool(guard["eligible_for_main_claim"]),
        "ineligible_reason": guard["ineligible_reason"],
        "calibration_split": calibration_split,
        "selection_split": selection_split,
        "evaluation_split": evaluation_split,
        "calibration_sample_count": int(calibration_sample_count),
        "effective_boresight_angle_degrees": float(calibration_state.get("boresight_angle_degrees", cfg.boresight_angle_degrees)),
        "effective_beam_direction": int(calibration_state.get("beam_direction", cfg.beam_direction)),
        "effective_beam_offset": int(calibration_state.get("beam_offset", cfg.beam_offset)),
        "gps_coverage": float(sum(coverage) / max(len(coverage), 1)),
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(len(predictions), 1)),
        "missing_field_count": int(sum(1 for sample in samples if sample.gps_coverage <= 0.0)),
        "fallback_status": "used" if fallback_count else "none",
        "calibration_state": calibration_state,
        **dict(cfg.artifact_metadata),
    }


def anchor_confidence_from_scores(
    beam_scores: torch.Tensor,
    *,
    gps_coverage: Iterable[float],
    cfg: GpsCoarseAnchorConfig,
) -> torch.Tensor:
    if beam_scores.numel() == 0:
        return torch.empty(*beam_scores.shape[:2], dtype=torch.float32)
    if cfg.confidence_from_score_margin:
        top2 = torch.topk(beam_scores, k=min(2, int(beam_scores.shape[-1])), dim=-1).values
        margin = top2[..., 0] - top2[..., 1] if top2.shape[-1] > 1 else top2[..., 0]
        confidence = torch.sigmoid(margin / max(float(cfg.confidence_temperature), 1e-6))
    else:
        confidence = torch.full(beam_scores.shape[:2], float(cfg.confidence), dtype=torch.float32)
    coverage = torch.tensor(list(gps_coverage), dtype=torch.float32).view(-1, 1)
    confidence = confidence * coverage.clamp(0.0, 1.0)
    low = coverage.lt(1.0).expand_as(confidence)
    confidence = torch.where(low, torch.minimum(confidence, torch.full_like(confidence, float(cfg.low_coverage_confidence))), confidence)
    return confidence.clamp(float(cfg.confidence_floor), 1.0)


def _window_cfg_from_anchor_cfg(cfg: GpsCoarseAnchorConfig) -> GpsWindowBaselineConfig:
    return GpsWindowBaselineConfig.from_mapping(
        {
            "algorithm": cfg.algorithm,
            "num_classes": cfg.num_classes,
            "group_size": cfg.group_size,
            "horizon": cfg.horizon,
            "beam_start_degrees": cfg.beam_start_degrees,
            "beam_direction": cfg.beam_direction,
            "beam_offset": cfg.beam_offset,
            "boresight_angle_degrees": cfg.boresight_angle_degrees,
            "auto_calibrate_boresight_angle": cfg.auto_calibrate_boresight_angle,
            "auto_calibrate_beam_mapping": cfg.auto_calibrate_beam_mapping,
            "auto_calibrate_beam_direction": cfg.auto_calibrate_beam_direction,
            "score_width": cfg.score_width,
            "score_temperature": cfg.score_temperature,
            "neighbor_top_k": cfg.neighbor_top_k,
            "calibration_mode": cfg.calibration_mode,
            "support_samples": cfg.support_samples,
            "calibration_holdout_fraction": cfg.calibration_holdout_fraction,
            "calibration_holdout_min_samples": cfg.calibration_holdout_min_samples,
            "calibration_holdout_strategy": cfg.calibration_holdout_strategy,
            "angle_lookup_k": cfg.angle_lookup_k,
        }
    )


def _sample_from_prediction_row(
    row: Mapping[str, Any],
    *,
    scenario: str,
    split: str,
    cfg: GpsCoarseAnchorConfig,
) -> GpsWindowSample:
    geometry = _geometry_from_row(row)
    return GpsWindowSample(
        sample_id=str(row.get("sample_id") or row.get("target_sample_id") or ""),
        scenario=str(row.get("scene") or row.get("scenario") or row.get("scene_slug") or scenario),
        split=str(row.get("split") or split),
        history_geometry=tuple(geometry),
        target_beams=tuple(_target_beams_from_row(row, horizon=cfg.horizon)),
        metadata={"used_prediction_fields": list(cfg.used_fields)},
    )


def _geometry_from_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("geometry"), dict):
        return [dict(row["geometry"])]
    for key in ("relative_geometry_json", "gps_rel_polar_json", "GPS-Rel-Polar"):
        payload = _json_dict(row.get(key))
        if payload:
            payload.setdefault("available", True)
            return [payload]
    if row.get("relative_azimuth") is not None:
        return [{"available": True, "relative_azimuth": _float(row.get("relative_azimuth"), 0.0)}]
    if row.get("azimuth") is not None:
        return [{"available": True, "relative_azimuth": _float(row.get("azimuth"), 0.0)}]
    if row.get("relative_x") is not None and row.get("relative_y") is not None:
        return [{"available": True, "relative_x": _float(row.get("relative_x"), 0.0), "relative_y": _float(row.get("relative_y"), 0.0)}]
    return []


def _target_beams_from_row(row: Mapping[str, Any], *, horizon: int) -> list[int]:
    labels = []
    for idx in range(1, int(horizon) + 1):
        value = row.get(f"future_beam_label{idx}")
        if value is None and idx == 1:
            value = row.get("target_beam", row.get("beam_label"))
        labels.append(int(_float(value, -100)))
    return labels


def _topk_center_hit(center: torch.Tensor, labels: torch.Tensor, *, cfg: GpsCoarseAnchorConfig, k: int) -> torch.Tensor:
    rows = []
    for pred, truth in zip(center.reshape(-1).tolist(), labels.reshape(-1).tolist()):
        rows.append(int(truth) in topk_neighbors(int(pred), num_classes=cfg.num_classes, k=k))
    return torch.tensor(rows, dtype=torch.bool).view_as(center)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(dtype=torch.bool)
    if not torch.any(mask):
        return 0.0
    return float(values.to(dtype=torch.float32)[mask].mean().item())


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None or value == "":
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return target


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
