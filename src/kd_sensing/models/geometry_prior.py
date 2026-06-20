from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from kd_sensing.registries import HEADS


CONDITION_ID_FIELDS = (
    "condition",
    "predictive_condition_id",
    "gps_condition",
    "image_condition",
    "c_idx",
    "d_idx",
)


def label_space_fingerprint(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, Mapping):
        parts = [f"{key}={label_space_fingerprint(value[key])}" for key in sorted(value)]
        return "|".join(parts)
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


@HEADS.register("gps_geometry_prior")
class GpsGeometryPriorBranch(nn.Module):
    """Deterministic GPS-angle beam prior with auditable diagnostics."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_pred: int = 1,
        feature_mode: str = "relative_polar",
        prior_mode: str = "circular_gaussian",
        fallback_mode: str = "uniform",
        angle_index: int | None = None,
        angle_unit: str = "radians",
        sigma: float = 2.5,
        temperature: float = 1.0,
        top_k: int = 5,
        label_space: Any = "beam64",
        class_order: str | list[int] | tuple[int, ...] = "circular",
        mapping_fingerprint: str | None = None,
        beam_label_space: Any = None,
        beam_mapping_fingerprint: str | None = None,
        calibration_source: str = "config",
        normalization_artifact: str | None = None,
        history_window: int | None = None,
        gps_source_window: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.feature_mode = str(feature_mode)
        self.prior_mode = str(prior_mode)
        self.fallback_mode = str(fallback_mode)
        self.angle_index = angle_index
        self.angle_unit = str(angle_unit).lower()
        self.sigma = float(sigma)
        self.temperature = float(temperature)
        self.top_k = max(1, min(int(top_k), self.num_classes))
        self.label_space = label_space
        self.class_order = class_order
        self.mapping_fingerprint = mapping_fingerprint or label_space_fingerprint(class_order)
        self.beam_label_space = beam_label_space
        self.beam_mapping_fingerprint = beam_mapping_fingerprint
        self.calibration_source = str(calibration_source)
        self.normalization_artifact = normalization_artifact
        self.history_window = history_window
        self.gps_source_window = gps_source_window
        if self.num_classes <= 0:
            raise ValueError(f"gps_geometry_prior num_classes must be positive, got {num_classes}.")
        if self.sigma <= 0.0:
            raise ValueError(f"gps_geometry_prior sigma must be positive, got {sigma}.")
        if self.temperature <= 0.0:
            raise ValueError(f"gps_geometry_prior temperature must be positive, got {temperature}.")
        self._validate_label_space()

    def forward(
        self,
        gps_batch: torch.Tensor | None,
        *,
        target_time: int | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        gps_delay_steps: torch.Tensor | None = None,
        gps_counterfactual_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if gps_batch is None:
            raise ValueError("gps_geometry_prior requires gps_batch when geometry_prior.enabled=true.")
        if gps_batch.ndim != 3:
            raise ValueError(f"gps_geometry_prior expects gps_batch [B, T, F], got {tuple(gps_batch.shape)}.")
        steps = int(target_time or self.num_pred)
        gps = self._select_time(gps_batch.to(dtype=torch.float32), steps)
        angle, feature_valid = self._angle_from_gps(gps)
        availability = feature_valid
        if gps_valid_mask is not None:
            availability = availability & self._select_mask(gps_valid_mask, steps).to(dtype=torch.bool, device=gps.device)
        if gps_counterfactual_mask is not None:
            counter = self._select_mask(gps_counterfactual_mask, steps).to(dtype=torch.bool, device=gps.device)
            availability = availability & ~counter
        if gps_delay_steps is not None:
            delay = self._select_mask(gps_delay_steps, steps).to(dtype=torch.float32, device=gps.device)
        else:
            delay = torch.zeros_like(angle)

        logits = self._angle_logits(angle)
        if self.fallback_mode in {"uniform", "zero_logits", "unavailable"}:
            logits = torch.where(availability.unsqueeze(-1), logits, torch.zeros_like(logits))
        else:
            raise ValueError("gps_geometry_prior fallback_mode must be uniform, zero_logits, or unavailable.")
        distribution = torch.softmax(logits, dim=-1)
        entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(dim=-1)
        top_values, top_indices = torch.topk(distribution, k=self.top_k, dim=-1)
        unavailable_reason = "gps_missing_or_invalid" if bool((~availability).any().detach().cpu().item()) else ""
        return {
            "logits": logits,
            "distribution": distribution,
            "entropy": entropy,
            "topk_indices": top_indices,
            "topk_probabilities": top_values,
            "availability_mask": availability,
            "unavailable_reason": unavailable_reason,
            "delay_steps": delay,
            "metadata": self.training_strategy_metadata(),
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "gps_geometry_prior",
            "component_role": "geometry_prior",
            "architecture_category": "component_baseline",
            "feature_mode": self.feature_mode,
            "prior_mode": self.prior_mode,
            "fallback_mode": self.fallback_mode,
            "normalization_artifact": self.normalization_artifact,
            "calibration_source": self.calibration_source,
            "history_window": self.history_window,
            "gps_source_window": self.gps_source_window,
            "label_space": self.label_space,
            "beam_label_space": self.beam_label_space or self.label_space,
            "num_beams": self.num_classes,
            "num_classes": self.num_classes,
            "class_order": self.class_order if isinstance(self.class_order, str) else list(self.class_order),
            "mapping_fingerprint": self.mapping_fingerprint,
            "beam_mapping_fingerprint": self.beam_mapping_fingerprint or self.mapping_fingerprint,
            "consumes_reliability_metadata": True,
            "reliability_fields": ["gps_valid_mask", "gps_delay_steps", "gps_counterfactual_mask"],
        }

    def _validate_label_space(self) -> None:
        label_text = str(self.label_space)
        if label_text.startswith("beam"):
            suffix = label_text.replace("beam", "", 1)
            if suffix.isdigit() and int(suffix) != self.num_classes:
                raise ValueError(
                    "gps_geometry_prior label space mismatch: "
                    f"label_space={self.label_space!r} implies {int(suffix)} beams, "
                    f"but beam head num_classes={self.num_classes}."
                )
        beam_fingerprint = self.beam_mapping_fingerprint or self.mapping_fingerprint
        if beam_fingerprint and self.mapping_fingerprint and beam_fingerprint != self.mapping_fingerprint:
            raise ValueError(
                "gps_geometry_prior label-space mapping fingerprint mismatch: "
                f"prior={self.mapping_fingerprint!r}, beam_head={beam_fingerprint!r}."
            )
        if isinstance(self.class_order, str) and self.class_order not in {"circular", "linear"}:
            raise ValueError("gps_geometry_prior class_order must be 'circular', 'linear', or an explicit index list.")
        if isinstance(self.class_order, (list, tuple)) and len(self.class_order) != self.num_classes:
            raise ValueError(
                "gps_geometry_prior explicit class_order length must match num_classes, "
                f"got {len(self.class_order)} and {self.num_classes}."
            )

    def _angle_from_gps(self, gps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_dim = int(gps.shape[-1])
        mode = self.feature_mode.lower().replace("-", "_")
        finite = torch.isfinite(gps).all(dim=-1)
        if mode in {"relative_cartesian", "cartesian", "xy"}:
            if feature_dim < 2:
                raise ValueError("relative_cartesian geometry prior requires GPS feature dim >= 2.")
            x = gps[..., 0]
            y = gps[..., 1]
            angle = torch.atan2(y, x)
            valid = finite & (x.abs() + y.abs()).gt(1e-12)
            return angle, valid
        if mode in {"calibrated_angle", "angle"}:
            index = int(self.angle_index if self.angle_index is not None else 0)
        else:
            index = int(self.angle_index if self.angle_index is not None else (1 if feature_dim > 1 else 0))
        if index < 0:
            index = feature_dim + index
        if index < 0 or index >= feature_dim:
            raise ValueError(f"gps_geometry_prior angle_index={self.angle_index} is outside feature dim {feature_dim}.")
        angle = gps[..., index]
        if self.angle_unit in {"degree", "degrees", "deg"}:
            angle = angle * math.pi / 180.0
        elif self.angle_unit not in {"radian", "radians", "rad"}:
            raise ValueError("gps_geometry_prior angle_unit must be radians or degrees.")
        return angle, finite

    def _angle_logits(self, angle: torch.Tensor) -> torch.Tensor:
        centers = torch.arange(self.num_classes, dtype=angle.dtype, device=angle.device)
        centers = (centers / float(self.num_classes)) * (2.0 * math.pi) - math.pi
        delta = torch.atan2(torch.sin(angle.unsqueeze(-1) - centers), torch.cos(angle.unsqueeze(-1) - centers))
        if self.prior_mode in {"circular_gaussian", "gaussian", "angle_gaussian"}:
            logits = -0.5 * (delta / self.sigma).pow(2)
        elif self.prior_mode in {"cosine", "von_mises"}:
            logits = torch.cos(delta) / self.sigma
        else:
            raise ValueError("gps_geometry_prior prior_mode must be circular_gaussian or cosine.")
        return logits / self.temperature

    @staticmethod
    def _select_time(value: torch.Tensor, steps: int) -> torch.Tensor:
        if int(value.shape[1]) >= steps:
            return value[:, -steps:, ...]
        pad = value[:, :1, ...].expand(-1, steps - int(value.shape[1]), *([-1] * (value.ndim - 2)))
        return torch.cat([pad, value], dim=1)

    @staticmethod
    def _select_mask(value: torch.Tensor, steps: int) -> torch.Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(1)
        if value.ndim != 2:
            raise ValueError(f"geometry prior reliability mask must have shape [B, T] or [B], got {tuple(value.shape)}.")
        if int(value.shape[1]) >= steps:
            return value[:, -steps:]
        pad = value[:, :1].expand(-1, steps - int(value.shape[1]))
        return torch.cat([pad, value], dim=1)


@HEADS.register("geometry_prior_logit_fusion")
class GeometryPriorLogitFusion(nn.Module):
    """Logit-level assistive fusion for image/fusion logits and GPS geometry prior."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        mode: str = "assistive",
        prior_weight: float = 0.25,
        max_prior_weight: float = 0.45,
        min_image_weight: float = 0.55,
        entropy_weight: float = 0.5,
        delay_scale: float = 4.0,
        counterfactual_weight: float = 0.05,
        image_observability_threshold: float = 0.35,
        disagreement_threshold: float = 0.35,
        disagreement_weight: float = 0.5,
        allow_image_downweight_when_invalid: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = None if num_classes is None else int(num_classes)
        self.mode = str(mode)
        self.prior_weight = float(prior_weight)
        self.max_prior_weight = float(max_prior_weight)
        self.min_image_weight = float(min_image_weight)
        self.entropy_weight = float(entropy_weight)
        self.delay_scale = max(float(delay_scale), 1e-6)
        self.counterfactual_weight = float(counterfactual_weight)
        self.image_observability_threshold = float(image_observability_threshold)
        self.disagreement_threshold = float(disagreement_threshold)
        self.disagreement_weight = float(disagreement_weight)
        self.allow_image_downweight_when_invalid = bool(allow_image_downweight_when_invalid)
        if self.mode not in {"assistive", "replace_when_image_invalid", "additive"}:
            raise ValueError("geometry_prior_logit_fusion mode must be assistive, replace_when_image_invalid, or additive.")

    def forward(
        self,
        *,
        image_logits: torch.Tensor,
        prior_logits: torch.Tensor,
        prior_distribution: torch.Tensor | None = None,
        prior_availability_mask: torch.Tensor | None = None,
        image_valid_mask: torch.Tensor | None = None,
        image_observability_score: torch.Tensor | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        gps_delay_steps: torch.Tensor | None = None,
        gps_counterfactual_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if image_logits.shape != prior_logits.shape:
            raise ValueError(
                "geometry_prior_logit_fusion requires image and prior logits with identical shape, "
                f"got {tuple(image_logits.shape)} and {tuple(prior_logits.shape)}."
            )
        if image_logits.ndim != 3:
            raise ValueError(f"geometry_prior_logit_fusion expects [B, T, C] logits, got {tuple(image_logits.shape)}.")
        if self.num_classes is not None and int(image_logits.shape[-1]) != self.num_classes:
            raise ValueError(
                "geometry_prior_logit_fusion class dimension mismatch: "
                f"expected {self.num_classes}, got {int(image_logits.shape[-1])}."
            )
        image_prob = torch.softmax(image_logits, dim=-1)
        prior_prob = prior_distribution if prior_distribution is not None else torch.softmax(prior_logits, dim=-1)
        prior_entropy = _normalized_entropy(prior_prob)
        image_entropy = _normalized_entropy(image_prob)
        disagreement = 1.0 - (image_prob * prior_prob).sum(dim=-1).clamp(0.0, 1.0)

        availability = torch.ones(image_logits.shape[:2], dtype=torch.bool, device=image_logits.device)
        if prior_availability_mask is not None:
            availability = availability & _temporal_like(prior_availability_mask, image_logits).to(dtype=torch.bool)
        if gps_valid_mask is not None:
            availability = availability & _temporal_like(gps_valid_mask, image_logits).to(dtype=torch.bool)
        delay = (
            _temporal_like(gps_delay_steps, image_logits).to(dtype=image_logits.dtype)
            if gps_delay_steps is not None
            else torch.zeros_like(prior_entropy)
        )
        counter = (
            _temporal_like(gps_counterfactual_mask, image_logits).to(dtype=torch.bool)
            if gps_counterfactual_mask is not None
            else torch.zeros_like(availability)
        )

        prior_weight = torch.full_like(prior_entropy, self.prior_weight)
        prior_weight = prior_weight * torch.exp(-self.entropy_weight * prior_entropy)
        prior_weight = prior_weight * torch.exp(-delay / self.delay_scale)
        prior_weight = torch.where(disagreement > self.disagreement_threshold, prior_weight * self.disagreement_weight, prior_weight)
        prior_weight = torch.where(counter, prior_weight * self.counterfactual_weight, prior_weight)
        prior_weight = torch.where(availability, prior_weight, torch.zeros_like(prior_weight))
        prior_weight = prior_weight.clamp(0.0, self.max_prior_weight)

        image_available = torch.ones_like(availability)
        if image_valid_mask is not None:
            image_available = image_available & _temporal_like(image_valid_mask, image_logits).to(dtype=torch.bool)
        if image_observability_score is not None:
            obs = _temporal_like(image_observability_score, image_logits).to(dtype=image_logits.dtype)
            image_available = image_available & obs.ge(self.image_observability_threshold)
        else:
            obs = torch.ones_like(prior_weight)
        image_weight = 1.0 - prior_weight
        if self.mode == "assistive":
            image_weight = torch.maximum(image_weight, torch.full_like(image_weight, self.min_image_weight))
        if self.allow_image_downweight_when_invalid:
            image_weight = torch.where(image_available, image_weight, torch.minimum(image_weight, obs.clamp(0.0, 1.0)))
        total = (image_weight + prior_weight).clamp_min(1e-6)
        image_weight = image_weight / total
        prior_weight = prior_weight / total
        fused = image_weight.unsqueeze(-1) * image_logits + prior_weight.unsqueeze(-1) * prior_logits
        return {
            "logits": fused,
            "image_weight": image_weight,
            "prior_weight": prior_weight,
            "diagnostics": {
                "branch_weights": torch.stack([image_weight, prior_weight], dim=-1),
                "image_weight": image_weight,
                "geometry_prior_weight": prior_weight,
                "image_entropy": image_entropy,
                "geometry_prior_entropy": prior_entropy,
                "prior_image_disagreement": disagreement,
                "prior_availability_mask": availability,
                "gps_counterfactual_mask": counter,
                "gps_delay_steps": delay,
                "condition_id_consumed": False,
                "blocked_condition_fields": list(CONDITION_ID_FIELDS),
                "fusion_mode": self.mode,
            },
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "geometry_prior_logit_fusion",
            "component_role": "logit_fusion",
            "fusion_mode": self.mode,
            "consumes_reliability_metadata": True,
            "prior_weight": self.prior_weight,
            "max_prior_weight": self.max_prior_weight,
            "min_image_weight": self.min_image_weight,
            "condition_id_consumed": False,
            "forbidden_condition_fields": list(CONDITION_ID_FIELDS),
        }


@HEADS.register("safe_residual_beam_reranker")
class SafeResidualBeamReranker(nn.Module):
    """Anchor-safe candidate reranker with bounded residual logits."""

    source_names = ("anchor", "prior", "neighborhood", "teacher")

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        anchor_top_k: int = 8,
        prior_top_k: int = 8,
        teacher_top_k: int = 0,
        neighborhood_radius: int = 1,
        max_candidates: int | None = None,
        max_residual_scale: float = 0.35,
        residual_hidden_size: int = 24,
        anchor_confidence_threshold: float = 0.55,
        image_observability_threshold: float = 0.75,
        disagreement_threshold: float = 0.35,
        min_gate_confidence: float = 0.05,
        delay_scale: float = 4.0,
        gps_invalid_residual_scale: float = 0.25,
        fallback_on_gps_invalid: bool = False,
        fallback_policy: str = "no_regret",
        diagnostics_mode: str = "full",
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = None if num_classes is None else int(num_classes)
        self.anchor_top_k = max(1, int(anchor_top_k))
        self.prior_top_k = max(0, int(prior_top_k))
        self.teacher_top_k = max(0, int(teacher_top_k))
        self.neighborhood_radius = max(0, int(neighborhood_radius))
        self.max_candidates = None if max_candidates is None else max(1, int(max_candidates))
        self.max_residual_scale = float(max_residual_scale)
        self.anchor_confidence_threshold = float(anchor_confidence_threshold)
        self.image_observability_threshold = float(image_observability_threshold)
        self.disagreement_threshold = float(disagreement_threshold)
        self.min_gate_confidence = float(min_gate_confidence)
        self.delay_scale = max(float(delay_scale), 1e-6)
        self.gps_invalid_residual_scale = float(gps_invalid_residual_scale)
        self.fallback_on_gps_invalid = bool(fallback_on_gps_invalid)
        self.fallback_policy = str(fallback_policy)
        self.diagnostics_mode = str(diagnostics_mode)
        feature_dim = 3 + len(self.source_names)
        hidden = max(4, int(residual_hidden_size))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        if self.max_residual_scale < 0.0:
            raise ValueError("safe_residual_beam_reranker max_residual_scale must be non-negative.")
        if self.fallback_policy not in {"no_regret", "anchor", "never"}:
            raise ValueError("safe residual reranker fallback_policy must be no_regret, anchor, or never.")

    def forward(
        self,
        *,
        anchor_logits: torch.Tensor,
        geometry_prior_logits: torch.Tensor | None = None,
        prior_logits: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        image_observability_score: torch.Tensor | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        gps_delay_steps: torch.Tensor | None = None,
        gps_counterfactual_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if anchor_logits.ndim != 3:
            raise ValueError(f"safe_residual_beam_reranker expects anchor_logits [B, T, C], got {tuple(anchor_logits.shape)}.")
        prior = geometry_prior_logits if prior_logits is None else prior_logits
        for name, value in (("prior_logits", prior), ("teacher_logits", teacher_logits)):
            if value is not None and value.shape != anchor_logits.shape:
                raise ValueError(
                    f"safe_residual_beam_reranker requires {name} to match anchor logits shape, "
                    f"got {tuple(value.shape)} and {tuple(anchor_logits.shape)}."
                )
        num_classes = int(anchor_logits.shape[-1])
        if self.num_classes is not None and num_classes != self.num_classes:
            raise ValueError(
                "safe_residual_beam_reranker class dimension mismatch: "
                f"expected {self.num_classes}, got {num_classes}."
            )
        candidate_mask, source_mask = self._candidate_masks(anchor_logits, prior, teacher_logits)
        candidate_ids, candidate_source_mask = self._compact_candidates(anchor_logits, candidate_mask, source_mask)
        valid_candidates = candidate_ids.ge(0)
        candidate_features = self._candidate_features(anchor_logits, prior, teacher_logits, candidate_ids, candidate_source_mask)
        residual_scores = torch.tanh(self.residual_head(candidate_features).squeeze(-1))
        residual_scores = torch.where(valid_candidates, residual_scores, torch.zeros_like(residual_scores))
        gate = self._gate(
            anchor_logits,
            prior,
            image_observability_score=image_observability_score,
            gps_valid_mask=gps_valid_mask,
            gps_delay_steps=gps_delay_steps,
            gps_counterfactual_mask=gps_counterfactual_mask,
            candidate_mask=candidate_mask,
        )
        full_residual = torch.zeros_like(anchor_logits)
        bounded_residual = (residual_scores * gate["residual_scale"].unsqueeze(-1)).to(dtype=full_residual.dtype)
        scatter_ids = candidate_ids.clamp_min(0)
        full_residual.scatter_(2, scatter_ids, bounded_residual)
        rerank_logits = anchor_logits + full_residual
        final_logits = torch.where(gate["fallback_to_anchor"].unsqueeze(-1), anchor_logits, rerank_logits)
        anchor_top1 = anchor_logits.argmax(dim=-1)
        final_top1 = final_logits.argmax(dim=-1)
        selected_source_mask = _gather_source_mask(source_mask, final_top1)
        selected_source = selected_source_mask.to(dtype=torch.long).argmax(dim=-1)
        selected_source = torch.where(
            selected_source_mask.any(dim=-1),
            selected_source,
            torch.full_like(selected_source, -1),
        )
        diagnostics = {
            "anchor_logits": anchor_logits,
            "prior_logits": prior,
            "geometry_prior_logits": prior,
            "teacher_logits": teacher_logits,
            "rerank_logits": final_logits,
            "candidate_ids": candidate_ids,
            "candidate_source_mask": candidate_source_mask,
            "candidate_mask": candidate_mask,
            "candidate_count": candidate_mask.sum(dim=-1),
            "residual_scores": bounded_residual,
            "full_residual": full_residual,
            "residual_magnitude": full_residual.abs().amax(dim=-1),
            "selected_source_mask": selected_source_mask,
            "selected_source": selected_source,
            "anchor_top1": anchor_top1,
            "rerank_top1": final_top1,
            "changed_from_anchor": final_top1.ne(anchor_top1),
            "fallback_to_anchor": gate["fallback_to_anchor"],
            "fallback_reason_code": gate["fallback_reason_code"],
            "gate_confidence": gate["gate_confidence"],
            "residual_scale": gate["residual_scale"],
            "anchor_confidence": gate["anchor_confidence"],
            "prior_entropy": gate["prior_entropy"],
            "branch_disagreement": gate["branch_disagreement"],
            "gps_reliability": gate["gps_reliability"],
            "condition_id_consumed": False,
            "blocked_condition_fields": list(CONDITION_ID_FIELDS),
            "source_names": list(self.source_names),
            "diagnostics_mode": self.diagnostics_mode,
        }
        return {"logits": final_logits, "diagnostics": diagnostics}

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "safe_residual_beam_reranker",
            "component_role": "safe_residual_reranker",
            "architecture_category": "component_baseline",
            "anchor_top_k": self.anchor_top_k,
            "prior_top_k": self.prior_top_k,
            "teacher_top_k": self.teacher_top_k,
            "neighborhood_radius": self.neighborhood_radius,
            "max_candidates": self.max_candidates,
            "max_residual_scale": self.max_residual_scale,
            "fallback_policy": self.fallback_policy,
            "diagnostics_mode": self.diagnostics_mode,
            "candidate_sources": list(self.source_names),
            "consumes_reliability_metadata": True,
            "condition_id_consumed": False,
            "forbidden_condition_fields": list(CONDITION_ID_FIELDS),
        }

    def _candidate_masks(
        self,
        anchor_logits: torch.Tensor,
        prior_logits: torch.Tensor | None,
        teacher_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, classes = anchor_logits.shape
        candidate_mask = torch.zeros(batch, steps, classes, dtype=torch.bool, device=anchor_logits.device)
        source_mask = torch.zeros(batch, steps, classes, len(self.source_names), dtype=torch.bool, device=anchor_logits.device)
        anchor_k = min(self.anchor_top_k, classes)
        anchor_top = anchor_logits.topk(anchor_k, dim=-1).indices
        candidate_mask.scatter_(2, anchor_top, True)
        source_mask[..., 0].scatter_(2, anchor_top, True)
        if prior_logits is not None and self.prior_top_k > 0:
            prior_top = prior_logits.topk(min(self.prior_top_k, classes), dim=-1).indices
            candidate_mask.scatter_(2, prior_top, True)
            source_mask[..., 1].scatter_(2, prior_top, True)
        if self.neighborhood_radius > 0:
            center = anchor_logits.argmax(dim=-1)
            offsets = torch.arange(
                -self.neighborhood_radius,
                self.neighborhood_radius + 1,
                device=anchor_logits.device,
                dtype=torch.long,
            )
            neighborhood = (center.unsqueeze(-1) + offsets).remainder(classes)
            candidate_mask.scatter_(2, neighborhood, True)
            source_mask[..., 2].scatter_(2, neighborhood, True)
        if teacher_logits is not None and self.teacher_top_k > 0:
            teacher_top = teacher_logits.topk(min(self.teacher_top_k, classes), dim=-1).indices
            candidate_mask.scatter_(2, teacher_top, True)
            source_mask[..., 3].scatter_(2, teacher_top, True)
        return candidate_mask, source_mask

    def _compact_candidates(
        self,
        anchor_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, classes = candidate_mask.shape
        limit = min(self.max_candidates or classes, classes)
        ids = torch.full((batch, steps, limit), -1, dtype=torch.long, device=anchor_logits.device)
        sources = torch.zeros(batch, steps, limit, len(self.source_names), dtype=torch.bool, device=anchor_logits.device)
        for b in range(batch):
            for t in range(steps):
                selected = torch.nonzero(candidate_mask[b, t], as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                scores = anchor_logits[b, t, selected]
                order = torch.argsort(scores, descending=True)
                selected = selected[order][:limit]
                count = int(selected.numel())
                ids[b, t, :count] = selected
                sources[b, t, :count] = source_mask[b, t, selected]
        return ids, sources

    def _candidate_features(
        self,
        anchor_logits: torch.Tensor,
        prior_logits: torch.Tensor | None,
        teacher_logits: torch.Tensor | None,
        candidate_ids: torch.Tensor,
        candidate_source_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_ids = candidate_ids.clamp_min(0)
        anchor_scores = torch.gather(anchor_logits, 2, safe_ids)
        if prior_logits is None:
            prior_scores = torch.zeros_like(anchor_scores)
        else:
            prior_scores = torch.gather(prior_logits, 2, safe_ids)
        if teacher_logits is None:
            teacher_scores = torch.zeros_like(anchor_scores)
        else:
            teacher_scores = torch.gather(teacher_logits, 2, safe_ids)
        score_features = torch.stack([anchor_scores, prior_scores, teacher_scores], dim=-1)
        source_features = candidate_source_mask.to(dtype=anchor_logits.dtype)
        features = torch.cat([score_features, source_features], dim=-1)
        return torch.where(candidate_ids.unsqueeze(-1).ge(0), features, torch.zeros_like(features))

    def _gate(
        self,
        anchor_logits: torch.Tensor,
        prior_logits: torch.Tensor | None,
        *,
        image_observability_score: torch.Tensor | None,
        gps_valid_mask: torch.Tensor | None,
        gps_delay_steps: torch.Tensor | None,
        gps_counterfactual_mask: torch.Tensor | None,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor_prob = torch.softmax(anchor_logits, dim=-1)
        anchor_top2 = anchor_prob.topk(min(2, int(anchor_prob.shape[-1])), dim=-1).values
        anchor_conf = anchor_top2[..., 0] - (anchor_top2[..., 1] if anchor_top2.shape[-1] > 1 else 0.0)
        if prior_logits is None:
            prior_prob = torch.full_like(anchor_prob, 1.0 / float(anchor_prob.shape[-1]))
        else:
            prior_prob = torch.softmax(prior_logits, dim=-1)
        prior_entropy = _normalized_entropy(prior_prob)
        disagreement = 1.0 - (anchor_prob * prior_prob).sum(dim=-1).clamp(0.0, 1.0)
        if image_observability_score is None:
            observability = torch.ones_like(anchor_conf)
        else:
            observability = _temporal_like(image_observability_score, anchor_logits).to(dtype=anchor_logits.dtype)
        if gps_valid_mask is None:
            gps_valid = torch.ones_like(anchor_conf, dtype=torch.bool)
        else:
            gps_valid = _temporal_like(gps_valid_mask, anchor_logits).to(dtype=torch.bool)
        if gps_counterfactual_mask is None:
            counter = torch.zeros_like(gps_valid)
        else:
            counter = _temporal_like(gps_counterfactual_mask, anchor_logits).to(dtype=torch.bool)
        if gps_delay_steps is None:
            delay = torch.zeros_like(anchor_conf)
        else:
            delay = _temporal_like(gps_delay_steps, anchor_logits).to(dtype=anchor_logits.dtype)
        gps_reliability = gps_valid.to(dtype=anchor_logits.dtype) * (~counter).to(dtype=anchor_logits.dtype)
        gps_reliability = gps_reliability * torch.exp(-delay.clamp_min(0.0) / self.delay_scale)
        uncertainty_gain = 1.0 - anchor_conf.clamp(0.0, 1.0)
        observability_gain = 1.0 - observability.clamp(0.0, 1.0)
        prior_gain = 1.0 - prior_entropy.clamp(0.0, 1.0)
        gate_conf = (0.45 * uncertainty_gain + 0.25 * observability_gain + 0.30 * prior_gain) * gps_reliability
        gate_conf = gate_conf.clamp(0.0, 1.0)
        empty_candidates = candidate_mask.sum(dim=-1).eq(0)
        clean_anchor_protected = (
            anchor_conf.ge(self.anchor_confidence_threshold)
            & observability.ge(self.image_observability_threshold)
            & disagreement.ge(self.disagreement_threshold)
        )
        gps_unreliable = (~gps_valid) | counter
        low_confidence = gate_conf.lt(self.min_gate_confidence)
        fallback = torch.zeros_like(empty_candidates)
        reason = torch.zeros_like(anchor_conf, dtype=torch.long)
        if self.fallback_policy == "anchor":
            fallback = torch.ones_like(empty_candidates)
            reason = torch.full_like(reason, 5)
        elif self.fallback_policy == "no_regret":
            fallback = empty_candidates | clean_anchor_protected | low_confidence
            if self.fallback_on_gps_invalid:
                fallback = fallback | gps_unreliable
            reason = torch.where(empty_candidates, torch.full_like(reason, 1), reason)
            reason = torch.where(clean_anchor_protected & reason.eq(0), torch.full_like(reason, 2), reason)
            reason = torch.where(gps_unreliable & self.fallback_on_gps_invalid & reason.eq(0), torch.full_like(reason, 3), reason)
            reason = torch.where(low_confidence & reason.eq(0), torch.full_like(reason, 4), reason)
        residual_scale = self.max_residual_scale * gate_conf
        residual_scale = torch.where(gps_unreliable, residual_scale * self.gps_invalid_residual_scale, residual_scale)
        residual_scale = torch.where(fallback, torch.zeros_like(residual_scale), residual_scale)
        return {
            "fallback_to_anchor": fallback,
            "fallback_reason_code": reason,
            "gate_confidence": gate_conf,
            "residual_scale": residual_scale,
            "anchor_confidence": anchor_conf,
            "prior_entropy": prior_entropy,
            "branch_disagreement": disagreement,
            "gps_reliability": gps_reliability,
        }


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    denom = math.log(max(int(probabilities.shape[-1]), 2))
    return entropy / denom


def _temporal_like(value: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2:
        raise ValueError(f"reliability metadata must have shape [B, T] or [B], got {tuple(value.shape)}.")
    target_time = int(logits.shape[1])
    if int(value.shape[1]) >= target_time:
        return value[:, -target_time:].to(device=logits.device)
    pad = value[:, :1].expand(-1, target_time - int(value.shape[1]))
    return torch.cat([pad, value], dim=1).to(device=logits.device)


def _gather_source_mask(source_mask: torch.Tensor, beam_ids: torch.Tensor) -> torch.Tensor:
    gather_ids = beam_ids.to(dtype=torch.long).unsqueeze(-1).unsqueeze(-1)
    gather_ids = gather_ids.expand(-1, -1, 1, int(source_mask.shape[-1]))
    return torch.gather(source_mask, 2, gather_ids).squeeze(2)


HEADS.register_removed("safe_residual_reranker", "Use 'safe_residual_beam_reranker'.")


__all__ = [
    "CONDITION_ID_FIELDS",
    "GeometryPriorLogitFusion",
    "GpsGeometryPriorBranch",
    "SafeResidualBeamReranker",
    "label_space_fingerprint",
]
