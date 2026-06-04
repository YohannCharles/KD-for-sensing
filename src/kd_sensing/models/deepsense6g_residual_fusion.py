from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.evaluation.metrics import circular_window
from kd_sensing.models.encoders import ArrayEncoder, ImageEncoder, TabularEncoder


class GPSAnchoredResidualFusion(nn.Module):
    def __init__(
        self,
        *,
        num_beams: int = 64,
        gps_context_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        correction_scale_init: float = 0.5,
        correction_scale_max: float = 3.0,
        use_gate: bool = True,
        use_anchor: bool = True,
        enabled_modalities: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.hidden_dim = int(hidden_dim)
        self.correction_scale_max = float(correction_scale_max)
        self.use_gate = bool(use_gate)
        self.use_anchor = bool(use_anchor)
        self.enabled_modalities = tuple(str(item) for item in (enabled_modalities or ("gps_context",)))
        self.gps_encoder = TabularEncoder(
            input_dim=int(gps_context_dim),
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=float(dropout),
        )
        self.image_encoder = ImageEncoder(output_dim=self.hidden_dim) if "image" in self.enabled_modalities else None
        self.lidar_encoder = ArrayEncoder(output_dim=self.hidden_dim) if "lidar" in self.enabled_modalities else None
        self.radar_encoder = ArrayEncoder(output_dim=self.hidden_dim) if "radar" in self.enabled_modalities else None
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.correction_head = nn.Linear(self.hidden_dim, self.num_beams)
        self.modality_head = nn.Linear(self.hidden_dim, self.num_beams)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        raw_scale = _inverse_softplus(max(float(correction_scale_init), 1e-6))
        self.raw_correction_scale = nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))

    @property
    def correction_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_correction_scale).clamp(max=self.correction_scale_max)

    def forward(
        self,
        *,
        gps_prior_logits: torch.Tensor,
        gps_context_features: torch.Tensor,
        image: torch.Tensor | None = None,
        lidar: torch.Tensor | None = None,
        radar: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if gps_prior_logits.ndim != 2 or int(gps_prior_logits.shape[-1]) != self.num_beams:
            raise ValueError(f"gps_prior_logits must have shape [B, {self.num_beams}], got {tuple(gps_prior_logits.shape)}.")
        encoded = self.gps_encoder(gps_context_features)
        modality_names = ["gps_context"]
        if self.image_encoder is not None and image is not None:
            encoded = encoded + self.image_encoder(image)
            modality_names.append("image")
        if self.lidar_encoder is not None and lidar is not None:
            encoded = encoded + self.lidar_encoder(lidar)
            modality_names.append("lidar")
        if self.radar_encoder is not None and radar is not None:
            encoded = encoded + self.radar_encoder(radar)
            modality_names.append("radar")
        features = self.fusion(encoded)
        correction_logits = self.correction_head(features)
        modality_only_logits = self.modality_head(features)
        gate = torch.sigmoid(self.gate_head(features)) if self.use_gate else torch.ones_like(self.gate_head(features))
        scale = self.correction_scale.to(device=gps_prior_logits.device, dtype=gps_prior_logits.dtype)
        correction_strength = scale.view(1, 1) * gate
        anchored = gps_prior_logits + correction_strength * correction_logits
        final_logits = anchored if self.use_anchor else correction_logits
        prior_probs = F.softmax(gps_prior_logits, dim=-1)
        prior_entropy = -(prior_probs * prior_probs.clamp_min(1e-12).log()).sum(dim=-1)
        return {
            "final_logits": final_logits,
            "correction_logits": correction_logits,
            "modality_only_logits": modality_only_logits,
            "correction_gate": gate,
            "correction_strength": correction_strength,
            "diagnostics": {
                "correction_scale": scale.detach(),
                "prior_entropy": prior_entropy.detach(),
                "enabled_modalities": tuple(modality_names),
                "use_anchor": self.use_anchor,
                "use_gate": self.use_gate,
            },
        }


class GPSAnchoredTopKReranker(nn.Module):
    def __init__(
        self,
        *,
        num_beams: int = 64,
        gps_top_k: int = 16,
        local_radius: int = 8,
        modality_top_m: int = 8,
        score_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.gps_top_k = int(gps_top_k)
        self.local_radius = int(local_radius)
        self.modality_top_m = int(modality_top_m)
        self.score_temperature = float(score_temperature)

    def candidate_set(
        self,
        gps_logits: torch.Tensor,
        *,
        modality_logits: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if gps_logits.ndim != 2 or int(gps_logits.shape[-1]) != self.num_beams:
            raise ValueError(f"gps_logits must have shape [B, {self.num_beams}], got {tuple(gps_logits.shape)}.")
        gps_top = torch.topk(gps_logits, min(self.gps_top_k, self.num_beams), dim=-1).indices.cpu()
        gps_top1 = gps_logits.argmax(dim=-1).cpu()
        modality_top = None
        if modality_logits is not None:
            modality_top = torch.topk(modality_logits, min(self.modality_top_m, self.num_beams), dim=-1).indices.cpu()
        candidates: list[list[int]] = []
        for idx in range(int(gps_logits.shape[0])):
            ordered: list[int] = []
            for beam in gps_top[idx].tolist():
                _append_unique(ordered, int(beam), self.num_beams)
            for beam in circular_window(int(gps_top1[idx]), radius=self.local_radius, num_beams=self.num_beams):
                _append_unique(ordered, int(beam), self.num_beams)
            if modality_top is not None:
                for beam in modality_top[idx].tolist():
                    _append_unique(ordered, int(beam), self.num_beams)
            candidates.append(ordered)
        return candidates

    def forward(
        self,
        gps_logits: torch.Tensor,
        *,
        modality_logits: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        candidates = self.candidate_set(gps_logits, modality_logits=modality_logits)
        score_source = gps_logits if modality_logits is None else gps_logits + modality_logits
        max_candidates = max(len(row) for row in candidates) if candidates else 0
        scores = gps_logits.new_full((len(candidates), max_candidates), -1e9)
        candidate_tensor = torch.full((len(candidates), max_candidates), -1, dtype=torch.long, device=gps_logits.device)
        for row_idx, beams in enumerate(candidates):
            idx = torch.as_tensor(beams, dtype=torch.long, device=gps_logits.device)
            candidate_tensor[row_idx, : len(beams)] = idx
            scores[row_idx, : len(beams)] = score_source[row_idx, idx] / max(self.score_temperature, 1e-6)
        result: dict[str, Any] = {
            "candidates": candidates,
            "candidate_tensor": candidate_tensor,
            "candidate_scores": scores,
        }
        if target is not None:
            target_cpu = target.detach().cpu().to(torch.long).reshape(-1)
            mask_values: list[bool] = []
            local_values: list[bool] = []
            gps_top_values: list[bool] = []
            target_positions: list[int] = []
            gps_top = torch.topk(gps_logits, min(self.gps_top_k, self.num_beams), dim=-1).indices.cpu()
            gps_top1 = gps_logits.argmax(dim=-1).cpu()
            for idx, beams in enumerate(candidates):
                truth = int(target_cpu[idx])
                mask_values.append(truth in beams)
                target_positions.append(beams.index(truth) if truth in beams else -100)
                gps_top_values.append(truth in {int(item) for item in gps_top[idx].tolist()})
                local_values.append(
                    truth in set(circular_window(int(gps_top1[idx]), radius=self.local_radius, num_beams=self.num_beams))
                )
            target_pos = torch.as_tensor(target_positions, dtype=torch.long, device=gps_logits.device)
            loss_mask = torch.as_tensor(mask_values, dtype=torch.bool, device=gps_logits.device)
            loss = gps_logits.new_tensor(0.0)
            if bool(loss_mask.any()):
                loss = F.cross_entropy(scores[loss_mask], target_pos[loss_mask])
            result.update(
                {
                    "loss": loss,
                    "loss_mask": loss_mask,
                    "target_in_gps_top16": torch.as_tensor(gps_top_values, dtype=torch.bool, device=gps_logits.device),
                    "target_in_local_radius8": torch.as_tensor(local_values, dtype=torch.bool, device=gps_logits.device),
                    "target_in_union_candidates": loss_mask,
                }
            )
        return result


def _inverse_softplus(value: float) -> float:
    return math.log(math.exp(float(value)) - 1.0)


def _append_unique(values: list[int], beam: int, num_beams: int) -> None:
    normalized = int(beam) % int(num_beams)
    if normalized not in values:
        values.append(normalized)

