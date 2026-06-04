from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.evaluation.metrics import residual_delta_class_count


class CameraGPSResidualFusion(nn.Module):
    """Camera-assisted local residual/gate correction on top of frozen GPS prior."""

    def __init__(
        self,
        *,
        num_beams: int = 64,
        gps_context_dim: int = 9,
        camera_feature_dim: int = 128,
        hidden_dim: int = 128,
        delta_radius: int = 8,
        dropout: float = 0.1,
        gate_bias_init: float = -2.0,
        overflow_policy: str = "uniform",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.gps_context_dim = int(gps_context_dim)
        self.camera_feature_dim = int(camera_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_radius = int(delta_radius)
        self.delta_class_count = int(residual_delta_class_count(radius=self.delta_radius))
        self.overflow_policy = str(overflow_policy or "uniform")
        self.eps = float(eps)
        self.gps_encoder = nn.Sequential(
            nn.Linear(self.gps_context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.camera_encoder = nn.Sequential(
            nn.Linear(self.camera_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.prior_encoder = nn.Sequential(
            nn.Linear(self.num_beams, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.residual_delta_head = nn.Linear(self.hidden_dim, self.delta_class_count)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.direct_beam_head = nn.Linear(self.hidden_dim, self.num_beams)
        nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

    def forward(
        self,
        *,
        gps_prior_logits: torch.Tensor,
        gps_pred_top1: torch.Tensor | None = None,
        gps_context: torch.Tensor | None = None,
        gps_context_features: torch.Tensor | None = None,
        camera_ae_feature: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if gps_prior_logits.ndim != 2 or int(gps_prior_logits.shape[-1]) != self.num_beams:
            raise ValueError(f"gps_prior_logits must have shape [B, {self.num_beams}], got {tuple(gps_prior_logits.shape)}.")
        batch_size = int(gps_prior_logits.shape[0])
        device = gps_prior_logits.device
        dtype = gps_prior_logits.dtype
        if gps_pred_top1 is None:
            gps_pred_top1 = gps_prior_logits.argmax(dim=-1)
        gps_pred_top1 = gps_pred_top1.to(device=device, dtype=torch.long).reshape(batch_size)
        context = gps_context if gps_context is not None else gps_context_features
        if context is None:
            context = gps_prior_logits.new_zeros((batch_size, self.gps_context_dim))
        context = _fit_feature_dim(context.to(device=device, dtype=dtype), self.gps_context_dim)
        camera = camera_ae_feature
        if camera is None:
            camera = gps_prior_logits.new_zeros((batch_size, self.camera_feature_dim))
        if camera.ndim > 2:
            camera = camera.flatten(start_dim=1)
        camera = _fit_feature_dim(camera.to(device=device, dtype=dtype), self.camera_feature_dim)

        p_gps = F.softmax(gps_prior_logits, dim=-1)
        features = self.fusion(
            torch.cat(
                [
                    self.gps_encoder(context),
                    self.camera_encoder(camera),
                    self.prior_encoder(p_gps),
                ],
                dim=-1,
            )
        )
        residual_delta_logits = self.residual_delta_head(features)
        gate_logit = self.gate_head(features)
        correction_gate = torch.sigmoid(gate_logit)
        p_corr = synthesize_correction_distribution(
            residual_delta_logits,
            gps_pred_top1,
            num_beams=self.num_beams,
            delta_radius=self.delta_radius,
            overflow_policy=self.overflow_policy,
            eps=self.eps,
        )
        p_final = (1.0 - correction_gate) * p_gps + correction_gate * p_corr
        p_final = p_final / p_final.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        final_logits = torch.log(p_final.clamp_min(self.eps))
        direct_beam_logits = self.direct_beam_head(features)
        return {
            "residual_delta_logits": residual_delta_logits,
            "gate_logit": gate_logit,
            "correction_gate": correction_gate,
            "p_corr": p_corr,
            "p_gps": p_gps,
            "p_final": p_final,
            "final_logits": final_logits,
            "direct_beam_logits": direct_beam_logits,
            "diagnostics": {
                "delta_radius": self.delta_radius,
                "delta_class_count": self.delta_class_count,
                "overflow_policy": self.overflow_policy,
                "gate_mean": correction_gate.detach().mean(),
                "p_corr_sum": p_corr.detach().sum(dim=-1),
                "p_final_sum": p_final.detach().sum(dim=-1),
            },
        }


def synthesize_correction_distribution(
    residual_delta_logits: torch.Tensor,
    gps_pred_top1: torch.Tensor,
    *,
    num_beams: int = 64,
    delta_radius: int = 8,
    overflow_policy: str = "uniform",
    eps: float = 1e-8,
) -> torch.Tensor:
    if residual_delta_logits.ndim != 2:
        raise ValueError(f"residual_delta_logits must have shape [B, C], got {tuple(residual_delta_logits.shape)}.")
    expected = residual_delta_class_count(radius=delta_radius)
    if int(residual_delta_logits.shape[-1]) != int(expected):
        raise ValueError(f"Expected {expected} delta classes, got {int(residual_delta_logits.shape[-1])}.")
    batch_size = int(residual_delta_logits.shape[0])
    device = residual_delta_logits.device
    probs = F.softmax(residual_delta_logits, dim=-1)
    local_probs = probs[:, :-1]
    overflow = probs[:, -1:]
    deltas = torch.arange(-int(delta_radius), int(delta_radius) + 1, device=device, dtype=torch.long)
    beams = (gps_pred_top1.to(device=device, dtype=torch.long).reshape(batch_size, 1) + deltas.view(1, -1)).remainder(int(num_beams))
    p_corr = residual_delta_logits.new_zeros((batch_size, int(num_beams)))
    p_corr.scatter_add_(dim=1, index=beams, src=local_probs)
    if str(overflow_policy or "uniform") == "uniform":
        p_corr = p_corr + overflow / float(num_beams)
    elif str(overflow_policy or "uniform") == "ignore":
        pass
    else:
        raise ValueError("overflow_policy must be one of uniform or ignore.")
    return p_corr / p_corr.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def _fit_feature_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
    if value.ndim != 2:
        value = value.flatten(start_dim=1)
    current = int(value.shape[-1])
    target = int(dim)
    if current == target:
        return value
    if current > target:
        return value[:, :target]
    pad = value.new_zeros((int(value.shape[0]), target - current))
    return torch.cat([value, pad], dim=-1)


__all__ = ["CameraGPSResidualFusion", "synthesize_correction_distribution"]
