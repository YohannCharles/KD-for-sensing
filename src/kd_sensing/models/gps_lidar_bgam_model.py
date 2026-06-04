from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.models.gps_lidar_bgam import GPSGuidedBGAM, GPSPriorEncoder, LidarBEVCrossAttention, assert_no_bgam_label_inputs
from kd_sensing.models.lidar_pillar_encoder import LidarBEVSpatialEncoder, SimplePillarEncoder, freeze_module
from kd_sensing.models.topk_candidate_selector import sparse_topk_scores_to_logits


class GPSLidarBGAMBeamPredictor(nn.Module):
    def __init__(
        self,
        *,
        topk: int = 8,
        num_beams: int = 64,
        d_model: int = 64,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        lambda_lidar_init: float = 0.5,
        lambda_lidar_max: float = 3.0,
        fusion: str = "concat_mlp",
        lidar_in_channels: int = 3,
        lidar_channels: Sequence[int] = (32, 64),
        freeze_lidar_encoder: bool = False,
        roi: Sequence[float] = (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0),
        bev_size: Sequence[int] = (64, 64),
        bgam_mode: str = "single_soft",
        bgam_sigma: float = 0.35,
        bgam_hard_half_width: float = 0.28,
        adaptive_sigma: Mapping[str, Any] | None = None,
        attention_heads: int = 4,
        attention_queries: int = 1,
        full64_head_enabled: bool = False,
        lidar_profile: str = "bev_cache",
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.num_beams = int(num_beams)
        self.d_model = int(d_model)
        self.fusion = str(fusion)
        self.lambda_lidar_max = float(lambda_lidar_max)
        self.full64_head_enabled = bool(full64_head_enabled)
        self.lidar_profile = str(lidar_profile)
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.lidar_in_channels = int(lidar_in_channels)
        self.pillar = SimplePillarEncoder(bev_size=bev_size, roi=roi)
        encoder_in = 6 if self.lidar_profile == "pillar6" else int(lidar_in_channels)
        self.lidar_encoder = LidarBEVSpatialEncoder(in_channels=encoder_in, channels=lidar_channels, dropout=dropout)
        if bool(freeze_lidar_encoder):
            freeze_module(self.lidar_encoder)
        self.bgam = GPSGuidedBGAM(
            roi=roi,
            bev_size=bev_size,
            num_beams=num_beams,
            mode=bgam_mode,
            sigma=bgam_sigma,
            hard_half_width=bgam_hard_half_width,
            adaptive_sigma=adaptive_sigma,
        )
        self.lidar_attention = LidarBEVCrossAttention(
            in_channels=self.lidar_encoder.out_channels,
            d_model=d_model,
            num_heads=attention_heads,
            num_queries=attention_queries,
            dropout=dropout,
        )
        self.gps_encoder = GPSPriorEncoder(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, d_model),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
        )
        self.cross_modal_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.cross_modal_attn = nn.MultiheadAttention(d_model, max(1, min(int(attention_heads), int(d_model))), batch_first=True)
        self.score_head = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.full64_head = nn.Sequential(nn.Linear(d_model * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_beams))
        self.lambda_lidar_param = nn.Parameter(torch.tensor(_inverse_softplus_or_floor(float(lambda_lidar_init)), dtype=torch.float32))

    @property
    def lambda_lidar(self) -> torch.Tensor:
        return F.softplus(self.lambda_lidar_param).clamp(min=0.0, max=self.lambda_lidar_max)

    def forward(
        self,
        *,
        candidate_beams: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        candidate_probs: torch.Tensor | None = None,
        theta_gps: torch.Tensor,
        distance_to_rsu: torch.Tensor,
        lidar_bev: torch.Tensor | None = None,
        raw_points: Sequence[torch.Tensor] | torch.Tensor | None = None,
        gps_probs: torch.Tensor | None = None,
        gps_logits: torch.Tensor | None = None,
        gps_entropy: torch.Tensor | None = None,
        history_pseudo_beams: torch.Tensor | None = None,
        history_pseudo_probs: torch.Tensor | None = None,
        history_pseudo_entropy: torch.Tensor | None = None,
        history_valid_mask: torch.Tensor | None = None,
        beam_angles: torch.Tensor | Sequence[float] | None = None,
        bgam_mode: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert_no_bgam_label_inputs(kwargs)
        if candidate_beams.ndim != 2:
            raise ValueError(f"candidate_beams must have shape [B,K], got {tuple(candidate_beams.shape)}.")
        batch, topk = candidate_beams.shape
        if int(topk) != self.topk:
            raise ValueError(f"candidate_beams K must be {self.topk}, got {topk}.")
        device = candidate_beams.device
        dtype = candidate_log_probs.dtype
        candidate_log_probs = candidate_log_probs.to(device=device, dtype=torch.float32)
        if candidate_probs is None:
            candidate_probs = torch.softmax(candidate_log_probs, dim=-1)
        candidate_probs = candidate_probs.to(device=device, dtype=torch.float32)
        theta_gps = theta_gps.to(device=device, dtype=torch.float32).reshape(-1)
        distance_to_rsu = distance_to_rsu.to(device=device, dtype=torch.float32).reshape(-1)
        lidar_input = self._lidar_input(lidar_bev=lidar_bev, raw_points=raw_points, batch=batch, device=device)
        spatial = self.lidar_encoder(lidar_input)
        bgam_out = self.bgam(
            spatial,
            theta_gps=theta_gps,
            gps_uncertainty=gps_entropy,
            gps_topk_beams=candidate_beams,
            gps_topk_probs=candidate_probs,
            history_pseudo_beams=history_pseudo_beams,
            history_pseudo_probs=history_pseudo_probs,
            history_pseudo_entropy=history_pseudo_entropy,
            history_valid_mask=history_valid_mask,
            beam_angles=beam_angles,
            mode=bgam_mode,
        )
        gps_emb = self.gps_encoder(
            theta_gps=theta_gps,
            distance_to_rsu=distance_to_rsu,
            gps_probs=gps_probs.to(device=device, dtype=torch.float32) if gps_probs is not None else None,
            gps_logits=gps_logits.to(device=device, dtype=torch.float32) if gps_logits is not None else None,
            candidate_probs=candidate_probs,
        )
        if "candidate_masked_bev_feat" in bgam_out:
            candidate_feat = bgam_out["candidate_masked_bev_feat"]
            flat = candidate_feat.reshape(batch * self.topk, candidate_feat.shape[2], candidate_feat.shape[3], candidate_feat.shape[4])
            lidar_emb = self.lidar_attention(flat).reshape(batch, self.topk, self.d_model)
        else:
            lidar_emb_single = self.lidar_attention(bgam_out["masked_bev_feat"])
            lidar_emb = lidar_emb_single.unsqueeze(1).expand(-1, self.topk, -1)
        candidate_emb = self.candidate_encoder(_candidate_features(candidate_beams, candidate_probs, candidate_log_probs, theta_gps, self.num_beams))
        gps_expanded = gps_emb.unsqueeze(1).expand(-1, self.topk, -1)
        fused = self._fuse(gps_expanded, lidar_emb, candidate_emb)
        lidar_scores = self.score_head(fused).squeeze(-1)
        lambda_value = self.lambda_lidar.to(device=device, dtype=torch.float32)
        final_scores = candidate_log_probs + lambda_value.view(1, 1) * lidar_scores
        final_probs = torch.softmax(final_scores, dim=-1)
        selected_index = final_probs.argmax(dim=-1, keepdim=True)
        selected_beam = torch.gather(candidate_beams.to(device=device), 1, selected_index).squeeze(-1)
        outputs: dict[str, Any] = {
            "final_candidate_scores": final_scores.to(dtype=dtype),
            "lidar_candidate_scores": lidar_scores.to(dtype=dtype),
            "candidate_probs": final_probs.to(dtype=dtype),
            "selected_beam": selected_beam,
            "lambda_lidar": lambda_value.detach(),
            "bgam_mask": bgam_out.get("mask"),
            "beam_angle_source": kwargs.get("beam_angle_source", ""),
            "diagnostics": {
                "lambda_lidar": float(lambda_value.detach().cpu()),
                "bgam_mode": str(bgam_mode or self.bgam.mode),
                "mask_source": str(bgam_out.get("mask_source", "")),
                "fusion": self.fusion,
                "full64_head_enabled": self.full64_head_enabled,
            },
        }
        outputs["sparse_logits64"] = sparse_topk_scores_to_logits(candidate_beams, final_scores, num_beams=self.num_beams)
        if self.full64_head_enabled:
            pooled_lidar = lidar_emb.mean(dim=1)
            outputs["logits64"] = self.full64_head(torch.cat([gps_emb, pooled_lidar], dim=-1)).to(dtype=dtype)
        return outputs

    def _lidar_input(
        self,
        *,
        lidar_bev: torch.Tensor | None,
        raw_points: Sequence[torch.Tensor] | torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.lidar_profile == "pillar6":
            if raw_points is None:
                return torch.zeros((batch, 6, *self.bev_size), dtype=torch.float32, device=device)
            return self.pillar(raw_points).to(device=device, dtype=torch.float32)
        if lidar_bev is None:
            return torch.zeros((batch, self.lidar_in_channels, *self.bev_size), dtype=torch.float32, device=device)
        return lidar_bev.to(device=device, dtype=torch.float32)

    def _fuse(self, gps_emb: torch.Tensor, lidar_emb: torch.Tensor, candidate_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion == "cross_attention":
            batch, topk, _ = gps_emb.shape
            tokens = torch.stack([gps_emb, lidar_emb, candidate_emb], dim=2).reshape(batch * topk, 3, self.d_model)
            query = self.cross_modal_query.unsqueeze(0).expand(batch * topk, -1, -1)
            out, _ = self.cross_modal_attn(query, tokens, tokens, need_weights=False)
            return out.reshape(batch, topk, self.d_model)
        return self.fusion_mlp(torch.cat([gps_emb, lidar_emb, candidate_emb], dim=-1))


def _candidate_features(
    candidate_beams: torch.Tensor,
    candidate_probs: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    theta_gps: torch.Tensor,
    num_beams: int,
) -> torch.Tensor:
    beams = candidate_beams.to(dtype=torch.float32)
    beam_angle = beams / float(num_beams) * 2.0 * math.pi - math.pi
    delta = torch.remainder(beam_angle - theta_gps.view(-1, 1) + math.pi, 2.0 * math.pi) - math.pi
    rank = torch.arange(candidate_beams.shape[1], device=candidate_beams.device, dtype=torch.float32).view(1, -1)
    rank = rank / max(candidate_beams.shape[1] - 1, 1)
    return torch.stack(
        [
            torch.sin(beam_angle),
            torch.cos(beam_angle),
            torch.sin(delta),
            torch.cos(delta),
            candidate_probs.to(torch.float32),
            candidate_log_probs.to(torch.float32),
        ],
        dim=-1,
    ) + torch.zeros((*candidate_beams.shape, 6), device=candidate_beams.device, dtype=torch.float32) + 0.0 * rank.unsqueeze(-1)


def _inverse_softplus_or_floor(value: float) -> float:
    if value <= 0:
        return -30.0
    return math.log(math.exp(float(value)) - 1.0)


__all__ = ["GPSLidarBGAMBeamPredictor"]
