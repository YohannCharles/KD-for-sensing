from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BGAM_MODES = {
    "single_soft",
    "single_hard",
    "topk_union_soft",
    "topk_per_candidate",
    "history_pseudo_soft",
    "history_pseudo_hard",
    "history_pseudo_topk_union",
    "history_pseudo_per_candidate",
    "none",
}
FORBIDDEN_MASK_INPUTS = {
    "gt_beam",
    "target_label",
    "future_beam",
    "oracle_beam",
    "oracle_history_label",
    "target_candidate_index",
}


class GPSGuidedBGAM(nn.Module):
    def __init__(
        self,
        *,
        roi: Sequence[float] = (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0),
        bev_size: Sequence[int] = (64, 64),
        num_beams: int = 64,
        mode: str = "single_soft",
        sigma: float = 0.35,
        hard_half_width: float = 0.28,
        adaptive_sigma: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.roi = tuple(float(value) for value in roi)
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.num_beams = int(num_beams)
        self.mode = _validate_mode(mode)
        self.sigma = float(sigma)
        self.hard_half_width = float(hard_half_width)
        self.adaptive_sigma = dict(adaptive_sigma or {})
        theta_cell = self._build_theta_cell(self.roi, self.bev_size)
        self.register_buffer("theta_cell", theta_cell, persistent=True)

    def forward(
        self,
        bev_feat: torch.Tensor,
        *,
        theta_gps: torch.Tensor,
        gps_uncertainty: torch.Tensor | None = None,
        gps_topk_beams: torch.Tensor | None = None,
        gps_topk_probs: torch.Tensor | None = None,
        history_pseudo_beams: torch.Tensor | None = None,
        history_pseudo_probs: torch.Tensor | None = None,
        history_pseudo_entropy: torch.Tensor | None = None,
        history_valid_mask: torch.Tensor | None = None,
        beam_angles: torch.Tensor | Sequence[float] | None = None,
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if bev_feat.ndim != 4:
            raise ValueError(f"bev_feat must have shape [B,C,H,W], got {tuple(bev_feat.shape)}.")
        selected_mode = _validate_mode(mode or self.mode)
        if selected_mode == "none":
            mask = torch.ones((bev_feat.shape[0], 1, bev_feat.shape[2], bev_feat.shape[3]), dtype=bev_feat.dtype, device=bev_feat.device)
            return {"masked_bev_feat": bev_feat, "mask": mask}
        theta = theta_gps.to(device=bev_feat.device, dtype=bev_feat.dtype).reshape(-1)
        history_uncertainty = _history_uncertainty(
            history_pseudo_entropy,
            history_valid_mask,
            dtype=bev_feat.dtype,
            device=bev_feat.device,
        )
        combined_uncertainty = _combine_uncertainty(gps_uncertainty, history_uncertainty)
        sigma = self._effective_sigma(theta, combined_uncertainty, dtype=bev_feat.dtype, device=bev_feat.device)
        if selected_mode == "single_soft":
            mask = self._soft_mask(theta, sigma=sigma, dtype=bev_feat.dtype, device=bev_feat.device)
            return {"masked_bev_feat": bev_feat * mask, "mask": mask, "sigma": sigma, "mask_source": "theta_gps"}
        if selected_mode == "single_hard":
            mask = self._hard_mask(theta, dtype=bev_feat.dtype, device=bev_feat.device)
            return {"masked_bev_feat": bev_feat * mask, "mask": mask, "half_width": bev_feat.new_tensor(self.hard_half_width), "mask_source": "theta_gps"}
        if selected_mode.startswith("history_pseudo"):
            if history_pseudo_beams is None:
                raise ValueError(f"{selected_mode} requires history_pseudo_beams.")
            history_angles = self._candidate_angles(
                history_pseudo_beams,
                beam_angles=beam_angles,
                dtype=bev_feat.dtype,
                device=bev_feat.device,
            )
            history_weights = _history_weights(
                history_pseudo_probs,
                history_valid_mask,
                shape=history_pseudo_beams.shape,
                dtype=bev_feat.dtype,
                device=bev_feat.device,
            )
            if selected_mode == "history_pseudo_hard":
                mask = self._history_hard_mask(history_angles, history_weights, dtype=bev_feat.dtype, device=bev_feat.device)
                return {"masked_bev_feat": bev_feat * mask, "mask": mask, "half_width": bev_feat.new_tensor(self.hard_half_width), "mask_source": "history_pseudo"}
            if selected_mode == "history_pseudo_topk_union":
                history_mask = self._topk_union_mask(history_angles, history_weights, sigma=sigma, dtype=bev_feat.dtype, device=bev_feat.device)
                if gps_topk_beams is not None:
                    candidate_angles = self._candidate_angles(gps_topk_beams, beam_angles=beam_angles, dtype=bev_feat.dtype, device=bev_feat.device)
                    candidate_mask = self._topk_union_mask(candidate_angles, gps_topk_probs, sigma=sigma, dtype=bev_feat.dtype, device=bev_feat.device)
                    mask = torch.maximum(history_mask, candidate_mask)
                else:
                    mask = history_mask
                return {"masked_bev_feat": bev_feat * mask, "mask": mask, "sigma": sigma, "mask_source": "history_pseudo_topk_union"}
            if selected_mode == "history_pseudo_per_candidate":
                if gps_topk_beams is None:
                    raise ValueError(f"{selected_mode} requires gps_topk_beams.")
                candidate_angles = self._candidate_angles(gps_topk_beams, beam_angles=beam_angles, dtype=bev_feat.dtype, device=bev_feat.device)
                history_support = _candidate_history_support(candidate_angles, history_angles, history_weights)
                candidate_masks = self._candidate_masks(candidate_angles, history_support, sigma=sigma, dtype=bev_feat.dtype, device=bev_feat.device)
                return {
                    "candidate_masked_bev_feat": bev_feat.unsqueeze(1) * candidate_masks,
                    "candidate_masks": candidate_masks,
                    "mask": candidate_masks.max(dim=1).values,
                    "sigma": sigma,
                    "mask_source": "history_pseudo_per_candidate",
                }
            mask = self._topk_union_mask(history_angles, history_weights, sigma=sigma, dtype=bev_feat.dtype, device=bev_feat.device)
            return {"masked_bev_feat": bev_feat * mask, "mask": mask, "sigma": sigma, "mask_source": "history_pseudo"}
        if gps_topk_beams is None:
            raise ValueError(f"{selected_mode} requires gps_topk_beams.")
        candidate_angles = self._candidate_angles(gps_topk_beams, beam_angles=beam_angles, dtype=bev_feat.dtype, device=bev_feat.device)
        if selected_mode == "topk_union_soft":
            mask = self._topk_union_mask(candidate_angles, gps_topk_probs, sigma=sigma, dtype=bev_feat.dtype, device=bev_feat.device)
            return {"masked_bev_feat": bev_feat * mask, "mask": mask, "sigma": sigma, "mask_source": "candidate_topk"}
        candidate_masks = self._candidate_masks(candidate_angles, gps_topk_probs, sigma=sigma, dtype=bev_feat.dtype, device=bev_feat.device)
        return {
            "candidate_masked_bev_feat": bev_feat.unsqueeze(1) * candidate_masks,
            "candidate_masks": candidate_masks,
            "mask": candidate_masks.max(dim=1).values,
            "sigma": sigma,
            "mask_source": "candidate_topk_per_candidate",
        }

    def _soft_mask(self, theta: torch.Tensor, *, sigma: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        theta_cell = self.theta_cell.to(device=device, dtype=dtype)
        delta = wrap_to_pi_tensor(theta_cell.unsqueeze(0) - theta.view(-1, 1, 1))
        sigma_view = sigma.view(-1, 1, 1).clamp_min(1e-6)
        return torch.exp(-0.5 * torch.square(delta / sigma_view)).unsqueeze(1)

    def _hard_mask(self, theta: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        theta_cell = self.theta_cell.to(device=device, dtype=dtype)
        delta = wrap_to_pi_tensor(theta_cell.unsqueeze(0) - theta.view(-1, 1, 1)).abs()
        return delta.le(float(self.hard_half_width)).to(dtype).unsqueeze(1)

    def _topk_union_mask(
        self,
        candidate_angles: torch.Tensor,
        gps_topk_probs: torch.Tensor | None,
        *,
        sigma: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        masks = self._candidate_masks(candidate_angles, gps_topk_probs, sigma=sigma, dtype=dtype, device=device)
        return masks.max(dim=1).values

    def _candidate_masks(
        self,
        candidate_angles: torch.Tensor,
        gps_topk_probs: torch.Tensor | None,
        *,
        sigma: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        theta_cell = self.theta_cell.to(device=device, dtype=dtype)
        delta = wrap_to_pi_tensor(theta_cell.view(1, 1, *theta_cell.shape) - candidate_angles.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1))
        sigma_view = sigma.view(-1, 1, 1, 1).clamp_min(1e-6)
        masks = torch.exp(-0.5 * torch.square(delta / sigma_view)).unsqueeze(2)
        if gps_topk_probs is not None:
            weights = gps_topk_probs.to(device=device, dtype=dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            masks = masks * weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
        return masks

    def _candidate_angles(
        self,
        beams: torch.Tensor,
        *,
        beam_angles: torch.Tensor | Sequence[float] | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if beam_angles is None:
            table = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, self.num_beams, device=device, dtype=dtype)
        else:
            table = torch.as_tensor(beam_angles, device=device, dtype=dtype)
        return table[beams.to(device=device, dtype=torch.long).remainder(int(table.numel()))]

    def _history_hard_mask(
        self,
        history_angles: torch.Tensor,
        history_weights: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        theta_cell = self.theta_cell.to(device=device, dtype=dtype)
        delta = wrap_to_pi_tensor(theta_cell.view(1, 1, *theta_cell.shape) - history_angles.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)).abs()
        masks = delta.le(float(self.hard_half_width)).to(dtype).unsqueeze(2)
        masks = masks * history_weights.to(device=device, dtype=dtype).view(history_weights.shape[0], history_weights.shape[1], 1, 1, 1)
        return masks.max(dim=1).values

    def _effective_sigma(
        self,
        theta: torch.Tensor,
        uncertainty: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        sigma = torch.full((int(theta.shape[0]),), float(self.sigma), dtype=dtype, device=device)
        cfg = self.adaptive_sigma
        if bool(cfg.get("enabled", False)) and uncertainty is not None:
            value = uncertainty.to(device=device, dtype=dtype).reshape(-1)
            scale = float(cfg.get("uncertainty_scale", cfg.get("entropy_scale", 0.0)))
            sigma = sigma + scale * value
            sigma = sigma.clamp(max=float(cfg.get("max_sigma", max(self.sigma, 1.0))))
        return sigma

    @staticmethod
    def _build_theta_cell(roi: Sequence[float], bev_size: Sequence[int]) -> torch.Tensor:
        height, width = int(bev_size[0]), int(bev_size[1])
        x_min, x_max, y_min, y_max = [float(value) for value in roi[:4]]
        x_centers = torch.linspace(x_min, x_max, width + 1)[:-1] + (x_max - x_min) / max(width, 1) / 2.0
        y_centers = torch.linspace(y_max, y_min, height + 1)[:-1] - (y_max - y_min) / max(height, 1) / 2.0
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")
        return torch.atan2(yy, xx).to(torch.float32)


class LidarBEVCrossAttention(nn.Module):
    def __init__(self, *, in_channels: int, d_model: int = 64, num_heads: int = 4, num_queries: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_proj = nn.Linear(int(in_channels), int(d_model))
        self.queries = nn.Parameter(torch.randn(int(num_queries), int(d_model)) * 0.02)
        self.attn = nn.MultiheadAttention(int(d_model), int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, masked_bev_feat: torch.Tensor) -> torch.Tensor:
        if masked_bev_feat.ndim != 4:
            raise ValueError(f"masked_bev_feat must have shape [B,C,H,W], got {tuple(masked_bev_feat.shape)}.")
        batch = int(masked_bev_feat.shape[0])
        tokens = masked_bev_feat.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        out, _ = self.attn(queries, tokens, tokens, need_weights=False)
        return self.norm(out.mean(dim=1))


class GPSPriorEncoder(nn.Module):
    def __init__(self, *, d_model: int = 64, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(d_model)),
            nn.LayerNorm(int(d_model)),
        )

    def forward(
        self,
        *,
        theta_gps: torch.Tensor,
        distance_to_rsu: torch.Tensor,
        gps_probs: torch.Tensor | None = None,
        gps_logits: torch.Tensor | None = None,
        candidate_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        theta = theta_gps.reshape(-1).to(torch.float32)
        distance = distance_to_rsu.reshape(-1).to(device=theta.device, dtype=torch.float32)
        features = [torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1), torch.log1p(distance.clamp_min(0.0)).unsqueeze(-1)]
        prior = gps_probs if gps_probs is not None else None
        if prior is None and gps_logits is not None:
            prior = torch.softmax(gps_logits.to(device=theta.device, dtype=torch.float32), dim=-1)
        if prior is not None:
            prior = prior.to(device=theta.device, dtype=torch.float32)
            entropy = -(prior.clamp_min(1e-12).log() * prior).sum(dim=-1, keepdim=True)
            top2 = torch.topk(prior, k=min(2, int(prior.shape[-1])), dim=-1).values
            margin = top2[:, :1] - (top2[:, 1:2] if top2.shape[-1] > 1 else torch.zeros_like(top2[:, :1]))
            features.extend([entropy, margin])
        elif candidate_probs is not None:
            probs = candidate_probs.to(device=theta.device, dtype=torch.float32)
            entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1, keepdim=True)
            top2 = torch.topk(probs, k=min(2, int(probs.shape[-1])), dim=-1).values
            margin = top2[:, :1] - (top2[:, 1:2] if top2.shape[-1] > 1 else torch.zeros_like(top2[:, :1]))
            features.extend([entropy, margin])
        return self.net(torch.cat(features, dim=-1))


def assert_no_bgam_label_inputs(inputs: Mapping[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_MASK_INPUTS.intersection(str(key) for key in inputs))
    if forbidden:
        raise ValueError(f"BGAM mask inputs must not contain future label fields: {', '.join(forbidden)}.")


def save_debug_masks(
    masks: torch.Tensor,
    *,
    output_dir: str | Path,
    sample_ids: Sequence[str],
    theta_gps: torch.Tensor,
    mode: str,
    sigma: torch.Tensor | float | None = None,
    half_width: float | None = None,
    beam_angle_source: str = "",
    max_samples: int = 8,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_tensor = masks.detach().cpu().to(torch.float32)
    if mask_tensor.ndim == 5:
        mask_tensor = mask_tensor.max(dim=1).values
    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor[:, 0]
    created: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    limit = min(int(max_samples), int(mask_tensor.shape[0]), len(sample_ids))
    sigma_values = _as_list(sigma, limit)
    theta_values = theta_gps.detach().cpu().reshape(-1).tolist()
    for idx in range(limit):
        sample_id = str(sample_ids[idx])
        safe = sample_id.replace("/", "_").replace(":", "_")
        npy_path = out / f"{idx:04d}_{safe}_mask.npy"
        np.save(npy_path, mask_tensor[idx].numpy())
        created.append(str(npy_path))
        png_path = out / f"{idx:04d}_{safe}_mask.png"
        try:
            from PIL import Image

            array = mask_tensor[idx].numpy()
            array = (255.0 * (array - array.min()) / max(float(array.max() - array.min()), 1e-8)).astype(np.uint8)
            Image.fromarray(array).save(png_path)
            created.append(str(png_path))
        except Exception:
            png_path = None
        metadata_rows.append(
            {
                "sample_id": sample_id,
                "theta_gps": float(theta_values[idx]) if idx < len(theta_values) else "",
                "sigma": sigma_values[idx],
                "half_width": half_width if half_width is not None else "",
                "bgam_mode": mode,
                "beam_angle_source": beam_angle_source,
                "mask_path": str(npy_path),
                "png_path": str(png_path or ""),
                "gt_beam_used_as_mask_source": False,
            }
        )
    metadata_path = out / "debug_mask_metadata.json"
    metadata_path.write_text(json.dumps(metadata_rows, indent=2, sort_keys=True), encoding="utf-8")
    return {"created": created, "metadata_path": str(metadata_path), "sample_count": limit}


def wrap_to_pi_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _validate_mode(mode: str) -> str:
    value = str(mode or "single_soft")
    aliases = {
        "gps_lidar_soft_bgam": "single_soft",
        "gps_lidar_hard_bgam": "single_hard",
        "gps_lidar_topk_union_bgam": "topk_union_soft",
        "gps_lidar_topk_per_candidate_rerank": "topk_per_candidate",
        "gps_pseudo_history_soft_bgam": "history_pseudo_soft",
        "gps_pseudo_history_topk_union_bgam": "history_pseudo_topk_union",
        "gps_pseudo_history_per_candidate_rerank": "history_pseudo_per_candidate",
        "history_pseudo_hard_bgam": "history_pseudo_hard",
        "gps_lidar_no_bgam": "none",
        "lidar_only_no_bgam": "none",
    }
    value = aliases.get(value, value)
    if value not in BGAM_MODES:
        raise ValueError(f"Unsupported BGAM mode: {mode}.")
    return value


def _history_uncertainty(
    entropy: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if entropy is None:
        return None
    values = entropy.to(device=device, dtype=dtype)
    if values.ndim == 1:
        return values
    valid = torch.ones_like(values, dtype=dtype) if valid_mask is None else valid_mask.to(device=device, dtype=dtype)
    denom = valid.sum(dim=-1).clamp_min(1.0)
    return (values * valid).sum(dim=-1) / denom


def _combine_uncertainty(gps_uncertainty: torch.Tensor | None, history_uncertainty: torch.Tensor | None) -> torch.Tensor | None:
    if gps_uncertainty is None:
        return history_uncertainty
    if history_uncertainty is None:
        return gps_uncertainty
    return 0.5 * (gps_uncertainty.reshape(-1) + history_uncertainty.reshape(-1).to(device=gps_uncertainty.device, dtype=gps_uncertainty.dtype))


def _history_weights(
    probs: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    *,
    shape: torch.Size | Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if probs is None:
        weights = torch.ones(tuple(shape), dtype=dtype, device=device)
    else:
        weights = probs.to(device=device, dtype=dtype)
    if valid_mask is not None:
        weights = weights * valid_mask.to(device=device, dtype=dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _candidate_history_support(
    candidate_angles: torch.Tensor,
    history_angles: torch.Tensor,
    history_weights: torch.Tensor,
) -> torch.Tensor:
    delta = wrap_to_pi_tensor(candidate_angles.unsqueeze(-1) - history_angles.unsqueeze(1)).abs()
    support = torch.exp(-0.5 * torch.square(delta / 0.35)) * history_weights.unsqueeze(1)
    support = support.max(dim=-1).values
    return support / support.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _as_list(value: torch.Tensor | float | None, limit: int) -> list[Any]:
    if value is None:
        return ["" for _ in range(limit)]
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
        return [float(values[idx]) if idx < len(values) else "" for idx in range(limit)]
    return [float(value) for _ in range(limit)]


__all__ = [
    "GPSGuidedBGAM",
    "GPSPriorEncoder",
    "LidarBEVCrossAttention",
    "assert_no_bgam_label_inputs",
    "save_debug_masks",
    "wrap_to_pi_tensor",
]
