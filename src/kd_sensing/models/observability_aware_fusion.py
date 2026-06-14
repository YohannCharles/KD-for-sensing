from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


JEPA_ADVANTAGE_GPS_CONDITIONS = {"C3_random_async", "C4_severe_async"}
JEPA_ADVANTAGE_IMAGE_CONDITIONS = {
    "D3_motion_blur",
    "D4_partial_occlusion",
    "D6_burst_missing",
    "D7_joint_worst_case",
}


class ObservabilityAwareFusion(nn.Module):
    supports_reliability_metadata = True

    def __init__(
        self,
        *,
        image_dim: int | None = None,
        gps_dim: int | None = None,
        fused_dim: int | None = None,
        image_observability_threshold: float = 0.35,
        gps_delay_scale: float = 4.0,
        enable_jepa_fallback: bool = True,
        fallback_unavailable: str = "raw",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.image_dim = None if image_dim is None else int(image_dim)
        self.gps_dim = None if gps_dim is None else int(gps_dim)
        self.fused_dim = None if fused_dim is None else int(fused_dim)
        self.image_observability_threshold = float(image_observability_threshold)
        self.gps_delay_scale = max(float(gps_delay_scale), float(eps))
        self.enable_jepa_fallback = bool(enable_jepa_fallback)
        self.fallback_unavailable = str(fallback_unavailable)
        self.eps = float(eps)
        if self.image_dim is not None and self.fused_dim is not None:
            self.image_projection: nn.Module = nn.Identity() if self.image_dim == self.fused_dim else nn.Linear(self.image_dim, self.fused_dim)
        else:
            self.image_projection = nn.Identity()
        if self.gps_dim is not None and self.fused_dim is not None:
            self.gps_projection: nn.Module = nn.Identity() if self.gps_dim == self.fused_dim else nn.Linear(self.gps_dim, self.fused_dim)
        else:
            self.gps_projection = nn.Identity()

    def forward(
        self,
        z_img: torch.Tensor | None = None,
        z_gps: torch.Tensor | None = None,
        *,
        image_latent: torch.Tensor | None = None,
        gps_latent: torch.Tensor | None = None,
        image_valid_mask: torch.Tensor | None = None,
        image_observability_score: torch.Tensor | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        gps_delay_steps: torch.Tensor | None = None,
        jepa_predicted_latent: torch.Tensor | None = None,
        temporal_jepa_latent: torch.Tensor | None = None,
        benchmark_condition_metadata: Mapping[str, Any] | None = None,
        image_degradation_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del image_degradation_metadata
        img = z_img if z_img is not None else image_latent
        gps = z_gps if z_gps is not None else gps_latent
        if img is None or gps is None:
            raise ValueError("ObservabilityAwareFusion requires z_img/image_latent and z_gps/gps_latent tensors.")
        img_3d, squeezed = _as_temporal_latent(img, name="z_img")
        gps_3d, gps_squeezed = _as_temporal_latent(gps, name="z_gps")
        if squeezed != gps_squeezed:
            raise ValueError("z_img and z_gps must both be rank-2 or both be rank-3 temporal latents.")
        _check_batch_time(img_3d, gps_3d)
        if self.image_dim is not None and int(img_3d.shape[-1]) != self.image_dim:
            raise ValueError(f"Expected z_img latent dim {self.image_dim}, got {int(img_3d.shape[-1])}.")
        if self.gps_dim is not None and int(gps_3d.shape[-1]) != self.gps_dim:
            raise ValueError(f"Expected z_gps latent dim {self.gps_dim}, got {int(gps_3d.shape[-1])}.")

        batch_size, steps, _ = img_3d.shape
        image_valid = _required_mask(
            image_valid_mask,
            name="image_valid_mask",
            batch_size=batch_size,
            steps=steps,
            device=img_3d.device,
        )
        image_score = _required_score(
            image_observability_score,
            name="image_observability_score",
            batch_size=batch_size,
            steps=steps,
            device=img_3d.device,
            dtype=img_3d.dtype,
        ).clamp(0.0, 1.0)
        gps_valid = _required_mask(
            gps_valid_mask,
            name="gps_valid_mask",
            batch_size=batch_size,
            steps=steps,
            device=img_3d.device,
        )
        delay = _required_score(
            gps_delay_steps,
            name="gps_delay_steps",
            batch_size=batch_size,
            steps=steps,
            device=img_3d.device,
            dtype=img_3d.dtype,
        ).clamp_min(0.0)

        predicted = temporal_jepa_latent if temporal_jepa_latent is not None else jepa_predicted_latent
        predicted_3d = None
        if predicted is not None:
            predicted_3d, predicted_squeezed = _as_temporal_latent(predicted, name="temporal_jepa_latent")
            if predicted_squeezed != squeezed:
                raise ValueError("temporal_jepa_latent rank must match z_img rank.")
            _check_batch_time(img_3d, predicted_3d)
            if int(predicted_3d.shape[-1]) != int(img_3d.shape[-1]):
                raise ValueError(
                    "temporal_jepa_latent must match z_img latent dim before projection; "
                    f"got {int(predicted_3d.shape[-1])} and {int(img_3d.shape[-1])}."
                )

        low_image = (~image_valid) | image_score.lt(self.image_observability_threshold)
        advantage = is_jepa_advantage_condition(benchmark_condition_metadata)
        fallback_mask = torch.zeros_like(low_image)
        warnings: list[str] = []
        if self.enable_jepa_fallback:
            advantage_request = (
                image_score.lt(max(self.image_observability_threshold, 0.5))
                if advantage
                else torch.zeros_like(low_image)
            )
            requested = low_image | advantage_request
            if predicted_3d is not None:
                fallback_mask = requested
                img_3d = torch.where(fallback_mask.unsqueeze(-1), predicted_3d.to(dtype=img_3d.dtype, device=img_3d.device), img_3d)
            elif bool(requested.any()):
                warnings.append(f"temporal_jepa_latent_unavailable:{self.fallback_unavailable}")

        projected_img = self.image_projection(img_3d)
        projected_gps = self.gps_projection(gps_3d)
        if tuple(projected_img.shape) != tuple(projected_gps.shape):
            if self.fused_dim is None:
                raise ValueError(
                    "Projected image/GPS latents must have identical shape. Provide image_dim, gps_dim and fused_dim "
                    f"when latent dims differ; got {tuple(projected_img.shape)} and {tuple(projected_gps.shape)}."
                )
            raise ValueError(
                f"Projected image/GPS latents must have identical shape, got {tuple(projected_img.shape)} and {tuple(projected_gps.shape)}."
            )

        image_reliability = image_score * image_valid.to(dtype=img_3d.dtype)
        gps_delay_reliability = 1.0 / (1.0 + delay / self.gps_delay_scale)
        gps_reliability = gps_valid.to(dtype=img_3d.dtype) * gps_delay_reliability
        denom = image_reliability + gps_reliability
        default = torch.full_like(denom, 0.5)
        w_img = torch.where(denom.gt(self.eps), image_reliability / denom.clamp_min(self.eps), default)
        w_gps = torch.where(denom.gt(self.eps), gps_reliability / denom.clamp_min(self.eps), default)
        z_fuse = w_img.unsqueeze(-1) * projected_img + w_gps.unsqueeze(-1) * projected_gps
        if squeezed:
            z_fuse = z_fuse.squeeze(1)
            w_img_out = w_img.squeeze(1)
            w_gps_out = w_gps.squeeze(1)
            image_score_out = image_score.squeeze(1)
            gps_reliability_out = gps_reliability.squeeze(1)
            fallback_out = fallback_mask.squeeze(1)
        else:
            w_img_out = w_img
            w_gps_out = w_gps
            image_score_out = image_score
            gps_reliability_out = gps_reliability
            fallback_out = fallback_mask
        diagnostics = {
            "w_img": w_img_out,
            "w_gps": w_gps_out,
            "image_observability_score": image_score_out,
            "gps_reliability": gps_reliability_out,
            "gps_delay_steps": delay.squeeze(1) if squeezed else delay,
            "image_missing_or_low_observability": low_image.squeeze(1) if squeezed else low_image,
            "jepa_fallback_triggered": fallback_out,
            "jepa_advantage_condition": bool(advantage),
            "latent_source": "temporal_jepa" if bool(fallback_mask.any()) else "current_image",
            "warnings": warnings,
            "gps_downweight_reason": "invalid_or_delayed" if (bool((~gps_valid).any()) or bool(delay.gt(0).any())) else "",
            "image_downweight_reason": "missing_or_low_observability" if bool(low_image.any()) else "",
        }
        return {
            "z_fuse": z_fuse,
            "fused": z_fuse,
            "w_img": w_img_out,
            "w_gps": w_gps_out,
            "diagnostics": diagnostics,
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "observability_aware_fusion",
            "supports_reliability_metadata": True,
            "image_observability_threshold": self.image_observability_threshold,
            "gps_delay_scale": self.gps_delay_scale,
            "enable_jepa_fallback": self.enable_jepa_fallback,
        }


def is_jepa_advantage_condition(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    gps_condition = _condition_value(metadata, ("gps_condition", "scenario_c_condition", "gps_async_condition"))
    image_condition = _condition_value(metadata, ("image_condition", "scenario_d_condition"))
    condition = _condition_value(metadata, ("condition", "difficulty_condition"))
    if not gps_condition and condition:
        gps_condition = _find_condition_token(condition, JEPA_ADVANTAGE_GPS_CONDITIONS)
    if not image_condition and condition:
        image_condition = _find_condition_token(condition, JEPA_ADVANTAGE_IMAGE_CONDITIONS)
    return gps_condition in JEPA_ADVANTAGE_GPS_CONDITIONS and image_condition in JEPA_ADVANTAGE_IMAGE_CONDITIONS


def _condition_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _find_condition_token(text: str, candidates: set[str]) -> str:
    for candidate in candidates:
        if candidate in str(text):
            return candidate
    return ""


def _as_temporal_latent(value: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor.")
    if value.ndim == 2:
        return value.unsqueeze(1), True
    if value.ndim == 3:
        return value, False
    raise ValueError(f"{name} must have shape [B, D] or [B, T, D], got {tuple(value.shape)}.")


def _check_batch_time(left: torch.Tensor, right: torch.Tensor) -> None:
    if tuple(left.shape[:2]) != tuple(right.shape[:2]):
        raise ValueError(
            "z_img and z_gps must share batch/time dimensions; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )


def _required_mask(
    value: torch.Tensor | None,
    *,
    name: str,
    batch_size: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"ObservabilityAwareFusion requires reliability metadata '{name}'.")
    mask = torch.as_tensor(value, dtype=torch.bool, device=device)
    return _align_bt(mask, name=name, batch_size=batch_size, steps=steps)


def _required_score(
    value: torch.Tensor | None,
    *,
    name: str,
    batch_size: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"ObservabilityAwareFusion requires reliability metadata '{name}'.")
    score = torch.as_tensor(value, dtype=dtype, device=device)
    return _align_bt(score, name=name, batch_size=batch_size, steps=steps)


def _align_bt(value: torch.Tensor, *, name: str, batch_size: int, steps: int) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, T] or [B], got {tuple(value.shape)}.")
    if int(value.shape[0]) != int(batch_size):
        raise ValueError(f"{name} batch dimension must be {batch_size}, got {int(value.shape[0])}.")
    if int(value.shape[1]) == int(steps):
        return value
    if int(value.shape[1]) == 1:
        return value.expand(-1, int(steps))
    raise ValueError(f"{name} time dimension must be {steps} or 1, got {int(value.shape[1])}.")


__all__ = [
    "JEPA_ADVANTAGE_GPS_CONDITIONS",
    "JEPA_ADVANTAGE_IMAGE_CONDITIONS",
    "ObservabilityAwareFusion",
    "is_jepa_advantage_condition",
]
