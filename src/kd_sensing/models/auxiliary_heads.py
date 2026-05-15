from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def resolve_auxiliary_heads(config: bool | dict[str, Any] | None) -> dict[str, bool]:
    if isinstance(config, bool):
        enabled = bool(config)
        return {"enabled": enabled, "occlusion": enabled, "position": enabled}
    if config is None:
        return {"enabled": False, "occlusion": False, "position": False}
    if not isinstance(config, dict):
        raise TypeError("auxiliary_heads must be a bool, mapping, or None.")
    enabled = bool(config.get("enabled", config.get("enable", False)))
    occlusion = bool(config.get("occlusion", config.get("occlusion_head", enabled)))
    position = bool(config.get("position", config.get("position_head", enabled)))
    if not enabled:
        occlusion = bool(config.get("occlusion", False))
        position = bool(config.get("position", False))
        enabled = occlusion or position
    return {"enabled": enabled, "occlusion": occlusion, "position": position}


class TemporalAuxiliaryHeads(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_pred: int,
        auxiliary_heads: bool | dict[str, Any] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_pred = int(num_pred)
        if self.num_pred <= 0:
            raise ValueError(f"num_pred must be positive, got {num_pred}.")
        self.config = resolve_auxiliary_heads(auxiliary_heads)
        self.occlusion_head = (
            nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Dropout(float(dropout)),
                nn.Linear(self.input_dim, 1),
            )
            if self.config["occlusion"]
            else None
        )
        self.position_head = (
            nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Dropout(float(dropout)),
                nn.Linear(self.input_dim, 2),
            )
            if self.config["position"]
            else None
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config["enabled"])

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        if features.ndim != 3:
            raise ValueError(f"auxiliary heads expect [B, T, D] features, got {tuple(features.shape)}.")
        if int(features.shape[1]) < self.num_pred:
            raise ValueError(
                f"auxiliary heads require at least {self.num_pred} temporal slots, got {int(features.shape[1])}."
            )
        future_features = features[:, -self.num_pred :, :]
        output: dict[str, torch.Tensor] = {}
        if self.occlusion_head is not None:
            output["occlusion_logits"] = self.occlusion_head(future_features).squeeze(-1)
        if self.position_head is not None:
            output["position"] = self.position_head(future_features)
        return output


def temporal_output_with_optional_auxiliary(
    *,
    logits: torch.Tensor,
    input_features: torch.Tensor,
    output_features: torch.Tensor,
    auxiliary_heads: TemporalAuxiliaryHeads,
):
    auxiliary = auxiliary_heads(output_features)
    if not auxiliary:
        return logits, input_features, output_features
    return {
        "logits": logits,
        "input_features": input_features,
        "output_features": output_features,
        **auxiliary,
    }


__all__ = [
    "TemporalAuxiliaryHeads",
    "resolve_auxiliary_heads",
    "temporal_output_with_optional_auxiliary",
]
