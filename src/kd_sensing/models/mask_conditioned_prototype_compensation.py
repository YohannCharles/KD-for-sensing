"""Static missing-mask compensation over shared beam-prototype evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TSPC_METHODS = ("M0", "M1", "M2", "M3")
TSPC_MASK_NAMES = (
    "missing_image",
    "image_only",
    "missing_lidar",
    "lidar_only",
    "missing_radar",
    "radar_only",
    "missing_gps",
    "gps_only",
    "missing_image_lidar",
    "missing_image_radar",
    "missing_image_gps",
    "missing_lidar_radar",
    "missing_lidar_gps",
    "missing_radar_gps",
)
TSPC_AVAILABLE_COUNTS = (3, 1, 3, 1, 3, 1, 3, 1, 2, 2, 2, 2, 2, 2)


def _logit(probability: float) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError("initial_weight must be strictly between zero and one.")
    return math.log(value / (1.0 - value))


class MaskConditionedPrototypeCompensation(nn.Module):
    """Learn at most one static interpolation weight per physical missing mask."""

    def __init__(
        self,
        method: str,
        *,
        initial_weight: float = 0.5,
        sensing_temperature: float = 1.0,
        radio_temperature: float = 1.0,
        mask_available_counts: Sequence[int] = TSPC_AVAILABLE_COUNTS,
    ) -> None:
        super().__init__()
        self.method = str(method)
        if self.method not in TSPC_METHODS:
            raise ValueError(f"Unknown TSPC method: {self.method}.")
        counts = torch.as_tensor(mask_available_counts, dtype=torch.long).reshape(-1)
        if len(counts) != len(TSPC_MASK_NAMES) or not bool(((counts >= 1) & (counts <= 3)).all()):
            raise ValueError("mask_available_counts must define the 14 non-Full masks with counts in [1,3].")
        if min(float(sensing_temperature), float(radio_temperature)) <= 0:
            raise ValueError("Evidence temperatures must be positive.")
        self.register_buffer("mask_available_counts", counts, persistent=True)
        self.register_buffer("sensing_temperature", torch.tensor(float(sensing_temperature)), persistent=True)
        self.register_buffer("radio_temperature", torch.tensor(float(radio_temperature)), persistent=True)

        initial = _logit(initial_weight)
        if self.method == "M0":
            self.global_logit = nn.Parameter(torch.tensor(initial))
        elif self.method in {"M1", "M3"}:
            self.alpha_count = nn.Parameter(torch.full((3,), initial))
            if self.method == "M3":
                self.delta_mask = nn.Parameter(torch.zeros(len(TSPC_MASK_NAMES)))
        else:
            self.mask_logits = nn.Parameter(torch.full((len(TSPC_MASK_NAMES),), initial))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def lambda_table(self) -> torch.Tensor:
        if self.method == "M0":
            return torch.sigmoid(self.global_logit).expand(len(TSPC_MASK_NAMES))
        if self.method == "M2":
            return torch.sigmoid(self.mask_logits)
        group_logits = self.alpha_count.index_select(0, self.mask_available_counts - 1)
        if self.method == "M3":
            group_logits = group_logits + self.delta_mask
        return torch.sigmoid(group_logits)

    def lambdas(self, mask_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(mask_ids, device=self.sensing_temperature.device, dtype=torch.long).reshape(-1)
        if bool(((ids < -1) | (ids >= len(TSPC_MASK_NAMES))).any()):
            raise ValueError("mask_ids must be -1 for Full or in [0,13].")
        safe_ids = ids.clamp_min(0)
        values = self.lambda_table().index_select(0, safe_ids)
        return torch.where(ids >= 0, values, torch.zeros_like(values))

    def regularization(self) -> dict[str, torch.Tensor]:
        zero = self.lambda_table().sum() * 0.0
        delta_l2 = zero
        group_mean = zero
        if self.method == "M3":
            delta_l2 = self.delta_mask.square().mean()
            group_mean = torch.stack(
                [self.delta_mask[self.mask_available_counts == count].mean().square() for count in (1, 2, 3)]
            ).sum()
        if self.method in {"M1", "M3"}:
            group_lambda = torch.sigmoid(self.alpha_count)
            severity = F.relu(group_lambda[1] - group_lambda[0]) + F.relu(group_lambda[2] - group_lambda[1])
        else:
            severity = zero
        return {"delta": delta_l2, "group": group_mean, "severity": severity}

    def forward(
        self,
        sensing_evidence: torch.Tensor,
        radio_evidence: torch.Tensor,
        mask_ids: torch.Tensor,
        csi_available: torch.Tensor,
        *,
        base_probability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sensing_input = torch.as_tensor(sensing_evidence)
        radio_input = torch.as_tensor(radio_evidence, device=sensing_input.device)
        if sensing_input.ndim != 2 or radio_input.shape != sensing_input.shape:
            raise ValueError("sensing_evidence and radio_evidence must share shape [B,num_beams].")
        ids = torch.as_tensor(mask_ids, device=sensing_input.device, dtype=torch.long).reshape(-1)
        available = torch.as_tensor(csi_available, device=sensing_input.device, dtype=torch.bool).reshape(-1)
        if ids.shape != available.shape or len(ids) != len(sensing_input):
            raise ValueError("mask_ids and csi_available must have shape [B].")
        full = ids.eq(-1)
        active = available & ~full

        with torch.autocast(device_type=sensing_input.device.type, enabled=False):
            sensing = sensing_input.float()
            radio = radio_input.float()
            sensing_calibrated = sensing / self.sensing_temperature.float()
            radio_calibrated = radio / self.radio_temperature.float()
            weight = self.lambdas(ids).float() * active.to(torch.float32)
            fused = sensing_calibrated + weight[:, None] * (radio_calibrated - sensing_calibrated)
            final = torch.where(active[:, None], fused, sensing)
            probability = torch.softmax(final, dim=-1)
            if base_probability is not None:
                base = torch.as_tensor(base_probability, device=final.device, dtype=torch.float32)
                if base.shape != final.shape:
                    raise ValueError("base_probability must have shape [B,num_beams].")
                probability = torch.where(active[:, None], probability, base)
        return {
            "sensing_evidence": sensing,
            "sensing_evidence_calibrated": sensing_calibrated,
            "radio_evidence_calibrated": radio_calibrated,
            "lambda": weight,
            "final_evidence": final,
            "final_probability": probability,
            "active": active,
        }


__all__ = [
    "MaskConditionedPrototypeCompensation",
    "TSPC_AVAILABLE_COUNTS",
    "TSPC_MASK_NAMES",
    "TSPC_METHODS",
]
