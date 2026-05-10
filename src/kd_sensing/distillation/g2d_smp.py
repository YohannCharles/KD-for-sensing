from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities


@dataclass(frozen=True)
class SMPMaskSummary:
    active_modalities: list[str]
    zeroed_parameters: int
    kept_parameters: int


class SMPScheduler:
    """Sequential Modality Prioritization scheduler.

    Epochs are zero-based. Each weak-to-strong modality is trained alone for
    ``per_modality_tau`` epochs, then all modalities are active.
    """

    def __init__(
        self,
        modalities: list[str] | tuple[str, ...] | None = None,
        *,
        per_modality_tau: int = 5,
        joint_tau: int = 30,
        prioritize_low_confidence_first: bool = True,
    ):
        self.modalities = list(normalize_modalities(tuple(modalities or MODALITY_ORDER), context="SMP modalities"))
        self.per_modality_tau = max(int(per_modality_tau), 1)
        self.joint_tau = max(int(joint_tau), 0)
        self.prioritize_low_confidence_first = bool(prioritize_low_confidence_first)

    def rank_modalities(self, confidence_avg: Mapping[str, float]) -> list[str]:
        missing = [name for name in self.modalities if name not in confidence_avg]
        if missing:
            raise KeyError(f"SMP confidence is missing modalities: {missing}.")
        return sorted(
            self.modalities,
            key=lambda name: (
                float(confidence_avg[name]) if self.prioritize_low_confidence_first else -float(confidence_avg[name]),
                self.modalities.index(name),
            ),
        )

    def active_modalities(self, epoch: int, confidence_avg: Mapping[str, float]) -> list[str]:
        ranking = self.rank_modalities(confidence_avg)
        phase = int(epoch) // self.per_modality_tau
        if phase < len(ranking):
            return [ranking[phase]]
        return list(self.modalities)


def apply_smp_gradient_mask(model, active_modalities: list[str] | tuple[str, ...]) -> SMPMaskSummary:
    active = set(normalize_modalities(tuple(active_modalities), context="active SMP modalities"))
    zeroed = 0
    kept = 0
    for name, param in model.named_parameters():
        modality = _parameter_modality(name)
        if modality is None:
            if param.grad is not None:
                kept += 1
            continue
        if modality not in active:
            if param.grad is not None:
                param.grad.zero_()
                zeroed += 1
        elif param.grad is not None:
            kept += 1
    return SMPMaskSummary(
        active_modalities=list(active_modalities),
        zeroed_parameters=zeroed,
        kept_parameters=kept,
    )


def _parameter_modality(name: str) -> str | None:
    for modality in MODALITY_ORDER:
        prefixes = (
            f"encoders.{modality}.",
            f"feature_projections.{modality}.",
            f"{modality}_encoder.",
            f"{modality}_feature_extractor.",
            f"{modality}_cnn_layers.",
            f"{modality}_projection.",
            f"{modality}_global_avg_pool.",
            f"{modality}_global_max_pool.",
        )
        if any(name.startswith(prefix) for prefix in prefixes):
            return modality
        if modality == "image" and name.startswith(("image_global_avg_pool.", "image_global_max_pool.")):
            return modality
        if modality == "radar" and name.startswith(("radar_global_avg_pool.", "radar_global_max_pool.")):
            return modality
        if modality == "lidar" and name.startswith(("lidar_global_avg_pool.", "lidar_global_max_pool.")):
            return modality
        if modality == "gps" and name.startswith("gps_projection."):
            return modality
        if modality == "mmwave" and name.startswith("mmwave_projection."):
            return modality
    return None


__all__ = [
    "SMPMaskSummary",
    "SMPScheduler",
    "apply_smp_gradient_mask",
]
