from __future__ import annotations

from kd_sensing.losses.circular import (
    circular_soft_ce_loss,
    circular_soft_target,
    class_balanced_weights,
    focal_circular_soft_ce_loss,
)
from kd_sensing.losses.beam import FocalLoss, SoftTargetCrossEntropyLoss
from kd_sensing.losses.jepa import JepaLossResult, jepa_latent_prediction_loss, jepa_loss_from_output
from kd_sensing.losses.residual import ResidualFusionLoss, ResidualLossConfig, gate_target_from_gps_error
from kd_sensing.losses.topk_candidate_losses import TopKCandidateSelectorLoss, TopKCandidateSelectorLossConfig

__all__ = [
    "FocalLoss",
    "JepaLossResult",
    "ResidualFusionLoss",
    "ResidualLossConfig",
    "SoftTargetCrossEntropyLoss",
    "TopKCandidateSelectorLoss",
    "TopKCandidateSelectorLossConfig",
    "circular_soft_ce_loss",
    "circular_soft_target",
    "class_balanced_weights",
    "focal_circular_soft_ce_loss",
    "gate_target_from_gps_error",
    "jepa_latent_prediction_loss",
    "jepa_loss_from_output",
]
