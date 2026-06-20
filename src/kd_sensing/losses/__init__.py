from __future__ import annotations

from kd_sensing.losses.circular import (
    circular_soft_ce_loss,
    circular_soft_target,
    class_balanced_weights,
    focal_circular_soft_ce_loss,
)
from kd_sensing.losses.beam import FocalLoss, SoftTargetCrossEntropyLoss
from kd_sensing.losses.jepa import JepaLossResult, jepa_latent_prediction_loss, jepa_loss_from_output

__all__ = [
    "FocalLoss",
    "JepaLossResult",
    "SoftTargetCrossEntropyLoss",
    "circular_soft_ce_loss",
    "circular_soft_target",
    "class_balanced_weights",
    "focal_circular_soft_ce_loss",
    "jepa_latent_prediction_loss",
    "jepa_loss_from_output",
]
