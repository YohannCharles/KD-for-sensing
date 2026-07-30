from dataclasses import dataclass
from typing import Any

import torch

from kd_sensing.engine.model_output import ModelOutput
from kd_sensing.losses.amber_full import amber_full_auxiliary_loss_from_output


@dataclass(frozen=True)
class PredictionLossBundle:
    total: torch.Tensor
    primary: torch.Tensor
    beam: torch.Tensor
    auxiliary: torch.Tensor
    diagnostics: dict[str, Any]


def compute_prediction_loss(
    model_output: ModelOutput,
    cfg: dict[str, Any],
    *,
    reference: torch.Tensor,
    beam_total_loss: torch.Tensor,
    beam_task_loss: torch.Tensor | None = None,
) -> PredictionLossBundle:
    beam = beam_task_loss if beam_task_loss is not None else beam_total_loss
    zero = reference.sum() * 0.0
    amber_auxiliary, diagnostics = amber_full_auxiliary_loss_from_output(model_output, cfg, zero)
    return PredictionLossBundle(
        total=beam_total_loss + amber_auxiliary,
        primary=beam,
        beam=beam,
        auxiliary=amber_auxiliary,
        diagnostics={"loss/beam": beam.detach(), "loss/primary": beam.detach(), **diagnostics},
    )


__all__ = ["PredictionLossBundle", "compute_prediction_loss"]
