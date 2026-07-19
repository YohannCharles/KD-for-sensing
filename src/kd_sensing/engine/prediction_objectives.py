from dataclasses import dataclass
from typing import Any

import torch

from kd_sensing.engine.model_output import ModelOutput
from kd_sensing.engine.objectives.metadata import resolve_prediction_objective
from kd_sensing.losses.amber_full import amber_full_auxiliary_loss_from_output
from kd_sensing.losses.amr_adapted import amr_adapted_auxiliary_loss_from_output


@dataclass(frozen=True)
class PredictionTargets:
    labels: torch.Tensor


@dataclass(frozen=True)
class PredictionLossBundle:
    total: torch.Tensor
    primary: torch.Tensor
    beam: torch.Tensor
    auxiliary: torch.Tensor
    diagnostics: dict[str, Any]


def prepare_prediction_targets(*, labels: torch.Tensor, auxiliary_targets: dict[str, torch.Tensor], cfg: dict[str, Any]) -> PredictionTargets:
    del auxiliary_targets
    resolve_prediction_objective(cfg)
    return PredictionTargets(labels=labels)


def compute_prediction_loss(
    model_output: ModelOutput,
    targets: PredictionTargets,
    cfg: dict[str, Any],
    *,
    reference: torch.Tensor,
    beam_total_loss: torch.Tensor,
    beam_task_loss: torch.Tensor | None = None,
) -> PredictionLossBundle:
    del targets
    resolve_prediction_objective(cfg)
    beam = beam_task_loss if beam_task_loss is not None else beam_total_loss
    zero = reference.sum() * 0.0
    amber_auxiliary, diagnostics = amber_full_auxiliary_loss_from_output(model_output, cfg, zero)
    amr_auxiliary, amr_diagnostics = amr_adapted_auxiliary_loss_from_output(model_output, cfg, zero)
    auxiliary = amber_auxiliary + amr_auxiliary
    diagnostics.update(amr_diagnostics)
    return PredictionLossBundle(
        total=beam_total_loss + auxiliary,
        primary=beam,
        beam=beam,
        auxiliary=auxiliary,
        diagnostics={"loss/beam": beam.detach(), "loss/primary": beam.detach(), **diagnostics},
    )


__all__ = ["PredictionLossBundle", "PredictionTargets", "compute_prediction_loss", "prepare_prediction_targets"]
