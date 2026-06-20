from typing import Any

import torch

from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, TrainingExtension
from kd_sensing.losses.jepa import jepa_loss_from_output


class JepaTrainingExtension(TrainingExtension):
    name = "gps_conditioned_jepa"

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        result = jepa_loss_from_output(batch_state.primary_output, context.cfg)
        zero = batch_state.primary_logits.sum() * 0.0
        diagnostics = dict(result.diagnostics)
        diagnostics.update(_scalar_jepa_diagnostics(batch_state.primary_output.diagnostics))
        return BaseLossResult(
            total_loss=result.loss,
            task_loss=result.loss,
            auxiliary_loss=zero,
            diagnostics=diagnostics,
        )

    def after_optimizer_step(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> None:
        updater = getattr(context.primary_model, "update_target_encoder", None)
        if callable(updater):
            updater()


def _scalar_jepa_diagnostics(diagnostics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in (
        "jepa/mask_context_ratio",
        "jepa/mask_target_ratio",
        "jepa/target_tokens",
        "jepa/ema_decay",
        "jepa/latent_norm",
    ):
        value = diagnostics.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().cpu().item())
    return result


__all__ = ["JepaTrainingExtension"]
