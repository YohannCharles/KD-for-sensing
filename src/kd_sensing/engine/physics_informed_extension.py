from __future__ import annotations

from typing import Any

from kd_sensing.engine.training_extensions import BatchState, ExtensionContext, LossBundle, TrainingExtension
from kd_sensing.losses.physics_informed import PhysicsInformedBeamLoss


class PhysicsInformedTrainingExtension(TrainingExtension):
    name = "physics_informed"

    def setup(self, context: ExtensionContext) -> PhysicsInformedBeamLoss:
        cfg = context.cfg.get("loss", {}).get("physics", {})
        return PhysicsInformedBeamLoss(**cfg)

    def after_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> LossBundle | None:
        if not isinstance(state, PhysicsInformedBeamLoss):
            return None
        output = {"logits": batch_state.primary_logits, **batch_state.primary_output.diagnostics}
        result = state.compute(output, batch_state.batch, batch_state.labels)
        components = {key: value for key, value in result["components"].items() if hasattr(value, "detach")}
        return LossBundle(
            total=result["loss"] - components.get("beam_loss", result["loss"]),
            components=components,
            diagnostics=dict(result["diagnostics"]),
        )


__all__ = ["PhysicsInformedTrainingExtension"]
