from __future__ import annotations

from typing import Any

import torch

from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss, hist_beam_enabled
from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, TrainingExtension


class HistBeamTrainingExtension(TrainingExtension):
    name = "hist_beam"

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        enabled = _config_enables_hist_beam(context.cfg)
        return {"enabled": enabled}

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: dict[str, Any],
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        if not state.get("enabled"):
            return None
        diagnostics = batch_state.student_output.diagnostics
        output = {
            "logits": batch_state.student_logits,
            **diagnostics,
        }
        if not hist_beam_enabled(context.cfg, output):
            return None
        result = compute_hist_beam_loss(
            output,
            batch_state.labels,
            cfg=context.cfg,
            scene_labels=_scene_labels_from_batch(batch_state.batch, device=context.device),
            num_classes=context.num_classes,
        )
        zero = batch_state.student_logits.sum() * 0.0
        return BaseLossResult(
            total_loss=result.total,
            task_loss=result.hierarchical if result.hierarchical is not None else result.total,
            distill_loss=zero,
            diagnostics=result.diagnostics,
        )


def _config_enables_hist_beam(cfg: dict[str, Any]) -> bool:
    hist_cfg = cfg.get("hist_beam")
    if isinstance(hist_cfg, dict) and hist_cfg.get("enabled") is False:
        return False
    return cfg.get("model", {}).get("student", {}).get("type") == "hist_beam_fusion" or bool(hist_cfg)


def _scene_labels_from_batch(batch: dict[str, Any], *, device: torch.device) -> torch.Tensor | None:
    raw = batch.get("scene_id")
    if raw is None and isinstance(batch.get("metadata"), dict):
        raw = batch["metadata"].get("scene_id")
    if raw is None:
        return None
    if torch.is_tensor(raw):
        labels = raw.to(device=device, dtype=torch.long)
    elif isinstance(raw, (list, tuple)):
        values = [int(item) for item in raw]
        labels = torch.tensor(values, device=device, dtype=torch.long)
    else:
        labels = torch.tensor([int(raw)], device=device, dtype=torch.long)
    unique = sorted(int(value) for value in labels.detach().cpu().unique().tolist())
    remap = {scene_id: index for index, scene_id in enumerate(unique)}
    return torch.tensor([remap[int(value)] for value in labels.detach().cpu().tolist()], device=device, dtype=torch.long)


__all__ = ["HistBeamTrainingExtension"]
