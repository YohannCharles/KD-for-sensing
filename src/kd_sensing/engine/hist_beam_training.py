from __future__ import annotations

from typing import Any

import torch

from kd_sensing.engine.batch import (
    prepare_history_anchor_inputs,
    prepare_path_descriptors,
    prepare_path_semantic_labels,
    prepare_radio_semantic_labels,
)
from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss, hist_beam_enabled
from kd_sensing.engine.hist_beam_residuals import history_anchor_enabled, num_delta_classes_from_config
from kd_sensing.engine.training_extensions import (
    BaseLossResult,
    BatchState,
    ExtensionContext,
    ForwardControls,
    TrainingExtension,
)


class HistBeamTrainingExtension(TrainingExtension):
    name = "hist_beam"

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        enabled = _config_enables_hist_beam(context.cfg)
        return {"enabled": enabled}

    def before_forward(
        self,
        context: ExtensionContext,
        state: dict[str, Any],
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
    ) -> ForwardControls:
        if not state.get("enabled") or not history_anchor_enabled(context.cfg):
            return ForwardControls()
        model_cfg = context.model_cfg.get("student", context.model_cfg)
        kwargs = prepare_history_anchor_inputs(
            batch,
            num_pred=context.num_pred,
            num_classes=num_delta_classes_from_config(context.cfg, default=context.num_classes),
            downsample_ratio=int(model_cfg.get("downsample_ratio", context.model_cfg.get("downsample_ratio", 1))),
            device=context.device,
            enabled=True,
            non_blocking=context.non_blocking,
            sample_ids=_sample_ids_from_batch(batch),
        )
        return ForwardControls(model_kwargs={key: value for key, value in kwargs.items() if key != "residual_labels"})

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
        path_targets = prepare_path_descriptors(
            batch_state.batch,
            num_pred=context.num_pred,
            device=context.device,
            non_blocking=context.non_blocking,
        )
        result = compute_hist_beam_loss(
            output,
            batch_state.labels,
            cfg=context.cfg,
            scene_labels=_scene_labels_from_batch(batch_state.batch, device=context.device),
            radio_semantic_labels=prepare_radio_semantic_labels(
                batch_state.batch,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
            ),
            path_semantic_labels=prepare_path_semantic_labels(
                batch_state.batch,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
            ),
            path_descriptors=path_targets[0] if path_targets is not None else None,
            path_descriptor_mask=path_targets[1] if path_targets is not None else None,
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
        values = [_stable_scene_label(item) for item in raw]
        labels = torch.tensor(values, device=device, dtype=torch.long)
    else:
        labels = torch.tensor([_stable_scene_label(raw)], device=device, dtype=torch.long)
    unique = sorted(int(value) for value in labels.detach().cpu().unique().tolist())
    remap = {scene_id: index for index, scene_id in enumerate(unique)}
    return torch.tensor([remap[int(value)] for value in labels.detach().cpu().tolist()], device=device, dtype=torch.long)


def _stable_scene_label(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value)
        return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 100000


def _sample_ids_from_batch(batch: dict[str, Any]) -> list[str] | None:
    metadata = batch.get("metadata")
    if isinstance(metadata, list):
        return [str(item.get("sample_id", index)) if isinstance(item, dict) else str(index) for index, item in enumerate(metadata)]
    if isinstance(metadata, dict):
        raw = metadata.get("sample_id")
        if isinstance(raw, (list, tuple)):
            return [str(item) for item in raw]
    return None


__all__ = ["HistBeamTrainingExtension"]
