from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kd_sensing.engine.debug_diagnostics import set_csi_debug_batch_source
from kd_sensing.engine.gps_coarse_anchor import (
    GpsCoarseAnchor,
    GpsCoarseAnchorConfig,
    compute_gps_coarse_anchor_loss,
)
from kd_sensing.engine.model_output import select_prediction_slots
from kd_sensing.engine.prediction_objectives import compute_prediction_loss, prepare_prediction_targets
from kd_sensing.engine.runtime import (
    autocast_context,
    prepare_task_auxiliary_targets,
    prepare_task_batch,
    prepare_task_labels,
    prepare_task_soft_beam_targets,
    run_model_step,
)
from kd_sensing.engine.training_extensions import (
    BaseLossResult,
    BatchState,
    ExtensionContext,
    ForwardControls,
)


@dataclass(frozen=True)
class BatchStepResult:
    batch: dict[str, torch.Tensor]
    labels: torch.Tensor
    primary_logits: torch.Tensor
    total_loss: torch.Tensor
    task_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    prediction_loss: Any
    extra_loss_values: dict[str, torch.Tensor]
    scalar_diagnostics: dict[str, float]
    accuracy: float


class BatchStepRunner:
    def __init__(
        self,
        *,
        cfg: dict,
        task: str,
        model_cfg: dict,
        training_cfg: dict,
        optimizer: torch.optim.Optimizer,
        grad_scaler,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
        extension_context: ExtensionContext,
        extensions: list,
        extension_states: list,
        health_tracker=None,
    ) -> None:
        self.cfg = cfg
        self.task = task
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.context = extension_context
        self.extensions = extensions
        self.extension_states = extension_states
        self.health_tracker = health_tracker

    def run(self, raw_batch, *, epoch: int, step: int, current_alpha: float) -> BatchStepResult:
        context = self.context
        batch = prepare_task_batch(raw_batch)
        labels = prepare_task_labels(
            batch,
            num_pred=context.num_pred,
            downsample_ratio=self.model_cfg.get("downsample_ratio", 1),
            device=context.device,
            non_blocking=context.non_blocking,
        )
        auxiliary_targets = prepare_task_auxiliary_targets(
            batch,
            num_pred=context.num_pred,
            device=context.device,
            non_blocking=context.non_blocking,
        )
        soft_beam_targets = prepare_task_soft_beam_targets(
            batch,
            cfg=self.cfg,
            num_pred=context.num_pred,
            num_classes=context.num_classes,
            downsample_ratio=self.model_cfg.get("downsample_ratio", 1),
            device=context.device,
            non_blocking=context.non_blocking,
        )
        prediction_targets = prepare_prediction_targets(
            labels=labels,
            auxiliary_targets=auxiliary_targets,
            cfg=self.cfg,
        )
        self.optimizer.zero_grad()
        with autocast_context(self.amp_enabled, context.device, self.amp_dtype):
            controls = ForwardControls()
            for extension, state in zip(self.extensions, self.extension_states):
                controls = controls.merge(
                    extension.before_forward(
                        context,
                        state,
                        batch,
                        labels,
                        epoch=epoch,
                    )
                )
            set_csi_debug_batch_source(context.primary_model, "train")
            primary_step = run_model_step(
                context.primary_model,
                self.task,
                batch,
                model_cfg=self.model_cfg["primary"],
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                force_modality_mask=controls.force_modality_mask,
                extra_model_kwargs=controls.model_kwargs,
            )
            primary_model_output = primary_step.model_output
            primary_outputs = primary_step.logits
            batch_state = BatchState(
                epoch=epoch,
                step=step,
                batch=batch,
                labels=labels,
                soft_beam_targets=soft_beam_targets,
                primary_output=primary_model_output,
                primary_logits=primary_outputs,
                controls=controls,
            )
            base_loss = self._compute_base_loss(
                batch_state=batch_state,
                batch=batch,
                labels=labels,
                primary_outputs=primary_outputs,
                current_alpha=current_alpha,
            )

            total_loss = base_loss.total_loss
            task_loss = base_loss.task_loss
            auxiliary_loss = base_loss.auxiliary_loss
            batch_state.total_loss = total_loss
            batch_state.task_loss = task_loss
            batch_state.auxiliary_loss = auxiliary_loss
            batch_state.active_modalities = base_loss.active_modalities
            scalar_diagnostics = dict(base_loss.diagnostics)
            extra_loss_values = {
                "beam_soft": primary_outputs.sum() * 0.0,
                "unimodal": primary_outputs.sum() * 0.0,
            }
            if "loss/beam_soft_target" in scalar_diagnostics:
                extra_loss_values["beam_soft"] = task_loss
            for extension, state in zip(self.extensions, self.extension_states):
                bundle = extension.after_forward(context, state, batch_state)
                if bundle is None:
                    continue
                total_loss = total_loss + bundle.total
                for key in extra_loss_values:
                    if key in bundle.components:
                        extra_loss_values[key] = bundle.components[key]
                scalar_diagnostics.update(bundle.diagnostics)
            prediction_loss = compute_prediction_loss(
                primary_model_output,
                prediction_targets,
                self.cfg,
                reference=primary_outputs,
                beam_total_loss=total_loss,
                beam_task_loss=task_loss,
            )
            total_loss = prediction_loss.total
            task_loss = prediction_loss.primary
            anchor_loss = _optional_gps_anchor_loss(
                primary_model_output.diagnostics,
                labels,
                self.cfg,
                reference=primary_outputs,
            )
            if anchor_loss is not None:
                total_loss = total_loss + anchor_loss[0]
                scalar_diagnostics.update(anchor_loss[1])
            scalar_diagnostics.update(prediction_loss.diagnostics)
            scalar_diagnostics.update(raymobtime_gate_scalar_diagnostics(primary_model_output.diagnostics))
            batch_state.total_loss = total_loss
            batch_state.task_loss = task_loss
            batch_state.auxiliary_loss = auxiliary_loss

        self._backward_and_step(total_loss, batch_state)
        prediction = torch.argmax(primary_outputs, dim=-1)
        valid = torch.sum(labels != -100).item()
        accuracy = (prediction == labels).sum().item() / max(valid, 1)
        return BatchStepResult(
            batch=batch,
            labels=labels,
            primary_logits=primary_outputs,
            total_loss=total_loss,
            task_loss=task_loss,
            auxiliary_loss=auxiliary_loss,
            prediction_loss=prediction_loss,
            extra_loss_values=extra_loss_values,
            scalar_diagnostics=scalar_diagnostics,
            accuracy=float(accuracy),
        )

    def _compute_base_loss(
        self,
        *,
        batch_state: BatchState,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        primary_outputs: torch.Tensor,
        current_alpha: float,
    ) -> BaseLossResult:
        context = self.context
        base_loss: BaseLossResult | None = None
        for extension, state in zip(self.extensions, self.extension_states):
            extension_loss = extension.compute_base_loss(context, state, batch_state)
            if extension_loss is None:
                continue
            if base_loss is not None:
                raise RuntimeError("Only one training extension may provide the base supervised loss.")
            base_loss = extension_loss

        if base_loss is not None:
            return base_loss

        primary_logits = primary_outputs.reshape(-1, context.num_classes)
        targets = labels.flatten()
        soft_targets = (
            batch_state.soft_beam_targets.reshape(-1, context.num_classes)
            if batch_state.soft_beam_targets is not None
            else None
        )
        task_loss = context.task_criterion(primary_logits, soft_targets if soft_targets is not None else targets)
        auxiliary_loss = primary_outputs.sum() * 0.0
        diagnostics = {}
        if soft_targets is not None:
            diagnostics["loss/beam_soft_target"] = float(task_loss.detach().cpu().item())
        return BaseLossResult(
            total_loss=task_loss,
            task_loss=task_loss,
            auxiliary_loss=auxiliary_loss,
            diagnostics=diagnostics,
        )

    def _backward_and_step(self, total_loss: torch.Tensor, batch_state: BatchState) -> None:
        grad_clip = self.training_cfg.get("grad_clip", None)
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(total_loss).backward()
            if grad_clip or batch_state.active_modalities is not None or self.health_tracker is not None:
                self.grad_scaler.unscale_(self.optimizer)
            for extension, state in zip(self.extensions, self.extension_states):
                extension.after_backward(self.context, state, batch_state)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.context.primary_model.parameters(), grad_clip)
            if self.health_tracker is not None:
                self.health_tracker.observe_gradients()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            return
        total_loss.backward()
        for extension, state in zip(self.extensions, self.extension_states):
            extension.after_backward(self.context, state, batch_state)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.context.primary_model.parameters(), grad_clip)
        if self.health_tracker is not None:
            self.health_tracker.observe_gradients()
        self.optimizer.step()


def raymobtime_gate_scalar_diagnostics(diagnostics: dict[str, Any]) -> dict[str, float]:
    gates = diagnostics.get("task_gates") or diagnostics.get("gates")
    modalities = diagnostics.get("gate_modalities") or diagnostics.get("modalities")
    if not isinstance(gates, dict) or not isinstance(modalities, (list, tuple)):
        return {}
    result: dict[str, float] = {}
    for task, gate in gates.items():
        if not torch.is_tensor(gate) or gate.ndim != 2:
            continue
        means = gate.detach().float().mean(dim=0).cpu().tolist()
        for modality, value in zip(modalities, means):
            result[f"raymobtime/gate/{task}/{modality}"] = float(value)
    return result


def _optional_gps_anchor_loss(
    diagnostics: dict[str, Any],
    labels: torch.Tensor,
    cfg: dict[str, Any],
    *,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]] | None:
    raw_cfg = cfg.get("coarse_anchor")
    if not isinstance(raw_cfg, dict):
        model_cfg = cfg.get("model", {}).get("primary", {}) if isinstance(cfg.get("model"), dict) else {}
        raw_cfg = model_cfg.get("coarse_anchor") if isinstance(model_cfg, dict) else None
    if not isinstance(raw_cfg, dict) or not bool(raw_cfg.get("enabled", False)):
        return None
    if "gps_anchor_coarse_logits" not in diagnostics:
        return None
    anchor_cfg = GpsCoarseAnchorConfig.from_mapping(raw_cfg)
    coarse_logits = select_prediction_slots(diagnostics["gps_anchor_coarse_logits"], labels.shape[1])
    beam_scores = diagnostics.get("gps_anchor_beam_scores")
    if torch.is_tensor(beam_scores):
        beam_scores = select_prediction_slots(beam_scores, labels.shape[1])
    center = diagnostics.get("gps_anchor_center_beam")
    confidence = diagnostics.get("gps_anchor_confidence")
    residual = diagnostics.get("gps_anchor_residual_anchor_beam")
    if not torch.is_tensor(center):
        center = torch.zeros(labels.shape, device=reference.device, dtype=torch.long)
    else:
        center = center[:, -labels.shape[1] :].to(device=reference.device)
    if not torch.is_tensor(confidence):
        confidence = torch.ones(labels.shape, device=reference.device, dtype=reference.dtype)
    else:
        confidence = confidence[:, -labels.shape[1] :].to(device=reference.device, dtype=reference.dtype)
    if not torch.is_tensor(residual):
        residual = center
    else:
        residual = residual[:, -labels.shape[1] :].to(device=reference.device)
    anchor = GpsCoarseAnchor(
        coarse_logits=coarse_logits,
        center_beam=center,
        confidence=confidence,
        residual_anchor_beam=residual,
        beam_scores=beam_scores,
        metadata={"anchor_source": "gps_neural_coarse"},
    )
    return compute_gps_coarse_anchor_loss(anchor, labels, anchor_cfg)
