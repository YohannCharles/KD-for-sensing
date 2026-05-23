from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kd_sensing.engine.debug_diagnostics import set_csi_debug_batch_source
from kd_sensing.engine.prediction_objectives import compute_prediction_loss, prepare_prediction_targets
from kd_sensing.engine.runtime import (
    autocast_context,
    prepare_task_auxiliary_targets,
    prepare_task_batch,
    prepare_task_labels,
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
    student_logits: torch.Tensor
    total_loss: torch.Tensor
    task_loss: torch.Tensor
    distill_loss: torch.Tensor
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
            set_csi_debug_batch_source(context.student_model, "train")
            student_step = run_model_step(
                context.student_model,
                self.task,
                batch,
                model_cfg=self.model_cfg["student"],
                seq_length=context.seq_length_student,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                force_modality_mask=controls.force_modality_mask,
                force_reliability_gate=controls.force_reliability_gate,
                gate_temperature=controls.gate_temperature,
            )
            student_model_output = student_step.model_output
            student_outputs = student_step.logits
            student_input_features = student_model_output.input_features
            student_out_features = student_model_output.output_features
            batch_state = BatchState(
                epoch=epoch,
                step=step,
                batch=batch,
                labels=labels,
                student_output=student_model_output,
                student_logits=student_outputs,
                controls=controls,
            )
            base_loss = self._compute_base_loss(
                batch_state=batch_state,
                batch=batch,
                labels=labels,
                student_outputs=student_outputs,
                student_input_features=student_input_features,
                student_out_features=student_out_features,
                current_alpha=current_alpha,
            )
            batch_state.teacher_diagnostics = base_loss.teacher_diagnostics

            total_loss = base_loss.total_loss
            task_loss = base_loss.task_loss
            distill_loss = base_loss.distill_loss
            batch_state.total_loss = total_loss
            batch_state.task_loss = task_loss
            batch_state.distill_loss = distill_loss
            batch_state.active_modalities = base_loss.active_modalities
            scalar_diagnostics = dict(base_loss.diagnostics)
            extra_loss_values = {
                "beam_soft": student_outputs.sum() * 0.0,
                "unimodal": student_outputs.sum() * 0.0,
                "counterfactual": student_outputs.sum() * 0.0,
                "prior_regularization": student_outputs.sum() * 0.0,
                "reliability_kd": student_outputs.sum() * 0.0,
            }
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
                student_model_output,
                prediction_targets,
                self.cfg,
                reference=student_outputs,
                beam_total_loss=total_loss,
                beam_task_loss=task_loss,
            )
            total_loss = prediction_loss.total
            task_loss = prediction_loss.primary
            if self.cfg.get("experiment", {}).get("objective", "beam") not in {"beam", "multitask"}:
                distill_loss = student_outputs.sum() * 0.0
            scalar_diagnostics.update(prediction_loss.diagnostics)
            scalar_diagnostics.update(raymobtime_gate_scalar_diagnostics(student_model_output.diagnostics))
            batch_state.total_loss = total_loss
            batch_state.task_loss = task_loss
            batch_state.distill_loss = distill_loss

        self._backward_and_step(total_loss, batch_state)
        prediction = torch.argmax(student_outputs, dim=-1)
        valid = torch.sum(labels != -100).item()
        accuracy = (prediction == labels).sum().item() / max(valid, 1)
        return BatchStepResult(
            batch=batch,
            labels=labels,
            student_logits=student_outputs,
            total_loss=total_loss,
            task_loss=task_loss,
            distill_loss=distill_loss,
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
        student_outputs: torch.Tensor,
        student_input_features: torch.Tensor | None,
        student_out_features: torch.Tensor | None,
        current_alpha: float,
    ) -> BaseLossResult:
        context = self.context
        base_loss: BaseLossResult | None = None
        for extension, state in zip(self.extensions, self.extension_states):
            extension_loss = extension.compute_base_loss(context, state, batch_state)
            if extension_loss is None:
                continue
            if base_loss is not None:
                raise RuntimeError("Only one training extension may provide the base distillation loss.")
            base_loss = extension_loss

        if base_loss is not None:
            return base_loss

        if context.teacher_model is not None:
            with torch.no_grad():
                set_csi_debug_batch_source(context.teacher_model, "train")
                teacher_step = run_model_step(
                    context.teacher_model,
                    self.task,
                    batch,
                    model_cfg=self.model_cfg["teacher"],
                    seq_length=context.seq_length_teacher,
                    num_pred=context.num_pred,
                    device=context.device,
                    non_blocking=context.non_blocking,
                )
                teacher_model_output = teacher_step.model_output
                teacher_outputs = teacher_step.logits
                teacher_input_features = teacher_model_output.input_features
                teacher_out_features = teacher_model_output.output_features
                teacher_diagnostics = teacher_model_output.diagnostics
        else:
            teacher_outputs, teacher_input_features, teacher_out_features = dummy_teacher(
                student_outputs,
                student_input_features,
                student_out_features,
            )
            teacher_diagnostics = {}
        batch_state.teacher_logits = teacher_outputs
        batch_state.teacher_input_features = teacher_input_features
        batch_state.teacher_output_features = teacher_out_features
        batch_state.teacher_diagnostics = teacher_diagnostics
        student_logits = student_outputs.reshape(-1, context.num_classes)
        teacher_logits = teacher_outputs.reshape(-1, context.num_classes)
        targets = labels.flatten()
        student_input_window = feature_prefix(
            student_input_features,
            context.seq_length_student - 1,
            name="student input_features",
        )
        teacher_input_window = feature_prefix(
            teacher_input_features,
            context.seq_length_teacher - 1,
            name="teacher input_features",
        )
        student_output_window = feature_tail(
            student_out_features,
            context.num_pred,
            name="student output_features",
        )
        teacher_output_window = feature_tail(
            teacher_out_features,
            context.num_pred,
            name="teacher output_features",
        )
        total_loss, task_loss, distill_loss = context.distiller(
            student_logits,
            teacher_logits,
            targets,
            student_input_window,
            teacher_input_window,
            student_output_window,
            teacher_output_window,
            current_alpha,
        )
        return BaseLossResult(
            total_loss=total_loss,
            task_loss=task_loss,
            distill_loss=distill_loss,
            teacher_diagnostics=teacher_diagnostics,
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
                torch.nn.utils.clip_grad_norm_(self.context.student_model.parameters(), grad_clip)
            if self.health_tracker is not None:
                self.health_tracker.observe_gradients()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            return
        total_loss.backward()
        for extension, state in zip(self.extensions, self.extension_states):
            extension.after_backward(self.context, state, batch_state)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.context.student_model.parameters(), grad_clip)
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


def dummy_teacher(
    student_outputs: torch.Tensor,
    student_input_features: torch.Tensor | None,
    student_out_features: torch.Tensor | None,
):
    return (
        torch.zeros_like(student_outputs),
        torch.zeros_like(student_input_features) if torch.is_tensor(student_input_features) else None,
        torch.zeros_like(student_out_features) if torch.is_tensor(student_out_features) else None,
    )


def feature_prefix(features: torch.Tensor | None, length: int, *, name: str) -> torch.Tensor | None:
    if features is None:
        return None
    if features.ndim < 2:
        raise ValueError(f"{name} must include a time dimension, got shape {tuple(features.shape)}.")
    if features.shape[1] < length:
        raise ValueError(f"{name} has {features.shape[1]} slots but {length} are required.")
    return features[:, :length, ...]


def feature_tail(features: torch.Tensor | None, length: int, *, name: str) -> torch.Tensor | None:
    if features is None:
        return None
    if features.ndim < 2:
        raise ValueError(f"{name} must include a time dimension, got shape {tuple(features.shape)}.")
    if features.shape[1] < length:
        raise ValueError(f"{name} has {features.shape[1]} slots but {length} are required.")
    return features[:, -length:, ...]
