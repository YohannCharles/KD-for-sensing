from dataclasses import dataclass
import time
from typing import Any

import torch

from kd_sensing.data.temporal_missing import apply_training_temporal_missing
from kd_sensing.engine.prediction_objectives import compute_prediction_loss, prepare_prediction_targets
from kd_sensing.engine.runtime import autocast_context, prepare_task_batch, prepare_task_labels, run_model_step
from kd_sensing.engine.scalar_metrics import materialize_batch_scalars, mean_metric_term
from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, ForwardControls


@dataclass(frozen=True)
class BatchStepResult:
    batch: dict[str, torch.Tensor]
    labels: torch.Tensor
    primary_logits: torch.Tensor
    total_loss: torch.Tensor
    task_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    scalar_diagnostics: dict[str, Any]
    accuracy: float
    metric_numerators: dict[str, float]
    metric_denominators: dict[str, float]
    timings: dict[str, float]


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
        self.timing_enabled = _batch_timing_enabled(training_cfg)

    def run(self, raw_batch, *, epoch: int, step: int) -> BatchStepResult:
        context = self.context
        batch = apply_training_temporal_missing(prepare_task_batch(raw_batch), self.cfg, epoch=epoch, step=step)
        labels = prepare_task_labels(
            batch,
            num_pred=context.num_pred,
            device=context.device,
            non_blocking=context.non_blocking,
        )
        observations = labels.ne(-100).sum().to(dtype=torch.float32)
        if not torch.is_nonzero(observations.gt(0)):
            raise ValueError("Training has zero effective beam-label observations.")
        self.optimizer.zero_grad()
        timings: dict[str, float] = {}
        with autocast_context(self.amp_enabled, context.device, self.amp_dtype):
            controls = ForwardControls()
            for extension, state in zip(self.extensions, self.extension_states):
                controls = controls.merge(
                    extension.before_forward(context, state, batch, labels, epoch=epoch, step=step)
                )
            controls = _respect_temporal_availability(controls, batch)
            started = time.perf_counter() if self.timing_enabled else None
            step_output = run_model_step(
                context.primary_model,
                self.task,
                batch,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                force_modality_mask=controls.force_modality_mask,
                extra_model_kwargs=controls.model_kwargs,
            )
            if started is not None:
                timings["forward_time"] = time.perf_counter() - started
            state = BatchState(
                epoch=epoch,
                step=step,
                batch=batch,
                labels=labels,
                primary_output=step_output.model_output,
                primary_logits=step_output.logits,
                controls=controls,
            )
            base = self._base_loss(state)
            total_loss = base.total_loss
            task_loss = base.task_loss
            diagnostics = dict(base.diagnostics)
            for extension, extension_state in zip(self.extensions, self.extension_states):
                extra = extension.after_forward(context, extension_state, state)
                if extra is None:
                    continue
                total_loss = total_loss + extra.total
                diagnostics.update(extra.diagnostics)
            combined = compute_prediction_loss(
                step_output.model_output,
                prepare_prediction_targets(labels=labels, auxiliary_targets={}, cfg=self.cfg),
                self.cfg,
                reference=step_output.logits,
                beam_total_loss=total_loss,
                beam_task_loss=task_loss,
            )
            total_loss = combined.total
            task_loss = combined.primary
            auxiliary_loss = total_loss - task_loss
            diagnostics.update(combined.diagnostics)
            state.total_loss = total_loss
            state.task_loss = task_loss
            state.auxiliary_loss = auxiliary_loss
        timings.update(self._backward_and_step(total_loss, state))
        terms = _metric_terms(total_loss, task_loss, auxiliary_loss, step_output.logits, labels, observations)
        numerators, denominators, diagnostics = materialize_batch_scalars(terms, diagnostics)
        accuracy = numerators["acc"] / max(denominators["acc"], 1.0)
        return BatchStepResult(
            batch=batch,
            labels=labels,
            primary_logits=step_output.logits,
            total_loss=total_loss,
            task_loss=task_loss,
            auxiliary_loss=auxiliary_loss,
            scalar_diagnostics=diagnostics,
            accuracy=float(accuracy),
            metric_numerators=numerators,
            metric_denominators=denominators,
            timings=timings,
        )

    def _base_loss(self, state: BatchState) -> BaseLossResult:
        for extension, extension_state in zip(self.extensions, self.extension_states):
            result = extension.compute_base_loss(self.context, extension_state, state)
            if result is not None:
                return result
        logits = state.primary_logits.reshape(-1, self.context.num_classes)
        loss = self.context.task_criterion(logits, state.labels.flatten())
        return BaseLossResult(total_loss=loss, task_loss=loss, auxiliary_loss=loss * 0.0)

    def _backward_and_step(self, total_loss: torch.Tensor, state: BatchState) -> dict[str, float]:
        started = time.perf_counter() if self.timing_enabled else None
        grad_clip = self.training_cfg.get("grad_clip")
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(total_loss).backward()
            if grad_clip or self.health_tracker is not None:
                self.grad_scaler.unscale_(self.optimizer)
            for extension, extension_state in zip(self.extensions, self.extension_states):
                extension.after_backward(self.context, extension_state, state)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.context.primary_model.parameters(), grad_clip)
            if self.health_tracker is not None:
                self.health_tracker.observe_gradients()
            backward = time.perf_counter() - started if started is not None else None
            step_started = time.perf_counter() if self.timing_enabled else None
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            for extension, extension_state in zip(self.extensions, self.extension_states):
                extension.after_backward(self.context, extension_state, state)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.context.primary_model.parameters(), grad_clip)
            if self.health_tracker is not None:
                self.health_tracker.observe_gradients()
            backward = time.perf_counter() - started if started is not None else None
            step_started = time.perf_counter() if self.timing_enabled else None
            self.optimizer.step()
        for extension, extension_state in zip(self.extensions, self.extension_states):
            extension.after_optimizer_step(self.context, extension_state, state)
        if backward is None or step_started is None:
            return {}
        return {"backward_time": backward, "optimizer_step_time": time.perf_counter() - step_started}


def _respect_temporal_availability(controls: ForwardControls, batch: dict[str, Any]) -> ForwardControls:
    missing = controls.model_kwargs.get("missing_mask")
    available = batch.get("available_modalities")
    if not torch.is_tensor(missing) or not torch.is_tensor(available):
        return controls
    mask = missing.to(dtype=torch.bool)
    usable = available.to(device=mask.device, dtype=torch.bool)
    if mask.shape != usable.shape:
        raise ValueError("U0 missing_mask must match temporal available_modalities.")
    if not bool(usable.any(dim=1).all().item()):
        raise ValueError("U0 temporal missing must retain one modality per sample.")
    mask = mask & usable
    empty = ~mask.any(dim=1)
    if bool(empty.any().item()):
        mask[empty, usable[empty].to(dtype=torch.int64).argmax(dim=1)] = True
    return ForwardControls(
        force_modality_mask=controls.force_modality_mask,
        model_kwargs={**controls.model_kwargs, "missing_mask": mask},
    )


def _metric_terms(
    total_loss: torch.Tensor,
    task_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    observations: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    valid = labels.ne(-100)
    correct = logits.argmax(dim=-1).eq(labels).logical_and(valid).sum().to(dtype=torch.float32).detach()
    return {
        "loss": mean_metric_term(total_loss, observations),
        "task_loss": mean_metric_term(task_loss, observations),
        "auxiliary_loss": mean_metric_term(auxiliary_loss, observations),
        "beam_loss": mean_metric_term(task_loss, observations),
        "acc": (correct, observations.detach()),
    }


def _batch_timing_enabled(training_cfg: dict[str, Any]) -> bool:
    timing = training_cfg.get("timing", {})
    return isinstance(timing, dict) and bool(timing.get("enabled")) and str(timing.get("profile", "")).lower() in {"host", "cuda_event"}
