from dataclasses import dataclass
import time
from typing import Any

import torch

from kd_sensing.data.difficulty import DifficultyContext, apply_configured_difficulty
from kd_sensing.engine.debug_diagnostics import set_csi_debug_batch_source
from kd_sensing.engine.objectives.metadata import resolve_prediction_objective
from kd_sensing.engine.prediction_objectives import (
    build_dba_aware_soft_targets,
    compute_prediction_loss,
    prepare_prediction_targets,
)
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
        self.objective = resolve_prediction_objective(cfg)
        self.difficulty_seed = int(cfg.get("experiment", {}).get("seed", 0))

    def run(self, raw_batch, *, epoch: int, step: int, current_alpha: float) -> BatchStepResult:
        if self.objective == "gps_conditioned_jepa":
            return self._run_jepa(raw_batch, epoch=epoch, step=step)
        return self._run_supervised(raw_batch, epoch=epoch, step=step, current_alpha=current_alpha)

    def _run_supervised(self, raw_batch, *, epoch: int, step: int, current_alpha: float) -> BatchStepResult:
        context = self.context
        batch = prepare_task_batch(raw_batch)
        batch = apply_configured_difficulty(
            batch,
            self.cfg,
            DifficultyContext(stage="train", split="train", seed=self.difficulty_seed, epoch=epoch, step=step),
        ).batch
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
        timings: dict[str, float] = {}
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
            forward_start = time.perf_counter()
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
            timings["forward_time"] = time.perf_counter() - forward_start
            loss_start = time.perf_counter()
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
            scalar_diagnostics.update(prediction_loss.diagnostics)
            batch_state.total_loss = total_loss
            batch_state.task_loss = task_loss
            batch_state.auxiliary_loss = auxiliary_loss
            timings["loss_time"] = time.perf_counter() - loss_start

        timings.update(self._backward_and_step(total_loss, batch_state))
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
            timings=timings,
        )

    def _run_jepa(self, raw_batch, *, epoch: int, step: int) -> BatchStepResult:
        context = self.context
        batch = prepare_task_batch(raw_batch)
        batch = apply_configured_difficulty(
            batch,
            self.cfg,
            DifficultyContext(stage="train", split="train", seed=self.difficulty_seed, epoch=epoch, step=step),
        ).batch
        self.optimizer.zero_grad()
        timings: dict[str, float] = {}
        with autocast_context(self.amp_enabled, context.device, self.amp_dtype):
            labels = _jepa_dummy_labels(batch, context)
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
            forward_start = time.perf_counter()
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
                extra_model_kwargs={
                    **controls.model_kwargs,
                    "jepa_epoch": int(epoch),
                    "jepa_step": int(step),
                },
            )
            timings["forward_time"] = time.perf_counter() - forward_start
            loss_start = time.perf_counter()
            primary_model_output = primary_step.model_output
            primary_outputs = primary_step.logits
            batch_state = BatchState(
                epoch=epoch,
                step=step,
                batch=batch,
                labels=labels,
                soft_beam_targets=None,
                primary_output=primary_model_output,
                primary_logits=primary_outputs,
                controls=controls,
            )
            base_loss = self._compute_base_loss(
                batch_state=batch_state,
                batch=batch,
                labels=labels,
                primary_outputs=primary_outputs,
                current_alpha=0.0,
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
            for extension, state in zip(self.extensions, self.extension_states):
                bundle = extension.after_forward(context, state, batch_state)
                if bundle is None:
                    continue
                total_loss = total_loss + bundle.total
                scalar_diagnostics.update(bundle.diagnostics)
            batch_state.total_loss = total_loss
            batch_state.task_loss = task_loss
            batch_state.auxiliary_loss = auxiliary_loss
            prediction_loss = _jepa_prediction_loss_bundle(total_loss, primary_outputs)
            timings["loss_time"] = time.perf_counter() - loss_start

        timings.update(self._backward_and_step(total_loss, batch_state))
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
            accuracy=0.0,
            timings=timings,
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
        dba_diagnostics: dict[str, float] = {}
        if soft_targets is None:
            dba_targets, dba_diagnostics = build_dba_aware_soft_targets(
                labels,
                num_classes=context.num_classes,
                cfg=self.cfg,
            )
            if dba_targets is not None:
                soft_targets = dba_targets.reshape(-1, context.num_classes)
        task_loss = context.task_criterion(primary_logits, soft_targets if soft_targets is not None else targets)
        auxiliary_loss = primary_outputs.sum() * 0.0
        diagnostics = dict(dba_diagnostics)
        if soft_targets is not None:
            if dba_diagnostics:
                diagnostics["loss/beam_circular_soft_ce"] = float(task_loss.detach().cpu().item())
                diagnostics["loss/beam_dba_aware"] = float(task_loss.detach().cpu().item())
            else:
                diagnostics["loss/beam_soft_target"] = float(task_loss.detach().cpu().item())
        return BaseLossResult(
            total_loss=task_loss,
            task_loss=task_loss,
            auxiliary_loss=auxiliary_loss,
            diagnostics=diagnostics,
        )

    def _backward_and_step(self, total_loss: torch.Tensor, batch_state: BatchState) -> dict[str, float]:
        grad_clip = self.training_cfg.get("grad_clip", None)
        backward_start = time.perf_counter()
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
            backward_time = time.perf_counter() - backward_start
            step_start = time.perf_counter()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            for extension, state in zip(self.extensions, self.extension_states):
                extension.after_optimizer_step(self.context, state, batch_state)
            return {"backward_time": backward_time, "optimizer_step_time": time.perf_counter() - step_start}
        total_loss.backward()
        for extension, state in zip(self.extensions, self.extension_states):
            extension.after_backward(self.context, state, batch_state)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.context.primary_model.parameters(), grad_clip)
        if self.health_tracker is not None:
            self.health_tracker.observe_gradients()
        backward_time = time.perf_counter() - backward_start
        step_start = time.perf_counter()
        self.optimizer.step()
        for extension, state in zip(self.extensions, self.extension_states):
            extension.after_optimizer_step(self.context, state, batch_state)
        return {"backward_time": backward_time, "optimizer_step_time": time.perf_counter() - step_start}

def _jepa_dummy_labels(batch: dict[str, torch.Tensor], context: ExtensionContext) -> torch.Tensor:
    missing = [key for key in ("image", "gps") if key not in batch]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"gps_conditioned_jepa objective requires image and GPS batch fields; missing: {missing_text}."
        )
    batch_size = int(batch["image"].shape[0])
    return torch.zeros(batch_size, context.num_pred, dtype=torch.long, device=context.device)


def _jepa_prediction_loss_bundle(total_loss: torch.Tensor, reference: torch.Tensor):
    from kd_sensing.engine.prediction_objectives import PredictionLossBundle

    zero = reference.sum() * 0.0
    loss_value = float(total_loss.detach().cpu().item())
    return PredictionLossBundle(
        total=total_loss,
        primary=total_loss,
        beam=zero,
        occlusion=zero,
        position=zero,
        multitask_total=zero,
        diagnostics={"loss/jepa": loss_value, "loss/primary": loss_value},
        los=zero,
        link_quality=zero,
        selection_multitask_total=zero,
        jepa=total_loss,
    )
