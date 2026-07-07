from dataclasses import dataclass
from pathlib import Path
import csv
import time
from typing import Any

import torch

from kd_sensing.data.difficulty.pipeline import apply_configured_difficulty
from kd_sensing.data.difficulty.schema import DifficultyContext
from kd_sensing.data.missing_mask import get_missing_pattern_name
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
        self._random_dropout_counts: dict[int, dict[tuple[str, int], int]] = {}
        self._amr_gate_rows: dict[int, list[dict[str, Any]]] = {}
        self._reliability_weight_rows: dict[int, list[dict[str, Any]]] = {}

    def run(self, raw_batch, *, epoch: int, step: int, current_alpha: float) -> BatchStepResult:
        if self.objective == "gps_conditioned_jepa":
            return self._run_jepa(raw_batch, epoch=epoch, step=step)
        return self._run_supervised(raw_batch, epoch=epoch, step=step, current_alpha=current_alpha)

    def _run_supervised(self, raw_batch, *, epoch: int, step: int, current_alpha: float) -> BatchStepResult:
        context = self.context
        batch = prepare_task_batch(raw_batch)
        difficulty_result = apply_configured_difficulty(
            batch,
            self.cfg,
            DifficultyContext(stage="train", split="train", seed=self.difficulty_seed, epoch=epoch, step=step),
        )
        batch = difficulty_result.batch
        self._collect_random_dropout_stats(batch, epoch=epoch)
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
            self._collect_amr_gate_stats(primary_model_output.diagnostics, epoch=epoch)
            self._collect_reliability_weight_stats(primary_model_output.diagnostics, epoch=epoch)
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
        difficulty_result = apply_configured_difficulty(
            batch,
            self.cfg,
            DifficultyContext(stage="train", split="train", seed=self.difficulty_seed, epoch=epoch, step=step),
        )
        batch = difficulty_result.batch
        self._collect_random_dropout_stats(batch, epoch=epoch)
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
            self._collect_amr_gate_stats(primary_model_output.diagnostics, epoch=epoch)
            self._collect_reliability_weight_stats(primary_model_output.diagnostics, epoch=epoch)
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

    def flush_epoch_artifacts(self, run_dir: str | Path, epoch: int) -> None:
        self._flush_random_dropout_stats(Path(run_dir), epoch)
        self._flush_amr_gate_stats(Path(run_dir), epoch)
        self._flush_reliability_weight_stats(Path(run_dir), epoch)

    def _collect_random_dropout_stats(self, batch: dict[str, Any], *, epoch: int) -> None:
        rows = batch.get("random_dropout_pattern_stats")
        if not isinstance(rows, list):
            return
        bucket = self._random_dropout_counts.setdefault(int(epoch), {})
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("pattern_or_available_set", "")).strip()
            if not name:
                continue
            missing_count = int(row.get("missing_count", 0) or 0)
            count = int(row.get("num_samples", 0) or 0)
            bucket[(name, missing_count)] = bucket.get((name, missing_count), 0) + count

    def _collect_amr_gate_stats(self, diagnostics: dict[str, Any], *, epoch: int) -> None:
        rows = diagnostics.get("amr_lite_gate_stats") if isinstance(diagnostics, dict) else None
        if not isinstance(rows, list):
            return
        target = self._amr_gate_rows.setdefault(int(epoch), [])
        for row in rows:
            if isinstance(row, dict):
                target.append({**row, "epoch": int(epoch)})

    def _collect_reliability_weight_stats(self, diagnostics: dict[str, Any], *, epoch: int) -> None:
        if not isinstance(diagnostics, dict):
            return
        weights = diagnostics.get("reliability_fusion_weights")
        mask = diagnostics.get("reliability_fusion_available_mask")
        if mask is None:
            mask = diagnostics.get("missing_mask")
        if not torch.is_tensor(weights) or not torch.is_tensor(mask):
            return
        weights_cpu = weights.detach().cpu().to(dtype=torch.float32)
        mask_cpu = mask.detach().cpu().to(dtype=torch.bool)
        if weights_cpu.ndim == 3 and weights_cpu.shape[-1] == 1:
            weights_cpu = weights_cpu.squeeze(-1)
        if weights_cpu.ndim != 2 or mask_cpu.ndim != 2 or tuple(weights_cpu.shape) != tuple(mask_cpu.shape):
            return
        metadata = diagnostics.get("metadata") if isinstance(diagnostics.get("metadata"), dict) else {}
        modalities = list(metadata.get("modalities") or metadata.get("enabled_modalities") or [])
        if len(modalities) != int(weights_cpu.shape[1]):
            modalities = [f"modality_{index}" for index in range(int(weights_cpu.shape[1]))]
        temperatures = diagnostics.get("bprr_modality_temperatures")
        temperatures_cpu = None
        if torch.is_tensor(temperatures) and int(temperatures.numel()) == int(weights_cpu.shape[1]):
            temperatures_cpu = temperatures.detach().cpu().to(dtype=torch.float32).flatten()
        gate_entropy = diagnostics.get("gate_entropy")
        gate_entropy_cpu = gate_entropy.detach().cpu().to(dtype=torch.float32).flatten() if torch.is_tensor(gate_entropy) else None
        target = self._reliability_weight_rows.setdefault(int(epoch), [])
        for row_mask in torch.unique(mask_cpu, dim=0):
            selected = (mask_cpu == row_mask).all(dim=1)
            if not bool(selected.any().item()):
                continue
            pattern = _safe_pattern_name(row_mask, modalities)
            selected_weights = weights_cpu[selected]
            selected_mask = mask_cpu[selected]
            selected_entropy = gate_entropy_cpu[selected] if gate_entropy_cpu is not None and int(gate_entropy_cpu.numel()) == int(mask_cpu.shape[0]) else None
            for index, modality in enumerate(modalities):
                values = selected_weights[:, index]
                available = selected_mask[:, index].to(dtype=torch.float32)
                target.append(
                    {
                        "epoch": int(epoch),
                        "pattern": pattern,
                        "modality": modality,
                        "mean_weight": float(values.mean().item()),
                        "std_weight": float(values.std(unbiased=False).item()) if int(values.numel()) > 1 else 0.0,
                        "available_rate": float(available.mean().item()),
                        "temperature": float(temperatures_cpu[index].item()) if temperatures_cpu is not None else "",
                        "gate_entropy": float(selected_entropy.mean().item()) if selected_entropy is not None else "",
                    }
                )

    def _flush_random_dropout_stats(self, run_dir: Path, epoch: int) -> None:
        bucket = self._random_dropout_counts.pop(int(epoch), None)
        if not bucket:
            return
        total = sum(bucket.values()) or 1
        rows = [
            {
                "epoch": int(epoch),
                "pattern_or_available_set": name,
                "num_samples": count,
                "fraction": float(count / total),
                "missing_count": missing_count,
            }
            for (name, missing_count), count in sorted(bucket.items())
        ]
        _append_csv(run_dir / "random_dropout_pattern_stats.csv", rows)

    def _flush_amr_gate_stats(self, run_dir: Path, epoch: int) -> None:
        rows = self._amr_gate_rows.pop(int(epoch), None)
        if rows:
            _append_csv(run_dir / "amr_lite_gate_stats.csv", rows)

    def _flush_reliability_weight_stats(self, run_dir: Path, epoch: int) -> None:
        rows = self._reliability_weight_rows.pop(int(epoch), None)
        if rows:
            _append_csv(run_dir / "reliability_weights_epoch.csv", rows)

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


def _safe_pattern_name(mask: torch.Tensor, modalities: list[str]) -> str:
    try:
        return get_missing_pattern_name(mask, modalities)
    except Exception:
        bits = "".join("1" if bool(value) else "0" for value in mask.flatten().tolist())
        return f"mask_{bits}"


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
