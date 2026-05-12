from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.distillation.craf_losses import (
    beam_soft_label_loss,
    counterfactual_sequence_ce,
    prior_regularization_loss,
    reliability_weighted_kd_loss,
    sequence_cross_entropy,
)
from kd_sensing.engine.runtime import run_model_step
from kd_sensing.engine.training_extensions import (
    BatchState,
    ExtensionContext,
    ForwardControls,
    LossBundle,
    TrainingExtension,
)
from kd_sensing.modalities import normalize_modalities


def generate_modality_dropout_mask(
    available_mask: torch.Tensor,
    *,
    drop_prob: float = 0.0,
    min_keep: int = 1,
) -> torch.Tensor:
    """Return a boolean keep mask for modality dropout."""

    available = available_mask.to(torch.bool)
    if available.ndim != 2:
        raise ValueError(f"available_mask must have shape [B, K], got {tuple(available.shape)}.")
    if not 0.0 <= float(drop_prob) <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}.")
    keep = available & (torch.rand(available.shape, device=available.device) >= float(drop_prob))
    min_keep = max(int(min_keep), 0)
    if min_keep > 0:
        keep = _enforce_min_keep(keep, available, min_keep=min_keep)
    return keep


def generate_counterfactual_drop_masks(
    available_mask: torch.Tensor,
    *,
    mode: str = "sample_one",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build counterfactual keep masks and one-hot dropped-modality masks."""

    available = available_mask.to(torch.bool)
    if available.ndim != 2:
        raise ValueError(f"available_mask must have shape [B, K], got {tuple(available.shape)}.")
    batch_size, modality_count = available.shape
    if mode == "sample_one":
        keep = available.clone()
        dropped = torch.zeros_like(available)
        for row in range(batch_size):
            candidates = torch.nonzero(available[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            choice = candidates[torch.randint(candidates.numel(), (1,), device=available.device).item()]
            keep[row, choice] = False
            dropped[row, choice] = True
        return [(keep, dropped)]
    if mode == "leave_one_out":
        masks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for modality_idx in range(modality_count):
            dropped = torch.zeros_like(available)
            dropped[:, modality_idx] = available[:, modality_idx]
            if not torch.any(dropped):
                continue
            keep = available.clone()
            keep[:, modality_idx] = False
            masks.append((keep, dropped))
        return masks
    raise ValueError("counterfactual mode must be 'sample_one' or 'leave_one_out'.")


def generate_context_marginal_masks(
    available_mask: torch.Tensor,
    *,
    num_samples: int = 1,
    min_keep: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Sample context masks A and paired masks A union {m} for marginal contribution."""

    available = available_mask.to(torch.bool)
    if available.ndim != 2:
        raise ValueError(f"available_mask must have shape [B, K], got {tuple(available.shape)}.")
    num_samples = max(int(num_samples), 0)
    if num_samples == 0:
        return []
    min_keep = max(int(min_keep), 0)
    masks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(num_samples):
        context = torch.zeros_like(available)
        with_target = torch.zeros_like(available)
        target = torch.zeros_like(available)
        for row in range(available.shape[0]):
            candidates = torch.nonzero(available[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            target_idx = candidates[torch.randint(candidates.numel(), (1,), device=available.device).item()]
            target[row, target_idx] = True
            context_candidates = candidates[candidates != target_idx]
            required = min(min_keep, int(context_candidates.numel()))
            if context_candidates.numel() > 0:
                max_keep = int(context_candidates.numel())
                keep_count = required
                if max_keep > required:
                    keep_count = int(
                        torch.randint(
                            required,
                            max_keep + 1,
                            (1,),
                            device=available.device,
                        ).item()
                    )
                order = torch.randperm(context_candidates.numel(), device=available.device)
                context[row, context_candidates[order[:keep_count]]] = True
            with_target[row] = context[row] | target[row]
        if torch.any(target):
            masks.append((context, with_target, target))
    return masks


def loss_delta_to_gate_target(
    full_loss: torch.Tensor,
    drop_loss: torch.Tensor,
    dropped_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    target_floor: float = 0.0,
    target_ceiling: float = 1.0,
) -> torch.Tensor:
    """Map per-sample loss degradation to modality gate targets."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if full_loss.shape != drop_loss.shape:
        raise ValueError("full_loss and drop_loss must have the same shape.")
    dropped = dropped_mask.to(torch.bool)
    if dropped.ndim != 2 or dropped.shape[0] != full_loss.shape[0]:
        raise ValueError("dropped_mask must have shape [B, K] aligned with full_loss.")
    contribution = torch.sigmoid((drop_loss - full_loss) / float(temperature))
    contribution = contribution.clamp(float(target_floor), float(target_ceiling))
    return dropped.to(contribution.dtype) * contribution.unsqueeze(1)


def loss_delta_to_binary_gate_target(
    delta: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    ignore_delta_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map CE deltas to binary reliability targets and an ignore-band valid mask."""

    if delta.ndim != 1:
        raise ValueError(f"delta must have shape [B], got {tuple(delta.shape)}.")
    mask = supervision_mask.to(torch.bool)
    if mask.ndim != 2 or mask.shape[0] != delta.shape[0]:
        raise ValueError("supervision_mask must have shape [B, K] aligned with delta.")
    eps = float(ignore_delta_eps)
    if eps < 0.0:
        raise ValueError(f"ignore_delta_eps must be non-negative, got {ignore_delta_eps}.")
    valid = delta.abs().gt(eps).unsqueeze(1) & mask
    target = delta.gt(eps).to(delta.dtype).unsqueeze(1).expand_as(mask)
    return target * mask.to(delta.dtype), valid


def masked_gate_mse_loss(
    reliability: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    mask = supervision_mask.to(torch.bool)
    if not torch.any(mask):
        return reliability.sum() * 0.0
    return F.mse_loss(reliability[mask], target.detach()[mask])


def _enforce_min_keep(keep: torch.Tensor, available: torch.Tensor, *, min_keep: int) -> torch.Tensor:
    keep = keep.clone()
    for row in range(keep.shape[0]):
        available_indices = torch.nonzero(available[row], as_tuple=False).flatten()
        if available_indices.numel() == 0:
            continue
        current = int(keep[row].sum().item())
        required = min(min_keep, int(available_indices.numel()))
        if current >= required:
            continue
        dropped_available = available_indices[~keep[row, available_indices]]
        if dropped_available.numel() == 0:
            continue
        restore_order = torch.randperm(dropped_available.numel(), device=keep.device)
        restore = dropped_available[restore_order[: required - current]]
        keep[row, restore] = True
    return keep


@dataclass
class CrafExtensionState:
    reliability_sums: dict[str, float]
    reliability_batches: int = 0


class CrafTrainingExtension(TrainingExtension):
    name = "craf"

    def setup(self, context: ExtensionContext) -> CrafExtensionState:
        del context
        return CrafExtensionState(reliability_sums={})

    def before_epoch(self, context: ExtensionContext, state: CrafExtensionState, *, epoch: int) -> None:
        del context, epoch
        state.reliability_sums.clear()
        state.reliability_batches = 0

    def before_forward(
        self,
        context: ExtensionContext,
        state: CrafExtensionState,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
    ) -> ForwardControls:
        del state, batch
        model = context.student_model
        supports_controls = getattr(model, "supports_reliability_controls", False)
        gate_temperature = (
            current_craf_gate_temperature(context.cfg, context.model_cfg["student"], epoch)
            if supports_controls
            else None
        )
        return ForwardControls(
            force_modality_mask=training_modality_mask(
                context.training_cfg,
                model,
                context.model_cfg["student"],
                batch_size=int(labels.shape[0]),
                device=context.device,
            ),
            force_reliability_gate=training_reliability_gate_override(
                context.training_cfg,
                model,
                epoch=epoch,
            ),
            gate_temperature=gate_temperature,
        )

    def after_forward(
        self,
        context: ExtensionContext,
        state: CrafExtensionState,
        batch_state: BatchState,
    ) -> LossBundle | None:
        losses = compute_craf_extra_losses(
            context.cfg,
            context.student_model,
            context.task,
            batch_state.batch,
            model_cfg=context.model_cfg["student"],
            seq_length=context.seq_length_student,
            num_pred=context.num_pred,
            num_classes=context.num_classes,
            labels=batch_state.labels,
            student_outputs=batch_state.student_logits,
            diagnostics=batch_state.student_output.diagnostics,
            teacher_diagnostics=batch_state.teacher_diagnostics,
            epoch=batch_state.epoch,
            gate_temperature=float(batch_state.controls.gate_temperature or 1.0),
            device=context.device,
            non_blocking=context.non_blocking,
        )
        reliability_summary = batch_reliability_summary(batch_state.student_output.diagnostics)
        if reliability_summary:
            state.reliability_batches += 1
            for modality, value in reliability_summary.items():
                state.reliability_sums[modality] = state.reliability_sums.get(modality, 0.0) + value
        return LossBundle(
            total=losses["total"],
            components={
                "beam_soft": losses["beam_soft"],
                "unimodal": losses["unimodal"],
                "counterfactual": losses["counterfactual"],
                "prior_regularization": losses["prior_regularization"],
                "reliability_kd": losses["reliability_kd"],
            },
            diagnostics=dict(losses.get("_diagnostics", {})),
        )

    def after_epoch(
        self,
        context: ExtensionContext,
        state: CrafExtensionState,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        del context, epoch
        if not state.reliability_batches:
            return {}
        return {
            "craf_reliability": {
                modality: float(value / state.reliability_batches)
                for modality, value in state.reliability_sums.items()
            }
        }


def training_modality_mask(
    training_cfg: dict,
    model,
    model_cfg: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not getattr(model, "supports_force_modality_mask", False):
        return None
    dropout_cfg = training_cfg.get("modality_dropout", {})
    enabled = bool(dropout_cfg.get("enabled", False))
    drop_prob = float(dropout_cfg.get("drop_prob", 0.0))
    if not enabled or drop_prob <= 0.0:
        return None
    modalities = normalize_modalities(model_cfg.get("modalities", ("image", "radar")), context="CRAF dropout modalities")
    available = torch.ones(batch_size, len(modalities), dtype=torch.bool, device=device)
    return generate_modality_dropout_mask(
        available,
        drop_prob=drop_prob,
        min_keep=int(dropout_cfg.get("min_keep", 1)),
    )


def training_reliability_gate_override(training_cfg: dict, model, *, epoch: int) -> float | None:
    if not getattr(model, "supports_reliability_controls", False):
        return None
    return 1.0 if epoch < craf_gate_start_epoch(training_cfg) else None


def craf_gate_start_epoch(training_cfg: dict) -> int:
    counterfactual_cfg = training_cfg.get("counterfactual", {})
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
    counterfactual_start = int(counterfactual_cfg.get("start_epoch", 0))
    return max(warmup_epochs, counterfactual_start)


def current_craf_gate_temperature(cfg: dict, model_cfg: dict, epoch: int) -> float:
    reliability_cfg = model_cfg.get("reliability", {})
    base_temperature = float(reliability_cfg.get("gate_temperature", 1.0))
    start_temperature = float(reliability_cfg.get("gate_temperature_start", base_temperature))
    end_temperature = float(reliability_cfg.get("gate_temperature_end", base_temperature))
    start_epoch = int(reliability_cfg.get("gate_temperature_start_epoch", craf_gate_start_epoch(cfg.get("training", {}))))
    default_anneal = max(int(cfg.get("training", {}).get("epochs", 0)) - start_epoch, 0)
    anneal_epochs = int(reliability_cfg.get("gate_temperature_anneal_epochs", default_anneal))
    if anneal_epochs <= 0:
        return max(end_temperature, 1e-6)
    progress = min(max((epoch - start_epoch) / float(anneal_epochs), 0.0), 1.0)
    return max(start_temperature + (end_temperature - start_temperature) * progress, 1e-6)


def compute_craf_extra_losses(
    cfg: dict,
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    num_classes: int,
    labels: torch.Tensor,
    student_outputs: torch.Tensor,
    diagnostics: dict,
    teacher_diagnostics: dict | None = None,
    epoch: int,
    gate_temperature: float,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor | dict[str, float]]:
    del num_classes
    zero = student_outputs.sum() * 0.0
    record_craf_diagnostics = getattr(model, "supports_reliability_controls", False)
    scalar_diagnostics: dict[str, float] = {}
    if record_craf_diagnostics:
        scalar_diagnostics["craf/gate_temperature"] = float(gate_temperature)
    losses = {
        "total": zero,
        "beam_soft": zero,
        "unimodal": zero,
        "counterfactual": zero,
        "prior_regularization": zero,
        "reliability_kd": zero,
        "_diagnostics": scalar_diagnostics,
    }

    loss_cfg = cfg.get("loss", {})
    beam_cfg = loss_cfg.get("beam_soft", {})
    beam_weight = float(beam_cfg.get("weight", 0.0))
    beam_enabled = bool(beam_cfg.get("enabled", beam_weight > 0.0)) and beam_weight > 0.0
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/beam_soft_weight"] = beam_weight if beam_enabled else 0.0
    if beam_enabled:
        losses["beam_soft"] = beam_soft_label_loss(
            student_outputs,
            labels,
            sigma=float(beam_cfg.get("sigma", 2.0)),
            circular=bool(beam_cfg.get("circular", True)),
            ignore_index=int(beam_cfg.get("ignore_index", -100)),
        )
        losses["total"] = losses["total"] + beam_weight * losses["beam_soft"]

    unimodal_cfg = loss_cfg.get("unimodal_aux", {})
    warmup_boundary = craf_gate_start_epoch(cfg.get("training", {}))
    unimodal_weight = scheduled_unimodal_weight(loss_cfg, model_cfg, epoch, warmup_boundary)
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/unimodal_aux_weight"] = float(unimodal_weight)
    if unimodal_weight > 0.0:
        unimodal_loss = unimodal_aux_loss(
            diagnostics.get("unimodal_logits"),
            labels,
            diagnostics.get("effective_modality_mask"),
            num_pred=num_pred,
            ignore_index=int(unimodal_cfg.get("ignore_index", -100)),
            zero=zero,
        )
        losses["unimodal"] = unimodal_loss
        losses["total"] = losses["total"] + unimodal_weight * unimodal_loss

    counterfactual_cfg = cfg.get("training", {}).get("counterfactual", {})
    counterfactual_weight = counterfactual_target_weight(loss_cfg, counterfactual_cfg)
    counterfactual_effective_weight = scheduled_gate_loss_weight(
        counterfactual_weight,
        epoch,
        start_epoch=warmup_boundary,
        ramp_epochs=int(loss_cfg.get("gate_ramp_epochs", counterfactual_cfg.get("gate_ramp_epochs", 0))),
    )
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/gate_weight_target"] = float(counterfactual_weight)
        losses["_diagnostics"]["loss/gate_weight_effective"] = float(counterfactual_effective_weight)
    counterfactual_enabled = bool(counterfactual_cfg.get("enabled", False))
    if (
        counterfactual_enabled
        and counterfactual_effective_weight > 0.0
        and epoch >= warmup_boundary
        and getattr(model, "supports_force_modality_mask", False)
    ):
        counterfactual_loss, counterfactual_diagnostics = counterfactual_gate_loss(
            model,
            task,
            batch,
            model_cfg=model_cfg,
            seq_length=seq_length,
            num_pred=num_pred,
            labels=labels,
            full_outputs=student_outputs,
            reliability=diagnostics.get("reliability"),
            effective_modality_mask=diagnostics.get("effective_modality_mask"),
            modalities=diagnostics.get("modalities"),
            mode=str(counterfactual_cfg.get("mode", "sample_one")),
            ignore_delta_eps=float(counterfactual_cfg.get("ignore_delta_eps", 0.0)),
            num_drop_per_batch=int(counterfactual_cfg.get("num_drop_per_batch", 1)),
            min_keep=int(
                counterfactual_cfg.get(
                    "min_keep",
                    cfg.get("training", {}).get("modality_dropout", {}).get("min_keep", 1),
                )
            ),
            no_grad_drop_forward=bool(counterfactual_cfg.get("no_grad_drop_forward", True)),
            gate_temperature=gate_temperature,
            device=device,
            non_blocking=non_blocking,
            zero=zero,
        )
        losses["counterfactual"] = counterfactual_loss
        losses["total"] = losses["total"] + counterfactual_effective_weight * counterfactual_loss
        losses["_diagnostics"].update(counterfactual_diagnostics)

    prior_cfg = loss_cfg.get("prior_regularization", {})
    prior_weight = float(prior_cfg.get("weight", 0.0))
    prior_enabled = bool(prior_cfg.get("enabled", prior_weight > 0.0)) and prior_weight > 0.0
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/prior_regularization_weight"] = prior_weight if prior_enabled else 0.0
        losses["_diagnostics"].update(craf_gate_scalar_diagnostics(diagnostics))
    if prior_enabled:
        gate = diagnostics.get("gate", diagnostics.get("reliability"))
        prior = diagnostics.get("prior")
        modality_mask = diagnostics.get("effective_modality_mask")
        if torch.is_tensor(gate) and torch.is_tensor(prior):
            losses["prior_regularization"] = prior_regularization_loss(
                gate,
                prior,
                modality_mask,
                loss_type=str(prior_cfg.get("loss_type", "mse")),
            )
            losses["total"] = losses["total"] + prior_weight * losses["prior_regularization"]

    kd_cfg = cfg.get("training", {}).get("reliability_kd", cfg.get("kd", {}))
    kd_weight = float(kd_cfg.get("weight", 0.0))
    kd_enabled = bool(kd_cfg.get("enabled", False)) and kd_weight > 0.0
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/reliability_kd_weight"] = kd_weight if kd_enabled else 0.0
    if kd_enabled:
        student_unimodal = diagnostics.get("unimodal_logits")
        teacher_unimodal = (teacher_diagnostics or {}).get("unimodal_logits")
        reliability = diagnostics.get("gate", diagnostics.get("reliability"))
        if not (torch.is_tensor(student_unimodal) and torch.is_tensor(teacher_unimodal) and torch.is_tensor(reliability)):
            raise ValueError("reliability_kd.enabled=true requires student and teacher unimodal CRAF logits.")
        losses["reliability_kd"] = reliability_weighted_kd_loss(
            student_unimodal,
            teacher_unimodal,
            reliability,
            modalities=diagnostics.get("modalities") or model_cfg.get("modalities") or [],
            use_modalities=kd_cfg.get("use_modalities", ["gps", "mmwave"]),
            temperature=float(kd_cfg.get("temperature", cfg.get("distillation", {}).get("temperature", 3.0))),
            modality_mask=diagnostics.get("effective_modality_mask"),
        )
        losses["total"] = losses["total"] + kd_weight * losses["reliability_kd"]
    return losses


def scheduled_unimodal_weight(loss_cfg: dict, model_cfg: dict, epoch: int, warmup_boundary: int) -> float:
    unimodal_cfg = loss_cfg.get("unimodal_aux", {})
    base_weight = float(unimodal_cfg.get("weight", model_cfg.get("unimodal_loss_weight", 0.0)))
    warmup_weight = optional_float(
        loss_cfg.get("uni_weight_warmup", unimodal_cfg.get("weight_warmup", None))
    )
    after_weight = optional_float(
        loss_cfg.get("uni_weight_after_warmup", unimodal_cfg.get("weight_after_warmup", None))
    )
    if epoch < warmup_boundary:
        return base_weight if warmup_weight is None else warmup_weight
    return base_weight if after_weight is None else after_weight


def counterfactual_target_weight(loss_cfg: dict, counterfactual_cfg: dict) -> float:
    configured = optional_float(loss_cfg.get("gate_weight", None))
    legacy = float(counterfactual_cfg.get("weight", counterfactual_cfg.get("gate_loss_weight", 0.0)))
    if configured is not None and configured > 0.0:
        return configured
    return legacy


def scheduled_gate_loss_weight(
    target_weight: float,
    epoch: int,
    *,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    if target_weight <= 0.0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(target_weight)
    progress = min(max((epoch - start_epoch + 1) / float(ramp_epochs), 0.0), 1.0)
    return float(target_weight) * progress


def optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def unimodal_aux_loss(
    unimodal_logits,
    labels: torch.Tensor,
    effective_modality_mask,
    *,
    num_pred: int,
    ignore_index: int,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(unimodal_logits) or unimodal_logits.numel() == 0:
        return zero
    if unimodal_logits.ndim != 4:
        raise ValueError(f"unimodal_logits must have shape [B, K, H, C], got {tuple(unimodal_logits.shape)}.")
    horizon = num_pred
    if unimodal_logits.shape[2] != horizon:
        raise ValueError(
            "unimodal_logits horizon must exactly match num_pred future slots; "
            f"got {unimodal_logits.shape[2]} slots for num_pred={horizon}."
        )
    batch_size, modality_count, _, num_classes = unimodal_logits.shape
    expanded_labels = labels.unsqueeze(1).expand(batch_size, modality_count, -1)
    _, per_modality_loss = sequence_cross_entropy(
        unimodal_logits.reshape(batch_size * modality_count, horizon, num_classes),
        expanded_labels.reshape(batch_size * modality_count, horizon),
        ignore_index=ignore_index,
    )
    if torch.is_tensor(effective_modality_mask):
        mask = effective_modality_mask.to(device=unimodal_logits.device, dtype=torch.bool).reshape(-1)
    else:
        mask = torch.ones(batch_size * modality_count, dtype=torch.bool, device=unimodal_logits.device)
    if not torch.any(mask):
        return zero
    return per_modality_loss[mask].mean()


def counterfactual_gate_loss(
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    labels: torch.Tensor,
    full_outputs: torch.Tensor,
    reliability,
    effective_modality_mask,
    modalities,
    mode: str,
    ignore_delta_eps: float,
    num_drop_per_batch: int,
    min_keep: int,
    no_grad_drop_forward: bool,
    gate_temperature: float,
    device: torch.device,
    non_blocking: bool,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not torch.is_tensor(reliability) or not torch.is_tensor(effective_modality_mask):
        return zero, {}
    modality_names = diagnostic_modalities(modalities, reliability.shape[1])
    available = effective_modality_mask.detach()
    gate_losses = []
    stats = CounterfactualStats(modality_names)
    if mode == "shuffle":
        full_per_sample = counterfactual_sequence_ce(full_outputs, labels)
        for modality_idx, modality in enumerate(modality_names):
            target_mask = torch.zeros_like(available)
            target_mask[:, modality_idx] = available[:, modality_idx]
            if not torch.any(target_mask):
                continue
            context_manager = torch.no_grad() if no_grad_drop_forward else nullcontext()
            with context_manager:
                shuffled_batch = shuffled_modality_batch(batch, modality)
                shuffled_per_sample = counterfactual_forward_ce(
                    model,
                    task,
                    shuffled_batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=available,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                delta = shuffled_per_sample - full_per_sample
                target, valid_mask = loss_delta_to_binary_gate_target(
                    delta.detach(),
                    target_mask,
                    ignore_delta_eps=ignore_delta_eps,
                )
            stats.update(delta.detach(), target.detach(), target_mask, valid_mask)
            gate_losses.append(masked_gate_mse_loss(reliability, target, valid_mask))
    elif mode == "context_marginal":
        mask_specs = generate_context_marginal_masks(
            available,
            num_samples=num_drop_per_batch,
            min_keep=min_keep,
        )
        for context_mask, with_target_mask, target_mask in mask_specs:
            context_manager = torch.no_grad() if no_grad_drop_forward else nullcontext()
            with context_manager:
                context_per_sample = counterfactual_forward_ce(
                    model,
                    task,
                    batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=context_mask,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                with_target_per_sample = counterfactual_forward_ce(
                    model,
                    task,
                    batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=with_target_mask,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                delta = context_per_sample - with_target_per_sample
                target, valid_mask = loss_delta_to_binary_gate_target(
                    delta.detach(),
                    target_mask,
                    ignore_delta_eps=ignore_delta_eps,
                )
            stats.update(delta.detach(), target.detach(), target_mask, valid_mask)
            gate_losses.append(masked_gate_mse_loss(reliability, target, valid_mask))
    else:
        full_per_sample = counterfactual_sequence_ce(full_outputs, labels)
        drop_specs = generate_counterfactual_drop_masks(available, mode=mode)
        for keep_mask, dropped_mask in drop_specs:
            context_manager = torch.no_grad() if no_grad_drop_forward else nullcontext()
            with context_manager:
                drop_per_sample = counterfactual_forward_ce(
                    model,
                    task,
                    batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=keep_mask,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                delta = drop_per_sample - full_per_sample
                target, valid_mask = loss_delta_to_binary_gate_target(
                    delta.detach(),
                    dropped_mask,
                    ignore_delta_eps=ignore_delta_eps,
                )
            stats.update(delta.detach(), target.detach(), dropped_mask, valid_mask)
            gate_losses.append(masked_gate_mse_loss(reliability, target, valid_mask))
    if not gate_losses:
        return zero, stats.to_diagnostics()
    return torch.stack(gate_losses).mean(), stats.to_diagnostics()


def counterfactual_forward_ce(
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    labels: torch.Tensor,
    force_modality_mask: torch.Tensor,
    gate_temperature: float,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    step = run_model_step(
        model,
        task,
        batch,
        model_cfg=model_cfg,
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
        force_modality_mask=force_modality_mask,
        gate_temperature=gate_temperature,
    )
    return counterfactual_sequence_ce(step.logits, labels)


def shuffled_modality_batch(batch: dict[str, torch.Tensor], modality: str) -> dict[str, torch.Tensor]:
    keys_by_modality = {
        "image": ("image",),
        "radar": ("radar_ra", "radar_da"),
        "gps": ("gps",),
        "lidar": ("lidar",),
        "mmwave": ("mmwave",),
    }
    keys = keys_by_modality.get(str(modality), ())
    if not keys:
        return batch
    first = next((batch[key] for key in keys if key in batch and torch.is_tensor(batch[key])), None)
    if first is None or first.shape[0] <= 1:
        return batch
    order = torch.randperm(first.shape[0], device=first.device)
    shuffled = dict(batch)
    for key in keys:
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[0] == first.shape[0]:
            shuffled[key] = value.index_select(0, order)
    return shuffled


class CounterfactualStats:
    def __init__(self, modalities: list[str]):
        self.modalities = modalities
        self.delta_sum = torch.zeros(len(modalities), dtype=torch.float64)
        self.delta_count = torch.zeros(len(modalities), dtype=torch.float64)
        self.target_sum = torch.zeros(len(modalities), dtype=torch.float64)
        self.valid_count = torch.zeros(len(modalities), dtype=torch.float64)
        self.candidate_count = torch.zeros(len(modalities), dtype=torch.float64)

    def update(
        self,
        delta: torch.Tensor,
        target: torch.Tensor,
        candidate_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        candidate = candidate_mask.detach().to(torch.bool).cpu()
        valid = valid_mask.detach().to(torch.bool).cpu()
        delta_cpu = delta.detach().double().cpu()
        target_cpu = target.detach().double().cpu()
        for idx in range(len(self.modalities)):
            candidate_idx = candidate[:, idx]
            if torch.any(candidate_idx):
                self.delta_sum[idx] += delta_cpu[candidate_idx].sum()
                self.delta_count[idx] += candidate_idx.sum()
                self.candidate_count[idx] += candidate_idx.sum()
            valid_idx = valid[:, idx]
            if torch.any(valid_idx):
                self.target_sum[idx] += target_cpu[:, idx][valid_idx].sum()
                self.valid_count[idx] += valid_idx.sum()

    def to_diagnostics(self) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for idx, modality in enumerate(self.modalities):
            if self.delta_count[idx].item() > 0:
                diagnostics[f"cf/delta_mean_{modality}"] = float(
                    (self.delta_sum[idx] / self.delta_count[idx]).item()
                )
            if self.valid_count[idx].item() > 0:
                diagnostics[f"cf/target_mean_{modality}"] = float(
                    (self.target_sum[idx] / self.valid_count[idx]).item()
                )
            if self.candidate_count[idx].item() > 0:
                diagnostics[f"cf/target_valid_rate_{modality}"] = float(
                    (self.valid_count[idx] / self.candidate_count[idx]).item()
                )
        return diagnostics


def batch_reliability_summary(diagnostics: dict) -> dict[str, float]:
    reliability = diagnostics.get("reliability")
    modalities = diagnostics.get("modalities")
    if not torch.is_tensor(reliability) or reliability.ndim != 2:
        return {}
    modalities = diagnostic_modalities(modalities, reliability.shape[1])
    means = reliability.detach().float().mean(dim=0).cpu()
    return {str(modality): float(means[idx].item()) for idx, modality in enumerate(modalities)}


def craf_gate_scalar_diagnostics(diagnostics: dict) -> dict[str, float]:
    gate = diagnostics.get("gate", diagnostics.get("reliability"))
    prior = diagnostics.get("prior")
    residual = diagnostics.get("residual_logits")
    mask = diagnostics.get("effective_modality_mask")
    modalities = diagnostics.get("modalities")
    if not torch.is_tensor(gate) or gate.ndim != 2:
        return {}
    modality_names = diagnostic_modalities(modalities, gate.shape[1])
    if torch.is_tensor(mask):
        available = mask.detach().to(device=gate.device, dtype=torch.bool)
    else:
        available = torch.ones_like(gate, dtype=torch.bool)
    scalars: dict[str, float] = {}
    for idx, modality in enumerate(modality_names):
        modality_mask = available[:, idx]
        if torch.any(modality_mask):
            scalars[f"craf/gate_mean/{modality}"] = float(gate[:, idx][modality_mask].detach().float().mean().cpu().item())
        if torch.is_tensor(prior):
            prior_values = prior[:, idx] if prior.ndim == 2 else prior[idx].view(1).expand(gate.shape[0])
            scalars[f"craf/prior/{modality}"] = float(prior_values.detach().float().mean().cpu().item())
        if torch.is_tensor(residual):
            residual_values = residual[:, idx]
            if torch.any(modality_mask):
                scalars[f"craf/residual_logit_mean/{modality}"] = float(
                    residual_values[modality_mask].detach().float().mean().cpu().item()
                )
    return scalars


def diagnostic_modalities(modalities, modality_count: int) -> list[str]:
    if not isinstance(modalities, (tuple, list)) or len(modalities) != modality_count:
        return [f"modality_{idx}" for idx in range(modality_count)]
    return [str(modality) for modality in modalities]


_compute_craf_extra_losses = compute_craf_extra_losses
_unimodal_aux_loss = unimodal_aux_loss
