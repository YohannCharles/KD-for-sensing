from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.data.missing_mask import sample_missing_mask
from kd_sensing.data.temporal_missing_contract import TEMPORAL_SUPERSET_PAYLOAD_KEY
from kd_sensing.engine.evaluation_pass_runtime import sample_ids_from_batch
from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, ForwardControls, TrainingExtension
from kd_sensing.losses.modality_alignment_contrastive import amber_cma_analogue_loss
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.losses.u_mask_beam_jepa_prototype import add_prototype_alignment_losses


def u_mask_beam_jepa_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    prototype_bank: torch.nn.Module | None = None,
    use_beam_prototype_alignment: bool = False,
    lambda_proto: float = 0.0,
    lambda_modality_proto: float = 0.0,
    beam_label_sigma: float = 1.0,
    prototype_target_circular: bool = True,
    use_amber_cma_analogue: bool = False,
    lambda_amber_cma: float = 0.2,
    amber_cma_temperature: float = 0.2,
    sample_ids: list[str] | tuple[str, ...] | None = None,
    superset_output: dict[str, torch.Tensor] | None = None,
    use_superset_confidence_gated_kl: bool = False,
    lambda_superset_consistency: float = 0.0,
    superset_temperature: float = 2.0,
    router_oracle_weight: float = 0.1,
    circular_beam_distance: bool = True,
) -> dict[str, Any]:
    if use_beam_prototype_alignment and use_amber_cma_analogue:
        raise ValueError("BPA and the AMBER CMA analogue are mutually exclusive.")
    if float(lambda_amber_cma) < 0.0 or float(lambda_superset_consistency) < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative.")
    if float(amber_cma_temperature) <= 0.0 or float(superset_temperature) <= 0.0:
        raise ValueError("auxiliary loss temperatures must be positive.")

    logits = _as_prediction_logits(output["logits"])
    labels = _as_prediction_labels(labels, logits)
    unweighted_loss_beam = _beam_supervised_loss(logits, labels)
    loss_beam = unweighted_loss_beam
    loss = loss_beam

    loss, prototype_diagnostics = add_prototype_alignment_losses(
        loss,
        output,
        labels,
        prototype_bank=prototype_bank,
        enabled=use_beam_prototype_alignment,
        lambda_proto=lambda_proto,
        lambda_modality_proto=lambda_modality_proto,
        beam_label_sigma=beam_label_sigma,
        prototype_target_circular=prototype_target_circular,
    )

    zero = logits.sum() * 0.0
    loss_amber_cma = zero
    diagnostics: dict[str, float] = dict(prototype_diagnostics)
    if use_amber_cma_analogue:
        loss_amber_cma, cma_diagnostics = amber_cma_analogue_loss(
            output["output_features"],
            output["modality_features"],
            output["missing_mask"],
            sample_ids,
            temperature=float(amber_cma_temperature),
        )
        weighted = float(lambda_amber_cma) * loss_amber_cma
        loss = loss + weighted
        diagnostics.update(cma_diagnostics)
        diagnostics["loss/amber_cma_weighted"] = float(weighted.detach().cpu().item())

    loss_superset = zero
    if use_superset_confidence_gated_kl:
        if superset_output is None or not torch.is_tensor(superset_output.get("logits")):
            raise ValueError("Enabled superset KL requires a same-model superset output.")
        loss_superset, raw_kl, gate = _confidence_gated_temperature_kl(
            logits,
            superset_output["logits"].detach(),
            labels,
            temperature=float(superset_temperature),
        )
        loss = loss + float(lambda_superset_consistency) * loss_superset
        diagnostics.update(
            {
                "loss/superset_consistency": float(loss_superset.detach().cpu().item()),
                "superset_consistency/raw_kl": float(raw_kl.detach().cpu().item()),
                "superset_consistency/weighted_kl": float(loss_superset.detach().cpu().item()),
                "superset_consistency/gate_mean": float(gate.mean().detach().cpu().item()),
                "superset_consistency/gate_active_ratio": float(gate.gt(0).float().mean().detach().cpu().item()),
            }
        )

    if float(router_oracle_weight) == 0.0:
        loss_router_oracle = zero
        router_diagnostics = {
            "loss/router_oracle": 0.0,
            "router_oracle_active_ratio": 0.0,
            "router_oracle_enabled": 0.0,
            "router_oracle_disabled": 1.0,
        }
    else:
        loss_router_oracle, router_diagnostics = _router_oracle_loss(
            output,
            labels,
            circular_beam_distance=bool(circular_beam_distance),
        )
        router_diagnostics["router_oracle_enabled"] = 1.0
        router_diagnostics["router_oracle_disabled"] = 0.0
    loss = loss + float(router_oracle_weight) * loss_router_oracle
    diagnostics.update(router_diagnostics)
    diagnostics["loss/router_oracle_weighted"] = float(
        (float(router_oracle_weight) * loss_router_oracle).detach().cpu().item()
    )

    diagnostics.update(_loss_diagnostics(logits, labels, loss_beam, unweighted_loss_beam))
    return {
        "loss": loss,
        "loss_beam": loss_beam,
        "unweighted_loss_beam": unweighted_loss_beam,
        "loss_amber_cma": loss_amber_cma,
        "loss_superset": loss_superset,
        "loss_router_oracle": loss_router_oracle,
        "diagnostics": diagnostics,
    }


class UMaskBeamJEPATrainingExtension(TrainingExtension):
    name = "u_mask_beam_jepa"

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        return {"config": u_mask_beam_jepa_config(context.cfg), "online_superset": None}

    def before_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
    ) -> ForwardControls:
        del epoch
        config = state.get("config", {}) if isinstance(state, dict) else {}
        if not bool(config.get("enabled", False)):
            return ForwardControls()
        modalities = tuple(getattr(context.primary_model, "modalities", ()))
        if not modalities:
            raise ValueError("u_mask_beam_jepa requires primary_model.modalities.")
        mask_config = config.get("missing_mask", {})
        mask = sample_missing_mask(
            int(labels.shape[0]),
            len(modalities),
            mask_config.get("p_missing", 0.25),
            always_available_indices=mask_config.get("always_available_indices"),
            ensure_at_least_one=bool(mask_config.get("ensure_at_least_one", True)),
            device=context.device,
        )
        superset = config.get("superset_consistency", {})
        active = bool(isinstance(superset, dict) and superset.get("enabled") and superset.get("confidence_gated_kl"))
        if active:
            state["online_superset"] = _online_superset(context, batch, modalities)
        else:
            state["online_superset"] = None
        return ForwardControls(model_kwargs={"missing_mask": mask})

    def compute_base_loss(self, context: ExtensionContext, state: Any, batch_state: BatchState) -> BaseLossResult | None:
        config = state.get("config", {}) if isinstance(state, dict) else {}
        if not bool(config.get("enabled", False)):
            return None
        output = {
            "logits": batch_state.primary_logits,
            "output_features": batch_state.primary_output.output_features,
            "input_features": batch_state.primary_output.input_features,
            **batch_state.primary_output.diagnostics,
        }
        superset = config.get("superset_consistency", {})
        result = u_mask_beam_jepa_loss(
            output,
            batch_state.labels,
            prototype_bank=getattr(context.primary_model, "prototype_bank", None),
            use_beam_prototype_alignment=bool(config.get("use_beam_prototype_alignment", False)),
            lambda_proto=float(config.get("lambda_proto", 0.0)),
            lambda_modality_proto=float(config.get("lambda_modality_proto", 0.0)),
            beam_label_sigma=float(config.get("beam_label_sigma", 1.0)),
            prototype_target_circular=bool(config.get("prototype_target_circular", True)),
            use_amber_cma_analogue=bool(config.get("use_amber_cma_analogue", False)),
            lambda_amber_cma=float(config.get("lambda_amber_cma", 0.2)),
            amber_cma_temperature=float(config.get("amber_cma_temperature", 0.2)),
            sample_ids=(sample_ids_from_batch(batch_state.batch) if config.get("use_amber_cma_analogue", False) else None),
            superset_output=state.get("online_superset"),
            use_superset_confidence_gated_kl=bool(
                isinstance(superset, dict) and superset.get("enabled") and superset.get("confidence_gated_kl")
            ),
            lambda_superset_consistency=float(superset.get("kl_weight", 0.0)),
            superset_temperature=float(superset.get("temperature", 2.0)),
            router_oracle_weight=float(config.get("router_oracle_weight", 0.1)),
            circular_beam_distance=bool(config.get("circular_beam_distance", True)),
        )
        total = result["loss"]
        return BaseLossResult(
            total_loss=total,
            task_loss=result["loss_beam"],
            auxiliary_loss=total - result["loss_beam"],
            diagnostics=result["diagnostics"],
        )


def _online_superset(
    context: ExtensionContext,
    batch: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    from kd_sensing.engine.runtime import run_model_step

    superset_batch, mask = _restore_temporal_superset(batch, modalities, context.device)
    module_states = [(module, module.training) for module in context.primary_model.modules()]
    try:
        context.primary_model.eval()
        with torch.no_grad():
            step = run_model_step(
                context.primary_model,
                context.task,
                superset_batch,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": mask},
            )
    finally:
        for module, training in module_states:
            module.training = training
    return {"logits": step.logits.detach()}


def _restore_temporal_superset(
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    payload = batch.get(TEMPORAL_SUPERSET_PAYLOAD_KEY)
    if not isinstance(payload, dict):
        raise ValueError("Superset KL requires temporal_missing.preserve_unmasked_for_superset=true.")
    inputs = payload.get("inputs")
    base_mask = payload.get("base_mask")
    payload_modalities = tuple(payload.get("modalities", ()))
    if not isinstance(inputs, dict) or not torch.is_tensor(base_mask) or payload_modalities != modalities:
        raise ValueError("Invalid temporal superset payload.")
    base_mask = base_mask.to(device=device, dtype=torch.bool)
    student_mask = batch.get("modality_temporal_mask")
    if not torch.is_tensor(student_mask) or tuple(student_mask.shape) != tuple(base_mask.shape):
        raise ValueError("Temporal superset and student masks must have matching [B,T,M] shapes.")
    student_mask = student_mask.to(device=device, dtype=torch.bool)
    if bool((student_mask & ~base_mask).any().item()):
        raise ValueError("Temporal student mask must be a subset of the preserved superset mask.")
    if not bool(student_mask.any(dim=(1, 2)).all().item()) or not bool(base_mask.any(dim=(1, 2)).all().item()):
        raise ValueError("Temporal student and superset masks must retain one cell per sample.")
    restored = dict(batch)
    restored.update(inputs)
    restored["modality_temporal_mask"] = base_mask
    restored["temporal_mask"] = base_mask.any(dim=2)
    restored["available_modalities"] = base_mask.any(dim=1)
    return restored, base_mask.any(dim=1)


def _as_prediction_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,C] or [B,T,C], got {tuple(logits.shape)}.")
    return logits


def _as_prediction_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2 or labels.shape[0] != logits.shape[0] or labels.shape[1] < logits.shape[1]:
        raise ValueError(f"labels must cover logits shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}.")
    return labels[:, : logits.shape[1]]


def _beam_supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def _confidence_gated_temperature_kl(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student = _as_prediction_logits(student_logits)
    reference = _as_prediction_logits(reference_logits).to(device=student.device, dtype=student.dtype).detach()
    labels = _as_prediction_labels(labels, student)
    if reference.shape != student.shape:
        raise ValueError(f"Superset logits shape {tuple(reference.shape)} must match student logits {tuple(student.shape)}.")
    confidence = torch.softmax(reference, dim=-1).amax(dim=-1)
    correct = reference.argmax(dim=-1).eq(labels)
    gate = confidence * correct.to(dtype=confidence.dtype)
    per_sample = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(reference / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean(dim=1) * temperature**2
    weighted = (per_sample * gate.mean(dim=1)).sum() / gate.mean(dim=1).sum().clamp_min(1e-6)
    return weighted, per_sample.mean(), gate.mean(dim=1)


def _router_oracle_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    circular_beam_distance: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output.get("router_gate_logits")
    unimodal = output.get("unimodal_logits")
    available = output.get("missing_mask")
    if not torch.is_tensor(logits) or not torch.is_tensor(unimodal) or not torch.is_tensor(available):
        zero = output["logits"].sum() * 0.0
        return zero, {"router_oracle_active_ratio": 0.0, "loss/router_oracle": 0.0}
    if logits.ndim != 2 or unimodal.ndim != 3 or tuple(logits.shape) != tuple(unimodal.shape[:2]):
        raise ValueError("supervised_router outputs must be [B,M] gate logits and [B,M,C] unimodal logits.")
    available = available.to(device=logits.device, dtype=torch.bool)
    if tuple(available.shape) != tuple(logits.shape):
        raise ValueError("missing_mask must match supervised_router gate logits.")
    target = labels[:, 0].to(device=logits.device, dtype=torch.long)
    predicted = unimodal.argmax(dim=-1)
    distance = (predicted - target.unsqueeze(1)).abs()
    if circular_beam_distance:
        distance = torch.minimum(distance, int(unimodal.shape[-1]) - distance)
    oracle = distance.masked_fill(~available, int(unimodal.shape[-1])).argmin(dim=1)
    active = target.ne(-100) & available.sum(dim=1).gt(1)
    if not bool(active.any().item()):
        zero = logits.sum() * 0.0
        return zero, {"router_oracle_active_ratio": 0.0, "loss/router_oracle": 0.0}
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    loss = F.cross_entropy(masked[active], oracle[active])
    accuracy = masked[active].argmax(dim=1).eq(oracle[active]).float().mean()
    return loss, {
        "loss/router_oracle": float(loss.detach().cpu().item()),
        "router_oracle_accuracy": float(accuracy.detach().cpu().item()),
        "router_oracle_active_ratio": float(active.float().mean().detach().cpu().item()),
    }


def _loss_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_beam: torch.Tensor,
    unweighted_loss_beam: torch.Tensor,
) -> dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    if not bool(valid.any().item()):
        top1 = top5 = 0.0
    else:
        active_logits = flat_logits[valid]
        active_labels = flat_labels[valid]
        top1 = float(active_logits.argmax(dim=-1).eq(active_labels).float().mean().detach().cpu().item())
        top5 = float(
            active_logits.topk(min(5, active_logits.shape[-1]), dim=-1).indices.eq(active_labels.unsqueeze(-1)).any(dim=-1).float().mean().detach().cpu().item()
        )
    return {
        "loss_beam": float(loss_beam.detach().cpu().item()),
        "ce_loss": float(unweighted_loss_beam.detach().cpu().item()),
        "weighted_ce_loss": float(loss_beam.detach().cpu().item()),
        "accuracy/top1": top1,
        "accuracy/top5": top5,
    }


__all__ = ["UMaskBeamJEPATrainingExtension", "u_mask_beam_jepa_config", "u_mask_beam_jepa_loss"]
