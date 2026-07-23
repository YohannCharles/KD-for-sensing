from typing import Any, Mapping

import torch
import torch.nn.functional as F

from kd_sensing.data.missing_mask import sample_missing_mask
from kd_sensing.data.temporal_missing_contract import TEMPORAL_SUPERSET_PAYLOAD_KEY
from kd_sensing.engine.training_extensions import (
    BaseLossResult,
    BatchState,
    ExtensionContext,
    ForwardControls,
    TrainingExtension,
)
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
    prototype_topology_id: str | None = None,
    prototype_topology_permutation: list[int] | tuple[int, ...] | None = None,
    superset_output: dict[str, torch.Tensor] | None = None,
    use_superset_confidence_gated_kl: bool = False,
    lambda_superset_consistency: float = 0.0,
    superset_temperature: float = 2.0,
    router_oracle_weight: float = 0.0,
    router_oracle_target_mode: str = "hard_first",
    router_oracle_temperature: float = 1.0,
) -> dict[str, Any]:
    if str(router_oracle_target_mode).strip().lower() != "hard_first":
        raise ValueError("Clean U0 supports router_oracle_target_mode=hard_first only.")
    if min(float(superset_temperature), float(router_oracle_temperature)) <= 0:
        raise ValueError("loss temperatures must be positive.")
    if min(float(lambda_superset_consistency), float(router_oracle_weight)) < 0:
        raise ValueError("loss weights must be non-negative.")

    logits = _as_prediction_logits(output["logits"])
    targets = _as_prediction_labels(labels, logits)
    per_sample_beam = _beam_supervised_loss_per_sample(logits, targets)
    loss_beam = per_sample_beam.mean()
    loss = loss_beam
    loss, prototype_diagnostics = add_prototype_alignment_losses(
        loss,
        output,
        targets,
        prototype_bank=prototype_bank,
        enabled=use_beam_prototype_alignment,
        lambda_proto=lambda_proto,
        lambda_modality_proto=lambda_modality_proto,
        beam_label_sigma=beam_label_sigma,
        prototype_target_circular=prototype_target_circular,
        prototype_topology_id=prototype_topology_id,
        prototype_topology_permutation=prototype_topology_permutation,
    )

    zero = logits.sum() * 0.0
    diagnostics = dict(prototype_diagnostics)
    loss_superset = zero
    if use_superset_confidence_gated_kl:
        if superset_output is None or not torch.is_tensor(superset_output.get("logits")):
            raise ValueError("Enabled superset KL requires a same-model superset output.")
        loss_superset, raw_kl, gate = _confidence_gated_temperature_kl(
            logits,
            superset_output["logits"].detach(),
            targets,
            temperature=float(superset_temperature),
        )
        loss = loss + float(lambda_superset_consistency) * loss_superset
        diagnostics.update(
            {
                "loss/superset_consistency": _scalar(loss_superset),
                "superset_consistency/raw_kl": _scalar(raw_kl),
                "superset_consistency/gate_mean": _scalar(gate.mean()),
                "superset_consistency/gate_active_ratio": _scalar(gate.gt(0).float().mean()),
            }
        )

    loss_router = zero
    if float(router_oracle_weight) > 0:
        loss_router, router_diagnostics = _router_oracle_loss(
            output,
            targets,
            temperature=float(router_oracle_temperature),
        )
        loss = loss + float(router_oracle_weight) * loss_router
    else:
        router_diagnostics = {
            "router_oracle_active_ratio": 0.0,
            "router_oracle_tie_ratio": 0.0,
            "router_oracle_enabled": 0.0,
            "router_oracle_disabled": 1.0,
        }
    diagnostics.update(router_diagnostics)
    diagnostics["loss/router_oracle"] = _scalar(loss_router)
    diagnostics["loss/router_oracle_weighted"] = _scalar(float(router_oracle_weight) * loss_router)
    diagnostics.update(_loss_diagnostics(logits, targets, loss_beam))

    return {
        "loss": loss,
        "loss_beam": loss_beam,
        "loss_superset": loss_superset,
        "loss_router_oracle": loss_router,
        "diagnostics": diagnostics,
    }


class UMaskBeamJEPATrainingExtension(TrainingExtension):
    name = "u_mask_beam_jepa"
    state_schema_version = 1

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        return {
            "config": u_mask_beam_jepa_config(context.cfg),
            "online_superset": None,
        }

    def state_dict(self, state: Any) -> dict[str, Any]:
        del state
        return {}

    def load_state_dict(self, state: Any, payload: Mapping[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("u_mask_beam_jepa extension state must be a mapping.")
        if payload:
            raise ValueError("Clean U0 cannot resume a checkpoint with retired extension state.")

    def before_epoch(self, context: ExtensionContext, state: Any, *, epoch: int) -> None:
        del context, epoch
        state["online_superset"] = None

    def before_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
        step: int = 0,
    ) -> ForwardControls:
        del epoch, step
        config = state.get("config", {})
        if not config.get("enabled"):
            return ForwardControls()
        modalities = tuple(getattr(context.primary_model, "modalities", ()))
        if not modalities:
            raise ValueError("u_mask_beam_jepa requires primary_model.modalities.")
        mask_config = config["missing_mask"]
        if mask_config["mode"] == "external":
            mask = _external_missing_mask(batch, labels, modalities, context.device)
        else:
            mask = sample_missing_mask(
                int(labels.shape[0]),
                len(modalities),
                mask_config["p_missing"],
                always_available_indices=mask_config.get("always_available_indices"),
                ensure_at_least_one=mask_config["ensure_at_least_one"],
                device=context.device,
            )
        superset = config["superset_consistency"]
        state["online_superset"] = (
            _online_superset(context, batch, modalities)
            if superset["enabled"] and superset["confidence_gated_kl"]
            else None
        )
        return ForwardControls(model_kwargs={"missing_mask": mask})

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        config = state.get("config", {})
        if not config.get("enabled"):
            return None
        output = {
            "logits": batch_state.primary_logits,
            "input_features": batch_state.primary_output.input_features,
            "output_features": batch_state.primary_output.output_features,
            **batch_state.primary_output.diagnostics,
        }
        superset = config["superset_consistency"]
        topology = config["prototype_topology"]
        result = u_mask_beam_jepa_loss(
            output,
            batch_state.labels,
            prototype_bank=getattr(context.primary_model, "prototype_bank", None),
            use_beam_prototype_alignment=config["use_beam_prototype_alignment"],
            lambda_proto=config["lambda_proto"],
            lambda_modality_proto=config["lambda_modality_proto"],
            beam_label_sigma=config["beam_label_sigma"],
            prototype_target_circular=config["prototype_target_circular"],
            prototype_topology_id=topology["id"] if topology["id"] != "not_applicable" else None,
            prototype_topology_permutation=topology["permutation"],
            superset_output=state.get("online_superset"),
            use_superset_confidence_gated_kl=superset["enabled"] and superset["confidence_gated_kl"],
            lambda_superset_consistency=superset["kl_weight"],
            superset_temperature=superset["temperature"],
            router_oracle_weight=config["router_oracle_weight"],
            router_oracle_target_mode=config["router_oracle_target_mode"],
            router_oracle_temperature=config["router_oracle_temperature"],
        )
        task_loss = result["loss_beam"]
        return BaseLossResult(
            total_loss=result["loss"],
            task_loss=task_loss,
            auxiliary_loss=result["loss"] - task_loss,
            diagnostics=dict(result["diagnostics"]),
        )

    def after_epoch(
        self,
        context: ExtensionContext,
        state: Any,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        del context, state, epoch
        return {}


def _online_superset(
    context: ExtensionContext,
    batch: dict[str, Any],
    modalities: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    from kd_sensing.engine.runtime import run_model_step

    restored, mask = _restore_temporal_superset(batch, modalities, context.device)
    states = [(module, module.training) for module in context.primary_model.modules()]
    try:
        context.primary_model.eval()
        with torch.no_grad():
            step = run_model_step(
                context.primary_model,
                context.task,
                restored,
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": mask},
            )
    finally:
        for module, training in states:
            module.training = training
    return {"logits": step.logits.detach()}


def _restore_temporal_superset(
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    payload = batch.get(TEMPORAL_SUPERSET_PAYLOAD_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError("Enabled temporal superset consistency requires preserved unmasked inputs.")
    if tuple(payload.get("modalities", ())) != modalities:
        raise ValueError("Temporal superset modalities do not match the model.")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Temporal superset payload is missing original inputs.")
    restored = dict(batch)
    restored.update(inputs)
    base_mask = torch.as_tensor(payload.get("base_mask"), dtype=torch.bool)
    restored["modality_temporal_mask"] = base_mask
    restored["temporal_mask"] = base_mask.any(dim=2)
    restored["available_modalities"] = base_mask.any(dim=1)
    mask = base_mask.any(dim=1).to(device=device)
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("Temporal superset requires at least one available modality per sample.")
    return restored, mask


def _external_missing_mask(
    batch: Mapping[str, Any],
    labels: torch.Tensor,
    modalities: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    value = batch.get("available_modalities")
    if value is None:
        temporal = batch.get("modality_temporal_mask")
        value = torch.as_tensor(temporal).any(dim=1) if temporal is not None else None
    if value is None:
        raise ValueError("external missing-mask mode requires batch.available_modalities.")
    mask = torch.as_tensor(value, device=device, dtype=torch.bool)
    expected = (int(labels.shape[0]), len(modalities))
    if tuple(mask.shape) != expected:
        raise ValueError(f"external missing mask must have shape {expected}.")
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("external missing mask requires at least one available modality per sample.")
    return mask


def _router_oracle_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    unimodal = output.get("unimodal_logits")
    router = output.get("router_gate_logits")
    available = output.get("missing_mask")
    if not all(torch.is_tensor(value) for value in (unimodal, router, available)):
        raise ValueError("Router oracle supervision requires unimodal logits, router logits, and missing mask.")
    target = labels[:, 0]
    safe = target.clamp_min(0)
    predictions = unimodal.detach().argmax(dim=-1)
    distance = (predictions - safe.unsqueeze(1)).abs()
    unavailable = ~available.to(dtype=torch.bool)
    distance = distance.masked_fill(unavailable, unimodal.shape[-1] + 1)
    minimum = distance.min(dim=1, keepdim=True).values
    ties = distance.eq(minimum) & ~unavailable
    oracle = distance.argmin(dim=1)
    valid = target.ne(-100)
    if bool(valid.any().item()):
        loss = F.cross_entropy(router[valid] / float(temperature), oracle[valid])
    else:
        loss = router.sum() * 0.0
    return loss, {
        "router_oracle_active_ratio": _scalar(valid.float().mean()),
        "router_oracle_tie_ratio": _scalar(ties.sum(dim=1).gt(1).float().mean()),
        "router_oracle_enabled": 1.0,
        "router_oracle_disabled": 0.0,
    }


def _confidence_gated_temperature_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher = _as_prediction_logits(teacher).to(device=student.device, dtype=student.dtype)
    if teacher.shape != student.shape:
        raise ValueError("Superset logits must match masked logits.")
    t = float(temperature)
    per_item = F.kl_div(
        F.log_softmax(student / t, dim=-1),
        F.softmax(teacher / t, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (t * t)
    safe = labels.clamp_min(0).unsqueeze(-1)
    teacher_true = F.softmax(teacher.detach(), dim=-1).gather(-1, safe).squeeze(-1)
    student_true = F.softmax(student.detach(), dim=-1).gather(-1, safe).squeeze(-1)
    gate = (teacher_true - student_true).clamp_min(0.0) * labels.ne(-100)
    raw = (per_item * labels.ne(-100)).sum() / labels.ne(-100).sum().clamp_min(1)
    weighted = (per_item * gate).sum() / gate.sum().clamp_min(torch.finfo(per_item.dtype).tiny)
    return weighted, raw, gate


def _as_prediction_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError("Beam logits must have shape [B,C] or [B,H,C].")
    return logits


def _as_prediction_labels(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    value = labels.to(device=logits.device, dtype=torch.long)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.shape != logits.shape[:2]:
        raise ValueError(f"Beam labels must have shape {tuple(logits.shape[:2])}.")
    return value


def _beam_supervised_loss_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    valid = labels.ne(-100)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _loss_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_beam: torch.Tensor,
) -> dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(-100)
    if bool(valid.any().item()):
        predictions = flat_logits[valid]
        targets = flat_labels[valid]
        top1 = predictions.argmax(dim=-1).eq(targets).float().mean()
        top5 = predictions.topk(min(5, predictions.shape[-1]), dim=-1).indices.eq(
            targets.unsqueeze(-1)
        ).any(dim=-1).float().mean()
    else:
        top1 = top5 = loss_beam.detach() * 0.0
    return {
        "loss_beam": _scalar(loss_beam),
        "ce_loss": _scalar(loss_beam),
        "accuracy/top1": _scalar(top1),
        "accuracy/top5": _scalar(top5),
    }


def _scalar(value: torch.Tensor | float | int) -> float:
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


__all__ = ["UMaskBeamJEPATrainingExtension", "u_mask_beam_jepa_loss"]
