from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, TrainingExtension
from kd_sensing.losses.beam_prototype_alignment import (
    BeamPrototypeBank,
    beam_topology_positions,
    make_soft_beam_labels,
    prototype_alignment_loss,
)
from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config


def topology_risk_target(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    available: torch.Tensor,
    *,
    topology_id: str = "cyclic_index_v1",
    topology_permutation: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Expected normalized topology error; the probability path is always detached."""
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [B,M,C].")
    mask = torch.as_tensor(available, device=probabilities.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(probabilities.shape[:2]):
        raise ValueError("available must match probabilities [B,M].")
    hard = labels.to(device=probabilities.device, dtype=torch.long).reshape(probabilities.shape[0], -1)[:, 0]
    valid_label = hard.ne(-100)
    safe_hard = hard.masked_fill(~valid_label, 0)
    classes = int(probabilities.shape[-1])
    positions = beam_topology_positions(
        classes,
        topology_id=topology_id,
        topology_permutation=topology_permutation,
        device=probabilities.device,
    )
    circular = str(topology_id).strip().lower() != "linear_index_v1"
    distance = (positions.view(1, -1) - positions[safe_hard].view(-1, 1)).abs()
    if circular:
        distance = torch.minimum(distance, float(classes) - distance)
        maximum = float(classes // 2)
    else:
        maximum = float(classes - 1)
    normalized_distance = distance / max(maximum, 1.0)
    with torch.autocast(device_type=probabilities.device.type, enabled=False):
        target = (probabilities.detach().float() * normalized_distance.unsqueeze(1)).sum(dim=-1)
        target = target * (mask & valid_label.unsqueeze(1)).to(torch.float32)
    return target


def pcpf_temporal_risk_loss(
    output: dict[str, Any],
    labels: torch.Tensor,
    *,
    prototype_bank: BeamPrototypeBank,
    config: dict[str, Any],
) -> dict[str, Any]:
    stage = config["training_stage"]
    if stage == "stage1_expert":
        return _stage1_loss(output, labels, prototype_bank=prototype_bank, config=config)
    if stage == "stage2_risk":
        return _stage2_loss(output, labels, config=config)
    if stage == "stage3_fusion":
        return _stage3_loss(output, labels, config=config)
    raise ValueError(f"Unsupported PCPF-T training stage {stage!r}.")


class PCPFTemporalRiskTrainingExtension(TrainingExtension):
    name = "pcpf_temporal_risk"
    state_schema_version = 1

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        config = pcpf_temporal_risk_config(context.cfg)
        model = context.primary_model
        if getattr(model, "training_stage", None) != config["training_stage"]:
            raise ValueError("PCPF-T model and loss training stages do not match.")
        model.assert_trainable_parameters()
        return {"config": config}

    def state_dict(self, state: Any) -> dict[str, Any]:
        del state
        return {}

    def load_state_dict(self, state: Any, payload: Mapping[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("PCPF-T extension state must be a mapping.")
        if payload:
            raise ValueError("PCPF-T extension has no mutable resume state.")

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        output = {
            "logits": batch_state.primary_logits,
            "input_features": batch_state.primary_output.input_features,
            "output_features": batch_state.primary_output.output_features,
            **batch_state.primary_output.diagnostics,
        }
        result = pcpf_temporal_risk_loss(
            output,
            batch_state.labels,
            prototype_bank=context.primary_model.prototype_bank,
            config=state["config"],
        )
        return BaseLossResult(
            total_loss=result["loss"],
            task_loss=result["task_loss"],
            auxiliary_loss=result["loss"] - result["task_loss"],
            diagnostics=dict(result["diagnostics"]),
        )


def _stage1_loss(
    output: dict[str, Any],
    labels: torch.Tensor,
    *,
    prototype_bank: BeamPrototypeBank,
    config: dict[str, Any],
) -> dict[str, Any]:
    logits = _prediction_logits(output["logits"])
    hard = _labels(labels, logits.shape[0])
    available = _available(output, logits.shape[0])
    unimodal_logits = output["unimodal_logits"]
    if unimodal_logits.ndim != 3 or tuple(unimodal_logits.shape[:2]) != tuple(available.shape):
        raise ValueError("unimodal_logits must have shape [B,M,C].")
    valid = hard.ne(-100)
    fused_hard = F.cross_entropy(logits, hard, ignore_index=-100)
    safe_hard = hard.masked_fill(~valid, 0)
    topology = config["prototype_topology"]
    soft_target = make_soft_beam_labels(
        safe_hard,
        int(unimodal_logits.shape[-1]),
        config["beam_label_sigma"],
        circular=True,
        topology_id=topology["id"],
        topology_permutation=topology["permutation"],
    ).to(device=logits.device, dtype=torch.float32)
    per_modality_hard = F.cross_entropy(
        unimodal_logits.reshape(-1, unimodal_logits.shape[-1]),
        safe_hard.unsqueeze(1).expand(-1, unimodal_logits.shape[1]).reshape(-1),
        reduction="none",
    ).reshape_as(available)
    per_modality_soft = -(
        soft_target.unsqueeze(1) * F.log_softmax(unimodal_logits.float(), dim=-1)
    ).sum(dim=-1)
    loss_mask = available & valid.unsqueeze(1)
    denominator = loss_mask.sum(dim=1).clamp_min(1).to(torch.float32)
    unimodal_hard = ((per_modality_hard * loss_mask).sum(dim=1) / denominator)[valid].mean()
    unimodal_soft = ((per_modality_soft * loss_mask).sum(dim=1) / denominator)[valid].mean()
    unimodal = (
        config["unimodal_hard_weight"] * unimodal_hard
        + config["unimodal_soft_weight"] * unimodal_soft
    )
    fused_soft = -(soft_target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)[valid].mean()
    prototype = logits.sum() * 0.0
    prototype_diagnostics: dict[str, float] = {}
    if config["use_beam_prototype_alignment"]:
        prototype, prototype_diagnostics = prototype_alignment_loss(
            prototype_bank,
            hard,
            fused_features=output["output_features"],
            modality_features=output["modality_features"],
            mask=available,
            beam_label_sigma=config["beam_label_sigma"],
            circular=True,
            topology_id=topology["id"],
            topology_permutation=topology["permutation"],
            lambda_proto=config["lambda_proto"],
            lambda_modality_proto=config["lambda_modality_proto"],
        )
    total = (
        config["lambda_fused_hard"] * fused_hard
        + config["lambda_unimodal"] * unimodal
        + config["fused_soft_weight"] * fused_soft
        + prototype
    )
    diagnostics = {
        "loss/pcpf_stage1_total": _scalar(total),
        "loss/pcpf_fused_hard": _scalar(fused_hard),
        "loss/pcpf_unimodal": _scalar(unimodal),
        "loss/pcpf_unimodal_hard": _scalar(unimodal_hard),
        "loss/pcpf_unimodal_soft": _scalar(unimodal_soft),
        "loss/pcpf_fused_soft": _scalar(fused_soft),
        **prototype_diagnostics,
    }
    return {"loss": total, "task_loss": fused_hard, "diagnostics": diagnostics}


def _stage2_loss(output: dict[str, Any], labels: torch.Tensor, *, config: dict[str, Any]) -> dict[str, Any]:
    available = _available(output, int(output["raw_risk"].shape[0]))
    topology = config["prototype_topology"]
    target = topology_risk_target(
        output["unimodal_probabilities"],
        labels,
        available,
        topology_id=topology["id"],
        topology_permutation=topology["permutation"],
    )
    valid_label = _labels(labels, available.shape[0]).ne(-100)
    mask = available & valid_label.unsqueeze(1)
    predicted = output["raw_risk"].float()
    loss_risk = _masked_mean(F.smooth_l1_loss(predicted, target, reduction="none"), mask)
    loss_rank, active_pairs = _pair_ranking_loss(predicted, target, mask, margin=config["rank_margin"])
    mu = output["probability_mu"].float()
    logvar = output["probability_logvar"].float()
    with torch.autocast(device_type=mu.device.type, enabled=False):
        gaussian_kl = 0.5 * (torch.exp(logvar) + mu.square() - 1.0 - logvar).mean(dim=-1)
    loss_kl = _masked_mean(gaussian_kl, mask)
    preserve_logits = output["sampled_unimodal_logits"].float()
    safe_hard = _labels(labels, available.shape[0]).masked_fill(~valid_label, 0)
    soft_target = make_soft_beam_labels(
        safe_hard,
        int(preserve_logits.shape[-1]),
        config["beam_label_sigma"],
        circular=True,
        topology_id=topology["id"],
        topology_permutation=topology["permutation"],
    ).to(device=preserve_logits.device, dtype=torch.float32)
    preserve = -(soft_target.unsqueeze(1) * F.log_softmax(preserve_logits, dim=-1)).sum(dim=-1)
    loss_preserve = _masked_mean(preserve, mask)
    supervision = 1.0 if config["risk_supervision_enabled"] else 0.0
    total = (
        supervision * config["lambda_risk"] * loss_risk
        + supervision * config["lambda_rank"] * loss_rank
        + config["beta_kl"] * loss_kl
        + config["lambda_preserve"] * loss_preserve
    )
    diagnostics = {
        "loss/pcpf_stage2_total": _scalar(total),
        "loss/pcpf_risk": _scalar(loss_risk),
        "loss/pcpf_rank": _scalar(loss_rank),
        "loss/pcpf_kl": _scalar(loss_kl),
        "loss/pcpf_preserve": _scalar(loss_preserve),
        "pcpf/rank_active_pairs": float(active_pairs),
        "pcpf/risk_target_mean": _scalar(_masked_mean(target, mask)),
        "pcpf/predicted_risk_mean": _scalar(_masked_mean(predicted, mask)),
        "pcpf/risk_supervision_enabled": supervision,
    }
    return {"loss": total, "task_loss": loss_risk, "diagnostics": diagnostics, "risk_target": target}


def _stage3_loss(output: dict[str, Any], labels: torch.Tensor, *, config: dict[str, Any]) -> dict[str, Any]:
    logits = _prediction_logits(output["logits"])
    hard = _labels(labels, logits.shape[0])
    valid = hard.ne(-100)
    nll = F.cross_entropy(logits, hard, ignore_index=-100)
    topology = config["prototype_topology"]
    safe_hard = hard.masked_fill(~valid, 0)
    target = make_soft_beam_labels(
        safe_hard,
        int(logits.shape[-1]),
        config["beam_label_sigma"],
        circular=True,
        topology_id=topology["id"],
        topology_permutation=topology["permutation"],
    ).to(device=logits.device, dtype=torch.float32)
    topology_ce = -(target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)[valid].mean()
    total = nll + config["stage3_topology_weight"] * topology_ce
    return {
        "loss": total,
        "task_loss": nll,
        "diagnostics": {
            "loss/pcpf_stage3_total": _scalar(total),
            "loss/pcpf_fusion_nll": _scalar(nll),
            "loss/pcpf_fusion_topology": _scalar(topology_ce),
        },
    }


def _pair_ranking_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    modalities = int(predicted.shape[1])
    for left in range(modalities):
        for right in range(left + 1, modalities):
            valid = available[:, left] & available[:, right]
            target_difference = target[:, right] - target[:, left]
            valid = valid & target_difference.abs().gt(float(margin))
            if bool(valid.any().item()):
                predicted_difference = predicted[:, right] - predicted[:, left]
                direction = target_difference.sign()
                losses.append(F.softplus(-direction[valid] * predicted_difference[valid]))
    if not losses:
        return predicted.sum() * 0.0, 0
    concatenated = torch.cat(losses)
    return concatenated.mean(), int(concatenated.numel())


def _prediction_logits(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        value = value[:, -1]
    if value.ndim != 2:
        raise ValueError("prediction logits must have shape [B,C] or [B,H,C].")
    return value.float()


def _labels(labels: torch.Tensor, batch_size: int) -> torch.Tensor:
    hard = labels.to(dtype=torch.long).reshape(batch_size, -1)[:, 0]
    if not bool(hard.ne(-100).any().item()):
        raise ValueError("PCPF-T loss requires at least one valid label.")
    return hard


def _available(output: dict[str, Any], batch_size: int) -> torch.Tensor:
    available = torch.as_tensor(output["available_modalities"], dtype=torch.bool, device=output["logits"].device)
    if available.ndim != 2 or int(available.shape[0]) != batch_size:
        raise ValueError("available_modalities must have shape [B,M].")
    return available


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=value.device, dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


__all__ = [
    "PCPFTemporalRiskTrainingExtension",
    "pcpf_temporal_risk_loss",
    "topology_risk_target",
]
