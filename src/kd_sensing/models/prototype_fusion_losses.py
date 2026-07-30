"""Losses for staged shared-prototype radio alignment and dynamic fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def topology_risk(logits: torch.Tensor, labels: torch.Tensor, topology_distance: torch.Tensor) -> torch.Tensor:
    distance = torch.as_tensor(topology_distance, device=logits.device, dtype=logits.dtype)
    scale = distance.max().clamp_min(1.0)
    return (torch.softmax(logits, dim=-1) * (distance[labels] / scale)).sum(dim=-1)


def radio_alignment_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    topology_distance: torch.Tensor,
    *,
    teacher_logits: torch.Tensor | None = None,
    topology_weight: float = 0.2,
    distillation_weight: float = 1.0,
    distillation_temperature: float = 2.0,
) -> dict[str, torch.Tensor]:
    ce = F.cross_entropy(student_logits, labels)
    topology = topology_risk(student_logits, labels, topology_distance).mean()
    kd = ce.new_zeros(())
    if teacher_logits is not None and float(distillation_weight):
        temperature = float(distillation_temperature)
        kd = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2
    total = ce + float(topology_weight) * topology + float(distillation_weight) * kd
    return {"total": total, "task": ce, "topology": topology, "radio_kd": kd}


def missing_monotonic_loss(rho_more_missing: torch.Tensor, rho_less_missing: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
    return F.relu(rho_less_missing - rho_more_missing + float(margin)).mean()


def quality_monotonic_loss(rho_clean: torch.Tensor, rho_degraded: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
    return F.relu(rho_degraded - rho_clean + float(margin)).mean()


def gate_smoothness(gate: torch.Tensor, labels_by_position: Sequence[int] | torch.Tensor) -> torch.Tensor:
    labels = torch.as_tensor(labels_by_position, device=gate.device, dtype=torch.long)
    ordered = gate.index_select(-1, labels)
    return (ordered - ordered.roll(1, dims=-1)).abs().mean()


def dynamic_fusion_loss(
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    physical_availability: torch.Tensor,
    topology_distance: torch.Tensor,
    *,
    weights: Mapping[str, float],
    cardinality_weights: Mapping[int | str, float],
    labels_by_position: Sequence[int] | torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    rho_more_missing: torch.Tensor | None = None,
    rho_less_missing: torch.Tensor | None = None,
    rho_clean: torch.Tensor | None = None,
    rho_degraded: torch.Tensor | None = None,
    trust_temperature: float = 1.0,
    min_rho_std: float = 0.05,
) -> dict[str, torch.Tensor]:
    final = output["final_evidence"]
    sensing = output["sensing_evidence"]
    radio = output["radio_evidence_calibrated"]
    rho = output["rho"]
    gate = output["prototype_gate"]
    available_count = physical_availability.sum(dim=-1)
    sample_weights = torch.tensor(
        [float(cardinality_weights.get(int(value), cardinality_weights.get(str(int(value)), 1.0))) for value in available_count],
        device=final.device,
        dtype=final.dtype,
    )
    task_items = F.cross_entropy(final, labels, reduction="none")
    task = (task_items * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
    topology = (topology_risk(final, labels, topology_distance) * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)

    sensing_items = F.cross_entropy(sensing.detach(), labels, reduction="none")
    radio_items = F.cross_entropy(radio.detach(), labels, reduction="none")
    trust_target = torch.sigmoid((sensing_items - radio_items) / float(trust_temperature))
    trust = F.binary_cross_entropy(rho, trust_target.detach())
    smooth = gate_smoothness(gate, labels_by_position)

    zero = final.sum() * 0.0
    missing_mono = (
        missing_monotonic_loss(rho_more_missing, rho_less_missing)
        if rho_more_missing is not None and rho_less_missing is not None
        else zero
    )
    quality_mono = (
        quality_monotonic_loss(rho_clean, rho_degraded)
        if rho_clean is not None and rho_degraded is not None
        else zero
    )
    p_s = torch.softmax(sensing.detach(), dim=-1)
    p_c = torch.softmax(radio.detach(), dim=-1)
    p_final = torch.softmax(final, dim=-1)
    pred_s = sensing.argmax(dim=-1)
    pred_c = radio.argmax(dim=-1)
    rescue_mask = pred_s.ne(labels) & (pred_c.eq(labels) | (p_c.gather(1, labels[:, None]).squeeze(1) > p_s.gather(1, labels[:, None]).squeeze(1) + 0.05))
    preserve_mask = pred_s.eq(labels) & pred_c.ne(labels)
    rescue = task_items[rescue_mask].mean() if bool(rescue_mask.any()) else zero
    preserve_items = F.kl_div(p_final.clamp_min(1e-12).log(), p_s, reduction="none").sum(dim=-1)
    preserve = preserve_items[preserve_mask].mean() if bool(preserve_mask.any()) else zero
    usage = F.relu(final.new_tensor(float(min_rho_std)) - rho.std(unbiased=False))

    radio_kd = zero
    if teacher_logits is not None and float(weights.get("radio_kd", 0.0)):
        temperature = 2.0
        radio_kd = F.kl_div(
            F.log_softmax(radio / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2
    total = (
        task
        + float(weights.get("topology", 0.0)) * topology
        + float(weights.get("trust", 0.0)) * trust
        + float(weights.get("missing_monotonic", 0.0)) * missing_mono
        + float(weights.get("quality_monotonic", 0.0)) * quality_mono
        + float(weights.get("gate_smooth", 0.0)) * smooth
        + float(weights.get("rescue", 0.0)) * rescue
        + float(weights.get("preserve", 0.0)) * preserve
        + float(weights.get("gate_usage", 0.0)) * usage
        + float(weights.get("radio_kd", 0.0)) * radio_kd
    )
    return {
        "total": total,
        "task": task,
        "topology": topology,
        "trust": trust,
        "missing_monotonic": missing_mono,
        "quality_monotonic": quality_mono,
        "gate_smooth": smooth,
        "rescue": rescue,
        "preserve": preserve,
        "gate_usage": usage,
        "radio_kd": radio_kd,
        "trust_target_mean": trust_target.mean(),
        "rho_mean": rho.mean(),
        "rho_std": rho.std(unbiased=False),
        "rescue_fraction": rescue_mask.float().mean(),
        "preserve_fraction": preserve_mask.float().mean(),
    }


__all__ = [
    "dynamic_fusion_loss",
    "gate_smoothness",
    "missing_monotonic_loss",
    "quality_monotonic_loss",
    "radio_alignment_loss",
    "topology_risk",
]
