from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamPrototypeBank(nn.Module):
    def __init__(self, d_model: int, num_beams: int, *, temperature: float = 0.2) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_beams = int(num_beams)
        self.temperature = float(temperature)
        if self.d_model <= 0 or self.num_beams <= 0 or self.temperature <= 0:
            raise ValueError("d_model, num_beams, and temperature must be positive.")
        self.prototypes = nn.Parameter(torch.randn(self.num_beams, self.d_model) * 0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.d_model:
            raise ValueError(f"features must have shape [B, {self.d_model}], got {tuple(features.shape)}.")
        feature = F.normalize(features, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return feature @ proto.t() / self.temperature


def make_soft_beam_labels(
    labels: torch.Tensor,
    num_beams: int,
    sigma: float,
    *,
    circular: bool = True,
) -> torch.Tensor:
    num_beams = int(num_beams)
    sigma = float(sigma)
    if num_beams <= 0 or sigma <= 0:
        raise ValueError("num_beams and sigma must be positive.")
    target = labels.to(dtype=torch.long).reshape(-1)
    if torch.any((target < 0) | (target >= num_beams)):
        raise ValueError(f"labels must be in [0, {num_beams - 1}].")
    beams = torch.arange(num_beams, device=target.device, dtype=torch.float32).view(1, -1)
    center = target.to(dtype=torch.float32).view(-1, 1)
    distance = (beams - center).abs()
    if circular:
        distance = torch.minimum(distance, float(num_beams) - distance)
    weights = torch.exp(-0.5 * (distance / sigma).pow(2))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def make_beam_topology_soft_targets(
    labels: torch.Tensor,
    num_beams: int,
    tau_beam: float,
    *,
    circular: bool = False,
) -> torch.Tensor:
    num_beams = int(num_beams)
    tau_beam = float(tau_beam)
    if num_beams <= 0 or tau_beam <= 0:
        raise ValueError("num_beams and tau_beam must be positive.")
    target = labels.to(dtype=torch.long).reshape(-1)
    if torch.any((target < 0) | (target >= num_beams)):
        raise ValueError(f"labels must be in [0, {num_beams - 1}].")
    beams = torch.arange(num_beams, device=target.device, dtype=torch.float32).view(1, -1)
    center = target.to(dtype=torch.float32).view(-1, 1)
    distance = (beams - center).abs()
    if circular:
        distance = torch.minimum(distance, float(num_beams) - distance)
    weights = torch.exp(-distance.pow(2) / tau_beam)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def prototype_alignment_loss(
    prototype_bank: BeamPrototypeBank,
    labels: torch.Tensor,
    *,
    fused_features: torch.Tensor,
    modality_features: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
    beam_label_sigma: float = 1.0,
    beam_label_circular: bool = True,
    proto_target_type: str = "gaussian",
    tau_beam: float = 2.0,
    circular_beam_distance: bool | None = None,
    lambda_proto: float = 1.0,
    lambda_modality_proto: float = 0.0,
    lambda_teacher_proto: float = 0.0,
    btapa_include_fusion: bool = True,
    btapa_include_modalities: bool = True,
    btapa_fusion_weight: float = 1.0,
    btapa_modality_weight: float | None = None,
    use_adba_aware_proto: bool = False,
    lambda_adba_proto: float = 0.0,
    adba_margin: int = 3,
    use_pattern_conditional_btapa: bool = False,
    pattern_names: list[str] | tuple[str, ...] | None = None,
    btapa_apply_patterns: list[str] | tuple[str, ...] | None = None,
    btapa_disable_on_patterns: list[str] | tuple[str, ...] | None = None,
    btapa_fallback_to_ordinary_proto: bool = True,
    ordinary_proto_target_type: str = "gaussian",
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = labels.to(device=fused_features.device, dtype=torch.long)
    if hard.ndim > 1:
        hard = hard[:, 0]
    target_type = str(proto_target_type or "gaussian").lower()
    circular = beam_label_circular if circular_beam_distance is None else bool(circular_beam_distance)
    btapa_active: torch.Tensor | None = None
    ordinary_loss_value = 0.0
    btapa_loss_value = 0.0
    if use_pattern_conditional_btapa:
        if pattern_names is None:
            raise ValueError("pattern_names are required when use_pattern_conditional_btapa=true.")
        if len(pattern_names) != int(hard.numel()):
            raise ValueError(f"pattern_names length must match batch size {int(hard.numel())}, got {len(pattern_names)}.")
        apply = {str(item) for item in (btapa_apply_patterns or ())}
        disable = {str(item) for item in (btapa_disable_on_patterns or ())}
        active_flags = [name in apply and name not in disable for name in pattern_names]
        btapa_active = torch.tensor(active_flags, dtype=torch.bool, device=fused_features.device)
        ordinary_type = str(ordinary_proto_target_type or "gaussian").lower()
        ordinary_target = _prototype_target(
            ordinary_type,
            hard,
            prototype_bank.num_beams,
            beam_label_sigma=beam_label_sigma,
            beam_label_circular=beam_label_circular,
            tau_beam=tau_beam,
            circular=circular,
        )
        btapa_target = make_beam_topology_soft_targets(
            hard,
            prototype_bank.num_beams,
            tau_beam,
            circular=circular,
        )
        if apply or not btapa_fallback_to_ordinary_proto:
            target = torch.where(btapa_active.view(-1, 1), btapa_target, ordinary_target)
        else:
            target = ordinary_target
        loss_fn = _soft_ce
        loss_per_sample_fn = _soft_ce_per_sample
        target_type = "pattern_conditional_btapa"
    elif target_type in {"beam_soft", "btapa"}:
        target = make_beam_topology_soft_targets(
            hard,
            prototype_bank.num_beams,
            tau_beam,
            circular=circular,
        )
        loss_fn = _soft_ce
        loss_per_sample_fn = _soft_ce_per_sample
    elif target_type in {"onehot", "hard"}:
        target = F.one_hot(hard, num_classes=prototype_bank.num_beams).to(dtype=torch.float32)
        loss_fn = _soft_ce
        loss_per_sample_fn = _soft_ce_per_sample
    elif target_type in {"gaussian", "soft", "legacy"}:
        target = make_soft_beam_labels(
            hard,
            prototype_bank.num_beams,
            beam_label_sigma,
            circular=beam_label_circular,
        )
        loss_fn = _soft_kl
        loss_per_sample_fn = _soft_ce_per_sample
    else:
        raise ValueError("proto_target_type must be onehot, beam_soft, or gaussian.")
    target = target.to(device=fused_features.device, dtype=fused_features.dtype)
    weights = _sample_weights(sample_weights, fused_features)
    fused_logits = prototype_bank(fused_features)
    if btapa_include_fusion:
        loss_fused = _weighted_mean(loss_per_sample_fn(fused_logits, target), weights) if weights is not None else loss_fn(fused_logits, target)
        if btapa_active is not None:
            fused_per_sample = loss_per_sample_fn(fused_logits, target).detach()
            ordinary_mask = ~btapa_active
            if bool(ordinary_mask.any().item()):
                ordinary_loss_value = float(fused_per_sample[ordinary_mask].mean().cpu().item())
            if bool(btapa_active.any().item()):
                btapa_loss_value = float(fused_per_sample[btapa_active].mean().cpu().item())
    else:
        loss_fused = fused_logits.sum() * 0.0
    zero = fused_logits.sum() * 0.0
    loss_modality = zero
    modality_count = 0
    if btapa_include_modalities and modality_features is not None and mask is not None:
        if modality_features.ndim != 3:
            raise ValueError(f"modality_features must have shape [B, M, D], got {tuple(modality_features.shape)}.")
        available = mask.to(device=modality_features.device, dtype=torch.bool)
        if tuple(available.shape) != tuple(modality_features.shape[:2]):
            raise ValueError(f"mask must have shape {tuple(modality_features.shape[:2])}, got {tuple(available.shape)}.")
        modality_count = int(available.sum().detach().cpu().item())
        if modality_count:
            flat_features = modality_features[available]
            flat_target = target.unsqueeze(1).expand(-1, modality_features.shape[1], -1)[available]
            flat_weight = weights.unsqueeze(1).expand(-1, modality_features.shape[1])[available] if weights is not None else None
            flat_logits = prototype_bank(flat_features)
            loss_modality = (
                _weighted_mean(loss_per_sample_fn(flat_logits, flat_target), flat_weight)
                if flat_weight is not None
                else loss_fn(flat_logits, flat_target)
            )
    loss_teacher = zero
    if teacher_features is not None and float(lambda_teacher_proto) != 0.0:
        loss_teacher = loss_fn(prototype_bank(teacher_features), target)
    modality_weight = float(lambda_modality_proto if btapa_modality_weight is None else btapa_modality_weight)
    proto_core = float(btapa_fusion_weight) * loss_fused + modality_weight * loss_modality
    loss_adba = zero
    if use_adba_aware_proto and float(lambda_adba_proto) != 0.0:
        loss_adba = _adba_proto_loss(
            fused_logits,
            hard,
            margin=int(adba_margin),
            circular=circular,
        )
    total = float(lambda_proto) * proto_core + float(lambda_teacher_proto) * loss_teacher + float(lambda_adba_proto) * loss_adba
    top1, top5 = _topk(fused_logits, hard)
    diagnostics = {
        "loss/prototype_alignment": float(loss_fused.detach().cpu().item()),
        "loss/prototype_modality": float(loss_modality.detach().cpu().item()),
        "loss/prototype_teacher": float(loss_teacher.detach().cpu().item()),
        "loss/prototype_total": float(total.detach().cpu().item()),
        "loss/btapa_fusion": float(loss_fused.detach().cpu().item()),
        "loss/btapa_modality": float(loss_modality.detach().cpu().item()),
        "loss/adba_proto": float(loss_adba.detach().cpu().item()),
        "prototype/top1": top1,
        "prototype/top5": top5,
        "prototype/sample_count": float(int(hard.numel())),
        "prototype/modality_sample_count": float(modality_count),
        "prototype/target_source_beam_topology": 1.0 if target_type in {"beam_soft", "btapa", "pattern_conditional_btapa"} else 0.0,
        "ordinary_proto_loss": ordinary_loss_value,
        "btapa_loss": btapa_loss_value,
        "btapa_active_ratio": float(btapa_active.float().mean().detach().cpu().item()) if btapa_active is not None else 0.0,
        "total_proto_loss": float(total.detach().cpu().item()),
    }
    return total, diagnostics


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [B, D], got {tuple(features.shape)}.")
    labels = labels.to(device=features.device, dtype=torch.long).reshape(-1)
    if labels.numel() != features.shape[0]:
        raise ValueError("labels and features batch size must match.")
    normalized = F.normalize(features, dim=-1)
    logits = normalized @ normalized.t() / float(temperature)
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    same = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye
    valid = same.any(dim=1)
    if not bool(valid.any().item()):
        zero = features.sum() * 0.0
        return zero, {"loss/prototype_supcon": 0.0, "prototype/supcon_anchor_count": 0.0}
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob * same.to(dtype=features.dtype)).sum(dim=1) / same.sum(dim=1).clamp_min(1)
    loss = per_anchor[valid].mean()
    return loss, {
        "loss/prototype_supcon": float(loss.detach().cpu().item()),
        "prototype/supcon_anchor_count": float(int(valid.sum().detach().cpu().item())),
    }


def _soft_kl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _soft_ce_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if int(weights.numel()) != int(values.numel()):
        raise ValueError(f"weights must have {int(values.numel())} values, got {int(weights.numel())}.")
    return (values.reshape(-1) * weights).sum() / weights.sum().clamp_min(1e-6)


def _sample_weights(sample_weights: torch.Tensor | None, fused_features: torch.Tensor) -> torch.Tensor | None:
    if sample_weights is None:
        return None
    weights = sample_weights.to(device=fused_features.device, dtype=fused_features.dtype).reshape(-1)
    if int(weights.numel()) != int(fused_features.shape[0]):
        raise ValueError(f"sample_weights must have shape [B], got {tuple(sample_weights.shape)} for B={fused_features.shape[0]}.")
    return weights


def _prototype_target(
    target_type: str,
    labels: torch.Tensor,
    num_beams: int,
    *,
    beam_label_sigma: float,
    beam_label_circular: bool,
    tau_beam: float,
    circular: bool,
) -> torch.Tensor:
    if target_type in {"beam_soft", "btapa"}:
        return make_beam_topology_soft_targets(labels, num_beams, tau_beam, circular=circular)
    if target_type in {"onehot", "hard"}:
        return F.one_hot(labels, num_classes=num_beams).to(dtype=torch.float32)
    if target_type in {"gaussian", "soft", "legacy"}:
        return make_soft_beam_labels(labels, num_beams, beam_label_sigma, circular=beam_label_circular)
    raise ValueError("ordinary_proto_target_type must be onehot, beam_soft, or gaussian.")


def _adba_proto_loss(logits: torch.Tensor, labels: torch.Tensor, *, margin: int, circular: bool) -> torch.Tensor:
    num_beams = int(logits.shape[-1])
    if margin < 0:
        raise ValueError("adba_margin must be non-negative.")
    beams = torch.arange(num_beams, device=labels.device, dtype=torch.float32).view(1, -1)
    center = labels.to(dtype=torch.float32).reshape(-1, 1)
    distance = (beams - center).abs()
    if circular:
        distance = torch.minimum(distance, float(num_beams) - distance)
    near = distance.le(float(margin)).to(dtype=logits.dtype)
    near_prob = (F.softmax(logits, dim=-1) * near).sum(dim=-1).clamp_min(1e-12)
    return -near_prob.log().mean()


def _topk(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    top1 = logits.argmax(dim=-1).eq(labels).float().mean()
    k = min(5, int(logits.shape[-1]))
    top5 = logits.topk(k, dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean()
    return float(top1.detach().cpu().item()), float(top5.detach().cpu().item())


__all__ = [
    "BeamPrototypeBank",
    "make_beam_topology_soft_targets",
    "make_soft_beam_labels",
    "prototype_alignment_loss",
    "supervised_contrastive_loss",
]
