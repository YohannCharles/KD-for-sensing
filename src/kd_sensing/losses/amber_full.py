from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.model_output import ModelOutput


def amber_full_auxiliary_loss_from_output(
    model_output: ModelOutput,
    cfg: dict[str, Any],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    aux_cfg = _amber_full_loss_cfg(cfg)
    if not aux_cfg["enabled"]:
        return zero, {}
    payload = model_output.diagnostics.get("amber_full_auxiliary")
    if not isinstance(payload, dict):
        if not torch.is_grad_enabled():
            return zero, {}
        raise ValueError("AMBER full auxiliary loss is enabled but 'amber_full_auxiliary' payload is missing.")
    l2_weight = float(aux_cfg.get("l2_weight", 0.0) or 0.0)
    cma_weight = float(aux_cfg.get("cma_weight", 0.0) or 0.0)
    l2 = _alignment_l2(payload, zero) * l2_weight if l2_weight > 0.0 else zero
    cma = _cma_contrastive(payload, zero) * cma_weight if cma_weight > 0.0 else zero
    total = l2 + cma
    return total, {
        "loss/amber_full_l2": float(l2.detach().cpu().item()),
        "loss/amber_full_cma": float(cma.detach().cpu().item()),
        "loss/amber_full_total": float(total.detach().cpu().item()),
        "amber_full_loss/l2_weight": l2_weight,
        "amber_full_loss/cma_weight": cma_weight,
    }


def _amber_full_loss_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    auxiliary = loss_cfg.get("auxiliary") if isinstance(loss_cfg.get("auxiliary"), dict) else {}
    raw = {
        **_mapping(auxiliary.get("amber_full")),
        **_mapping(auxiliary.get("amber_l2")),
        **_mapping(auxiliary.get("amber_cma_contrastive")),
        **_mapping(loss_cfg.get("amber_full")),
    }
    if not raw:
        return {"enabled": False}
    l2_weight = float(raw.get("l2_weight", raw.get("alignment_weight", raw.get("weight_l2", 0.0))) or 0.0)
    cma_weight = float(raw.get("cma_weight", raw.get("contrastive_weight", raw.get("weight_cma", 0.0))) or 0.0)
    enabled = bool(raw.get("enabled", l2_weight > 0.0 or cma_weight > 0.0))
    return {**raw, "enabled": enabled, "l2_weight": l2_weight, "cma_weight": cma_weight}


def _alignment_l2(payload: dict[str, Any], zero: torch.Tensor) -> torch.Tensor:
    fusion = _tensor(payload, "fusion_features")
    target = _tensor(payload, "alignment_target").to(device=fusion.device, dtype=fusion.dtype)
    if target.shape != fusion.shape:
        raise ValueError(f"AMBER full alignment target shape {tuple(target.shape)} does not match fusion {tuple(fusion.shape)}.")
    finite = torch.isfinite(fusion).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    if not bool(finite.any().detach().cpu().item()):
        return zero
    return (fusion - target).pow(2).mean(dim=-1)[finite].mean()


def _cma_contrastive(payload: dict[str, Any], zero: torch.Tensor) -> torch.Tensor:
    logits = _tensor(payload, "cma_logits")
    fusion_query = _tensor(payload, "cma_fusion_query_embeddings").to(device=logits.device, dtype=logits.dtype)
    modality_query = _tensor(payload, "cma_modality_query_embeddings").to(device=logits.device, dtype=logits.dtype)
    availability = _tensor(payload, "availability_mask").to(device=logits.device, dtype=torch.bool)
    if availability.ndim != 3:
        raise ValueError(f"AMBER full availability_mask must be [B,K,T], got {tuple(availability.shape)}.")
    if fusion_query.ndim != 3:
        raise ValueError(f"AMBER full cma_fusion_query_embeddings must be [B,T,D], got {tuple(fusion_query.shape)}.")
    if modality_query.ndim != 4:
        raise ValueError(f"AMBER full cma_modality_query_embeddings must be [B,K,T,D], got {tuple(modality_query.shape)}.")
    positive = availability.permute(0, 2, 1).contiguous()
    if positive.shape != logits.shape:
        raise ValueError(f"AMBER full cma_logits shape {tuple(logits.shape)} does not match availability {tuple(positive.shape)}.")
    valid = positive.any(dim=-1)
    if not bool(valid.any().detach().cpu().item()):
        return zero
    log_prob = F.log_softmax(logits, dim=-1)
    positive_f = positive.to(dtype=logits.dtype)
    per_slot = -(log_prob * positive_f).sum(dim=-1) / positive_f.sum(dim=-1).clamp_min(1.0)
    return per_slot[valid].mean()


def _tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"AMBER full auxiliary loss requires payload tensor '{key}'.")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["amber_full_auxiliary_loss_from_output"]
