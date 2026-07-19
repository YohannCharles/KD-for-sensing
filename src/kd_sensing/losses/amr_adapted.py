from typing import Any

import torch

from kd_sensing.engine.model_output import ModelOutput


def amr_adapted_auxiliary_loss_from_output(
    model_output: ModelOutput,
    cfg: dict[str, Any],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    auxiliary = loss_cfg.get("auxiliary", {}) if isinstance(loss_cfg.get("auxiliary"), dict) else {}
    raw = auxiliary.get("amr_adapted", {}) if isinstance(auxiliary.get("amr_adapted"), dict) else {}
    if not bool(raw.get("enabled", False)):
        return zero, {}
    payload = model_output.diagnostics.get("amr_adapted_auxiliary")
    if not isinstance(payload, dict):
        if not torch.is_grad_enabled():
            return zero, {}
        raise ValueError("AMR adapted auxiliary loss is enabled but payload is missing.")
    mu = _tensor(payload, "mu")
    logvar = _tensor(payload, "logvar").to(device=mu.device, dtype=mu.dtype)
    availability = _tensor(payload, "availability").to(device=mu.device, dtype=torch.bool)
    fused_mu = _tensor(payload, "fused_mu").to(device=mu.device, dtype=mu.dtype)
    active = availability.unsqueeze(-1).to(dtype=mu.dtype)
    count = active.sum().clamp_min(1.0)
    kl = 0.5 * ((mu.pow(2) + logvar.exp() - 1.0 - logvar) * active).sum() / (count * mu.shape[-1])
    consistency = ((mu - fused_mu.unsqueeze(1)).pow(2) * active).sum() / (count * mu.shape[-1])
    kl_weight = float(raw.get("kl_weight", 0.01))
    consistency_weight = float(raw.get("consistency_weight", 0.05))
    total = kl_weight * kl + consistency_weight * consistency
    return total, {
        "loss/amr_adapted_kl": float((kl_weight * kl).detach().cpu().item()),
        "loss/amr_adapted_consistency": float((consistency_weight * consistency).detach().cpu().item()),
        "loss/amr_adapted_total": float(total.detach().cpu().item()),
    }


def _tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"AMR adapted auxiliary loss requires payload tensor {key!r}.")
    return value


__all__ = ["amr_adapted_auxiliary_loss_from_output"]
