from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def masked_latent_smooth_l1_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    ema_momentum: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if predicted.shape != target.shape:
        raise ValueError(
            f"JEPA-MSAC masked latent loss requires matching predicted/target shapes, "
            f"got {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if loss_mask is None:
        selected_predicted = predicted
        selected_target = target.detach()
        mask_ratio = 1.0
        target_token_count = int(predicted.shape[1]) if predicted.ndim >= 2 else int(predicted.numel())
    else:
        if loss_mask.shape != predicted.shape[: loss_mask.ndim]:
            expected = tuple(predicted.shape[: loss_mask.ndim])
            raise ValueError(f"JEPA-MSAC loss_mask shape must prefix predicted shape {expected}, got {tuple(loss_mask.shape)}.")
        mask = loss_mask.to(device=predicted.device, dtype=torch.bool)
        selected_predicted = predicted[mask]
        selected_target = target.detach()[mask]
        mask_ratio = float(mask.float().mean().detach().cpu().item())
        target_token_count = int(mask.sum().detach().cpu().item())
    if selected_predicted.numel() == 0:
        raise ValueError("JEPA-MSAC masked latent loss received an empty masked token set.")
    loss = F.smooth_l1_loss(selected_predicted, selected_target, beta=float(beta))
    diagnostics = {
        "loss_name": "jepa_msac_masked_smooth_l1",
        "mask_ratio": mask_ratio,
        "target_token_count": target_token_count,
        "ema_momentum": None if ema_momentum is None else float(ema_momentum),
        "predicted_latent_norm": float(predicted.detach().norm(dim=-1).mean().cpu().item()),
        "target_latent_norm": float(target.detach().norm(dim=-1).mean().cpu().item()),
    }
    return loss, diagnostics


def jepa_msac_stage2_losses(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    if "predicted_location" in outputs and "future_location" in targets:
        losses["localization_l1"] = F.l1_loss(outputs["predicted_location"], targets["future_location"].to(outputs["predicted_location"].device))
    if "beam_logits" in outputs and "future_beam" in targets:
        logits = outputs["beam_logits"]
        target = targets["future_beam"].to(device=logits.device, dtype=torch.long)
        losses["beam_cross_entropy"] = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    if "rssi_profile" in outputs and "future_rssi_profile" in targets:
        losses["rssi_profile_smooth_l1"] = F.smooth_l1_loss(
            outputs["rssi_profile"],
            targets["future_rssi_profile"].to(outputs["rssi_profile"].device),
        )
    if "scalar_rssi" in outputs and "future_rssi_scalar" in targets:
        losses["rssi_scalar_smooth_l1"] = F.smooth_l1_loss(
            outputs["scalar_rssi"],
            targets["future_rssi_scalar"].to(outputs["scalar_rssi"].device),
        )
    if losses:
        losses["total"] = sum(losses.values())
    return losses


def jepa_msac_pretraining_metadata(*, loss_name: str = "jepa_msac_masked_smooth_l1") -> dict[str, Any]:
    return {
        "objective": "jepa_msac_pretraining",
        "primary_loss": loss_name,
        "early_stopping_metric": "val_jepa_msac_loss",
        "early_stopping_mode": "min",
        "self_supervised": True,
        "paper_workflow_baseline": "jepa_msac",
    }


__all__ = [
    "masked_latent_smooth_l1_loss",
    "jepa_msac_stage2_losses",
    "jepa_msac_pretraining_metadata",
]
