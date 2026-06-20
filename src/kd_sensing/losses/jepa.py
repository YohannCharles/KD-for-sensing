from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JepaLossResult:
    loss: torch.Tensor
    diagnostics: dict[str, float]


def jepa_latent_prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    cfg: dict[str, Any] | None = None,
) -> JepaLossResult:
    """Compute masked latent-space JEPA loss without silently accepting empty masks."""

    if predicted.shape != target.shape:
        raise ValueError(
            "JEPA predicted_target_latent and target_latent must have identical shape, "
            f"got {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.ndim != 4:
        raise ValueError(f"JEPA latents must have shape [B, T, N_tgt, D], got {tuple(predicted.shape)}.")
    if loss_mask.shape != predicted.shape[:-1]:
        raise ValueError(
            f"JEPA loss_mask must have shape {tuple(predicted.shape[:-1])}, got {tuple(loss_mask.shape)}."
        )

    loss_cfg = _jepa_loss_cfg(cfg)
    mask = loss_mask.to(device=predicted.device, dtype=torch.bool)
    finite = torch.isfinite(predicted).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    mask = mask & finite
    valid_tokens = int(mask.sum().detach().cpu().item())
    if valid_tokens <= 0:
        raise ValueError("JEPA loss_mask has no valid target tokens; refusing to produce a NaN loss.")

    pred = predicted
    tgt = target.detach()
    if bool(loss_cfg.get("latent_normalize", loss_cfg.get("normalize_latent", False))):
        pred = F.normalize(pred, dim=-1)
        tgt = F.normalize(tgt, dim=-1)

    loss_type = str(loss_cfg.get("type", "mse")).strip().lower()
    if loss_type in {"smooth_l1", "huber"}:
        per_dim = F.smooth_l1_loss(
            pred,
            tgt,
            reduction="none",
            beta=float(loss_cfg.get("beta", 1.0)),
        )
    elif loss_type == "mse":
        per_dim = (pred - tgt).pow(2)
    else:
        raise ValueError("loss.jepa.type must be one of mse, smooth_l1, or huber.")
    per_token = per_dim.mean(dim=-1)
    loss = per_token[mask].mean() * float(loss_cfg.get("weight", 1.0))
    diagnostics = {
        "loss/jepa": float(loss.detach().cpu().item()),
        "jepa/valid_target_tokens": float(valid_tokens),
        "jepa/loss_mask_ratio": float(mask.float().mean().detach().cpu().item()),
    }
    return JepaLossResult(loss=loss, diagnostics=diagnostics)


def jepa_loss_from_output(output: Any, cfg: dict[str, Any] | None = None) -> JepaLossResult:
    diagnostics = getattr(output, "diagnostics", output)
    if not isinstance(diagnostics, dict):
        raise TypeError("JEPA loss expects a ModelOutput or diagnostics dict.")
    missing = [
        key
        for key in ("predicted_target_latent", "target_latent", "loss_mask")
        if key not in diagnostics or not torch.is_tensor(diagnostics[key])
    ]
    if missing:
        raise ValueError(f"JEPA output is missing tensor field(s): {', '.join(missing)}.")
    return jepa_latent_prediction_loss(
        diagnostics["predicted_target_latent"],
        diagnostics["target_latent"],
        diagnostics["loss_mask"],
        cfg,
    )


def _jepa_loss_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    loss_cfg = cfg.get("loss")
    if not isinstance(loss_cfg, dict):
        return {}
    jepa_cfg = loss_cfg.get("jepa")
    if isinstance(jepa_cfg, dict):
        return jepa_cfg
    objective_cfg = loss_cfg.get("objective")
    if isinstance(objective_cfg, dict) and isinstance(objective_cfg.get("jepa"), dict):
        return objective_cfg["jepa"]
    return {}


__all__ = ["JepaLossResult", "jepa_latent_prediction_loss", "jepa_loss_from_output"]
