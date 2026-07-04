
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from kd_sensing.engine.model_output import ModelOutput


def amr_net_loss_from_output(
    output: ModelOutput,
    labels: torch.Tensor,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    amr = output.diagnostics.get("amr")
    zero = output.logits.sum() * 0.0
    if not isinstance(amr, Mapping):
        return zero, {}
    cfg = _loss_cfg(cfg)
    if not bool(cfg.get("enabled", False)):
        return zero, {}
    modality_logits = _mapping(amr.get("modality_logits"))
    mu = _mapping(amr.get("mu"))
    logvar = _mapping(amr.get("logvar"))
    modalities = [str(item) for item in amr.get("modalities", modality_logits.keys())]
    targets = _match_time(labels.to(device=output.logits.device, dtype=torch.long), int(output.logits.shape[1]))
    valid = targets.ge(0) & targets.lt(int(output.logits.shape[-1]))

    total_ce = zero
    total_kl = zero
    diagnostics: dict[str, float] = {}
    for name in modalities:
        logits = modality_logits.get(name)
        if not torch.is_tensor(logits):
            continue
        logits = _match_time(logits.to(device=output.logits.device), int(targets.shape[1]))
        if bool(valid.any().detach().cpu().item()):
            ce = F.cross_entropy(logits[valid], targets[valid])
        else:
            ce = zero
        total_ce = total_ce + ce
        kl = _gaussian_kl(mu.get(name), logvar.get(name), zero)
        total_kl = total_kl + kl
        diagnostics[f"loss/amr_{name}_ce"] = float(ce.detach().cpu().item())
        diagnostics[f"loss/amr_{name}_kl"] = float(kl.detach().cpu().item())

    pre, skipped = _pre_loss(mu, logvar, labels, modalities, zero, cfg)
    alpha = float(cfg.get("alpha", 0.01))
    beta = float(cfg.get("beta", 1.0))
    weight = float(cfg.get("weight", 1.0))
    total = weight * (total_ce + alpha * total_kl + beta * pre)
    diagnostics.update(
        {
            "loss/amr_ce": float(total_ce.detach().cpu().item()),
            "loss/amr_kl": float(total_kl.detach().cpu().item()),
            "loss/amr_pre": float(pre.detach().cpu().item()),
            "loss/amr_total": float(total.detach().cpu().item()),
            "amr/pre_skipped_anchors": float(skipped),
            "amr/pre_samples": float(max(int(cfg.get("pre_samples", 2)), 1)),
            "amr/loss_weight": float(weight),
            "amr/alpha": float(alpha),
            "amr/beta": float(beta),
        }
    )
    return total, diagnostics


def _loss_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    loss = cfg.get("loss", {}) if isinstance(cfg, Mapping) else {}
    raw = loss.get("amr", loss.get("amr_net", {})) if isinstance(loss, Mapping) else {}
    return raw if isinstance(raw, dict) else {}


def _gaussian_kl(mu: Any, logvar: Any, zero: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mu) or not torch.is_tensor(logvar):
        return zero
    latent_dim = max(int(mu.shape[-1]), 1)
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1) / float(latent_dim)).mean()


def _pre_loss(
    mu: Mapping[str, Any],
    logvar: Mapping[str, Any],
    labels: torch.Tensor,
    modalities: list[str],
    zero: torch.Tensor,
    cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, int]:
    if not bool(cfg.get("pre_enabled", cfg.get("pre", True))):
        return zero, 0
    targets = labels[:, -1].to(device=zero.device, dtype=torch.long)
    temperature = max(float(cfg.get("temperature", 0.1)), 1e-6)
    samples = max(int(cfg.get("pre_samples", 2)), 1)
    losses: list[torch.Tensor] = []
    skipped = 0
    for name in modalities:
        modality_mu = mu.get(name)
        modality_logvar = logvar.get(name)
        if not torch.is_tensor(modality_mu) or not torch.is_tensor(modality_logvar):
            continue
        modality_mu = modality_mu.to(device=zero.device)
        modality_logvar = modality_logvar.to(device=zero.device, dtype=modality_mu.dtype)
        noise = torch.randn(
            int(modality_mu.shape[0]),
            samples,
            int(modality_mu.shape[-1]),
            dtype=modality_mu.dtype,
            device=modality_mu.device,
        )
        sampled = modality_mu.unsqueeze(1) + noise * torch.exp(0.5 * modality_logvar).unsqueeze(1)
        features = F.normalize(sampled.reshape(int(modality_mu.shape[0]) * samples, -1), dim=-1)
        flat_labels = targets.repeat_interleave(samples)
        flat_valid = flat_labels.ge(0)
        for index in range(features.shape[0]):
            if not bool(flat_valid[index]):
                skipped += 1
                continue
            positives = flat_labels.eq(flat_labels[index]) & flat_valid
            positives[index] = False
            if not bool(positives.any().detach().cpu().item()):
                skipped += 1
                continue
            candidates = flat_valid.clone()
            candidates[index] = False
            logits = features[index].matmul(features[candidates].T) / temperature
            local_positive = positives[candidates]
            losses.append(-(logits.log_softmax(dim=0)[local_positive]).mean())
    if not losses:
        return zero, skipped
    return torch.stack(losses).mean(), skipped


def _match_time(value: torch.Tensor, target_steps: int) -> torch.Tensor:
    if value.ndim < 2 or int(value.shape[1]) == int(target_steps):
        return value
    if int(value.shape[1]) > int(target_steps):
        return value[:, -int(target_steps) :, ...]
    pad_shape = (int(value.shape[0]), int(target_steps) - int(value.shape[1]), *tuple(value.shape[2:]))
    pad = torch.full(pad_shape, -100, dtype=value.dtype, device=value.device)
    return torch.cat([pad, value], dim=1)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
