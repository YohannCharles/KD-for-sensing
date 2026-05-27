from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from kd_sensing.engine.hist_beam_losses import (
    compute_hist_beam_loss,
    entropy_minimization_loss,
    prototype_consistency_loss,
)
from kd_sensing.engine.runtime import run_model_step, transfer_non_blocking


@dataclass(frozen=True)
class TrainableParameterSummary:
    trainable: int
    total: int
    ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "trainable_params": self.trainable,
            "total_params": self.total,
            "trainable_ratio": self.ratio,
        }


def trainable_parameter_summary(model) -> TrainableParameterSummary:
    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    return TrainableParameterSummary(trainable=trainable, total=total, ratio=float(trainable / max(total, 1)))


def apply_hist_beam_adaptation_strategy(
    model,
    strategy: str,
    *,
    train_layernorm_affine: bool = False,
) -> dict[str, Any]:
    normalized = str(strategy).strip().lower()
    if normalized in {"v6_full_finetune", "full_finetune", "full"}:
        for param in model.parameters():
            param.requires_grad = True
        summary = trainable_parameter_summary(model)
        return {"strategy": normalized, **summary.to_dict()}
    if normalized not in {"v4_adapter", "adapter", "v5_adapter_proto", "adapter_proto"}:
        raise ValueError(f"Unsupported HiST-Beam adaptation strategy '{strategy}'.")
    for _, param in model.named_parameters():
        param.requires_grad = False
    trainable_prefixes = ("private_adapter", "fine_head")
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad = True
        if train_layernorm_affine and ("norm" in name.lower() or "layernorm" in name.lower()):
            param.requires_grad = True
    summary = trainable_parameter_summary(model)
    return {"strategy": normalized, **summary.to_dict()}


def adapt_hist_beam_target(
    model,
    labeled_dataloader,
    unlabeled_dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    prototypes: dict[str, Any] | None = None,
    epochs: int = 1,
    confidence_threshold: float = 0.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    model.train()
    objective_cfg = cfg.get("hist_beam", {}).get("adaptation", {})
    entropy_weight = float(objective_cfg.get("entropy_weight", 0.01))
    prototype_weight = float(objective_cfg.get("prototype_weight", 0.1 if prototypes is not None else 0.0))
    diagnostics: dict[str, float] = {}
    total_epochs = int(epochs)
    for epoch_index in range(total_epochs):
        epoch_start = time.perf_counter()
        epoch_losses: list[float] = []
        labeled_batches = 0
        unlabeled_batches = 0
        if labeled_dataloader is not None:
            for batch in labeled_dataloader:
                labeled_batches += 1
                optimizer.zero_grad()
                step = _target_step(model, batch, cfg, device)
                supervised = compute_hist_beam_loss(
                    {"logits": step.logits, **step.model_output.diagnostics},
                    step.labels,
                    cfg=cfg,
                )
                loss = supervised.total
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
                diagnostics.update(supervised.diagnostics)
        if unlabeled_dataloader is not None:
            for batch in unlabeled_dataloader:
                unlabeled_batches += 1
                optimizer.zero_grad()
                step = _target_step(model, batch, cfg, device)
                loss = step.logits.sum() * 0.0
                if entropy_weight > 0:
                    entropy = entropy_minimization_loss(step.logits)
                    loss = loss + entropy_weight * entropy
                    diagnostics["adaptation/entropy_loss"] = float(entropy.detach().cpu().item())
                if prototypes is not None and prototype_weight > 0:
                    proto = prototypes["shared_prototypes"]
                    rep = step.model_output.diagnostics.get("shared_representation")
                    if torch.is_tensor(rep):
                        proto_loss, proto_metrics = prototype_consistency_loss(
                            rep,
                            proto,
                            confidence_threshold=confidence_threshold,
                        )
                        loss = loss + prototype_weight * proto_loss
                        diagnostics["adaptation/prototype_loss"] = float(proto_loss.detach().cpu().item())
                        diagnostics.update({f"adaptation/{k}": float(v) for k, v in proto_metrics.items()})
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch_index + 1,
                    "epochs": total_epochs,
                    "duration_seconds": float(time.perf_counter() - epoch_start),
                    "loss_last": epoch_losses[-1] if epoch_losses else None,
                    "loss_mean": float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else None,
                    "labeled_batches": labeled_batches,
                    "unlabeled_batches": unlabeled_batches,
                }
            )
    elapsed = time.perf_counter() - start
    params = trainable_parameter_summary(model)
    return {
        "epochs": int(epochs),
        "adaptation_time_seconds": float(elapsed),
        "adaptation_time_per_epoch": float(elapsed / max(int(epochs), 1)),
        **params.to_dict(),
        "diagnostics": diagnostics,
    }


def _target_step(model, batch, cfg: dict[str, Any], device: torch.device):
    model_cfg = cfg["model"]
    return run_model_step(
        model,
        cfg["experiment"].get("task", "fusion"),
        batch,
        model_cfg=model_cfg.get("student", model_cfg),
        seq_length=model_cfg.get("seq_length_student", cfg.get("data", {}).get("dataset", {}).get("seq_len", 8)),
        num_pred=model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)),
        device=device,
        downsample_ratio=model_cfg.get("downsample_ratio", 1),
        non_blocking=transfer_non_blocking(cfg),
    )


__all__ = [
    "TrainableParameterSummary",
    "adapt_hist_beam_target",
    "apply_hist_beam_adaptation_strategy",
    "trainable_parameter_summary",
]
