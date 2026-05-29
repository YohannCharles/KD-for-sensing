from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from kd_sensing.engine.batch import (
    assert_sensitive_fields_allowed,
    prepare_path_descriptors,
    prepare_path_semantic_labels,
    prepare_radio_semantic_labels,
)
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
    if normalized not in {"v4_adapter", "adapter", "v5_adapter_proto", "adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}:
        raise ValueError(f"Unsupported HiST-Beam adaptation strategy '{strategy}'.")
    for _, param in model.named_parameters():
        param.requires_grad = False
    trainable_prefixes = ("private_adapter", "fine_head")
    if normalized in {"v6_radio_proto", "adapter_radio_proto"}:
        trainable_prefixes = ("private_adapter", "fine_head", "radio_embedding")
    if normalized in {"v8_path_proto", "adapter_path_proto"}:
        trainable_prefixes = ("private_adapter", "fine_head", "path_embedding")
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
    label_budget: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    model.train()
    objective_cfg = cfg.get("hist_beam", {}).get("adaptation", {})
    entropy_weight = float(objective_cfg.get("entropy_weight", 0.01))
    prototype_weight = float(objective_cfg.get("prototype_weight", 0.1 if prototypes is not None else 0.0))
    proto_type = _resolve_proto_type(cfg, prototypes)
    tau = float(objective_cfg.get("proto_tau", objective_cfg.get("radio_tau", cfg.get("hist_beam", {}).get("radio_tau", 1.0))))
    bank_momentum = float(objective_cfg.get("target_proto_momentum", objective_cfg.get("target_private_momentum", objective_cfg.get("momentum", 0.9))))
    bank_warmup = int(objective_cfg.get("proto_warmup_epochs", objective_cfg.get("target_private_warmup_epochs", objective_cfg.get("warmup_epochs", 0))))
    bank_min_count = int(objective_cfg.get("target_private_min_count", objective_cfg.get("min_count", 1)))
    budget = int(label_budget if label_budget is not None else 0)
    allow_supervised_target = budget > 0
    allow_labeled_target_path_supervision = bool(objective_cfg.get("allow_labeled_target_path_supervision", False))
    diagnostics: dict[str, float] = {}
    leakage_flags = {
        "used_target_labels": False,
        "used_target_beam_for_training": False,
        "used_target_beam_power_for_training": False,
        "used_target_csi_for_training": False,
        "used_target_path_params_for_training": False,
        "used_target_path_descriptor_for_training": False,
        "used_target_path_label_for_training": False,
        "used_target_radio_label_for_training": False,
    }
    target_bank: TargetPrivatePrototypeBank | None = None
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
                step = _target_step(model, batch, cfg, device, require_labels=allow_supervised_target)
                if allow_supervised_target:
                    assert_sensitive_fields_allowed(
                        step.batch,
                        split="target_labeled",
                        label_budget=budget,
                        fields=("target_beam",),
                    )
                    radio_labels = prepare_radio_semantic_labels(
                        step.batch,
                        num_pred=step.labels.shape[1],
                        device=device,
                        non_blocking=transfer_non_blocking(cfg),
                    )
                    path_labels = None
                    path_targets = None
                    if allow_labeled_target_path_supervision:
                        assert_sensitive_fields_allowed(
                            step.batch,
                            split="target_labeled",
                            label_budget=budget,
                            fields=("path_semantic_label", "path_descriptor", "path_params"),
                            allow_labeled_target_path_supervision=True,
                        )
                        path_labels = prepare_path_semantic_labels(
                            step.batch,
                            num_pred=step.labels.shape[1],
                            device=device,
                            non_blocking=transfer_non_blocking(cfg),
                        )
                        path_targets = prepare_path_descriptors(
                            step.batch,
                            num_pred=step.labels.shape[1],
                            device=device,
                            non_blocking=transfer_non_blocking(cfg),
                        )
                    supervised = compute_hist_beam_loss(
                        {"logits": step.logits, **step.model_output.diagnostics},
                        step.labels,
                        cfg=cfg,
                        radio_semantic_labels=radio_labels,
                        path_semantic_labels=path_labels,
                        path_descriptors=path_targets[0] if path_targets is not None else None,
                        path_descriptor_mask=path_targets[1] if path_targets is not None else None,
                    )
                    loss = supervised.total
                    diagnostics.update(supervised.diagnostics)
                    leakage_flags["used_target_labels"] = True
                    leakage_flags["used_target_beam_for_training"] = True
                    if (
                        radio_labels is not None
                        and torch.is_tensor(step.model_output.diagnostics.get("radio_logits"))
                        and bool(radio_labels.ge(0).any().detach().cpu().item())
                    ):
                        leakage_flags["used_target_radio_label_for_training"] = True
                    if path_labels is not None and bool(path_labels.ge(0).any().detach().cpu().item()):
                        leakage_flags["used_target_path_label_for_training"] = True
                    if path_targets is not None and bool(path_targets[1].any().detach().cpu().item()):
                        leakage_flags["used_target_path_descriptor_for_training"] = True
                else:
                    loss = step.logits.sum() * 0.0
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
        if unlabeled_dataloader is not None:
            for batch in unlabeled_dataloader:
                unlabeled_batches += 1
                optimizer.zero_grad()
                step = _target_step(model, batch, cfg, device, require_labels=False)
                loss = step.logits.sum() * 0.0
                if entropy_weight > 0:
                    entropy = entropy_minimization_loss(step.logits)
                    loss = loss + entropy_weight * entropy
                    diagnostics["adaptation/entropy_loss"] = float(entropy.detach().cpu().item())
                if prototypes is not None and prototype_weight > 0 and proto_type == "path":
                    shared = step.model_output.diagnostics.get("shared_representation")
                    rep = step.model_output.diagnostics.get("adapter_representation")
                    if not torch.is_tensor(rep):
                        rep = step.model_output.diagnostics.get("private_representation")
                    mu_path = _path_prototypes_from_artifact(prototypes)
                    count_path = _path_counts_from_artifact(prototypes)
                    if torch.is_tensor(shared) and torch.is_tensor(rep) and torch.is_tensor(mu_path):
                        alpha, assign_metrics = path_prototype_assignment(
                            shared,
                            mu_path,
                            tau=tau,
                            counts=count_path,
                        )
                        confidence, k_hat = alpha.max(dim=-1)
                        if target_bank is None:
                            target_bank = TargetPrivatePrototypeBank(
                                num_classes=int(mu_path.shape[0]),
                                dim=int(rep.shape[-1]),
                                device=device,
                                dtype=rep.dtype,
                                momentum=bank_momentum,
                                min_count=bank_min_count,
                            )
                        update_metrics = target_bank.update(rep, k_hat, confidence, threshold=confidence_threshold)
                        diagnostics.update({f"adaptation/{k}": float(v) for k, v in assign_metrics.items()})
                        diagnostics.update({f"adaptation/{k}": float(v) for k, v in update_metrics.items()})
                        diagnostics["adaptation/path_assignment_confidence_mean"] = float(confidence.detach().mean().cpu().item())
                        diagnostics["adaptation/path_assignment_used_sample_count"] = float(
                            confidence.ge(confidence_threshold).sum().detach().cpu().item()
                        )
                        diagnostics["adaptation/path_assignment_histogram"] = [
                            float(item) for item in torch.bincount(k_hat.detach().cpu(), minlength=int(mu_path.shape[0])).tolist()
                        ]
                        if epoch_index + 1 > bank_warmup:
                            proto_loss, proto_metrics = target_bank.loss(rep, k_hat)
                            loss = loss + prototype_weight * proto_loss
                            diagnostics["adaptation/prototype_loss"] = float(proto_loss.detach().cpu().item())
                            diagnostics.update({f"adaptation/{k}": float(v) for k, v in proto_metrics.items()})
                            diagnostics["adaptation/prototype_status"] = (
                                1.0 if proto_metrics.get("target_private_prototype_used", 0.0) > 0.0 else 0.0
                            )
                    else:
                        diagnostics["adaptation/prototype_status"] = 0.0
                        diagnostics["adaptation/prototype_coverage"] = 0.0
                elif prototypes is not None and prototype_weight > 0 and proto_type == "radio_semantic":
                    shared = step.model_output.diagnostics.get("shared_representation")
                    rep = step.model_output.diagnostics.get("adapter_representation")
                    if not torch.is_tensor(rep):
                        rep = step.model_output.diagnostics.get("private_representation")
                    mu_radio = _radio_prototypes_from_artifact(prototypes)
                    count_radio = _radio_counts_from_artifact(prototypes)
                    if torch.is_tensor(shared) and torch.is_tensor(rep) and torch.is_tensor(mu_radio):
                        alpha, assign_metrics = radio_prototype_assignment(
                            shared,
                            mu_radio,
                            tau=tau,
                            counts=count_radio,
                        )
                        confidence, r_hat = alpha.max(dim=-1)
                        if target_bank is None:
                            target_bank = TargetPrivatePrototypeBank(
                                num_classes=int(mu_radio.shape[0]),
                                dim=int(rep.shape[-1]),
                                device=device,
                                dtype=rep.dtype,
                                momentum=bank_momentum,
                                min_count=bank_min_count,
                            )
                        update_metrics = target_bank.update(rep, r_hat, confidence, threshold=confidence_threshold)
                        diagnostics.update({f"adaptation/{k}": float(v) for k, v in assign_metrics.items()})
                        diagnostics.update({f"adaptation/{k}": float(v) for k, v in update_metrics.items()})
                        diagnostics["adaptation/radio_assignment_confidence_mean"] = float(confidence.detach().mean().cpu().item())
                        diagnostics["adaptation/radio_assignment_used_sample_count"] = float(
                            confidence.ge(confidence_threshold).sum().detach().cpu().item()
                        )
                        if epoch_index + 1 > bank_warmup:
                            proto_loss, proto_metrics = target_bank.loss(rep, r_hat)
                            loss = loss + prototype_weight * proto_loss
                            diagnostics["adaptation/prototype_loss"] = float(proto_loss.detach().cpu().item())
                            diagnostics.update({f"adaptation/{k}": float(v) for k, v in proto_metrics.items()})
                            diagnostics["adaptation/prototype_status"] = (
                                1.0 if proto_metrics.get("target_private_prototype_used", 0.0) > 0.0 else 0.0
                            )
                    else:
                        diagnostics["adaptation/prototype_status"] = 0.0
                        diagnostics["adaptation/prototype_coverage"] = 0.0
                elif prototypes is not None and prototype_weight > 0 and proto_type == "coarse":
                    proto = prototypes.get("adapter_prototypes", prototypes.get("private_prototypes"))
                    rep = step.model_output.diagnostics.get("adapter_representation")
                    if not torch.is_tensor(rep):
                        rep = step.model_output.diagnostics.get("private_representation")
                    coarse_logits = step.model_output.diagnostics.get("coarse_logits")
                    if torch.is_tensor(rep):
                        proto_loss, proto_metrics = prototype_consistency_loss(
                            rep,
                            proto,
                            confidence_threshold=confidence_threshold,
                            coarse_logits=coarse_logits if torch.is_tensor(coarse_logits) else None,
                            counts=prototypes.get("counts"),
                        )
                        loss = loss + prototype_weight * proto_loss
                        diagnostics["adaptation/prototype_loss"] = float(proto_loss.detach().cpu().item())
                        diagnostics.update({f"adaptation/{k}": float(v) for k, v in proto_metrics.items()})
                        diagnostics["adaptation/prototype_status"] = (
                            1.0 if proto_metrics.get("prototype_used", 0.0) > 0.0 else 0.0
                        )
                    else:
                        diagnostics["adaptation/prototype_status"] = 0.0
                elif prototypes is None:
                    diagnostics.setdefault("adaptation/prototype_status", 0.0)
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
        "proto_type": proto_type,
        **leakage_flags,
        "diagnostics": diagnostics,
    }


def radio_prototype_assignment(
    representation: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    tau: float = 1.0,
    counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if representation.ndim == 3:
        representation = representation[:, 0, :]
    if prototypes.ndim != 2:
        raise ValueError(f"radio prototypes must have shape [C, D], got {tuple(prototypes.shape)}.")
    rep = F.normalize(representation, dim=-1)
    proto = F.normalize(prototypes.to(device=rep.device, dtype=rep.dtype), dim=-1)
    scores = rep @ proto.t()
    available = torch.ones(proto.shape[0], dtype=torch.bool, device=rep.device)
    if counts is not None:
        available = counts.to(device=rep.device).reshape(-1).gt(0)
        if available.numel() != proto.shape[0]:
            available = torch.ones(proto.shape[0], dtype=torch.bool, device=rep.device)
    if not bool(available.any().detach().cpu().item()):
        alpha = torch.zeros(scores.shape, dtype=rep.dtype, device=rep.device)
        return alpha, {
            "prototype_coverage": 0.0,
            "prototype_used": 0.0,
            "prototype_confidence_mean": 0.0,
            "radio_prototype_available_classes": 0.0,
        }
    scores = scores.masked_fill(~available.view(1, -1), -1e9)
    alpha = torch.softmax(scores / max(float(tau), 1e-6), dim=-1)
    confidence, _ = alpha.max(dim=-1)
    return alpha, {
        "prototype_coverage": float(available.float().mean().detach().cpu().item()),
        "prototype_used": float(representation.shape[0]),
        "prototype_confidence_mean": float(confidence.detach().mean().cpu().item()),
        "radio_prototype_available_classes": float(available.sum().detach().cpu().item()),
    }


def path_prototype_assignment(
    representation: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    tau: float = 1.0,
    counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if representation.ndim == 3:
        representation = representation[:, 0, :]
    if prototypes.ndim != 2:
        raise ValueError(f"path prototypes must have shape [K, D], got {tuple(prototypes.shape)}.")
    rep = F.normalize(representation, dim=-1)
    proto = F.normalize(prototypes.to(device=rep.device, dtype=rep.dtype), dim=-1)
    scores = rep @ proto.t()
    available = torch.ones(proto.shape[0], dtype=torch.bool, device=rep.device)
    if counts is not None:
        available = counts.to(device=rep.device).reshape(-1).gt(0)
        if available.numel() != proto.shape[0]:
            available = torch.ones(proto.shape[0], dtype=torch.bool, device=rep.device)
    if not bool(available.any().detach().cpu().item()):
        alpha = torch.zeros(scores.shape, dtype=rep.dtype, device=rep.device)
        return alpha, {
            "prototype_coverage": 0.0,
            "prototype_used": 0.0,
            "prototype_confidence_mean": 0.0,
            "path_prototype_available_classes": 0.0,
        }
    scores = scores.masked_fill(~available.view(1, -1), -1e9)
    alpha = torch.softmax(scores / max(float(tau), 1e-6), dim=-1)
    confidence, _ = alpha.max(dim=-1)
    return alpha, {
        "prototype_coverage": float(available.float().mean().detach().cpu().item()),
        "prototype_used": float(representation.shape[0]),
        "prototype_confidence_mean": float(confidence.detach().mean().cpu().item()),
        "path_prototype_available_classes": float(available.sum().detach().cpu().item()),
    }


class TargetPrivatePrototypeBank:
    def __init__(
        self,
        *,
        num_classes: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        momentum: float = 0.9,
        min_count: int = 1,
    ) -> None:
        self.prototypes = torch.zeros(int(num_classes), int(dim), device=device, dtype=dtype)
        self.counts = torch.zeros(int(num_classes), device=device, dtype=torch.long)
        self.momentum = float(momentum)
        self.min_count = int(min_count)

    def update(
        self,
        representation: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        *,
        threshold: float,
    ) -> dict[str, float]:
        if representation.ndim == 3:
            representation = representation[:, 0, :]
        labels = labels.to(device=representation.device, dtype=torch.long).reshape(-1)
        confidence = confidence.to(device=representation.device).reshape(-1)
        used = 0
        for class_index in range(self.prototypes.shape[0]):
            mask = labels.eq(class_index) & confidence.ge(float(threshold))
            if not torch.any(mask):
                continue
            mean_rep = representation[mask].detach().mean(dim=0)
            if self.counts[class_index] <= 0:
                self.prototypes[class_index] = mean_rep
            else:
                self.prototypes[class_index] = self.momentum * self.prototypes[class_index] + (1.0 - self.momentum) * mean_rep
            self.counts[class_index] += int(mask.sum().item())
            used += int(mask.sum().item())
        initialized = self.counts.ge(self.min_count)
        return {
            "target_private_update_used": float(used),
            "target_private_initialized_count": float(initialized.sum().detach().cpu().item()),
            "target_private_initialized_ratio": float(initialized.float().mean().detach().cpu().item()),
        }

    def loss(self, representation: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if representation.ndim == 3:
            representation = representation[:, 0, :]
        labels = labels.to(device=representation.device, dtype=torch.long).reshape(-1)
        in_range = labels.ge(0) & labels.lt(self.prototypes.shape[0])
        initialized = torch.zeros_like(in_range)
        if torch.any(in_range):
            initialized[in_range] = self.counts[labels[in_range]].ge(self.min_count)
        valid = in_range & initialized
        if not torch.any(valid):
            return representation.sum() * 0.0, {
                "target_private_prototype_used": 0.0,
                "target_private_prototype_coverage": 0.0,
            }
        selected = self.prototypes.detach()[labels[valid]]
        loss = 1.0 - F.cosine_similarity(F.normalize(representation[valid], dim=-1), F.normalize(selected, dim=-1), dim=-1).mean()
        return loss, {
            "target_private_prototype_used": float(valid.sum().detach().cpu().item()),
            "target_private_prototype_coverage": float(valid.float().mean().detach().cpu().item()),
        }


def _resolve_proto_type(cfg: dict[str, Any], prototypes: dict[str, Any] | None) -> str:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    proto_cfg = hist_cfg.get("prototype", {}) if isinstance(hist_cfg.get("prototype"), dict) else {}
    adapt_cfg = hist_cfg.get("adaptation", {}) if isinstance(hist_cfg.get("adaptation"), dict) else {}
    explicit = hist_cfg.get("proto_type", adapt_cfg.get("proto_type", proto_cfg.get("proto_type")))
    if explicit:
        return str(explicit).strip().lower()
    if prototypes is None:
        return "none"
    metadata = prototypes.get("metadata", {}) if isinstance(prototypes, dict) else {}
    if str(metadata.get("prototype_space", "")).strip().lower() == "shared_path_physical":
        return "path"
    if str(metadata.get("prototype_space", "")).strip().lower() == "shared_radio_semantic":
        return "radio_semantic"
    return "coarse"


def _radio_prototypes_from_artifact(prototypes: dict[str, Any]) -> torch.Tensor | None:
    value = prototypes.get("mu_radio_c", prototypes.get("radio_prototypes"))
    if value is None and str((prototypes.get("metadata") or {}).get("prototype_space", "")) == "shared_radio_semantic":
        value = prototypes.get("shared_prototypes")
    return value if torch.is_tensor(value) else None


def _radio_counts_from_artifact(prototypes: dict[str, Any]) -> torch.Tensor | None:
    value = prototypes.get("count_radio", prototypes.get("radio_counts"))
    if value is None and str((prototypes.get("metadata") or {}).get("prototype_space", "")) == "shared_radio_semantic":
        value = prototypes.get("counts")
    return value if torch.is_tensor(value) else None


def _path_prototypes_from_artifact(prototypes: dict[str, Any]) -> torch.Tensor | None:
    value = prototypes.get("mu_path_c", prototypes.get("path_prototypes"))
    if value is None and str((prototypes.get("metadata") or {}).get("prototype_space", "")) == "shared_path_physical":
        value = prototypes.get("shared_prototypes")
    return value if torch.is_tensor(value) else None


def _path_counts_from_artifact(prototypes: dict[str, Any]) -> torch.Tensor | None:
    value = prototypes.get("count_path", prototypes.get("path_counts"))
    if value is None and str((prototypes.get("metadata") or {}).get("prototype_space", "")) == "shared_path_physical":
        value = prototypes.get("counts")
    return value if torch.is_tensor(value) else None


def _target_step(model, batch, cfg: dict[str, Any], device: torch.device, *, require_labels: bool = True):
    model_cfg = cfg["model"]
    return run_model_step(
        model,
        cfg["experiment"].get("task", "fusion"),
        batch,
        model_cfg=model_cfg.get("student", model_cfg),
        seq_length=model_cfg.get("seq_length_student", cfg.get("data", {}).get("dataset", {}).get("seq_len", 8)),
        num_pred=model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)),
        device=device,
        downsample_ratio=model_cfg.get("downsample_ratio", 1) if require_labels else None,
        non_blocking=transfer_non_blocking(cfg),
    )


__all__ = [
    "TrainableParameterSummary",
    "TargetPrivatePrototypeBank",
    "adapt_hist_beam_target",
    "apply_hist_beam_adaptation_strategy",
    "path_prototype_assignment",
    "radio_prototype_assignment",
    "trainable_parameter_summary",
]
