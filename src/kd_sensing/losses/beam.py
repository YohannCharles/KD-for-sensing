import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import LOSSES


@LOSSES.register("focal_loss")
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = "mean", ignore_index: int = -100, **_: object):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype.is_floating_point:
            return self._forward_soft(inputs, targets)
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction="none")
        valid = targets.ne(self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss[valid].mean() if torch.any(valid) else focal_loss.sum() * 0.0
        if self.reduction == "sum":
            return focal_loss[valid].sum()
        return focal_loss

    def _forward_soft(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != inputs.shape:
            raise ValueError(f"soft targets must have shape {tuple(inputs.shape)}, got {tuple(targets.shape)}.")
        valid = torch.isfinite(targets).all(dim=-1) & targets.sum(dim=-1).gt(0)
        normalized = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()
        ce_loss = -(normalized * log_probs).sum(dim=-1)
        pt = (normalized * probs).sum(dim=-1).clamp(min=0.0, max=1.0)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).masked_fill(~valid, 0.0)
        if self.reduction == "mean":
            return focal_loss[valid].mean() if torch.any(valid) else focal_loss.sum() * 0.0
        if self.reduction == "sum":
            return focal_loss[valid].sum()
        return focal_loss


@LOSSES.register("soft_cross_entropy")
class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        **_: object,
    ):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not targets.dtype.is_floating_point:
            return F.cross_entropy(
                inputs,
                targets,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )
        if targets.shape != inputs.shape:
            raise ValueError(f"soft targets must have shape {tuple(inputs.shape)}, got {tuple(targets.shape)}.")
        valid = torch.isfinite(targets).all(dim=-1) & targets.sum(dim=-1).gt(0)
        normalized = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.label_smoothing > 0:
            smooth = min(max(self.label_smoothing, 0.0), 1.0)
            normalized = normalized * (1.0 - smooth) + smooth / max(int(inputs.shape[-1]), 1)
        loss = -(normalized * F.log_softmax(inputs, dim=-1)).sum(dim=-1).masked_fill(~valid, 0.0)
        if self.reduction == "mean":
            return loss[valid].mean() if torch.any(valid) else loss.sum() * 0.0
        if self.reduction == "sum":
            return loss[valid].sum()
        return loss


LOSSES.register("cross_entropy")(SoftTargetCrossEntropyLoss)


__all__ = ["FocalLoss", "SoftTargetCrossEntropyLoss"]
