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


@LOSSES.register("beam_neighborhood_ce")
class BeamNeighborhoodCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        sigma: float = 1.5,
        circular: bool = True,
        mix_ce: float = 0.5,
        reduction: str = "mean",
        ignore_index: int = -100,
        **_: object,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.circular = bool(circular)
        self.mix_ce = float(mix_ce)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)
        self._printed_shape: int | None = None
        if self.sigma <= 0.0:
            raise ValueError(f"beam_neighborhood_ce sigma must be positive, got {self.sigma}.")
        if not 0.0 <= self.mix_ce <= 1.0:
            raise ValueError(f"beam_neighborhood_ce mix_ce must be in [0, 1], got {self.mix_ce}.")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_beams = int(inputs.shape[-1])
        if self._printed_shape != num_beams:
            print(
                "[BeamNeighborhoodLoss] "
                f"num_beams={num_beams}, sigma={self.sigma:g}, circular={str(self.circular).lower()}, "
                f"mix_ce={self.mix_ce:g}"
            )
            self._printed_shape = num_beams
        if targets.dtype.is_floating_point:
            return self._soft_ce(inputs, targets)
        flat_inputs = inputs.reshape(-1, num_beams)
        flat_targets = targets.reshape(-1).to(device=flat_inputs.device, dtype=torch.long)
        valid = flat_targets.ne(self.ignore_index) & flat_targets.ge(0) & flat_targets.lt(num_beams)
        safe_targets = flat_targets.masked_fill(~valid, self.ignore_index)
        hard_loss = F.cross_entropy(
            flat_inputs,
            safe_targets,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        soft_targets = self.soft_targets(flat_inputs, flat_targets)
        soft_loss = self._soft_ce(flat_inputs, soft_targets)
        return (1.0 - self.mix_ce) * hard_loss + self.mix_ce * soft_loss

    def soft_targets(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_beams = int(inputs.shape[-1])
        flat_targets = targets.reshape(-1).to(device=inputs.device, dtype=torch.long)
        valid = flat_targets.ne(self.ignore_index) & flat_targets.ge(0) & flat_targets.lt(num_beams)
        classes = torch.arange(num_beams, device=inputs.device, dtype=torch.float32).view(1, num_beams)
        center = flat_targets.to(dtype=torch.float32).view(-1, 1)
        distance = (classes - center).abs()
        if self.circular:
            distance = torch.minimum(distance, float(num_beams) - distance)
        logits = -0.5 * (distance / self.sigma).pow(2)
        target = torch.softmax(logits, dim=-1).to(dtype=inputs.dtype)
        return torch.where(valid.view(-1, 1), target, torch.zeros_like(target))

    def _soft_ce(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_targets = targets.reshape(-1, inputs.shape[-1]).to(device=flat_inputs.device, dtype=flat_inputs.dtype)
        valid = torch.isfinite(flat_targets).all(dim=-1) & flat_targets.sum(dim=-1).gt(0)
        normalized = flat_targets / flat_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        loss = -(normalized * F.log_softmax(flat_inputs, dim=-1)).sum(dim=-1).masked_fill(~valid, 0.0)
        if self.reduction == "mean":
            return loss[valid].mean() if torch.any(valid) else loss.sum() * 0.0
        if self.reduction == "sum":
            return loss[valid].sum()
        return loss


@LOSSES.register("label_smoothing_ce")
class LabelSmoothingCrossEntropyLoss(SoftTargetCrossEntropyLoss):
    def __init__(self, smoothing: float = 0.05, **kwargs: object) -> None:
        super().__init__(label_smoothing=float(smoothing), **kwargs)


__all__ = [
    "BeamNeighborhoodCrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingCrossEntropyLoss",
    "SoftTargetCrossEntropyLoss",
]
