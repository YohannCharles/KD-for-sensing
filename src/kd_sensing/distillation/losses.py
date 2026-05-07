from __future__ import annotations

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
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction="none")
        valid = targets.ne(self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss[valid].mean() if torch.any(valid) else focal_loss.sum() * 0.0
        if self.reduction == "sum":
            return focal_loss[valid].sum()
        return focal_loss


LOSSES.register("cross_entropy")(nn.CrossEntropyLoss)
