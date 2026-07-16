import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import LOSSES


@LOSSES.register("focal_loss")
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100, **_: object) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits, labels, ignore_index=self.ignore_index, reduction="none")
        valid = labels.ne(self.ignore_index)
        if not bool(valid.any().item()):
            return loss.sum() * 0.0
        return (self.alpha * (1.0 - torch.exp(-loss)) ** self.gamma * loss)[valid].mean()


@LOSSES.register("cross_entropy")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            weight=kwargs.get("weight"),
            ignore_index=int(kwargs.get("ignore_index", -100)),
            reduction=str(kwargs.get("reduction", "mean")),
            label_smoothing=float(kwargs.get("label_smoothing", 0.0)),
        )


__all__ = ["CrossEntropyLoss", "FocalLoss"]
