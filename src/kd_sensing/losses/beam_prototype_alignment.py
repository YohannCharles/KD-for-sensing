import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamPrototypeBank(nn.Module):
    def __init__(self, d_model: int, num_beams: int, *, temperature: float = 0.2) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_beams = int(num_beams)
        self.temperature = float(temperature)
        if min(self.d_model, self.num_beams) <= 0 or self.temperature <= 0:
            raise ValueError("d_model, num_beams, and temperature must be positive.")
        self.prototypes = nn.Parameter(torch.randn(self.num_beams, self.d_model) * 0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[-1]) != self.d_model:
            raise ValueError(f"features must have shape [B, {self.d_model}], got {tuple(features.shape)}.")
        return F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).t() / self.temperature


def make_soft_beam_labels(
    labels: torch.Tensor,
    num_beams: int,
    sigma: float,
    *,
    circular: bool = True,
) -> torch.Tensor:
    labels = labels.to(dtype=torch.long).reshape(-1)
    if int(num_beams) <= 0 or float(sigma) <= 0:
        raise ValueError("num_beams and sigma must be positive.")
    if bool(((labels < 0) | (labels >= int(num_beams))).any().item()):
        raise ValueError(f"labels must be in [0, {int(num_beams) - 1}].")
    beams = torch.arange(int(num_beams), device=labels.device, dtype=torch.float32).view(1, -1)
    distance = (beams - labels.to(dtype=torch.float32).view(-1, 1)).abs()
    if circular:
        distance = torch.minimum(distance, float(num_beams) - distance)
    weights = torch.exp(-0.5 * (distance / float(sigma)).pow(2))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def prototype_alignment_loss(
    prototype_bank: BeamPrototypeBank,
    labels: torch.Tensor,
    *,
    fused_features: torch.Tensor,
    modality_features: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    beam_label_sigma: float = 1.0,
    circular: bool = True,
    lambda_proto: float = 1.0,
    lambda_modality_proto: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = labels.to(device=fused_features.device, dtype=torch.long)
    if hard.ndim > 1:
        hard = hard[:, 0]
    target = make_soft_beam_labels(
        hard,
        prototype_bank.num_beams,
        beam_label_sigma,
        circular=circular,
    ).to(dtype=fused_features.dtype)
    fused_loss = _soft_ce(prototype_bank(fused_features), target)
    zero = fused_loss * 0.0
    modality_loss = zero
    modality_count = 0
    if modality_features is not None and mask is not None:
        available = mask.to(device=modality_features.device, dtype=torch.bool)
        if modality_features.ndim != 3 or tuple(available.shape) != tuple(modality_features.shape[:2]):
            raise ValueError("modality_features and mask must have shapes [B, M, D] and [B, M].")
        modality_count = int(available.sum().detach().cpu().item())
        if modality_count:
            targets = target.unsqueeze(1).expand(-1, modality_features.shape[1], -1)[available]
            modality_loss = _soft_ce(prototype_bank(modality_features[available]), targets)
    total = float(lambda_proto) * fused_loss + float(lambda_modality_proto) * modality_loss
    return total, {
        "loss/prototype_alignment": float(fused_loss.detach().cpu().item()),
        "loss/prototype_modality": float(modality_loss.detach().cpu().item()),
        "loss/prototype_total": float(total.detach().cpu().item()),
        "prototype/sample_count": float(hard.numel()),
        "prototype/modality_sample_count": float(modality_count),
    }


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


__all__ = ["BeamPrototypeBank", "make_soft_beam_labels", "prototype_alignment_loss"]
