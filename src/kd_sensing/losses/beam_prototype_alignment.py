import torch
import torch.nn as nn
import torch.nn.functional as F


TOPOLOGY_IDS = frozenset(("linear_index_v1", "cyclic_index_v1", "permuted_index_v1", "ula_dft_phase_cycle_v1"))


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
    topology_id: str | None = None,
    topology_permutation: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    labels = labels.to(dtype=torch.long).reshape(-1)
    if int(num_beams) <= 0 or float(sigma) <= 0:
        raise ValueError("num_beams and sigma must be positive.")
    if bool(((labels < 0) | (labels >= int(num_beams))).any().item()):
        raise ValueError(f"labels must be in [0, {int(num_beams) - 1}].")
    resolved_topology, use_cyclic, positions = _resolve_topology(
        int(num_beams),
        circular=bool(circular),
        topology_id=topology_id,
        topology_permutation=topology_permutation,
        device=labels.device,
    )
    del resolved_topology
    beams = positions.view(1, -1)
    target_positions = positions[labels].to(dtype=torch.float32).view(-1, 1)
    distance = (beams - target_positions).abs()
    if use_cyclic:
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
    topology_id: str | None = None,
    topology_permutation: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    lambda_proto: float = 1.0,
    lambda_modality_proto: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = labels.to(device=fused_features.device, dtype=torch.long)
    if hard.ndim > 1:
        hard = hard[:, 0]
    resolved_topology, _, _ = _resolve_topology(
        prototype_bank.num_beams,
        circular=bool(circular),
        topology_id=topology_id,
        topology_permutation=topology_permutation,
        device=fused_features.device,
    )
    target = make_soft_beam_labels(
        hard,
        prototype_bank.num_beams,
        beam_label_sigma,
        circular=circular,
        topology_id=resolved_topology,
        topology_permutation=topology_permutation,
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
        "prototype/topology_is_cyclic": float(resolved_topology in {"cyclic_index_v1", "permuted_index_v1", "ula_dft_phase_cycle_v1"}),
        "prototype/topology_is_permuted": float(resolved_topology == "permuted_index_v1"),
    }


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def beam_topology_positions(
    num_beams: int,
    *,
    topology_id: str,
    topology_permutation: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return each beam label's position in the declared codebook topology."""
    count = int(num_beams)
    topology = str(topology_id).strip().lower()
    if count <= 0:
        raise ValueError("num_beams must be positive.")
    if topology not in TOPOLOGY_IDS:
        raise ValueError(f"Unsupported beam prototype topology {topology!r}.")
    positions = torch.arange(count, device=device, dtype=torch.float32)
    if topology == "permuted_index_v1":
        if topology_permutation is None:
            raise ValueError("permuted_index_v1 requires a topology permutation.")
        permutation = torch.as_tensor(topology_permutation, device=device, dtype=torch.long).reshape(-1)
        if int(permutation.numel()) != count:
            raise ValueError(f"Topology permutation must contain {count} labels.")
        if torch.any(permutation < 0) or torch.any(permutation >= count) or int(torch.unique(permutation).numel()) != count:
            raise ValueError("Topology permutation must be a bijection over beam labels.")
        return permutation.to(dtype=torch.float32)
    if topology_permutation is not None:
        raise ValueError(f"Topology {topology!r} does not accept a permutation.")
    return positions


def _resolve_topology(
    num_beams: int,
    *,
    circular: bool,
    topology_id: str | None,
    topology_permutation: torch.Tensor | list[int] | tuple[int, ...] | None,
    device: torch.device,
) -> tuple[str, bool, torch.Tensor]:
    topology = str(topology_id or ("cyclic_index_v1" if circular else "linear_index_v1")).strip().lower()
    if topology not in TOPOLOGY_IDS:
        raise ValueError(f"Unsupported beam prototype topology {topology!r}.")
    expected_circular = topology != "linear_index_v1"
    if bool(circular) != expected_circular:
        raise ValueError(
            f"prototype_target_circular={circular} conflicts with topology_id={topology!r}."
        )
    positions = beam_topology_positions(
        num_beams,
        topology_id=topology,
        topology_permutation=topology_permutation,
        device=device,
    )
    return topology, expected_circular, positions


__all__ = [
    "BeamPrototypeBank",
    "TOPOLOGY_IDS",
    "beam_topology_positions",
    "make_soft_beam_labels",
    "prototype_alignment_loss",
]
