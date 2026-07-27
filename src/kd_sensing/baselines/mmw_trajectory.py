"""Four fixed Candidate12-head baselines for the trajectory protocol."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.baselines.btma_assignment import fixed_proportion_assignment
from kd_sensing.baselines.full_pool_candidate12 import Candidate12Model, MODALITIES
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank, prototype_alignment_loss


METHODS = (
    "m0_plain_linear",
    "m1_ordinary_prototype",
    "m2_topology_prototype",
    "m3_topology_prototype_random_balanced",
)
TOPOLOGY_METHODS = frozenset(METHODS[2:])
RANDOM_BALANCED_METHOD = METHODS[3]


class TrajectoryBaselineModel(Candidate12Model):
    """Candidate12 encoders/fusion with only the requested classification head."""

    def __init__(self, method: str, *, d_model: int = 64, seq_len: int = 5, dropout: float = 0.1) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown trajectory baseline method: {method}")
        super().__init__(d_model=d_model, seq_len=seq_len, dropout=dropout)
        self.method = method
        del self.motion
        del self.prototype_bank
        if method == METHODS[0]:
            self.linear_head = nn.Linear(self.d_model, 64)
            self.prototype_bank = None
        else:
            self.linear_head = None
            self.prototype_bank = BeamPrototypeBank(self.d_model, 64, temperature=0.1)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        availability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_tokens(self.encode(inputs), availability=availability)

    def forward_tokens(
        self,
        tokens: Mapping[str, torch.Tensor],
        *,
        availability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        first = next(iter(tokens.values()))
        batch = first.shape[0]
        if availability is None:
            availability = torch.ones(batch, 4, dtype=torch.bool, device=first.device)
        availability = torch.as_tensor(availability, device=first.device, dtype=torch.bool)
        if tuple(availability.shape) != (batch, 4) or not bool(availability.any(dim=1).all()):
            raise ValueError("Trajectory baseline availability must be non-empty [B,4].")
        stacked = torch.stack([tokens[name] for name in MODALITIES], dim=2)
        positioned = stacked + self.time_embedding[None, :, None, :] + self.modality_embedding[None, None, :, :]
        fused = self.fusion((positioned * availability[:, None, :, None].to(positioned)).flatten(1))
        modality_features = torch.stack([tokens[name].mean(dim=1) for name in MODALITIES], dim=1)
        if self.linear_head is not None:
            logits = self.linear_head(fused)
            unimodal_logits = self.linear_head(modality_features.flatten(0, 1)).view(batch, 4, 64)
        else:
            assert self.prototype_bank is not None
            logits = self.prototype_bank(fused)
            unimodal_logits = self.prototype_bank(modality_features.flatten(0, 1)).view(batch, 4, 64)
        return {
            "tokens": stacked,
            "modality_features": modality_features,
            "fused_features": fused,
            "unimodal_logits": unimodal_logits,
            "logits": logits,
            "availability": availability,
        }


def baseline_loss(
    model: TrajectoryBaselineModel,
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(output["logits"], labels)
    if model.method not in TOPOLOGY_METHODS:
        return ce, {"ce": float(ce.detach()), "topology_alignment": 0.0}
    assert model.prototype_bank is not None
    topology, diagnostics = prototype_alignment_loss(
        model.prototype_bank,
        labels,
        fused_features=output["fused_features"],
        modality_features=output["modality_features"],
        mask=output["availability"],
        beam_label_sigma=2.0,
        circular=True,
        topology_id="ula_dft_phase_cycle_v1",
        lambda_proto=0.2,
        lambda_modality_proto=0.1,
    )
    return ce + topology, {
        "ce": float(ce.detach()),
        "topology_alignment": float(topology.detach()),
        **diagnostics,
    }


def random_balanced_assignment(sample_ids: Sequence[str]) -> dict[str, int]:
    values = [str(value) for value in sample_ids]
    assigned = fixed_proportion_assignment(values, {name: 0.25 for name in MODALITIES})
    return dict(zip(values, (int(value) for value in assigned)))


def availability_for_assignments(assignments: Sequence[int], device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(assignments, dtype=torch.long, device=device)
    if values.ndim != 1 or not bool(((values >= 0) & (values < 4)).all()):
        raise ValueError("Random-balanced assignments must be one modality index per sample.")
    return F.one_hot(values, num_classes=4).to(dtype=torch.bool)


def model_contract(model: TrajectoryBaselineModel) -> dict[str, Any]:
    return {
        "method": model.method,
        "modalities": list(MODALITIES),
        "history_window": model.seq_len,
        "num_classes": 64,
        "head": "linear" if model.linear_head is not None else "prototype",
        "topology_alignment": model.method in TOPOLOGY_METHODS,
        "random_balanced_single_modality": model.method == RANDOM_BALANCED_METHOD,
        "motion_branch_present": hasattr(model, "motion"),
        "channel_input_present": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


__all__ = [
    "METHODS",
    "RANDOM_BALANCED_METHOD",
    "TOPOLOGY_METHODS",
    "TrajectoryBaselineModel",
    "availability_for_assignments",
    "baseline_loss",
    "model_contract",
    "random_balanced_assignment",
]
