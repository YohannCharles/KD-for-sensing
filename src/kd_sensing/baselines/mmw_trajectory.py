"""Fixed Candidate12-head methods for the trajectory protocol."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.baselines.btma_assignment import fixed_proportion_assignment
from kd_sensing.baselines.full_pool_candidate12 import Candidate12Model, MODALITIES
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank, prototype_alignment_loss


M4_UNIFORM_METHOD = "m4a_uniform_all_masks"
M4_BALANCED_METHOD = "m4b_availability_balanced"
M4_GENERIC_KL_METHOD = "m4c_availability_balanced_generic_kl"
ABTC_METHOD = "m4_availability_balanced_topology_consistency"
METHODS = (
    "m0_plain_linear",
    "m1_ordinary_prototype",
    "m2_topology_prototype",
    "m3_topology_prototype_random_balanced",
    M4_UNIFORM_METHOD,
    M4_BALANCED_METHOD,
    M4_GENERIC_KL_METHOD,
    ABTC_METHOD,
)
TOPOLOGY_METHODS = frozenset(METHODS[2:])
RANDOM_BALANCED_METHOD = METHODS[3]
PAIRED_MISSING_METHODS = frozenset((M4_UNIFORM_METHOD, M4_BALANCED_METHOD, M4_GENERIC_KL_METHOD, ABTC_METHOD))
ABTC_CONSISTENCY_WEIGHT = 0.2
ABTC_TEMPERATURE = 2.0
ABTC_TOPOLOGY_SIGMA = 2.0


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

    def forward_paired(
        self,
        inputs: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        tokens = self.encode(inputs)
        return self.forward_tokens(tokens), self.forward_tokens(tokens, availability=availability)


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


def availability_balanced_assignment(
    sample_ids: Sequence[str],
    *,
    epoch: int,
    seed: int = 2026,
) -> dict[str, tuple[int, ...]]:
    """Assign every train sample to one of 14 masks with balanced availability levels."""
    values = [str(value) for value in sample_ids]
    if len(values) != len(set(values)):
        raise ValueError("ABTC assignments require unique train sample ids.")
    order = sorted(
        values,
        key=lambda value: (hashlib.sha256(f"abtc-v1:{seed}:{epoch}:{value}".encode()).digest(), value),
    )
    masks = {
        count: tuple(
            tuple(int(index in available) for index in range(len(MODALITIES)))
            for available in combinations(range(len(MODALITIES)), count)
        )
        for count in (1, 2, 3)
    }
    quotas = [len(values) // 3 + int(index < len(values) % 3) for index in range(3)]
    result: dict[str, tuple[int, ...]] = {}
    start = 0
    for available_count, quota in zip((1, 2, 3), quotas):
        level_masks = masks[available_count]
        for offset, sample_id in enumerate(order[start : start + quota]):
            result[sample_id] = level_masks[offset % len(level_masks)]
        start += quota
    if len(result) != len(values):
        raise AssertionError("ABTC availability assignment left samples unassigned.")
    return result


def uniform_mask_assignment(
    sample_ids: Sequence[str],
    *,
    epoch: int,
    seed: int = 2026,
) -> dict[str, tuple[int, ...]]:
    """Assign train samples uniformly across all 14 non-full masks."""
    values = [str(value) for value in sample_ids]
    if len(values) != len(set(values)):
        raise ValueError("Uniform mask assignments require unique train sample ids.")
    order = sorted(
        values,
        key=lambda value: (hashlib.sha256(f"m4a-v1:{seed}:{epoch}:{value}".encode()).digest(), value),
    )
    masks = tuple(
        tuple(int(index in available) for index in range(len(MODALITIES)))
        for count in (1, 2, 3)
        for available in combinations(range(len(MODALITIES)), count)
    )
    return {sample_id: masks[index % len(masks)] for index, sample_id in enumerate(order)}


def topology_smoothed_consistency_loss(
    masked_logits: torch.Tensor,
    full_logits: torch.Tensor,
    topology_distance: torch.Tensor,
    *,
    temperature: float = ABTC_TEMPERATURE,
    sigma: float = ABTC_TOPOLOGY_SIGMA,
) -> torch.Tensor:
    """Match masked and detached-full distributions after topology-aware smoothing."""
    if masked_logits.ndim != 2 or full_logits.shape != masked_logits.shape:
        raise ValueError("ABTC logits must have matching [B,C] shapes.")
    if topology_distance.shape != (masked_logits.shape[-1], masked_logits.shape[-1]):
        raise ValueError("ABTC topology distance must have shape [C,C].")
    if min(float(temperature), float(sigma)) <= 0:
        raise ValueError("ABTC temperature and topology sigma must be positive.")
    distance = topology_distance.to(device=masked_logits.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (distance / float(sigma)).square())
    kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(kernel.dtype).tiny)
    scale = float(temperature)
    teacher = torch.softmax(full_logits.detach().float() / scale, dim=-1) @ kernel
    student = torch.softmax(masked_logits.float() / scale, dim=-1) @ kernel
    return F.kl_div(student.clamp_min(1e-12).log(), teacher, reduction="batchmean") * scale**2


def generic_consistency_loss(
    masked_logits: torch.Tensor,
    full_logits: torch.Tensor,
    *,
    temperature: float = ABTC_TEMPERATURE,
) -> torch.Tensor:
    if masked_logits.ndim != 2 or full_logits.shape != masked_logits.shape:
        raise ValueError("Paired consistency logits must have matching [B,C] shapes.")
    if float(temperature) <= 0:
        raise ValueError("Paired consistency temperature must be positive.")
    scale = float(temperature)
    return F.kl_div(
        F.log_softmax(masked_logits.float() / scale, dim=-1),
        F.softmax(full_logits.detach().float() / scale, dim=-1),
        reduction="batchmean",
    ) * scale**2


def paired_missing_loss(
    model: TrajectoryBaselineModel,
    full_output: Mapping[str, torch.Tensor],
    masked_output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    topology_distance: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if model.method not in PAIRED_MISSING_METHODS:
        raise ValueError("Paired missing loss requires an M4-family trajectory method.")
    full_loss, full_report = baseline_loss(model, full_output, labels)
    masked_loss, masked_report = baseline_loss(model, masked_output, labels)
    if model.method == M4_GENERIC_KL_METHOD:
        consistency = generic_consistency_loss(masked_output["logits"], full_output["logits"])
    elif model.method == ABTC_METHOD:
        consistency = topology_smoothed_consistency_loss(
            masked_output["logits"],
            full_output["logits"],
            topology_distance,
        )
    else:
        consistency = masked_output["logits"].sum() * 0.0
    total = 0.5 * (full_loss + masked_loss) + ABTC_CONSISTENCY_WEIGHT * consistency
    return total, {
        "ce": 0.5 * (full_report["ce"] + masked_report["ce"]),
        "topology_alignment": 0.5
        * (full_report["topology_alignment"] + masked_report["topology_alignment"]),
        "topology_consistency": float(consistency.detach()),
        "full_ce": full_report["ce"],
        "masked_ce": masked_report["ce"],
    }


def abtc_loss(
    model: TrajectoryBaselineModel,
    full_output: Mapping[str, torch.Tensor],
    masked_output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    topology_distance: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if model.method != ABTC_METHOD:
        raise ValueError("ABTC loss requires the M4 trajectory method.")
    return paired_missing_loss(model, full_output, masked_output, labels, topology_distance)


def model_contract(model: TrajectoryBaselineModel) -> dict[str, Any]:
    return {
        "method": model.method,
        "modalities": list(MODALITIES),
        "history_window": model.seq_len,
        "num_classes": 64,
        "head": "linear" if model.linear_head is not None else "prototype",
        "topology_alignment": model.method in TOPOLOGY_METHODS,
        "random_balanced_single_modality": model.method == RANDOM_BALANCED_METHOD,
        "availability_balanced_topology_consistency": model.method == ABTC_METHOD,
        "paired_missing_training": model.method in PAIRED_MISSING_METHODS,
        "motion_branch_present": hasattr(model, "motion"),
        "channel_input_present": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


__all__ = [
    "ABTC_CONSISTENCY_WEIGHT",
    "ABTC_METHOD",
    "ABTC_TEMPERATURE",
    "ABTC_TOPOLOGY_SIGMA",
    "M4_BALANCED_METHOD",
    "M4_GENERIC_KL_METHOD",
    "M4_UNIFORM_METHOD",
    "METHODS",
    "PAIRED_MISSING_METHODS",
    "RANDOM_BALANCED_METHOD",
    "TOPOLOGY_METHODS",
    "TrajectoryBaselineModel",
    "abtc_loss",
    "availability_balanced_assignment",
    "availability_for_assignments",
    "baseline_loss",
    "model_contract",
    "paired_missing_loss",
    "random_balanced_assignment",
    "topology_smoothed_consistency_loss",
    "uniform_mask_assignment",
]
