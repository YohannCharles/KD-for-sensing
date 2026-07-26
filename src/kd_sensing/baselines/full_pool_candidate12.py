"""Local Full-pool BTPR-Mix and PAMR research components."""

from __future__ import annotations

import csv
import heapq
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import (
    BeamPrototypeBank,
    make_soft_beam_labels,
    prototype_alignment_loss,
)
from kd_sensing.registries import ENCODERS, import_default_components


MODALITIES = ("image", "lidar", "radar", "gps")
METHODS = (
    "a0_prototype_baseline",
    "a1_kl_data_remixing",
    "a2_prototype_risk_assignment",
    "a3_btpr_mix",
    "a4_prototype_anchored_motion",
    "a5_btpr_mix_motion",
)
REMIX_METHODS = frozenset((METHODS[1], METHODS[2], METHODS[3], METHODS[5]))
RISK_METHODS = frozenset((METHODS[2], METHODS[3], METHODS[5]))
MOTION_METHODS = frozenset((METHODS[4], METHODS[5]))


def pamr_candidate_gate(criteria: Mapping[str, bool]) -> bool:
    """Apply the count gate plus the protocol's sample-specific-motion hard stop."""

    required = (
        "full_top1_plus_0_5pp",
        "dynamic_beats_mean_0_3pp",
        "dynamic_beats_shuffle_0_3pp",
    )
    return all(bool(criteria[name]) for name in required) and sum(bool(value) for value in criteria.values()) >= 5


class Candidate12Model(nn.Module):
    """U0 encoders with fixed MLP fusion and the current shared prototype bank."""

    def __init__(self, *, d_model: int = 64, seq_len: int = 5, dropout: float = 0.1, motion_radius: int = 3) -> None:
        super().__init__()
        import_default_components()
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.motion_radius = int(motion_radius)
        configs = {
            "image": {"type": "tinyvit_5m_scratch_rgb", "output_dim": self.d_model, "pretrained": False, "freeze_backbone": False},
            "lidar": {"type": "tinyvit_5m_scratch_rgb", "output_dim": self.d_model, "pretrained": False, "freeze_backbone": False},
            "radar": {"type": "radar_cnn", "output_dim": self.d_model, "pretrained": False, "freeze_backbone": False},
            "gps": {"type": "gps_mlp", "output_dim": self.d_model, "hidden_size": self.d_model, "dropout": float(dropout), "pretrained": False, "freeze_backbone": False},
        }
        self.encoders = nn.ModuleDict({name: ENCODERS.build(configs[name]) for name in MODALITIES})
        self.projections = nn.ModuleDict({name: nn.Identity() for name in MODALITIES})
        self.modality_embedding = nn.Parameter(torch.randn(4, self.d_model) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(self.seq_len, self.d_model) * 0.02)
        fusion_input = self.seq_len * len(MODALITIES) * self.d_model
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, 1024),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(512, self.d_model),
        )
        self.prototype_bank = BeamPrototypeBank(self.d_model, 64, temperature=0.1)
        motion_input = 4 * len(MODALITIES) * self.d_model
        self.motion = nn.Sequential(
            nn.LayerNorm(motion_input),
            nn.Linear(motion_input, 512),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2 * self.motion_radius + 1),
        )

    def encode(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens: dict[str, torch.Tensor] = {}
        for name in MODALITIES:
            if name not in inputs:
                raise ValueError(f"Candidate12 is missing {name} input.")
            token = self.projections[name](self.encoders[name](inputs[name]))
            if tuple(token.shape[1:]) != (self.seq_len, self.d_model):
                raise ValueError(f"Candidate12 {name} tokens must be [B,{self.seq_len},{self.d_model}], got {tuple(token.shape)}.")
            tokens[name] = token
        return tokens

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        availability: torch.Tensor | None = None,
        signed_order: Sequence[int] | None = None,
        apply_motion: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.forward_tokens(
            self.encode(inputs),
            availability=availability,
            signed_order=signed_order,
            apply_motion=apply_motion,
        )

    def forward_tokens(
        self,
        tokens: Mapping[str, torch.Tensor],
        *,
        availability: torch.Tensor | None = None,
        signed_order: Sequence[int] | None = None,
        apply_motion: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch = next(iter(tokens.values())).shape[0]
        if availability is None:
            availability = torch.ones(batch, 4, dtype=torch.bool, device=next(iter(tokens.values())).device)
        availability = torch.as_tensor(availability, device=next(iter(tokens.values())).device, dtype=torch.bool)
        if tuple(availability.shape) != (batch, 4) or not bool(availability.any(dim=1).all()):
            raise ValueError("Candidate12 availability must be non-empty [B,4].")
        stacked = torch.stack([tokens[name] for name in MODALITIES], dim=2)
        positioned = stacked + self.time_embedding[None, :, None, :] + self.modality_embedding[None, None, :, :]
        masked = positioned * availability[:, None, :, None].to(positioned)
        fused = self.fusion(masked.flatten(1))
        modality_features = torch.stack([tokens[name].mean(dim=1) for name in MODALITIES], dim=1)
        unimodal_logits = self.prototype_bank(modality_features.flatten(0, 1)).view(batch, 4, 64)
        anchor_logits = self.prototype_bank(fused)
        output = {
            "tokens": stacked,
            "modality_features": modality_features,
            "fused_features": fused,
            "unimodal_logits": unimodal_logits,
            "anchor_logits": anchor_logits,
            "availability": availability,
        }
        if apply_motion:
            if signed_order is None:
                raise ValueError("Candidate12 motion requires an audited signed beam order.")
            shift_logits = self.motion(motion_features(stacked, availability))
            final_probability = motion_mixture(
                torch.softmax(anchor_logits, dim=-1), shift_logits, signed_order, radius=self.motion_radius
            )
            output.update(
                shift_logits=shift_logits,
                final_probability=final_probability,
                final_logits=final_probability.clamp_min(1e-8).log(),
            )
        return output


def motion_features(tokens: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4 or tokens.shape[2] != 4:
        raise ValueError("motion tokens must be [B,T,4,D].")
    last = tokens[:, -1]
    mean = tokens.mean(dim=1)
    mean_delta = (tokens[:, 1:] - tokens[:, :-1]).mean(dim=1)
    long_delta = tokens[:, -1] - tokens[:, 0]
    values = torch.cat((last, mean, mean_delta, long_delta), dim=-1)
    values = values * availability[:, :, None].to(values)
    return values.flatten(1)


def load_signed_angle_order(topology_table: str | Path) -> tuple[int, ...]:
    with Path(topology_table).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise ValueError("PAMR signed order requires exactly 64 topology rows.")
    try:
        ordered = sorted(
            ((int(row["label"]), float(row["principal_local_angle_deg"])) for row in rows),
            key=lambda item: item[1],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PAMR topology table lacks valid principal_local_angle_deg values.") from exc
    labels, angles = zip(*ordered)
    if set(labels) != set(range(64)) or not all(left < right for left, right in zip(angles, angles[1:])):
        raise ValueError("PAMR signed angle order must be a strict 64-label bijection.")
    return tuple(labels)


def noncircular_shift(probability: torch.Tensor, delta: int, signed_order: Sequence[int]) -> torch.Tensor:
    labels = torch.as_tensor(signed_order, device=probability.device, dtype=torch.long)
    if probability.ndim != 2 or probability.shape[1] != labels.numel() or torch.unique(labels).numel() != labels.numel():
        raise ValueError("noncircular shift requires [B,64] probabilities and a label bijection.")
    amount = int(delta)
    if abs(amount) >= labels.numel():
        raise ValueError("shift magnitude exceeds the beam order.")
    if amount == 0:
        return probability.clone()
    ordered = probability[:, labels]
    shifted = torch.zeros_like(ordered)
    if amount > 0:
        shifted[:, amount:] = ordered[:, :-amount]
    elif amount < 0:
        shifted[:, :amount] = ordered[:, -amount:]
    mass = shifted.sum(dim=-1, keepdim=True)
    shifted = torch.where(mass > 1e-12, shifted / mass.clamp_min(1e-12), ordered)
    restored = torch.empty_like(shifted)
    restored[:, labels] = shifted
    return restored


def motion_mixture(
    anchor_probability: torch.Tensor,
    shift_logits: torch.Tensor,
    signed_order: Sequence[int],
    *,
    radius: int = 3,
) -> torch.Tensor:
    if tuple(shift_logits.shape) != (anchor_probability.shape[0], 2 * int(radius) + 1):
        raise ValueError("shift logits shape does not match the configured motion radius.")
    candidates = torch.stack(
        [noncircular_shift(anchor_probability, delta, signed_order) for delta in range(-int(radius), int(radius) + 1)],
        dim=1,
    )
    result = (torch.softmax(shift_logits, dim=-1)[:, :, None] * candidates).sum(dim=1)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def signed_offset_targets(
    anchor_probability: torch.Tensor,
    labels: torch.Tensor,
    signed_order: Sequence[int],
    *,
    radius: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position = torch.empty(64, dtype=torch.long, device=anchor_probability.device)
    position[torch.as_tensor(signed_order, device=anchor_probability.device, dtype=torch.long)] = torch.arange(64, device=anchor_probability.device)
    raw = position[labels] - position[anchor_probability.detach().argmax(dim=-1)]
    valid = raw.abs().le(int(radius))
    target = raw + int(radius)
    return target, valid, raw


def common_loss(
    model: Candidate12Model,
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(output["anchor_logits"], labels)
    prototype, diagnostics = prototype_alignment_loss(
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
    return ce + prototype, {"ce_anchor": float(ce.detach()), **diagnostics}


def remix_loss(
    model: Candidate12Model,
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    assigned: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = torch.arange(labels.shape[0], device=labels.device)
    features = output["modality_features"][rows, assigned]
    logits = (
        F.normalize(features, dim=-1)
        @ F.normalize(model.prototype_bank.prototypes.detach(), dim=-1).t()
        / model.prototype_bank.temperature
    )
    ce = F.cross_entropy(logits, labels)
    target = make_soft_beam_labels(
        labels, 64, 2.0, circular=True, topology_id="ula_dft_phase_cycle_v1"
    ).to(logits)
    alignment = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return ce + 0.1 * alignment, {"ce_remix": float(ce.detach()), "prototype_remix": float(alignment.detach())}


def motion_loss(
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    signed_order: Sequence[int],
    *,
    radius: int = 3,
) -> tuple[torch.Tensor, dict[str, float]]:
    final = F.nll_loss(output["final_logits"], labels)
    target, valid, _ = signed_offset_targets(output["anchor_logits"].softmax(-1), labels, signed_order, radius=radius)
    offset = F.cross_entropy(output["shift_logits"][valid], target[valid]) if bool(valid.any()) else final * 0.0
    return final + 0.5 * offset, {
        "final_nll": float(final.detach()),
        "offset_ce": float(offset.detach()),
        "offset_valid_ratio": float(valid.float().mean()),
    }


def percentile_ranks(values: np.ndarray, sample_ids: Sequence[str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(sample_ids):
        raise ValueError("percentile ranks require [N,M] values and N sample ids.")
    ranks = np.empty_like(array)
    tie = np.asarray([str(value) for value in sample_ids])
    denominator = max(array.shape[0] - 1, 1)
    for modality in range(array.shape[1]):
        order = np.lexsort((tie, array[:, modality]))
        ranks[order, modality] = np.arange(array.shape[0], dtype=np.float64) / denominator
    return ranks


def capacity_constrained_assignment(
    scores: np.ndarray,
    sample_ids: Sequence[str],
    *,
    minimum: float = 0.15,
    maximum: float = 0.40,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(sample_ids), 4) or not np.isfinite(values).all():
        raise ValueError("capacity assignment requires finite [N,4] scores.")
    count = values.shape[0]
    lower, upper = int(np.ceil(count * float(minimum))), int(np.floor(count * float(maximum)))
    if 4 * lower > count or 4 * upper < count:
        raise ValueError("assignment capacity bounds are infeasible.")
    ranking = np.argsort(-values, axis=1, kind="stable")
    assigned = ranking[:, 0].copy()
    ids = np.asarray([str(value) for value in sample_ids])

    counts = np.bincount(assigned, minlength=4)
    heap: list[tuple[float, str, int, int, int]] = []
    for row in range(count):
        donor = int(assigned[row])
        if counts[donor] > upper:
            for target in ranking[row, 1:]:
                target = int(target)
                heapq.heappush(
                    heap,
                    (values[row, donor] - values[row, target], ids[row], row, donor, target),
                )
    while bool(np.any(counts > upper)):
        if not heap:
            raise RuntimeError("capacity maximum adjustment has no feasible transfer.")
        _, _, row, donor, target = heapq.heappop(heap)
        if int(assigned[row]) != donor or counts[donor] <= upper or counts[target] >= upper:
            continue
        assigned[row] = target
        counts[donor] -= 1
        counts[target] += 1

    while int(counts.min()) < lower:
        target = int(np.flatnonzero(counts < lower)[0])
        candidates = [
            (values[row, int(assigned[row])] - values[row, target], ids[row], int(row))
            for row in range(count)
            if assigned[row] != target
        ]
        heapq.heapify(candidates)
        while counts[target] < lower:
            if not candidates:
                raise RuntimeError("capacity minimum adjustment has no feasible transfer.")
            _, _, row = heapq.heappop(candidates)
            donor = int(assigned[row])
            if donor == target or counts[donor] <= lower:
                continue
            assigned[row] = target
            counts[donor] -= 1
            counts[target] += 1

    if np.any(counts < lower) or np.any(counts > upper):
        raise AssertionError("capacity assignment failed its final bounds check.")
    return assigned


def assignment_diagnostics(
    unimodal_logits: np.ndarray,
    modality_features: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    topology_distance: np.ndarray,
    sample_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    logits = np.asarray(unimodal_logits, dtype=np.float64)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    risk = np.einsum("nmk,nk->nm", probabilities, topology_distance[np.asarray(labels, dtype=np.int64)])
    normalized_features = modality_features / np.clip(np.linalg.norm(modality_features, axis=-1, keepdims=True), 1e-12, None)
    normalized_prototypes = prototypes / np.clip(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-12, None)
    cosine = np.einsum("nmd,kd->nmk", normalized_features, normalized_prototypes)
    rows = np.arange(logits.shape[0])[:, None]
    modalities = np.arange(4)[None, :]
    true = cosine[rows, modalities, np.asarray(labels)[:, None]]
    cosine[rows, modalities, np.asarray(labels)[:, None]] = -np.inf
    margin = true - cosine.max(axis=-1)
    hardness = np.maximum(-margin, 0.0)
    risk_rank = percentile_ranks(risk, sample_ids)
    hardness_rank = percentile_ranks(hardness, sample_ids)
    combined = 0.5 * risk_rank + 0.5 * hardness_rank
    kl_uniform = np.sum(probabilities * np.log(np.clip(probabilities * 64.0, 1e-12, None)), axis=-1)
    return {
        "kl_uniform": kl_uniform,
        "risk": risk,
        "margin": margin,
        "margin_hardness": hardness,
        "risk_rank": risk_rank,
        "margin_rank": hardness_rank,
        "combined_hardness": combined,
    }


__all__ = [
    "Candidate12Model",
    "METHODS",
    "MODALITIES",
    "MOTION_METHODS",
    "REMIX_METHODS",
    "RISK_METHODS",
    "assignment_diagnostics",
    "capacity_constrained_assignment",
    "common_loss",
    "load_signed_angle_order",
    "motion_loss",
    "motion_mixture",
    "noncircular_shift",
    "percentile_ranks",
    "remix_loss",
    "signed_offset_targets",
]
