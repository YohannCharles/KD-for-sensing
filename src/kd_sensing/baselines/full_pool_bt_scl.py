"""Local Full-pool Beam-Topology-Aware Subset Consistency Learning workflow.

This module intentionally does not register a model or extend a public CLI.  It
owns the experimental BT-SCL architecture, topology binding, nested subset
schedule and losses so the retained U0/Adapter surfaces remain untouched.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.baselines import full_pool_common as _common


MODALITIES = ("image", "lidar", "radar", "gps")
METHODS = (
    "r0_subset_task_only",
    "r1_available_evidence",
    "r2_topology_monotonicity",
    "r3_coarse_to_fine",
    "r4_mono_c2f",
    "r5_full_bt_scl",
    "r6_topological_stochastic_dominance",
)
PATTERNS = {
    "full": (1, 1, 1, 1),
    "missing_image": (0, 1, 1, 1),
    "missing_lidar": (1, 0, 1, 1),
    "missing_radar": (1, 1, 0, 1),
    "missing_gps": (1, 1, 1, 0),
    "missing_image_lidar": (0, 0, 1, 1),
    "missing_image_radar": (0, 1, 0, 1),
    "missing_image_gps": (0, 1, 1, 0),
    "missing_lidar_radar": (1, 0, 0, 1),
    "missing_lidar_gps": (1, 0, 1, 0),
    "missing_radar_gps": (1, 1, 0, 0),
    "only_image": (1, 0, 0, 0),
    "only_lidar": (0, 1, 0, 0),
    "only_radar": (0, 0, 1, 0),
    "only_gps": (0, 0, 0, 1),
}


# Re-exported so the Candidate12 and BTMA workflows keep importing these from
# here.  The implementations now live in `full_pool_common` and are shared with
# the rest of the local experiment surface; BT-SCL artifacts stay in insertion
# order, so `write_json` intentionally keeps `sort_keys=False`.
sha256_json = _common.sha256_json
sha256_file = _common.sha256_file
write_json = _common.write_json


@dataclass(frozen=True)
class BeamTopology:
    manifest_path: str
    manifest_sha256: str
    descriptor_sha256: str
    labels_by_position: tuple[int, ...]
    distance: torch.Tensor

    @property
    def num_beams(self) -> int:
        return len(self.labels_by_position)

    def sector_labels(self, count: int) -> tuple[tuple[int, ...], ...]:
        if count <= 0 or self.num_beams % int(count):
            raise ValueError("sector count must divide the number of beams.")
        width = self.num_beams // int(count)
        return tuple(tuple(self.labels_by_position[index : index + width]) for index in range(0, self.num_beams, width))


def load_audited_topology(path: str | Path) -> BeamTopology:
    """Bind losses to the local ULA-DFT topology evidence, not a Boolean flag."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor_payload = payload.get("descriptor")
    if not isinstance(descriptor_payload, Mapping):
        raise ValueError("BT-SCL topology manifest has no descriptor mapping.")
    if descriptor_payload.get("topology_id") != "ula_dft_phase_cycle_v1":
        raise ValueError("BT-SCL requires topology_id=ula_dft_phase_cycle_v1.")
    if int(descriptor_payload.get("num_beams", -1)) != 64 or int(descriptor_payload.get("num_antennas", -1)) != 64:
        raise ValueError("BT-SCL requires an audited 64-beam, 64-antenna ULA-DFT topology.")
    domains = payload.get("domains")
    if payload.get("metadata_consistent") is not True or not isinstance(domains, list) or int(payload.get("domain_count", -1)) != 15 or len(domains) != 15:
        raise ValueError("BT-SCL topology audit must be metadata-consistent across all 15 domains.")
    if any(item.get("metadata_status") != "verified" for item in domains if isinstance(item, Mapping)):
        raise ValueError("BT-SCL topology audit has an unverified domain.")
    descriptor = str(payload.get("descriptor_sha256", ""))
    if len(descriptor) != 64:
        raise ValueError("BT-SCL topology manifest has no descriptor SHA256.")
    table_path = manifest_path.parent / "topology_table.csv"
    edge_path = manifest_path.parent / "topology_edges.csv"
    if not table_path.is_file() or not edge_path.is_file():
        raise FileNotFoundError("BT-SCL topology table or edge audit is missing.")
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise ValueError("BT-SCL topology table must contain exactly 64 labels.")
    try:
        ordered = sorted(rows, key=lambda row: float(row["phase_coordinate"]))
        labels = tuple(int(row["label"]) for row in ordered)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BT-SCL topology table is not a valid label/phase mapping.") from exc
    if set(labels) != set(range(64)):
        raise ValueError("BT-SCL topology labels must be a bijection over [0, 63].")
    with edge_path.open(newline="", encoding="utf-8") as handle:
        edges = list(csv.DictReader(handle))
    normalized_edges = {
        frozenset((int(row["left_label"]), int(row["right_label"])))
        for row in edges
    }
    expected = {frozenset((labels[index], labels[(index + 1) % 64])) for index in range(64)}
    if normalized_edges != expected:
        raise ValueError("BT-SCL topology edges do not form the audited 64-label phase cycle.")
    positions = torch.empty(64, dtype=torch.long)
    positions[torch.tensor(labels, dtype=torch.long)] = torch.arange(64, dtype=torch.long)
    delta = (positions[:, None] - positions[None, :]).abs()
    distance = torch.minimum(delta, 64 - delta).to(dtype=torch.float32)
    return BeamTopology(
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        descriptor_sha256=descriptor,
        labels_by_position=labels,
        distance=distance,
    )


def _chain_from_permutation(permutation: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in permutation)
    if set(values) != set(range(4)):
        raise ValueError("nested subset permutation must contain each modality exactly once.")
    return tuple(tuple(sorted(values[:count])) for count in range(1, 5))


def generate_nested_schedule(sample_ids: Iterable[str], *, seed: int, split: str) -> dict[str, Any]:
    """Allocate all 24 modality permutations evenly over stable sample identities."""
    values = [str(value) for value in sample_ids]
    ids = sorted(set(values))
    if not ids or len(ids) != len(values):
        raise ValueError("nested schedule requires unique non-empty stable sample ids.")
    shuffled = list(ids)
    random.Random(int(seed)).shuffle(shuffled)
    permutations = list(itertools.permutations(range(4)))
    assignment = {sample_id: list(permutations[index % len(permutations)]) for index, sample_id in enumerate(shuffled)}
    counts = {
        "single_start": {MODALITIES[index]: 0 for index in range(4)},
        "double_subset": {},
        "triple_subset": {},
        "added_modality": {MODALITIES[index]: 0 for index in range(4)},
    }
    for permutation in assignment.values():
        chain = _chain_from_permutation(permutation)
        counts["single_start"][MODALITIES[chain[0][0]]] += 1
        counts["double_subset"]["+".join(MODALITIES[index] for index in chain[1])] = counts["double_subset"].get(
            "+".join(MODALITIES[index] for index in chain[1]), 0
        ) + 1
        counts["triple_subset"]["+".join(MODALITIES[index] for index in chain[2])] = counts["triple_subset"].get(
            "+".join(MODALITIES[index] for index in chain[2]), 0
        ) + 1
        for index in permutation:
            counts["added_modality"][MODALITIES[index]] += 1
    payload = {
        "schema_version": 1,
        "schedule_id": "bt_scl_nested_permutation_v1",
        "split": str(split),
        "seed": int(seed),
        "modalities": list(MODALITIES),
        "sample_count": len(ids),
        "assignments": assignment,
        "balance_counts": counts,
    }
    payload["schedule_sha256"] = sha256_json(payload)
    return payload


def schedule_masks(sample_ids: Iterable[str], schedule: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    assignments = schedule.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("BT-SCL nested schedule assignments are unavailable.")
    masks = []
    for sample_id in sample_ids:
        try:
            chain = _chain_from_permutation(assignments[str(sample_id)])
        except KeyError as exc:
            raise ValueError(f"BT-SCL schedule has no assignment for {sample_id!r}.") from exc
        masks.append([[int(index in subset) for index in range(4)] for subset in chain])
    return torch.tensor(masks, dtype=torch.bool, device=device)


class SpatialTokenEncoder(nn.Module):
    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(96, int(output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"spatial input must be [B,T,C,H,W], got {tuple(value.shape)}.")
        batch, steps = value.shape[:2]
        return self.output(self.net(value.reshape(batch * steps, *value.shape[2:])).flatten(1)).view(batch, steps, -1)


class GPSTokenEncoder(nn.Module):
    def __init__(self, input_size: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_size)),
            nn.Linear(int(input_size), int(output_dim)),
            nn.GELU(),
            nn.Linear(int(output_dim), int(output_dim)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError(f"GPS input must be [B,T,F], got {tuple(value.shape)}.")
        return self.net(value)


class BeamPrototypeHead(nn.Module):
    def __init__(self, d_model: int, *, num_classes: int = 64, temperature: float = 0.1) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(int(num_classes), int(d_model)) * 0.02)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature)), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(min=0.02, max=1.0)
        return F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).t() / temperature


class AuxiliaryPrototypeHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), int(d_model)), nn.GELU())

    def forward(self, tokens: torch.Tensor, prototype: BeamPrototypeHead) -> torch.Tensor:
        feature = self.net(tokens.mean(dim=1))
        temperature = prototype.log_temperature.detach().exp().clamp(min=0.02, max=1.0)
        return F.normalize(feature, dim=-1) @ F.normalize(prototype.prototypes.detach(), dim=-1).t() / temperature


class BTSCLModel(nn.Module):
    """Four encoders, fixed time-modality tokens, MLP fusion and shared prototypes."""

    def __init__(self, *, d_model: int = 256, seq_len: int = 5, gps_input_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model, self.seq_len = int(d_model), int(seq_len)
        raw_dim = 128
        self.encoders = nn.ModuleDict(
            {
                "image": SpatialTokenEncoder(3, raw_dim),
                "lidar": SpatialTokenEncoder(3, raw_dim),
                "radar": SpatialTokenEncoder(2, raw_dim),
                "gps": GPSTokenEncoder(gps_input_size, raw_dim),
            }
        )
        self.projections = nn.ModuleDict({name: nn.Linear(raw_dim, self.d_model) for name in MODALITIES})
        self.modality_embedding = nn.Parameter(torch.randn(4, self.d_model) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(self.seq_len, self.d_model) * 0.02)
        fusion_input = self.seq_len * 4 * self.d_model + self.seq_len * 4
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
        self.prototype_bank = BeamPrototypeHead(self.d_model)
        self.auxiliary = nn.ModuleDict({name: AuxiliaryPrototypeHead(self.d_model) for name in MODALITIES})

    def encode(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens: dict[str, torch.Tensor] = {}
        for name in MODALITIES:
            if name not in inputs:
                raise ValueError(f"BT-SCL is missing {name} input.")
            token = self.projections[name](self.encoders[name](inputs[name]))
            if token.shape[1] != self.seq_len:
                raise ValueError(f"BT-SCL {name} has {token.shape[1]} time steps, expected {self.seq_len}.")
            tokens[name] = token
        return tokens

    def logits_from_tokens(self, tokens: Mapping[str, torch.Tensor], availability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if availability.ndim != 2 or availability.shape[1] != 4:
            raise ValueError("BT-SCL availability must have shape [B,4].")
        if not bool(availability.any(dim=1).all()):
            raise ValueError("BT-SCL rejects all-missing inputs.")
        batch = availability.shape[0]
        stacked = torch.stack([tokens[name] for name in MODALITIES], dim=2)
        if stacked.shape[:2] != (batch, self.seq_len):
            raise ValueError("BT-SCL tokens and availability disagree on batch/time shape.")
        positioned = stacked + self.time_embedding.view(1, self.seq_len, 1, self.d_model) + self.modality_embedding.view(1, 1, 4, self.d_model)
        token_mask = availability[:, None, :, None].to(dtype=positioned.dtype)
        masked = positioned * token_mask
        block_mask = availability[:, None, :].expand(-1, self.seq_len, -1).to(dtype=positioned.dtype)
        fused = self.fusion(torch.cat((masked.flatten(1), block_mask.flatten(1)), dim=1))
        return self.prototype_bank(fused), masked

    def forward_views(self, inputs: Mapping[str, torch.Tensor], chain_masks: torch.Tensor) -> dict[str, Any]:
        tokens = self.encode(inputs)
        if chain_masks.ndim != 3 or tuple(chain_masks.shape[1:]) != (4, 4):
            raise ValueError("BT-SCL chain masks must have shape [B,4,4].")
        logits, token_views = [], []
        for index in range(4):
            view_logits, view_tokens = self.logits_from_tokens(tokens, chain_masks[:, index])
            logits.append(view_logits)
            token_views.append(view_tokens)
        return {"logits": torch.stack(logits, dim=1), "tokens": tokens, "masked_tokens": torch.stack(token_views, dim=1)}

    def forward_pattern(self, inputs: Mapping[str, torch.Tensor], availability: torch.Tensor) -> dict[str, Any]:
        tokens = self.encode(inputs)
        logits, masked = self.logits_from_tokens(tokens, availability)
        return {"logits": logits, "tokens": tokens, "masked_tokens": masked}

    def auxiliary_logits(self, tokens: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.auxiliary[name](tokens[name], self.prototype_bank) for name in MODALITIES}


def topology_risk(logits: torch.Tensor, labels: torch.Tensor, topology: BeamTopology) -> torch.Tensor:
    distance = topology.distance.to(device=logits.device, dtype=logits.dtype) / 32.0
    return (torch.softmax(logits, dim=-1) * distance[labels]).sum(dim=-1)


def task_loss(logits: torch.Tensor, labels: torch.Tensor, topology: BeamTopology, *, topology_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ce = F.cross_entropy(logits, labels, reduction="none")
    risk = topology_risk(logits, labels, topology)
    return ce + float(topology_weight) * risk, {"ce": ce, "risk": risk}


def monotonicity_loss(logits: torch.Tensor, labels: torch.Tensor, topology: BeamTopology) -> torch.Tensor:
    risk = torch.stack([topology_risk(logits[:, index], labels, topology) for index in range(4)], dim=1)
    return F.relu(risk[:, 1:] - risk[:, :-1].detach()).mean()


def _aggregate_sectors(probabilities: torch.Tensor, sectors: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    return torch.stack([probabilities[:, list(sector)].sum(dim=-1) for sector in sectors], dim=-1)


def coarse_to_fine_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topology: BeamTopology,
    *,
    temperature: float = 2.0,
    local_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probabilities = torch.softmax(logits / float(temperature), dim=-1)
    sector_losses = []
    for lower, upper, sectors in ((0, 1, 4), (1, 2, 8), (2, 3, 16)):
        teacher = _aggregate_sectors(probabilities[:, upper].detach(), topology.sector_labels(sectors))
        student = _aggregate_sectors(probabilities[:, lower], topology.sector_labels(sectors))
        sector_losses.append(F.kl_div(student.clamp_min(1e-8).log(), teacher, reduction="batchmean") * float(temperature) ** 2)
    distance = topology.distance.to(device=logits.device)
    neighborhood = distance[labels].le(3)
    full_local = probabilities[:, 3].detach().masked_fill(~neighborhood, 0.0)
    small_local = probabilities[:, 2].masked_fill(~neighborhood, 0.0)
    full_local = full_local / full_local.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    small_local = small_local / small_local.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    local = F.kl_div(small_local.clamp_min(1e-8).log(), full_local, reduction="batchmean")
    sectors_mean = torch.stack(sector_losses).mean()
    return sectors_mean + float(local_weight) * local, {
        "sector_4": sector_losses[0],
        "sector_8": sector_losses[1],
        "sector_16": sector_losses[2],
        "local": local,
    }


def hierarchical_sector_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topology: BeamTopology,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Anchor increasingly informative subset views to 4/8/16 label sectors."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    losses: dict[str, torch.Tensor] = {}
    for view, count in enumerate((4, 8, 16)):
        sectors = topology.sector_labels(count)
        sector_probabilities = _aggregate_sectors(probabilities[:, view], sectors)
        label_to_sector = torch.empty(64, dtype=torch.long, device=labels.device)
        for sector_index, sector_labels in enumerate(sectors):
            label_to_sector[list(sector_labels)] = sector_index
        losses[f"sector_{count}"] = F.nll_loss(
            sector_probabilities.clamp_min(1e-8).log(),
            label_to_sector[labels],
        )
    return torch.stack(list(losses.values())).mean(), losses


def stochastic_dominance_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topology: BeamTopology,
    *,
    radii: tuple[int, ...] = (0, 3, 5),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Require each larger subset to retain ground-truth neighborhood mass."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    distance = topology.distance.to(device=logits.device)[labels]
    losses: dict[str, torch.Tensor] = {}
    for radius in radii:
        mass = (probabilities * distance.le(radius)[:, None, :]).sum(dim=-1)
        losses[f"radius_{radius}"] = F.relu(mass[:, :-1].detach() - mass[:, 1:]).mean()
    return torch.stack(list(losses.values())).mean(), losses


def auxiliary_loss(aux_logits: Mapping[str, torch.Tensor], labels: torch.Tensor, topology: BeamTopology, *, topology_weight: float) -> torch.Tensor:
    return torch.stack([task_loss(aux_logits[name], labels, topology, topology_weight=topology_weight)[0].mean() for name in MODALITIES]).mean()


def btscl_losses(
    model: BTSCLModel,
    views: Mapping[str, Any],
    labels: torch.Tensor,
    topology: BeamTopology,
    method: str,
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    if method not in METHODS:
        raise ValueError(f"Unknown BT-SCL method {method!r}.")
    logits = views["logits"]
    per_view = [task_loss(logits[:, index], labels, topology, topology_weight=float(weights["task_topology"]))[0].mean() for index in range(4)]
    chain = torch.stack(per_view).mean()
    base = 0.75 * chain + 0.25 * per_view[3]
    total, report = base, {"base": float(base.detach())}
    if method in {"r1_available_evidence", "r5_full_bt_scl"}:
        uni = auxiliary_loss(model.auxiliary_logits(views["tokens"]), labels, topology, topology_weight=float(weights["task_topology"]))
        total = total + float(weights["uni"]) * uni
        report["uni"] = float(uni.detach())
    if method in {"r2_topology_monotonicity", "r4_mono_c2f", "r5_full_bt_scl"}:
        mono = monotonicity_loss(logits, labels, topology)
        total = total + float(weights["mono"]) * mono
        report["mono"] = float(mono.detach())
    if method in {"r3_coarse_to_fine", "r4_mono_c2f", "r5_full_bt_scl"}:
        c2f, detail = coarse_to_fine_loss(logits, labels, topology, local_weight=float(weights["local"]))
        total = total + float(weights["c2f"]) * c2f
        report.update({f"c2f_{name}": float(value.detach()) for name, value in detail.items()})
        report["c2f"] = float(c2f.detach())
    if method == "r6_topological_stochastic_dominance":
        hierarchy, hierarchy_detail = hierarchical_sector_loss(logits, labels, topology)
        dominance, dominance_detail = stochastic_dominance_loss(logits, labels, topology)
        total = total + float(weights["hierarchy"]) * hierarchy + float(weights["dominance"]) * dominance
        report.update({f"hierarchy_{name}": float(value.detach()) for name, value in hierarchy_detail.items()})
        report.update({f"dominance_{name}": float(value.detach()) for name, value in dominance_detail.items()})
        report["hierarchy"] = float(hierarchy.detach())
        report["dominance"] = float(dominance.detach())
    report["total"] = float(total.detach())
    return total, report


def parameter_rows(model: BTSCLModel, method: str) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_children():
        total = sum(parameter.numel() for parameter in module.parameters())
        auxiliary = name == "auxiliary"
        trainable = method in {"r1_available_evidence", "r5_full_bt_scl"} if auxiliary else True
        rows.append(
            {
                "method": method,
                "module": name,
                "total_params": total,
                "trainable_params": total if trainable else 0,
                "requires_grad": trainable,
                "used_at_inference": not auxiliary,
            }
        )
    return rows


def check_missing_token_invariance(model: BTSCLModel, inputs: Mapping[str, torch.Tensor], availability: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        reference = model.forward_pattern(inputs, availability)["logits"]
        altered = {name: value.clone() for name, value in inputs.items()}
        for index, name in enumerate(MODALITIES):
            if bool((~availability[:, index]).any()):
                value = altered[name]
                value[~availability[:, index]] = torch.randn_like(value[~availability[:, index]]) * 100.0
        candidate = model.forward_pattern(altered, availability)["logits"]
    if not torch.equal(reference, candidate):
        raise AssertionError("Missing-modality token invariance failed.")


__all__ = [
    "BTSCLModel",
    "BeamTopology",
    "METHODS",
    "MODALITIES",
    "PATTERNS",
    "auxiliary_loss",
    "btscl_losses",
    "check_missing_token_invariance",
    "coarse_to_fine_loss",
    "generate_nested_schedule",
    "hierarchical_sector_loss",
    "load_audited_topology",
    "monotonicity_loss",
    "parameter_rows",
    "schedule_masks",
    "sha256_file",
    "sha256_json",
    "stochastic_dominance_loss",
    "task_loss",
    "topology_risk",
    "write_json",
]
