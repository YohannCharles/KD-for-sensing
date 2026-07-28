from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PrototypeTransition(nn.Module):
    def __init__(
        self,
        *,
        sensing_dim: int,
        csi_dim: int,
        quality_dim: int,
        prototype_dim: int,
        topology_radius: int = 3,
        circular_topology: bool = True,
        use_dual_route: bool = True,
        use_reliability_gate: bool = True,
        use_csi_only_fallback: bool = False,
        use_availability_gate: bool = False,
    ) -> None:
        super().__init__()
        self.topology_radius = int(topology_radius)
        self.circular_topology = bool(circular_topology)
        self.use_dual_route = bool(use_dual_route)
        self.use_reliability_gate = bool(use_reliability_gate)
        self.use_csi_only_fallback = bool(use_csi_only_fallback)
        self.use_availability_gate = bool(use_availability_gate)
        query_dim = int(sensing_dim) + int(csi_dim)
        self.local_query = nn.Linear(query_dim, int(prototype_dim))
        self.global_query = nn.Linear(query_dim, int(prototype_dim))
        self.csi_query = nn.Linear(int(csi_dim), int(prototype_dim))
        route_inputs = int(csi_dim) + int(quality_dim) + 2
        self.route_head = nn.Sequential(nn.Linear(route_inputs, 64), nn.GELU(), nn.Linear(64, 1))
        reliability_inputs = int(quality_dim) + 2 + int(self.use_availability_gate)
        self.reliability_head = nn.Sequential(nn.Linear(reliability_inputs, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(
        self,
        z_sensing: torch.Tensor,
        p0: torch.Tensor,
        prototype_bank: torch.Tensor,
        csi_feature: torch.Tensor,
        csi_quality: torch.Tensor,
        topology_positions: torch.Tensor,
        *,
        csi_available: torch.Tensor,
        quality_confidence: torch.Tensor,
        sensing_availability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if p0.ndim != 2 or prototype_bank.ndim != 2 or p0.shape[1] != prototype_bank.shape[0]:
            raise ValueError("p0 and prototype_bank must have shapes [B,P] and [P,D].")
        fallback_p0 = p0
        p0 = p0.clamp_min(0.0)
        p0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        query_input = torch.cat((z_sensing, csi_feature), dim=-1)
        prototypes = F.normalize(prototype_bank, dim=-1)
        local_scores = F.normalize(self.local_query(query_input), dim=-1) @ prototypes.t()
        global_scores = F.normalize(self.global_query(query_input), dim=-1) @ prototypes.t()
        csi_scores = F.normalize(self.csi_query(csi_feature), dim=-1) @ prototypes.t()

        major = p0.argmax(dim=-1)
        positions = torch.as_tensor(topology_positions, device=p0.device, dtype=p0.dtype).reshape(-1)
        if positions.numel() != p0.shape[1]:
            raise ValueError("topology_positions must contain one position per prototype.")
        distance = (positions[None, :] - positions[major, None]).abs()
        if self.circular_topology:
            distance = torch.minimum(distance, float(p0.shape[1]) - distance)
        local_mask = distance <= float(self.topology_radius)
        q_local = F.softmax(local_scores.masked_fill(~local_mask, torch.finfo(local_scores.dtype).min), dim=-1)
        q_global = F.softmax(global_scores, dim=-1)

        entropy = -(p0 * p0.clamp_min(torch.finfo(p0.dtype).tiny).log()).sum(dim=-1, keepdim=True)
        top = p0.topk(min(2, p0.shape[1]), dim=-1).values
        margin = top[:, :1] if p0.shape[1] == 1 else top[:, :1] - top[:, 1:2]
        route_input = torch.cat((csi_feature, csi_quality, entropy, margin), dim=-1)
        r_global = torch.sigmoid(self.route_head(route_input)).squeeze(-1)
        if self.use_dual_route:
            q_transition = (1.0 - r_global[:, None]) * q_local + r_global[:, None] * q_global
        else:
            r_global = torch.zeros_like(r_global)
            q_transition = q_local
        q_csi = F.softmax(csi_scores, dim=-1)
        q_fallback = q_csi if self.use_csi_only_fallback else q_transition
        disagreement = 0.5 * (p0 - q_fallback).abs().sum(dim=-1, keepdim=True)
        reliability_parts = [csi_quality, entropy, disagreement]
        if self.use_availability_gate:
            availability = (
                torch.ones_like(r_global)
                if sensing_availability is None
                else torch.as_tensor(sensing_availability, device=p0.device, dtype=p0.dtype).reshape(-1)
            ).clamp(0.0, 1.0)
            reliability_parts.append(availability[:, None])
        else:
            availability = torch.ones_like(r_global)
        reliability_input = torch.cat(reliability_parts, dim=-1)
        available = torch.as_tensor(csi_available, device=p0.device, dtype=torch.bool).reshape(-1)
        confidence = torch.as_tensor(quality_confidence, device=p0.device, dtype=p0.dtype).reshape(-1).clamp(0.0, 1.0)
        if self.use_reliability_gate:
            alpha = torch.sigmoid(self.reliability_head(reliability_input)).squeeze(-1) * confidence
        else:
            alpha = torch.ones_like(confidence)
        alpha = alpha * available.to(alpha.dtype)
        p_final = (1.0 - alpha[:, None]) * p0 + alpha[:, None] * q_fallback
        p_final = p_final / p_final.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p_final = torch.where(available[:, None], p_final, fallback_p0)
        return {
            "q_local": q_local,
            "q_global": q_global,
            "q_csi": q_csi,
            "q_fallback": q_fallback,
            "r_global": r_global,
            "alpha": alpha,
            "p_final": p_final,
            "q_transition": q_transition,
            "local_mask": local_mask,
            "sensing_availability": availability,
        }


def prototype_transition_losses(
    outputs: dict[str, torch.Tensor],
    p0: torch.Tensor,
    target: torch.Tensor,
    topology_positions: torch.Tensor,
    *,
    topology_radius: int = 3,
) -> dict[str, torch.Tensor]:
    p_final = outputs["p_final"]
    labels = target.to(device=p_final.device, dtype=torch.long).reshape(-1)
    final_ce = F.nll_loss(p_final.clamp_min(1e-12).log(), labels)
    positions = torch.as_tensor(topology_positions, device=p_final.device, dtype=p_final.dtype).reshape(-1)
    target_positions = positions[labels]
    distance = (positions[None, :] - target_positions[:, None]).abs()
    distance = torch.minimum(distance, float(p_final.shape[1]) - distance)
    final_topology = (p_final * distance).sum(dim=-1).mean()
    base = p0.argmax(dim=-1)
    base_distance = (positions[base] - target_positions).abs()
    base_distance = torch.minimum(base_distance, float(p_final.shape[1]) - base_distance)
    route_target = (base_distance > float(topology_radius)).to(p_final.dtype)
    route = F.binary_cross_entropy(outputs["r_global"], route_target)
    correct = base.eq(labels).to(p_final.dtype)
    quality_weight = 1.0 - outputs["alpha"].detach()
    preserve_kl = (
        p0.detach() * (p0.detach().clamp_min(1e-12).log() - p_final.clamp_min(1e-12).log())
    ).sum(dim=-1)
    preserve = (correct * quality_weight * preserve_kl).sum() / correct.sum().clamp_min(1.0)
    q_fallback = outputs.get("q_fallback", outputs["q_transition"])
    fallback_ce = F.nll_loss(q_fallback.clamp_min(1e-12).log(), labels)
    fallback_topology = (q_fallback * distance).sum(dim=-1).mean()
    gate_target = 1.0 - correct
    available = outputs.get("csi_available")
    if available is None:
        gate = F.binary_cross_entropy(outputs["alpha"].clamp(1e-6, 1.0 - 1e-6), gate_target)
    else:
        available_weight = available.to(device=p_final.device, dtype=p_final.dtype).reshape(-1)
        gate_values = F.binary_cross_entropy(
            outputs["alpha"].clamp(1e-6, 1.0 - 1e-6), gate_target, reduction="none"
        )
        gate = (gate_values * available_weight).sum() / available_weight.sum().clamp_min(1.0)
    return {
        "final_ce": final_ce,
        "final_topology": final_topology,
        "route": route,
        "preserve": preserve,
        "fallback_ce": fallback_ce,
        "fallback_topology": fallback_topology,
        "gate": gate,
    }


__all__ = ["PrototypeTransition", "prototype_transition_losses"]
