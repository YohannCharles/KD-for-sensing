"""Quality-aware fusion of sensing and radio evidence over shared prototypes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.radio_prototype_expert import PositiveTemperature, RadioPrototypeExpert
from kd_sensing.models.radio_trust_estimator import RadioTrustEstimator, build_trust_features
from kd_sensing.models.topology_prototype_gate import (
    IndependentPrototypeGate,
    TopologyPrototypeGate,
    prototype_gate_input,
)


QTPR_METHODS = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")


class DynamicPrototypeFusion(nn.Module):
    """Fuse two shared-bank experts without adding a free decision head."""

    def __init__(
        self,
        method: str,
        *,
        labels_by_position: Sequence[int] | torch.Tensor,
        radio_dim: int = 128,
        prototype_dim: int = 64,
        radio_hidden_dim: int = 128,
        trust_hidden_dim: int = 32,
        gate_hidden_channels: int = 16,
        gate_kernel_size: int = 3,
        gate_initial_probability: float = 0.9,
        structured_trust_base_bias: float = -2.0,
        structured_trust_raw_quality: float = 1.5,
        radio_temperature: float = 0.1,
        sensing_temperature: float = 1.0,
        fixed_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.method = str(method)
        if self.method not in QTPR_METHODS:
            raise ValueError(f"Unknown QTPR method: {self.method}.")
        self.num_beams = int(prototype_dim)
        self.fixed_weight = float(fixed_weight)
        if not 0.0 <= self.fixed_weight <= 1.0:
            raise ValueError("fixed_weight must be in [0,1].")
        self.radio_expert = RadioPrototypeExpert(
            radio_dim=int(radio_dim),
            hidden_dim=int(radio_hidden_dim),
            prototype_dim=self.num_beams,
            temperature=float(radio_temperature),
        )
        self.sensing_temperature = PositiveTemperature(float(sensing_temperature))
        structured = self.method in {"F5", "F6", "F7"}
        self.trust_estimator = (
            RadioTrustEstimator(
                hidden_dim=int(trust_hidden_dim),
                structured=structured,
                structured_base_bias=float(structured_trust_base_bias),
                structured_raw_quality=float(structured_trust_raw_quality),
            )
            if self.method in {"F2", "F3", "F4", "F5", "F6", "F7"}
            else None
        )
        self.prototype_gate: nn.Module | None
        if self.method == "F3":
            self.prototype_gate = IndependentPrototypeGate(
                num_beams=self.num_beams,
                initial_probability=float(gate_initial_probability),
            )
        elif self.method in {"F4", "F5", "F6", "F7"}:
            self.prototype_gate = TopologyPrototypeGate(
                num_beams=self.num_beams,
                hidden_channels=int(gate_hidden_channels),
                kernel_size=int(gate_kernel_size),
                labels_by_position=labels_by_position,
                circular=True,
                initial_probability=float(gate_initial_probability),
            )
        else:
            self.prototype_gate = None
        self.register_buffer(
            "labels_by_position",
            torch.as_tensor(labels_by_position, dtype=torch.long).reshape(-1),
            persistent=True,
        )

    def forward(
        self,
        sensing_embedding: torch.Tensor,
        sensing_evidence: torch.Tensor,
        c_radio: torch.Tensor,
        csi_quality: torch.Tensor,
        csi_available: torch.Tensor,
        physical_availability: torch.Tensor,
        prototype_bank: BeamPrototypeBank,
        topology_distance: torch.Tensor,
        *,
        rho_floor: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        sensing = torch.as_tensor(sensing_evidence)
        physical = torch.as_tensor(physical_availability, device=sensing.device, dtype=torch.bool)
        if sensing.ndim != 2 or sensing.shape[-1] != self.num_beams:
            raise ValueError(f"sensing_evidence must have shape [B,{self.num_beams}].")
        if physical.shape != (sensing.shape[0], 4) or not bool(physical.any(dim=-1).all()):
            raise ValueError("physical_availability must be a non-empty [B,4] mask.")
        if bool(physical.all(dim=-1).any()):
            raise ValueError("Full samples must bypass DynamicPrototypeFusion.")
        available = torch.as_tensor(csi_available, device=sensing.device, dtype=torch.bool).reshape(-1)
        radio = self.radio_expert(c_radio, prototype_bank)
        sensing_calibrated = sensing / self.sensing_temperature()
        radio_calibrated = radio["radio_evidence"]

        statistics: dict[str, torch.Tensor] = {}
        if self.method == "F0":
            rho = available.to(sensing.dtype)
            gate = rho[:, None].expand_as(sensing)
            final = radio_calibrated
        elif self.method == "F1":
            rho = available.to(sensing.dtype) * self.fixed_weight
            gate = rho[:, None].expand_as(sensing)
            final = sensing_calibrated + gate * (radio_calibrated - sensing_calibrated)
        else:
            assert self.trust_estimator is not None
            statistics = build_trust_features(
                physical,
                sensing_embedding,
                sensing_calibrated,
                radio_calibrated,
                csi_quality,
                prototype_bank.prototypes,
                topology_distance,
            )
            trust = self.trust_estimator(statistics, available, rho_floor=float(rho_floor))
            statistics = trust
            rho = trust["rho"]
            if self.method == "F2":
                gate = rho[:, None].expand_as(sensing)
            else:
                assert self.prototype_gate is not None
                gate_values = prototype_gate_input(sensing_calibrated, radio_calibrated)
                gate = self.prototype_gate(gate_values, rho)["prototype_gate"]
            final = sensing_calibrated + gate * (radio_calibrated - sensing_calibrated)

        inactive = ~available
        if bool(inactive.any()):
            final = torch.where(inactive[:, None], sensing, final)
            gate = torch.where(inactive[:, None], torch.zeros_like(gate), gate)
            rho = torch.where(inactive, torch.zeros_like(rho), rho)
        return {
            **radio,
            **statistics,
            "sensing_evidence": sensing,
            "sensing_evidence_calibrated": sensing_calibrated,
            "radio_evidence_calibrated": radio_calibrated,
            "rho": rho,
            "prototype_gate": gate,
            "final_evidence": final,
            "final_probability": torch.softmax(final, dim=-1),
            "csi_available": available,
        }


class MatchedConcatHead(nn.Module):
    """Budget-matched free-classifier control; never used by QTPR methods."""

    def __init__(self, *, sensing_dim: int = 64, radio_dim: int = 128, hidden_dim: int = 128, num_beams: int = 64) -> None:
        super().__init__()
        input_dim = int(sensing_dim) + int(radio_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_beams)),
        )

    def forward(self, sensing_embedding: torch.Tensor, c_radio: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((sensing_embedding, c_radio), dim=-1))


__all__ = ["DynamicPrototypeFusion", "MatchedConcatHead", "QTPR_METHODS"]
