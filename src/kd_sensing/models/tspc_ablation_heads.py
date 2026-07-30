"""Prototype and fusion controls for final TSPC ablations."""

from __future__ import annotations

import torch
from torch import nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.radio_prototype_expert import PositiveTemperature, RadioPrototypeExpert
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder
from kd_sensing.models.temporal_radio_encoders import TemporalRadioEncoder


PROTOTYPE_ABLATION_METHODS = ("P0", "P1", "P2", "P3", "P4", "P5")
FUSION_LOCATION_METHODS = ("L0", "L1", "L2", "L3", "L4")


def random_frozen_prototype_bank(*, seed: int, dimension: int = 64) -> BeamPrototypeBank:
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(dimension, dimension, generator=generator)
    orthogonal = torch.linalg.qr(matrix).Q
    bank = BeamPrototypeBank(dimension, dimension, temperature=0.1)
    with torch.no_grad():
        bank.prototypes.copy_(orthogonal)
    bank.prototypes.requires_grad_(False)
    return bank


class RadioAblationHead(nn.Module):
    """P0--P5 radio decision heads; P3 is handled by ``ConcatAblationHead``."""

    def __init__(self, method: str, *, radio_dim: int = 128, prototype_dim: int = 64, seed: int = 1) -> None:
        super().__init__()
        self.method = str(method).upper()
        if self.method not in PROTOTYPE_ABLATION_METHODS or self.method == "P3":
            raise ValueError(f"RadioAblationHead does not support {method}.")
        self.radio_dim = int(radio_dim)
        self.prototype_dim = int(prototype_dim)
        self.expert: RadioPrototypeExpert | None = None
        self.prototype_bank: BeamPrototypeBank | None = None
        self.classifier: nn.Module | None = None
        self.embedding: nn.Module | None = None
        self.temperature: PositiveTemperature | None = None
        if self.method in {"P0", "P1", "P4"}:
            self.expert = RadioPrototypeExpert(
                radio_dim=self.radio_dim,
                hidden_dim=self.radio_dim,
                prototype_dim=self.prototype_dim,
                temperature=0.1,
            )
            if self.method == "P1":
                self.prototype_bank = BeamPrototypeBank(self.prototype_dim, 64, temperature=0.1)
            elif self.method == "P4":
                self.prototype_bank = random_frozen_prototype_bank(seed=seed, dimension=self.prototype_dim)
        elif self.method == "P2":
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.radio_dim),
                nn.Linear(self.radio_dim, self.radio_dim),
                nn.GELU(),
                nn.Linear(self.radio_dim, 64),
            )
            self.temperature = PositiveTemperature(1.0)
        else:
            self.embedding = nn.Sequential(nn.LayerNorm(self.radio_dim), nn.Linear(self.radio_dim, self.prototype_dim))
            self.classifier = nn.Linear(self.prototype_dim, 64)
            self.temperature = PositiveTemperature(1.0)

    def decision_bank(self, shared_bank: BeamPrototypeBank) -> BeamPrototypeBank | None:
        if self.method == "P0":
            return shared_bank
        return self.prototype_bank

    def forward(self, c_radio: torch.Tensor, shared_bank: BeamPrototypeBank) -> dict[str, torch.Tensor]:
        radio = torch.as_tensor(c_radio)
        if radio.ndim != 2 or radio.shape[-1] != self.radio_dim:
            raise ValueError(f"c_radio must have shape [B,{self.radio_dim}].")
        with torch.autocast(device_type=radio.device.type, enabled=False):
            radio = radio.float()
            if self.expert is not None:
                bank = self.decision_bank(shared_bank)
                assert bank is not None
                return self.expert(radio, bank)
            if self.method == "P5":
                assert self.embedding is not None
                z_radio = self.embedding(radio)
                raw = self.classifier(z_radio)
            else:
                z_radio = radio.new_zeros(radio.shape[0], self.prototype_dim)
                raw = self.classifier(radio)
            assert self.temperature is not None
            evidence = raw.float() / self.temperature().float()
            return {
                "z_radio": z_radio,
                "radio_evidence": evidence,
                "radio_probability": torch.softmax(evidence, dim=-1),
                "radio_temperature": self.temperature().float(),
            }


class ConcatAblationHead(nn.Module):
    def __init__(self, *, sensing_dim: int = 64, radio_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        total = int(sensing_dim) + int(radio_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 64),
        )

    def forward(self, z_sensing: torch.Tensor, c_radio: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((z_sensing.float(), c_radio.float()), dim=-1))


class SparseRadioAblationModel(nn.Module):
    """Shared per-frame encoder, one temporal encoder, and one P0--P5 head."""

    def __init__(
        self,
        temporal_method: str,
        head_method: str,
        *,
        hidden_dim: int = 128,
        encoder_layers: int = 0,
        num_candidate_patterns: int = 32,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.temporal_method = str(temporal_method).lower()
        self.head_method = str(head_method).upper()
        self.frame_encoder = SparsePilotEncoder(
            num_candidate_patterns=int(num_candidate_patterns),
            hidden_dim=int(hidden_dim),
            num_layers=int(encoder_layers),
        )
        self.temporal_encoder = TemporalRadioEncoder(self.temporal_method, hidden_dim=int(hidden_dim))
        if self.head_method == "P3":
            self.radio_head: nn.Module = ConcatAblationHead(radio_dim=int(hidden_dim))
        else:
            self.radio_head = RadioAblationHead(
                self.head_method,
                radio_dim=int(hidden_dim),
                prototype_dim=64,
                seed=int(seed),
            )

    def freeze_radio_backbone(self) -> None:
        for module in (self.frame_encoder, self.temporal_encoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.frame_encoder.parameters()):
            self.frame_encoder.eval()
        if not any(parameter.requires_grad for parameter in self.temporal_encoder.parameters()):
            self.temporal_encoder.eval()
        return self

    def forward(
        self,
        pilot_observations: torch.Tensor,
        pattern_ids: torch.Tensor,
        frequency_positions: torch.Tensor,
        pilot_mask: torch.Tensor,
        snr_db: torch.Tensor,
        shared_bank: BeamPrototypeBank,
        *,
        z_sensing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_radio(
            pilot_observations,
            pattern_ids,
            frequency_positions,
            pilot_mask,
            snr_db,
        )
        c_radio = encoded["c_radio"]
        batch = c_radio.shape[0]
        if self.head_method == "P3":
            if z_sensing is None or z_sensing.shape != (batch, 64):
                raise ValueError("P3 requires z_sensing [B,64].")
            evidence = self.radio_head(z_sensing, c_radio).float()
            result = {
                "z_radio": evidence.new_zeros(batch, 64),
                "radio_evidence": evidence,
                "radio_probability": torch.softmax(evidence, dim=-1),
                "radio_temperature": evidence.new_tensor(1.0),
            }
        else:
            result = self.radio_head(c_radio, shared_bank)
        return {**result, **encoded}

    def encode_radio(
        self,
        pilot_observations: torch.Tensor,
        pattern_ids: torch.Tensor,
        frequency_positions: torch.Tensor,
        pilot_mask: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        observations = torch.as_tensor(pilot_observations)
        if not torch.is_complex(observations) or observations.ndim != 4:
            raise ValueError("pilot_observations must be complex [B,T,M,K].")
        batch, history, patterns, frequencies = observations.shape
        ids = torch.as_tensor(pattern_ids, device=observations.device, dtype=torch.long)
        valid = torch.as_tensor(pilot_mask, device=observations.device, dtype=torch.bool)
        snr = torch.as_tensor(snr_db, device=observations.device, dtype=observations.real.dtype)
        if ids.ndim == 2:
            ids = ids[:, None].expand(-1, history, -1)
        if snr.ndim == 1:
            snr = snr[:, None].expand(-1, history)
        if ids.shape != (batch, history, patterns) or valid.shape != observations.shape or snr.shape != (batch, history):
            raise ValueError("pattern_ids, pilot_mask, or snr_db does not match [B,T,M,K].")
        encoded = self.frame_encoder(
            observations.reshape(batch * history, patterns, frequencies),
            ids.reshape(batch * history, patterns),
            frequency_positions,
            valid.reshape(batch * history, patterns, frequencies),
            snr.reshape(batch * history),
        )
        frames = encoded["csi_feature"].reshape(batch, history, -1)
        c_radio = self.temporal_encoder(frames)
        available = valid.flatten(1).any(dim=-1)
        return {
            "frame_features": frames,
            "c_radio": c_radio,
            "csi_available": available,
        }


def fuse_expert_probabilities(
    method: str,
    sensing_probability: torch.Tensor,
    radio_evidence: torch.Tensor,
    *,
    weight: float = 0.5,
    z_sensing: torch.Tensor | None = None,
    z_radio: torch.Tensor | None = None,
    shared_bank: BeamPrototypeBank | None = None,
    sensing_temperature: float = 1.0,
) -> torch.Tensor:
    """Apply L1--L4 in FP32; inputs are already temperature-calibrated once."""
    mode = str(method).upper()
    if mode not in {"L1", "L2", "L3", "L4"}:
        raise ValueError(f"Unknown analytic fusion method: {method}.")
    value = float(weight)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Fusion weight must be in [0,1].")
    device_type = sensing_probability.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        p_s_base = sensing_probability.float().clamp_min(1e-12)
        e_s = p_s_base.log() / float(sensing_temperature)
        p_s = torch.softmax(e_s, dim=-1)
        e_c = radio_evidence.float()
        if mode == "L1":
            if z_sensing is None or z_radio is None or shared_bank is None:
                raise ValueError("L1 requires sensing/radio embeddings and the shared bank.")
            embedding = (1.0 - value) * z_sensing.float() + value * z_radio.float()
            return torch.softmax(shared_bank(embedding).float(), dim=-1)
        if mode == "L2":
            return torch.softmax((1.0 - value) * e_s + value * e_c, dim=-1)
        p_c = torch.softmax(e_c, dim=-1)
        if mode == "L3":
            probability = (1.0 - value) * p_s + value * p_c
            return probability / probability.sum(dim=-1, keepdim=True)
        log_probability = (1.0 - value) * p_s.log() + value * p_c.clamp_min(1e-12).log()
        return torch.softmax(log_probability, dim=-1)


def apply_exact_fallback(
    base_probability: torch.Tensor,
    candidate_probability: torch.Tensor,
    *,
    csi_available: torch.Tensor,
    full: torch.Tensor | None = None,
) -> torch.Tensor:
    active = torch.as_tensor(csi_available, device=base_probability.device, dtype=torch.bool).reshape(-1)
    if full is not None:
        active = active & ~torch.as_tensor(full, device=base_probability.device, dtype=torch.bool).reshape(-1)
    return torch.where(active[:, None], candidate_probability, base_probability)


def expected_calibration_error(probability: torch.Tensor, labels: torch.Tensor, *, bins: int = 15) -> float:
    confidence, prediction = probability.float().max(dim=-1)
    correct = prediction.eq(labels).float()
    result = confidence.new_zeros(())
    for index in range(int(bins)):
        left, right = index / bins, (index + 1) / bins
        selected = confidence.ge(left) & (confidence.lt(right) if index + 1 < bins else confidence.le(right))
        if bool(selected.any()):
            result += selected.float().mean() * (confidence[selected].mean() - correct[selected].mean()).abs()
    return float(result.item())


__all__ = [
    "ConcatAblationHead",
    "FUSION_LOCATION_METHODS",
    "PROTOTYPE_ABLATION_METHODS",
    "RadioAblationHead",
    "SparseRadioAblationModel",
    "apply_exact_fallback",
    "expected_calibration_error",
    "fuse_expert_probabilities",
    "random_frozen_prototype_bank",
]
