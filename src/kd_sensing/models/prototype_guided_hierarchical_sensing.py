"""Hierarchical sensing encoders used by the local TSPC-V2 workflow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank


PROTOTYPE_MODES = ("shared_frozen", "independent", "no_prototype", "random_frozen")
SENSING_TEMPORAL_METHODS = ("mean", "gru", "lstm", "tcn", "transformer")


@dataclass(frozen=True)
class HierarchicalSensingConfig:
    """Validated dimensions and controls for fixed-slot sensing fusion."""

    feature_dim: int = 64
    history_length: int = 5
    num_modalities: int = 4
    num_heads: int = 4
    frame_layers: int = 2
    temporal_method: str = "lstm"
    temporal_hidden_dim: int = 128
    temporal_layers: int = 2
    dropout: float = 0.1
    prototype_mode: str = "shared_frozen"
    random_seed: int = 1

    def __post_init__(self) -> None:
        if self.feature_dim <= 0 or self.history_length <= 0 or self.num_modalities <= 0:
            raise ValueError("Sensing dimensions must be positive.")
        if self.feature_dim % self.num_heads:
            raise ValueError("feature_dim must be divisible by num_heads.")
        if self.frame_layers <= 0 or self.temporal_layers <= 0 or self.temporal_hidden_dim <= 0:
            raise ValueError("Sensing layer counts and hidden dimension must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1).")
        if self.temporal_method not in SENSING_TEMPORAL_METHODS:
            raise ValueError(f"Unsupported sensing temporal method: {self.temporal_method}.")
        if self.prototype_mode not in PROTOTYPE_MODES:
            raise ValueError(f"Unsupported prototype mode: {self.prototype_mode}.")


def _coerce_availability(
    availability_mask: torch.Tensor,
    *,
    batch_size: int,
    history_length: int,
    num_modalities: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a non-empty bool availability tensor with shape [B,T,M]."""

    availability = torch.as_tensor(availability_mask, device=device, dtype=torch.bool)
    if tuple(availability.shape) == (batch_size, num_modalities):
        availability = availability[:, None, :].expand(-1, history_length, -1)
    if tuple(availability.shape) != (batch_size, history_length, num_modalities):
        raise ValueError(
            "availability_mask must have shape "
            f"[B,{num_modalities}] or [B,{history_length},{num_modalities}]."
        )
    if not bool(availability.any(dim=(1, 2)).all()):
        raise ValueError("Each sensing sample must retain at least one available modality token.")
    return availability


def _random_frozen_bank(*, feature_dim: int, num_beams: int, seed: int) -> BeamPrototypeBank:
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(feature_dim, feature_dim, generator=generator)
    orthogonal = torch.linalg.qr(matrix).Q[:num_beams]
    bank = BeamPrototypeBank(feature_dim, num_beams, temperature=0.1)
    with torch.no_grad():
        bank.prototypes.copy_(orthogonal)
    bank.prototypes.requires_grad_(False)
    return bank


class SensingPrototypeHead(nn.Module):
    """Map [B,D] sensing embeddings to 64 beam evidence under a selected control."""

    def __init__(self, *, feature_dim: int, prototype_mode: str, random_seed: int, num_beams: int = 64) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_beams = int(num_beams)
        self.prototype_mode = str(prototype_mode)
        if self.prototype_mode not in PROTOTYPE_MODES:
            raise ValueError(f"Unsupported prototype mode: {prototype_mode}.")
        self.prototype_bank: BeamPrototypeBank | None = None
        self.classifier: nn.Module | None = None
        if self.prototype_mode == "independent":
            self.prototype_bank = BeamPrototypeBank(self.feature_dim, self.num_beams, temperature=0.1)
        elif self.prototype_mode == "random_frozen":
            self.prototype_bank = _random_frozen_bank(
                feature_dim=self.feature_dim,
                num_beams=self.num_beams,
                seed=int(random_seed),
            )
        elif self.prototype_mode == "no_prototype":
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, self.num_beams),
            )

    def decision_bank(self, shared_prototype_bank: BeamPrototypeBank | None) -> BeamPrototypeBank | None:
        if self.prototype_mode == "shared_frozen":
            if shared_prototype_bank is None:
                raise ValueError("shared_frozen sensing head requires a shared prototype bank.")
            if shared_prototype_bank.d_model != self.feature_dim or shared_prototype_bank.num_beams != self.num_beams:
                raise ValueError("Shared prototype bank dimensions do not match the sensing head.")
            return shared_prototype_bank
        return self.prototype_bank

    def forward(self, embedding: torch.Tensor, shared_prototype_bank: BeamPrototypeBank | None) -> torch.Tensor:
        features = torch.as_tensor(embedding)
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"embedding must have shape [B,{self.feature_dim}].")
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            bank = self.decision_bank(shared_prototype_bank)
            if bank is not None:
                if self.prototype_mode == "shared_frozen":
                    # Keep the external M4 bank immutable even if a caller
                    # forgot to set requires_grad=False on its parameters.
                    prototypes = F.normalize(bank.prototypes.detach().float().to(features.device), dim=-1)
                    return F.normalize(features, dim=-1) @ prototypes.t() / bank.temperature
                return bank(features).float()
            assert self.classifier is not None
            return self.classifier(features).float()


class _AsymmetricFrameBlock(nn.Module):
    """One pre-norm frame block where missing modality tokens are never keys."""

    def __init__(self, *, feature_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.norm_attention = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, frame_tokens: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        """Fuse [B,1+M,D] tokens with availability [B,M]."""

        tokens = torch.as_tensor(frame_tokens)
        available = torch.as_tensor(availability, device=tokens.device, dtype=torch.bool)
        batch, sequence_length, _ = tokens.shape
        modalities = sequence_length - 1
        if available.shape != (batch, modalities):
            raise ValueError("Frame availability must have shape [B,M].")

        # Key 0 is the fusion token. All missing modality keys are blocked for
        # every query; missing queries can still read the observed context.
        allowed = torch.zeros(batch, sequence_length, sequence_length, dtype=torch.bool, device=tokens.device)
        allowed[:, :, 0] = True
        allowed[:, :, 1:] = available[:, None, :]
        attention_mask = (~allowed[:, None]).expand(-1, self.num_heads, -1, -1).reshape(
            batch * self.num_heads, sequence_length, sequence_length
        )
        normalized = self.norm_attention(tokens)
        attended = self.attention(normalized, normalized, normalized, attn_mask=attention_mask, need_weights=False)[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_mlp(tokens))


class _TemporalSensingAggregator(nn.Module):
    """Aggregate frame features [B,T,D] into one sensing embedding [B,D]."""

    def __init__(self, config: HierarchicalSensingConfig) -> None:
        super().__init__()
        self.method = config.temporal_method
        self.feature_dim = int(config.feature_dim)
        self.history_length = int(config.history_length)
        hidden = int(config.temporal_hidden_dim)
        dropout = float(config.dropout)
        self.recurrent: nn.Module | None = None
        self.projection: nn.Module | None = None
        self.temporal: nn.Module | None = None
        if self.method == "mean":
            self.temporal = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
            )
        elif self.method in {"gru", "lstm"}:
            recurrent = nn.GRU if self.method == "gru" else nn.LSTM
            self.recurrent = recurrent(
                self.feature_dim,
                hidden,
                num_layers=int(config.temporal_layers),
                dropout=dropout if config.temporal_layers > 1 else 0.0,
                batch_first=True,
            )
            self.projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.feature_dim))
        elif self.method == "tcn":
            self.temporal = nn.Sequential(
                nn.Conv1d(self.feature_dim, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, self.feature_dim, kernel_size=3, padding=1),
            )
            self.projection = nn.LayerNorm(self.feature_dim)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=int(config.num_heads),
                dim_feedforward=self.feature_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=int(config.temporal_layers), enable_nested_tensor=False)
            self.position = nn.Parameter(torch.zeros(1, self.history_length, self.feature_dim))
            self.projection = nn.LayerNorm(self.feature_dim)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        frames = torch.as_tensor(frame_features)
        if frames.ndim != 3 or frames.shape[-1] != self.feature_dim or frames.shape[1] != self.history_length:
            raise ValueError(f"frame_features must have shape [B,{self.history_length},{self.feature_dim}].")
        if self.method == "mean":
            assert self.temporal is not None
            pooled = frames.mean(dim=1)
            return pooled + self.temporal(pooled)
        if self.method in {"gru", "lstm"}:
            assert self.recurrent is not None and self.projection is not None
            return self.projection(self.recurrent(frames)[0][:, -1])
        if self.method == "tcn":
            assert self.temporal is not None and self.projection is not None
            residual = frames[:, -1]
            encoded = self.temporal(frames.transpose(1, 2)).transpose(1, 2)[:, -1]
            return self.projection(residual + encoded)
        assert self.temporal is not None and self.projection is not None
        encoded = self.temporal(frames + self.position[:, : frames.shape[1]])
        return self.projection(encoded[:, -1])


class PrototypeGuidedHierarchicalSensingEncoder(nn.Module):
    """Mask-aware fixed-slot sensing fusion from [B,T,4,64] to beam evidence."""

    def __init__(self, config: HierarchicalSensingConfig) -> None:
        super().__init__()
        self.config = config
        feature_dim = int(config.feature_dim)
        self.missing_tokens = nn.Parameter(torch.randn(config.num_modalities, feature_dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(config.num_modalities, feature_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(config.history_length, feature_dim) * 0.02)
        self.availability_embedding = nn.Embedding(2, feature_dim)
        self.frame_fusion_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.frame_blocks = nn.ModuleList(
            [
                _AsymmetricFrameBlock(
                    feature_dim=feature_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.frame_layers)
            ]
        )
        self.frame_norm = nn.LayerNorm(feature_dim)
        self.temporal_aggregator = _TemporalSensingAggregator(config)
        self.evidence_head = SensingPrototypeHead(
            feature_dim=feature_dim,
            prototype_mode=config.prototype_mode,
            random_seed=config.random_seed,
        )

    def decision_bank(self, shared_prototype_bank: BeamPrototypeBank | None) -> BeamPrototypeBank | None:
        return self.evidence_head.decision_bank(shared_prototype_bank)

    def forward(
        self,
        sensing_features: torch.Tensor,
        availability_mask: torch.Tensor,
        shared_prototype_bank: BeamPrototypeBank | None,
    ) -> dict[str, torch.Tensor]:
        """Return frame features [B,T,D], embedding [B,D], and evidence [B,64]."""

        features = torch.as_tensor(sensing_features)
        if features.ndim != 4:
            raise ValueError("sensing_features must have shape [B,T,M,D].")
        batch, history, modalities, feature_dim = features.shape
        expected = (self.config.history_length, self.config.num_modalities, self.config.feature_dim)
        if (history, modalities, feature_dim) != expected:
            raise ValueError(f"sensing_features must have shape [B,{expected[0]},{expected[1]},{expected[2]}].")
        availability = _coerce_availability(
            availability_mask,
            batch_size=batch,
            history_length=history,
            num_modalities=modalities,
            device=features.device,
        )
        missing = self.missing_tokens[None, None].expand(batch, history, -1, -1)
        slots = torch.where(availability[..., None], features, missing)
        slots = slots + self.modality_embedding[None, None] + self.time_embedding[None, :, None]
        slots = slots + self.availability_embedding(availability.long())
        fusion = self.frame_fusion_token.expand(batch, history, -1, -1) + self.time_embedding[None, :, None]
        tokens = torch.cat((fusion, slots), dim=2).reshape(batch * history, modalities + 1, feature_dim)
        frame_availability = availability.reshape(batch * history, modalities)
        for block in self.frame_blocks:
            tokens = block(tokens, frame_availability)
        frame_features = self.frame_norm(tokens[:, 0]).reshape(batch, history, feature_dim)
        embedding = self.temporal_aggregator(frame_features)
        evidence = self.evidence_head(embedding, shared_prototype_bank)
        return {
            "frame_features": frame_features,
            "z_sensing": embedding,
            "sensing_evidence": evidence,
            "availability": availability,
        }


class LegacyFlattenSensingEncoder(nn.Module):
    """A0 control retaining the old flatten-then-MLP sensing structure."""

    def __init__(self, config: HierarchicalSensingConfig) -> None:
        super().__init__()
        self.config = config
        feature_dim = int(config.feature_dim)
        fusion_input = config.history_length * config.num_modalities * feature_dim
        self.modality_embedding = nn.Parameter(torch.randn(config.num_modalities, feature_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(config.history_length, feature_dim) * 0.02)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, 1024),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, feature_dim),
        )
        self.evidence_head = SensingPrototypeHead(
            feature_dim=feature_dim,
            prototype_mode=config.prototype_mode,
            random_seed=config.random_seed,
        )

    def decision_bank(self, shared_prototype_bank: BeamPrototypeBank | None) -> BeamPrototypeBank | None:
        return self.evidence_head.decision_bank(shared_prototype_bank)

    def forward(
        self,
        sensing_features: torch.Tensor,
        availability_mask: torch.Tensor,
        shared_prototype_bank: BeamPrototypeBank | None,
    ) -> dict[str, torch.Tensor]:
        features = torch.as_tensor(sensing_features)
        if features.ndim != 4:
            raise ValueError("sensing_features must have shape [B,T,M,D].")
        batch, history, modalities, feature_dim = features.shape
        expected = (self.config.history_length, self.config.num_modalities, self.config.feature_dim)
        if (history, modalities, feature_dim) != expected:
            raise ValueError(f"sensing_features must have shape [B,{expected[0]},{expected[1]},{expected[2]}].")
        availability = _coerce_availability(
            availability_mask,
            batch_size=batch,
            history_length=history,
            num_modalities=modalities,
            device=features.device,
        )
        positioned = features + self.modality_embedding[None, None] + self.time_embedding[None, :, None]
        positioned = positioned * availability[..., None].to(positioned)
        embedding = self.fusion(positioned.flatten(1))
        weights = availability.to(features.dtype)
        frame_features = (features * weights[..., None]).sum(dim=2) / weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        evidence = self.evidence_head(embedding, shared_prototype_bank)
        return {
            "frame_features": frame_features,
            "z_sensing": embedding,
            "sensing_evidence": evidence,
            "availability": availability,
        }


__all__ = [
    "HierarchicalSensingConfig",
    "LegacyFlattenSensingEncoder",
    "PROTOTYPE_MODES",
    "PrototypeGuidedHierarchicalSensingEncoder",
    "SENSING_TEMPORAL_METHODS",
    "SensingPrototypeHead",
]
