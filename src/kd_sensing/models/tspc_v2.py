"""TSPC-V2 sensing/CSI residual model for local development experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank, prototype_alignment_loss
from kd_sensing.models.prototype_guided_hierarchical_sensing import (
    HierarchicalSensingConfig,
    LegacyFlattenSensingEncoder,
    PROTOTYPE_MODES,
    PrototypeGuidedHierarchicalSensingEncoder,
)
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder


RESIDUAL_MODES = ("cross_attention", "residual_mlp")
SENSING_ARCHITECTURES = ("hierarchical", "flatten_mlp")


@dataclass(frozen=True)
class TSPCV2ModelConfig:
    """Validated architecture controls for the local V2 model."""

    feature_dim: int = 64
    history_length: int = 5
    num_modalities: int = 4
    num_beams: int = 64
    sensing_architecture: str = "hierarchical"
    prototype_mode: str = "shared_frozen"
    sensing_frame_layers: int = 2
    sensing_heads: int = 4
    sensing_temporal_method: str = "lstm"
    sensing_temporal_hidden_dim: int = 128
    sensing_temporal_layers: int = 2
    sensing_dropout: float = 0.1
    csi_dim: int = 128
    csi_num_candidate_patterns: int = 32
    csi_num_frequency_indices: int = 16
    csi_encoder_layers: int = 0
    csi_heads: int = 4
    csi_temporal_layers: int = 2
    csi_dropout: float = 0.1
    residual_mode: str = "cross_attention"
    residual_heads: int = 4
    residual_dropout: float = 0.1
    use_mask_context: bool = True
    use_sensing_context: bool = True
    use_csi_temporal_tokens: bool = True
    random_seed: int = 1

    def __post_init__(self) -> None:
        if self.feature_dim <= 0 or self.history_length <= 0 or self.num_modalities <= 0 or self.num_beams <= 0:
            raise ValueError("V2 dimensions must be positive.")
        if self.feature_dim % self.sensing_heads or self.feature_dim % self.residual_heads:
            raise ValueError("feature_dim must divide sensing_heads and residual_heads.")
        if self.csi_dim <= 0 or self.csi_dim % self.csi_heads:
            raise ValueError("csi_dim must be positive and divisible by csi_heads.")
        if self.sensing_architecture not in SENSING_ARCHITECTURES:
            raise ValueError(f"Unsupported sensing architecture: {self.sensing_architecture}.")
        if self.prototype_mode not in PROTOTYPE_MODES:
            raise ValueError(f"Unsupported prototype mode: {self.prototype_mode}.")
        if self.residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"Unsupported residual mode: {self.residual_mode}.")
        if self.sensing_temporal_layers <= 0 or self.csi_temporal_layers <= 0:
            raise ValueError("Temporal layer counts must be positive.")
        if not 0.0 <= self.sensing_dropout < 1.0 or not 0.0 <= self.csi_dropout < 1.0:
            raise ValueError("Dropout must be in [0,1).")


@dataclass(frozen=True)
class TSPCV2LossConfig:
    """Configurable V2 losses; labels always remain the task target."""

    final_ce_weight: float = 1.0
    sensing_ce_weight: float = 1.0
    prototype_weight: float = 0.2
    compensation_kl_weight: float = 0.0
    residual_regression_weight: float = 0.0
    distillation_temperature: float = 2.0

    def __post_init__(self) -> None:
        if min(
            self.final_ce_weight,
            self.sensing_ce_weight,
            self.prototype_weight,
            self.compensation_kl_weight,
            self.residual_regression_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative.")
        if self.distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive.")


class TemporalSparseCSIEncoder(nn.Module):
    """Encode complex pilot history [B,T,M,K] into temporal tokens and a summary."""

    def __init__(self, config: TSPCV2ModelConfig) -> None:
        super().__init__()
        self.history_length = int(config.history_length)
        self.hidden_dim = int(config.csi_dim)
        self.frame_encoder = SparsePilotEncoder(
            num_candidate_patterns=int(config.csi_num_candidate_patterns),
            hidden_dim=self.hidden_dim,
            num_heads=int(config.csi_heads),
            num_layers=int(config.csi_encoder_layers),
            dropout=float(config.csi_dropout),
            include_index_embeddings=True,
            num_frequency_indices=int(config.csi_num_frequency_indices),
            maximum_time_steps=self.history_length,
        )
        self.temporal = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=int(config.csi_temporal_layers),
            dropout=float(config.csi_dropout) if config.csi_temporal_layers > 1 else 0.0,
            batch_first=True,
        )

    @staticmethod
    def _expand_pattern_ids(pattern_ids: torch.Tensor, *, batch: int, history: int, patterns: int, device: torch.device) -> torch.Tensor:
        ids = torch.as_tensor(pattern_ids, device=device, dtype=torch.long)
        if tuple(ids.shape) == (batch, patterns):
            ids = ids[:, None, :].expand(-1, history, -1)
        if tuple(ids.shape) != (batch, history, patterns):
            raise ValueError("pattern_ids must have shape [B,M] or [B,T,M].")
        return ids

    @staticmethod
    def _expand_snr(snr_db: torch.Tensor | float, *, batch: int, history: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        snr = torch.as_tensor(snr_db, device=device, dtype=dtype)
        if snr.ndim == 0:
            snr = snr.expand(batch, history)
        elif tuple(snr.shape) == (batch,):
            snr = snr[:, None].expand(-1, history)
        if tuple(snr.shape) != (batch, history):
            raise ValueError("snr_db must be scalar, [B], or [B,T].")
        return snr

    @staticmethod
    def _flatten_frequency_values(
        values: torch.Tensor,
        *,
        batch: int,
        history: int,
        frequencies: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(values, device=device, dtype=dtype)
        if tuple(tensor.shape) == (frequencies,):
            return tensor
        if tuple(tensor.shape) == (batch, frequencies):
            return tensor[:, None, :].expand(-1, history, -1).reshape(batch * history, frequencies)
        if tuple(tensor.shape) == (batch, history, frequencies):
            return tensor.reshape(batch * history, frequencies)
        raise ValueError(f"{name} must have shape [K], [B,K], or [B,T,K].")

    def forward(
        self,
        pilot_observations: torch.Tensor,
        pattern_ids: torch.Tensor,
        frequency_positions: torch.Tensor,
        pilot_mask: torch.Tensor,
        snr_db: torch.Tensor | float,
        *,
        frequency_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(pilot_observations)
        if not torch.is_complex(values) or values.ndim != 4:
            raise ValueError("pilot_observations must be complex [B,T,M,K].")
        batch, history, patterns, frequencies = values.shape
        if history != self.history_length:
            raise ValueError(f"pilot history must have length {self.history_length}.")
        ids = self._expand_pattern_ids(pattern_ids, batch=batch, history=history, patterns=patterns, device=values.device)
        valid = torch.as_tensor(pilot_mask, device=values.device, dtype=torch.bool)
        if tuple(valid.shape) != tuple(values.shape):
            raise ValueError("pilot_mask must have shape [B,T,M,K].")
        snr = self._expand_snr(snr_db, batch=batch, history=history, dtype=values.real.dtype, device=values.device)
        positions = self._flatten_frequency_values(
            frequency_positions,
            batch=batch,
            history=history,
            frequencies=frequencies,
            device=values.device,
            dtype=values.real.dtype,
            name="frequency_positions",
        )
        if frequency_ids is None:
            frequency_ids = torch.arange(frequencies, device=values.device)
        index_values = self._flatten_frequency_values(
            frequency_ids,
            batch=batch,
            history=history,
            frequencies=frequencies,
            device=values.device,
            dtype=torch.long,
            name="frequency_ids",
        )
        time_ids = torch.arange(history, device=values.device).view(1, history).expand(batch, -1).reshape(-1)
        encoded = self.frame_encoder(
            values.reshape(batch * history, patterns, frequencies),
            ids.reshape(batch * history, patterns),
            positions,
            valid.reshape(batch * history, patterns, frequencies),
            snr.reshape(batch * history),
            frequency_ids=index_values,
            time_ids=time_ids,
        )
        frame_features = encoded["csi_feature"].reshape(batch, history, self.hidden_dim)
        frame_available = encoded["csi_available"].reshape(batch, history)
        layers = int(self.temporal.num_layers)
        hidden = frame_features.new_zeros(layers, batch, self.hidden_dim)
        cell = frame_features.new_zeros(layers, batch, self.hidden_dim)
        temporal_steps = []
        for time_index in range(history):
            _, (candidate_hidden, candidate_cell) = self.temporal(
                frame_features[:, time_index : time_index + 1],
                (hidden, cell),
            )
            update = frame_available[:, time_index][None, :, None]
            hidden = torch.where(update, candidate_hidden, hidden)
            cell = torch.where(update, candidate_cell, cell)
            temporal_steps.append(hidden[-1])
        temporal_tokens = torch.stack(temporal_steps, dim=1)
        csi_available = frame_available.any(dim=1)
        temporal_tokens = temporal_tokens * csi_available[:, None, None].to(temporal_tokens)
        z_csi = temporal_tokens[:, -1]
        return {
            "csi_frame_features": frame_features,
            "csi_temporal_tokens": temporal_tokens,
            "z_csi": z_csi,
            "csi_available": csi_available,
            "csi_frame_available": frame_available,
        }


class PrototypeConditionedResidualCompensator(nn.Module):
    """Use beam prototypes as queries over sensing, CSI, and missing context."""

    def __init__(self, config: TSPCV2ModelConfig) -> None:
        super().__init__()
        self.feature_dim = int(config.feature_dim)
        self.csi_dim = int(config.csi_dim)
        self.history_length = int(config.history_length)
        self.num_modalities = int(config.num_modalities)
        self.shared_frozen = config.prototype_mode == "shared_frozen"
        self.use_mask_context = bool(config.use_mask_context)
        self.use_sensing_context = bool(config.use_sensing_context)
        self.use_csi_temporal_tokens = bool(config.use_csi_temporal_tokens)
        self.sensing_summary_projection = nn.Linear(self.feature_dim, self.feature_dim)
        self.sensing_frame_projection = nn.Linear(self.feature_dim, self.feature_dim)
        self.csi_projection = nn.Linear(self.csi_dim, self.feature_dim)
        self.mask_projection = nn.Sequential(
            nn.Linear(self.history_length * self.num_modalities, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.prototype_projection = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        with torch.no_grad():
            self.prototype_projection.weight.copy_(torch.eye(self.feature_dim))
        self.learned_queries: nn.Parameter | None = None
        if config.prototype_mode == "no_prototype":
            self.learned_queries = nn.Parameter(torch.randn(64, self.feature_dim) * 0.02)
        self.context_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.feature_dim,
            int(config.residual_heads),
            dropout=float(config.residual_dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.feature_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(float(config.residual_dropout)),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )
        self.scalar_hidden = nn.Sequential(nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, self.feature_dim), nn.GELU())
        self.scalar_output = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.scalar_output.weight)
        nn.init.zeros_(self.scalar_output.bias)

    def _prototype_queries(self, prototype_bank: BeamPrototypeBank | None, batch: int, device: torch.device) -> torch.Tensor:
        if prototype_bank is None:
            if self.learned_queries is None:
                raise ValueError("Prototype-conditioned compensation requires a decision bank.")
            prototypes = self.learned_queries
        else:
            if prototype_bank.d_model != self.feature_dim or prototype_bank.num_beams != 64:
                raise ValueError("Prototype bank must have shape [64,feature_dim].")
            prototypes = prototype_bank.prototypes.detach() if self.shared_frozen else prototype_bank.prototypes
        with torch.autocast(device_type=device.type, enabled=False):
            normalized = F.normalize(prototypes.float().to(device), dim=-1)
            projected = self.prototype_projection(normalized)
            projected = F.normalize(projected, dim=-1)
        return projected[None].expand(batch, -1, -1)

    def forward(
        self,
        z_sensing: torch.Tensor,
        frame_features: torch.Tensor,
        csi_temporal_tokens: torch.Tensor,
        z_csi: torch.Tensor,
        availability: torch.Tensor,
        prototype_bank: BeamPrototypeBank | None,
    ) -> dict[str, torch.Tensor]:
        """Return delta evidence [B,64] from prototype-to-context cross-attention."""

        sensing = torch.as_tensor(z_sensing)
        frames = torch.as_tensor(frame_features, device=sensing.device)
        csi_tokens = torch.as_tensor(csi_temporal_tokens, device=sensing.device)
        csi = torch.as_tensor(z_csi, device=sensing.device)
        mask = torch.as_tensor(availability, device=sensing.device, dtype=torch.bool)
        batch = sensing.shape[0]
        if sensing.shape != (batch, self.feature_dim) or frames.shape != (
            batch,
            self.history_length,
            self.feature_dim,
        ):
            raise ValueError("Sensing context must be [B,D] and [B,T,D].")
        if csi_tokens.shape != (batch, self.history_length, self.csi_dim) or csi.shape != (batch, self.csi_dim):
            raise ValueError("CSI context must be [B,T,Dc] and [B,Dc].")
        if mask.shape != (batch, self.history_length, self.num_modalities):
            raise ValueError("availability must have shape [B,T,M].")

        with torch.autocast(device_type=sensing.device.type, enabled=False):
            context_parts = []
            if self.use_sensing_context:
                context_parts.extend(
                    (
                        self.sensing_summary_projection(sensing.float())[:, None],
                        self.sensing_frame_projection(frames.float()),
                    )
                )
            if self.use_csi_temporal_tokens:
                csi_context = self.csi_projection(csi_tokens.float())
            else:
                csi_context = self.csi_projection(csi[:, None].float())
            context_parts.append(csi_context)
            if self.use_mask_context:
                mask_token = self.mask_projection((~mask).float().flatten(1))[:, None]
                context_parts.append(mask_token)
            else:
                mask_token = sensing.float().new_zeros(batch, 1, self.feature_dim)
            context = torch.cat(context_parts, dim=1)
            query = self._prototype_queries(prototype_bank, batch, sensing.device)
            attended = self.cross_attention(
                self.query_norm(query),
                self.context_norm(context),
                self.context_norm(context),
                need_weights=False,
            )[0]
            prototype_context = self.output_norm(query + attended)
            prototype_context = prototype_context + self.output_mlp(prototype_context)
            delta = self.scalar_output(self.scalar_hidden(prototype_context)).squeeze(-1)
        return {
            "delta_evidence": delta,
            "prototype_context": prototype_context,
            "missing_pattern_token": mask_token.squeeze(1),
        }


class ResidualMLPCompensator(nn.Module):
    """B1 control without prototype queries."""

    def __init__(self, config: TSPCV2ModelConfig) -> None:
        super().__init__()
        self.feature_dim = int(config.feature_dim)
        self.csi_dim = int(config.csi_dim)
        self.history_length = int(config.history_length)
        self.num_modalities = int(config.num_modalities)
        self.use_mask_context = bool(config.use_mask_context)
        self.use_sensing_context = bool(config.use_sensing_context)
        self.sensing_projection = nn.Linear(self.feature_dim, self.feature_dim)
        self.csi_projection = nn.Linear(self.csi_dim, self.feature_dim)
        self.mask_projection = nn.Linear(self.history_length * self.num_modalities, self.feature_dim)
        self.hidden = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 3),
            nn.Linear(self.feature_dim * 3, self.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(float(config.residual_dropout)),
        )
        self.output = nn.Linear(self.feature_dim * 2, 64)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        z_sensing: torch.Tensor,
        frame_features: torch.Tensor,
        csi_temporal_tokens: torch.Tensor,
        z_csi: torch.Tensor,
        availability: torch.Tensor,
        prototype_bank: BeamPrototypeBank | None,
    ) -> dict[str, torch.Tensor]:
        del frame_features, csi_temporal_tokens, prototype_bank
        sensing = torch.as_tensor(z_sensing)
        csi = torch.as_tensor(z_csi, device=sensing.device)
        mask = torch.as_tensor(availability, device=sensing.device, dtype=torch.bool)
        batch = sensing.shape[0]
        if sensing.shape != (batch, self.feature_dim) or csi.shape != (batch, self.csi_dim):
            raise ValueError("Residual MLP inputs must be [B,D] and [B,Dc].")
        if mask.shape != (batch, self.history_length, self.num_modalities):
            raise ValueError("availability must have shape [B,T,M].")
        with torch.autocast(device_type=sensing.device.type, enabled=False):
            sensing_context = self.sensing_projection(sensing.float())
            if not self.use_sensing_context:
                sensing_context = torch.zeros_like(sensing_context)
            csi_context = self.csi_projection(csi.float())
            if self.use_mask_context:
                mask_context = self.mask_projection((~mask).float().flatten(1))
            else:
                mask_context = torch.zeros_like(sensing_context)
            delta = self.output(self.hidden(torch.cat((sensing_context, csi_context, mask_context), dim=-1)))
        return {
            "delta_evidence": delta,
            "prototype_context": sensing.float().new_zeros(batch, 64, self.feature_dim),
            "missing_pattern_token": mask_context,
        }


class TSPCV2Model(nn.Module):
    """V2 model with Full bypass and exact missing-sensing CSI-off fallback."""

    def __init__(self, config: TSPCV2ModelConfig) -> None:
        super().__init__()
        self.config = config
        sensing_config = HierarchicalSensingConfig(
            feature_dim=config.feature_dim,
            history_length=config.history_length,
            num_modalities=config.num_modalities,
            num_heads=config.sensing_heads,
            frame_layers=config.sensing_frame_layers,
            temporal_method=config.sensing_temporal_method,
            temporal_hidden_dim=config.sensing_temporal_hidden_dim,
            temporal_layers=config.sensing_temporal_layers,
            dropout=config.sensing_dropout,
            prototype_mode=config.prototype_mode,
            random_seed=config.random_seed,
        )
        if config.sensing_architecture == "hierarchical":
            self.sensing: nn.Module = PrototypeGuidedHierarchicalSensingEncoder(sensing_config)
        else:
            self.sensing = LegacyFlattenSensingEncoder(sensing_config)
        self.csi_encoder = TemporalSparseCSIEncoder(config)
        self.residual: nn.Module
        if config.residual_mode == "cross_attention":
            self.residual = PrototypeConditionedResidualCompensator(config)
        else:
            self.residual = ResidualMLPCompensator(config)

    @staticmethod
    def _availability(
        availability_mask: torch.Tensor,
        *,
        batch: int,
        history: int,
        modalities: int,
        device: torch.device,
    ) -> torch.Tensor:
        availability = torch.as_tensor(availability_mask, device=device, dtype=torch.bool)
        if tuple(availability.shape) == (batch, modalities):
            availability = availability[:, None, :].expand(-1, history, -1)
        if tuple(availability.shape) != (batch, history, modalities):
            raise ValueError(f"availability_mask must have shape [B,{modalities}] or [B,{history},{modalities}].")
        if not bool(availability.any(dim=(1, 2)).all()):
            raise ValueError("Each sample must retain at least one sensing slot.")
        return availability

    @staticmethod
    def _slice_batch(
        value: torch.Tensor | float | int | None,
        rows: torch.Tensor,
        batch: int,
        *,
        preserve_shared_vector: bool = False,
    ):
        if value is None:
            return None
        tensor = torch.as_tensor(value)
        if tensor.ndim and tensor.shape[0] == batch and not (preserve_shared_vector and tensor.ndim == 1):
            return tensor.index_select(0, rows)
        return tensor

    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
        if not enabled:
            module.eval()

    def configure_stage(self, stage: str) -> None:
        """Configure the documented Stage A/B/C trainable modules."""

        normalized = str(stage).strip().lower()
        aliases = {"a": "stage_a", "b": "stage_b", "c": "stage_c"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"stage_a", "stage_b", "stage_c"}:
            raise ValueError("stage must be stage_a, stage_b, or stage_c.")
        self._set_trainable(self.sensing, normalized in {"stage_a", "stage_c"})
        self._set_trainable(self.csi_encoder, normalized in {"stage_b", "stage_c"})
        self._set_trainable(self.residual, normalized in {"stage_b", "stage_c"})
        if self.config.prototype_mode == "random_frozen":
            head = getattr(self.sensing, "evidence_head")
            assert head.prototype_bank is not None
            head.prototype_bank.prototypes.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in (self.sensing, self.csi_encoder, self.residual):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
        return self

    def prototype_bank_for_loss(self, shared_prototype_bank: BeamPrototypeBank | None) -> BeamPrototypeBank | None:
        return self.sensing.decision_bank(shared_prototype_bank)

    def forward(
        self,
        sensing_features: torch.Tensor,
        availability_mask: torch.Tensor,
        *,
        shared_prototype_bank: BeamPrototypeBank | None = None,
        pilot_observations: torch.Tensor | None = None,
        pattern_ids: torch.Tensor | None = None,
        frequency_positions: torch.Tensor | None = None,
        pilot_mask: torch.Tensor | None = None,
        snr_db: torch.Tensor | float | None = None,
        frequency_ids: torch.Tensor | None = None,
        full_probability: torch.Tensor | None = None,
        full_evidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward V2 without using pilots for Full rows."""

        features = torch.as_tensor(sensing_features)
        if features.ndim != 4:
            raise ValueError("sensing_features must have shape [B,T,M,D].")
        batch, history, modalities, feature_dim = features.shape
        if (history, modalities, feature_dim) != (
            self.config.history_length,
            self.config.num_modalities,
            self.config.feature_dim,
        ):
            raise ValueError(
                "sensing_features must have shape "
                f"[B,{self.config.history_length},{self.config.num_modalities},{self.config.feature_dim}]."
            )
        availability = self._availability(
            availability_mask,
            batch=batch,
            history=history,
            modalities=modalities,
            device=features.device,
        )
        full_rows = availability.all(dim=(1, 2))
        missing_rows = (~full_rows).nonzero(as_tuple=False).squeeze(1)
        evidence = features.float().new_zeros(batch, self.config.num_beams)
        frame_features = features.float().new_zeros(batch, history, feature_dim)
        z_sensing = features.float().new_zeros(batch, feature_dim)
        delta = features.float().new_zeros(batch, self.config.num_beams)
        csi_frame_features = features.float().new_zeros(batch, history, self.config.csi_dim)
        csi_temporal_tokens = features.float().new_zeros(batch, history, self.config.csi_dim)
        z_csi = features.float().new_zeros(batch, self.config.csi_dim)
        prototype_context = features.float().new_zeros(batch, self.config.num_beams, feature_dim)
        missing_pattern_token = features.float().new_zeros(batch, feature_dim)
        csi_available = torch.zeros(batch, dtype=torch.bool, device=features.device)
        csi_frame_available = torch.zeros(batch, history, dtype=torch.bool, device=features.device)
        pilot_re_per_frame = torch.zeros(batch, history, dtype=torch.long, device=features.device)

        if missing_rows.numel():
            sensing_output = self.sensing(
                features.index_select(0, missing_rows),
                availability.index_select(0, missing_rows),
                shared_prototype_bank,
            )
            evidence.index_copy_(0, missing_rows, sensing_output["sensing_evidence"])
            frame_features.index_copy_(0, missing_rows, sensing_output["frame_features"].float())
            z_sensing.index_copy_(0, missing_rows, sensing_output["z_sensing"].float())
            active_local_rows = missing_rows.new_empty(0)
            if pilot_mask is not None:
                selected_pilot_mask = torch.as_tensor(
                    self._slice_batch(pilot_mask, missing_rows, batch),
                    device=features.device,
                    dtype=torch.bool,
                )
                if selected_pilot_mask.ndim != 4 or selected_pilot_mask.shape[:2] != (len(missing_rows), history):
                    raise ValueError("pilot_mask must have shape [B,T,M,K].")
                active_local_rows = selected_pilot_mask.any(dim=(1, 2, 3)).nonzero(as_tuple=False).squeeze(1)
            if active_local_rows.numel():
                required = {
                    "pilot_observations": pilot_observations,
                    "pattern_ids": pattern_ids,
                    "frequency_positions": frequency_positions,
                    "snr_db": snr_db,
                }
                absent = [name for name, value in required.items() if value is None]
                if absent:
                    raise ValueError(f"Missing CSI inputs for active non-Full rows: {absent}.")
                active_rows = missing_rows.index_select(0, active_local_rows)
                radio_output = self.csi_encoder(
                    self._slice_batch(pilot_observations, active_rows, batch),
                    self._slice_batch(pattern_ids, active_rows, batch),
                    self._slice_batch(frequency_positions, active_rows, batch, preserve_shared_vector=True),
                    self._slice_batch(pilot_mask, active_rows, batch),
                    self._slice_batch(snr_db, active_rows, batch),
                    frequency_ids=self._slice_batch(frequency_ids, active_rows, batch, preserve_shared_vector=True),
                )
                csi_frame_features.index_copy_(0, active_rows, radio_output["csi_frame_features"].float())
                csi_temporal_tokens.index_copy_(0, active_rows, radio_output["csi_temporal_tokens"].float())
                z_csi.index_copy_(0, active_rows, radio_output["z_csi"].float())
                csi_available.index_copy_(0, active_rows, radio_output["csi_available"])
                csi_frame_available.index_copy_(0, active_rows, radio_output["csi_frame_available"])
                active_pilot_mask = torch.as_tensor(
                    self._slice_batch(pilot_mask, active_rows, batch),
                    device=features.device,
                    dtype=torch.bool,
                )
                pilot_re_per_frame.index_copy_(0, active_rows, active_pilot_mask.sum(dim=(-2, -1)).long())
                prototype_bank = self.prototype_bank_for_loss(shared_prototype_bank)
                residual_output = self.residual(
                    sensing_output["z_sensing"].index_select(0, active_local_rows),
                    sensing_output["frame_features"].index_select(0, active_local_rows),
                    radio_output["csi_temporal_tokens"],
                    radio_output["z_csi"],
                    sensing_output["availability"].index_select(0, active_local_rows),
                    prototype_bank,
                )
                prototype_context.index_copy_(0, active_rows, residual_output["prototype_context"].float())
                missing_pattern_token.index_copy_(0, active_rows, residual_output["missing_pattern_token"].float())
                subset_delta = torch.where(
                    radio_output["csi_available"][:, None],
                    residual_output["delta_evidence"],
                    torch.zeros_like(residual_output["delta_evidence"]),
                )
                delta.index_copy_(0, active_rows, subset_delta)

        if bool(full_rows.any()):
            if full_probability is None:
                raise ValueError("full_probability is required when a Full row is present.")
            full_probability_value = torch.as_tensor(full_probability, device=features.device, dtype=features.float().dtype)
            if full_probability_value.shape != (batch, self.config.num_beams):
                raise ValueError(f"full_probability must have shape [B,{self.config.num_beams}].")
            if full_evidence is None:
                full_evidence_value = full_probability_value.clamp_min(torch.finfo(full_probability_value.dtype).tiny).log()
            else:
                full_evidence_value = torch.as_tensor(full_evidence, device=features.device, dtype=features.float().dtype)
                if full_evidence_value.shape != (batch, self.config.num_beams):
                    raise ValueError(f"full_evidence must have shape [B,{self.config.num_beams}].")
            evidence.index_copy_(0, full_rows.nonzero(as_tuple=False).squeeze(1), full_evidence_value[full_rows])

        # Exact CSI-off fallback selects the sensing tensor rather than adding zero.
        final_evidence = evidence.clone()
        active_rows = (~full_rows & csi_available).nonzero(as_tuple=False).squeeze(1)
        if active_rows.numel():
            final_evidence.index_copy_(0, active_rows, evidence.index_select(0, active_rows) + delta.index_select(0, active_rows))
        final_probability = torch.softmax(final_evidence.float(), dim=-1)
        if bool(full_rows.any()):
            assert full_probability is not None
            final_probability.index_copy_(
                0,
                full_rows.nonzero(as_tuple=False).squeeze(1),
                torch.as_tensor(full_probability, device=features.device, dtype=final_probability.dtype)[full_rows],
            )
        return {
            "frame_features": frame_features,
            "z_sensing": z_sensing,
            "sensing_evidence": evidence,
            "csi_frame_features": csi_frame_features,
            "csi_temporal_tokens": csi_temporal_tokens,
            "z_csi": z_csi,
            "prototype_context": prototype_context,
            "missing_pattern_token": missing_pattern_token,
            "delta_evidence": delta,
            "final_evidence": final_evidence,
            "final_probability": final_probability,
            "availability": availability,
            "full_bypass": full_rows,
            "csi_available": csi_available,
            "csi_frame_available": csi_frame_available,
            "pilot_re_per_frame": pilot_re_per_frame,
            "pilot_re_window": pilot_re_per_frame.sum(dim=1),
        }


def tspc_v2_losses(
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    config: TSPCV2LossConfig,
    prototype_bank: BeamPrototypeBank | None = None,
    teacher_probability: torch.Tensor | None = None,
    teacher_evidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute V2 losses without feeding labels or teacher values into forward."""

    final_evidence = torch.as_tensor(output["final_evidence"])
    sensing_evidence = torch.as_tensor(output["sensing_evidence"], device=final_evidence.device)
    target = torch.as_tensor(labels, device=final_evidence.device, dtype=torch.long).reshape(-1)
    full = torch.as_tensor(output["full_bypass"], device=final_evidence.device, dtype=torch.bool).reshape(-1)
    csi_available = torch.as_tensor(output["csi_available"], device=final_evidence.device, dtype=torch.bool).reshape(-1)
    if final_evidence.shape != sensing_evidence.shape or final_evidence.shape != (target.numel(), 64):
        raise ValueError("V2 evidence and labels must have shapes [B,64] and [B].")
    missing = ~full
    compensated = missing & csi_available
    zero = final_evidence.sum() * 0.0
    final_ce = F.cross_entropy(final_evidence[compensated], target[compensated]) if bool(compensated.any()) else zero
    sensing_ce = F.cross_entropy(sensing_evidence[missing], target[missing]) if bool(missing.any()) else zero
    prototype = zero
    if prototype_bank is not None and bool(missing.any()):
        prototype, _ = prototype_alignment_loss(
            prototype_bank,
            target[missing],
            fused_features=torch.as_tensor(output["z_sensing"], device=final_evidence.device)[missing],
            lambda_proto=1.0,
            lambda_modality_proto=0.0,
            beam_label_sigma=2.0,
            topology_id="ula_dft_phase_cycle_v1",
        )
    kl = zero
    if config.compensation_kl_weight and teacher_probability is None:
        raise ValueError("compensation_kl_weight requires a stop-gradient teacher_probability.")
    if config.compensation_kl_weight and bool(compensated.any()):
        assert teacher_probability is not None
        teacher = torch.as_tensor(teacher_probability, device=final_evidence.device, dtype=torch.float32)
        if teacher.shape != final_evidence.shape:
            raise ValueError("teacher_probability must have shape [B,64].")
        temperature = float(config.distillation_temperature)
        teacher_at_temperature = torch.softmax(teacher.detach().clamp_min(1e-12).log() / temperature, dim=-1)
        kl = F.kl_div(
            F.log_softmax(final_evidence[compensated].float() / temperature, dim=-1),
            teacher_at_temperature[compensated],
            reduction="batchmean",
        ) * temperature**2
    residual = zero
    if config.residual_regression_weight and teacher_evidence is None:
        raise ValueError("residual_regression_weight requires stop-gradient teacher_evidence.")
    if config.residual_regression_weight and bool(compensated.any()):
        assert teacher_evidence is not None
        teacher = torch.as_tensor(teacher_evidence, device=final_evidence.device, dtype=torch.float32)
        if teacher.shape != final_evidence.shape:
            raise ValueError("teacher_evidence must have shape [B,64].")
        teacher = teacher.detach() - teacher.detach().mean(dim=-1, keepdim=True)
        sensing = sensing_evidence.float() - sensing_evidence.float().mean(dim=-1, keepdim=True)
        target_delta = teacher[compensated] - sensing[compensated]
        predicted_delta = output["delta_evidence"].float()
        predicted_delta = predicted_delta - predicted_delta.mean(dim=-1, keepdim=True)
        residual = F.smooth_l1_loss(predicted_delta[compensated], target_delta)
    total = (
        float(config.final_ce_weight) * final_ce
        + float(config.sensing_ce_weight) * sensing_ce
        + float(config.prototype_weight) * prototype
        + float(config.compensation_kl_weight) * kl
        + float(config.residual_regression_weight) * residual
    )
    return {
        "loss_total": total,
        "loss_final_ce": final_ce,
        "loss_sensing_ce": sensing_ce,
        "loss_prototype": prototype,
        "loss_compensation_kl": kl,
        "loss_residual_regression": residual,
        "loss_full_consistency": zero,
    }


__all__ = [
    "PrototypeConditionedResidualCompensator",
    "RESIDUAL_MODES",
    "ResidualMLPCompensator",
    "SENSING_ARCHITECTURES",
    "TSPCV2LossConfig",
    "TSPCV2Model",
    "TSPCV2ModelConfig",
    "TemporalSparseCSIEncoder",
    "tspc_v2_losses",
]
