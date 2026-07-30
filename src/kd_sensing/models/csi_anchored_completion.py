"""CSI-anchored beam-semantic completion for missing sensing slots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.models.available_modality_context import AvailableModalityContext
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder


class SparsePilotRadioEncoder(nn.Module):
    """The validated pilot encoder and GRU, deliberately without a beam head."""

    def __init__(
        self,
        *,
        history_length: int = 5,
        hidden_dim: int = 128,
        num_candidate_patterns: int = 32,
        encoder_layers: int = 0,
        quality_dim: int = 16,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.hidden_dim = int(hidden_dim)
        self.frame_quality_dim = int(quality_dim) + 4
        self.quality_dim = self.frame_quality_dim + 1
        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")
        self.csi_encoder = SparsePilotEncoder(
            num_candidate_patterns=int(num_candidate_patterns),
            hidden_dim=self.hidden_dim,
            num_layers=int(encoder_layers),
            quality_dim=int(quality_dim),
        )
        self.temporal = (
            nn.GRU(
                self.hidden_dim,
                self.hidden_dim,
                num_layers=2,
                dropout=0.1,
                batch_first=True,
            )
            if self.history_length > 1
            else None
        )

    def forward(
        self,
        pilot_observations: torch.Tensor,
        pattern_ids: torch.Tensor,
        frequency_positions: torch.Tensor,
        pilot_mask: torch.Tensor,
        snr_db: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        observations = torch.as_tensor(pilot_observations)
        if observations.ndim == 3:
            observations = observations[:, None]
        if not torch.is_complex(observations) or observations.ndim != 4:
            raise ValueError("pilot_observations must be complex [B,T,M,K].")
        batch, history, patterns, frequencies = observations.shape
        if history != self.history_length:
            raise ValueError("pilot history does not match the configured history_length.")

        ids = torch.as_tensor(pattern_ids, device=observations.device, dtype=torch.long)
        if ids.ndim == 2:
            ids = ids[:, None].expand(-1, history, -1)
        valid = torch.as_tensor(pilot_mask, device=observations.device, dtype=torch.bool)
        if valid.ndim == 3:
            valid = valid[:, None].expand(-1, history, -1, -1)
        snr = torch.as_tensor(snr_db, device=observations.device, dtype=observations.real.dtype)
        if snr.ndim == 0:
            snr = snr.expand(batch, history)
        elif snr.ndim == 1:
            snr = snr[:, None].expand(-1, history)
        if tuple(ids.shape) != (batch, history, patterns) or tuple(valid.shape) != tuple(observations.shape):
            raise ValueError("pattern_ids and pilot_mask must match [B,T,M,K].")
        if tuple(snr.shape) != (batch, history):
            raise ValueError("snr_db must be scalar, [B], or [B,T].")

        positions = torch.as_tensor(
            frequency_positions,
            device=observations.device,
            dtype=observations.real.dtype,
        )
        if positions.ndim == 2:
            if tuple(positions.shape) != (batch, frequencies):
                raise ValueError("batched frequency_positions must be [B,K].")
            positions = positions[:, None].expand(-1, history, -1).reshape(batch * history, frequencies)
        elif positions.ndim != 1 or positions.numel() != frequencies:
            raise ValueError("frequency_positions must be [K] or [B,K].")

        encoded = self.csi_encoder(
            observations.reshape(batch * history, patterns, frequencies),
            ids.reshape(batch * history, patterns),
            positions,
            valid.reshape(batch * history, patterns, frequencies),
            snr.reshape(batch * history),
        )
        frame_features = encoded["csi_feature"].reshape(batch, history, self.hidden_dim)
        frame_available = encoded["csi_available"].reshape(batch, history)
        if self.temporal is None:
            radio = frame_features[:, -1]
        else:
            radio = self.temporal(frame_features)[0][:, -1]
        radio_available = frame_available.any(dim=1)
        radio = radio * radio_available[:, None].to(radio)

        frame_quality = encoded["csi_quality"].reshape(batch, history, self.frame_quality_dim)
        quality_weight = frame_available.to(frame_quality.dtype)
        mean_quality = (frame_quality * quality_weight[:, :, None]).sum(dim=1)
        mean_quality = mean_quality / quality_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        if history > 1:
            consistency = F.cosine_similarity(frame_features[:, 1:], frame_features[:, :-1], dim=-1)
            pair_valid = frame_available[:, 1:] & frame_available[:, :-1]
            consistency = ((consistency + 1.0) * 0.5).clamp(0.0, 1.0)
            consistency = (consistency * pair_valid).sum(dim=1) / pair_valid.sum(dim=1).clamp_min(1)
        else:
            consistency = radio_available.to(frame_quality.dtype)
        quality = torch.cat((mean_quality, consistency[:, None]), dim=-1)
        quality = quality * radio_available[:, None].to(quality)
        return {
            "c_radio": radio,
            "frame_csi_features": frame_features,
            "csi_quality": quality,
            "temporal_consistency": consistency,
            "csi_available": radio_available,
        }

    def load_information_checkpoint(self, checkpoint: str | Path) -> dict[str, Any]:
        """Load only CSI encoder and GRU weights from the successful classifier."""
        source = Path(checkpoint)
        payload = torch.load(source, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload)
        if not isinstance(state, Mapping):
            raise ValueError("CSI checkpoint does not contain a model state mapping.")
        allowed = ("csi_encoder.", "temporal.")
        filtered = {key: value for key, value in state.items() if key.startswith(allowed)}
        expected = set(self.state_dict())
        missing = sorted(expected - set(filtered))
        incompatible = sorted(
            key for key, value in filtered.items() if key in expected and self.state_dict()[key].shape != value.shape
        )
        if missing or incompatible:
            raise ValueError(f"CSI checkpoint is incompatible: missing={missing}, incompatible={incompatible}.")
        self.load_state_dict({key: filtered[key] for key in expected}, strict=True)
        return {
            "checkpoint": str(source.resolve()),
            "loaded_parameter_tensors": len(expected),
            "classifier_loaded": False,
        }

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update, weights = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + self.attention_dropout(update)
        query = query + self.ffn(self.ffn_norm(query))
        return query, weights


class RadioReliability(nn.Module):
    """Monotone main effects plus a bounded sensing/radio agreement term."""

    def __init__(self, num_beams: int = 64) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.raw_severity_weight = nn.Parameter(torch.tensor(0.5))
        self.raw_quality_weight = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(-0.5))
        self.context = nn.Linear(2, 1)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)

    def forward(
        self,
        quality: torch.Tensor,
        availability: torch.Tensor,
        sensing_probability: torch.Tensor,
        radio_probability: torch.Tensor,
        csi_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if quality.ndim != 2 or quality.shape[1] < 5:
            raise ValueError("quality must contain learned values plus SNR/valid/RMS/confidence/consistency.")
        eps = torch.finfo(sensing_probability.dtype).tiny
        entropy = -(sensing_probability * sensing_probability.clamp_min(eps).log()).sum(dim=-1)
        entropy = entropy / math.log(self.num_beams)
        midpoint = 0.5 * (sensing_probability + radio_probability)
        disagreement = 0.5 * (
            (sensing_probability * (sensing_probability.clamp_min(eps) / midpoint.clamp_min(eps)).log()).sum(dim=-1)
            + (radio_probability * (radio_probability.clamp_min(eps) / midpoint.clamp_min(eps)).log()).sum(dim=-1)
        )
        severity = 1.0 - availability.to(quality.dtype).mean(dim=1)
        snr_score = torch.sigmoid(3.0 * quality[:, -5])
        valid_ratio = quality[:, -4].clamp(0.0, 1.0)
        rms_score = torch.sigmoid(quality[:, -3])
        confidence = quality[:, -2].clamp(0.0, 1.0)
        consistency = quality[:, -1].clamp(0.0, 1.0)
        quality_score = torch.stack((snr_score, valid_ratio, rms_score, confidence, consistency), dim=-1).mean(dim=-1)
        context = 2.0 * torch.tanh(self.context(torch.stack((1.0 - entropy, 1.0 - disagreement), dim=-1))).squeeze(-1)
        logit = (
            self.bias
            + F.softplus(self.raw_severity_weight) * severity
            + F.softplus(self.raw_quality_weight) * quality_score
            + context
        )
        rho = torch.sigmoid(logit) * csi_available.to(logit.dtype)
        return {
            "rho": rho,
            "severity": severity,
            "quality_score": quality_score,
            "sensing_entropy": entropy,
            "prototype_disagreement": disagreement,
        }


class CSIAnchoredPrototypeCompletion(nn.Module):
    """Generate only missing sensing slots from available sensing and CSI evidence."""

    def __init__(
        self,
        *,
        feature_dim: int = 64,
        radio_dim: int = 128,
        quality_dim: int = 21,
        hidden_dim: int = 128,
        num_modalities: int = 4,
        num_beams: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        top_k: int = 8,
        sensing_temperature: float = 0.1,
        radio_temperature: float = 0.1,
        use_radio: bool = True,
        use_prototype_memory: bool = True,
        use_cross_attention: bool = True,
        evidence_fusion: str = "poe",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.radio_dim = int(radio_dim)
        self.quality_dim = int(quality_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_modalities = int(num_modalities)
        self.num_beams = int(num_beams)
        self.top_k = min(int(top_k), self.num_beams)
        self.sensing_temperature = float(sensing_temperature)
        self.radio_temperature = float(radio_temperature)
        self.use_radio = bool(use_radio)
        self.use_prototype_memory = bool(use_prototype_memory)
        self.use_cross_attention = bool(use_cross_attention)
        self.evidence_fusion = str(evidence_fusion)
        if self.evidence_fusion not in {"poe", "arithmetic", "sensing", "radio"}:
            raise ValueError("evidence_fusion must be poe, arithmetic, sensing, or radio.")
        if min(self.top_k, self.sensing_temperature, self.radio_temperature) <= 0:
            raise ValueError("top_k and evidence temperatures must be positive.")

        self.available_context = AvailableModalityContext(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_modalities=self.num_modalities,
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.radio_projection = nn.Linear(self.radio_dim, self.hidden_dim)
        self.quality_projection = nn.Linear(self.quality_dim, self.hidden_dim)
        self.prototype_projection = (
            nn.Linear(self.feature_dim, self.hidden_dim) if self.use_prototype_memory else None
        )
        self.sensing_query_projection = (
            nn.Linear(self.hidden_dim, self.hidden_dim) if self.use_prototype_memory else None
        )
        self.radio_query_projection = (
            nn.Linear(self.radio_dim, self.hidden_dim) if self.use_prototype_memory else None
        )
        self.missing_query = nn.Parameter(torch.randn(self.num_modalities, self.hidden_dim) * 0.02)
        self.query_modality_embedding = nn.Parameter(torch.randn(self.num_modalities, self.hidden_dim) * 0.02)
        self.severity_embedding = nn.Embedding(self.num_modalities, self.hidden_dim)
        self.mask_embedding = nn.Embedding(2**self.num_modalities, self.hidden_dim)
        self.query_context_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder = nn.ModuleList(
            _CrossAttentionBlock(self.hidden_dim, int(num_heads), int(ffn_dim), float(dropout))
            for _ in range(int(num_layers))
        )
        self.output_heads = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.feature_dim) for _ in range(self.num_modalities)
        )
        self.no_prototype_heads = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.feature_dim) for _ in range(self.num_modalities)
        )
        self.output_norms = nn.ModuleList(nn.LayerNorm(self.feature_dim) for _ in range(self.num_modalities))
        self.gamma_logit = nn.Parameter(torch.full((self.num_modalities,), math.log(0.4 / 0.6)))
        self.reliability = RadioReliability(self.num_beams)

    def _decode(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_cross_attention:
            return query.squeeze(1), query.new_zeros(query.shape[0], 0, context.shape[1])
        layer_weights = []
        for layer in self.decoder:
            query, weights = layer(query, context, padding_mask)
            layer_weights.append(weights.mean(dim=1).squeeze(1))
        return query.squeeze(1), torch.stack(layer_weights, dim=1)

    def _candidate(
        self,
        modality: int,
        hidden: torch.Tensor,
        prototype_probability: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_prototype_memory:
            return self.output_norms[modality](self.no_prototype_heads[modality](hidden))
        prototype_base = prototype_probability @ prototypes
        gamma = 0.25 * torch.sigmoid(self.gamma_logit[modality])
        residual = gamma * torch.tanh(self.output_heads[modality](hidden))
        return self.output_norms[modality](prototype_base + residual)

    def forward(
        self,
        modality_tokens: torch.Tensor,
        availability: torch.Tensor,
        c_radio: torch.Tensor,
        csi_quality: torch.Tensor,
        csi_available: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(modality_tokens)
        physical = torch.as_tensor(availability, device=values.device, dtype=torch.bool)
        radio = torch.as_tensor(c_radio, device=values.device, dtype=values.dtype)
        quality = torch.as_tensor(csi_quality, device=values.device, dtype=values.dtype)
        radio_available = torch.as_tensor(csi_available, device=values.device, dtype=torch.bool).reshape(-1)
        prototype_values = torch.as_tensor(prototypes, device=values.device, dtype=values.dtype).detach()
        batch = values.shape[0]
        if tuple(values.shape) != (batch, self.num_modalities, self.feature_dim):
            raise ValueError("modality_tokens must be [B,num_modalities,feature_dim].")
        if tuple(physical.shape) != (batch, self.num_modalities) or not bool(physical.any(dim=1).all()):
            raise ValueError("availability must be a non-empty [B,num_modalities] mask.")
        if bool(physical.all()):
            raise ValueError("Full samples must bypass the completion module.")
        if tuple(radio.shape) != (batch, self.radio_dim) or tuple(quality.shape) != (batch, self.quality_dim):
            raise ValueError("radio and quality features have incompatible shapes.")
        if tuple(radio_available.shape) != (batch,):
            raise ValueError("csi_available must have shape [B].")
        if tuple(prototype_values.shape) != (self.num_beams, self.feature_dim):
            raise ValueError("prototypes must have shape [num_beams,feature_dim].")

        context = self.available_context(values, physical)
        if self.use_prototype_memory:
            assert self.prototype_projection is not None
            assert self.sensing_query_projection is not None
            assert self.radio_query_projection is not None
            prototype_memory = self.prototype_projection(prototype_values)
            sensing_query = F.normalize(self.sensing_query_projection(context["z_available"]), dim=-1)
            normalized_memory = F.normalize(prototype_memory, dim=-1)
            sensing_logits = sensing_query @ normalized_memory.t() / self.sensing_temperature
            sensing_probability = torch.softmax(sensing_logits, dim=-1)
            radio_logits = F.normalize(self.radio_query_projection(radio), dim=-1) @ normalized_memory.t()
            radio_probability = torch.softmax(radio_logits / self.radio_temperature, dim=-1)
        else:
            prototype_memory = values.new_zeros(self.num_beams, self.hidden_dim)
            sensing_probability = values.new_full((batch, self.num_beams), 1.0 / self.num_beams)
            radio_probability = sensing_probability
        reliability = self.reliability(
            quality,
            physical,
            sensing_probability,
            radio_probability,
            radio_available,
        )
        rho = reliability["rho"] if self.use_radio else reliability["rho"].new_zeros(batch)
        eps = torch.finfo(sensing_probability.dtype).tiny
        if self.evidence_fusion == "poe":
            anchor_logits = (
                (1.0 - rho[:, None]) * sensing_probability.clamp_min(eps).log()
                + rho[:, None] * radio_probability.clamp_min(eps).log()
            )
            anchor_probability = torch.softmax(anchor_logits, dim=-1)
        elif self.evidence_fusion == "arithmetic":
            anchor_probability = (1.0 - rho[:, None]) * sensing_probability + rho[:, None] * radio_probability
        elif self.evidence_fusion == "radio":
            anchor_probability = torch.where(radio_available[:, None], radio_probability, sensing_probability)
        else:
            anchor_probability = sensing_probability

        sensing_top = sensing_probability.topk(self.top_k, dim=-1).indices
        anchor_top = anchor_probability.topk(self.top_k, dim=-1).indices
        sensing_prototypes = prototype_memory[sensing_top]
        anchor_prototypes = prototype_memory[anchor_top]
        available_tokens = context["available_tokens"]
        available_padding = ~physical
        if self.use_prototype_memory:
            base_context = torch.cat((available_tokens, sensing_prototypes), dim=1)
            base_padding = torch.cat(
                (available_padding, torch.zeros(batch, self.top_k, dtype=torch.bool, device=values.device)),
                dim=1,
            )
        else:
            base_context = available_tokens
            base_padding = available_padding

        radio_token = self.radio_projection(radio)[:, None]
        quality_token = self.quality_projection(quality)[:, None]
        radio_padding = ~radio_available[:, None]
        radio_parts = (available_tokens, radio_token, quality_token)
        radio_masks = (available_padding, radio_padding, radio_padding)
        if self.use_prototype_memory:
            radio_parts += (anchor_prototypes,)
            radio_masks += (torch.zeros(batch, self.top_k, dtype=torch.bool, device=values.device),)
        anchored_context = torch.cat(radio_parts, dim=1)
        anchored_padding = torch.cat(radio_masks, dim=1)

        mask_weights = 2 ** torch.arange(self.num_modalities, device=values.device, dtype=torch.long)
        mask_id = (physical.to(torch.long) * mask_weights).sum(dim=1)
        severity_id = (~physical).sum(dim=1).clamp(1, self.num_modalities).to(torch.long) - 1
        completed_slots = []
        distributions = values.new_zeros(batch, self.num_modalities, self.num_beams)
        base_distributions = values.new_zeros(batch, self.num_modalities, self.num_beams)
        slot_rho = values.new_zeros(batch, self.num_modalities)
        query_active = ~physical
        base_attention = values.new_zeros(batch, self.num_modalities, 2)
        radio_attention = values.new_zeros(batch, self.num_modalities, 4)
        gamma = 0.25 * torch.sigmoid(self.gamma_logit)

        for modality in range(self.num_modalities):
            rows = (~physical[:, modality]).nonzero(as_tuple=False).squeeze(1)
            if not rows.numel():
                completed_slots.append(values[:, modality])
                continue
            query = (
                self.missing_query[modality]
                + self.query_modality_embedding[modality]
                + self.severity_embedding(severity_id[rows])
                + self.mask_embedding(mask_id[rows])
                + self.query_context_projection(context["z_available"][rows])
            )[:, None]
            base_hidden, base_weights = self._decode(query, base_context[rows], base_padding[rows])
            base_candidate = self._candidate(
                modality,
                base_hidden,
                sensing_probability[rows],
                prototype_values,
            )
            if self.use_radio and bool(radio_available[rows].any()):
                anchored_hidden, anchored_weights = self._decode(
                    query,
                    anchored_context[rows],
                    anchored_padding[rows],
                )
                radio_candidate = self._candidate(
                    modality,
                    anchored_hidden,
                    anchor_probability[rows],
                    prototype_values,
                )
            else:
                anchored_weights = values.new_zeros(rows.numel(), 0, anchored_context.shape[1])
                radio_candidate = base_candidate
            candidate = (1.0 - rho[rows, None]) * base_candidate + rho[rows, None] * radio_candidate
            slot = values[:, modality].index_copy(0, rows, candidate)
            completed_slots.append(slot)
            distributions[:, modality] = distributions[:, modality].index_copy(
                0, rows, anchor_probability[rows]
            )
            base_distributions[:, modality] = base_distributions[:, modality].index_copy(
                0, rows, sensing_probability[rows]
            )
            slot_rho[:, modality] = slot_rho[:, modality].index_copy(0, rows, rho[rows])
            if base_weights.numel():
                averaged = base_weights.mean(dim=1)
                available_mass = averaged[:, : self.num_modalities].sum(dim=-1)
                prototype_mass = averaged[:, self.num_modalities :].sum(dim=-1)
                masses = torch.stack((available_mass, prototype_mass), dim=-1)
                base_attention[:, modality] = base_attention[:, modality].index_copy(0, rows, masses)
            if anchored_weights.numel():
                averaged = anchored_weights.mean(dim=1)
                start = self.num_modalities
                masses = torch.stack(
                    (
                        averaged[:, :start].sum(dim=-1),
                        averaged[:, start],
                        averaged[:, start + 1],
                        averaged[:, start + 2 :].sum(dim=-1),
                    ),
                    dim=-1,
                )
                radio_attention[:, modality] = radio_attention[:, modality].index_copy(0, rows, masses)

        completed = torch.stack(completed_slots, dim=1)
        if not bool(torch.isfinite(completed).all()):
            raise RuntimeError("Completion produced non-finite slot features.")
        return {
            "completed_tokens": completed,
            "physical_availability": physical,
            "semantic_slot_mask": torch.ones_like(physical),
            "query_active": query_active,
            "prototype_distribution": distributions,
            "sensing_prototype_distribution": base_distributions,
            "radio_reliability": slot_rho,
            "sample_radio_reliability": rho,
            "sensing_probability": sensing_probability,
            "radio_probability": radio_probability,
            "anchor_probability": anchor_probability,
            "top_k_indices": anchor_top,
            "base_attention_mass": base_attention,
            "radio_attention_mass": radio_attention,
            "gamma": gamma,
            **{f"reliability_{key}": value for key, value in reliability.items() if key != "rho"},
        }


class MissingPathAdapter(nn.Module):
    """Zero-initialized feature residual used only by the gated B8 experiment."""

    def __init__(self, feature_dim: int = 64, num_modalities: int = 4, bottleneck_dim: int = 32) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_modalities = int(num_modalities)
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim + self.num_modalities),
            nn.Linear(self.feature_dim + self.num_modalities, int(bottleneck_dim)),
            nn.GELU(),
            nn.Linear(int(bottleneck_dim), self.feature_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, feature: torch.Tensor, physical_availability: torch.Tensor) -> torch.Tensor:
        physical = physical_availability.to(device=feature.device, dtype=torch.bool)
        if tuple(physical.shape) != (feature.shape[0], self.num_modalities):
            raise ValueError("physical_availability has an incompatible shape.")
        synthetic = (~physical).to(feature.dtype)
        update = self.network(torch.cat((feature, synthetic), dim=-1))
        return feature + (~physical.all(dim=1))[:, None].to(feature.dtype) * update


__all__ = [
    "CSIAnchoredPrototypeCompletion",
    "MissingPathAdapter",
    "RadioReliability",
    "SparsePilotRadioEncoder",
]
