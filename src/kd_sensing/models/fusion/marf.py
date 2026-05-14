from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.models.fusion.craf import (
    ModalityTokenizer,
    UniModalHead,
    _build_modality_encoder,
    _check_temporal_features,
    _effective_modality_mask,
    _masked_modality_mean,
    _safe_transformer_padding_mask,
    compute_unimodal_confidence,
)
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE
from kd_sensing.registries import MODELS


class ModalityRouter(nn.Module):
    """Sample-wise and horizon-wise anchor/residual router."""

    def __init__(
        self,
        d_model: int,
        modality_count: int,
        horizon: int,
        *,
        modalities: tuple[str, ...] = (),
        hidden_size: int | None = None,
        temperature: float = 1.0,
        use_prior_bias: bool = True,
        prior_anchor_scale: float = 0.5,
        prior_residual_scale: float = 0.25,
        dataset_prior: dict[str, float] | list[float] | tuple[float, ...] | None = None,
        use_confidence_features: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        if float(temperature) <= 0.0:
            raise ValueError(f"router temperature must be positive, got {temperature}.")
        self.modality_count = int(modality_count)
        self.horizon = int(horizon)
        self.temperature = float(temperature)
        self.use_prior_bias = bool(use_prior_bias)
        self.prior_anchor_scale = float(prior_anchor_scale)
        self.prior_residual_scale = float(prior_residual_scale)
        self.use_confidence_features = bool(use_confidence_features)
        input_size = int(d_model) + (2 if self.use_confidence_features else 0)
        hidden = int(hidden_size or max(int(d_model), 16))
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.horizon * 2),
        )
        if zero_init:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        prior = _resolve_prior_values(dataset_prior, modalities, self.modality_count, default=0.5)
        self.register_buffer("prior", prior, persistent=True)

    def set_prior(self, priors: dict[str, float] | list[float] | tuple[float, ...], modalities: tuple[str, ...]) -> None:
        prior = _resolve_prior_values(priors, modalities, self.modality_count, default=0.5)
        self.prior.copy_(prior.to(device=self.prior.device, dtype=self.prior.dtype))

    def forward(
        self,
        modality_summary: torch.Tensor,
        confidence: torch.Tensor,
        modality_mask: torch.Tensor,
        *,
        temperature: float | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if modality_summary.ndim != 3:
            raise ValueError(
                f"modality_summary must have shape [B, K, D], got {tuple(modality_summary.shape)}."
            )
        if modality_summary.shape[1] != self.modality_count:
            raise ValueError(
                f"router expected {self.modality_count} modalities, got {modality_summary.shape[1]}."
            )
        if self.use_confidence_features and (
            confidence.shape[:2] != modality_summary.shape[:2] or confidence.shape[-1] != 2
        ):
            raise ValueError("confidence must have shape [B, K, 2] aligned with modality_summary.")
        available = modality_mask.to(device=modality_summary.device, dtype=torch.bool)
        if available.shape != modality_summary.shape[:2]:
            raise ValueError("modality_mask must have shape [B, K] aligned with modality_summary.")
        features = [modality_summary]
        if self.use_confidence_features:
            features.append(confidence.to(dtype=modality_summary.dtype))
        logits = self.net(torch.cat(features, dim=-1)).view(
            modality_summary.shape[0],
            self.modality_count,
            self.horizon,
            2,
        )
        anchor_logits = logits[..., 0].transpose(1, 2).contiguous()
        residual_logits = logits[..., 1].transpose(1, 2).contiguous()
        prior = self.prior.to(device=modality_summary.device, dtype=modality_summary.dtype)
        if self.use_prior_bias:
            prior_logits = _prior_to_logit(prior)
            anchor_logits = anchor_logits + self.prior_anchor_scale * prior_logits.view(1, 1, -1)
            residual_logits = residual_logits + self.prior_residual_scale * prior_logits.view(1, 1, -1)
        temp = _temperature_value(temperature, self.temperature)
        anchor_weights = _masked_softmax(anchor_logits / temp, available)
        residual_weights = torch.sigmoid(residual_logits / temp)
        residual_weights = residual_weights.masked_fill(~available.unsqueeze(1), 0.0)
        prior_batch = prior.view(1, -1).expand(modality_summary.shape[0], -1)
        return {
            "anchor_logits": anchor_logits,
            "anchor_weights": anchor_weights,
            "residual_logits": residual_logits,
            "residual_weights": residual_weights,
            "prior": prior_batch,
        }


class AnchorFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        horizon: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.query = nn.Parameter(torch.randn(self.horizon, int(d_model)) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(
        self,
        tokens: torch.Tensor,
        anchor_weights: torch.Tensor,
        token_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"tokens must have shape [B, K, T, D], got {tuple(tokens.shape)}.")
        batch_size, modality_count, seq_len, d_model = tokens.shape
        if anchor_weights.shape != (batch_size, self.horizon, modality_count):
            raise ValueError(
                "anchor_weights must have shape "
                f"{(batch_size, self.horizon, modality_count)}, got {tuple(anchor_weights.shape)}."
            )
        scaled_tokens = tokens.unsqueeze(1) * anchor_weights.view(batch_size, self.horizon, modality_count, 1, 1)
        memory = scaled_tokens.reshape(batch_size * self.horizon, modality_count * seq_len, d_model)
        padding = token_padding_mask.unsqueeze(1).expand(batch_size, self.horizon, modality_count, seq_len)
        padding = padding.reshape(batch_size * self.horizon, modality_count * seq_len)
        safe_padding = _safe_transformer_padding_mask(padding)
        query = self.query.view(1, self.horizon, d_model).expand(batch_size, -1, -1)
        query = query.reshape(batch_size * self.horizon, 1, d_model)
        attended, _ = self.attention(query, memory, memory, key_padding_mask=safe_padding, need_weights=False)
        attended = self.norm(attended.squeeze(1))
        return attended.view(batch_size, self.horizon, d_model)


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        horizon: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_scale: float = 0.2,
        enabled: bool = True,
    ):
        super().__init__()
        self.modality_count = int(modality_count)
        self.horizon = int(horizon)
        self.residual_scale = float(residual_scale)
        self.enabled = bool(enabled)
        self.modality_embedding = nn.Embedding(self.modality_count, int(d_model))
        self.attention = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), int(d_model)),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        h_anchor: torch.Tensor,
        residual_weights: torch.Tensor,
        token_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, modality_count, seq_len, d_model = tokens.shape
        if residual_weights.shape != (batch_size, self.horizon, modality_count):
            raise ValueError(
                "residual_weights must have shape "
                f"{(batch_size, self.horizon, modality_count)}, got {tuple(residual_weights.shape)}."
            )
        if not self.enabled:
            zeros = tokens.new_zeros(batch_size, self.horizon, modality_count, d_model)
            return h_anchor, zeros
        modality_ids = torch.arange(modality_count, device=tokens.device)
        modality_embed = self.modality_embedding(modality_ids).view(1, 1, modality_count, d_model)
        query = h_anchor.unsqueeze(2) + modality_embed
        query = query.reshape(batch_size * self.horizon * modality_count, 1, d_model)
        memory = tokens.unsqueeze(1).expand(batch_size, self.horizon, modality_count, seq_len, d_model)
        memory = memory.reshape(batch_size * self.horizon * modality_count, seq_len, d_model)
        padding = token_padding_mask.unsqueeze(1).expand(batch_size, self.horizon, modality_count, seq_len)
        padding = padding.reshape(batch_size * self.horizon * modality_count, seq_len)
        safe_padding = _safe_transformer_padding_mask(padding)
        attended, _ = self.attention(query, memory, memory, key_padding_mask=safe_padding, need_weights=False)
        residual_delta = self.delta(attended.squeeze(1)).view(batch_size, self.horizon, modality_count, d_model)
        available = ~token_padding_mask.all(dim=-1)
        residual_delta = residual_delta.masked_fill(~available.view(batch_size, 1, modality_count, 1), 0.0)
        weighted = residual_delta * residual_weights.view(batch_size, self.horizon, modality_count, 1)
        h_final = h_anchor + self.residual_scale * weighted.sum(dim=2)
        return h_final, residual_delta


@MODELS.register("marf_fusion")
class MARFFusionNet(nn.Module):
    supports_force_modality_mask = True
    supports_marf_routing = True

    def __init__(
        self,
        *,
        feature_size: int,
        num_classes: int,
        num_pred: int = 3,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        image_channels: int = 3,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        router: dict[str, Any] | None = None,
        anchor_fusion: dict[str, Any] | None = None,
        residual_adapter: dict[str, Any] | None = None,
        return_unimodal: bool = True,
        **_: Any,
    ):
        super().__init__()
        self.name = "MARFFusionNet"
        self.modalities = normalize_modalities(
            tuple(modalities or ("image", "radar")),
            context="MARF fusion modalities",
        )
        self.modality_count = len(self.modalities)
        self.feature_size = int(feature_size)
        self.d_model = int(d_model or feature_size)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.horizon = self.num_pred
        self.return_unimodal = bool(return_unimodal)
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")

        self.encoders = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        for modality in self.modalities:
            self.encoders[modality] = _build_modality_encoder(
                modality,
                self.feature_size,
                image_channels=image_channels,
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                lidar_channels=lidar_channels,
                mmwave_input_size=mmwave_input_size,
            )
            self.feature_projections[modality] = (
                nn.Identity() if self.feature_size == self.d_model else nn.Linear(self.feature_size, self.d_model)
            )

        self.tokenizer = ModalityTokenizer(
            self.modality_count,
            self.d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.unimodal_head = UniModalHead(self.d_model, self.horizon, self.num_classes, dropout=dropout)
        router_cfg = dict(router or {})
        self.router = ModalityRouter(
            self.d_model,
            self.modality_count,
            self.horizon,
            modalities=self.modalities,
            **router_cfg,
        )
        anchor_cfg = dict(anchor_fusion or {})
        self.anchor_fusion = AnchorFusion(
            self.d_model,
            self.horizon,
            num_heads=int(anchor_cfg.pop("num_heads", num_heads)),
            dropout=float(anchor_cfg.pop("dropout", dropout)),
        )
        if anchor_cfg:
            unknown = ", ".join(sorted(anchor_cfg))
            raise ValueError(f"Unknown MARF anchor_fusion fields: {unknown}.")
        residual_cfg = dict(residual_adapter or {})
        self.residual_adapter = ResidualAdapter(
            self.d_model,
            self.modality_count,
            self.horizon,
            num_heads=int(residual_cfg.pop("num_heads", num_heads)),
            dropout=float(residual_cfg.pop("dropout", dropout)),
            **residual_cfg,
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.num_classes),
        )

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        raw_inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
            "mmwave": mmwave_batch,
        }
        modality_features = []
        batch_size = None
        seq_len = None
        for modality in self.modalities:
            tensor = raw_inputs[modality]
            if tensor is None:
                raise ValueError(f"MARF fusion requires '{modality}' input because it is enabled.")
            features = self.encoders[modality](tensor)
            features = self.feature_projections[modality](features)
            batch_size, seq_len = _check_temporal_features(features, modality, batch_size, seq_len)
            modality_features.append(features)
        assert batch_size is not None and seq_len is not None

        stacked_features = torch.stack(modality_features, dim=1)
        effective_mask = _effective_modality_mask(
            batch_size,
            self.modality_count,
            device=stacked_features.device,
            force_modality_mask=force_modality_mask,
        )
        token_padding_mask = ~effective_mask.unsqueeze(-1).expand(batch_size, self.modality_count, seq_len)
        tokens = self.tokenizer(stacked_features)
        modality_summary = _masked_modality_mean(tokens, token_padding_mask)
        unimodal_logits = self.unimodal_head(modality_summary)
        confidence = compute_unimodal_confidence(unimodal_logits)
        router_output = self.router(modality_summary, confidence, effective_mask)
        h_anchor = self.anchor_fusion(tokens, router_output["anchor_weights"], token_padding_mask)
        h_final, residual_delta = self.residual_adapter(
            tokens,
            h_anchor,
            router_output["residual_weights"],
            token_padding_mask,
        )
        logits = self.prediction_head(h_final)
        input_features = _available_timewise_mean(tokens, effective_mask)
        return {
            "logits": logits,
            "input_features": input_features,
            "output_features": h_final,
            "token_features": tokens,
            "anchor_logits": router_output["anchor_logits"],
            "anchor_weights": router_output["anchor_weights"],
            "residual_logits": router_output["residual_logits"],
            "residual_weights": router_output["residual_weights"],
            "h_anchor": h_anchor,
            "h_final": h_final,
            "residual_delta": residual_delta,
            "prior": router_output["prior"],
            "effective_modality_mask": effective_mask,
            "unimodal_logits": unimodal_logits if self.return_unimodal else torch.empty(0, device=logits.device),
            "confidence": confidence,
            "token_padding_mask": token_padding_mask,
            "modalities": self.modalities,
        }

    def set_reliability_prior(self, priors: dict[str, float] | list[float] | tuple[float, ...]) -> None:
        self.router.set_prior(priors, self.modalities)


def _resolve_prior_values(
    dataset_prior: dict[str, float] | list[float] | tuple[float, ...] | None,
    modalities: tuple[str, ...],
    modality_count: int,
    *,
    default: float,
) -> torch.Tensor:
    if dataset_prior is None:
        return torch.full((int(modality_count),), float(default), dtype=torch.float32)
    if isinstance(dataset_prior, dict):
        return torch.tensor([float(dataset_prior.get(name, default)) for name in modalities], dtype=torch.float32)
    values = torch.tensor([float(value) for value in dataset_prior], dtype=torch.float32)
    if values.numel() != int(modality_count):
        raise ValueError(f"dataset_prior must contain {modality_count} values, got {values.numel()}.")
    return values


def _prior_to_logit(prior: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(prior.dtype).eps
    return torch.logit(prior.clamp(eps, 1.0 - eps))


def _temperature_value(value: float | torch.Tensor | None, default: float) -> float:
    if torch.is_tensor(value):
        value = float(value.detach().item())
    if value is None:
        value = default
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"router temperature must be positive, got {value}.")
    return value


def _masked_softmax(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    mask = available.unsqueeze(1).expand_as(logits)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    flat_logits = masked_logits.reshape(-1, masked_logits.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1])
    all_masked = ~flat_mask.any(dim=1)
    if torch.any(all_masked):
        flat_logits = flat_logits.clone()
        flat_logits[all_masked] = 0.0
    weights = F.softmax(flat_logits, dim=-1).view_as(logits)
    return weights.masked_fill(~mask, 0.0)


def _available_timewise_mean(tokens: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
    valid = effective_mask.to(device=tokens.device, dtype=tokens.dtype).view(tokens.shape[0], tokens.shape[1], 1, 1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return (tokens * valid).sum(dim=1) / counts


__all__ = [
    "AnchorFusion",
    "MARFFusionNet",
    "ModalityRouter",
    "ResidualAdapter",
]
