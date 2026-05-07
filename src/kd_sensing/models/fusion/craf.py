from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image import ImageFeatureExtractor
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE, MmWaveFeatureExtractor
from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.registries import MODELS


def compute_unimodal_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Return entropy confidence and top-probability margin for [B, K, H, C] logits."""

    if logits.ndim != 4:
        raise ValueError(f"unimodal logits must have shape [B, K, H, C], got {tuple(logits.shape)}.")
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(logits.shape[-1]), device=logits.device, dtype=logits.dtype))
    entropy_confidence = 1.0 - entropy / max_entropy.clamp_min(1e-12)
    top2 = probs.topk(k=min(2, logits.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = top2[..., 0]
    else:
        margin = top2[..., 0] - top2[..., 1]
    return torch.stack([entropy_confidence.mean(dim=-1), margin.mean(dim=-1)], dim=-1)


class UniModalHead(nn.Module):
    def __init__(self, d_model: int, horizon: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.horizon = int(horizon)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.horizon * self.num_classes),
        )

    def forward(self, modality_repr: torch.Tensor) -> torch.Tensor:
        if modality_repr.ndim != 3:
            raise ValueError(f"modality_repr must have shape [B, K, D], got {tuple(modality_repr.shape)}.")
        batch_size, modality_count, _ = modality_repr.shape
        logits = self.net(modality_repr)
        return logits.view(batch_size, modality_count, self.horizon, self.num_classes)


class ReliabilityEstimator(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        *,
        hidden_size: int | None = None,
        min_gate: float = 0.0,
        use_dataset_prior: bool = False,
        dataset_prior: list[float] | tuple[float, ...] | dict[str, float] | None = None,
        modalities: tuple[str, ...] = (),
    ):
        super().__init__()
        if not 0.0 <= float(min_gate) < 1.0:
            raise ValueError(f"min_gate must be in [0, 1), got {min_gate}.")
        self.min_gate = float(min_gate)
        self.use_dataset_prior = bool(use_dataset_prior)
        self.modality_count = int(modality_count)
        hidden = int(hidden_size or max(d_model // 2, 16))
        prior_feature_count = 1 if self.use_dataset_prior else 0
        self.net = nn.Sequential(
            nn.Linear(d_model + 2 + prior_feature_count, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        prior = torch.zeros(self.modality_count, dtype=torch.float32)
        if dataset_prior is not None:
            prior = _resolve_dataset_prior(dataset_prior, modalities, self.modality_count)
        self.register_buffer("dataset_prior", prior, persistent=True)

    def forward(
        self,
        modality_repr: torch.Tensor,
        confidence: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> torch.Tensor:
        if modality_repr.ndim != 3:
            raise ValueError(f"modality_repr must have shape [B, K, D], got {tuple(modality_repr.shape)}.")
        if confidence.shape[:2] != modality_repr.shape[:2] or confidence.shape[-1] != 2:
            raise ValueError("confidence must have shape [B, K, 2] aligned with modality_repr.")
        available = available_mask.to(torch.bool)
        if available.shape != modality_repr.shape[:2]:
            raise ValueError("available_mask must have shape [B, K] aligned with modality_repr.")

        features = [modality_repr, confidence]
        if self.use_dataset_prior:
            prior = self.dataset_prior.to(dtype=modality_repr.dtype, device=modality_repr.device)
            prior = prior.view(1, self.modality_count, 1).expand(modality_repr.shape[0], -1, -1)
            features.append(prior)
        raw_gate = torch.sigmoid(self.net(torch.cat(features, dim=-1)).squeeze(-1))
        if self.min_gate > 0:
            raw_gate = raw_gate * (1.0 - self.min_gate) + self.min_gate
        return raw_gate.masked_fill(~available, 0.0)


class ModalityTokenizer(nn.Module):
    def __init__(self, modality_count: int, d_model: int, *, max_seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.time_embedding = nn.Embedding(int(max_seq_len), d_model)
        self.modality_embedding = nn.Embedding(int(modality_count), d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must have shape [B, K, T, D], got {tuple(features.shape)}.")
        _, modality_count, seq_len, _ = features.shape
        if seq_len > self.time_embedding.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.time_embedding.num_embeddings}."
            )
        time_ids = torch.arange(seq_len, device=features.device)
        modality_ids = torch.arange(modality_count, device=features.device)
        time = self.time_embedding(time_ids).view(1, 1, seq_len, -1)
        modality = self.modality_embedding(modality_ids).view(1, modality_count, 1, -1)
        return self.dropout(self.layer_norm(features + time + modality))


class HorizonPredictionHead(nn.Module):
    def __init__(self, d_model: int, horizon: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.horizon = int(horizon)
        self.query = nn.Parameter(torch.randn(self.horizon, d_model) * 0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_features = context.unsqueeze(1) + self.query.unsqueeze(0)
        return self.net(query_features), query_features


class CRAFTTokenFusionBase(nn.Module):
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        feature_size: int,
        num_classes: int,
        num_pred: int = 3,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int | None = None,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        image_channels: int = 1,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        return_unimodal: bool = True,
        use_reliability: bool = True,
        reliability: dict[str, Any] | None = None,
        **_: Any,
    ):
        super().__init__()
        self.modalities = normalize_modalities(
            tuple(modalities or ("image", "radar")),
            context="CRAF fusion modalities",
        )
        self.modality_count = len(self.modalities)
        self.feature_size = int(feature_size)
        self.d_model = int(d_model or feature_size)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.horizon = self.num_pred + 1
        self.return_unimodal = bool(return_unimodal)
        self.use_reliability = bool(use_reliability)

        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")

        self.encoders = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        for modality in self.modalities:
            encoder = _build_modality_encoder(
                modality,
                self.feature_size,
                image_channels=image_channels,
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                lidar_channels=lidar_channels,
                mmwave_input_size=mmwave_input_size,
            )
            self.encoders[modality] = encoder
            if self.feature_size == self.d_model:
                self.feature_projections[modality] = nn.Identity()
            else:
                self.feature_projections[modality] = nn.Linear(self.feature_size, self.d_model)

        self.tokenizer = ModalityTokenizer(
            self.modality_count,
            self.d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=max(self.d_model * 4, 64),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.unimodal_head = UniModalHead(self.d_model, self.horizon, self.num_classes, dropout=dropout)
        reliability_cfg = dict(reliability or {})
        self.reliability_estimator = ReliabilityEstimator(
            self.d_model,
            self.modality_count,
            modalities=self.modalities,
            **reliability_cfg,
        )
        self.prediction_head = HorizonPredictionHead(self.d_model, self.horizon, self.num_classes, dropout=dropout)

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
                raise ValueError(f"CRAF fusion requires '{modality}' input because it is enabled.")
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

        modality_repr = _masked_modality_mean(tokens, token_padding_mask)
        unimodal_logits = self.unimodal_head(modality_repr)
        confidence = compute_unimodal_confidence(unimodal_logits)
        if self.use_reliability:
            reliability = self.reliability_estimator(modality_repr, confidence, effective_mask)
            gated_tokens = tokens * reliability.view(batch_size, self.modality_count, 1, 1)
        else:
            reliability = effective_mask.to(tokens.dtype)
            gated_tokens = tokens

        flat_tokens = gated_tokens.reshape(batch_size, self.modality_count * seq_len, self.d_model)
        flat_padding_mask = token_padding_mask.reshape(batch_size, self.modality_count * seq_len)
        safe_padding_mask = _safe_transformer_padding_mask(flat_padding_mask)
        memory = self.transformer(flat_tokens, src_key_padding_mask=safe_padding_mask)
        context = _masked_sequence_mean(memory, flat_padding_mask)
        logits, query_features = self.prediction_head(context)
        sequence_features = _timewise_memory(memory, flat_padding_mask, self.modality_count, seq_len)

        return {
            "logits": logits,
            "input_features": sequence_features,
            "output_features": query_features,
            "reliability": reliability,
            "effective_modality_mask": effective_mask,
            "unimodal_logits": unimodal_logits if self.return_unimodal else torch.empty(0, device=logits.device),
            "confidence": confidence,
            "fusion_memory": memory,
            "token_features": tokens,
            "token_padding_mask": token_padding_mask,
            "modalities": self.modalities,
        }


@MODELS.register("craf_fusion")
class CRAFFusionNet(CRAFTTokenFusionBase):
    def __init__(self, **kwargs: Any):
        super().__init__(use_reliability=True, **kwargs)
        self.name = "CRAFFusionNet"


@MODELS.register("token_transformer_fusion")
class TokenTransformerFusionNet(CRAFTTokenFusionBase):
    def __init__(self, **kwargs: Any):
        kwargs.pop("reliability", None)
        super().__init__(use_reliability=False, reliability={"min_gate": 0.0}, **kwargs)
        self.name = "TokenTransformerFusionNet"


def _build_modality_encoder(
    modality: str,
    feature_size: int,
    *,
    image_channels: int,
    radar_channels: int,
    gps_input_size: int,
    lidar_channels: int,
    mmwave_input_size: int,
) -> nn.Module:
    if modality == "image":
        return ImageFeatureExtractor(feature_size, image_channels)
    if modality == "radar":
        return RadarFeatureExtractor(feature_size, radar_channels)
    if modality == "gps":
        return GpsFeatureExtractor(feature_size, gps_input_size=gps_input_size)
    if modality == "lidar":
        return LidarFeatureExtractor(feature_size, in_channels=lidar_channels)
    if modality == "mmwave":
        return MmWaveFeatureExtractor(feature_size=feature_size, mmwave_input_size=mmwave_input_size)
    available = ", ".join(MODALITY_ORDER)
    raise ValueError(f"Unknown CRAF modality '{modality}'. Available modalities: {available}.")


def _resolve_dataset_prior(
    dataset_prior: list[float] | tuple[float, ...] | dict[str, float],
    modalities: tuple[str, ...],
    modality_count: int,
) -> torch.Tensor:
    if isinstance(dataset_prior, dict):
        return torch.tensor([float(dataset_prior.get(name, 0.0)) for name in modalities], dtype=torch.float32)
    values = torch.tensor([float(value) for value in dataset_prior], dtype=torch.float32)
    if values.numel() != modality_count:
        raise ValueError(f"dataset_prior must contain {modality_count} values, got {values.numel()}.")
    return values


def _effective_modality_mask(
    batch_size: int,
    modality_count: int,
    *,
    device: torch.device,
    force_modality_mask: torch.Tensor | None,
) -> torch.Tensor:
    mask = torch.ones(batch_size, modality_count, dtype=torch.bool, device=device)
    if force_modality_mask is None:
        return mask
    forced = force_modality_mask.to(device=device, dtype=torch.bool)
    if forced.ndim == 1:
        if forced.shape[0] != modality_count:
            raise ValueError(f"force_modality_mask shape must be [K] or [B, K], got {tuple(forced.shape)}.")
        forced = forced.unsqueeze(0).expand(batch_size, -1)
    if forced.shape != mask.shape:
        raise ValueError(f"force_modality_mask shape must be {tuple(mask.shape)}, got {tuple(forced.shape)}.")
    return mask & forced


def _check_temporal_features(
    features: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
) -> tuple[int, int]:
    if features.ndim != 3:
        raise ValueError(f"{modality} features must have shape [B, T, D], got {tuple(features.shape)}.")
    current_batch = int(features.shape[0])
    current_seq = int(features.shape[1])
    if batch_size is not None and (batch_size != current_batch or seq_len != current_seq):
        raise ValueError("Enabled CRAF modalities must share batch and sequence dimensions.")
    return current_batch, current_seq


def _masked_modality_mean(tokens: torch.Tensor, token_padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~token_padding_mask).to(tokens.dtype).unsqueeze(-1)
    summed = (tokens * valid).sum(dim=2)
    counts = valid.sum(dim=2).clamp_min(1.0)
    return summed / counts


def _masked_sequence_mean(memory: torch.Tensor, flat_padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~flat_padding_mask).to(memory.dtype).unsqueeze(-1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return (memory * valid).sum(dim=1) / counts


def _timewise_memory(
    memory: torch.Tensor,
    flat_padding_mask: torch.Tensor,
    modality_count: int,
    seq_len: int,
) -> torch.Tensor:
    batch_size, _, d_model = memory.shape
    memory = memory.view(batch_size, modality_count, seq_len, d_model)
    padding = flat_padding_mask.view(batch_size, modality_count, seq_len)
    valid = (~padding).to(memory.dtype).unsqueeze(-1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return (memory * valid).sum(dim=1) / counts


def _safe_transformer_padding_mask(flat_padding_mask: torch.Tensor) -> torch.Tensor:
    mask = flat_padding_mask.clone()
    all_masked = mask.all(dim=1)
    if torch.any(all_masked):
        mask[all_masked, :] = False
    return mask
