from typing import Any
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.models.reliability_biased_missing_attention import ReliabilityBiasedMissingAwareAttention
from kd_sensing.registries import ENCODERS, MODELS
from kd_sensing.utils.checkpoint import load_torch_payload


DEFAULT_MODALITIES = ("image", "radar", "lidar", "gps")
TEMPORAL_MASK_STATISTIC_NAMES = (
    "coverage",
    "last_age",
    "longest_gap",
    "trailing_gap",
    "missing_block_count",
)
TEMPORAL_SCORER_STATISTIC_NAMES = TEMPORAL_MASK_STATISTIC_NAMES + (
    "relative_age",
    "distance_since_previous_valid",
)


def temporal_mask_statistics(modality_temporal_mask: torch.Tensor) -> torch.Tensor:
    """Return normalized [coverage, recency, gap, trailing-gap, block-count] per modality."""
    mask = torch.as_tensor(modality_temporal_mask, dtype=torch.bool)
    if mask.ndim != 3 or int(mask.shape[1]) <= 0:
        raise ValueError(
            "modality_temporal_mask must have shape [B,T,M] with T > 0, "
            f"got {tuple(mask.shape)}."
        )
    batch_size, steps, num_modalities = mask.shape
    values = mask.to(dtype=torch.float32)
    coverage = values.mean(dim=1)

    positions = torch.arange(steps, device=mask.device).view(1, steps, 1)
    last_valid = torch.where(mask, positions, torch.full_like(positions, -1)).amax(dim=1)
    last_age = (steps - 1 - last_valid).to(dtype=torch.float32) / max(steps - 1, 1)
    last_age = torch.where(last_valid >= 0, last_age, torch.ones_like(last_age))

    missing = ~mask
    run = torch.zeros(batch_size, num_modalities, dtype=torch.long, device=mask.device)
    longest = torch.zeros_like(run)
    for step in range(steps):
        run = torch.where(missing[:, step], run + 1, torch.zeros_like(run))
        longest = torch.maximum(longest, run)
    trailing = run
    block_starts = missing & torch.cat(
        [torch.ones(batch_size, 1, num_modalities, dtype=torch.bool, device=mask.device), mask[:, :-1]],
        dim=1,
    )
    blocks = block_starts.sum(dim=1)
    return torch.stack(
        [
            coverage,
            last_age,
            longest.to(dtype=torch.float32) / steps,
            trailing.to(dtype=torch.float32) / steps,
            blocks.to(dtype=torch.float32) / max(math.ceil(steps / 2), 1),
        ],
        dim=-1,
    )


def _temporal_scorer_statistics(mask: torch.Tensor, statistics: torch.Tensor) -> torch.Tensor:
    batch_size, steps, num_modalities = mask.shape
    denominator = max(steps - 1, 1)
    relative_age = torch.arange(
        steps - 1,
        -1,
        -1,
        device=mask.device,
        dtype=statistics.dtype,
    ).view(1, steps, 1) / denominator
    relative_age = relative_age.expand(batch_size, -1, num_modalities)
    previous = torch.full((batch_size, num_modalities), -1, dtype=torch.long, device=mask.device)
    distances = torch.zeros(batch_size, steps, num_modalities, dtype=statistics.dtype, device=mask.device)
    for step in range(steps):
        valid = mask[:, step]
        gap = (step - previous - 1).clamp_min(0).to(dtype=statistics.dtype) / denominator
        distances[:, step] = torch.where(valid & previous.ge(0), gap, torch.zeros_like(gap))
        previous = torch.where(valid, torch.full_like(previous, step), previous)
    return torch.cat(
        [
            statistics.unsqueeze(1).expand(-1, steps, -1, -1),
            relative_age.unsqueeze(-1),
            distances.unsqueeze(-1),
        ],
        dim=-1,
    )


class ModalityReliabilityHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), int(d_model) * 2))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(z).chunk(2, dim=-1)
        return mu, logvar


class FullModalTeacher(nn.Module):
    def __init__(self, d_model: int, num_beams: int, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=max(int(d_model) * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = BeamPredictionHead(int(d_model), int(num_beams), dropout=dropout)

    def forward(self, tokens: torch.Tensor, modality_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(tokens.shape[0])
        query = self.query.expand(batch_size, -1, -1)
        memory = self.encoder(torch.cat([query, tokens + modality_embedding.unsqueeze(0)], dim=1))
        u_star = memory[:, 0, :]
        return u_star, self.head(u_star)


class SetContextEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.reliability_projection = nn.Linear(1, int(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=max(int(d_model) * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        modality_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(tokens.shape[0])
        query = self.query.expand(batch_size, -1, -1)
        token_input = tokens + modality_embedding.unsqueeze(0) + self.reliability_projection(reliability)
        padding = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        memory = self.encoder(torch.cat([query, token_input], dim=1), src_key_padding_mask=padding)
        return memory[:, 0, :]


class GaussianJEPAPredictor(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None, logvar_min: float = -6.0, logvar_max: float = 2.0):
        super().__init__()
        hidden = int(hidden_dim or max(int(d_model) * 2, 64))
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.net = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), hidden), nn.GELU())
        self.mu = nn.Linear(hidden, int(d_model))
        self.logvar = nn.Linear(hidden, int(d_model))

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(context)
        return self.mu(hidden), self.logvar(hidden).clamp(self.logvar_min, self.logvar_max)


class ReliabilityGatedCrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, beta: float = 1.0, eps: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)
        self.query = nn.Parameter(torch.zeros(int(d_model)))
        self.key = nn.Linear(int(d_model), int(d_model))
        self.value = nn.Linear(int(d_model), int(d_model))
        self.out = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Dropout(float(dropout)), nn.Linear(int(d_model), int(d_model)))

    def forward(
        self,
        tokens: torch.Tensor,
        mu_token: torch.Tensor,
        reliability: torch.Tensor,
        global_reliability: torch.Tensor,
    ) -> torch.Tensor:
        all_tokens = torch.cat([tokens, mu_token.unsqueeze(1)], dim=1)
        all_reliability = torch.cat([reliability.squeeze(-1), global_reliability.view(-1, 1)], dim=1)
        keys = self.key(all_tokens)
        scores = (keys * self.query.view(1, 1, -1)).sum(dim=-1) / (keys.shape[-1] ** 0.5)
        scores = scores + self.beta * all_reliability.clamp_min(self.eps).log()
        scores = scores.masked_fill(all_reliability <= 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        fused = (weights.unsqueeze(-1) * self.value(all_tokens)).sum(dim=1)
        return self.out(fused)


class BeamPredictionHead(nn.Module):
    def __init__(self, d_model: int, num_beams: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Dropout(float(dropout)), nn.Linear(int(d_model), int(num_beams)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PatternConditionedPrototypeGate(nn.Module):
    def __init__(self, num_modalities: int, reliability_dim: int, pattern_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.net = nn.Sequential(
            nn.LayerNorm(int(reliability_dim) + int(pattern_dim) + self.num_modalities),
            nn.Linear(int(reliability_dim) + int(pattern_dim) + self.num_modalities, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        reliability_features: torch.Tensor,
        pattern_features: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> torch.Tensor:
        if reliability_features.ndim != 3:
            raise ValueError("pcpg reliability_features must have shape [B, M, F].")
        batch_size, num_modalities, _ = reliability_features.shape
        if num_modalities != self.num_modalities:
            raise ValueError(f"pcpg expected {self.num_modalities} modalities, got {num_modalities}.")
        if pattern_features.ndim != 2 or pattern_features.shape[0] != batch_size:
            raise ValueError("pcpg pattern_features must have shape [B, P].")
        modality_eye = torch.eye(
            num_modalities,
            device=reliability_features.device,
            dtype=reliability_features.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        pattern = pattern_features.to(device=reliability_features.device, dtype=reliability_features.dtype)
        pattern = pattern.unsqueeze(1).expand(-1, num_modalities, -1)
        logits = self.net(torch.cat([reliability_features, pattern, modality_eye], dim=-1)).squeeze(-1)
        return masked_pcpg_softmax(logits, available_mask)


class BeamPrototypeReliabilityRouter(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        feature_dim: int,
        pattern_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.feature_dim = int(feature_dim)
        self.pattern_dim = int(pattern_dim)
        router_dim = self.feature_dim + self.num_modalities
        self.net = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.pattern_bias = nn.Linear(self.pattern_dim, self.num_modalities, bias=False)

    def forward_logits(
        self,
        reliability_features: torch.Tensor,
        available_mask: torch.Tensor,
        pattern_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reliability_features.ndim != 3:
            raise ValueError("bprr reliability_features must have shape [B, M, F].")
        batch_size, num_modalities, feature_dim = reliability_features.shape
        if num_modalities != self.num_modalities or feature_dim != self.feature_dim:
            raise ValueError(
                "bprr expected reliability_features shape "
                f"[B, {self.num_modalities}, {self.feature_dim}], got {tuple(reliability_features.shape)}."
            )
        modality_eye = torch.eye(
            num_modalities,
            device=reliability_features.device,
            dtype=reliability_features.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        logits = self.net(torch.cat([reliability_features, modality_eye], dim=-1)).squeeze(-1)
        if pattern_features is not None:
            if pattern_features.ndim != 2 or tuple(pattern_features.shape) != (batch_size, self.pattern_dim):
                raise ValueError(
                    f"bprr pattern_features must have shape {(batch_size, self.pattern_dim)}, "
                    f"got {tuple(pattern_features.shape)}."
                )
            logits = logits + self.pattern_bias(pattern_features.to(device=logits.device, dtype=logits.dtype))
        return logits

    def forward(
        self,
        reliability_features: torch.Tensor,
        available_mask: torch.Tensor,
        pattern_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward_logits(reliability_features, available_mask, pattern_features)
        return masked_pcpg_softmax(logits, available_mask)


class BPRRTemperatureCalibration(nn.Module):
    def __init__(self, num_modalities: int, init_temperature: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        init = max(float(init_temperature) - self.eps, self.eps)
        raw = math.log(math.expm1(init))
        self.raw_temperature = nn.Parameter(torch.full((int(num_modalities),), float(raw)))

    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + self.eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or int(logits.shape[1]) != int(self.raw_temperature.numel()):
            raise ValueError(
                "bprr temperature calibration expects logits [B, M, C] with "
                f"M={int(self.raw_temperature.numel())}, got {tuple(logits.shape)}."
            )
        return logits / self.temperatures().to(device=logits.device, dtype=logits.dtype).view(1, -1, 1)


def masked_pcpg_softmax(logits: torch.Tensor, available_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    available = available_mask.to(device=logits.device, dtype=torch.bool)
    if logits.shape != available.shape:
        raise ValueError(f"pcpg gate logits shape {tuple(logits.shape)} must match mask {tuple(available.shape)}.")
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=-1) * available.to(dtype=logits.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return torch.where(available.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))


def _masked_temporal_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    available = mask.to(device=scores.device, dtype=torch.bool)
    if scores.shape != available.shape:
        raise ValueError(f"temporal scores shape {tuple(scores.shape)} must match mask {tuple(available.shape)}.")
    masked = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=1) * available.to(dtype=scores.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
    return torch.where(available.any(dim=1, keepdim=True), weights, torch.zeros_like(weights))


class MaskConditionedAdapter(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        d_model: int,
        hidden_dim: int = 16,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
        init_identity: bool = False,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Linear(int(num_modalities), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(d_model) * 2),
        )
        if init_identity:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(mask.to(device=features.device, dtype=features.dtype)).chunk(2, dim=-1)
        scale = self.residual_scale
        return features * (1.0 + scale * gamma) + scale * beta


@MODELS.register("u_mask_beam_jepa")
class UMaskBeamJEPA(nn.Module):
    supports_modality_kwargs = True
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int = 64,
        num_classes: int = 64,
        num_pred: int = 1,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        image_channels: int = 3,
        radar_channels: int = 2,
        lidar_channels: int = 3,
        gps_input_size: int = 3,
        fusion_type: str = "reliability_gated_cross_attention",
        context_type: str = "set_transformer_simplified",
        use_teacher: bool = True,
        use_jepa_loss: bool = True,
        use_modality_uncertainty: bool = True,
        use_global_uncertainty: bool = True,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        eval_missing_pattern: dict[str, Any] | None = None,
        beta: float = 1.0,
        eps: float = 1e-6,
        use_beam_prototype_alignment: bool = False,
        beam_proto_temperature: float = 0.2,
        tau_proto: float | None = None,
        mask_sampler: str | None = None,
        use_mask_adapter: bool = False,
        mask_adapter_dim: int = 16,
        mask_adapter_apply: str = "after_fusion",
        mask_adapter_residual_scale: float = 1.0,
        mask_adapter_dropout: float = 0.0,
        pattern_film: dict[str, Any] | bool | None = None,
        pcpg_fuse_level: str = "logits",
        pcpg_hidden_dim: int = 64,
        raw_conf_temperature: float = 1.0,
        bprr_fuse_level: str = "logits",
        bprr_calibration: str = "none",
        bprr_init_temperature: float = 1.0,
        bprr_hidden_dim: int = 64,
        bprr_dropout: float = 0.1,
        router_supervision: str = "none",
        router_distill_weight: float = 0.0,
        router_distill_temperature: float = 1.0,
        router_focus_patterns: list[str] | tuple[str, ...] | str | None = None,
        router_fuse_level: str = "logits",
        router_use_pattern_features: bool = True,
        router_use_reliability_features: bool = True,
        router_use_prototype_margin: bool = True,
        router_use_entropy: bool = True,
        router_use_confidence: bool = True,
        router_use_logit_norm: bool = True,
        temporal_pooling: dict[str, Any] | bool | None = None,
        use_mask_statistics: bool = False,
        coverage_shrinkage: dict[str, Any] | bool | None = None,
        head_type: str = "legacy",
        ablation_id: str | None = None,
        encoders: dict[str, Any] | None = None,
        encoder_checkpoint_paths: dict[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__()
        retired_temporal_router = str(extra.get("temporal_router_type") or "none").strip().lower()
        if retired_temporal_router != "none":
            raise ValueError(
                "temporal_router_type is retired and is not mapped to the current model; "
                "use the explicit model.primary.temporal_pooling mapping."
            )
        self.modalities = _validate_modalities(DEFAULT_MODALITIES if modalities is None else modalities)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.fusion_type = str(fusion_type)
        self.context_type = str(context_type)
        self.use_teacher = bool(use_teacher)
        self.use_jepa_loss = bool(use_jepa_loss)
        self.use_modality_uncertainty = bool(use_modality_uncertainty)
        self.use_global_uncertainty = bool(use_global_uncertainty)
        self.use_beam_prototype_alignment = bool(use_beam_prototype_alignment)
        self.mask_sampler = mask_sampler
        self.use_mask_adapter = bool(use_mask_adapter)
        self.mask_adapter_apply = str(mask_adapter_apply)
        self.pattern_film_config = _pattern_film_config(pattern_film)
        self.pcpg_fuse_level = str(pcpg_fuse_level)
        self.raw_conf_temperature = max(float(raw_conf_temperature), 1e-6)
        self.bprr_fuse_level = str(bprr_fuse_level)
        self.bprr_calibration = str(bprr_calibration or "none").strip().lower()
        self.bprr_init_temperature = float(bprr_init_temperature)
        self.router_supervision = str(router_supervision or "none").strip().lower()
        self.router_distill_weight = float(router_distill_weight)
        self.router_distill_temperature = float(router_distill_temperature)
        self.router_focus_patterns = router_focus_patterns
        self.router_fuse_level = str(router_fuse_level or "logits").strip().lower()
        self.router_use_pattern_features = _bool(router_use_pattern_features)
        self.router_use_reliability_features = _bool(router_use_reliability_features)
        self.router_use_prototype_margin = _bool(router_use_prototype_margin)
        self.router_use_entropy = _bool(router_use_entropy)
        self.router_use_confidence = _bool(router_use_confidence)
        self.router_use_logit_norm = _bool(router_use_logit_norm)
        self.temporal_pooling_config = _temporal_pooling_config(temporal_pooling)
        self.temporal_pooling_enabled = bool(self.temporal_pooling_config["enabled"])
        self.temporal_pooling_type = str(self.temporal_pooling_config["type"])
        self.use_mask_statistics = _bool(use_mask_statistics)
        self.coverage_shrinkage_config = _coverage_shrinkage_config(coverage_shrinkage)
        self.coverage_shrinkage_enabled = bool(self.coverage_shrinkage_config["enabled"])
        self.head_type = str(head_type or "legacy").strip().lower()
        self.ablation_id = ablation_id
        self.eval_missing_pattern = dict(eval_missing_pattern or {})
        _validate_context_type(self.context_type)
        if self.fusion_type not in {
            "reliability_gated_cross_attention",
            "reliability_biased_missing_attention",
            "concat_mlp",
            "weighted_sum",
            "average",
            "pcpg",
            "raw_conf_gate",
            "bprr",
            "supervised_router",
        }:
            raise ValueError(
                "fusion_type must be reliability_gated_cross_attention, reliability_biased_missing_attention, "
                "concat_mlp, weighted_sum, average, pcpg, raw_conf_gate, bprr, or supervised_router."
            )
        if self.fusion_type == "pcpg" and self.pcpg_fuse_level != "logits":
            raise ValueError("fusion_type='pcpg' currently supports pcpg_fuse_level='logits' only.")
        if self.fusion_type == "bprr" and self.bprr_fuse_level != "logits":
            raise ValueError("fusion_type='bprr' currently supports bprr_fuse_level='logits' only.")
        if self.fusion_type == "supervised_router" and self.router_fuse_level != "logits":
            raise ValueError("fusion_type='supervised_router' currently supports router_fuse_level='logits' only.")
        if self.temporal_pooling_enabled and self.fusion_type != "supervised_router":
            raise ValueError("temporal_pooling.enabled=true requires fusion_type='supervised_router'.")
        if self.use_mask_statistics and not self.temporal_pooling_enabled:
            raise ValueError("use_mask_statistics=true requires temporal_pooling.enabled=true.")
        if self.coverage_shrinkage_enabled and not self.temporal_pooling_enabled:
            raise ValueError("coverage_shrinkage.enabled=true requires temporal_pooling.enabled=true.")
        if self.router_supervision not in {"oracle", "pattern_best", "none"}:
            raise ValueError("router_supervision must be one of oracle, pattern_best, or none.")
        if self.head_type not in {"legacy", "prototype", "classifier"}:
            raise ValueError("head_type must be one of legacy, prototype, or classifier.")
        if self.bprr_calibration not in {"none", "temperature"}:
            raise ValueError("bprr_calibration currently supports 'none' or 'temperature'.")
        if self.d_model <= 0 or self.num_classes <= 0 or self.num_pred <= 0:
            raise ValueError("d_model, num_classes, and num_pred must be positive.")

        self.encoder_configs = {name: dict((encoders or {}).get(name, {})) for name in self.modalities}
        missing_encoder_cfg = [name for name in self.modalities if not self.encoder_configs[name]]
        if missing_encoder_cfg:
            raise ValueError(f"u_mask_beam_jepa requires model.primary.encoders for modalities {missing_encoder_cfg}.")
        self.use_registry_encoders = True
        self.encoders = nn.ModuleDict()
        self.encoder_projections = nn.ModuleDict()
        for name in self.modalities:
            cfg = dict(self.encoder_configs[name])
            cfg.setdefault("output_dim", self.d_model)
            if name == "image":
                cfg.setdefault("image_channels", image_channels)
            elif name == "radar":
                cfg.setdefault("radar_channels", radar_channels)
            elif name == "lidar":
                cfg.setdefault("lidar_channels", lidar_channels)
            elif name == "gps":
                cfg.setdefault("gps_input_size", gps_input_size)
            encoder = ENCODERS.build(cfg)
            self.encoders[name] = encoder
            output_dim = int(getattr(encoder, "output_dim", cfg.get("output_dim", self.d_model)))
            self.encoder_projections[name] = (
                nn.Identity() if output_dim == self.d_model else nn.Linear(output_dim, self.d_model)
            )
            self.encoder_configs[name] = cfg
        self.encoder_checkpoint_loads = self._load_encoder_checkpoints(encoder_checkpoint_paths or {})
        self.modality_embedding = nn.Parameter(torch.zeros(len(self.modalities), self.d_model))
        self.reliability_heads = nn.ModuleDict(
            {name: ModalityReliabilityHead(self.d_model) for name in self.modalities}
        )
        self.teacher = FullModalTeacher(self.d_model, self.num_classes, num_heads, num_layers, dropout)
        self.context_encoder = SetContextEncoder(self.d_model, num_heads, num_layers, dropout)
        self.predictor = GaussianJEPAPredictor(self.d_model, logvar_min=logvar_min, logvar_max=logvar_max)
        self.cross_attention_fusion = ReliabilityGatedCrossAttentionFusion(self.d_model, beta=beta, eps=eps, dropout=dropout)
        self.rbma_fusion = ReliabilityBiasedMissingAwareAttention(
            self.d_model,
            len(self.modalities),
            num_heads=num_heads,
            beta_reliability=beta,
            eps=eps,
            dropout=dropout,
        )
        self.concat_fusion = nn.Sequential(nn.LayerNorm(self.d_model * 2), nn.Linear(self.d_model * 2, self.d_model), nn.GELU())
        if self.use_mask_adapter and self.mask_adapter_apply != "after_fusion":
            raise ValueError("mask_adapter_apply currently supports only 'after_fusion'.")
        self.mask_adapter = (
            MaskConditionedAdapter(
                len(self.modalities),
                self.d_model,
                hidden_dim=int(mask_adapter_dim),
                residual_scale=float(mask_adapter_residual_scale),
                dropout=float(mask_adapter_dropout),
            )
            if self.use_mask_adapter
            else None
        )
        self.pattern_film = (
            MaskConditionedAdapter(
                len(self.modalities),
                self.d_model,
                hidden_dim=int(self.pattern_film_config["dim"]),
                residual_scale=1.0,
                dropout=float(self.pattern_film_config.get("dropout", 0.0)),
                init_identity=bool(self.pattern_film_config.get("init_identity", True)),
            )
            if self.pattern_film_config
            else None
        )
        self.beam_head = BeamPredictionHead(self.d_model, self.num_classes, dropout=dropout)
        self.prototype_bank = BeamPrototypeBank(
            self.d_model,
            self.num_classes,
            temperature=beam_proto_temperature if tau_proto is None else float(tau_proto),
        )
        gap_hidden_dim = int(self.temporal_pooling_config["hidden_dim"])
        self.temporal_content_projection = (
            nn.Linear(self.d_model, gap_hidden_dim, bias=False)
            if self.temporal_pooling_enabled and self.temporal_pooling_type == "gap_aware_residual"
            else None
        )
        self.temporal_statistics_projection = (
            nn.Linear(len(TEMPORAL_SCORER_STATISTIC_NAMES), gap_hidden_dim)
            if self.temporal_pooling_enabled and self.temporal_pooling_type == "gap_aware_residual"
            else None
        )
        self.temporal_score_projection = (
            nn.Linear(gap_hidden_dim, 1, bias=False)
            if self.temporal_pooling_enabled and self.temporal_pooling_type == "gap_aware_residual"
            else None
        )
        self.temporal_residual_gate = (
            nn.Parameter(torch.zeros(len(self.modalities)))
            if self.temporal_pooling_enabled and self.temporal_pooling_type == "gap_aware_residual"
            else None
        )
        self.pcpg_gate = (
            PatternConditionedPrototypeGate(
                len(self.modalities),
                reliability_dim=6,
                pattern_dim=len(self.modalities) + 2,
                hidden_dim=int(pcpg_hidden_dim),
            )
            if self.fusion_type == "pcpg"
            else None
        )
        self.bprr_router = (
            BeamPrototypeReliabilityRouter(
                len(self.modalities),
                feature_dim=8 + (len(TEMPORAL_MASK_STATISTIC_NAMES) if self.use_mask_statistics else 0),
                pattern_dim=len(self.modalities) + 2,
                hidden_dim=int(bprr_hidden_dim),
                dropout=float(bprr_dropout),
            )
            if self.fusion_type in {"bprr", "supervised_router"}
            else None
        )
        shrinkage_hidden_dim = int(self.coverage_shrinkage_config["hidden_dim"])
        self.coverage_shrinkage_net = (
            nn.Sequential(
                nn.Linear(3, shrinkage_hidden_dim),
                nn.GELU(),
                nn.Linear(shrinkage_hidden_dim, 1),
            )
            if self.coverage_shrinkage_enabled
            else None
        )
        self.bprr_temperature = (
            BPRRTemperatureCalibration(
                len(self.modalities),
                init_temperature=self.bprr_init_temperature,
            )
            if self.fusion_type in {"bprr", "supervised_router"} and self.bprr_calibration == "temperature"
            else None
        )
        self._total_param_count = sum(param.numel() for param in self.parameters())
        self._trainable_param_count = sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        modality_temporal_mask: torch.Tensor | None = None,
        available_modalities: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        inputs = {"image": image_batch, "radar": radar_batch, "lidar": lidar_batch, "gps": gps_batch}
        temporal_diagnostics: dict[str, Any] = {}
        mask_statistics = None
        if self.temporal_pooling_enabled:
            latent_sequence = torch.stack(
                [self._encode_sequence(name, inputs[name]) for name in self.modalities],
                dim=2,
            )
            modality_level_mask = missing_mask if missing_mask is not None else force_modality_mask
            if modality_level_mask is None:
                modality_level_mask = available_modalities
            modality_level_mask = self._resolve_mask(modality_level_mask, latent_sequence[:, 0])
            resolved_temporal_mask = self._resolve_modality_temporal_mask(
                latent_sequence,
                modality_mask=modality_level_mask,
                temporal_mask=temporal_mask,
                modality_temporal_mask=modality_temporal_mask,
            )
            mask_statistics = temporal_mask_statistics(resolved_temporal_mask).to(
                device=latent_sequence.device,
                dtype=latent_sequence.dtype,
            )
            latent, temporal_weights = self._pool_temporal_sequence(
                latent_sequence,
                resolved_temporal_mask,
                mask_statistics,
            )
            mask = resolved_temporal_mask.any(dim=1)
            temporal_diagnostics = self._temporal_pooling_diagnostics(
                resolved_temporal_mask,
                mask_statistics,
                temporal_weights,
            )
        else:
            latent = torch.stack([self._encode(name, inputs[name]) for name in self.modalities], dim=1)
            mask = self._resolve_mask(
                missing_mask if missing_mask is not None else force_modality_mask,
                latent,
                allow_all_missing=self.fusion_type in {"reliability_biased_missing_attention", "weighted_sum"},
            )
        reliability, modality_mu_b, modality_logvar_b = self._modality_reliability(latent, mask)
        if self.temporal_pooling_enabled:
            time_weights = resolved_temporal_mask.any(dim=2).to(dtype=latent_sequence.dtype).unsqueeze(-1)
            mu_b = (latent_sequence.mean(dim=2) * time_weights).sum(dim=1) / time_weights.sum(
                dim=1
            ).clamp_min(1.0)
            logvar_b = torch.zeros_like(mu_b)
            modality_mu_b = latent
            modality_logvar_b = torch.zeros_like(latent)
            global_reliability = torch.ones(mu_b.shape[0], device=mu_b.device, dtype=mu_b.dtype)
        else:
            u_star, teacher_logits = self.teacher(latent, self.modality_embedding)
            c_a = self.context_encoder(latent, mask, reliability, self.modality_embedding)
            mu_b, logvar_b = self.predictor(c_a)
            global_reliability = (
                torch.exp(-F.softplus(logvar_b).mean(dim=-1))
                if self.use_global_uncertainty
                else torch.ones_like(mu_b[:, 0])
            )
        fused, fusion_diagnostics = self._fuse(
            latent,
            mask,
            reliability,
            mu_b,
            global_reliability,
            mask_statistics=mask_statistics,
        )
        if self.mask_adapter is not None:
            fused = self.mask_adapter(fused, mask)
            fusion_diagnostics["mask_adapter_param_count"] = sum(param.numel() for param in self.mask_adapter.parameters())
        if self.pattern_film is not None:
            fused = self.pattern_film(fused, mask)
            fusion_diagnostics["pattern_film_param_count"] = sum(param.numel() for param in self.pattern_film.parameters())
            fusion_diagnostics["pattern_film_dim"] = int(self.pattern_film_config.get("dim", 0))
            fusion_diagnostics["pattern_film_apply_at"] = str(self.pattern_film_config.get("apply_at", "pre_head"))
        base_logits = fusion_diagnostics.get("fused_logits", fusion_diagnostics.get("pcpg_fused_logits"))
        if not torch.is_tensor(base_logits):
            base_logits = self._head_logits(fused)
        if self.temporal_pooling_enabled:
            u_star = fused.detach()
            mu_b = fused
            logvar_b = torch.zeros_like(fused)
            teacher_logits = self._head_logits(fused)
        logits = base_logits.unsqueeze(1).expand(-1, self.num_pred, -1)
        teacher_logits = teacher_logits.unsqueeze(1).expand(-1, self.num_pred, -1)
        return {
            "logits": logits,
            "input_features": latent,
            "output_features": fused,
            "teacher_logits": teacher_logits,
            "u_star": u_star.detach(),
            "mu_B": mu_b,
            "logvar_B": logvar_b,
            "modality_mu_B": modality_mu_b,
            "modality_logvar_B": modality_logvar_b,
            "modality_reliability": reliability,
            "global_reliability": global_reliability,
            "missing_mask": mask,
            "modality_features": latent,
            "student_feature": fused,
            **temporal_diagnostics,
            **fusion_diagnostics,
            "metadata": self.training_strategy_metadata(),
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "u_mask_beam_jepa",
            "architecture_category": "whole_model_exception",
            "enabled_modalities": list(self.modalities),
            "modalities": list(self.modalities),
            "consumes_missing_mask": True,
            "consumes_missing_modality_metadata": self.temporal_pooling_enabled,
            "consumes_reliability_metadata": self.fusion_type in {"reliability_biased_missing_attention", "bprr", "supervised_router"},
            "reliability_metadata_consumption": "internal_modality_uncertainty"
            if self.fusion_type in {"reliability_biased_missing_attention", "bprr", "supervised_router"}
            else "none",
            "same_model_full_modal_teacher_auxiliary": True,
            "use_teacher": self.use_teacher,
            "use_jepa_loss": self.use_jepa_loss,
            "use_beam_prototype_alignment": self.use_beam_prototype_alignment,
            "mask_sampler": self.mask_sampler,
            "use_mask_adapter": self.use_mask_adapter,
            "mask_adapter_apply": self.mask_adapter_apply,
            "mask_adapter_param_count": sum(param.numel() for param in self.mask_adapter.parameters()) if self.mask_adapter is not None else 0,
            "pattern_film": self.pattern_film_config or {"enabled": False},
            "pattern_film_param_count": sum(param.numel() for param in self.pattern_film.parameters()) if self.pattern_film is not None else 0,
            "ablation_id": self.ablation_id,
            "use_modality_uncertainty": self.use_modality_uncertainty,
            "use_global_uncertainty": self.use_global_uncertainty,
            "fusion_type": self.fusion_type,
            "head_type": self.head_type,
            "prototype_margin_enabled": self._prototype_margin_enabled(),
            "router_use_pattern_features": self.router_use_pattern_features
            if self.fusion_type == "supervised_router"
            else None,
            "router_use_reliability_features": self.router_use_reliability_features
            if self.fusion_type == "supervised_router"
            else None,
            "router_use_prototype_margin": self.router_use_prototype_margin
            if self.fusion_type == "supervised_router"
            else None,
            "router_use_entropy": self.router_use_entropy if self.fusion_type == "supervised_router" else None,
            "router_use_confidence": self.router_use_confidence if self.fusion_type == "supervised_router" else None,
            "router_use_logit_norm": self.router_use_logit_norm if self.fusion_type == "supervised_router" else None,
            "pcpg_fuse_level": self.pcpg_fuse_level if self.fusion_type == "pcpg" else None,
            "bprr_fuse_level": self.bprr_fuse_level if self.fusion_type == "bprr" else None,
            "bprr_calibration": self.bprr_calibration if self.fusion_type == "bprr" else None,
            "bprr_temperature_count": len(self.modalities) if self.bprr_temperature is not None else 0,
            "router_supervision": self.router_supervision if self.fusion_type == "supervised_router" else None,
            "router_distill_weight": self.router_distill_weight if self.fusion_type == "supervised_router" else None,
            "router_distill_temperature": self.router_distill_temperature if self.fusion_type == "supervised_router" else None,
            "router_focus_patterns": self.router_focus_patterns if self.fusion_type == "supervised_router" else None,
            "router_fuse_level": self.router_fuse_level if self.fusion_type == "supervised_router" else None,
            "temporal_pooling": dict(self.temporal_pooling_config),
            "temporal_pooling_type": self.temporal_pooling_type if self.temporal_pooling_enabled else None,
            "temporal_pooling_param_count": self._temporal_pooling_parameter_count(),
            "temporal_pooling_scorer_features": list(TEMPORAL_SCORER_STATISTIC_NAMES)
            if self.temporal_pooling_type == "gap_aware_residual" and self.temporal_pooling_enabled
            else [],
            "temporal_pooling_residual_gate": "tanh_per_modality_zero_initialized"
            if self.temporal_residual_gate is not None
            else None,
            "temporal_pooling_recency_decay": self.temporal_pooling_config["recency_decay"]
            if self.temporal_pooling_type == "fixed_recency" and self.temporal_pooling_enabled
            else None,
            "use_mask_statistics": self.use_mask_statistics,
            "router_mask_statistic_features": list(TEMPORAL_MASK_STATISTIC_NAMES)
            if self.use_mask_statistics
            else [],
            "coverage_shrinkage": dict(self.coverage_shrinkage_config),
            "coverage_shrinkage_param_count": sum(
                param.numel() for param in self.coverage_shrinkage_net.parameters()
            )
            if self.coverage_shrinkage_net is not None
            else 0,
            "coverage_shrinkage_rho_max": self.coverage_shrinkage_config["rho_max"]
            if self.coverage_shrinkage_enabled
            else None,
            "total_params": self._total_param_count,
            "trainable_params": self._trainable_param_count,
            "context_type": self.context_type,
            "use_registry_encoders": self.use_registry_encoders,
            "encoder_configs": self.encoder_configs,
            "encoder_checkpoint_loads": self.encoder_checkpoint_loads,
        }

    def _encode(self, modality: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"u_mask_beam_jepa requires {modality}_batch for enabled modalities {list(self.modalities)}.")
        features = self.encoders[modality](value)
        if features.ndim == 3:
            features = features.mean(dim=1)
        elif features.ndim != 2:
            raise ValueError(f"{modality} encoder must return [B,T,D] or [B,D], got {tuple(features.shape)}.")
        return self.encoder_projections[modality](features)

    def _encode_sequence(self, modality: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"u_mask_beam_jepa requires {modality}_batch for enabled modalities {list(self.modalities)}.")
        features = self.encoders[modality](value)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        elif features.ndim != 3:
            raise ValueError(f"{modality} encoder must return [B,T,D] or [B,D], got {tuple(features.shape)}.")
        return self.encoder_projections[modality](features)

    def _resolve_modality_temporal_mask(
        self,
        latent_sequence: torch.Tensor,
        *,
        modality_mask: torch.Tensor,
        temporal_mask: torch.Tensor | None,
        modality_temporal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, steps, num_modalities, _ = latent_sequence.shape
        expected = (batch_size, steps, num_modalities)
        if modality_temporal_mask is None:
            mask = torch.ones(expected, dtype=torch.bool, device=latent_sequence.device)
        else:
            mask = torch.as_tensor(modality_temporal_mask, device=latent_sequence.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            if tuple(mask.shape) != expected:
                raise ValueError(f"modality_temporal_mask must have shape {expected}, got {tuple(mask.shape)}.")
        mask = mask & modality_mask.unsqueeze(1)
        if temporal_mask is not None:
            time_mask = torch.as_tensor(temporal_mask, device=latent_sequence.device, dtype=torch.bool)
            if time_mask.ndim == 1:
                time_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
            if tuple(time_mask.shape) != (batch_size, steps):
                raise ValueError(f"temporal_mask must have shape {(batch_size, steps)}, got {tuple(time_mask.shape)}.")
            mask = mask & time_mask.unsqueeze(-1)
        empty = (~mask.any(dim=(1, 2))).nonzero(as_tuple=False).flatten()
        if int(empty.numel()) > 0:
            raise ValueError(
                "modality_temporal_mask has no available cells for sample indices "
                f"{empty.detach().cpu().tolist()}."
            )
        return mask

    def _pool_temporal_sequence(
        self,
        latent_sequence: torch.Tensor,
        mask: torch.Tensor,
        statistics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask.to(device=latent_sequence.device, dtype=latent_sequence.dtype)
        if self.temporal_pooling_type == "fixed_recency":
            age = torch.arange(
                int(latent_sequence.shape[1]) - 1,
                -1,
                -1,
                device=latent_sequence.device,
                dtype=latent_sequence.dtype,
            )
            scores = -float(self.temporal_pooling_config["recency_decay"]) * age.view(1, -1, 1)
            weights = _masked_temporal_softmax(scores.expand_as(valid), mask)
        else:
            weights = valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (latent_sequence * weights.unsqueeze(-1)).sum(dim=1)
        if self.temporal_pooling_type != "gap_aware_residual":
            return mean, weights
        if (
            self.temporal_content_projection is None
            or self.temporal_statistics_projection is None
            or self.temporal_score_projection is None
            or self.temporal_residual_gate is None
        ):
            raise RuntimeError("gap_aware_residual pooling requested without scorer parameters.")
        scores = self.temporal_score_projection(
            torch.tanh(
                self.temporal_content_projection(latent_sequence)
                + self.temporal_statistics_projection(_temporal_scorer_statistics(mask, statistics))
            )
        ).squeeze(-1)
        temporal_weights = _masked_temporal_softmax(scores, mask)
        residual = (temporal_weights.unsqueeze(-1) * (latent_sequence - mean.unsqueeze(1))).sum(dim=1)
        gate = torch.tanh(self.temporal_residual_gate).view(1, -1, 1)
        return mean + gate * residual, temporal_weights

    def _temporal_pooling_diagnostics(
        self,
        mask: torch.Tensor,
        statistics: torch.Tensor,
        weights: torch.Tensor,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "temporal_pooling_type": self.temporal_pooling_type,
            "temporal_pooling_weights": weights.detach(),
            "temporal_pooling_param_count": self._temporal_pooling_parameter_count(),
            "modality_temporal_mask": mask.detach(),
            "temporal_mask": mask.any(dim=2).detach(),
            "modality_mask": mask.any(dim=1).detach(),
            "available_modalities": mask.any(dim=1).detach(),
            "temporal_mask_statistics": statistics.detach(),
            "temporal_mask_statistic_names": TEMPORAL_MASK_STATISTIC_NAMES,
            "temporal_scorer_statistic_names": TEMPORAL_SCORER_STATISTIC_NAMES,
            "temporal_coverage": statistics[..., 0].detach(),
            "temporal_last_age": statistics[..., 1].detach(),
            "temporal_longest_gap": statistics[..., 2].detach(),
            "temporal_trailing_gap": statistics[..., 3].detach(),
            "temporal_missing_block_count": statistics[..., 4].detach(),
        }
        if self.temporal_pooling_type == "fixed_recency":
            diagnostics["temporal_recency_decay"] = float(self.temporal_pooling_config["recency_decay"])
        if self.temporal_residual_gate is not None:
            diagnostics["temporal_residual_gate"] = torch.tanh(self.temporal_residual_gate).detach()
        return diagnostics

    def _temporal_pooling_parameter_count(self) -> int:
        modules = (
            self.temporal_content_projection,
            self.temporal_statistics_projection,
            self.temporal_score_projection,
        )
        count = sum(param.numel() for module in modules if module is not None for param in module.parameters())
        if self.temporal_residual_gate is not None:
            count += int(self.temporal_residual_gate.numel())
        return count

    def _load_encoder_checkpoints(self, paths: dict[str, str]) -> dict[str, Any]:
        if not paths:
            return {}
        loads: dict[str, Any] = {}
        for modality, raw_path in paths.items():
            if modality not in self.encoders:
                raise ValueError(f"encoder checkpoint configured for disabled modality '{modality}'.")
            checkpoint_path = Path(str(raw_path))
            checkpoint = load_torch_payload(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state_dict, dict):
                raise ValueError(f"Encoder checkpoint {checkpoint_path} does not contain a state dict.")
            prefix = f"encoders.{modality}."
            target = self.encoders[modality].state_dict()
            matched = {}
            skipped = []
            for key, tensor in state_dict.items():
                if not key.startswith(prefix):
                    continue
                local_key = key[len(prefix):]
                if local_key in target and tuple(target[local_key].shape) == tuple(tensor.shape):
                    matched[local_key] = tensor
                else:
                    skipped.append(local_key)
            if not matched:
                raise ValueError(f"No matching encoder weights for '{modality}' in {checkpoint_path}.")
            incompatible = self.encoders[modality].load_state_dict(matched, strict=False)
            loads[modality] = {
                "path": str(checkpoint_path),
                "loaded_keys": len(matched),
                "missing_keys": sorted(incompatible.missing_keys),
                "unexpected_keys": sorted(incompatible.unexpected_keys),
                "skipped_keys": sorted(skipped),
            }
        return loads

    def _resolve_mask(
        self,
        missing_mask: torch.Tensor | None,
        latent: torch.Tensor,
        *,
        allow_all_missing: bool = False,
    ) -> torch.Tensor:
        if missing_mask is None and (not self.training) and self.eval_missing_pattern:
            from kd_sensing.data.missing_mask import make_pattern_mask

            missing_mask = make_pattern_mask(
                int(latent.shape[0]),
                self.modalities,
                available_modalities=self.eval_missing_pattern.get("available_modalities"),
                pattern_mask=self.eval_missing_pattern.get("pattern_mask"),
                device=latent.device,
            )
        if missing_mask is None:
            mask = torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device)
        else:
            mask = missing_mask.to(device=latent.device, dtype=torch.bool)
        expected = (int(latent.shape[0]), len(self.modalities))
        if tuple(mask.shape) == (len(self.modalities),):
            mask = mask.unsqueeze(0).expand(expected)
        if tuple(mask.shape) != expected:
            raise ValueError(f"missing_mask must have shape {expected}, got {tuple(mask.shape)}.")
        empty = (~mask.any(dim=1)).nonzero(as_tuple=False).flatten()
        if int(empty.numel()) > 0 and not allow_all_missing:
            raise ValueError(f"missing_mask has no available modalities for sample indices {empty.detach().cpu().tolist()}.")
        return mask

    def _modality_reliability(self, latent: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = [self.reliability_heads[name](latent[:, index, :]) for index, name in enumerate(self.modalities)]
        modality_mu_b = torch.stack([item[0] for item in outputs], dim=1)
        modality_logvar_b = torch.stack([item[1] for item in outputs], dim=1)
        mask_values = mask.unsqueeze(-1).to(dtype=latent.dtype)
        if not self.use_modality_uncertainty:
            return mask_values, modality_mu_b, modality_logvar_b
        reliability = torch.exp(-F.softplus(modality_logvar_b).mean(dim=-1, keepdim=True))
        return reliability * mask_values, modality_mu_b, modality_logvar_b

    def _fuse(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        mu_b: torch.Tensor,
        global_reliability: torch.Tensor,
        *,
        mask_statistics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.fusion_type == "reliability_biased_missing_attention":
            result = self.rbma_fusion(
                latent,
                mask,
                reliability,
                global_token=mu_b if self.use_jepa_loss else None,
                global_reliability=global_reliability if self.use_jepa_loss else None,
            )
            diagnostics = dict(result["diagnostics"])
            diagnostics["rbma_mask_provenance"] = "missing_mask"
            diagnostics["rbma_modality_reliability_mean"] = reliability.detach().mean(dim=(1, 2))
            return result["fused"], diagnostics
        if self.fusion_type == "reliability_gated_cross_attention":
            return self.cross_attention_fusion(latent, mu_b, reliability, global_reliability), {}
        if self.fusion_type == "pcpg":
            return self._pcpg_fuse(latent, mask, reliability, mu_b)
        if self.fusion_type == "average":
            return self._average_fuse(latent, mask, reliability)
        if self.fusion_type == "raw_conf_gate":
            return self._raw_conf_gate_fuse(latent, mask, reliability, mu_b)
        if self.fusion_type == "bprr":
            return self._bprr_fuse(latent, mask, reliability, mu_b)
        if self.fusion_type == "supervised_router":
            return self._supervised_router_fuse(
                latent,
                mask,
                reliability,
                mu_b,
                mask_statistics=mask_statistics,
            )
        weights = reliability.squeeze(-1)
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / weight_sum.clamp_min(1e-6)
        pooled = (latent * weights.unsqueeze(-1)).sum(dim=1)
        pooled = torch.where(weight_sum.gt(0), pooled, mu_b)
        diagnostics = {
            "reliability_fusion_mode": self.fusion_type,
            "reliability_fusion_weights": weights.detach(),
            "reliability_fusion_available_mask": mask.detach(),
            "reliability_fusion_weight_sum": weights.detach().sum(dim=1),
        }
        if self.fusion_type == "weighted_sum":
            return 0.5 * (pooled + mu_b), diagnostics
        return self.concat_fusion(torch.cat([pooled, mu_b], dim=-1)), diagnostics

    def _average_fuse(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        stats = self._unimodal_branch_stats(latent)
        available = mask.to(device=latent.device, dtype=latent.dtype)
        counts = available.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = available / counts
        pooled = (latent * weights.unsqueeze(-1)).sum(dim=1)
        fused_logits = (stats["unimodal_logits"] * weights.unsqueeze(-1)).sum(dim=1)
        diagnostics = self._router_diagnostics(
            mode="average",
            weights=weights,
            mask=mask,
            stats=stats,
            reliability=reliability,
            fused_logits=fused_logits,
            pattern_features=_pattern_features(available),
        )
        diagnostics["average_fusion"] = 1.0
        return pooled, diagnostics

    def _pcpg_fuse(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        mu_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.pcpg_gate is None:
            raise RuntimeError("pcpg fusion requested without a pcpg gate.")
        stats = self._unimodal_branch_stats(latent)
        unimodal_logits = stats["unimodal_logits"]
        prototype_scores = stats["prototype_scores"]
        available = mask.to(device=latent.device, dtype=latent.dtype)
        reliability_features = torch.stack(
            [
                reliability.squeeze(-1),
                available,
                stats["entropy"],
                stats["margin"],
                stats["confidence"],
                stats["prototype_confidence"],
            ],
            dim=-1,
        )
        pattern_features = _pattern_features(available)
        weights = self.pcpg_gate(reliability_features, pattern_features, mask)
        pooled = (latent * weights.unsqueeze(-1)).sum(dim=1)
        weighted_logits = (unimodal_logits * weights.unsqueeze(-1)).sum(dim=1)
        weighted_prototypes = (prototype_scores * weights.unsqueeze(-1)).sum(dim=1)
        fused_logits = self._combine_decision_and_prototype(weighted_logits, weighted_prototypes)
        diagnostics = {
            "reliability_fusion_mode": "pcpg",
            "reliability_fusion_weights": weights.detach(),
            "reliability_fusion_available_mask": mask.detach(),
            "reliability_fusion_weight_sum": weights.detach().sum(dim=1),
            "pcpg_gate_weights": weights.detach(),
            "pcpg_available_mask": mask.detach(),
            "pcpg_pattern_features": pattern_features.detach(),
            "pcpg_unimodal_logits": unimodal_logits,
            "unimodal_logits": unimodal_logits,
            "pcpg_unimodal_prototype_scores": prototype_scores,
            "unimodal_prototype_scores": prototype_scores,
            "pcpg_unimodal_entropy": stats["entropy"].detach(),
            "pcpg_unimodal_margin": stats["margin"].detach(),
            "pcpg_gate_mean": weights.detach().mean(dim=0),
            "pcpg_fused_logits": fused_logits,
            "fused_logits": fused_logits,
            "head_type": self.head_type,
            "prototype_margin_enabled": self._prototype_margin_enabled(),
        }
        return 0.5 * (pooled + mu_b), diagnostics

    def _supervised_router_fuse(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        mu_b: torch.Tensor,
        *,
        mask_statistics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.bprr_router is None:
            raise RuntimeError("supervised_router fusion requested without a router.")
        stats = self._unimodal_branch_stats(latent)
        feature_logits = stats["unimodal_logits"]
        if self.bprr_temperature is not None:
            feature_logits = self.bprr_temperature(feature_logits)
            feature_stats = self._logit_reliability_stats(feature_logits, stats["prototype_scores"])
        else:
            feature_stats = stats
        available = mask.to(device=latent.device, dtype=latent.dtype)
        reliability_features = self._router_reliability_features(
            feature_stats,
            available,
            reliability,
            mask_statistics=mask_statistics,
        )
        pattern_features = _pattern_features(available) if self.router_use_pattern_features else None
        gate_logits = self.bprr_router.forward_logits(reliability_features, mask, pattern_features)
        pre_shrinkage_weights = masked_pcpg_softmax(gate_logits, mask)
        weights, shrinkage_diagnostics = self._apply_coverage_shrinkage(
            pre_shrinkage_weights,
            mask,
            mask_statistics,
        )
        pooled = (latent * weights.unsqueeze(-1)).sum(dim=1)
        weighted_logits = (stats["unimodal_logits"] * weights.unsqueeze(-1)).sum(dim=1)
        diagnostics = self._router_diagnostics(
            mode="supervised_router",
            weights=weights,
            mask=mask,
            stats=stats,
            reliability=reliability,
            fused_logits=weighted_logits,
            pattern_features=pattern_features
            if pattern_features is not None
            else torch.zeros(available.shape[0], available.shape[1] + 2, device=latent.device, dtype=latent.dtype),
        )
        diagnostics.update(
            {
                "supervised_router_gate_logits": gate_logits,
                "router_gate_logits": gate_logits,
                "supervised_router_reliability_features": reliability_features.detach(),
                "supervised_router_feature_names": self._router_feature_names(),
                "router_use_pattern_features": self.router_use_pattern_features,
                "router_use_reliability_features": self.router_use_reliability_features,
                "router_use_prototype_margin": self.router_use_prototype_margin,
                "router_use_entropy": self.router_use_entropy,
                "router_use_confidence": self.router_use_confidence,
                "router_use_logit_norm": self.router_use_logit_norm,
                "router_pattern_feature_fallback": 0.0 if self.router_use_pattern_features else 1.0,
                "prototype_margin_enabled": self._prototype_margin_enabled(),
                "prototype_margin_fallback": 0.0 if self._prototype_margin_enabled() else 1.0,
                "head_type": self.head_type,
                "router_supervision": self.router_supervision,
                "router_distill_weight": float(self.router_distill_weight),
                "router_distill_temperature": float(self.router_distill_temperature),
                "router_focus_patterns": self.router_focus_patterns,
                "router_fuse_level": self.router_fuse_level,
                **shrinkage_diagnostics,
            }
        )
        if self.bprr_temperature is not None:
            temperatures = self.bprr_temperature.temperatures().to(device=latent.device, dtype=latent.dtype)
            diagnostics["supervised_router_modality_temperatures"] = temperatures.detach()
        return 0.5 * (pooled + mu_b), diagnostics

    def _raw_conf_gate_fuse(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        mu_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        stats = self._unimodal_branch_stats(latent)
        weights = masked_pcpg_softmax(stats["margin"] / self.raw_conf_temperature, mask)
        pooled = (latent * weights.unsqueeze(-1)).sum(dim=1)
        weighted_logits = (stats["unimodal_logits"] * weights.unsqueeze(-1)).sum(dim=1)
        diagnostics = self._router_diagnostics(
            mode="raw_conf_gate",
            weights=weights,
            mask=mask,
            stats=stats,
            reliability=reliability,
            fused_logits=weighted_logits,
            pattern_features=_pattern_features(mask.to(device=latent.device, dtype=latent.dtype)),
        )
        diagnostics["raw_conf_temperature"] = float(self.raw_conf_temperature)
        diagnostics["raw_conf_gate_scores"] = (stats["margin"] / self.raw_conf_temperature).detach()
        diagnostics["raw_conf_gate_weights"] = weights.detach()
        return 0.5 * (pooled + mu_b), diagnostics

    def _bprr_fuse(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        mu_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.bprr_router is None:
            raise RuntimeError("bprr fusion requested without a BPRR router.")
        stats = self._unimodal_branch_stats(latent)
        feature_logits = stats["unimodal_logits"]
        if self.bprr_temperature is not None:
            # The first BPRR implementation calibrates logits only for router reliability
            # features; final fused logits stay on the model's original classification scale.
            feature_logits = self.bprr_temperature(feature_logits)
            feature_stats = self._logit_reliability_stats(feature_logits, stats["prototype_scores"])
        else:
            feature_stats = stats
        available = mask.to(device=latent.device, dtype=latent.dtype)
        zero_proto_distance = torch.zeros_like(feature_stats["margin"])
        reliability_features = torch.stack(
            [
                available,
                feature_stats["entropy"],
                feature_stats["margin"],
                feature_stats["confidence"],
                feature_stats["logit_norm"],
                zero_proto_distance,
                feature_stats["prototype_margin"],
                reliability.squeeze(-1),
            ],
            dim=-1,
        )
        pattern_features = _pattern_features(available)
        weights = self.bprr_router(reliability_features, mask, pattern_features)
        pooled = (latent * weights.unsqueeze(-1)).sum(dim=1)
        weighted_logits = (stats["unimodal_logits"] * weights.unsqueeze(-1)).sum(dim=1)
        diagnostics = self._router_diagnostics(
            mode="bprr",
            weights=weights,
            mask=mask,
            stats=stats,
            reliability=reliability,
            fused_logits=weighted_logits,
            pattern_features=pattern_features,
        )
        diagnostics.update(
            {
                "bprr_reliability_features": reliability_features.detach(),
                "bprr_feature_names": (
                    "availability",
                    "entropy",
                    "top1_top2_margin",
                    "max_prob",
                    "logit_norm",
                    "prototype_min_distance",
                    "prototype_margin",
                    "modality_reliability",
                ),
                "bprr_prototype_distance_available": 0.0,
                "bprr_prototype_distance_todo": "prototype_min_distance falls back to zero until per-modality distances are exposed",
                "bprr_calibration": self.bprr_calibration,
                "bprr_calibration_applies_to": "router_reliability_features",
            }
        )
        if self.bprr_temperature is not None:
            temperatures = self.bprr_temperature.temperatures().to(device=latent.device, dtype=latent.dtype)
            diagnostics["bprr_modality_temperatures"] = temperatures.detach()
            for index, modality in enumerate(self.modalities):
                diagnostics[f"bprr_temperature_{modality}"] = float(temperatures[index].detach().cpu().item())
        return 0.5 * (pooled + mu_b), diagnostics

    def _head_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.head_type == "prototype":
            return self.prototype_bank(features)
        return self.beam_head(features)

    def _prototype_scores(self, features: torch.Tensor) -> torch.Tensor:
        if self.head_type == "classifier":
            return self.beam_head(features).detach().new_zeros(features.shape[0], self.num_classes)
        return self.prototype_bank(features)

    def _combine_decision_and_prototype(self, logits: torch.Tensor, prototype_scores: torch.Tensor) -> torch.Tensor:
        if self.head_type == "legacy":
            return 0.5 * (logits + prototype_scores)
        return logits

    def _prototype_margin_enabled(self) -> bool:
        return self.head_type != "classifier"

    def _router_reliability_features(
        self,
        feature_stats: dict[str, torch.Tensor],
        available: torch.Tensor,
        reliability: torch.Tensor,
        *,
        mask_statistics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zero = torch.zeros_like(feature_stats["margin"])
        use_rel = self.router_use_reliability_features
        use_proto_margin = use_rel and self.router_use_prototype_margin and self._prototype_margin_enabled()
        features = torch.stack(
            [
                available if self.router_use_pattern_features else zero,
                feature_stats["entropy"] if use_rel and self.router_use_entropy else zero,
                feature_stats["margin"] if use_rel and self.router_use_confidence else zero,
                feature_stats["confidence"] if use_rel and self.router_use_confidence else zero,
                feature_stats["logit_norm"] if use_rel and self.router_use_logit_norm else zero,
                zero,
                feature_stats["prototype_margin"] if use_proto_margin else zero,
                reliability.squeeze(-1) if use_rel else zero,
            ],
            dim=-1,
        )
        if not self.use_mask_statistics:
            return features
        if mask_statistics is None:
            raise ValueError("use_mask_statistics=true requires modality temporal mask statistics.")
        return torch.cat([features, mask_statistics.to(device=features.device, dtype=features.dtype)], dim=-1)

    def _router_feature_names(self) -> tuple[str, ...]:
        names = (
            "availability" if self.router_use_pattern_features else "availability_disabled",
            "entropy" if self.router_use_reliability_features and self.router_use_entropy else "entropy_disabled",
            "top1_top2_margin" if self.router_use_reliability_features and self.router_use_confidence else "top1_top2_margin_disabled",
            "max_prob" if self.router_use_reliability_features and self.router_use_confidence else "max_prob_disabled",
            "logit_norm" if self.router_use_reliability_features and self.router_use_logit_norm else "logit_norm_disabled",
            "prototype_min_distance_disabled",
            "prototype_margin"
            if self.router_use_reliability_features and self.router_use_prototype_margin and self._prototype_margin_enabled()
            else "prototype_margin_disabled",
            "modality_reliability" if self.router_use_reliability_features else "modality_reliability_disabled",
        )
        if not self.use_mask_statistics:
            return names
        return names + tuple(f"temporal_{name}" for name in TEMPORAL_MASK_STATISTIC_NAMES)

    def _apply_coverage_shrinkage(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor,
        mask_statistics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.coverage_shrinkage_enabled:
            return weights, {"coverage_shrinkage_enabled": False}
        if self.coverage_shrinkage_net is None or mask_statistics is None:
            raise ValueError("coverage_shrinkage.enabled=true requires temporal mask statistics.")
        mean_coverage = mask_statistics[..., 0].mean(dim=1)
        gate_entropy = _gate_entropy(weights)
        topk = weights.topk(min(2, int(weights.shape[1])), dim=1).values
        gate_margin = topk[:, 0] - (topk[:, 1] if int(topk.shape[1]) > 1 else 0.0)
        shrinkage_features = torch.stack([mean_coverage, gate_entropy, gate_margin], dim=-1)
        rho = (
            float(self.coverage_shrinkage_config["rho_max"])
            * (1.0 - mean_coverage)
            * torch.sigmoid(self.coverage_shrinkage_net(shrinkage_features).squeeze(-1))
        )
        available = mask.to(device=weights.device, dtype=weights.dtype)
        uniform = available / available.sum(dim=1, keepdim=True).clamp_min(1.0)
        shrunk = (1.0 - rho.unsqueeze(-1)) * weights + rho.unsqueeze(-1) * uniform
        single = mask.sum(dim=1, keepdim=True) == 1
        shrunk = torch.where(single, weights, shrunk)
        return shrunk, {
            "coverage_shrinkage_enabled": True,
            "coverage_shrinkage_rho": rho.detach(),
            "coverage_shrinkage_rho_max": float(self.coverage_shrinkage_config["rho_max"]),
            "coverage_shrinkage_mean_coverage": mean_coverage.detach(),
            "coverage_shrinkage_gate_entropy": gate_entropy.detach(),
            "coverage_shrinkage_gate_margin": gate_margin.detach(),
            "coverage_shrinkage_pre_weights": weights.detach(),
            "coverage_shrinkage_uniform_weights": uniform.detach(),
        }

    def _unimodal_branch_stats(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_modalities, feature_dim = latent.shape
        flat = latent.reshape(batch_size * num_modalities, feature_dim)
        classifier_logits = self.beam_head(flat).view(batch_size, num_modalities, self.num_classes)
        prototype_scores = self._prototype_scores(flat).view(batch_size, num_modalities, self.num_classes)
        unimodal_logits = prototype_scores if self.head_type == "prototype" else classifier_logits
        return {
            **self._logit_reliability_stats(unimodal_logits, prototype_scores),
            "unimodal_logits": unimodal_logits,
            "classifier_logits": classifier_logits,
            "prototype_scores": prototype_scores,
        }

    def _logit_reliability_stats(
        self,
        unimodal_logits: torch.Tensor,
        prototype_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        probabilities = torch.softmax(unimodal_logits, dim=-1)
        entropy_scale = max(math.log(max(self.num_classes, 2)), 1e-6)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1) / entropy_scale
        topk = probabilities.topk(min(2, self.num_classes), dim=-1).values
        margin = topk[..., 0] - (topk[..., 1] if topk.shape[-1] > 1 else 0.0)
        confidence = topk[..., 0]
        if self._prototype_margin_enabled():
            prototype_probs = torch.softmax(prototype_scores, dim=-1)
            proto_topk = prototype_probs.topk(min(2, self.num_classes), dim=-1).values
            prototype_confidence = prototype_probs.amax(dim=-1)
            prototype_margin = proto_topk[..., 0] - (proto_topk[..., 1] if proto_topk.shape[-1] > 1 else 0.0)
        else:
            prototype_confidence = torch.zeros_like(confidence)
            prototype_margin = torch.zeros_like(margin)
        return {
            "entropy": entropy,
            "margin": margin,
            "confidence": confidence,
            "logit_norm": unimodal_logits.norm(dim=-1),
            "prototype_confidence": prototype_confidence,
            "prototype_margin": prototype_margin,
        }

    def _router_diagnostics(
        self,
        *,
        mode: str,
        weights: torch.Tensor,
        mask: torch.Tensor,
        stats: dict[str, torch.Tensor],
        reliability: torch.Tensor,
        fused_logits: torch.Tensor,
        pattern_features: torch.Tensor,
    ) -> dict[str, Any]:
        gate_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return {
            "reliability_fusion_mode": mode,
            "reliability_fusion_weights": weights.detach(),
            "reliability_fusion_available_mask": mask.detach(),
            "reliability_fusion_weight_sum": weights.detach().sum(dim=1),
            "unimodal_logits": stats["unimodal_logits"],
            "unimodal_prototype_scores": stats["prototype_scores"],
            f"{mode}_gate_weights": weights.detach(),
            f"{mode}_available_mask": mask.detach(),
            f"{mode}_unimodal_entropy": stats["entropy"].detach(),
            f"{mode}_unimodal_margin": stats["margin"].detach(),
            f"{mode}_max_prob": stats["confidence"].detach(),
            f"{mode}_logit_norm": stats["logit_norm"].detach(),
            f"{mode}_prototype_margin": stats["prototype_margin"].detach(),
            f"{mode}_pattern_features": pattern_features.detach(),
            f"{mode}_gate_mean": weights.detach().mean(dim=0),
            f"{mode}_gate_entropy": gate_entropy.detach(),
            "gate_entropy": gate_entropy.detach(),
            "modality_reliability": reliability.detach(),
            "fused_logits": fused_logits,
        }


def _validate_modalities(modalities: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(str(item) for item in modalities)
    if not values:
        raise ValueError(f"u_mask_beam_jepa modalities must be non-empty. Available canonical modalities: {list(DEFAULT_MODALITIES)}.")
    duplicates = sorted({item for item in values if values.count(item) > 1})
    invalid = [item for item in values if item not in DEFAULT_MODALITIES]
    if duplicates or invalid:
        raise ValueError(
            "Invalid u_mask_beam_jepa modalities. "
            f"duplicates={duplicates}, invalid={invalid}, available canonical modalities={list(DEFAULT_MODALITIES)}, "
            f"project canonical order={list(MODALITY_ORDER)}."
        )
    return values


def _pattern_features(available: torch.Tensor) -> torch.Tensor:
    if available.ndim != 2:
        raise ValueError(f"pattern features expect available mask [B, M], got {tuple(available.shape)}.")
    values = available.to(dtype=torch.float32)
    available_fraction = values.mean(dim=1, keepdim=True)
    return torch.cat([values, available_fraction, 1.0 - available_fraction], dim=-1)


def _gate_entropy(weights: torch.Tensor) -> torch.Tensor:
    return -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)


def _modality_gate_diagnostics(weights: torch.Tensor, modalities: tuple[str, ...]) -> dict[str, float]:
    return {
        f"mean_gate_{modality}": float(weights[:, index].detach().mean().cpu().item())
        for index, modality in enumerate(modalities)
    }


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "none", ""}
    return bool(value)


def _temporal_pooling_config(raw: dict[str, Any] | bool | None) -> dict[str, Any]:
    if raw in (None, False, "", "none"):
        return {"enabled": False, "type": "masked_mean", "recency_decay": 1.0, "hidden_dim": 32}
    if not isinstance(raw, dict):
        raise ValueError(f"temporal_pooling must be a mapping or false/null, got {type(raw).__name__}.")
    cfg = {
        "enabled": _bool(raw.get("enabled", False)),
        "type": str(raw.get("type", "masked_mean")).strip().lower(),
        "recency_decay": float(raw.get("recency_decay", 1.0)),
        "hidden_dim": int(raw.get("hidden_dim", 32)),
    }
    if cfg["type"] not in {"masked_mean", "fixed_recency", "gap_aware_residual"}:
        raise ValueError("temporal_pooling.type must be masked_mean, fixed_recency, or gap_aware_residual.")
    if not math.isfinite(cfg["recency_decay"]) or cfg["recency_decay"] < 0:
        raise ValueError("temporal_pooling.recency_decay must be finite and non-negative.")
    if cfg["hidden_dim"] <= 0:
        raise ValueError("temporal_pooling.hidden_dim must be positive.")
    return cfg


def _coverage_shrinkage_config(raw: dict[str, Any] | bool | None) -> dict[str, Any]:
    if raw in (None, False, "", "none"):
        return {"enabled": False, "rho_max": 0.5, "hidden_dim": 16}
    if not isinstance(raw, dict):
        raise ValueError(f"coverage_shrinkage must be a mapping or false/null, got {type(raw).__name__}.")
    cfg = {
        "enabled": _bool(raw.get("enabled", False)),
        "rho_max": float(raw.get("rho_max", 0.5)),
        "hidden_dim": int(raw.get("hidden_dim", 16)),
    }
    if not math.isfinite(cfg["rho_max"]) or not 0.0 <= cfg["rho_max"] <= 1.0:
        raise ValueError("coverage_shrinkage.rho_max must be finite and within [0, 1].")
    if cfg["hidden_dim"] <= 0:
        raise ValueError("coverage_shrinkage.hidden_dim must be positive.")
    return cfg


def _pattern_film_config(raw: dict[str, Any] | bool | None) -> dict[str, Any]:
    if raw in (None, False, "", "none"):
        return {}
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, dict):
        raise ValueError(f"pattern_film must be a mapping, bool, or null, got {type(raw).__name__}.")
    cfg = dict(raw)
    if not bool(cfg.get("enabled", False)):
        return {}
    cfg.setdefault("dim", 8)
    cfg.setdefault("init_identity", True)
    cfg.setdefault("apply_at", "pre_head")
    if int(cfg["dim"]) <= 0:
        raise ValueError(f"pattern_film.dim must be positive, got {cfg['dim']}.")
    if str(cfg["apply_at"]) != "pre_head":
        raise ValueError("pattern_film.apply_at currently supports only 'pre_head'.")
    return cfg


def _validate_context_type(value: str) -> None:
    if value in {"set_transformer_simplified", "beam_query_transformer"}:
        return
    if value == "mask_transformer":
        raise ValueError("context_type='mask_transformer' is not implemented for u_mask_beam_jepa.")
    raise ValueError("context_type must be set_transformer_simplified or beam_query_transformer.")
