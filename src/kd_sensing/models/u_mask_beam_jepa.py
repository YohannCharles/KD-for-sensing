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


DEFAULT_MODALITIES = ("image", "radar", "lidar", "gps")


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


class TemporalScalarRouter(nn.Module):
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"temporal scalar router expected feature_dim={self.feature_dim}, got {tuple(features.shape)}.")
        return self.net(features).squeeze(-1)


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


class LatentPredictionProbe(nn.Module):
    def __init__(self, d_model: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


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
        use_full_to_partial_kd: bool = False,
        kd_teacher_mode: str = "disabled",
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
        temporal_router_type: str = "none",
        temporal_router_distill_weight: float = 0.0,
        temporal_aggregation: str = "masked_mean",
        head_type: str = "legacy",
        use_light_latent_pred: bool = False,
        latent_pred_target: str = "full_fused",
        latent_pred_hidden_dim: int = 256,
        ablation_id: str | None = None,
        encoders: dict[str, Any] | None = None,
        encoder_checkpoint_paths: dict[str, str] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
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
        self.use_full_to_partial_kd = bool(use_full_to_partial_kd)
        self.kd_teacher_mode = str(kd_teacher_mode)
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
        self.temporal_router_type = str(temporal_router_type or "none").strip().lower()
        self.temporal_router_distill_weight = float(temporal_router_distill_weight)
        self.temporal_aggregation = str(temporal_aggregation or "masked_mean").strip().lower()
        self.head_type = str(head_type or "legacy").strip().lower()
        self.use_light_latent_pred = bool(use_light_latent_pred)
        self.latent_pred_target = str(latent_pred_target)
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
        if self.router_supervision not in {"oracle", "pattern_best", "none"}:
            raise ValueError("router_supervision must be one of oracle, pattern_best, or none.")
        if self.temporal_router_type not in {
            "none",
            "s1_temporalagg_modality",
            "s2_pertime_modality",
            "s3_two_level",
            "s4_global",
        }:
            raise ValueError(
                "temporal_router_type must be none, s1_temporalagg_modality, "
                "s2_pertime_modality, s3_two_level, or s4_global."
            )
        if self.temporal_aggregation not in {"masked_mean", "attention"}:
            raise ValueError("temporal_aggregation must be masked_mean or attention.")
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
                feature_dim=8,
                pattern_dim=len(self.modalities) + 2,
                hidden_dim=int(bprr_hidden_dim),
                dropout=float(bprr_dropout),
            )
            if self.fusion_type in {"bprr", "supervised_router"}
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
        self.temporal_router = (
            TemporalScalarRouter(feature_dim=8, hidden_dim=int(bprr_hidden_dim), dropout=float(bprr_dropout))
            if self.temporal_router_type == "s3_two_level"
            else None
        )
        self.global_temporal_router = (
            TemporalScalarRouter(feature_dim=8, hidden_dim=int(bprr_hidden_dim), dropout=float(bprr_dropout))
            if self.temporal_router_type == "s4_global"
            else None
        )
        latent_output_dim = self.num_classes if self.latent_pred_target == "prototype_distribution" else self.d_model
        self.latent_predictor = (
            LatentPredictionProbe(self.d_model, latent_output_dim, hidden_dim=int(latent_pred_hidden_dim))
            if self.use_light_latent_pred
            else None
        )

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
        if self.temporal_router_type != "none":
            return self._temporal_router_forward(
                inputs,
                missing_mask=missing_mask if missing_mask is not None else force_modality_mask,
                temporal_mask=temporal_mask,
                modality_temporal_mask=modality_temporal_mask,
                available_modalities=available_modalities,
            )
        latent = torch.stack([self._encode(name, inputs[name]) for name in self.modalities], dim=1)
        mask = self._resolve_mask(
            missing_mask if missing_mask is not None else force_modality_mask,
            latent,
            allow_all_missing=self.fusion_type in {"reliability_biased_missing_attention", "weighted_sum"},
        )
        reliability, modality_mu_b, modality_logvar_b = self._modality_reliability(latent, mask)
        u_star, teacher_logits = self.teacher(latent, self.modality_embedding)
        c_a = self.context_encoder(latent, mask, reliability, self.modality_embedding)
        mu_b, logvar_b = self.predictor(c_a)
        global_reliability = (
            torch.exp(-F.softplus(logvar_b).mean(dim=-1)) if self.use_global_uncertainty else torch.ones_like(mu_b[:, 0])
        )
        fused, fusion_diagnostics = self._fuse(latent, mask, reliability, mu_b, global_reliability)
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
            "consumes_missing_modality_metadata": False,
            "consumes_reliability_metadata": self.fusion_type in {"reliability_biased_missing_attention", "bprr", "supervised_router"},
            "reliability_metadata_consumption": "internal_modality_uncertainty"
            if self.fusion_type in {"reliability_biased_missing_attention", "bprr", "supervised_router"}
            else "none",
            "same_model_full_modal_teacher_auxiliary": True,
            "use_teacher": self.use_teacher,
            "use_jepa_loss": self.use_jepa_loss,
            "use_beam_prototype_alignment": self.use_beam_prototype_alignment,
            "use_full_to_partial_kd": self.use_full_to_partial_kd,
            "kd_teacher_mode": self.kd_teacher_mode,
            "teacher_checkpoint_pending_reason": "checkpoint teacher is not implemented"
            if self.kd_teacher_mode == "checkpoint"
            else None,
            "mask_sampler": self.mask_sampler,
            "use_mask_adapter": self.use_mask_adapter,
            "mask_adapter_apply": self.mask_adapter_apply,
            "mask_adapter_param_count": sum(param.numel() for param in self.mask_adapter.parameters()) if self.mask_adapter is not None else 0,
            "pattern_film": self.pattern_film_config or {"enabled": False},
            "pattern_film_param_count": sum(param.numel() for param in self.pattern_film.parameters()) if self.pattern_film is not None else 0,
            "use_light_latent_pred": self.use_light_latent_pred,
            "latent_pred_target": self.latent_pred_target,
            "ablation_id": self.ablation_id,
            "use_modality_uncertainty": self.use_modality_uncertainty,
            "use_global_uncertainty": self.use_global_uncertainty,
            "fusion_type": self.fusion_type,
            "temporal_router_type": self.temporal_router_type,
            "temporal_aggregation": self.temporal_aggregation if self.temporal_router_type != "none" else None,
            "temporal_router_distill_weight": self.temporal_router_distill_weight
            if self.temporal_router_type != "none"
            else None,
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

    def _temporal_router_forward(
        self,
        inputs: dict[str, torch.Tensor | None],
        *,
        missing_mask: torch.Tensor | None,
        temporal_mask: torch.Tensor | None,
        modality_temporal_mask: torch.Tensor | None,
        available_modalities: torch.Tensor | None,
    ) -> dict[str, Any]:
        latent_seq = torch.stack([self._encode_sequence(name, inputs[name]) for name in self.modalities], dim=2)
        # latent_seq: [B, T, M, D]
        mt_mask = self._resolve_modality_temporal_mask(
            latent_seq,
            missing_mask=missing_mask,
            temporal_mask=temporal_mask,
            modality_temporal_mask=modality_temporal_mask,
            available_modalities=available_modalities,
        )
        if self.temporal_router_type == "s1_temporalagg_modality":
            logits, fused, diagnostics = self._s1_temporalagg_modality(latent_seq, mt_mask)
        elif self.temporal_router_type == "s2_pertime_modality":
            logits, fused, diagnostics = self._s2_pertime_modality(latent_seq, mt_mask)
        elif self.temporal_router_type == "s3_two_level":
            logits, fused, diagnostics = self._s3_two_level(latent_seq, mt_mask)
        else:
            logits, fused, diagnostics = self._s4_global(latent_seq, mt_mask)
        modality_features = self._masked_modality_temporal_mean(latent_seq, mt_mask)
        modality_mask = mt_mask.any(dim=1)
        teacher_logits = self._head_logits(fused).unsqueeze(1).expand(-1, self.num_pred, -1)
        logits = logits.unsqueeze(1).expand(-1, self.num_pred, -1)
        zero_logvar = torch.zeros_like(modality_features)
        return {
            "logits": logits,
            "input_features": modality_features,
            "output_features": fused,
            "teacher_logits": teacher_logits,
            "u_star": fused.detach(),
            "mu_B": fused,
            "logvar_B": torch.zeros_like(fused),
            "modality_mu_B": modality_features,
            "modality_logvar_B": zero_logvar,
            "modality_reliability": modality_mask.unsqueeze(-1).to(dtype=fused.dtype),
            "global_reliability": torch.ones(fused.shape[0], device=fused.device, dtype=fused.dtype),
            "missing_mask": modality_mask,
            "modality_temporal_mask": mt_mask,
            "temporal_mask": mt_mask.any(dim=2),
            "modality_mask": modality_mask,
            "available_modalities": modality_mask,
            "modality_features": modality_features,
            "student_feature": fused,
            **diagnostics,
            "metadata": self.training_strategy_metadata(),
        }

    def _resolve_modality_temporal_mask(
        self,
        latent_seq: torch.Tensor,
        *,
        missing_mask: torch.Tensor | None,
        temporal_mask: torch.Tensor | None,
        modality_temporal_mask: torch.Tensor | None,
        available_modalities: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, steps, num_modalities, _ = latent_seq.shape
        if modality_temporal_mask is None:
            mask = torch.ones(batch_size, steps, num_modalities, dtype=torch.bool, device=latent_seq.device)
        else:
            mask = torch.as_tensor(modality_temporal_mask, device=latent_seq.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            if tuple(mask.shape) != (batch_size, steps, num_modalities):
                raise ValueError(
                    "modality_temporal_mask must have shape "
                    f"{(batch_size, steps, num_modalities)}, got {tuple(mask.shape)}."
                )
        modality_level = missing_mask if missing_mask is not None else available_modalities
        if modality_level is not None:
            mod_mask = torch.as_tensor(modality_level, device=latent_seq.device, dtype=torch.bool)
            if mod_mask.ndim == 1:
                mod_mask = mod_mask.unsqueeze(0).expand(batch_size, -1)
            if tuple(mod_mask.shape) != (batch_size, num_modalities):
                raise ValueError(f"missing/available modality mask must have shape {(batch_size, num_modalities)}, got {tuple(mod_mask.shape)}.")
            mask = mask & mod_mask.unsqueeze(1)
        if temporal_mask is not None:
            time_mask = torch.as_tensor(temporal_mask, device=latent_seq.device, dtype=torch.bool)
            if time_mask.ndim == 1:
                time_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
            if tuple(time_mask.shape) != (batch_size, steps):
                raise ValueError(f"temporal_mask must have shape {(batch_size, steps)}, got {tuple(time_mask.shape)}.")
            mask = mask & time_mask.unsqueeze(-1)
        empty = (~mask.any(dim=(1, 2))).nonzero(as_tuple=False).flatten()
        if int(empty.numel()) > 0:
            raise ValueError(f"modality_temporal_mask has no available cells for sample indices {empty.detach().cpu().tolist()}.")
        return mask

    def _masked_modality_temporal_mean(self, latent_seq: torch.Tensor, mt_mask: torch.Tensor) -> torch.Tensor:
        weights = mt_mask.to(device=latent_seq.device, dtype=latent_seq.dtype).unsqueeze(-1)
        return (latent_seq * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _masked_time_mean(self, values: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        weights = temporal_mask.to(device=values.device, dtype=values.dtype).view(
            values.shape[0],
            values.shape[1],
            *([1] * (values.ndim - 2)),
        )
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _s1_temporalagg_modality(
        self,
        latent_seq: torch.Tensor,
        mt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        latent = self._masked_modality_temporal_mean(latent_seq, mt_mask)
        mask = mt_mask.any(dim=1)
        reliability, _, _ = self._modality_reliability(latent, mask)
        mu_b = self._masked_time_mean(latent_seq.mean(dim=2), mt_mask.any(dim=2))
        fused, diagnostics = self._supervised_router_fuse(latent, mask, reliability, mu_b)
        logits = diagnostics["fused_logits"]
        weights = diagnostics["supervised_router_gate_weights"]
        diagnostics.update(self._temporal_common_diagnostics(mt_mask, "s1_temporalagg_modality"))
        diagnostics.update(_modality_gate_diagnostics(weights, self.modalities))
        diagnostics["temporal_router_modality_gate"] = weights.detach()
        diagnostics["temporal_router_modality_gate_logits"] = diagnostics["router_gate_logits"]
        diagnostics["temporal_router_unimodal_logits"] = diagnostics["unimodal_logits"]
        diagnostics["temporal_router_oracle_kind"] = "hard_modality"
        return logits, fused, diagnostics

    def _per_time_modality_route(self, latent_seq: torch.Tensor, mt_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, steps, num_modalities, feature_dim = latent_seq.shape
        flat_latent = latent_seq.reshape(batch_size * steps, num_modalities, feature_dim)
        flat_mask = mt_mask.reshape(batch_size * steps, num_modalities)
        reliability, _, _ = self._modality_reliability(flat_latent, flat_mask)
        mu_b = flat_latent.mean(dim=1)
        _, diagnostics = self._supervised_router_fuse(flat_latent, flat_mask, reliability, mu_b)
        return {
            "features": (flat_latent * diagnostics["supervised_router_gate_weights"].unsqueeze(-1)).sum(dim=1).view(batch_size, steps, feature_dim),
            "logits": diagnostics["fused_logits"].view(batch_size, steps, self.num_classes),
            "gate": diagnostics["supervised_router_gate_weights"].view(batch_size, steps, num_modalities),
            "gate_logits": diagnostics["router_gate_logits"].view(batch_size, steps, num_modalities),
            "unimodal_logits": diagnostics["unimodal_logits"].view(batch_size, steps, num_modalities, self.num_classes),
            "unimodal_entropy": diagnostics["supervised_router_unimodal_entropy"].view(batch_size, steps, num_modalities),
            "unimodal_margin": diagnostics["supervised_router_unimodal_margin"].view(batch_size, steps, num_modalities),
            "max_prob": diagnostics["supervised_router_max_prob"].view(batch_size, steps, num_modalities),
            "logit_norm": diagnostics["supervised_router_logit_norm"].view(batch_size, steps, num_modalities),
            "prototype_margin": diagnostics["supervised_router_prototype_margin"].view(batch_size, steps, num_modalities),
        }

    def _s2_pertime_modality(
        self,
        latent_seq: torch.Tensor,
        mt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        routed = self._per_time_modality_route(latent_seq, mt_mask)
        temporal = mt_mask.any(dim=2)
        logits = self._masked_time_mean(routed["logits"], temporal)
        fused = self._masked_time_mean(routed["features"], temporal)
        diagnostics = self._temporal_common_diagnostics(mt_mask, "s2_pertime_modality")
        diagnostics.update(
            {
                "temporal_router_modality_gate": routed["gate"].detach(),
                "temporal_router_modality_gate_logits": routed["gate_logits"],
                "temporal_router_unimodal_logits": routed["unimodal_logits"],
                "temporal_router_per_time_logits": routed["logits"],
                "temporal_aggregation_fallback": 1.0 if self.temporal_aggregation == "attention" else 0.0,
                "temporal_router_oracle_kind": "hard_per_time_modality",
            }
        )
        diagnostics["gate_entropy_modality"] = _gate_entropy(routed["gate"]).detach()
        for index, modality in enumerate(self.modalities):
            diagnostics[f"mean_gate_{modality}"] = float(routed["gate"][..., index].detach().mean().cpu().item())
        return logits, fused, diagnostics

    def _s3_two_level(
        self,
        latent_seq: torch.Tensor,
        mt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.temporal_router is None:
            raise RuntimeError("s3_two_level requested without temporal router.")
        routed = self._per_time_modality_route(latent_seq, mt_mask)
        temporal = mt_mask.any(dim=2)
        temporal_features = self._temporal_reliability_features(routed, mt_mask, temporal)
        temporal_gate_logits = self.temporal_router.forward_logits(temporal_features)
        temporal_gate = masked_pcpg_softmax(temporal_gate_logits, temporal)
        logits = (routed["logits"] * temporal_gate.unsqueeze(-1)).sum(dim=1)
        fused = (routed["features"] * temporal_gate.unsqueeze(-1)).sum(dim=1)
        diagnostics = self._temporal_common_diagnostics(mt_mask, "s3_two_level")
        diagnostics.update(
            {
                "temporal_router_modality_gate": routed["gate"].detach(),
                "temporal_router_modality_gate_logits": routed["gate_logits"],
                "temporal_router_unimodal_logits": routed["unimodal_logits"],
                "temporal_router_per_time_logits": routed["logits"],
                "temporal_gate": temporal_gate.detach(),
                "temporal_gate_logits": temporal_gate_logits,
                "temporal_router_temporal_features": temporal_features.detach(),
                "gate_entropy_modality": _gate_entropy(routed["gate"]).detach(),
                "gate_entropy_temporal": _gate_entropy(temporal_gate).detach(),
                "temporal_router_oracle_kind": "hard_temporal",
                "temporal_router_soft_target_fallback": 1.0,
            }
        )
        for index, modality in enumerate(self.modalities):
            diagnostics[f"mean_gate_{modality}"] = float(routed["gate"][..., index].detach().mean().cpu().item())
        for step in range(int(temporal_gate.shape[1])):
            diagnostics[f"mean_temporal_gate_t{step}"] = float(temporal_gate[:, step].detach().mean().cpu().item())
        return logits, fused, diagnostics

    def _s4_global(
        self,
        latent_seq: torch.Tensor,
        mt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.global_temporal_router is None:
            raise RuntimeError("s4_global requested without global router.")
        batch_size, steps, num_modalities, feature_dim = latent_seq.shape
        flat = latent_seq.reshape(batch_size * steps * num_modalities, feature_dim)
        cell_logits = self._head_logits(flat).view(batch_size, steps, num_modalities, self.num_classes)
        cell_stats = self._logit_reliability_stats(cell_logits, cell_logits)
        cell_features = self._global_cell_features(cell_stats, mt_mask)
        gate_logits = self.global_temporal_router.forward_logits(cell_features.reshape(batch_size, steps * num_modalities, -1))
        flat_mask = mt_mask.reshape(batch_size, steps * num_modalities)
        gate = masked_pcpg_softmax(gate_logits, flat_mask).view(batch_size, steps, num_modalities)
        logits = (cell_logits * gate.unsqueeze(-1)).sum(dim=(1, 2))
        fused = (latent_seq * gate.unsqueeze(-1)).sum(dim=(1, 2))
        diagnostics = self._temporal_common_diagnostics(mt_mask, "s4_global")
        diagnostics.update(
            {
                "global_gate": gate.detach(),
                "global_gate_logits": gate_logits.view(batch_size, steps, num_modalities),
                "global_unimodal_logits": cell_logits,
                "global_gate_entropy": _gate_entropy(gate.reshape(batch_size, steps * num_modalities)).detach(),
                "temporal_router_oracle_kind": "hard_global_cell",
                "temporal_router_soft_target_fallback": 1.0,
            }
        )
        for index, modality in enumerate(self.modalities):
            diagnostics[f"mean_gate_{modality}"] = float(gate[..., index].detach().mean().cpu().item())
        return logits, fused, diagnostics

    def _temporal_reliability_features(
        self,
        routed: dict[str, torch.Tensor],
        mt_mask: torch.Tensor,
        temporal: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(routed["logits"], dim=-1)
        topk = probs.topk(min(2, self.num_classes), dim=-1).values
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / max(math.log(max(self.num_classes, 2)), 1e-6)
        margin = topk[..., 0] - (topk[..., 1] if topk.shape[-1] > 1 else 0.0)
        count = mt_mask.to(dtype=routed["logits"].dtype).sum(dim=2) / max(len(self.modalities), 1)
        time_pos = torch.linspace(0.0, 1.0, steps=mt_mask.shape[1], device=mt_mask.device, dtype=routed["logits"].dtype)
        time_pos = time_pos.view(1, -1).expand(mt_mask.shape[0], -1)
        return torch.stack(
            [
                temporal.to(dtype=routed["logits"].dtype),
                count,
                entropy,
                probs.amax(dim=-1),
                margin,
                routed["logits"].norm(dim=-1),
                routed["prototype_margin"].amax(dim=-1),
                time_pos,
            ],
            dim=-1,
        )

    def _global_cell_features(self, stats: dict[str, torch.Tensor], mt_mask: torch.Tensor) -> torch.Tensor:
        batch_size, steps, num_modalities = mt_mask.shape
        dtype = stats["entropy"].dtype
        device = mt_mask.device
        time_pos = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=dtype).view(1, steps, 1).expand(batch_size, -1, num_modalities)
        modality_pos = torch.linspace(0.0, 1.0, steps=num_modalities, device=device, dtype=dtype).view(1, 1, num_modalities).expand(batch_size, steps, -1)
        return torch.stack(
            [
                mt_mask.to(dtype=dtype),
                stats["entropy"],
                stats["margin"],
                stats["confidence"],
                stats["logit_norm"],
                stats["prototype_margin"],
                time_pos,
                modality_pos,
            ],
            dim=-1,
        )

    def _temporal_common_diagnostics(self, mt_mask: torch.Tensor, router_type: str) -> dict[str, Any]:
        return {
            "temporal_router_type": router_type,
            "temporal_aggregation": self.temporal_aggregation,
            "modality_temporal_mask": mt_mask.detach(),
            "temporal_mask": mt_mask.any(dim=2).detach(),
            "modality_mask": mt_mask.any(dim=1).detach(),
            "temporal_router_distill_weight": float(self.temporal_router_distill_weight),
            "router_supervision": self.router_supervision,
            "router_distill_weight": float(self.router_distill_weight),
            "prototype_margin_enabled": self._prototype_margin_enabled(),
            "prototype_margin_fallback": 0.0 if self._prototype_margin_enabled() else 1.0,
        }

    def _load_encoder_checkpoints(self, paths: dict[str, str]) -> dict[str, Any]:
        if not paths:
            return {}
        loads: dict[str, Any] = {}
        for modality, raw_path in paths.items():
            if modality not in self.encoders:
                raise ValueError(f"encoder checkpoint configured for disabled modality '{modality}'.")
            checkpoint_path = Path(str(raw_path))
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
            return self._supervised_router_fuse(latent, mask, reliability, mu_b)
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
        reliability_features = self._router_reliability_features(feature_stats, available, reliability)
        pattern_features = _pattern_features(available) if self.router_use_pattern_features else None
        gate_logits = self.bprr_router.forward_logits(reliability_features, mask, pattern_features)
        weights = masked_pcpg_softmax(gate_logits, mask)
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
    ) -> torch.Tensor:
        zero = torch.zeros_like(feature_stats["margin"])
        use_rel = self.router_use_reliability_features
        use_proto_margin = use_rel and self.router_use_prototype_margin and self._prototype_margin_enabled()
        return torch.stack(
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

    def _router_feature_names(self) -> tuple[str, ...]:
        return (
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


def circular_beam_error_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    target = labels.to(device=logits.device, dtype=torch.long)
    if target.ndim > 1:
        target = target[:, 0]
    pred = logits.argmax(dim=-1)
    while target.ndim < pred.ndim:
        target = target.unsqueeze(-1)
    diff = (pred - target).abs()
    return torch.minimum(diff, int(num_classes) - diff).to(dtype=logits.dtype)


def modality_oracle_targets(unimodal_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    errors = circular_beam_error_from_logits(unimodal_logits, labels, int(unimodal_logits.shape[-1]))
    return _masked_argmin(errors, mask)


def per_time_modality_oracle_targets(
    unimodal_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, steps, _, _ = unimodal_logits.shape
    flat = modality_oracle_targets(
        unimodal_logits.reshape(batch_size * steps, unimodal_logits.shape[2], unimodal_logits.shape[3]),
        labels.reshape(batch_size, -1)[:, :1].expand(-1, steps).reshape(-1),
        mask.reshape(batch_size * steps, mask.shape[2]),
    )
    return flat.view(batch_size, steps)


def temporal_oracle_targets(per_time_logits: torch.Tensor, labels: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
    errors = circular_beam_error_from_logits(per_time_logits, labels, int(per_time_logits.shape[-1]))
    return _masked_argmin(errors, temporal_mask)


def global_oracle_targets(cell_logits: torch.Tensor, labels: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
    batch_size, steps, modalities, classes = cell_logits.shape
    flat_logits = cell_logits.reshape(batch_size, steps * modalities, classes)
    flat_mask = cell_mask.reshape(batch_size, steps * modalities)
    errors = circular_beam_error_from_logits(flat_logits, labels, int(classes))
    return _masked_argmin(errors, flat_mask)


def _masked_argmin(errors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    available = mask.to(device=errors.device, dtype=torch.bool)
    masked_errors = errors.masked_fill(~available, torch.finfo(errors.dtype).max)
    target = masked_errors.argmin(dim=-1)
    return torch.where(available.any(dim=-1), target, torch.full_like(target, -100))


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "none", ""}
    return bool(value)


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
