from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.registries import ENCODERS, MODELS


DEFAULT_MODALITIES = MODALITY_ORDER


def _masked_softmax(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    available = available.to(device=logits.device, dtype=torch.bool)
    if logits.shape != available.shape:
        raise ValueError(f"router logits {tuple(logits.shape)} must match mask {tuple(available.shape)}.")
    if not bool(available.any(dim=1).all().item()):
        raise ValueError("u_mask_beam_jepa requires one available modality per sample.")
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=1) * available.to(dtype=logits.dtype)


@MODELS.register("u_mask_beam_jepa")
class UMaskBeamJEPA(nn.Module):
    """T2/S1 temporal masked-mean model with a supervised modality router."""

    supports_modality_kwargs = True
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int = 64,
        num_classes: int = 64,
        num_pred: int = 1,
        dropout: float = 0.1,
        encoders: dict[str, dict[str, Any]] | None = None,
        temporal_pooling: dict[str, Any] | bool | None = None,
        fusion_type: str = "supervised_router",
        head_type: str = "prototype",
        beam_proto_temperature: float = 0.2,
        router_use_pattern_features: bool = True,
        router_use_reliability_features: bool = True,
        router_use_prototype_margin: bool = True,
        router_use_entropy: bool = True,
        router_use_confidence: bool = True,
        router_use_logit_norm: bool = True,
        router_hidden_dim: int = 64,
        image_channels: int = 3,
        radar_channels: int = 2,
        lidar_channels: int = 3,
        gps_input_size: int = 3,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modalities = _validate_modalities(modalities or DEFAULT_MODALITIES)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.fusion_type = str(fusion_type).strip().lower()
        self.head_type = str(head_type).strip().lower()
        self.router_use_pattern_features = bool(router_use_pattern_features)
        self.router_use_reliability_features = bool(router_use_reliability_features)
        self.router_use_prototype_margin = bool(router_use_prototype_margin)
        self.router_use_entropy = bool(router_use_entropy)
        self.router_use_confidence = bool(router_use_confidence)
        self.router_use_logit_norm = bool(router_use_logit_norm)
        self.temporal_pooling = _temporal_pooling_config(temporal_pooling)

        if self.d_model <= 0 or self.num_classes <= 0 or self.num_pred <= 0:
            raise ValueError("d_model, num_classes, and num_pred must be positive.")
        if self.fusion_type != "supervised_router":
            raise ValueError("T2 only supports fusion_type='supervised_router'.")
        if self.head_type not in {"prototype", "classifier"}:
            raise ValueError("T2 head_type must be 'prototype' or 'classifier'.")

        encoder_configs = {name: dict((encoders or {}).get(name, {})) for name in self.modalities}
        missing = [name for name, config in encoder_configs.items() if not config]
        if missing:
            raise ValueError(f"u_mask_beam_jepa requires encoders for {missing}.")
        defaults = {
            "image": {"image_channels": image_channels},
            "radar": {"radar_channels": radar_channels},
            "lidar": {"lidar_channels": lidar_channels},
            "gps": {"gps_input_size": gps_input_size},
        }
        self.encoder_configs: dict[str, dict[str, Any]] = {}
        self.encoders = nn.ModuleDict()
        self.encoder_projections = nn.ModuleDict()
        for name in self.modalities:
            config = {**defaults[name], **encoder_configs[name]}
            config.setdefault("output_dim", self.d_model)
            encoder = ENCODERS.build(config)
            output_dim = int(getattr(encoder, "output_dim", config["output_dim"]))
            self.encoders[name] = encoder
            self.encoder_projections[name] = nn.Identity() if output_dim == self.d_model else nn.Linear(output_dim, self.d_model)
            self.encoder_configs[name] = config

        self.reliability_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))
                for name in self.modalities
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.num_classes),
        )
        self.prototype_bank = BeamPrototypeBank(self.d_model, self.num_classes, temperature=float(beam_proto_temperature))

        self.router_feature_names = _router_feature_names(
            reliability=self.router_use_reliability_features,
            prototype_margin=self.router_use_prototype_margin and self.head_type == "prototype",
            entropy=self.router_use_entropy,
            confidence=self.router_use_confidence,
            logit_norm=self.router_use_logit_norm,
        )
        router_input_dim = len(self.router_feature_names) + len(self.modalities)
        self.supervised_router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, int(router_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), 1),
        )
        self.router_pattern_bias = (
            nn.Linear(len(self.modalities), len(self.modalities), bias=False)
            if self.router_use_pattern_features
            else None
        )

    def forward(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        modality_temporal_mask: torch.Tensor | None = None,
        available_modalities: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
        }
        sequences = [self._encode_sequence(name, inputs[name]) for name in self.modalities]
        latent_sequence = torch.stack(sequences, dim=2)
        available = self._resolve_modality_mask(
            missing_mask if missing_mask is not None else force_modality_mask,
            available_modalities,
            latent_sequence,
        )
        cell_mask = self._resolve_temporal_mask(latent_sequence, available, temporal_mask, modality_temporal_mask)
        latent = _masked_mean(latent_sequence, cell_mask)
        reliability = torch.stack(
            [torch.sigmoid(self.reliability_heads[name](latent[:, index])) for index, name in enumerate(self.modalities)],
            dim=1,
        )
        reliability = reliability * available.unsqueeze(-1).to(dtype=reliability.dtype)
        unimodal_logits = self._head_logits(latent.reshape(-1, self.d_model)).reshape(
            latent.shape[0], len(self.modalities), self.num_classes
        )
        router_features = self._router_features(unimodal_logits, reliability, available)
        eye = torch.eye(len(self.modalities), device=latent.device, dtype=latent.dtype).unsqueeze(0).expand(latent.shape[0], -1, -1)
        router_logits = self.supervised_router(torch.cat([router_features, eye], dim=-1)).squeeze(-1)
        if self.router_pattern_bias is not None:
            router_logits = router_logits + self.router_pattern_bias(available.to(dtype=latent.dtype))
        router_weights = _masked_softmax(router_logits, available)
        fused_features = (router_weights.unsqueeze(-1) * latent).sum(dim=1)
        fused_logits = (router_weights.unsqueeze(-1) * unimodal_logits).sum(dim=1)
        return {
            "logits": fused_logits.unsqueeze(1).expand(-1, self.num_pred, -1),
            "input_features": latent,
            "output_features": fused_features,
            "modality_features": latent,
            "missing_mask": available,
            "available_modalities": available,
            "modality_temporal_mask": cell_mask,
            "temporal_mask": cell_mask.any(dim=2),
            "temporal_pooling_type": "masked_mean",
            "temporal_pooling_param_count": 0,
            "modality_reliability": reliability,
            "global_reliability": reliability.squeeze(-1).sum(dim=1) / available.sum(dim=1).clamp_min(1),
            "unimodal_logits": unimodal_logits,
            "router_gate_logits": router_logits,
            "router_gate_weights": router_weights,
            "supervised_router_gate_logits": router_logits,
            "supervised_router_gate_weights": router_weights,
            "supervised_router_reliability_features": router_features,
            "supervised_router_feature_names": self.router_feature_names,
            "reliability_fusion_mode": "supervised_router",
            "reliability_fusion_weights": router_weights,
            "metadata": self.training_strategy_metadata(),
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "type": "u_mask_beam_jepa",
            "architecture_category": "t2_temporal_supervised_router",
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "consumes_missing_mask": True,
            "consumes_missing_modality_metadata": True,
            "fusion_type": self.fusion_type,
            "head_type": self.head_type,
            "router_feature_names": list(self.router_feature_names),
            "temporal_pooling": dict(self.temporal_pooling),
            "temporal_pooling_type": "masked_mean",
            "temporal_pooling_param_count": 0,
            "encoder_configs": self.encoder_configs,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }

    def _encode_sequence(self, modality: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"u_mask_beam_jepa requires {modality}_batch.")
        features = self.encoders[modality](value)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError(f"{modality} encoder must return [B,T,D] or [B,D], got {tuple(features.shape)}.")
        return self.encoder_projections[modality](features)

    def _resolve_modality_mask(
        self,
        missing_mask: torch.Tensor | None,
        available_modalities: torch.Tensor | None,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sequence.shape[0]
        count = len(self.modalities)
        raw = missing_mask if missing_mask is not None else available_modalities
        if raw is None:
            return torch.ones(batch_size, count, dtype=torch.bool, device=sequence.device)
        mask = torch.as_tensor(raw, device=sequence.device, dtype=torch.bool)
        if tuple(mask.shape) == (count,):
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, count):
            raise ValueError(f"missing_mask must have shape {(batch_size, count)}, got {tuple(mask.shape)}.")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("u_mask_beam_jepa requires at least one available modality per sample.")
        return mask

    def _resolve_temporal_mask(
        self,
        sequence: torch.Tensor,
        available: torch.Tensor,
        temporal_mask: torch.Tensor | None,
        modality_temporal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, steps, modalities, _ = sequence.shape
        if modality_temporal_mask is not None:
            mask = torch.as_tensor(modality_temporal_mask, device=sequence.device, dtype=torch.bool)
            if tuple(mask.shape) != (batch_size, steps, modalities):
                raise ValueError(
                    "modality_temporal_mask must have shape "
                    f"{(batch_size, steps, modalities)}, got {tuple(mask.shape)}."
                )
        elif temporal_mask is not None:
            time = torch.as_tensor(temporal_mask, device=sequence.device, dtype=torch.bool)
            if tuple(time.shape) != (batch_size, steps):
                raise ValueError(f"temporal_mask must have shape {(batch_size, steps)}, got {tuple(time.shape)}.")
            mask = time.unsqueeze(-1).expand(-1, -1, modalities)
        else:
            mask = torch.ones(batch_size, steps, modalities, dtype=torch.bool, device=sequence.device)
        mask = mask & available.unsqueeze(1)
        if not bool(mask.any(dim=(1, 2)).all().item()):
            raise ValueError("u_mask_beam_jepa requires at least one available temporal cell per sample.")
        return mask

    def _head_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.prototype_bank(features) if self.head_type == "prototype" else self.classifier(features)

    def _router_features(
        self,
        unimodal_logits: torch.Tensor,
        reliability: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        probabilities = torch.softmax(unimodal_logits, dim=-1)
        if self.router_use_reliability_features:
            features.append(reliability)
        if self.router_use_prototype_margin and self.head_type == "prototype":
            top2 = unimodal_logits.topk(min(2, self.num_classes), dim=-1).values
            margin = top2[..., :1] if self.num_classes == 1 else (top2[..., :1] - top2[..., 1:2])
            features.append(margin)
        if self.router_use_entropy:
            entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=-1, keepdim=True)
            features.append(entropy)
        if self.router_use_confidence:
            features.append(probabilities.amax(dim=-1, keepdim=True))
        if self.router_use_logit_norm:
            features.append(unimodal_logits.norm(dim=-1, keepdim=True))
        if not features:
            features.append(available.unsqueeze(-1).to(dtype=unimodal_logits.dtype))
        return torch.cat(features, dim=-1)


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=sequence.device, dtype=sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _temporal_pooling_config(raw: dict[str, Any] | bool | None) -> dict[str, Any]:
    config = dict(raw) if isinstance(raw, dict) else {"enabled": bool(raw)}
    if not bool(config.get("enabled", False)) or str(config.get("type", "masked_mean")).strip().lower() != "masked_mean":
        raise ValueError("T2 requires temporal_pooling.enabled=true and temporal_pooling.type='masked_mean'.")
    return {"enabled": True, "type": "masked_mean"}


def _router_feature_names(
    *,
    reliability: bool,
    prototype_margin: bool,
    entropy: bool,
    confidence: bool,
    logit_norm: bool,
) -> tuple[str, ...]:
    names = []
    if reliability:
        names.append("reliability")
    if prototype_margin:
        names.append("prototype_margin")
    if entropy:
        names.append("entropy")
    if confidence:
        names.append("confidence")
    if logit_norm:
        names.append("logit_norm")
    return tuple(names or ["availability"])


def _validate_modalities(modalities: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(str(item) for item in modalities)
    unknown = sorted(set(values) - set(DEFAULT_MODALITIES))
    if not values or unknown or len(set(values)) != len(values):
        raise ValueError(
            f"u_mask_beam_jepa modalities must be a non-empty unique subset of {list(DEFAULT_MODALITIES)}; got {list(values)}."
        )
    return values
