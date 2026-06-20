from typing import Any

import torch
import torch.nn as nn

from kd_sensing.models.fusion.cls_token_transformer import CLSTokenTransformerFusionNet
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image_encoders import CameraAEImageEncoder, ResNet18ImageEncoder  # noqa: F401
from kd_sensing.registries import ENCODERS, MODELS


VISION_POSITION_PRESETS = (
    "camera_ae_gps",
    "resnet_gps",
    "transformer_image_gps",
    "gps_only_neural",
)


def _positive_int(value: int | None, *, name: str, default: int) -> int:
    resolved = default if value is None else int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return resolved


def _as_config(raw: Any, *, default_type: str) -> dict[str, Any]:
    if raw is None:
        return {"type": default_type}
    if isinstance(raw, str):
        return {"type": raw}
    if not isinstance(raw, dict):
        raise ValueError("Encoder config must be a dict or string.")
    cfg = dict(raw)
    cfg.setdefault("type", default_type)
    return cfg


def _build_image_encoder(
    raw_cfg: Any,
    *,
    default_type: str,
    output_dim: int,
    image_profile: str | None,
    image_channels: int,
) -> nn.Module:
    cfg = _as_config(raw_cfg, default_type=default_type)
    cfg.setdefault("output_dim", output_dim)
    cfg.setdefault("image_profile", image_profile)
    cfg.setdefault("image_channels", image_channels)
    return ENCODERS.build(cfg)


def _build_gps_encoder(
    raw_cfg: Any,
    *,
    output_dim: int,
    gps_input_size: int,
    dropout: float,
) -> nn.Module:
    cfg = _as_config(raw_cfg, default_type="gps_mlp")
    encoder_type = str(cfg.get("type", "gps_mlp"))
    if encoder_type not in {"gps_mlp", "gps_feature_extractor", "direct_mlp", "gps_direct_mlp"}:
        raise ValueError(
            "Vision-position GPS encoder must be one of gps_mlp, gps_feature_extractor, "
            f"direct_mlp, or gps_direct_mlp; got {encoder_type!r}."
        )
    hidden_size = int(cfg.get("hidden_size", max(output_dim, 64)))
    return GpsFeatureExtractor(
        int(cfg.get("output_dim", cfg.get("feature_size", output_dim))),
        gps_input_size=int(cfg.get("gps_input_size", gps_input_size)),
        hidden_size=hidden_size,
        dropout=float(cfg.get("dropout", dropout)),
    )


def _encoder_output_dim(encoder: nn.Module, fallback: int) -> int:
    return int(getattr(encoder, "output_dim", fallback))


def _history_features(features: torch.Tensor, *, history_length: int | None, num_pred: int, name: str) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"{name} features must have shape [B, T, D], got {tuple(features.shape)}.")
    total_steps = int(features.shape[1])
    if total_steps <= 0:
        raise ValueError(f"{name} sequence must contain at least one step.")
    if history_length is not None:
        steps = int(history_length)
        if steps <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}.")
        if total_steps < steps:
            raise ValueError(
                f"{name} sequence has {total_steps} steps but history_length={steps} was configured."
            )
        return features[:, :steps, :]
    inferred = total_steps - max(int(num_pred) - 1, 0)
    return features[:, : max(inferred, 1), :]


def _check_shared_time(left: torch.Tensor, right: torch.Tensor, *, left_name: str, right_name: str) -> None:
    if left.ndim < 2 or right.ndim < 2:
        raise ValueError(f"{left_name} and {right_name} inputs must expose batch and sequence dimensions.")
    if tuple(left.shape[:2]) != tuple(right.shape[:2]):
        raise ValueError(
            f"{left_name} and {right_name} sequence dimensions must match; "
            f"got {left_name} {tuple(left.shape)} and {right_name} {tuple(right.shape)}."
        )


def _validate_num_classes(num_classes: int) -> int:
    classes = int(num_classes)
    if classes != 64:
        raise ValueError(f"Vision-position baselines use the 64-beam label space; got num_classes={classes}.")
    return classes


class _HorizonClassifierMixin:
    num_pred: int
    num_classes: int

    def _horizon_features(self, context: torch.Tensor, embedding: nn.Parameter) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"context features must have shape [B, D], got {tuple(context.shape)}.")
        return context.unsqueeze(1) + embedding[:, : self.num_pred, :]


@MODELS.register("vision_position_late_fusion")
class VisionPositionLateFusionNet(nn.Module, _HorizonClassifierMixin):
    supports_modality_kwargs = True

    def __init__(
        self,
        *,
        image_encoder: dict[str, Any] | str | None = None,
        image_encoder_type: str = "resnet18_imagenet_rgb",
        gps_encoder: dict[str, Any] | str | None = None,
        modalities: list[str] | tuple[str, ...] | None = None,
        feature_size: int = 64,
        fusion_hidden_size: int | None = None,
        temporal_hidden_size: int | None = None,
        temporal_layers: int = 1,
        temporal_aggregation: str = "mean",
        num_classes: int = 64,
        num_pred: int = 3,
        history_length: int | None = None,
        seq_length: int | None = None,
        image_profile: str | None = "rgb_imagenet",
        image_channels: int = 3,
        gps_input_size: int = 3,
        dropout: float = 0.1,
        baseline_preset: str | None = None,
        model_name: str | None = None,
        gps_feature_mode: str = "relative_polar",
        paper_model_group: str | None = None,
        target_source: str | None = None,
        metric_profile: str | None = None,
        claim_status: str | None = None,
        paper_reported_row: bool | None = None,
        fusion_type: str = "late_concat_mlp",
        gps_normalizer_provenance: str | None = None,
        uses_lidar: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        selected = tuple(modalities or ("image", "gps"))
        if selected != ("image", "gps"):
            raise ValueError("vision_position_late_fusion requires modalities ['image', 'gps'].")
        self.name = "VisionPositionLateFusionNet"
        self.modalities = selected
        self.feature_size = _positive_int(feature_size, name="feature_size", default=64)
        self.num_classes = _validate_num_classes(num_classes)
        self.num_pred = _positive_int(num_pred, name="num_pred", default=3)
        self.history_length = int(history_length or seq_length) if (history_length or seq_length) else None
        self.temporal_aggregation = str(temporal_aggregation or "mean").lower()
        self.baseline_preset = baseline_preset
        self.model_name = model_name
        self.gps_feature_mode = str(gps_feature_mode)
        self.image_profile = image_profile
        self.image_encoder_type = str(image_encoder_type)
        self.paper_model_group = paper_model_group
        self.target_source = target_source
        self.metric_profile = metric_profile
        self.claim_status = claim_status
        self.paper_reported_row = paper_reported_row
        self.fusion_type = str(fusion_type)
        self.gps_normalizer_provenance = gps_normalizer_provenance
        self.uses_lidar = bool(uses_lidar)

        self.image_encoder = _build_image_encoder(
            image_encoder,
            default_type=self.image_encoder_type,
            output_dim=self.feature_size,
            image_profile=image_profile,
            image_channels=int(image_channels),
        )
        self.gps_encoder = _build_gps_encoder(
            gps_encoder,
            output_dim=self.feature_size,
            gps_input_size=int(gps_input_size),
            dropout=float(dropout),
        )
        image_dim = _encoder_output_dim(self.image_encoder, self.feature_size)
        gps_dim = _encoder_output_dim(self.gps_encoder, self.feature_size)
        fusion_dim = _positive_int(fusion_hidden_size, name="fusion_hidden_size", default=self.feature_size)
        self.fusion = nn.Sequential(
            nn.LayerNorm(image_dim + gps_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(image_dim + gps_dim, fusion_dim),
            nn.GELU(),
        )
        temporal_dim = _positive_int(temporal_hidden_size, name="temporal_hidden_size", default=fusion_dim)
        if self.temporal_aggregation == "mean":
            self.temporal = None
            context_dim = fusion_dim
        elif self.temporal_aggregation == "last":
            self.temporal = None
            context_dim = fusion_dim
        elif self.temporal_aggregation == "gru":
            self.temporal = nn.GRU(
                input_size=fusion_dim,
                hidden_size=temporal_dim,
                num_layers=int(temporal_layers),
                dropout=float(dropout) if int(temporal_layers) > 1 else 0.0,
                batch_first=True,
            )
            context_dim = temporal_dim
        elif self.temporal_aggregation == "lstm":
            self.temporal = nn.LSTM(
                input_size=fusion_dim,
                hidden_size=temporal_dim,
                num_layers=int(temporal_layers),
                dropout=float(dropout) if int(temporal_layers) > 1 else 0.0,
                batch_first=True,
            )
            context_dim = temporal_dim
        else:
            raise ValueError("temporal_aggregation must be one of mean, last, gru, or lstm.")
        self.context_norm = nn.LayerNorm(context_dim)
        self.horizon_embedding = nn.Parameter(torch.zeros(1, self.num_pred, context_dim))
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(context_dim, self.num_classes),
        )

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if image_batch is None:
            raise ValueError("vision_position_late_fusion requires image_batch.")
        if gps_batch is None:
            raise ValueError("vision_position_late_fusion requires gps_batch.")
        _check_shared_time(image_batch, gps_batch, left_name="image", right_name="GPS")
        image_features = _history_features(
            self.image_encoder(image_batch),
            history_length=self.history_length,
            num_pred=self.num_pred,
            name="image",
        )
        gps_features = _history_features(
            self.gps_encoder(gps_batch),
            history_length=self.history_length,
            num_pred=self.num_pred,
            name="GPS",
        )
        _check_shared_time(image_features, gps_features, left_name="image features", right_name="GPS features")
        fused = self.fusion(torch.cat([image_features, gps_features], dim=-1))
        if self.temporal_aggregation == "mean":
            context = fused.mean(dim=1)
            output_features = fused
        elif self.temporal_aggregation == "last":
            context = fused[:, -1, :]
            output_features = fused
        else:
            assert self.temporal is not None
            output_features, state = self.temporal(fused)
            if isinstance(state, tuple):
                context = state[0][-1]
            else:
                context = state[-1]
        context = self.context_norm(context)
        horizon_features = self._horizon_features(context, self.horizon_embedding)
        logits = self.classifier(horizon_features)
        return {
            "logits": logits,
            "input_features": fused,
            "output_features": horizon_features,
            "image_features": image_features,
            "gps_features": gps_features,
            "modalities": self.modalities,
            "baseline_preset": self.baseline_preset,
            "model_name": self.model_name,
            "encoder_type": self.encoder_type,
            "gps_feature_mode": self.gps_feature_mode,
            "temporal_aggregation": self.temporal_aggregation,
            "label_space": "64_beam",
            "paper_model_group": self.paper_model_group,
            "claim_status": self.claim_status,
        }

    @property
    def encoder_type(self) -> str:
        return str(getattr(self.image_encoder, "expected_image_profile", None) and self.image_encoder.__class__.__name__)

    def training_strategy_metadata(self) -> dict[str, Any]:
        image_metadata = (
            self.image_encoder.training_strategy_metadata()
            if hasattr(self.image_encoder, "training_strategy_metadata")
            else {"type": self.image_encoder.__class__.__name__}
        )
        return {
            "type": "vision_position_late_fusion",
            "model_registry_name": "vision_position_late_fusion",
            "architecture_category": "whole_model_exception",
            "baseline_preset": self.baseline_preset,
            "model_name": self.model_name,
            "paper_model_group": self.paper_model_group,
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "encoder_type": image_metadata.get("encoder") or image_metadata.get("type") or self.image_encoder.__class__.__name__,
            "image_encoder": image_metadata,
            "gps_encoder_type": self.gps_encoder.__class__.__name__,
            "gps_normalizer_provenance": self.gps_normalizer_provenance,
            "uses_external_checkpoint": bool(image_metadata.get("checkpoint_path") or image_metadata.get("pretrained", False)),
            "freeze_policy": {
                "image_encoder": bool(
                    image_metadata.get("freeze_encoder", image_metadata.get("freeze_backbone", False))
                ),
                "gps_encoder": False,
            },
            "consumes_reliability_metadata": False,
            "gps_feature_mode": self.gps_feature_mode,
            "fusion_type": self.fusion_type,
            "target_source": self.target_source,
            "metric_profile": self.metric_profile,
            "claim_status": self.claim_status,
            "paper_reported_row": self.paper_reported_row,
            "uses_lidar": self.uses_lidar,
            "temporal_aggregation": self.temporal_aggregation,
            "num_classes": self.num_classes,
            "num_pred": self.num_pred,
            "label_space": "64_beam",
        }


@MODELS.register("gps_sequence_baseline")
class GpsSequenceBaselineNet(nn.Module, _HorizonClassifierMixin):
    supports_modality_kwargs = True

    def __init__(
        self,
        *,
        gps_input_size: int = 3,
        feature_size: int = 64,
        hidden_size: int = 64,
        temporal_model: str = "gru",
        num_layers: int = 1,
        num_classes: int = 64,
        num_pred: int = 3,
        history_length: int | None = None,
        seq_length: int | None = None,
        dropout: float = 0.1,
        baseline_preset: str | None = "gps_only_neural",
        model_name: str | None = None,
        gps_feature_mode: str = "relative_polar",
        paper_model_group: str | None = None,
        target_source: str | None = None,
        metric_profile: str | None = None,
        claim_status: str | None = None,
        paper_reported_row: bool | None = None,
        gps_normalizer_provenance: str | None = None,
        uses_lidar: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.name = "GpsSequenceBaselineNet"
        self.modalities = ("gps",)
        self.feature_size = _positive_int(feature_size, name="feature_size", default=64)
        self.hidden_size = _positive_int(hidden_size, name="hidden_size", default=self.feature_size)
        self.num_classes = _validate_num_classes(num_classes)
        self.num_pred = _positive_int(num_pred, name="num_pred", default=3)
        self.history_length = int(history_length or seq_length) if (history_length or seq_length) else None
        self.temporal_model = str(temporal_model or "gru").lower()
        self.baseline_preset = baseline_preset
        self.model_name = model_name
        self.gps_feature_mode = str(gps_feature_mode)
        self.paper_model_group = paper_model_group
        self.target_source = target_source
        self.metric_profile = metric_profile
        self.claim_status = claim_status
        self.paper_reported_row = paper_reported_row
        self.gps_normalizer_provenance = gps_normalizer_provenance
        self.uses_lidar = bool(uses_lidar)
        self.gps_encoder = GpsFeatureExtractor(
            self.feature_size,
            gps_input_size=int(gps_input_size),
            hidden_size=max(self.feature_size, self.hidden_size),
            dropout=float(dropout),
        )
        if self.temporal_model == "mlp":
            self.temporal = None
            context_dim = self.feature_size
        elif self.temporal_model == "gru":
            self.temporal = nn.GRU(
                input_size=self.feature_size,
                hidden_size=self.hidden_size,
                num_layers=int(num_layers),
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
                batch_first=True,
            )
            context_dim = self.hidden_size
        elif self.temporal_model == "lstm":
            self.temporal = nn.LSTM(
                input_size=self.feature_size,
                hidden_size=self.hidden_size,
                num_layers=int(num_layers),
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
                batch_first=True,
            )
            context_dim = self.hidden_size
        else:
            raise ValueError("temporal_model must be one of mlp, gru, or lstm.")
        self.context_norm = nn.LayerNorm(context_dim)
        self.horizon_embedding = nn.Parameter(torch.zeros(1, self.num_pred, context_dim))
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(context_dim, self.num_classes),
        )

    def forward(self, gps_batch: torch.Tensor | None = None, **_: Any) -> dict[str, Any]:
        if gps_batch is None:
            raise ValueError("gps_sequence_baseline requires gps_batch.")
        features = _history_features(
            self.gps_encoder(gps_batch),
            history_length=self.history_length,
            num_pred=self.num_pred,
            name="GPS",
        )
        if self.temporal_model == "mlp":
            context = features.mean(dim=1)
            sequence_features = features
        else:
            assert self.temporal is not None
            sequence_features, state = self.temporal(features)
            if isinstance(state, tuple):
                context = state[0][-1]
            else:
                context = state[-1]
        context = self.context_norm(context)
        horizon_features = self._horizon_features(context, self.horizon_embedding)
        logits = self.classifier(horizon_features)
        return {
            "logits": logits,
            "input_features": features,
            "output_features": horizon_features,
            "sequence_features": sequence_features,
            "modalities": self.modalities,
            "baseline_preset": self.baseline_preset,
            "model_name": self.model_name,
            "uses_neural_network": True,
            "gps_feature_mode": self.gps_feature_mode,
            "temporal_aggregation": self.temporal_model,
            "label_space": "64_beam",
            "paper_model_group": self.paper_model_group,
            "claim_status": self.claim_status,
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "gps_sequence_baseline",
            "model_registry_name": "gps_sequence_baseline",
            "architecture_category": "whole_model_exception",
            "baseline_preset": self.baseline_preset,
            "model_name": self.model_name,
            "paper_model_group": self.paper_model_group,
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "uses_neural_network": True,
            "uses_external_checkpoint": False,
            "freeze_policy": "none",
            "consumes_reliability_metadata": False,
            "non_neural_window_baseline": False,
            "gps_feature_mode": self.gps_feature_mode,
            "gps_normalizer_provenance": self.gps_normalizer_provenance,
            "target_source": self.target_source,
            "metric_profile": self.metric_profile,
            "claim_status": self.claim_status,
            "paper_reported_row": self.paper_reported_row,
            "uses_lidar": self.uses_lidar,
            "temporal_aggregation": self.temporal_model,
            "num_classes": self.num_classes,
            "num_pred": self.num_pred,
            "label_space": "64_beam",
        }


MODELS.register_removed(
    "gps_only_neural_baseline",
    "Use 'gps_sequence_baseline' for the retained AMR-Net GPS-only baseline.",
)


@MODELS.register("vision_position_transformer_fusion")
class VisionPositionTransformerFusionNet(CLSTokenTransformerFusionNet):
    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        history_length: int | None = None,
        seq_length: int | None = None,
        token_organization: str = "cls_time_major_image_gps_tokens",
        baseline_preset: str | None = "transformer_image_gps",
        model_name: str | None = None,
        gps_feature_mode: str = "relative_polar",
        paper_model_group: str | None = None,
        target_source: str | None = None,
        metric_profile: str | None = None,
        claim_status: str | None = None,
        paper_reported_row: bool | None = None,
        uses_lidar: bool = False,
        **kwargs: Any,
    ) -> None:
        selected = tuple(modalities or ("image", "gps"))
        if selected != ("image", "gps"):
            raise ValueError("vision_position_transformer_fusion requires modalities ['image', 'gps'].")
        self.history_length = int(history_length or seq_length) if (history_length or seq_length) else None
        self.token_organization = str(token_organization)
        self.baseline_preset = baseline_preset
        self.model_name = model_name
        self.gps_feature_mode = str(gps_feature_mode)
        self.paper_model_group = paper_model_group
        self.target_source = target_source
        self.metric_profile = metric_profile
        self.claim_status = claim_status
        self.paper_reported_row = paper_reported_row
        self.uses_lidar = bool(uses_lidar)
        super().__init__(modalities=list(selected), **kwargs)

    def forward(self, image_batch: torch.Tensor | None = None, gps_batch: torch.Tensor | None = None, **kwargs: Any):
        if image_batch is None:
            raise ValueError("vision_position_transformer_fusion requires image_batch.")
        if gps_batch is None:
            raise ValueError("vision_position_transformer_fusion requires gps_batch.")
        _check_shared_time(image_batch, gps_batch, left_name="image", right_name="GPS")
        if self.history_length is not None:
            if image_batch.shape[1] < self.history_length or gps_batch.shape[1] < self.history_length:
                raise ValueError(
                    "vision_position_transformer_fusion configured history_length="
                    f"{self.history_length} but got image {tuple(image_batch.shape)} and GPS {tuple(gps_batch.shape)}."
                )
            image_batch = image_batch[:, : self.history_length, ...]
            gps_batch = gps_batch[:, : self.history_length, :]
        output = super().forward(image_batch=image_batch, gps_batch=gps_batch, **kwargs)
        output["baseline_preset"] = self.baseline_preset
        output["model_name"] = self.model_name
        output["token_organization"] = self.token_organization
        output["gps_feature_mode"] = self.gps_feature_mode
        output["label_space"] = "64_beam"
        output["paper_model_group"] = self.paper_model_group
        output["claim_status"] = self.claim_status
        return output

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "vision_position_transformer_fusion",
            "model_registry_name": "vision_position_transformer_fusion",
            "architecture_category": "whole_model_exception",
            "baseline_preset": self.baseline_preset,
            "model_name": self.model_name,
            "paper_model_group": self.paper_model_group,
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "uses_external_checkpoint": False,
            "freeze_policy": "none",
            "consumes_reliability_metadata": False,
            "token_organization": self.token_organization,
            "d_model": self.d_model,
            "num_heads": self.transformer.layers[0].self_attn.num_heads if self.transformer.layers else None,
            "num_layers": len(self.transformer.layers),
            "max_seq_len": self.max_seq_len,
            "gps_feature_mode": self.gps_feature_mode,
            "target_source": self.target_source,
            "metric_profile": self.metric_profile,
            "claim_status": self.claim_status,
            "paper_reported_row": self.paper_reported_row,
            "uses_lidar": self.uses_lidar,
            "num_classes": self.num_classes,
            "num_pred": self.num_pred,
            "label_space": "64_beam",
        }


__all__ = [
    "GpsSequenceBaselineNet",
    "VISION_POSITION_PRESETS",
    "VisionPositionLateFusionNet",
    "VisionPositionTransformerFusionNet",
]
