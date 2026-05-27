from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image import ImageFeatureExtractor
from kd_sensing.models.image_encoders import ResNet18ImageEncoder
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE, MmWaveFeatureExtractor
from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.registries import MODELS


HIST_BEAM_VARIANTS = {
    "v0_flat",
    "flat",
    "v1_hierarchical",
    "hierarchical",
    "v2_shared_private",
    "shared_private",
    "v3_decoupled",
    "decoupled",
    "v4_adapter",
    "adapter",
    "v5_adapter_proto",
    "adapter_proto",
    "v6_full_finetune",
    "full_finetune",
}
DEFAULT_HIST_MODALITIES = ("image", "radar", "gps")


@dataclass(frozen=True)
class HistBeamConfig:
    num_classes: int = 64
    group_size: int = 8
    variant: str = "v3_decoupled"
    modalities: tuple[str, ...] = DEFAULT_HIST_MODALITIES
    lambda_hier: float = 1.0
    lambda_flat: float = 0.2
    lambda_orth: float = 0.01
    lambda_scene_c: float = 0.05
    lambda_scene_s: float = 0.05
    adapter_enabled: bool = False
    prototype_enabled: bool = False

    @property
    def num_groups(self) -> int:
        return self.num_classes // self.group_size

    @property
    def hierarchical_enabled(self) -> bool:
        return self.variant not in {"v0_flat", "flat"}

    @property
    def shared_private_enabled(self) -> bool:
        return self.variant in {
            "v2_shared_private",
            "shared_private",
            "v3_decoupled",
            "decoupled",
            "v4_adapter",
            "adapter",
            "v5_adapter_proto",
            "adapter_proto",
            "v6_full_finetune",
            "full_finetune",
        }

    @property
    def decoupled_enabled(self) -> bool:
        return self.variant in {
            "v3_decoupled",
            "decoupled",
            "v4_adapter",
            "adapter",
            "v5_adapter_proto",
            "adapter_proto",
            "v6_full_finetune",
            "full_finetune",
        }


def resolve_hist_beam_config(
    *,
    num_classes: int = 64,
    group_size: int = 8,
    variant: str = "v3_decoupled",
    modalities: list[str] | tuple[str, ...] | None = None,
    loss_weights: dict[str, Any] | None = None,
    adapter: bool | dict[str, Any] | None = None,
    prototype: bool | dict[str, Any] | None = None,
    **_: Any,
) -> HistBeamConfig:
    classes = int(num_classes)
    group = int(group_size)
    if classes <= 0:
        raise ValueError(f"HiST-Beam num_classes must be positive, got {classes}.")
    if group <= 0:
        raise ValueError(f"HiST-Beam group_size must be positive, got {group}.")
    if classes % group != 0:
        raise ValueError(
            f"HiST-Beam num_classes ({classes}) must be divisible by group_size ({group}). "
            "Use a group_size that evenly partitions the beam classes, for example 8 for 64 classes."
        )
    normalized_variant = str(variant).strip().lower()
    if normalized_variant not in HIST_BEAM_VARIANTS:
        raise ValueError(
            f"Unknown HiST-Beam variant '{variant}'. Available variants: {sorted(HIST_BEAM_VARIANTS)}."
        )
    selected_modalities = normalize_modalities(
        modalities or DEFAULT_HIST_MODALITIES,
        context="HiST-Beam modalities",
    )
    weights = loss_weights or {}
    return HistBeamConfig(
        num_classes=classes,
        group_size=group,
        variant=normalized_variant,
        modalities=selected_modalities,
        lambda_hier=float(weights.get("hierarchical", weights.get("lambda_hier", 1.0))),
        lambda_flat=float(weights.get("flat", weights.get("lambda_flat", 0.2))),
        lambda_orth=float(weights.get("orthogonality", weights.get("lambda_orth", 0.01))),
        lambda_scene_c=float(weights.get("scene_confusion", weights.get("lambda_scene_c", 0.05))),
        lambda_scene_s=float(weights.get("scene_private", weights.get("lambda_scene_s", 0.05))),
        adapter_enabled=_mapping_enabled(adapter)
        or normalized_variant in {"v4_adapter", "adapter", "v5_adapter_proto", "adapter_proto"},
        prototype_enabled=_mapping_enabled(prototype) or normalized_variant in {"v5_adapter_proto", "adapter_proto"},
    )


@MODELS.register("hist_beam_fusion")
class HistBeamFusionNet(nn.Module):
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        feature_size: int = 64,
        d_model: int = 256,
        num_classes: int = 64,
        num_pred: int = 1,
        group_size: int = 8,
        variant: str = "v3_decoupled",
        modalities: list[str] | tuple[str, ...] | None = None,
        loss_weights: dict[str, Any] | None = None,
        adapter: bool | dict[str, Any] | None = None,
        prototype: bool | dict[str, Any] | None = None,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        image_channels: int = 3,
        image_encoder: str | dict[str, Any] | None = None,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        num_scenes: int = 4,
        grl_lambda: float = 1.0,
        image_profile: str | None = "rgb_imagenet",
        **_: Any,
    ):
        super().__init__()
        self.name = "HistBeamFusionNet"
        self.hist_config = resolve_hist_beam_config(
            num_classes=num_classes,
            group_size=group_size,
            variant=variant,
            modalities=modalities,
            loss_weights=loss_weights,
            adapter=adapter,
            prototype=prototype,
        )
        self.modalities = self.hist_config.modalities
        self.feature_size = int(feature_size)
        self.d_model = int(d_model)
        self.num_classes = self.hist_config.num_classes
        self.group_size = self.hist_config.group_size
        self.num_groups = self.hist_config.num_groups
        self.num_pred = int(num_pred)
        self.horizon = self.num_pred
        self.max_seq_len = int(max_seq_len)
        self.cls_type_id = len(MODALITY_ORDER)
        self.num_scenes = int(num_scenes)
        self.grl_lambda = float(grl_lambda)
        if self.num_pred <= 0:
            raise ValueError(f"num_pred must be positive, got {num_pred}.")
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")

        self.encoders = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        for modality in self.modalities:
            self.encoders[modality] = _build_hist_modality_encoder(
                modality,
                self.feature_size,
                image_channels=image_channels,
                image_encoder=image_encoder,
                image_profile=image_profile,
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                lidar_channels=lidar_channels,
                mmwave_input_size=mmwave_input_size,
            )
            self.feature_projections[modality] = (
                nn.Identity() if self.feature_size == self.d_model else nn.Linear(self.feature_size, self.d_model)
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.token_type_embedding = nn.Embedding(len(MODALITY_ORDER) + 1, self.d_model)
        self.time_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=max(self.d_model * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(self.d_model)

        self.flat_head = nn.Linear(self.d_model, self.num_pred * self.num_classes)
        self.shared_branch = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model), nn.GELU())
        self.private_branch = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model), nn.GELU())
        self.private_adapter = BottleneckPrivateAdapter(
            self.d_model,
            bottleneck_dim=_adapter_bottleneck_dim(adapter, self.d_model),
        )
        self.coarse_head = nn.Linear(self.d_model, self.num_pred * self.num_groups)
        self.fine_head = nn.Linear(self.d_model * 2, self.num_pred * self.num_groups * self.group_size)
        self.shared_scene_classifier = nn.Linear(self.d_model, self.num_scenes) if self.num_scenes > 0 else None
        self.private_scene_classifier = nn.Linear(self.d_model, self.num_scenes) if self.num_scenes > 0 else None

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...] | dict[str, Any]]:
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
                raise ValueError(f"HiST-Beam requires '{modality}' input because it is enabled.")
            features = self.feature_projections[modality](self.encoders[modality](tensor))
            batch_size, seq_len = _check_temporal_features(features, modality, batch_size, seq_len)
            modality_features.append(features)
        assert batch_size is not None and seq_len is not None
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")

        stacked = torch.stack(modality_features, dim=1)
        effective_mask = _effective_modality_mask(
            batch_size,
            len(self.modalities),
            device=stacked.device,
            force_modality_mask=force_modality_mask,
        )
        if torch.any(~effective_mask.any(dim=1)):
            raise ValueError("force_modality_mask leaves no available modalities for at least one sample.")
        tokens = self._embed_modality_tokens(stacked)
        token_padding_mask = ~effective_mask.unsqueeze(-1).expand(batch_size, len(self.modalities), seq_len)
        diagnostic_tokens = tokens.masked_fill(token_padding_mask.unsqueeze(-1), 0.0)
        flat_tokens = _serialize_time_first(tokens)
        flat_padding_mask = _serialize_mask_time_first(token_padding_mask)
        cls_ids = torch.full((batch_size, 1), self.cls_type_id, dtype=torch.long, device=stacked.device)
        cls = self.input_dropout(
            self.input_norm(self.cls_token.expand(batch_size, -1, -1) + self.token_type_embedding(cls_ids))
        )
        memory = self.transformer(
            torch.cat([cls, flat_tokens], dim=1),
            src_key_padding_mask=torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=stacked.device), flat_padding_mask],
                dim=1,
            ),
        )
        fused = self.output_norm(memory[:, 0, :])
        flat_logits = self.flat_head(fused).view(batch_size, self.num_pred, self.num_classes)

        shared = self.shared_branch(fused)
        private = self.private_branch(fused)
        adapter_rep = self.private_adapter(private) if self.hist_config.adapter_enabled else private
        coarse_logits = self.coarse_head(shared).view(batch_size, self.num_pred, self.num_groups)
        fine_input = torch.cat([shared, adapter_rep], dim=-1)
        fine_logits = self.fine_head(fine_input).view(batch_size, self.num_pred, self.num_groups, self.group_size)
        beam_log_probs = hierarchical_beam_log_probs(coarse_logits, fine_logits)
        logits = flat_logits if not self.hist_config.hierarchical_enabled else beam_log_probs

        output_features = fused.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous()
        result: dict[str, torch.Tensor | tuple[str, ...] | dict[str, Any]] = {
            "logits": logits,
            "beam_logits": logits,
            "beam_log_probs": beam_log_probs,
            "flat_logits": flat_logits,
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "shared_representation": shared.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous(),
            "private_representation": private.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous(),
            "adapter_representation": adapter_rep.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous(),
            "input_features": _available_timewise_mean(diagnostic_tokens, effective_mask),
            "output_features": output_features,
            "token_features": diagnostic_tokens,
            "modalities": self.modalities,
            "effective_modality_mask": effective_mask,
            "fusion_memory": memory,
            "scene_diagnostics": self.scene_diagnostics(shared, private),
            "hist_beam": {
                "variant": self.hist_config.variant,
                "num_classes": self.num_classes,
                "group_size": self.group_size,
                "num_groups": self.num_groups,
                "adapter_enabled": self.hist_config.adapter_enabled,
                "prototype_enabled": self.hist_config.prototype_enabled,
            },
        }
        if self.shared_scene_classifier is not None:
            result["shared_scene_logits"] = self.shared_scene_classifier(
                gradient_reverse(shared, lambda_=self.grl_lambda)
            )
        if self.private_scene_classifier is not None:
            result["private_scene_logits"] = self.private_scene_classifier(private)
        return result

    def scene_diagnostics(self, shared: torch.Tensor, private: torch.Tensor) -> dict[str, Any]:
        return {
            "shared_norm": float(shared.detach().norm(dim=-1).mean().cpu().item()),
            "private_norm": float(private.detach().norm(dim=-1).mean().cpu().item()),
            "shared_private_cosine": float(
                F.cosine_similarity(shared.detach(), private.detach(), dim=-1).mean().cpu().item()
            ),
        }

    def _embed_modality_tokens(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, seq_len, _ = features.shape
        time_ids = torch.arange(seq_len, device=features.device)
        time = self.time_embedding(time_ids).view(1, 1, seq_len, self.d_model)
        type_ids = torch.tensor(
            [MODALITY_ORDER.index(name) for name in self.modalities],
            dtype=torch.long,
            device=features.device,
        )
        token_type = self.token_type_embedding(type_ids).view(1, modality_count, 1, self.d_model)
        return self.input_dropout(self.input_norm(features + time + token_type))


class BottleneckPrivateAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int | None = None):
        super().__init__()
        hidden = int(bottleneck_dim or max(dim // 4, 1))
        self.down = nn.Linear(dim, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.activation(self.down(x)))


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: torch.Tensor, *, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, float(lambda_))


def hierarchical_beam_log_probs(coarse_logits: torch.Tensor, fine_logits: torch.Tensor) -> torch.Tensor:
    if coarse_logits.ndim != 3:
        raise ValueError(f"coarse_logits must have shape [B, H, G], got {tuple(coarse_logits.shape)}.")
    if fine_logits.ndim != 4:
        raise ValueError(f"fine_logits must have shape [B, H, G, S], got {tuple(fine_logits.shape)}.")
    if coarse_logits.shape[:3] != fine_logits.shape[:3]:
        raise ValueError("coarse_logits and fine_logits must share [B, H, G] dimensions.")
    coarse_lp = F.log_softmax(coarse_logits, dim=-1).unsqueeze(-1)
    fine_lp = F.log_softmax(fine_logits, dim=-1)
    return (coarse_lp + fine_lp).reshape(*coarse_logits.shape[:2], -1)


def _build_hist_modality_encoder(
    modality: str,
    feature_size: int,
    *,
    image_channels: int,
    image_encoder: str | dict[str, Any] | None,
    image_profile: str | None,
    radar_channels: int,
    gps_input_size: int,
    lidar_channels: int,
    mmwave_input_size: int,
) -> nn.Module:
    if modality == "image":
        encoder_cfg = image_encoder
        if encoder_cfg is None:
            encoder_cfg = {
                "type": "resnet18_imagenet_rgb",
                "output_dim": feature_size,
                "pretrained": False,
                "weights": None,
            }
        if isinstance(encoder_cfg, str):
            encoder_cfg = {"type": encoder_cfg}
        if isinstance(encoder_cfg, dict) and encoder_cfg.get("type") == "resnet18_imagenet_rgb":
            cfg = dict(encoder_cfg)
            cfg.pop("type", None)
            cfg.setdefault("output_dim", feature_size)
            cfg.setdefault("image_profile", image_profile)
            cfg.setdefault("image_channels", image_channels)
            return ResNet18ImageEncoder(**cfg)
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
    raise ValueError(f"Unknown HiST-Beam modality '{modality}'. Available modalities: {available}.")


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
        raise ValueError("Enabled HiST-Beam modalities must share batch and sequence dimensions.")
    return current_batch, current_seq


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


def _serialize_time_first(tokens: torch.Tensor) -> torch.Tensor:
    batch_size, modality_count, seq_len, d_model = tokens.shape
    return tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len * modality_count, d_model)


def _serialize_mask_time_first(mask: torch.Tensor) -> torch.Tensor:
    batch_size, modality_count, seq_len = mask.shape
    return mask.permute(0, 2, 1).contiguous().view(batch_size, seq_len * modality_count)


def _available_timewise_mean(tokens: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
    valid = effective_mask.to(device=tokens.device, dtype=tokens.dtype).view(tokens.shape[0], tokens.shape[1], 1, 1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return (tokens * valid).sum(dim=1) / counts


def _mapping_enabled(value: bool | dict[str, Any] | None) -> bool:
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return bool(value)


def _adapter_bottleneck_dim(adapter: bool | dict[str, Any] | None, dim: int) -> int:
    if isinstance(adapter, dict):
        return int(adapter.get("bottleneck_dim", adapter.get("hidden_dim", max(dim // 4, 1))))
    return max(dim // 4, 1)


__all__ = [
    "DEFAULT_HIST_MODALITIES",
    "HIST_BEAM_VARIANTS",
    "BottleneckPrivateAdapter",
    "HistBeamConfig",
    "HistBeamFusionNet",
    "gradient_reverse",
    "hierarchical_beam_log_probs",
    "resolve_hist_beam_config",
]
