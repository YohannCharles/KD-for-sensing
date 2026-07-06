import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from kd_sensing.modalities import image_profile_spec, validate_image_encoder_profile
from kd_sensing.models.jepa_downstream import (
    GPSQueryPool,
    PredictiveGPSQueryPool,
    build_jepa_downstream_adapter,
    build_jepa_downstream_pooler,
    normalize_jepa_downstream_adapter_config,
    normalize_jepa_downstream_pooler_config,
)
from kd_sensing.models.jepa_helpers import (
    CHECKPOINT_POLICIES,
    VisualTokenMetadata,
    _apply_temporal_context_fallback,
    _compute_temporal_auxiliary_prediction,
    _conv_grid,
    _image_size_pair,
    _load_context_encoder_state,
    _metadata_dict,
    _metadata_token_count,
    _metadata_token_grid,
    _normalize_checkpoint_policy,
    _normalize_temporal_auxiliary_config,
    _normalize_temporal_fallback_config,
    _normalize_visual_encoder_type,
    _positive_int,
    _token_budget_error,
    _visual_token_diagnostics,
    normalize_visual_token_encoder_config,
    visual_token_metadata_from_encoder,
)
from kd_sensing.registries import ENCODERS, JEPA_VISUAL_TOKEN_ENCODERS, MODELS


@dataclass(frozen=True)
class JepaMaskSample:
    context_mask: torch.Tensor
    target_mask: torch.Tensor
    loss_mask: torch.Tensor
    context_indices: torch.Tensor
    target_indices: torch.Tensor
    diagnostics: dict[str, float | str]


def build_visual_token_encoder(cfg: Any = None, **extra_kwargs: Any) -> nn.Module:
    return JEPA_VISUAL_TOKEN_ENCODERS.build(
        normalize_visual_token_encoder_config(cfg, **extra_kwargs)
    )


@JEPA_VISUAL_TOKEN_ENCODERS.register("patch_vit")
@JEPA_VISUAL_TOKEN_ENCODERS.register("patch")
class VisualPatchTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_channels: int = 3,
        latent_dim: int = 64,
        patch_size: int = 16,
        kernel_size: int | None = None,
        stride: int | None = None,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_tokens: int = 256,
        image_profile: str | None = None,
        image_size: int | list[int] | tuple[int, int] | None = None,
        variant_id: str | None = None,
        token_source: str = "patch_vit",
        visual_encoder_type: str = "patch_vit",
        positional_encoding: str = "learned_absolute",
        checkpoint_policy: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        validate_image_encoder_profile(
            encoder_name="gps_conditioned_jepa",
            image_profile=image_profile,
            expected_channels=image_profile_spec(image_profile).channels,
            actual_channels=image_channels,
        )
        self.latent_dim = int(latent_dim)
        self.patch_size = _positive_int(patch_size, "patch_size")
        self.kernel_size = _positive_int(kernel_size or patch_size, "kernel_size")
        self.stride = _positive_int(stride or patch_size, "stride")
        self.max_tokens = _positive_int(max_tokens, "max_tokens")
        self.image_size = _image_size_pair(image_size)
        self.variant_id = str(variant_id or f"patch{self.patch_size}").strip() or f"patch{self.patch_size}"
        self.token_source = str(token_source).strip() or "patch_vit"
        self.visual_encoder_type = _normalize_visual_encoder_type(visual_encoder_type)
        self.positional_encoding = str(positional_encoding).strip() or "learned_absolute"
        default_policy = "exact_reuse" if self.patch_size == 16 and self.kernel_size == 16 and self.stride == 16 else "fresh_stage1_required"
        self.checkpoint_policy = _normalize_checkpoint_policy(checkpoint_policy, default=default_policy)
        self.patch_embed = nn.Conv2d(
            int(image_channels),
            self.latent_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=max(int(self.latent_dim * float(mlp_ratio)), self.latent_dim),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(depth), 0)) if int(depth) > 0 else nn.Identity()
        self.norm = nn.LayerNorm(self.latent_dim)
        self.last_metadata: VisualTokenMetadata | None = None
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if image_batch.ndim != 5:
            raise ValueError(f"JEPA image_batch must have shape [B, T, C, H, W], got {tuple(image_batch.shape)}.")
        batch_size, seq_len, channels, height, width = image_batch.shape
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width)
        patches = self._patches_from_frames(frames)
        patches = self._mix_patch_grid(patches)
        return self._finish_tokens(patches, batch_size=batch_size, seq_len=seq_len, image_size=(int(height), int(width)))

    def _patches_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(frames)

    def _mix_patch_grid(self, patches: torch.Tensor) -> torch.Tensor:
        return patches

    def _finish_tokens(
        self,
        patches: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        image_size: tuple[int, int],
        scale_token_counts: dict[str, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        grid_size = (int(patches.shape[-2]), int(patches.shape[-1]))
        tokens = patches.flatten(2).transpose(1, 2)
        if tokens.shape[1] > self.max_tokens:
            raise _token_budget_error(
                token_count=int(tokens.shape[1]),
                max_tokens=self.max_tokens,
                image_size=image_size,
                variant_type=self.visual_encoder_type,
            )
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :].to(dtype=tokens.dtype, device=tokens.device)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        self.last_metadata = self._metadata(
            image_size=image_size,
            grid_size=grid_size,
            token_count=int(tokens.shape[1]),
            scale_token_counts=scale_token_counts,
        )
        return tokens.reshape(batch_size, seq_len, tokens.shape[1], self.latent_dim), grid_size

    def _metadata(
        self,
        *,
        image_size: tuple[int, int] | None = None,
        grid_size: tuple[int, int] | None = None,
        token_count: int | None = None,
        scale_token_counts: dict[str, int] | None = None,
    ) -> VisualTokenMetadata:
        resolved_image = image_size or self.image_size
        resolved_grid = grid_size or _conv_grid(resolved_image, kernel_size=self.kernel_size, stride=self.stride)
        resolved_count = int(token_count if token_count is not None else resolved_grid[0] * resolved_grid[1])
        return VisualTokenMetadata(
            variant_id=self.variant_id,
            visual_encoder_type=self.visual_encoder_type,
            token_source=self.token_source,
            image_size=resolved_image,
            effective_stride=(self.stride, self.stride),
            token_grid=resolved_grid,
            token_count=resolved_count,
            positional_encoding=self.positional_encoding,
            checkpoint_policy=self.checkpoint_policy,
            max_tokens=self.max_tokens,
            scale_token_counts=scale_token_counts,
        )

    def visual_token_metadata(self) -> dict[str, Any]:
        return (self.last_metadata or self._metadata()).to_dict()

    def training_strategy_metadata(self) -> dict[str, Any]:
        return self.visual_token_metadata()


@JEPA_VISUAL_TOKEN_ENCODERS.register("overlap_patch")
class OverlapPatchTokenEncoder(VisualPatchTokenEncoder):
    def __init__(
        self,
        *,
        kernel_size: int = 16,
        stride: int = 8,
        patch_size: int | None = None,
        variant_id: str | None = None,
        checkpoint_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            patch_size=int(patch_size or kernel_size),
            kernel_size=kernel_size,
            stride=stride,
            variant_id=variant_id or f"overlap_k{int(kernel_size)}_s{int(stride)}",
            token_source="overlap_patch",
            visual_encoder_type="overlap_patch",
            checkpoint_policy=checkpoint_policy or "fresh_stage1_required",
            **kwargs,
        )


@JEPA_VISUAL_TOKEN_ENCODERS.register("local_token_mixing")
class LocalTokenMixingEncoder(VisualPatchTokenEncoder):
    def __init__(
        self,
        *,
        local_kernel_size: int = 3,
        variant_id: str | None = None,
        checkpoint_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            variant_id=variant_id,
            token_source="patch_vit_with_local_mixing",
            visual_encoder_type="local_token_mixing",
            checkpoint_policy=checkpoint_policy or "partial_reuse",
            **kwargs,
        )
        kernel = _positive_int(local_kernel_size, "local_kernel_size")
        self.local_mixer = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=kernel, padding=kernel // 2, groups=self.latent_dim),
            nn.GELU(),
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=1),
        )
        if self.variant_id == f"patch{self.patch_size}":
            self.variant_id = f"patch{self.patch_size}_local_k{kernel}"

    def _mix_patch_grid(self, patches: torch.Tensor) -> torch.Tensor:
        return patches + self.local_mixer(patches)


@JEPA_VISUAL_TOKEN_ENCODERS.register("cvt")
class CvTStyleTokenEncoder(VisualPatchTokenEncoder):
    def __init__(
        self,
        *,
        cvt_kernel_size: int = 3,
        variant_id: str | None = None,
        checkpoint_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            variant_id=variant_id,
            token_source="patch_vit_with_cvt_depthwise_projection",
            visual_encoder_type="cvt",
            checkpoint_policy=checkpoint_policy or "partial_reuse",
            **kwargs,
        )
        kernel = _positive_int(cvt_kernel_size, "cvt_kernel_size")
        self.depthwise_projection = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=kernel, padding=kernel // 2, groups=self.latent_dim),
            nn.BatchNorm2d(self.latent_dim),
            nn.GELU(),
        )
        if self.variant_id == f"patch{self.patch_size}":
            self.variant_id = f"patch{self.patch_size}_cvt_k{kernel}"

    def _mix_patch_grid(self, patches: torch.Tensor) -> torch.Tensor:
        return self.depthwise_projection(patches)


@JEPA_VISUAL_TOKEN_ENCODERS.register("conv_stem")
class ConvStemTokenEncoder(VisualPatchTokenEncoder):
    def __init__(
        self,
        *,
        image_channels: int = 3,
        latent_dim: int = 64,
        stem_channels: list[int] | tuple[int, ...] | None = None,
        stem_strides: list[int] | tuple[int, ...] | None = None,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_tokens: int = 256,
        image_profile: str | None = None,
        image_size: int | list[int] | tuple[int, int] | None = None,
        variant_id: str | None = None,
        positional_encoding: str = "learned_absolute",
        checkpoint_policy: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            image_channels=image_channels,
            latent_dim=latent_dim,
            patch_size=16,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_tokens=max_tokens,
            image_profile=image_profile,
            image_size=image_size,
            variant_id=variant_id or "conv_stem_s16",
            token_source="conv_stem",
            visual_encoder_type="conv_stem",
            positional_encoding=positional_encoding,
            checkpoint_policy=checkpoint_policy or "fresh_stage1_required",
        )
        channels = [int(item) for item in (stem_channels or (latent_dim // 2, latent_dim // 2, latent_dim))]
        strides = [int(item) for item in (stem_strides or (2, 2, 4))]
        if len(channels) != len(strides):
            raise ValueError("conv_stem stem_channels and stem_strides must have the same length.")
        layers: list[nn.Module] = []
        in_channels = int(image_channels)
        effective_stride = 1
        for out_channels, stride in zip(channels, strides):
            if out_channels <= 0 or stride <= 0:
                raise ValueError("conv_stem channels and strides must be positive.")
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
            effective_stride *= int(stride)
        if in_channels != self.latent_dim:
            layers.append(nn.Conv2d(in_channels, self.latent_dim, kernel_size=1))
        self.stem = nn.Sequential(*layers)
        self.effective_stem_stride = int(effective_stride)
        self.patch_embed = nn.Identity()
        self.kernel_size = 1
        self.stride = self.effective_stem_stride

    def _patches_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stem(frames)


def _build_torchvision_resnet_feature_extractor(
    *,
    backbone: str,
    stage: str,
    pretrained: bool,
    weights: str | None,
) -> tuple[nn.Module, int]:
    try:
        import torchvision.models as tv_models
    except Exception as exc:  # pragma: no cover - environment-dependent.
        raise RuntimeError(
            "JEPA CNN feature-map token encoder requires torchvision in the kd_mm_beam environment."
        ) from exc
    backbone_name = str(backbone).strip().lower()
    if backbone_name not in {"resnet18", "resnet34"}:
        raise ValueError("JEPA CNN feature-map token encoder supports resnet18 or resnet34.")
    stage = str(stage).strip().lower()
    if stage not in {"layer3", "layer4"}:
        raise ValueError("JEPA CNN feature-map token encoder stage must be layer3 or layer4.")
    builder = getattr(tv_models, backbone_name)
    weights_obj = None
    if pretrained:
        enum_name = "ResNet18_Weights" if backbone_name == "resnet18" else "ResNet34_Weights"
        enum = getattr(tv_models, enum_name, None)
        if enum is None:
            model = builder(pretrained=True)
        else:
            if weights in (None, "", "none", "None"):
                weights_obj = None
            elif weights in ("DEFAULT", "default"):
                weights_obj = enum.DEFAULT
            else:
                weights_obj = getattr(enum, str(weights))
            model = builder(weights=weights_obj)
    else:
        model = builder(weights=None)
    modules = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3]
    channels = 256
    if stage == "layer4":
        modules.append(model.layer4)
        channels = 512
    return nn.Sequential(*modules), channels


@JEPA_VISUAL_TOKEN_ENCODERS.register("cnn_feature_map")
class CNNFeatureMapTokenEncoder(VisualPatchTokenEncoder):
    def __init__(
        self,
        *,
        image_channels: int = 3,
        latent_dim: int = 64,
        backbone: str = "resnet18",
        stage: str = "layer4",
        pretrained: bool = False,
        weights: str | None = None,
        freeze_backbone: bool = True,
        max_tokens: int = 256,
        image_profile: str | None = "rgb_imagenet",
        image_size: int | list[int] | tuple[int, int] | None = None,
        variant_id: str | None = None,
        token_source: str = "cnn_feature_map",
        visual_encoder_type: str = "cnn_feature_map",
        positional_encoding: str = "learned_absolute",
        checkpoint_policy: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("patch_size", None)
        kwargs.pop("kernel_size", None)
        kwargs.pop("stride", None)
        kwargs.pop("depth", None)
        super().__init__(
            image_channels=image_channels,
            latent_dim=latent_dim,
            patch_size=16,
            depth=0,
            max_tokens=max_tokens,
            image_profile=image_profile,
            image_size=image_size,
            variant_id=variant_id or f"{backbone}_{stage}_tokens",
            token_source=token_source,
            visual_encoder_type=visual_encoder_type,
            positional_encoding=positional_encoding,
            checkpoint_policy=checkpoint_policy or "supervised_only_anchor",
            **kwargs,
        )
        self.backbone_name = str(backbone).strip().lower()
        self.stage = str(stage).strip().lower()
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone, backbone_dim = _build_torchvision_resnet_feature_extractor(
            backbone=self.backbone_name,
            stage=self.stage,
            pretrained=self.pretrained,
            weights=weights,
        )
        if int(image_channels) != 3:
            raise ValueError("JEPA CNN feature-map token encoder currently requires RGB image_channels=3.")
        self.projection = nn.Conv2d(backbone_dim, self.latent_dim, kernel_size=1)
        self.patch_embed = nn.Identity()
        self.stride = 16 if self.stage == "layer3" else 32
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _patches_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(frames))

    def _metadata(
        self,
        *,
        image_size: tuple[int, int] | None = None,
        grid_size: tuple[int, int] | None = None,
        token_count: int | None = None,
        scale_token_counts: dict[str, int] | None = None,
    ) -> VisualTokenMetadata:
        base = super()._metadata(
            image_size=image_size,
            grid_size=grid_size,
            token_count=token_count,
            scale_token_counts=scale_token_counts,
        )
        return VisualTokenMetadata(
            variant_id=base.variant_id,
            visual_encoder_type=base.visual_encoder_type,
            token_source=base.token_source,
            image_size=base.image_size,
            effective_stride=base.effective_stride,
            token_grid=base.token_grid,
            token_count=base.token_count,
            positional_encoding=base.positional_encoding,
            checkpoint_policy=base.checkpoint_policy,
            max_tokens=base.max_tokens,
            backbone=self.backbone_name,
            stages=(self.stage,),
            pretrained=self.pretrained,
            freeze_backbone=self.freeze_backbone,
        )


@JEPA_VISUAL_TOKEN_ENCODERS.register("tinyvit_frame")
class TinyViTFrameTokenEncoder(nn.Module):
    """Expose a TinyViT frame encoder as a single JEPA visual token."""

    def __init__(
        self,
        *,
        image_channels: int = 3,
        latent_dim: int = 64,
        encoder_type: str = "tinyvit_5m_scratch_rgb",
        output_dim: int | None = None,
        variant_id: str | None = None,
        token_source: str = "tinyvit_frame",
        visual_encoder_type: str = "tinyvit_frame",
        image_profile: str | None = "rgb_imagenet",
        max_tokens: int = 1,
        checkpoint_policy: str | None = None,
        freeze_backbone: bool = False,
        allow_download: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        import kd_sensing.models.tinyvit  # noqa: F401

        validate_image_encoder_profile(
            encoder_name=str(encoder_type),
            image_profile=image_profile,
            expected_channels=3,
            actual_channels=image_channels,
        )
        self.latent_dim = int(output_dim if output_dim is not None else latent_dim)
        self.encoder_type = str(encoder_type)
        self.variant_id = str(variant_id or self.encoder_type).strip() or self.encoder_type
        self.token_source = str(token_source).strip() or "tinyvit_frame"
        self.visual_encoder_type = _normalize_visual_encoder_type(visual_encoder_type)
        self.max_tokens = max(1, int(max_tokens))
        self.checkpoint_policy = _normalize_checkpoint_policy(
            checkpoint_policy,
            default="fresh_stage1_required",
        )
        for stale_key in ("patch_size", "kernel_size", "stride", "depth"):
            kwargs.pop(stale_key, None)
        self.encoder = ENCODERS.build(
            {
                "type": self.encoder_type,
                "output_dim": self.latent_dim,
                "image_channels": int(image_channels),
                "image_profile": image_profile,
                "freeze_backbone": bool(freeze_backbone),
                "allow_download": bool(allow_download),
                **kwargs,
            }
        )
        self.last_metadata: VisualTokenMetadata | None = None

    def forward(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        features = self.encoder(image_batch)
        if features.ndim != 3:
            raise ValueError(
                "TinyViT JEPA frame token encoder expects [B, T, D] features from the image encoder, "
                f"got {tuple(features.shape)}."
            )
        tokens = features.unsqueeze(2)
        self.last_metadata = VisualTokenMetadata(
            variant_id=self.variant_id,
            visual_encoder_type=self.visual_encoder_type,
            token_source=self.token_source,
            image_size=(224, 224),
            effective_stride=(224, 224),
            token_grid=(1, 1),
            token_count=1,
            positional_encoding="tinyvit_global_frame",
            checkpoint_policy=self.checkpoint_policy,
            max_tokens=self.max_tokens,
            backbone=self.encoder_type,
            stages=("global_frame",),
            pretrained="22k" in self.encoder_type,
            freeze_backbone=bool(getattr(self.encoder, "freeze_backbone", False)),
        )
        return tokens, (1, 1)

    def visual_token_metadata(self) -> dict[str, Any]:
        return (self.last_metadata or self._metadata()).to_dict()

    def training_strategy_metadata(self) -> dict[str, Any]:
        metadata = self.visual_token_metadata()
        if hasattr(self.encoder, "training_strategy_metadata"):
            metadata["image_encoder"] = self.encoder.training_strategy_metadata()
        return metadata

    def _metadata(self) -> VisualTokenMetadata:
        return VisualTokenMetadata(
            variant_id=self.variant_id,
            visual_encoder_type=self.visual_encoder_type,
            token_source=self.token_source,
            image_size=(224, 224),
            effective_stride=(224, 224),
            token_grid=(1, 1),
            token_count=1,
            positional_encoding="tinyvit_global_frame",
            checkpoint_policy=self.checkpoint_policy,
            max_tokens=self.max_tokens,
            backbone=self.encoder_type,
            stages=("global_frame",),
            pretrained="22k" in self.encoder_type,
            freeze_backbone=bool(getattr(self.encoder, "freeze_backbone", False)),
        )


@JEPA_VISUAL_TOKEN_ENCODERS.register("multi_scale_cnn")
class MultiScaleCNNTokenEncoder(CNNFeatureMapTokenEncoder):
    def __init__(
        self,
        *,
        stages: list[str] | tuple[str, ...] | None = None,
        backbone: str = "resnet18",
        latent_dim: int = 64,
        pretrained: bool = False,
        weights: str | None = None,
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            stage="layer4",
            latent_dim=latent_dim,
            pretrained=pretrained,
            weights=weights,
            freeze_backbone=freeze_backbone,
            variant_id=kwargs.pop("variant_id", None) or f"{backbone}_layer3_layer4_tokens",
            token_source="multi_scale_cnn",
            visual_encoder_type="multi_scale_cnn",
            **kwargs,
        )
        selected = tuple(str(item).strip().lower() for item in (stages or ("layer3", "layer4")))
        if selected != ("layer3", "layer4"):
            raise ValueError("JEPA multi_scale_cnn currently supports stages ['layer3', 'layer4'].")
        self.stages = selected
        self.backbone_l3, dim3 = _build_torchvision_resnet_feature_extractor(
            backbone=self.backbone_name,
            stage="layer3",
            pretrained=self.pretrained,
            weights=weights,
        )
        self.projection_l3 = nn.Conv2d(dim3, self.latent_dim, kernel_size=1)
        self.scale_embedding = nn.Parameter(torch.zeros(len(self.stages), self.latent_dim))
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)
        if self.freeze_backbone:
            for param in self.backbone_l3.parameters():
                param.requires_grad_(False)

    def forward(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if image_batch.ndim != 5:
            raise ValueError(f"JEPA image_batch must have shape [B, T, C, H, W], got {tuple(image_batch.shape)}.")
        batch_size, seq_len, channels, height, width = image_batch.shape
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width)
        l3 = self.projection_l3(self.backbone_l3(frames))
        l4 = self.projection(self.backbone(frames))
        pieces = []
        scale_counts: dict[str, int] = {}
        for scale_index, (name, fmap) in enumerate((("layer3", l3), ("layer4", l4))):
            grid = (int(fmap.shape[-2]), int(fmap.shape[-1]))
            tokens = fmap.flatten(2).transpose(1, 2)
            tokens = tokens + self.scale_embedding[scale_index].to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)
            pieces.append(tokens)
            scale_counts[name] = int(grid[0] * grid[1])
        tokens = torch.cat(pieces, dim=1)
        if tokens.shape[1] > self.max_tokens:
            raise _token_budget_error(
                token_count=int(tokens.shape[1]),
                max_tokens=self.max_tokens,
                image_size=(int(height), int(width)),
                variant_type=self.visual_encoder_type,
            )
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :].to(dtype=tokens.dtype, device=tokens.device)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        grid_size = (int(l3.shape[-2]), int(l3.shape[-1]))
        self.last_metadata = self._metadata(
            image_size=(int(height), int(width)),
            grid_size=grid_size,
            token_count=int(tokens.shape[1]),
            scale_token_counts=scale_counts,
        )
        return tokens.reshape(batch_size, seq_len, tokens.shape[1], self.latent_dim), grid_size

    def _metadata(
        self,
        *,
        image_size: tuple[int, int] | None = None,
        grid_size: tuple[int, int] | None = None,
        token_count: int | None = None,
        scale_token_counts: dict[str, int] | None = None,
    ) -> VisualTokenMetadata:
        base = super()._metadata(
            image_size=image_size,
            grid_size=grid_size,
            token_count=token_count,
            scale_token_counts=scale_token_counts,
        )
        return VisualTokenMetadata(
            variant_id=base.variant_id,
            visual_encoder_type="multi_scale_cnn",
            token_source="multi_scale_cnn",
            image_size=base.image_size,
            effective_stride=base.effective_stride,
            token_grid=base.token_grid,
            token_count=base.token_count,
            positional_encoding=base.positional_encoding,
            checkpoint_policy=base.checkpoint_policy,
            max_tokens=base.max_tokens,
            backbone=self.backbone_name,
            stages=self.stages,
            pretrained=self.pretrained,
            freeze_backbone=self.freeze_backbone,
            scale_token_counts=scale_token_counts,
        )


@ENCODERS.register("patchvit_frame")
@ENCODERS.register("lightweight_patchvit_frame")
class LightweightPatchViTFrameEncoder(nn.Module):
    """Frame-level wrapper around the existing lightweight patch ViT tokenizer."""

    expected_image_profile = "rgb_imagenet"

    def __init__(
        self,
        *,
        output_dim: int | None = None,
        latent_dim: int = 64,
        image_channels: int | None = None,
        lidar_channels: int | None = None,
        in_channels: int | None = None,
        image_profile: str | None = "rgb_imagenet",
        visual_encoder: dict[str, Any] | None = None,
        patch_size: int = 16,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_tokens: int = 256,
        pooling: str = "mean",
        variant_id: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim if output_dim is not None else latent_dim)
        self.latent_dim = int(latent_dim)
        self.input_channels = int(in_channels or image_channels or lidar_channels or 3)
        self.pooling = str(pooling or "mean").strip().lower()
        if self.pooling != "mean":
            raise ValueError("lightweight_patchvit_frame currently supports pooling='mean' only.")
        encoder_cfg = dict(visual_encoder or {})
        encoder_cfg.setdefault("type", "patch_vit")
        encoder_cfg.setdefault("image_channels", self.input_channels)
        encoder_cfg.setdefault("latent_dim", self.latent_dim)
        encoder_cfg.setdefault("image_profile", image_profile)
        encoder_cfg.setdefault("patch_size", int(patch_size))
        encoder_cfg.setdefault("depth", int(depth))
        encoder_cfg.setdefault("num_heads", int(num_heads))
        encoder_cfg.setdefault("mlp_ratio", float(mlp_ratio))
        encoder_cfg.setdefault("dropout", float(dropout))
        encoder_cfg.setdefault("max_tokens", int(max_tokens))
        if variant_id is not None:
            encoder_cfg.setdefault("variant_id", str(variant_id))
        self.visual_encoder_config = normalize_visual_token_encoder_config(encoder_cfg)
        self.context_encoder = build_visual_token_encoder(self.visual_encoder_config)
        self.dropout = nn.Dropout(float(dropout))
        self.adapter = nn.Identity() if self.output_dim == self.latent_dim else nn.Linear(self.latent_dim, self.output_dim)
        self.last_visual_token_metadata: dict[str, Any] = {}

    def forward(self, frame_batch: torch.Tensor) -> torch.Tensor:
        tokens, token_info = self.context_encoder(frame_batch)
        metadata = visual_token_metadata_from_encoder(self.context_encoder, token_info, tokens)
        self.last_visual_token_metadata = metadata
        pooled = tokens.mean(dim=2)
        return self.adapter(self.dropout(pooled))

    def training_strategy_metadata(self) -> dict[str, Any]:
        visual_metadata = (
            self.context_encoder.visual_token_metadata()
            if hasattr(self.context_encoder, "visual_token_metadata")
            else dict(self.last_visual_token_metadata)
        )
        return {
            "encoder": "lightweight_patchvit_frame",
            "visual_token_encoder": dict(visual_metadata),
            "visual_token_metadata": dict(visual_metadata),
            "pooling": self.pooling,
            "input_channels": self.input_channels,
            "latent_dim": self.latent_dim,
            "output_dim": self.output_dim,
        }


class JepaMaskSampler(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "random",
        context_ratio: float = 0.6,
        target_ratio: float = 0.2,
        seed: int = 0,
        angle_feature_index: int = 1,
        angle_concentration: float = 3.0,
    ) -> None:
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"random", "gps_angle_biased"}:
            raise ValueError("JEPA mask sampler mode must be random or gps_angle_biased.")
        self.context_ratio = _ratio(context_ratio, "context_ratio")
        self.target_ratio = _ratio(target_ratio, "target_ratio")
        if self.context_ratio + self.target_ratio > 1.0:
            raise ValueError("JEPA context_ratio + target_ratio must not exceed 1.0.")
        self.seed = int(seed)
        self.angle_feature_index = int(angle_feature_index)
        self.angle_concentration = float(angle_concentration)

    def sample(
        self,
        *,
        batch_size: int,
        seq_len: int,
        num_tokens: int,
        grid_size: tuple[int, int],
        gps_batch: torch.Tensor,
        token_metadata: VisualTokenMetadata | Mapping[str, Any] | None = None,
        epoch: int = 0,
        step: int = 0,
        device: torch.device | None = None,
    ) -> JepaMaskSample:
        device = device or gps_batch.device
        grid_size = _metadata_token_grid(token_metadata, grid_size)
        num_tokens = _metadata_token_count(token_metadata, num_tokens)
        if num_tokens <= 0:
            raise ValueError("JEPA mask sampler requires at least one visual token.")
        if num_tokens == 1:
            indices = torch.zeros(batch_size, seq_len, 1, dtype=torch.long, device=device)
            mask = torch.ones(batch_size, seq_len, 1, dtype=torch.bool, device=device)
            diagnostics = {
                "jepa/mask_mode": self.mode,
                "jepa/mask_context_ratio": 1.0,
                "jepa/mask_target_ratio": 1.0,
                "jepa/target_tokens": 1.0,
                "jepa/token_count": 1.0,
                "jepa/token_grid_h": float(grid_size[0]),
                "jepa/token_grid_w": float(grid_size[1]),
                "jepa/degenerate_single_token_mask": 1.0,
            }
            metadata_payload = _metadata_dict(token_metadata)
            if metadata_payload:
                diagnostics["jepa/visual_encoder_type"] = str(
                    metadata_payload.get("visual_encoder_type") or metadata_payload.get("visual_encoder.type") or ""
                )
                diagnostics["jepa/checkpoint_policy"] = str(metadata_payload.get("checkpoint_policy") or "")
            return JepaMaskSample(
                context_mask=mask,
                target_mask=mask,
                loss_mask=mask,
                context_indices=indices,
                target_indices=indices,
                diagnostics=diagnostics,
            )
        n_context = min(max(1, int(round(num_tokens * self.context_ratio))), max(num_tokens - 1, 1))
        n_target = min(max(1, int(round(num_tokens * self.target_ratio))), max(num_tokens - n_context, 1))
        context_indices = torch.empty(batch_size, seq_len, n_context, dtype=torch.long, device=device)
        target_indices = torch.empty(batch_size, seq_len, n_target, dtype=torch.long, device=device)
        gps_cpu = gps_batch.detach().cpu()
        for batch_idx in range(batch_size):
            for time_idx in range(seq_len):
                gen = torch.Generator().manual_seed(self._sample_seed(epoch, step, batch_idx, time_idx))
                target = self._sample_target_indices(
                    n_target,
                    num_tokens=num_tokens,
                    grid_size=grid_size,
                    gps=gps_cpu[batch_idx, time_idx],
                    generator=gen,
                )
                target_set = set(int(value) for value in target.tolist())
                remaining = torch.tensor([idx for idx in range(num_tokens) if idx not in target_set], dtype=torch.long)
                order = remaining[torch.randperm(len(remaining), generator=gen)]
                context = order[:n_context]
                target_indices[batch_idx, time_idx] = target.to(device=device)
                context_indices[batch_idx, time_idx] = context.to(device=device)
        context_mask = _indices_to_mask(context_indices, num_tokens)
        target_mask = _indices_to_mask(target_indices, num_tokens)
        if torch.any(context_mask & target_mask):
            raise RuntimeError("JEPA sampler produced overlapping context and target masks.")
        loss_mask = torch.ones(batch_size, seq_len, n_target, dtype=torch.bool, device=device)
        diagnostics = {
            "jepa/mask_mode": self.mode,
            "jepa/mask_context_ratio": float(context_mask.float().mean().detach().cpu().item()),
            "jepa/mask_target_ratio": float(target_mask.float().mean().detach().cpu().item()),
            "jepa/target_tokens": float(n_target),
            "jepa/token_count": float(num_tokens),
            "jepa/token_grid_h": float(grid_size[0]),
            "jepa/token_grid_w": float(grid_size[1]),
        }
        metadata_payload = _metadata_dict(token_metadata)
        if metadata_payload:
            diagnostics["jepa/visual_encoder_type"] = str(
                metadata_payload.get("visual_encoder_type") or metadata_payload.get("visual_encoder.type") or ""
            )
            diagnostics["jepa/checkpoint_policy"] = str(metadata_payload.get("checkpoint_policy") or "")
            if isinstance(metadata_payload.get("scale_token_counts"), dict):
                diagnostics["jepa/multiscale_token_count"] = float(
                    sum(int(value) for value in metadata_payload["scale_token_counts"].values())
                )
        return JepaMaskSample(
            context_mask=context_mask,
            target_mask=target_mask,
            loss_mask=loss_mask,
            context_indices=context_indices,
            target_indices=target_indices,
            diagnostics=diagnostics,
        )

    def _sample_seed(self, epoch: int, step: int, batch_idx: int, time_idx: int) -> int:
        return int(self.seed + int(epoch) * 1_000_003 + int(step) * 9_176 + batch_idx * 97 + time_idx * 13)

    def _sample_target_indices(
        self,
        count: int,
        *,
        num_tokens: int,
        grid_size: tuple[int, int],
        gps: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if self.mode == "random":
            return torch.randperm(num_tokens, generator=generator)[:count]
        weights = self._gps_angle_weights(num_tokens, grid_size=grid_size, gps=gps)
        return torch.multinomial(weights, num_samples=count, replacement=False, generator=generator)

    def _gps_angle_weights(self, num_tokens: int, *, grid_size: tuple[int, int], gps: torch.Tensor) -> torch.Tensor:
        rows, cols = grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, rows),
            torch.linspace(-1.0, 1.0, cols),
            indexing="ij",
        )
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:num_tokens]
        angle_index = min(max(self.angle_feature_index, 0), max(int(gps.numel()) - 1, 0))
        angle = float(gps.reshape(-1)[angle_index].item()) if gps.numel() else 0.0
        direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
        weights = torch.exp(self.angle_concentration * (coords @ direction))
        return weights.clamp_min(1e-6)


class GpsConditioner(nn.Module):
    def __init__(
        self,
        *,
        conditioning_type: str = "film",
        gps_input_size: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.conditioning_type = str(conditioning_type).strip().lower()
        self.gps_input_size = int(gps_input_size)
        self.latent_dim = int(latent_dim)
        hidden = int(hidden_dim)
        if self.conditioning_type == "film":
            self.net = nn.Sequential(nn.Linear(self.gps_input_size, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.latent_dim))
        elif self.conditioning_type == "concat_mlp":
            self.net = nn.Sequential(
                nn.Linear(self.gps_input_size + self.latent_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.latent_dim),
            )
        else:
            raise ValueError("JEPA GPS conditioner type must be film or concat_mlp.")

    def forward(self, context_latent: torch.Tensor, gps_batch: torch.Tensor) -> torch.Tensor:
        if gps_batch.shape[-1] != self.gps_input_size:
            raise ValueError(
                f"GPS-conditioned JEPA expected GPS feature dim {self.gps_input_size}, got {gps_batch.shape[-1]}."
            )
        gps = gps_batch.to(device=context_latent.device, dtype=context_latent.dtype)
        if self.conditioning_type == "film":
            gamma_beta = self.net(gps).unsqueeze(2)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            return context_latent * (1.0 + gamma) + beta
        expanded = gps.unsqueeze(2).expand(*context_latent.shape[:-1], gps.shape[-1])
        return self.net(torch.cat([context_latent, expanded], dim=-1))


class TargetLatentPredictor(nn.Module):
    def __init__(self, *, latent_dim: int = 64, hidden_dim: int = 128, max_tokens: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.target_pos_embed = nn.Embedding(int(max_tokens), int(latent_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, context_latent: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        summary = context_latent.mean(dim=2, keepdim=True)
        position = self.target_pos_embed(target_indices.clamp_min(0).clamp_max(self.target_pos_embed.num_embeddings - 1))
        return self.net(summary + position)


@ENCODERS.register("jepa_context_image")
@MODELS.register("jepa_context_image")
class JepaContextImageEncoder(nn.Module):
    expected_image_profile = "rgb_imagenet"
    input_channels = 3

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        checkpoint: str | None = None,
        output_dim: int | None = None,
        latent_dim: int = 64,
        image_channels: int = 3,
        image_profile: str | None = "rgb_imagenet",
        visual_encoder: dict[str, Any] | None = None,
        freeze_encoder: bool = False,
        strict: bool = True,
        state_dict_prefix: str = "context_encoder",
        pooling: str = "mean",
        pooler: dict[str, Any] | str | None = None,
        adapter: dict[str, Any] | str | None = None,
        gps_query_pool: dict[str, Any] | None = None,
        temporal_fallback: dict[str, Any] | None = None,
        temporal_auxiliary: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim if output_dim is not None else self.latent_dim)
        self.checkpoint_path = str(checkpoint_path or checkpoint or "")
        self.freeze_encoder = bool(freeze_encoder)
        self.strict = bool(strict)
        self.state_dict_prefix = str(state_dict_prefix).strip().rstrip(".") or "context_encoder"
        self.pooler_config = normalize_jepa_downstream_pooler_config(
            pooler=pooler,
            pooling=pooling,
            gps_query_pool=gps_query_pool,
            latent_dim=self.latent_dim,
        )
        self.adapter_config = normalize_jepa_downstream_adapter_config(
            adapter=adapter,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
        )
        self.pooling = str(self.pooler_config.get("type", "mean")).strip().lower()
        self.temporal_fallback_config = _normalize_temporal_fallback_config(temporal_fallback)
        self.temporal_fallback_enabled = bool(self.temporal_fallback_config.get("enabled", False))
        self.temporal_auxiliary_config = _normalize_temporal_auxiliary_config(temporal_auxiliary)
        self.temporal_auxiliary_enabled = bool(self.temporal_auxiliary_config.get("enabled", False))
        self.supports_observability_metadata = self.temporal_fallback_enabled or self.temporal_auxiliary_enabled
        self.last_temporal_fallback_metadata: dict[str, Any] = {"enabled": self.temporal_fallback_enabled, "affected_count": 0}
        self.last_current_latent: torch.Tensor | None = None
        self.last_temporal_predicted_latent: torch.Tensor | None = None
        self.last_temporal_auxiliary_metadata: dict[str, Any] = {
            "enabled": self.temporal_auxiliary_enabled,
            "available": False,
            "insufficient_history_count": 0,
        }
        self.last_predictive_gps_query_diagnostics: dict[str, Any] | None = None
        if self.output_dim != self.latent_dim:
            raise ValueError(
                "jepa_context_image requires output_dim to equal latent_dim because it reuses the JEPA "
                f"context encoder projection directly; got output_dim={self.output_dim}, latent_dim={self.latent_dim}."
            )
        self.pooler = build_jepa_downstream_pooler(self.pooler_config)
        self.adapter = build_jepa_downstream_adapter(self.adapter_config)
        self.supports_observability_metadata = self.supports_observability_metadata or isinstance(
            self.pooler,
            PredictiveGPSQueryPool,
        )
        self.required_context_modalities = tuple(getattr(self.pooler, "required_context_modalities", ()))
        self.context_feature_source = str(getattr(self.pooler, "context_feature_source", "none"))
        raw_kwargs = getattr(self.pooler, "context_feature_kwargs", {})
        self.context_feature_kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        self.gps_query_pool_config: dict[str, Any] = {}
        object.__setattr__(
            self,
            "gps_query_pool",
            self.pooler if isinstance(self.pooler, GPSQueryPool) else None,
        )
        self.last_attention_map: torch.Tensor | None = None
        if self.pooling == "gps_query_attention":
            self.gps_query_pool_config = {
                "latent_dim": getattr(self.pooler, "latent_dim", self.latent_dim),
                "condition_dim": getattr(self.pooler, "condition_dim", self.latent_dim),
                "k_queries": getattr(self.pooler, "k_queries", None),
                "num_heads": getattr(self.pooler, "num_heads", None),
                "dropout": self.pooler_config.get("dropout", 0.0),
                "return_attention": getattr(self.pooler, "return_attention", False),
                "condition_source": getattr(self.pooler, "condition_source", "projected_gps"),
            }
        encoder_cfg = dict(visual_encoder or {})
        encoder_cfg.setdefault("image_channels", image_channels)
        encoder_cfg.setdefault("latent_dim", self.latent_dim)
        encoder_cfg.setdefault("image_profile", image_profile)
        self.visual_encoder_config = normalize_visual_token_encoder_config(
            encoder_cfg,
            image_channels=image_channels,
            latent_dim=self.latent_dim,
            image_profile=image_profile,
        )
        self.context_encoder = build_visual_token_encoder(self.visual_encoder_config)
        self.last_visual_token_metadata: dict[str, Any] = {}
        self.last_visual_token_diagnostics: dict[str, Any] = {}
        if self.checkpoint_path:
            _load_context_encoder_state(
                Path(self.checkpoint_path),
                self.context_encoder,
                prefix=self.state_dict_prefix,
                strict=self.strict,
            )
        if self.freeze_encoder:
            for param in self.context_encoder.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        image_batch: torch.Tensor,
        gps_condition_features: torch.Tensor | None = None,
        *,
        image_valid_mask: torch.Tensor | None = None,
        image_observability_score: torch.Tensor | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        gps_counterfactual_mask: torch.Tensor | None = None,
        benchmark_condition_metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        tokens, token_info = self.context_encoder(image_batch)
        token_metadata = visual_token_metadata_from_encoder(self.context_encoder, token_info, tokens)
        self.last_visual_token_metadata = token_metadata
        if self.required_context_modalities and gps_condition_features is None:
            if self.pooling == "gps_query_attention":
                raise ValueError("jepa_context_image GPS-query pooling requires GPS condition feature.")
            if self.pooling == "hybrid_residual_query":
                raise ValueError("jepa_context_image hybrid residual query pooling requires GPS condition feature.")
            if self.pooling == "predictive_gps_query":
                raise ValueError("jepa_context_image Predictive GPS-query++ pooling requires GPS condition feature.")
            raise ValueError(f"jepa_context_image pooler {self.pooling!r} requires condition features.")
        if self.pooling == "predictive_gps_query":
            result = self.pooler(
                tokens,
                condition_features=gps_condition_features,
                image_valid_mask=image_valid_mask,
                image_observability_score=image_observability_score,
                gps_valid_mask=gps_valid_mask,
                gps_counterfactual_mask=gps_counterfactual_mask,
                benchmark_condition_metadata=benchmark_condition_metadata,
                token_metadata=token_metadata,
            )
            self.last_predictive_gps_query_diagnostics = getattr(self.pooler, "last_diagnostics", None)
        else:
            result = self.pooler(tokens, condition_features=gps_condition_features, token_metadata=token_metadata)
            self.last_predictive_gps_query_diagnostics = None
        if isinstance(result, tuple):
            pooled, attention_map = result
            self.last_attention_map = attention_map
        else:
            self.last_attention_map = getattr(self.pooler, "last_attention_map", None)
            pooled = result
        features = self.adapter(pooled)
        self.last_visual_token_diagnostics = _visual_token_diagnostics(
            token_metadata=token_metadata,
            pooler=self.pooler,
            attention_map=self.last_attention_map,
        )
        if features.ndim != 3:
            if self.temporal_auxiliary_enabled or self.temporal_fallback_enabled:
                raise ValueError(
                    "jepa_context_image token output mode is incompatible with temporal auxiliary or fallback; "
                    "disable temporal options or use frame output mode."
                )
            self.last_temporal_auxiliary_metadata = {
                "enabled": False,
                "available": False,
                "insufficient_history_count": 0,
            }
            self.last_temporal_fallback_metadata = {"enabled": False, "affected_count": 0}
            return features
        self._update_temporal_auxiliary(features)
        if self.pooling == "predictive_gps_query":
            self._update_predictive_gps_query_auxiliary()
        return self._maybe_apply_temporal_fallback(
            features,
            image_valid_mask=image_valid_mask,
            image_observability_score=image_observability_score,
            benchmark_condition_metadata=benchmark_condition_metadata,
        )

    def _update_temporal_auxiliary(self, features: torch.Tensor) -> None:
        if not self.temporal_auxiliary_enabled:
            self.last_current_latent = None
            self.last_temporal_predicted_latent = None
            self.last_temporal_auxiliary_metadata = {
                "enabled": False,
                "available": False,
                "insufficient_history_count": 0,
            }
            return
        predicted, metadata = _compute_temporal_auxiliary_prediction(features, self.temporal_auxiliary_config)
        self.last_current_latent = features
        self.last_temporal_predicted_latent = predicted
        self.last_temporal_auxiliary_metadata = metadata

    def _update_predictive_gps_query_auxiliary(self) -> None:
        diagnostics = self.last_predictive_gps_query_diagnostics or {}
        current = getattr(self.pooler, "last_current_latent", None)
        predicted = getattr(self.pooler, "last_temporal_predicted_latent", None)
        self.last_current_latent = current if torch.is_tensor(current) else None
        self.last_temporal_predicted_latent = predicted if torch.is_tensor(predicted) else None
        self.last_temporal_auxiliary_metadata = {
            "enabled": True,
            "objective": "predictive_gps_query_temporal_latent",
            "available": bool(diagnostics.get("branch_availability", {}).get("temporal_predicted", False)),
            "history_window": diagnostics.get("pooler", {}).get("history_window", None)
            or diagnostics.get("history_window", None),
            "source_history_range": diagnostics.get("temporal_source_history_range", []),
            "availability_mask": diagnostics.get("temporal_availability_mask", []),
            "insufficient_history_count": int(diagnostics.get("insufficient_history_count", 0) or 0),
            "fallback_strategy": diagnostics.get("fallback_strategy", "zero"),
        }

    def _maybe_apply_temporal_fallback(
        self,
        features: torch.Tensor,
        *,
        image_valid_mask: torch.Tensor | None,
        image_observability_score: torch.Tensor | None,
        benchmark_condition_metadata: dict[str, Any] | None,
    ) -> torch.Tensor:
        if not self.temporal_fallback_enabled:
            self.last_temporal_fallback_metadata = {"enabled": False, "affected_count": 0}
            return features
        output, metadata = _apply_temporal_context_fallback(
            features,
            image_valid_mask=image_valid_mask,
            image_observability_score=image_observability_score,
            benchmark_condition_metadata=benchmark_condition_metadata,
            config=self.temporal_fallback_config,
        )
        self.last_temporal_fallback_metadata = metadata
        return output

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        legacy_prefix = f"{prefix}gps_query_pool."
        pooler_prefix = f"{prefix}pooler."
        if self.pooling == "predictive_gps_query" and any(key.startswith(legacy_prefix) for key in state_dict):
            message = (
                "Cannot silently load legacy gps_query_attention checkpoint keys into predictive_gps_query; "
                "use strict=False only for an explicit non-strict transfer."
            )
            if strict:
                error_msgs.append(message)
                return
        if any(key.startswith(legacy_prefix) for key in state_dict) and not any(
            key.startswith(pooler_prefix) for key in state_dict
        ):
            for key, value in list(state_dict.items()):
                if key.startswith(legacy_prefix):
                    state_dict[f"{pooler_prefix}{key[len(legacy_prefix):]}"] = value
                    state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def training_strategy_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "encoder": "jepa_context_image",
            "checkpoint_path": self.checkpoint_path,
            "state_dict_prefix": self.state_dict_prefix,
            "freeze_encoder": self.freeze_encoder,
            "pooling": self.pooling,
            "pooler_type": self.pooling,
            "adapter_type": str(self.adapter_config.get("type", "identity")),
            "condition_source": self.gps_query_pool_config.get("condition_source"),
            "attention_diagnostics": bool(self.gps_query_pool_config.get("return_attention", False)),
            "latent_dim": self.latent_dim,
            "temporal_fallback": dict(self.temporal_fallback_config),
            "temporal_auxiliary": dict(self.temporal_auxiliary_config),
            "temporal_auxiliary_enabled": self.temporal_auxiliary_enabled,
        }
        visual_metadata = (
            self.context_encoder.visual_token_metadata()
            if hasattr(self.context_encoder, "visual_token_metadata")
            else dict(self.last_visual_token_metadata)
        )
        metadata["visual_token_encoder"] = dict(visual_metadata)
        metadata["visual_token_metadata"] = dict(visual_metadata)
        metadata["checkpoint_policy"] = visual_metadata.get("checkpoint_policy")
        metadata["token_source"] = visual_metadata.get("token_source")
        metadata["token_count"] = visual_metadata.get("token_count")
        metadata["token_grid"] = visual_metadata.get("token_grid")
        metadata["pooler_output_mode"] = getattr(self.pooler, "output_mode", "frame")
        pooler_metadata = (
            self.pooler.training_strategy_metadata()
            if hasattr(self.pooler, "training_strategy_metadata")
            else {"type": self.pooling}
        )
        adapter_metadata = (
            self.adapter.training_strategy_metadata()
            if hasattr(self.adapter, "training_strategy_metadata")
            else {"type": self.adapter_config.get("type", "identity")}
        )
        metadata["pooler"] = pooler_metadata
        metadata["adapter"] = adapter_metadata
        if self.pooling == "gps_query_attention":
            metadata["gps_query_pooling_enabled"] = True
            metadata["gps_query_pool"] = dict(self.gps_query_pool_config)
            metadata["required_context_modalities"] = list(self.required_context_modalities)
            metadata["context_feature_source"] = self.context_feature_source
        else:
            metadata["gps_query_pooling_enabled"] = False
        if self.pooling == "hybrid_residual_query":
            metadata["hybrid_residual_query_enabled"] = True
            metadata["required_context_modalities"] = list(self.required_context_modalities)
            metadata["context_feature_source"] = self.context_feature_source
        else:
            metadata["hybrid_residual_query_enabled"] = False
        if self.pooling == "predictive_gps_query":
            metadata["predictive_gps_query_enabled"] = True
            metadata["gps_query_plus_plus_enabled"] = True
            metadata["required_context_modalities"] = list(self.required_context_modalities)
            metadata["context_feature_source"] = self.context_feature_source
            metadata["condition_source"] = pooler_metadata.get("condition_source")
            metadata["content_query_count"] = pooler_metadata.get("content_queries")
            metadata["gps_query_count"] = pooler_metadata.get("gps_queries")
            metadata["temporal_predictor_type"] = pooler_metadata.get("temporal_predictor_type")
            metadata["reliability_gate_type"] = pooler_metadata.get("reliability_gate_type")
            metadata["residual_scale"] = pooler_metadata.get("residual_scale")
            metadata["auxiliary_losses"] = {
                "temporal_auxiliary_enabled": self.temporal_auxiliary_enabled,
                "temporal_auxiliary": dict(self.temporal_auxiliary_config),
            }
            metadata["jepa_checkpoint_path"] = self.checkpoint_path
            metadata["context_encoder_frozen"] = self.freeze_encoder
        else:
            metadata["predictive_gps_query_enabled"] = False
            metadata["gps_query_plus_plus_enabled"] = False
        return metadata


@MODELS.register("gps_conditioned_jepa")
class GPSConditionedJEPA(nn.Module):
    supports_modality_kwargs = True

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        image_channels: int = 3,
        gps_input_size: int = 3,
        image_profile: str | None = None,
        visual_encoder: dict[str, Any] | None = None,
        conditioning: dict[str, Any] | None = None,
        predictor: dict[str, Any] | None = None,
        mask_sampler: dict[str, Any] | None = None,
        ema_decay: float = 0.996,
        num_classes: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.gps_input_size = int(gps_input_size)
        self.ema_decay = float(ema_decay)
        self.num_classes = int(num_classes)
        encoder_cfg = dict(visual_encoder or {})
        encoder_cfg.setdefault("image_channels", image_channels)
        encoder_cfg.setdefault("latent_dim", self.latent_dim)
        encoder_cfg.setdefault("image_profile", image_profile)
        self.visual_encoder_config = normalize_visual_token_encoder_config(
            encoder_cfg,
            image_channels=image_channels,
            latent_dim=self.latent_dim,
            image_profile=image_profile,
        )
        self.context_encoder = build_visual_token_encoder(self.visual_encoder_config)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        conditioner_cfg = dict(conditioning or {})
        conditioner_cfg.setdefault("conditioning_type", conditioner_cfg.pop("type", "film"))
        conditioner_cfg.setdefault("gps_input_size", self.gps_input_size)
        conditioner_cfg.setdefault("latent_dim", self.latent_dim)
        self.gps_conditioner = GpsConditioner(**conditioner_cfg)
        predictor_cfg = dict(predictor or {})
        predictor_cfg.setdefault("latent_dim", self.latent_dim)
        predictor_cfg.setdefault("max_tokens", getattr(self.context_encoder, "max_tokens", encoder_cfg.get("max_tokens", 256)))
        self.predictor = TargetLatentPredictor(**predictor_cfg)
        sampler_cfg = dict(mask_sampler or {})
        self.mask_sampler = JepaMaskSampler(**sampler_cfg)

    def forward(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        jepa_epoch: int = 0,
        jepa_step: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        if image_batch is None:
            raise ValueError("gps_conditioned_jepa objective requires image input in 'image_batch'.")
        if gps_batch is None:
            raise ValueError("GPS-conditioned JEPA requires GPS-Rel-Polar input in 'gps_batch'.")
        if gps_batch.ndim != 3:
            raise ValueError(f"GPS-conditioned JEPA expects gps_batch shape [B, T, F], got {tuple(gps_batch.shape)}.")
        if gps_batch.shape[-1] != self.gps_input_size:
            raise ValueError(
                f"GPS-conditioned JEPA expected GPS feature dim {self.gps_input_size}, got {gps_batch.shape[-1]}."
            )
        context_tokens, token_info = self.context_encoder(image_batch)
        context_metadata = visual_token_metadata_from_encoder(self.context_encoder, token_info, context_tokens)
        grid_size = _metadata_token_grid(context_metadata, token_info)
        with torch.no_grad():
            target_tokens, _ = self.target_encoder(image_batch)
        batch_size, seq_len, num_tokens, _ = context_tokens.shape
        masks = self.mask_sampler.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            num_tokens=num_tokens,
            grid_size=grid_size,
            gps_batch=gps_batch,
            token_metadata=context_metadata,
            epoch=int(jepa_epoch),
            step=int(jepa_step),
            device=context_tokens.device,
        )
        context_latent = _gather_tokens(context_tokens, masks.context_indices)
        target_latent = _gather_tokens(target_tokens, masks.target_indices).detach()
        conditioned_context = self.gps_conditioner(context_latent, gps_batch)
        predicted = self.predictor(conditioned_context, masks.target_indices)
        logits = predicted.mean(dim=(2, 3), keepdim=False).unsqueeze(-1).expand(-1, -1, self.num_classes)
        diagnostics: dict[str, Any] = {
            "predicted_target_latent": predicted,
            "target_latent": target_latent,
            "context_mask": masks.context_mask,
            "target_mask": masks.target_mask,
            "loss_mask": masks.loss_mask,
            "ema_decay": float(self.ema_decay),
            "jepa/ema_decay": float(self.ema_decay),
            "jepa/latent_norm": float(predicted.detach().norm(dim=-1).mean().cpu().item()),
            "visual_token_metadata": context_metadata,
            "jepa/variant_id": str(context_metadata.get("variant_id", "")),
            **masks.diagnostics,
        }
        return {
            "logits": logits,
            "input_features": context_latent,
            "output_features": predicted,
            **diagnostics,
        }

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = float(self.ema_decay)
        for target_param, context_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_param.data.mul_(decay).add_(context_param.data, alpha=1.0 - decay)
        for target_buffer, context_buffer in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            target_buffer.copy_(context_buffer)


def _ratio(value: float, name: str) -> float:
    ratio = float(value)
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"JEPA mask {name} must be in (0, 1), got {value}.")
    return ratio


def _indices_to_mask(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    mask = torch.zeros(*indices.shape[:-1], int(num_tokens), dtype=torch.bool, device=indices.device)
    return mask.scatter(-1, indices, True)


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).expand(*indices.shape, tokens.shape[-1])
    return torch.gather(tokens, dim=2, index=expanded)


__all__ = [
    "GPSConditionedJEPA",
    "GPSQueryPool",
    "GpsConditioner",
    "JepaContextImageEncoder",
    "JepaMaskSample",
    "JepaMaskSampler",
    "LightweightPatchViTFrameEncoder",
    "OverlapPatchTokenEncoder",
    "TargetLatentPredictor",
    "TinyViTFrameTokenEncoder",
    "VisualTokenMetadata",
    "VisualPatchTokenEncoder",
    "build_visual_token_encoder",
    "normalize_visual_token_encoder_config",
]
