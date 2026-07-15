from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import validate_image_encoder_profile
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder
from kd_sensing.registries import ENCODERS, MODELS
from kd_sensing.utils.checkpoint import load_torch_payload


RESNET18_STAGES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
RESNET34_STAGES = RESNET18_STAGES


def _resolve_output_dim(
    output_dim: int | None = None,
    feature_size: int | None = None,
    d_model: int | None = None,
) -> int:
    value = output_dim if output_dim is not None else feature_size if feature_size is not None else d_model
    if value is None:
        value = 64
    value = int(value)
    if value <= 0:
        raise ValueError(f"output_dim must be positive, got {value}.")
    return value


@ENCODERS.register("resnet18_imagenet_rgb")
@MODELS.register("resnet18_imagenet_rgb")
class ResNet18ImageEncoder(nn.Module):
    expected_image_profile = "rgb_imagenet"
    input_channels = 3
    input_size = (224, 224)

    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        pretrained: bool = True,
        weights: str | None = "DEFAULT",
        freeze_backbone: bool = True,
        unfreeze_stages: list[str] | tuple[str, ...] | None = None,
        unfreeze_last_n_stages: int = 0,
        image_profile: str | None = "rgb_imagenet",
        image_channels: int = 3,
        **_: Any,
    ):
        super().__init__()
        validate_image_encoder_profile(
            encoder_name="resnet18_imagenet_rgb",
            image_profile=image_profile,
            expected_channels=3,
            actual_channels=image_channels,
        )
        self.output_dim = _resolve_output_dim(output_dim, feature_size, d_model)
        self.image_profile = "rgb_imagenet"
        self.image_channels = int(image_channels)
        self.pretrained = bool(pretrained)
        self.weights = weights
        self.freeze_backbone = bool(freeze_backbone)
        self.requested_unfreeze_stages = tuple(str(stage) for stage in (unfreeze_stages or ()))
        self.unfreeze_last_n_stages = int(unfreeze_last_n_stages)

        self.backbone, backbone_dim = _build_resnet18_backbone(pretrained=self.pretrained, weights=weights)
        self.projection = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(backbone_dim, self.output_dim),
        )
        self.trainable_stages = self._configure_trainable_backbone()

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.ndim != 5:
            raise ValueError(f"ResNet-18 image input must have shape [B, T, 3, 224, 224], got {tuple(image_batch.shape)}.")
        batch_size, seq_len, channels, height, width = image_batch.shape
        if int(channels) != 3 or (int(height), int(width)) != self.input_size:
            raise ValueError(
                "ResNet-18 ImageNet encoder requires [B, T, 3, 224, 224] input, "
                f"got {tuple(image_batch.shape)}."
            )
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width)
        features = self.backbone(frames)
        projected = self.projection(features)
        return projected.view(batch_size, seq_len, self.output_dim)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "pretrained": self.pretrained,
            "weights": self.weights,
            "freeze_backbone": self.freeze_backbone,
            "trainable_stages": list(self.trainable_stages),
            "unfreeze_last_n_stages": self.unfreeze_last_n_stages,
        }

    def _configure_trainable_backbone(self) -> tuple[str, ...]:
        if not self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return RESNET18_STAGES

        for param in self.backbone.parameters():
            param.requires_grad = False
        requested = set(self.requested_unfreeze_stages)
        if self.unfreeze_last_n_stages > 0:
            requested.update(RESNET18_STAGES[-self.unfreeze_last_n_stages :])
        invalid = sorted(requested - set(RESNET18_STAGES))
        if invalid:
            raise ValueError(f"Unknown ResNet-18 stages {invalid}. Available stages: {list(RESNET18_STAGES)}.")
        for stage in requested:
            module = getattr(self.backbone, stage)
            for param in module.parameters():
                param.requires_grad = True
        return tuple(stage for stage in RESNET18_STAGES if stage in requested)


@ENCODERS.register("resnet18_spatial_tokens")
class ResNet18SpatialTokenEncoder(nn.Module):
    input_size: tuple[int, int] | None = None

    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        pretrained: bool = True,
        weights: str | None = "DEFAULT",
        freeze_backbone: bool = True,
        unfreeze_stages: list[str] | tuple[str, ...] | None = None,
        unfreeze_last_n_stages: int = 0,
        in_channels: int | None = None,
        image_channels: int | None = None,
        radar_channels: int | None = None,
        lidar_channels: int | None = None,
        image_size: list[int] | tuple[int, int] | None = None,
        token_pool_size: list[int] | tuple[int, int] | int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.output_dim = _resolve_output_dim(output_dim, feature_size, d_model)
        self.in_channels = int(in_channels or image_channels or radar_channels or lidar_channels or 3)
        self.pretrained = bool(pretrained)
        self.weights = weights
        self.freeze_backbone = bool(freeze_backbone)
        self.requested_unfreeze_stages = tuple(str(stage) for stage in (unfreeze_stages or ()))
        self.unfreeze_last_n_stages = int(unfreeze_last_n_stages)
        self.input_size = tuple(int(value) for value in image_size) if image_size is not None else None
        self.token_pool_size = _normalize_token_pool_size(token_pool_size)

        self.backbone, backbone_dim = _build_resnet18_backbone(pretrained=self.pretrained, weights=weights)
        _adapt_resnet_input_channels(self.backbone, self.in_channels)
        self.projection = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(backbone_dim, self.output_dim),
        )
        self.trainable_stages = self._configure_trainable_backbone()

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.ndim != 5:
            raise ValueError(
                "ResNet-18 spatial token encoder input must have shape [B, T, C, H, W], "
                f"got {tuple(image_batch.shape)}."
            )
        batch_size, seq_len, channels, height, width = image_batch.shape
        if int(channels) != self.in_channels:
            raise ValueError(
                f"ResNet-18 spatial token encoder expected {self.in_channels} channels, got {int(channels)}."
            )
        if self.input_size is not None and (int(height), int(width)) != self.input_size:
            raise ValueError(
                "ResNet-18 spatial token encoder expected spatial size "
                f"{self.input_size[0]}x{self.input_size[1]}, got {int(height)}x{int(width)}."
            )
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32)
        feature_map = _resnet_feature_map(self.backbone, frames)
        if self.token_pool_size is not None:
            feature_map = nn.functional.adaptive_avg_pool2d(feature_map, self.token_pool_size)
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        projected = self.projection(tokens)
        return projected.view(batch_size, seq_len, int(projected.shape[1]), self.output_dim)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "encoder": "resnet18_spatial_tokens",
            "pretrained": self.pretrained,
            "weights": self.weights,
            "freeze_backbone": self.freeze_backbone,
            "trainable_stages": list(self.trainable_stages),
            "unfreeze_last_n_stages": self.unfreeze_last_n_stages,
            "output_mode": "spatial_tokens",
            "token_pool_size": list(self.token_pool_size) if self.token_pool_size is not None else None,
        }

    def _configure_trainable_backbone(self) -> tuple[str, ...]:
        if not self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return RESNET18_STAGES

        for param in self.backbone.parameters():
            param.requires_grad = False
        requested = set(self.requested_unfreeze_stages)
        if self.unfreeze_last_n_stages > 0:
            requested.update(RESNET18_STAGES[-self.unfreeze_last_n_stages :])
        invalid = sorted(requested - set(RESNET18_STAGES))
        if invalid:
            raise ValueError(f"Unknown ResNet-18 stages {invalid}. Available stages: {list(RESNET18_STAGES)}.")
        for stage in requested:
            module = getattr(self.backbone, stage)
            for param in module.parameters():
                param.requires_grad = True
        return tuple(stage for stage in RESNET18_STAGES if stage in requested)


@ENCODERS.register("resnet34_spatial_tokens")
class ResNet34SpatialTokenEncoder(nn.Module):
    input_size: tuple[int, int] | None = None

    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        pretrained: bool = True,
        weights: str | None = "DEFAULT",
        freeze_backbone: bool = True,
        unfreeze_stages: list[str] | tuple[str, ...] | None = None,
        unfreeze_last_n_stages: int = 0,
        in_channels: int | None = None,
        image_channels: int | None = None,
        radar_channels: int | None = None,
        lidar_channels: int | None = None,
        image_size: list[int] | tuple[int, int] | None = None,
        token_pool_size: list[int] | tuple[int, int] | int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.output_dim = _resolve_output_dim(output_dim, feature_size, d_model)
        self.in_channels = int(in_channels or image_channels or radar_channels or lidar_channels or 3)
        self.pretrained = bool(pretrained)
        self.weights = weights
        self.freeze_backbone = bool(freeze_backbone)
        self.requested_unfreeze_stages = tuple(str(stage) for stage in (unfreeze_stages or ()))
        self.unfreeze_last_n_stages = int(unfreeze_last_n_stages)
        self.input_size = tuple(int(value) for value in image_size) if image_size is not None else None
        self.token_pool_size = _normalize_token_pool_size(token_pool_size)

        self.backbone, backbone_dim = _build_resnet34_backbone(pretrained=self.pretrained, weights=weights)
        _adapt_resnet_input_channels(self.backbone, self.in_channels)
        self.projection = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(backbone_dim, self.output_dim),
        )
        self.trainable_stages = self._configure_trainable_backbone()

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.ndim != 5:
            raise ValueError(
                "ResNet-34 spatial token encoder input must have shape [B, T, C, H, W], "
                f"got {tuple(image_batch.shape)}."
            )
        batch_size, seq_len, channels, height, width = image_batch.shape
        if int(channels) != self.in_channels:
            raise ValueError(
                f"ResNet-34 spatial token encoder expected {self.in_channels} channels, got {int(channels)}."
            )
        if self.input_size is not None and (int(height), int(width)) != self.input_size:
            raise ValueError(
                "ResNet-34 spatial token encoder expected spatial size "
                f"{self.input_size[0]}x{self.input_size[1]}, got {int(height)}x{int(width)}."
            )
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32)
        feature_map = _resnet_feature_map(self.backbone, frames)
        if self.token_pool_size is not None:
            feature_map = nn.functional.adaptive_avg_pool2d(feature_map, self.token_pool_size)
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        projected = self.projection(tokens)
        return projected.view(batch_size, seq_len, int(projected.shape[1]), self.output_dim)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "encoder": "resnet34_spatial_tokens",
            "backbone": "resnet34",
            "pretrained": self.pretrained,
            "weights": self.weights,
            "freeze_backbone": self.freeze_backbone,
            "trainable_stages": list(self.trainable_stages),
            "unfreeze_last_n_stages": self.unfreeze_last_n_stages,
            "output_mode": "spatial_tokens",
            "token_pool_size": list(self.token_pool_size) if self.token_pool_size is not None else None,
        }

    def _configure_trainable_backbone(self) -> tuple[str, ...]:
        if not self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return RESNET34_STAGES

        for param in self.backbone.parameters():
            param.requires_grad = False
        requested = set(self.requested_unfreeze_stages)
        if self.unfreeze_last_n_stages > 0:
            requested.update(RESNET34_STAGES[-self.unfreeze_last_n_stages :])
        invalid = sorted(requested - set(RESNET34_STAGES))
        if invalid:
            raise ValueError(f"Unknown ResNet-34 stages {invalid}. Available stages: {list(RESNET34_STAGES)}.")
        for stage in requested:
            module = getattr(self.backbone, stage)
            for param in module.parameters():
                param.requires_grad = True
        return tuple(stage for stage in RESNET34_STAGES if stage in requested)


def _build_resnet18_backbone(*, pretrained: bool, weights: str | None) -> tuple[nn.Module, int]:
    try:
        import torchvision.models as tv_models
    except Exception as exc:  # pragma: no cover - environment-dependent.
        raise RuntimeError(
            "ResNet-18 ImageNet encoder requires torchvision in the kd_mm_beam environment. "
            "Install or repair torchvision before using resnet18_imagenet_rgb."
        ) from exc

    weights_obj = None
    if pretrained:
        try:
            enum = tv_models.ResNet18_Weights
        except AttributeError:
            model = tv_models.resnet18(pretrained=True)
            feature_dim = int(model.fc.in_features)
            model.fc = nn.Identity()
            return model, feature_dim
        if weights in (None, "", "none", "None"):
            weights_obj = None
        elif weights in ("DEFAULT", "default"):
            weights_obj = enum.DEFAULT
        else:
            try:
                weights_obj = getattr(enum, str(weights))
            except AttributeError as exc:
                available = [name for name in dir(enum) if name.isupper()]
                raise RuntimeError(
                    f"Unknown torchvision ResNet18 weights '{weights}'. Available weights: {available}."
                ) from exc
    model = tv_models.resnet18(weights=weights_obj)
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, feature_dim


def _build_resnet34_backbone(*, pretrained: bool, weights: str | None) -> tuple[nn.Module, int]:
    try:
        import torchvision.models as tv_models
    except Exception as exc:  # pragma: no cover - environment-dependent.
        raise RuntimeError(
            "ResNet-34 ImageNet encoder requires torchvision in the kd_mm_beam environment. "
            "Install or repair torchvision before using resnet34_spatial_tokens."
        ) from exc

    weights_obj = None
    if pretrained:
        try:
            enum = tv_models.ResNet34_Weights
        except AttributeError:
            model = tv_models.resnet34(pretrained=True)
            feature_dim = int(model.fc.in_features)
            model.fc = nn.Identity()
            return model, feature_dim
        if weights in (None, "", "none", "None"):
            weights_obj = None
        elif weights in ("DEFAULT", "default"):
            weights_obj = enum.DEFAULT
        else:
            try:
                weights_obj = getattr(enum, str(weights))
            except AttributeError as exc:
                available = [name for name in dir(enum) if name.isupper()]
                raise RuntimeError(
                    f"Unknown torchvision ResNet34 weights '{weights}'. Available weights: {available}."
                ) from exc
    model = tv_models.resnet34(weights=weights_obj)
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, feature_dim


def _resnet_feature_map(backbone: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    x = backbone.conv1(frames)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    return backbone.layer4(x)


def _resnet18_feature_map(backbone: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    return _resnet_feature_map(backbone, frames)


def _adapt_resnet_input_channels(backbone: nn.Module, in_channels: int) -> None:
    if int(in_channels) == int(backbone.conv1.in_channels):
        return
    old = backbone.conv1
    new = nn.Conv2d(
        int(in_channels),
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        base = old.weight.mean(dim=1, keepdim=True).expand(-1, int(in_channels), -1, -1)
        new.weight.copy_(base)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    backbone.conv1 = new


def _adapt_resnet18_input_channels(backbone: nn.Module, in_channels: int) -> None:
    _adapt_resnet_input_channels(backbone, in_channels)


def _normalize_token_pool_size(value: list[int] | tuple[int, int] | int | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        size = (int(value), int(value))
    else:
        size = tuple(int(item) for item in value)
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"token_pool_size must be a positive int or pair, got {value!r}.")
    return size


@ENCODERS.register("camera_ae_frozen")
@MODELS.register("camera_ae_frozen")
class CameraAEImageEncoder(nn.Module):
    """Frozen CameraAutoEncoder encoder for BeamBench-style image-AE features."""

    expected_image_profile = "rgb_imagenet"
    input_channels = 3

    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        latent_dim: int = 128,
        image_channels: int = 3,
        image_size: int = 64,
        checkpoint_path: str | None = None,
        checkpoint: str | None = None,
        require_checkpoint: bool = True,
        freeze_encoder: bool = True,
        dropout: float = 0.0,
        image_profile: str | None = "rgb_imagenet",
        **_: Any,
    ) -> None:
        super().__init__()
        validate_image_encoder_profile(
            encoder_name="camera_ae_frozen",
            image_profile=image_profile,
            expected_channels=int(image_channels),
            actual_channels=int(image_channels),
        )
        self.latent_dim = int(latent_dim)
        self.output_dim = _resolve_output_dim(output_dim, feature_size, d_model) if output_dim or feature_size or d_model else self.latent_dim
        self.image_size = int(image_size)
        self.checkpoint_path = str(checkpoint_path or checkpoint or "")
        self.require_checkpoint = bool(require_checkpoint)
        self.freeze_encoder = bool(freeze_encoder)
        self.autoencoder = CameraAutoEncoder(
            latent_dim=self.latent_dim,
            image_channels=int(image_channels),
            image_size=self.image_size,
        )
        if self.checkpoint_path:
            checkpoint_file = Path(self.checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    "Camera AE checkpoint is required for camera_ae_frozen but was not found at "
                    f"{checkpoint_file}. Provide a trained Camera AE checkpoint or set "
                    "require_checkpoint=false for mock/smoke runs."
                )
            payload = load_torch_payload(self.checkpoint_path, map_location="cpu")
            state_dict = payload.get("model_state_dict", payload)
            self.autoencoder.load_state_dict(state_dict)
        elif self.require_checkpoint:
            raise FileNotFoundError(
                "Camera AE checkpoint is required for camera_ae_frozen. Provide checkpoint_path "
                "for a trained Camera AE encoder, or set require_checkpoint=false for mock/smoke runs."
            )
        if self.freeze_encoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        self.projection = (
            nn.Identity()
            if self.output_dim == self.latent_dim and float(dropout) == 0.0
            else nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(self.latent_dim, self.output_dim))
        )

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.ndim != 5:
            raise ValueError(f"Camera AE image input must have shape [B, T, C, H, W], got {tuple(image_batch.shape)}.")
        batch_size, seq_len, channels, height, width = image_batch.shape
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32)
        if (int(height), int(width)) != (self.image_size, self.image_size):
            frames = nn.functional.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        with torch.set_grad_enabled(not self.freeze_encoder):
            latent = self.autoencoder.encode(frames)
        features = self.projection(latent)
        return features.view(batch_size, seq_len, self.output_dim)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "encoder": "camera_ae_frozen",
            "checkpoint_path": self.checkpoint_path,
            "require_checkpoint": self.require_checkpoint,
            "freeze_encoder": self.freeze_encoder,
            "latent_dim": self.latent_dim,
            "output_dim": self.output_dim,
            "image_size": self.image_size,
        }


__all__ = [
    "CameraAEImageEncoder",
    "RESNET18_STAGES",
    "RESNET34_STAGES",
    "ResNet18ImageEncoder",
    "ResNet18SpatialTokenEncoder",
    "ResNet34SpatialTokenEncoder",
]
