from typing import Any, Callable

import torch
import torch.nn as nn

from kd_sensing.modalities import validate_image_encoder_profile
from kd_sensing.registries import ENCODERS


def _resolve_output_dim(
    output_dim: int | None = None,
    feature_size: int | None = None,
    d_model: int | None = None,
) -> int:
    value = output_dim if output_dim is not None else feature_size if feature_size is not None else d_model
    value = 64 if value is None else int(value)
    if value <= 0:
        raise ValueError(f"output_dim must be positive, got {value}.")
    return value


@ENCODERS.register("resnet18_imagenet_rgb")
class ResNet18ImageEncoder(nn.Module):
    input_size = (224, 224)

    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        image_profile: str | None = "rgb_imagenet",
        image_channels: int = 3,
        **_: Any,
    ) -> None:
        super().__init__()
        validate_image_encoder_profile(
            encoder_name="resnet18_imagenet_rgb",
            image_profile=image_profile,
            expected_channels=3,
            actual_channels=image_channels,
        )
        self.output_dim = _resolve_output_dim(output_dim, feature_size, d_model)
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone, backbone_dim = _build_resnet18_backbone(pretrained=self.pretrained, weights=None)
        self.projection = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(backbone_dim, self.output_dim))
        _set_trainable(self.backbone, frozen=self.freeze_backbone)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.ndim != 5 or tuple(image_batch.shape[2:]) != (3, *self.input_size):
            raise ValueError(f"ResNet-18 image input must be [B,T,3,224,224], got {tuple(image_batch.shape)}.")
        batch_size, seq_len = image_batch.shape[:2]
        features = self.backbone(image_batch.reshape(batch_size * seq_len, *image_batch.shape[2:]))
        return self.projection(features).view(batch_size, seq_len, self.output_dim)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {"encoder": "resnet18_imagenet_rgb", "pretrained": self.pretrained, "freeze_backbone": self.freeze_backbone}


class _SpatialResNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        builder: Callable[..., tuple[nn.Module, int]],
        output_dim: int | None = None,
        feature_size: int | None = None,
        d_model: int | None = None,
        dropout: float = 0.0,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        in_channels: int | None = None,
        image_channels: int | None = None,
        radar_channels: int | None = None,
        lidar_channels: int | None = None,
        image_size: list[int] | tuple[int, int] | None = None,
        token_pool_size: list[int] | tuple[int, int] | int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.name = name
        self.output_dim = _resolve_output_dim(output_dim, feature_size, d_model)
        self.in_channels = int(in_channels or image_channels or radar_channels or lidar_channels or 3)
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.input_size = tuple(int(value) for value in image_size) if image_size is not None else None
        self.token_pool_size = _token_pool_size(token_pool_size)
        self.backbone, backbone_dim = builder(pretrained=self.pretrained, weights=None)
        _adapt_input_channels(self.backbone, self.in_channels)
        self.projection = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(backbone_dim, self.output_dim),
        )
        _set_trainable(self.backbone, frozen=self.freeze_backbone)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.ndim != 5:
            raise ValueError(f"{self.name} input must be [B,T,C,H,W], got {tuple(image_batch.shape)}.")
        batch_size, seq_len, channels, height, width = image_batch.shape
        if int(channels) != self.in_channels:
            raise ValueError(f"{self.name} expected {self.in_channels} input channels, got {int(channels)}.")
        if self.input_size is not None and (int(height), int(width)) != self.input_size:
            raise ValueError(f"{self.name} expected spatial size {self.input_size}, got {(int(height), int(width))}.")
        feature_map = _resnet_feature_map(
            self.backbone,
            image_batch.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32),
        )
        if self.token_pool_size is not None:
            feature_map = nn.functional.adaptive_avg_pool2d(feature_map, self.token_pool_size)
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        projected = self.projection(tokens)
        return projected.view(batch_size, seq_len, int(projected.shape[1]), self.output_dim)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "output_mode": "spatial_tokens",
            "token_pool_size": list(self.token_pool_size) if self.token_pool_size is not None else None,
        }


def _resnet18_spatial_tokens(**kwargs: Any) -> _SpatialResNetEncoder:
    return _SpatialResNetEncoder(name="resnet18_spatial_tokens", builder=_build_resnet18_backbone, **kwargs)


def _resnet34_spatial_tokens(**kwargs: Any) -> _SpatialResNetEncoder:
    return _SpatialResNetEncoder(name="resnet34_spatial_tokens", builder=_build_resnet34_backbone, **kwargs)


ENCODERS.register("resnet18_spatial_tokens")(_resnet18_spatial_tokens)
ENCODERS.register("resnet34_spatial_tokens")(_resnet34_spatial_tokens)


def _build_resnet18_backbone(*, pretrained: bool, weights: str | None) -> tuple[nn.Module, int]:
    del weights
    if pretrained:
        raise ValueError("Current T2/baseline recipes use scratch ResNet encoders only.")
    try:
        import torchvision.models as models
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("torchvision is required for the retained ResNet encoders.") from exc
    backbone = models.resnet18(weights=None)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _build_resnet34_backbone(*, pretrained: bool, weights: str | None) -> tuple[nn.Module, int]:
    del weights
    if pretrained:
        raise ValueError("Current T2/baseline recipes use scratch ResNet encoders only.")
    try:
        import torchvision.models as models
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("torchvision is required for the retained ResNet encoders.") from exc
    backbone = models.resnet34(weights=None)
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _set_trainable(backbone: nn.Module, *, frozen: bool) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = not frozen


def _resnet_feature_map(backbone: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    value = backbone.conv1(frames)
    value = backbone.bn1(value)
    value = backbone.relu(value)
    value = backbone.maxpool(value)
    value = backbone.layer1(value)
    value = backbone.layer2(value)
    value = backbone.layer3(value)
    return backbone.layer4(value)


def _adapt_input_channels(backbone: nn.Module, in_channels: int) -> None:
    if int(backbone.conv1.in_channels) == int(in_channels):
        return
    old = backbone.conv1
    replacement = nn.Conv2d(
        int(in_channels),
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        replacement.weight.copy_(old.weight.mean(dim=1, keepdim=True).expand(-1, int(in_channels), -1, -1))
        if old.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(old.bias)
    backbone.conv1 = replacement


def _token_pool_size(value: list[int] | tuple[int, int] | int | None) -> tuple[int, int] | None:
    if value is None:
        return None
    result = (int(value), int(value)) if isinstance(value, int) else tuple(int(item) for item in value)
    if len(result) != 2 or min(result) <= 0:
        raise ValueError(f"token_pool_size must be a positive int or pair, got {value!r}.")
    return result


__all__ = ["ResNet18ImageEncoder"]
