"""TinyViT image encoders for RGB/ImageNet modular sequence baselines.

TinyViT architecture adapted from the Microsoft TinyViT implementation:
https://github.com/microsoft/Cream/tree/main/TinyViT

Copyright (c) 2022 Microsoft.
Licensed under the MIT License.
"""

import itertools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.modalities import validate_image_encoder_profile
from kd_sensing.models.image_encoders import _resolve_output_dim
from kd_sensing.registries import ENCODERS


TINYVIT_IMAGE_SIZE = (224, 224)
TINYVIT_5M = {
    "embed_dims": (64, 128, 160, 320),
    "depths": (2, 2, 6, 2),
    "num_heads": (2, 4, 5, 10),
    "window_sizes": (7, 7, 14, 7),
}


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:g}"


class Conv2dBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            ),
        )
        bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0.0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, resolution: int, activation: type[nn.Module]) -> None:
        super().__init__()
        img_size = _to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.seq = nn.Sequential(
            Conv2dBN(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            activation(),
            Conv2dBN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        expand_ratio: float,
        activation: type[nn.Module],
        drop_path: float,
    ) -> None:
        super().__init__()
        hidden_chans = int(in_chans * expand_ratio)
        self.conv1 = Conv2dBN(in_chans, hidden_chans, kernel_size=1)
        self.act1 = activation()
        self.conv2 = Conv2dBN(hidden_chans, hidden_chans, kernel_size=3, padding=1, groups=hidden_chans)
        self.act2 = activation()
        self.conv3 = Conv2dBN(hidden_chans, out_chans, kernel_size=1, bn_weight_init=0.0)
        self.act3 = activation()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x = x + shortcut
        return self.act3(x)


class PatchMerging(nn.Module):
    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        out_dim: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.input_resolution = tuple(int(value) for value in input_resolution)
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.act = activation()
        self.conv1 = Conv2dBN(dim, out_dim, kernel_size=1)
        self.conv2 = Conv2dBN(out_dim, out_dim, kernel_size=3, stride=2, padding=1, groups=out_dim)
        self.conv3 = Conv2dBN(out_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            height, width = self.input_resolution
            batch = int(x.shape[0])
            x = x.view(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class ConvLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        activation: type[nn.Module],
        drop_path: list[float] | float = 0.0,
        downsample: type[nn.Module] | None = None,
        out_dim: int | None = None,
        conv_expand_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else float(drop_path),
                )
                for i in range(int(depth))
            ]
        )
        self.downsample = (
            downsample(input_resolution, dim=dim, out_dim=int(out_dim or dim), activation=activation)
            if downsample is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = int(out_features or in_features)
        hidden_features = int(hidden_features or in_features)
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: float = 4.0,
        resolution: tuple[int, int] = (14, 14),
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.scale = key_dim**-0.5
        self.key_dim = int(key_dim)
        self.value_dim = int(attn_ratio * key_dim)
        self.value_heads_dim = self.value_dim * self.num_heads
        qkv_dim = self.value_heads_dim + self.key_dim * self.num_heads * 2
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, qkv_dim)
        self.proj = nn.Linear(self.value_heads_dim, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs: list[int] = []
        for point_a in points:
            for point_b in points:
                offset = (abs(point_a[0] - point_b[0]), abs(point_a[1] - point_b[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        token_count = len(points)
        self.attention_biases = nn.Parameter(torch.zeros(self.num_heads, len(attention_offsets)))
        self.register_buffer(
            "attention_bias_idxs",
            torch.LongTensor(idxs).view(token_count, token_count),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.view(batch, tokens, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.value_dim],
            dim=3,
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        bias = self.attention_biases[:, self.attention_bias_idxs]
        attn = ((q @ k.transpose(-2, -1)) * self.scale + bias).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, self.value_heads_dim)
        return self.proj(x)


class TinyViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.input_resolution = tuple(int(value) for value in input_resolution)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads}).")
        head_dim = self.dim // self.num_heads
        self.attn = Attention(self.dim, head_dim, self.num_heads, attn_ratio=1.0, resolution=(self.window_size, self.window_size))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(self.dim, hidden_features=int(self.dim * mlp_ratio), act_layer=activation, drop=drop)
        pad = int(local_conv_size) // 2
        self.local_conv = Conv2dBN(self.dim, self.dim, kernel_size=local_conv_size, padding=pad, groups=self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(f"TinyViT block expected {height * width} tokens, got {tokens}.")
        residual = x
        if height == self.window_size and width == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(batch, height, width, channels)
            pad_b = (self.window_size - height % self.window_size) % self.window_size
            pad_r = (self.window_size - width % self.window_size) % self.window_size
            if pad_b > 0 or pad_r > 0:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            padded_h, padded_w = height + pad_b, width + pad_r
            n_h = padded_h // self.window_size
            n_w = padded_w // self.window_size
            x = (
                x.view(batch, n_h, self.window_size, n_w, self.window_size, channels)
                .transpose(2, 3)
                .reshape(batch * n_h * n_w, self.window_size * self.window_size, channels)
            )
            x = self.attn(x)
            x = (
                x.view(batch, n_h, n_w, self.window_size, self.window_size, channels)
                .transpose(2, 3)
                .reshape(batch, padded_h, padded_w, channels)
            )
            if pad_b > 0 or pad_r > 0:
                x = x[:, :height, :width].contiguous()
            x = x.view(batch, tokens, channels)
        x = residual + self.drop_path(x)
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.local_conv(x)
        x = x.view(batch, channels, tokens).transpose(1, 2)
        return x + self.drop_path(self.mlp(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: list[float] | float = 0.0,
        downsample: type[nn.Module] | None = None,
        local_conv_size: int = 3,
        activation: type[nn.Module] = nn.GELU,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else float(drop_path),
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(int(depth))
            ]
        )
        self.downsample = (
            downsample(input_resolution, dim=dim, out_dim=int(out_dim or dim), activation=activation)
            if downsample is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dims: tuple[int, int, int, int] = (64, 128, 160, 320),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: tuple[int, int, int, int] = (2, 4, 5, 10),
        window_sizes: tuple[int, int, int, int] = (7, 7, 14, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.depths = tuple(int(value) for value in depths)
        self.num_layers = len(self.depths)
        self.mlp_ratio = float(mlp_ratio)
        activation = nn.GELU
        self.patch_embed = PatchEmbed(
            in_chans=int(in_chans),
            embed_dim=int(embed_dims[0]),
            resolution=int(img_size),
            activation=activation,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        dpr = [float(value.item()) for value in torch.linspace(0, float(drop_path_rate), sum(self.depths))]
        self.layers = nn.ModuleList()
        for layer_index in range(self.num_layers):
            input_resolution = (
                patches_resolution[0] // (2**layer_index),
                patches_resolution[1] // (2**layer_index),
            )
            kwargs = dict(
                dim=int(embed_dims[layer_index]),
                input_resolution=input_resolution,
                depth=self.depths[layer_index],
                drop_path=dpr[sum(self.depths[:layer_index]) : sum(self.depths[: layer_index + 1])],
                downsample=PatchMerging if layer_index < self.num_layers - 1 else None,
                out_dim=int(embed_dims[min(layer_index + 1, len(embed_dims) - 1)]),
                activation=activation,
            )
            if layer_index == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    num_heads=int(num_heads[layer_index]),
                    window_size=int(window_sizes[layer_index]),
                    mlp_ratio=self.mlp_ratio,
                    drop=float(drop_rate),
                    local_conv_size=int(local_conv_size),
                    **kwargs,
                )
            self.layers.append(layer)
        self.norm_head = nn.LayerNorm(int(embed_dims[-1]))
        self.head = nn.Linear(int(embed_dims[-1]), self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_head(self.forward_features(x))
        return self.head(x)


def _build_tinyvit_backbone(variant: str, *, in_chans: int = 3) -> tuple[TinyViT, int]:
    if str(variant).strip().lower() not in {"5m", "tinyvit_5m", "tinyvit-5m"}:
        raise ValueError("The retained U0 route uses the TinyViT-5M scratch encoder only.")
    model = TinyViT(
        img_size=TINYVIT_IMAGE_SIZE[0],
        in_chans=int(in_chans),
        embed_dims=TINYVIT_5M["embed_dims"],
        depths=TINYVIT_5M["depths"],
        num_heads=TINYVIT_5M["num_heads"],
        window_sizes=TINYVIT_5M["window_sizes"],
    )
    return model, 320


class TinyViTImageEncoder(nn.Module):
    expected_image_profile = "rgb_imagenet"
    input_channels = 3
    input_size = TINYVIT_IMAGE_SIZE

    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        pretrained: bool = False,
        freeze_backbone: bool = True,
        dropout: float = 0.0,
        image_profile: str | None = "rgb_imagenet",
        image_channels: int = 3,
        registry_name: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("The retained U0 route supports the scratch TinyViT encoder only.")
        self.variant = "5m"
        self.registry_name = registry_name or "tinyvit_5m_scratch_rgb"
        validate_image_encoder_profile(
            encoder_name=self.registry_name,
            image_profile=image_profile,
            expected_channels=3,
            actual_channels=image_channels,
        )
        self.output_dim = _resolve_output_dim(output_dim, feature_size, d_model)
        self.image_profile = "rgb_imagenet"
        self.image_channels = int(image_channels)
        self.pretrained = False
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone, self.backbone_dim = _build_tinyvit_backbone("5m", in_chans=3)
        self.projection = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.backbone_dim, self.output_dim),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not self.freeze_backbone

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.ndim != 5:
            raise ValueError(f"TinyViT image input must have shape [B, T, 3, 224, 224], got {tuple(image_batch.shape)}.")
        batch_size, seq_len, channels, height, width = image_batch.shape
        if int(channels) != 3 or (int(height), int(width)) != self.input_size:
            raise ValueError(
                "TinyViT ImageNet encoder requires [B, T, 3, 224, 224] input, "
                f"got {tuple(image_batch.shape)}."
            )
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32)
        features = self.backbone.norm_head(self.backbone.forward_features(frames))
        projected = self.projection(features)
        return projected.view(batch_size, seq_len, self.output_dim)

    def training_strategy_metadata(self) -> dict[str, Any]:
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in self.parameters())
        return {
            "variant": self.variant,
            "pretrained": False,
            "freeze_backbone": self.freeze_backbone,
            "freeze_policy": "frozen_backbone" if self.freeze_backbone else "full_finetune",
            "backbone_dim": self.backbone_dim,
            "output_dim": self.output_dim,
            "consumes_reliability_metadata": False,
            "reliability_metadata": {"consumed": False, "fields": []},
            "trainable_parameter_count": int(trainable_params),
            "total_parameter_count": int(total_params),
        }

def _tinyvit_5m_scratch_rgb(**kwargs: Any) -> TinyViTImageEncoder:
    kwargs.pop("variant", None)
    return TinyViTImageEncoder(registry_name="tinyvit_5m_scratch_rgb", **kwargs)


ENCODERS.register("tinyvit_5m_scratch_rgb")(_tinyvit_5m_scratch_rgb)
