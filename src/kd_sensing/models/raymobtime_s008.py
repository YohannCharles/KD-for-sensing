from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import (
    image_profile_spec,
    normalize_modalities,
    resolve_image_profile,
    validate_image_encoder_profile,
)
from kd_sensing.models.image_encoders import ResNet18ImageEncoder  # noqa: F401 - ensure encoder registration
from kd_sensing.registries import ENCODERS, MODELS


def _resolve_dim(value: int | None, *fallbacks: int | None, default: int = 64) -> int:
    for candidate in (value, *fallbacks):
        if candidate is not None:
            resolved = int(candidate)
            if resolved <= 0:
                raise ValueError(f"dimension must be positive, got {resolved}.")
            return resolved
    return int(default)


class SnapshotVectorMLPEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_dim: int = 64,
        hidden_size: int = 64,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_size),
            nn.Linear(self.input_size, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), self.output_dim),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _require_snapshot_time(batch, "Raymobtime s008 vector encoder")
        if int(batch.shape[-1]) != self.input_size:
            raise ValueError(f"Expected vector feature size {self.input_size}, got {tuple(batch.shape)}.")
        return self.net(batch.to(dtype=torch.float32))


@ENCODERS.register("coord_mlp")
class CoordMLPEncoder(SnapshotVectorMLPEncoder):
    def __init__(self, coord_input_size: int = 3, output_dim: int = 64, **kwargs: Any) -> None:
        super().__init__(input_size=coord_input_size, output_dim=output_dim, **kwargs)


@ENCODERS.register("ray_mlp")
class RayMLPEncoder(SnapshotVectorMLPEncoder):
    def __init__(self, ray_input_size: int = 14, output_dim: int = 64, **kwargs: Any) -> None:
        super().__init__(input_size=ray_input_size, output_dim=output_dim, **kwargs)


@ENCODERS.register("raymobtime_lidar_3d_cnn")
class RaymobtimeLidar3DCNNEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        lidar_channels: int = 1,
        stem_channels: int = 16,
        block_channels: list[int] | tuple[int, ...] = (16, 32),
        hidden_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        self.lidar_channels = int(lidar_channels)
        stem = int(stem_channels)
        if self.lidar_channels <= 0 or stem <= 0:
            raise ValueError("Raymobtime LiDAR 3D CNN channel counts must be positive.")
        channels = [int(value) for value in block_channels]
        if not channels or any(value <= 0 for value in channels):
            raise ValueError("block_channels must contain positive channel counts.")
        self.conv_stem = nn.Sequential(
            nn.Conv3d(self.lidar_channels, stem, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(stem),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        in_channels = stem
        for out_channels in channels:
            blocks.append(_Residual3DBlock(in_channels, out_channels))
            in_channels = out_channels
        self.residual_blocks = nn.Sequential(*blocks)
        self.channel_attention = _ChannelAttention3D(in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.projection_head = nn.Sequential(
            nn.LayerNorm(in_channels * 2),
            nn.Linear(in_channels * 2, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), self.output_dim),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _require_snapshot_time(batch, "Raymobtime s008 LiDAR 3D CNN encoder")
        if batch.ndim == 5:
            batch = batch.unsqueeze(2)
        if batch.ndim != 6:
            raise ValueError(
                "Raymobtime s008 LiDAR input must have shape [B, 1, C, D, H, W], "
                f"got {tuple(batch.shape)}."
            )
        if int(batch.shape[2]) != self.lidar_channels:
            raise ValueError(
                f"Raymobtime s008 LiDAR 3D CNN expected {self.lidar_channels} occupancy channel(s), "
                f"got {tuple(batch.shape)}."
            )
        batch_size = int(batch.shape[0])
        grid = batch[:, 0, ...].to(dtype=torch.float32)
        features = self.conv_stem(grid)
        features = self.residual_blocks(features)
        features = features * self.channel_attention(features)
        pooled = torch.cat(
            [
                self.global_avg_pool(features).flatten(1),
                self.global_max_pool(features).flatten(1),
            ],
            dim=1,
        )
        embedding = self.projection_head(pooled)
        return embedding.reshape(batch_size, 1, self.output_dim)


class _Residual3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut: nn.Module
        if int(in_channels) == int(out_channels):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.shortcut(x))


class _ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(int(channels) // int(reduction), 1)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(int(channels), hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, int(channels), kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RaymobtimeSelectionBase(nn.Module):
    supports_modality_kwargs = True

    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...],
        feature_size: int = 64,
        num_classes: int = 256,
        coord_input_size: int = 3,
        ray_input_size: int = 14,
        image_profile: str | None = None,
        image_channels: int | None = None,
        lidar_channels: int = 1,
        hidden_size: int = 128,
        dropout: float = 0.1,
        encoders: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modalities = normalize_modalities(tuple(modalities), context="Raymobtime selection modalities")
        if not self.modalities:
            raise ValueError("Raymobtime selection model requires at least one modality.")
        unsupported = [name for name in self.modalities if name not in {"coord", "image", "lidar", "ray"}]
        if unsupported:
            raise ValueError(f"Raymobtime s008 models support coord/image/lidar/ray only, got {unsupported}.")
        self.feature_size = int(feature_size)
        self.num_classes = int(num_classes)
        self.image_profile = resolve_image_profile(image_profile)
        self.image_channels = int(image_channels or image_profile_spec(self.image_profile).channels)
        self.encoders = nn.ModuleDict()
        encoder_cfgs = dict(encoders or {})
        for modality in self.modalities:
            encoder_cfg = self._encoder_config(
                modality,
                encoder_cfgs.get(modality),
                coord_input_size=coord_input_size,
                ray_input_size=ray_input_size,
                lidar_channels=lidar_channels,
                hidden_size=hidden_size,
                dropout=dropout,
            )
            self._validate_encoder_config(modality, encoder_cfg)
            self.encoders[modality] = ENCODERS.build(encoder_cfg)

    def _encode_modalities(
        self,
        *,
        coord_batch: torch.Tensor | None = None,
        image_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        ray_batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "coord": coord_batch,
            "image": image_batch,
            "lidar": lidar_batch,
            "ray": ray_batch,
        }
        encoded: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            tensor = inputs[modality]
            if tensor is None:
                raise ValueError(f"Raymobtime s008 model requires '{modality}' input because it is enabled.")
            encoded[modality] = self.encoders[modality](tensor)
        _validate_shared_shape(encoded)
        return encoded

    def _encoder_config(
        self,
        modality: str,
        raw_cfg: Any,
        *,
        coord_input_size: int,
        ray_input_size: int,
        lidar_channels: int,
        hidden_size: int,
        dropout: float,
    ) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {"type": _default_raymobtime_encoder_type(modality)}
        if isinstance(raw_cfg, str):
            raw_cfg = {"type": raw_cfg}
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Encoder config for Raymobtime modality '{modality}' must be a dict or string.")
        cfg = dict(raw_cfg)
        cfg.setdefault("output_dim", self.feature_size)
        if modality == "coord":
            cfg.setdefault("coord_input_size", coord_input_size)
            cfg.setdefault("hidden_size", hidden_size)
            cfg.setdefault("dropout", dropout)
        elif modality == "ray":
            cfg.setdefault("ray_input_size", ray_input_size)
            cfg.setdefault("hidden_size", hidden_size)
            cfg.setdefault("dropout", dropout)
        elif modality == "image":
            cfg.setdefault("image_profile", self.image_profile)
            cfg.setdefault("image_channels", self.image_channels)
        elif modality == "lidar":
            cfg.setdefault("lidar_channels", lidar_channels)
            cfg.setdefault("hidden_size", hidden_size)
            cfg.setdefault("dropout", dropout)
        return cfg

    def _validate_encoder_config(self, modality: str, encoder_cfg: dict[str, Any]) -> None:
        encoder_type = str(encoder_cfg.get("type"))
        if modality == "image":
            if encoder_type != "resnet18_imagenet_rgb":
                raise ValueError(
                    "Raymobtime s008 image modality must use the shared image encoder "
                    "'resnet18_imagenet_rgb'."
                )
            validate_image_encoder_profile(
                encoder_name=encoder_type,
                image_profile=self.image_profile,
                expected_channels=3,
                actual_channels=encoder_cfg.get("image_channels", self.image_channels),
            )
        if modality == "lidar" and encoder_type != "raymobtime_lidar_3d_cnn":
            raise ValueError(
                "Raymobtime s008 LiDAR modality must use 'raymobtime_lidar_3d_cnn' for 3D occupancy grids."
            )


@MODELS.register("simple_concat_multitask_selection")
class SimpleConcatMultiTaskSelection(RaymobtimeSelectionBase):
    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...],
        feature_size: int = 64,
        num_classes: int = 256,
        hidden_size: int = 128,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            modalities=modalities,
            feature_size=feature_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            **kwargs,
        )
        fused_dim = int(feature_size) * len(self.modalities)
        self.projection = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(feature_size)),
            nn.GELU(),
        )
        self.beam_head = nn.Linear(int(feature_size), int(num_classes))
        self.los_head = nn.Linear(int(feature_size), 1)
        self.link_head = nn.Linear(int(feature_size), 1)

    def forward(
        self,
        coord_batch: torch.Tensor | None = None,
        image_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        ray_batch: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        encoded = self._encode_modalities(
            coord_batch=coord_batch,
            image_batch=image_batch,
            lidar_batch=lidar_batch,
            ray_batch=ray_batch,
        )
        ordered = [encoded[modality] for modality in self.modalities]
        concat = torch.cat(ordered, dim=-1)
        fused = self.projection(concat)
        return _selection_output(
            logits=self.beam_head(fused),
            fused=fused,
            input_features=concat,
            encoded=encoded,
            modalities=self.modalities,
            los_logits=self.los_head(fused).squeeze(-1),
            link_quality=self.link_head(fused).squeeze(-1),
        )


@MODELS.register("task_aware_gated_multitask_selection")
class TaskAwareGatedMultiTaskSelection(RaymobtimeSelectionBase):
    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...],
        feature_size: int = 64,
        num_classes: int = 256,
        hidden_size: int = 128,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            modalities=modalities,
            feature_size=feature_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=dropout,
            **kwargs,
        )
        tasks = ("beam_selection", "los", "link_quality")
        self.gates = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.LayerNorm(int(feature_size)),
                    nn.Linear(int(feature_size), int(hidden_size)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_size), 1),
                )
                for task in tasks
            }
        )
        self.task_projections = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.LayerNorm(int(feature_size)),
                    nn.Linear(int(feature_size), int(feature_size)),
                    nn.GELU(),
                )
                for task in tasks
            }
        )
        self.beam_head = nn.Linear(int(feature_size), int(num_classes))
        self.los_head = nn.Linear(int(feature_size), 1)
        self.link_head = nn.Linear(int(feature_size), 1)

    def forward(
        self,
        coord_batch: torch.Tensor | None = None,
        image_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        ray_batch: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        encoded = self._encode_modalities(
            coord_batch=coord_batch,
            image_batch=image_batch,
            lidar_batch=lidar_batch,
            ray_batch=ray_batch,
        )
        stacked = torch.stack([encoded[modality][:, 0, :] for modality in self.modalities], dim=1)
        fused_by_task = {}
        gates = {}
        for task, gate_net in self.gates.items():
            gate_logits = gate_net(stacked).squeeze(-1)
            gate = torch.softmax(gate_logits, dim=1)
            gates[task] = gate
            fused = torch.sum(stacked * gate.unsqueeze(-1), dim=1)
            fused_by_task[task] = self.task_projections[task](fused).unsqueeze(1)
        beam_features = fused_by_task["beam_selection"]
        los_features = fused_by_task["los"]
        link_features = fused_by_task["link_quality"]
        output = _selection_output(
            logits=self.beam_head(beam_features),
            fused=beam_features,
            input_features=stacked.reshape(stacked.shape[0], 1, -1),
            encoded=encoded,
            modalities=self.modalities,
            los_logits=self.los_head(los_features).squeeze(-1),
            link_quality=self.link_head(link_features).squeeze(-1),
        )
        output["gates"] = gates
        output["gate_modalities"] = list(self.modalities)
        output["gate_tasks"] = list(gates)
        output["task_gates"] = gates
        return output


def _selection_output(
    *,
    logits: torch.Tensor,
    fused: torch.Tensor,
    input_features: torch.Tensor,
    encoded: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
    los_logits: torch.Tensor,
    link_quality: torch.Tensor,
) -> dict[str, Any]:
    return {
        "logits": logits,
        "input_features": input_features,
        "output_features": fused,
        "modalities": modalities,
        "modality_features": encoded,
        "los_logits": los_logits,
        "link_quality": link_quality,
        "link_prediction": link_quality,
        "model_heads": ["beam_selection", "los", "link_quality"],
    }


def _require_snapshot_time(batch: torch.Tensor, context: str) -> None:
    if batch.ndim < 3:
        raise ValueError(f"{context} expects a batch with snapshot time dimension, got {tuple(batch.shape)}.")
    if int(batch.shape[1]) != 1:
        raise ValueError(
            "Raymobtime s008 snapshot model only accepts current snapshot inputs with time dimension 1; "
            f"got shape {tuple(batch.shape)}."
        )


def _validate_shared_shape(encoded: dict[str, torch.Tensor]) -> None:
    batch_size = None
    for modality, features in encoded.items():
        if features.ndim != 3:
            raise ValueError(f"{modality} encoder must return [B, 1, D], got {tuple(features.shape)}.")
        if int(features.shape[1]) != 1:
            raise ValueError(
                "Raymobtime s008 snapshot model only accepts current snapshot encoder outputs; "
                f"{modality} returned {tuple(features.shape)}."
            )
        if batch_size is None:
            batch_size = int(features.shape[0])
        elif int(features.shape[0]) != batch_size:
            raise ValueError("Raymobtime s008 enabled modalities must share batch size.")


def _default_raymobtime_encoder_type(modality: str) -> str:
    return {
        "coord": "coord_mlp",
        "image": "resnet18_imagenet_rgb",
        "lidar": "raymobtime_lidar_3d_cnn",
        "ray": "ray_mlp",
    }[modality]


__all__ = [
    "CoordMLPEncoder",
    "RayMLPEncoder",
    "RaymobtimeLidar3DCNNEncoder",
    "SimpleConcatMultiTaskSelection",
    "SnapshotVectorMLPEncoder",
    "TaskAwareGatedMultiTaskSelection",
]
