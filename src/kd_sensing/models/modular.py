from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import (
    MODALITY_ORDER,
    REMOVED_IMAGE_ENCODERS,
    image_profile_spec,
    normalize_modalities,
    resolve_image_profile,
    validate_image_encoder_profile,
)
from kd_sensing.models.auxiliary_heads import TemporalAuxiliaryHeads
from kd_sensing.models.csi import PilotDualViewCSIEncoder
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image_encoders import ResNet18ImageEncoder
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE, MmWaveFeatureExtractor
from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.registries import ENCODERS, HEADS, MODELS, PROJECTORS, REPRESENTATION_CORES


def _resolve_dim(value: int | None, *fallbacks: int | None, default: int = 64) -> int:
    for candidate in (value, *fallbacks):
        if candidate is not None:
            resolved = int(candidate)
            if resolved <= 0:
                raise ValueError(f"dimension must be positive, got {resolved}.")
            return resolved
    return int(default)


@ENCODERS.register("radar_cnn")
class RadarCNNEncoder(RadarFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        radar_channels: int = 2,
        in_channels: int | None = None,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(self.output_dim, in_channels=int(in_channels or radar_channels))


@ENCODERS.register("gps_mlp")
class GpsMLPEncoder(GpsFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        gps_input_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.1,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(self.output_dim, gps_input_size=gps_input_size, hidden_size=hidden_size, dropout=dropout)


@ENCODERS.register("lidar_cnn")
class LidarCNNEncoder(LidarFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        lidar_channels: int = 3,
        in_channels: int | None = None,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(self.output_dim, in_channels=int(in_channels or lidar_channels))


@ENCODERS.register("mmwave_mlp")
class MmWaveMLPEncoder(MmWaveFeatureExtractor):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        feature_size: int | None = None,
        d_model: int | None = None,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        csi_train_rms: float = 1.0,
        hidden_size: int = 128,
        dropout: float = 0.1,
        **_: Any,
    ):
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(
            feature_size=self.output_dim,
            mmwave_input_size=mmwave_input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )


@PROJECTORS.register("linear")
class LinearProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True, **_: Any):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(d_model)
        layers: list[nn.Module] = []
        if bool(layer_norm):
            layers.append(nn.LayerNorm(self.input_dim))
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(self.input_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Projector input must have shape [B, T, D], got {tuple(features.shape)}.")
        return self.net(features)


@PROJECTORS.register("identity")
class IdentityProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int | None = None, **_: Any):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim if d_model is None else d_model)
        if self.input_dim != self.output_dim:
            raise ValueError(
                f"identity projector requires input_dim == d_model, got {self.input_dim} and {self.output_dim}."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Projector input must have shape [B, T, D], got {tuple(features.shape)}.")
        return features


@REPRESENTATION_CORES.register("single_gru")
class SingleGRUCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.output_dim = int(hidden_size or d_model)
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.output_dim,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"single_gru core expects [B, T, D], got {tuple(features.shape)}.")
        output, _ = self.gru(features)
        return output


@REPRESENTATION_CORES.register("early_concat_gru")
class EarlyConcatGRUCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.output_dim = int(hidden_size or d_model)
        self.gru = nn.GRU(
            input_size=self.d_model * self.modality_count,
            hidden_size=self.output_dim,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"early_concat_gru core expects [B, K, T, D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count or int(d_model) != self.d_model:
            raise ValueError(
                "early_concat_gru core received incompatible features: "
                f"expected K={self.modality_count}, D={self.d_model}, got {tuple(features.shape)}."
            )
        concat = features.permute(0, 2, 1, 3).reshape(batch_size, seq_len, modality_count * d_model)
        output, _ = self.gru(concat)
        return output


@REPRESENTATION_CORES.register("snapshot_frame")
class SnapshotFrameCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int | None = None,
        hidden_size: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = None if modality_count is None else int(modality_count)
        self.output_dim = int(output_dim or hidden_size or d_model)
        if self.d_model <= 0 or self.output_dim <= 0:
            raise ValueError("snapshot_frame dimensions must be positive.")
        self.dropout = nn.Dropout(float(dropout))
        self.activation = str(activation)
        self.single_norm = nn.LayerNorm(self.d_model)
        self.single_projection = _snapshot_projection(
            self.d_model,
            self.output_dim,
            dropout=float(dropout),
            activation=activation,
        )
        self._multi_fusion: nn.Module | None = (
            self._build_multi_fusion(self.modality_count, dropout=float(dropout), activation=activation)
            if self.modality_count is not None and self.modality_count > 1
            else None
        )
        if self._multi_fusion is not None:
            self.add_module("multi_fusion", self._multi_fusion)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            batch_size, seq_len, d_model = features.shape
            self._validate_time_dim(seq_len, features.shape)
            if int(d_model) != self.d_model:
                raise ValueError(
                    f"snapshot_frame core expected D={self.d_model}, got {tuple(features.shape)}."
                )
            current = self.single_norm(features)
            return self.single_projection(current)
        if features.ndim == 4:
            batch_size, modality_count, seq_len, d_model = features.shape
            self._validate_time_dim(seq_len, features.shape)
            if int(d_model) != self.d_model:
                raise ValueError(
                    f"snapshot_frame core expected D={self.d_model}, got {tuple(features.shape)}."
                )
            if self.modality_count is not None and int(modality_count) != self.modality_count:
                raise ValueError(
                    "snapshot_frame core received incompatible modality count: "
                    f"expected K={self.modality_count}, got {tuple(features.shape)}."
                )
            fusion = self._multi_fusion_module(int(modality_count), features.device)
            current_tokens = features[:, :, 0, :].reshape(batch_size, modality_count * self.d_model)
            fused = fusion(current_tokens)
            return fused.unsqueeze(1)
        raise ValueError(f"snapshot_frame core expects [B, 1, D] or [B, K, 1, D], got {tuple(features.shape)}.")

    @staticmethod
    def _validate_time_dim(seq_len: int, shape: torch.Size) -> None:
        if int(seq_len) != 1:
            raise ValueError(
                "snapshot_frame baseline requires seq_len=1 and num_pred=1; "
                f"received time dimension T={int(seq_len)} from input shape {tuple(shape)}."
            )

    def _multi_fusion_module(self, modality_count: int, device: torch.device) -> nn.Module:
        input_dim = int(modality_count) * self.d_model
        if self._multi_fusion is None:
            self._multi_fusion = self._build_multi_fusion(
                modality_count,
                dropout=float(self.dropout.p),
                activation=self.activation,
            )
            self.add_module("multi_fusion", self._multi_fusion)
        first_linear = next(module for module in self._multi_fusion.modules() if isinstance(module, nn.Linear))
        if int(first_linear.in_features) != input_dim:
            raise ValueError(
                "snapshot_frame core was initialized for a different modality count; "
                f"expected fused input {first_linear.in_features}, got {input_dim}."
            )
        return self._multi_fusion.to(device)

    def _build_multi_fusion(self, modality_count: int, *, dropout: float, activation: str) -> nn.Module:
        input_dim = int(modality_count) * self.d_model
        hidden_dim = max(input_dim, self.output_dim)
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            _snapshot_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.output_dim),
        )


@REPRESENTATION_CORES.register("token_transformer")
class TokenTransformerCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        modality_count: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dropout=float(dropout),
            dim_feedforward=max(self.d_model * 4, 64),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_dim = self.d_model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"token_transformer core expects [B, K, T, D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, d_model = features.shape
        tokens = features.permute(0, 2, 1, 3).reshape(batch_size, seq_len * modality_count, d_model)
        memory = self.transformer(tokens)
        return memory.view(batch_size, seq_len, modality_count, d_model).mean(dim=2)


@HEADS.register("beam_head")
@HEADS.register("beam")
class BeamClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0, **_: Any):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.input_dim, self.num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"beam head expects [B, T, D], got {tuple(features.shape)}.")
        return self.net(features)


@MODELS.register("modular_sequence")
@MODELS.register("modular_sequence_model")
class ModularSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        encoders: dict[str, Any] | None = None,
        projectors: dict[str, Any] | None = None,
        representation_core: dict[str, Any] | None = None,
        heads: dict[str, Any] | None = None,
        feature_size: int = 64,
        d_model: int | None = None,
        num_classes: int = 64,
        num_pred: int = 3,
        image_profile: str | None = None,
        image_channels: int | None = None,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        coord_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        ray_input_size: int = 14,
        csi_train_rms: float = 1.0,
        auxiliary_heads: bool | dict[str, Any] | None = None,
        **_: Any,
    ):
        super().__init__()
        self.supports_modality_kwargs = True
        self.modalities = normalize_modalities(tuple(modalities or ("image",)), context="modular sequence modalities")
        self.feature_size = int(feature_size)
        self.d_model = int(d_model or feature_size)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.image_profile = resolve_image_profile(image_profile)
        self.image_channels = int(image_channels or image_profile_spec(self.image_profile).channels)

        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.encoder_output_dims: dict[str, int] = {}

        encoder_cfgs = dict(encoders or {})
        projector_cfgs = dict(projectors or {})
        for modality in self.modalities:
            encoder_cfg = self._encoder_config(
                modality,
                encoder_cfgs.get(modality),
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                coord_input_size=coord_input_size,
                lidar_channels=lidar_channels,
                mmwave_input_size=mmwave_input_size,
                ray_input_size=ray_input_size,
                csi_train_rms=csi_train_rms,
            )
            self._validate_modality_encoder_profile(modality, encoder_cfg)
            encoder = ENCODERS.build(encoder_cfg)
            self.encoders[modality] = encoder
            raw_dim = int(getattr(encoder, "output_dim", encoder_cfg.get("output_dim", self.feature_size)))
            self.encoder_output_dims[modality] = raw_dim
            projector_cfg = self._projector_config(projector_cfgs.get(modality), input_dim=raw_dim)
            self.projectors[modality] = PROJECTORS.build(projector_cfg)

        core_cfg = self._core_config(representation_core)
        self.representation_core = REPRESENTATION_CORES.build(core_cfg)
        core_output_dim = int(getattr(self.representation_core, "output_dim", self.d_model))
        head_cfgs = dict(heads or {})
        beam_cfg = dict(head_cfgs.get("beam") or head_cfgs.get("beam_head") or {"type": "beam_head"})
        beam_cfg.setdefault("input_dim", core_output_dim)
        beam_cfg.setdefault("num_classes", self.num_classes)
        self.heads = nn.ModuleDict({"beam": HEADS.build(beam_cfg)})
        self.auxiliary_heads = TemporalAuxiliaryHeads(
            core_output_dim,
            num_pred=self.num_pred,
            auxiliary_heads=auxiliary_heads,
            dropout=float(beam_cfg.get("dropout", 0.0)),
        )

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        csi_batch: torch.Tensor | None = None,
        coord_batch: torch.Tensor | None = None,
        ray_batch: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        raw_inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
            "mmwave": mmwave_batch,
            "csi": csi_batch,
            "coord": coord_batch,
            "ray": ray_batch,
        }
        encoded: dict[str, torch.Tensor] = {}
        projected: dict[str, torch.Tensor] = {}
        batch_size = None
        seq_len = None
        for modality in self.modalities:
            tensor = raw_inputs[modality]
            if tensor is None:
                raise ValueError(f"Modular sequence model requires '{modality}' input because it is enabled.")
            features = self.encoders[modality](tensor)
            batch_size, seq_len = _check_temporal_features(features, modality, batch_size, seq_len)
            encoded[modality] = features
            projected_features = self.projectors[modality](features)
            _check_projected_features(projected_features, modality, self.d_model)
            projected[modality] = projected_features
        ordered = [projected[modality] for modality in self.modalities]
        if len(ordered) == 1:
            core_input = ordered[0]
            input_features = core_input
        else:
            stacked = torch.stack(ordered, dim=1)
            core_input = stacked
            input_features = torch.cat(ordered, dim=-1)
        output_features = self.representation_core(core_input)
        logits = self.heads["beam"](output_features)
        output = {
            "logits": logits,
            "input_features": input_features,
            "output_features": output_features,
            "modalities": self.modalities,
            "modality_features": projected,
            "encoder_features": encoded,
            "image_profile": self.image_profile,
        }
        output.update(self.auxiliary_heads(output_features))
        return output

    def _encoder_config(
        self,
        modality: str,
        raw_cfg: Any,
        *,
        radar_channels: int,
        gps_input_size: int,
        coord_input_size: int,
        lidar_channels: int,
        mmwave_input_size: int,
        ray_input_size: int,
        csi_train_rms: float,
    ) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {"type": _default_encoder_type(modality, self.image_profile)}
        if isinstance(raw_cfg, str):
            raw_cfg = {"type": raw_cfg}
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Encoder config for modality '{modality}' must be a dict or string.")
        cfg = dict(raw_cfg)
        cfg.setdefault("output_dim", self.feature_size)
        if modality == "image":
            cfg.setdefault("image_profile", self.image_profile)
            cfg.setdefault("image_channels", self.image_channels)
        elif modality == "radar":
            cfg.setdefault("radar_channels", radar_channels)
        elif modality == "gps":
            cfg.setdefault("gps_input_size", gps_input_size)
        elif modality == "coord":
            cfg.setdefault("coord_input_size", coord_input_size)
        elif modality == "lidar":
            cfg.setdefault("lidar_channels", lidar_channels)
        elif modality == "mmwave":
            cfg.setdefault("mmwave_input_size", mmwave_input_size)
        elif modality == "ray":
            cfg.setdefault("ray_input_size", ray_input_size)
        elif modality == "csi":
            cfg.setdefault("train_rms", csi_train_rms)
        return cfg

    def _projector_config(self, raw_cfg: Any, *, input_dim: int) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {"type": "linear"}
        if isinstance(raw_cfg, str):
            raw_cfg = {"type": raw_cfg}
        if not isinstance(raw_cfg, dict):
            raise ValueError("Projector config must be a dict or string.")
        cfg = dict(raw_cfg)
        cfg.setdefault("input_dim", input_dim)
        cfg.setdefault("d_model", self.d_model)
        return cfg

    def _core_config(self, raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
        if raw_cfg is None:
            raw_cfg = {"type": "single_gru" if len(self.modalities) == 1 else "early_concat_gru"}
        cfg = dict(raw_cfg)
        cfg.setdefault("d_model", self.d_model)
        cfg.setdefault("modality_count", len(self.modalities))
        return cfg

    def _validate_modality_encoder_profile(self, modality: str, encoder_cfg: dict[str, Any]) -> None:
        if modality != "image":
            return
        encoder_name = str(encoder_cfg.get("type"))
        if encoder_name == "resnet18_imagenet_rgb":
            validate_image_encoder_profile(
                encoder_name=encoder_name,
                image_profile=self.image_profile,
                expected_channels=3,
                actual_channels=encoder_cfg.get("image_channels", self.image_channels),
            )
        elif encoder_name in REMOVED_IMAGE_ENCODERS:
            raise ValueError(
                f"Removed image encoder '{encoder_name}' is no longer supported. "
                "Use 'resnet18_imagenet_rgb' with image_profile 'rgb_imagenet'."
            )


def _default_encoder_type(modality: str, image_profile: str) -> str:
    if modality == "image":
        return "resnet18_imagenet_rgb"
    return {
        "radar": "radar_cnn",
        "gps": "gps_mlp",
        "coord": "coord_mlp",
        "lidar": "lidar_cnn",
        "mmwave": "mmwave_mlp",
        "csi": "pilot_dual_view_csi",
        "ray": "ray_mlp",
    }[modality]


def _snapshot_projection(input_dim: int, output_dim: int, *, dropout: float, activation: str) -> nn.Module:
    hidden_dim = max(int(input_dim), int(output_dim))
    return nn.Sequential(
        nn.Linear(int(input_dim), hidden_dim),
        _snapshot_activation(activation),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden_dim, int(output_dim)),
    )


def _snapshot_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    return nn.GELU()


def _check_temporal_features(
    features: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
) -> tuple[int, int]:
    if features.ndim != 3:
        raise ValueError(f"{modality} encoder output must have shape [B, T, D], got {tuple(features.shape)}.")
    current_batch, current_seq = int(features.shape[0]), int(features.shape[1])
    if batch_size is not None and (current_batch != batch_size or current_seq != seq_len):
        raise ValueError(
            "Modular sequence modalities must share batch/time dimensions; "
            f"modality '{modality}' produced shape {tuple(features.shape)}, "
            f"expected batch={batch_size}, time={seq_len}."
        )
    return current_batch, current_seq


def _check_projected_features(features: torch.Tensor, modality: str, d_model: int) -> None:
    if features.ndim != 3 or int(features.shape[-1]) != int(d_model):
        raise ValueError(
            f"{modality} projector output must have shape [B, T, {int(d_model)}], got {tuple(features.shape)}."
        )


__all__ = [
    "BeamClassificationHead",
    "EarlyConcatGRUCore",
    "GpsMLPEncoder",
    "IdentityProjector",
    "LidarCNNEncoder",
    "LinearProjector",
    "MmWaveMLPEncoder",
    "ModularSequenceModel",
    "PilotDualViewCSIEncoder",
    "RadarCNNEncoder",
    "ResNet18ImageEncoder",
    "SingleGRUCore",
    "SnapshotFrameCore",
    "TokenTransformerCore",
]
