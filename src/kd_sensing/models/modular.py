from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import MODALITY_ORDER, image_profile_spec, normalize_modalities, resolve_image_profile
import kd_sensing.models.amber_full  # noqa: F401
import kd_sensing.models.rmbp_mm  # noqa: F401
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.lidar import LidarFeatureExtractor
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
        self.output_dim = _resolve_dim(output_dim, feature_size, d_model)
        super().__init__(self.output_dim, in_channels=int(in_channels or lidar_channels))


@PROJECTORS.register("linear")
class LinearProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.0, **_: Any) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.input_dim, self.output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim not in {3, 4}:
            raise ValueError(f"projector input must be [B,T,D] or [B,T,S,D], got {tuple(features.shape)}.")
        return self.net(features)


@PROJECTORS.register("identity")
class IdentityProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int | None = None, **_: Any) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim if d_model is None else d_model)
        if self.input_dim != self.output_dim:
            raise ValueError("identity projector requires input_dim == d_model.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


@HEADS.register("beam_head")
class BeamClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0, **_: Any) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(input_dim), int(num_classes)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"beam head expects [B,T,D], got {tuple(features.shape)}.")
        return self.net(features)


@MODELS.register("modular_sequence")
class ModularSequenceModel(nn.Module):
    """AMBER-Full and RMBP-MM's shared four-modality wrapper."""

    supports_modality_kwargs = True

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
        num_pred: int = 1,
        image_profile: str | None = None,
        image_channels: int | None = None,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        paper_metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modalities = normalize_modalities(tuple(modalities or MODALITY_ORDER), context="baseline modalities")
        if self.modalities != MODALITY_ORDER:
            raise ValueError(f"modular_sequence requires {list(MODALITY_ORDER)}, got {list(self.modalities)}.")
        self.feature_size = int(feature_size)
        self.d_model = int(d_model or feature_size)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.image_profile = resolve_image_profile(image_profile)
        self.image_channels = int(image_channels or image_profile_spec(self.image_profile).channels)
        self.paper_metadata = dict(paper_metadata or {})

        encoder_cfgs = dict(encoders or {})
        projector_cfgs = dict(projectors or {})
        defaults = {
            "image": {"image_channels": self.image_channels, "image_profile": self.image_profile},
            "radar": {"radar_channels": int(radar_channels)},
            "gps": {"gps_input_size": int(gps_input_size)},
            "lidar": {"lidar_channels": int(lidar_channels)},
        }
        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.encoder_configs: dict[str, dict[str, Any]] = {}
        self.projector_configs: dict[str, dict[str, Any]] = {}
        for modality in self.modalities:
            raw_encoder = encoder_cfgs.get(modality)
            if not isinstance(raw_encoder, dict) or not raw_encoder.get("type"):
                raise ValueError(f"modular_sequence requires an encoder config for '{modality}'.")
            encoder_cfg = {**defaults[modality], **raw_encoder}
            encoder_cfg.setdefault("output_dim", self.feature_size)
            encoder = ENCODERS.build(encoder_cfg)
            raw_dim = int(getattr(encoder, "output_dim", encoder_cfg["output_dim"]))
            projector_cfg = dict(projector_cfgs.get(modality) or {"type": "linear"})
            projector_cfg.setdefault("input_dim", raw_dim)
            projector_cfg.setdefault("d_model", self.d_model)
            self.encoders[modality] = encoder
            self.projectors[modality] = PROJECTORS.build(projector_cfg)
            self.encoder_configs[modality] = encoder_cfg
            self.projector_configs[modality] = projector_cfg

        if not isinstance(representation_core, dict):
            raise ValueError("modular_sequence requires representation_core for AMBER-Full or RMBP-MM.")
        core_cfg = dict(representation_core)
        core_type = str(core_cfg.get("type", ""))
        if core_type not in {"amber_full_adaptive_mask_transformer", "rmbp_channel_attention_fusion"}:
            raise ValueError(f"Unsupported baseline representation_core {core_type!r}.")
        core_cfg.setdefault("d_model", self.d_model)
        core_cfg.setdefault("modality_count", len(self.modalities))
        self.representation_core_config = core_cfg
        self.representation_core = REPRESENTATION_CORES.build(core_cfg)

        head_cfg = dict((heads or {}).get("beam") or {"type": "beam_head"})
        head_cfg.setdefault("input_dim", int(getattr(self.representation_core, "output_dim", self.d_model)))
        head_cfg.setdefault("num_classes", self.num_classes)
        if head_cfg.get("type") != "beam_head":
            raise ValueError("modular_sequence only supports heads.beam.type='beam_head'.")
        self.head_configs = {"beam": head_cfg}
        self.heads = nn.ModuleDict({"beam": HEADS.build(head_cfg)})

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        image_valid_mask: torch.Tensor | None = None,
        radar_valid_mask: torch.Tensor | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        lidar_valid_mask: torch.Tensor | None = None,
        image_dropout_mask: torch.Tensor | None = None,
        radar_dropout_mask: torch.Tensor | None = None,
        gps_dropout_mask: torch.Tensor | None = None,
        lidar_dropout_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        modality_temporal_mask: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
        available_modalities: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
        missing_modality_metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        raw_inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
        }
        encoded: dict[str, torch.Tensor] = {}
        projected: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            value = raw_inputs[modality]
            if value is None:
                raise ValueError(f"modular_sequence requires {modality}_batch.")
            features = self.encoders[modality](value)
            if features.ndim not in {3, 4}:
                raise ValueError(f"{modality} encoder must return [B,T,D] or [B,T,S,D].")
            encoded[modality] = features
            projected[modality] = self.projectors[modality](features)

        batch_size, seq_len = projected["image"].shape[:2]
        availability = _availability_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=projected["image"].device,
            valid_masks={
                "image": image_valid_mask,
                "radar": radar_valid_mask,
                "gps": gps_valid_mask,
                "lidar": lidar_valid_mask,
            },
            dropout_masks={
                "image": image_dropout_mask,
                "radar": radar_dropout_mask,
                "gps": gps_dropout_mask,
                "lidar": lidar_dropout_mask,
            },
            temporal_mask=temporal_mask,
            modality_temporal_mask=modality_temporal_mask,
            missing_mask=missing_mask,
            modality_mask=modality_mask,
            available_modalities=available_modalities,
        )

        if self.representation_core_config["type"] == "amber_full_adaptive_mask_transformer":
            core_input, core_availability, input_features = _amber_input(projected, availability)
            output_features = self.representation_core(core_input, modality_available=core_availability)
        else:
            core_input = torch.stack([_frame_features(projected[modality]) for modality in self.modalities], dim=1)
            core_input = core_input * availability.unsqueeze(-1).to(dtype=core_input.dtype)
            input_features = torch.cat([core_input[:, index] for index in range(len(self.modalities))], dim=-1)
            output_features = self.representation_core(core_input, modality_available=availability)
            core_availability = availability
        logits = self.heads["beam"](output_features)
        output: dict[str, Any] = {
            "logits": logits,
            "input_features": input_features,
            "output_features": output_features,
            "modalities": self.modalities,
            "modality_features": projected,
            "encoder_features": encoded,
            "image_profile": self.image_profile,
            "missing_modality_metadata": _missing_metadata(availability, missing_modality_metadata),
        }
        if self.representation_core_config["type"] == "amber_full_adaptive_mask_transformer":
            output["token_features"] = core_input
            auxiliary = getattr(self.representation_core, "last_amber_full_auxiliary", None)
            if isinstance(auxiliary, dict):
                output["amber_full_auxiliary"] = auxiliary
            attention_mask = getattr(self.representation_core, "last_amber_full_attention_mask", None)
            if torch.is_tensor(attention_mask):
                output["amber_full_attention_key_padding_mask"] = attention_mask
        else:
            weights = getattr(self.representation_core, "last_attention_weights", None)
            if torch.is_tensor(weights):
                output["rmbp_attention_weights"] = weights
        return output

    def training_strategy_metadata(self) -> dict[str, Any]:
        core_metadata = _component_metadata(
            self.representation_core,
            self.representation_core_config,
            role="representation_core",
        )
        metadata = {
            "type": "modular_sequence",
            "architecture_category": "component_baseline",
            "model_group": "modular_sequence",
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "d_model": self.d_model,
            "encoders": {
                modality: _component_metadata(self.encoders[modality], self.encoder_configs[modality], role="encoder")
                for modality in self.modalities
            },
            "projectors": {
                modality: _component_metadata(self.projectors[modality], self.projector_configs[modality], role="projector")
                for modality in self.modalities
            },
            "representation_core_type": self.representation_core_config["type"],
            "representation_core_class": self.representation_core.__class__.__name__,
            "representation_core": core_metadata,
            "heads": {"beam": _component_metadata(self.heads["beam"], self.head_configs["beam"], role="head")},
            "consumes_missing_modality_metadata": True,
            "missing_modality_metadata": {
                "consumed": True,
                "consumers": ["representation_core"],
                "fields": [f"{modality}_valid_mask" for modality in self.modalities],
            },
        }
        metadata.update(self.paper_metadata)
        return metadata


def _component_metadata(module: nn.Module, config: dict[str, Any], *, role: str) -> dict[str, Any]:
    strategy = getattr(module, "training_strategy_metadata", None)
    metadata = dict(strategy()) if callable(strategy) else {}
    return {"type": config["type"], "component_role": role, **metadata}


def _frame_features(features: torch.Tensor) -> torch.Tensor:
    return features if features.ndim == 3 else features.mean(dim=2)


def _amber_input(
    projected: dict[str, torch.Tensor], availability: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pieces = [projected[modality] if projected[modality].ndim == 4 else projected[modality].unsqueeze(2) for modality in MODALITY_ORDER]
    max_tokens = max(int(piece.shape[2]) for piece in pieces)
    padded = []
    for piece in pieces:
        if int(piece.shape[2]) < max_tokens:
            pad = piece.new_zeros((*piece.shape[:2], max_tokens - int(piece.shape[2]), piece.shape[-1]))
            piece = torch.cat([piece, pad], dim=2)
        padded.append(piece)
    core_input = torch.stack(padded, dim=1)
    core_availability = availability.unsqueeze(-1).expand(-1, -1, -1, max_tokens)
    core_input = core_input * core_availability.unsqueeze(-1).to(dtype=core_input.dtype)
    input_features = torch.cat([piece.mean(dim=2) for piece in padded], dim=-1)
    return core_input, core_availability, input_features


def _availability_mask(
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    valid_masks: dict[str, torch.Tensor | None],
    dropout_masks: dict[str, torch.Tensor | None],
    temporal_mask: torch.Tensor | None,
    modality_temporal_mask: torch.Tensor | None,
    missing_mask: torch.Tensor | None,
    modality_mask: torch.Tensor | None,
    available_modalities: torch.Tensor | None,
) -> torch.Tensor:
    available = torch.ones(batch_size, len(MODALITY_ORDER), seq_len, dtype=torch.bool, device=device)
    for index, modality in enumerate(MODALITY_ORDER):
        if valid_masks[modality] is not None:
            available[:, index] &= _coerce_temporal_mask(valid_masks[modality], batch_size, seq_len, device)
        if dropout_masks[modality] is not None:
            available[:, index] &= ~_coerce_temporal_mask(dropout_masks[modality], batch_size, seq_len, device)
    if temporal_mask is not None:
        available &= _coerce_temporal_mask(temporal_mask, batch_size, seq_len, device).unsqueeze(1)
    for value in (modality_temporal_mask,):
        if value is not None:
            available &= _coerce_modality_temporal_mask(value, batch_size, seq_len, device)
    for value in (missing_mask, modality_mask, available_modalities):
        if value is not None:
            available &= _coerce_modality_mask(value, batch_size, device).unsqueeze(-1)
    return available


def _coerce_temporal_mask(value: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.as_tensor(value, dtype=torch.bool, device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(1)
    if tuple(mask.shape) != (batch_size, seq_len):
        raise ValueError(f"temporal mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}.")
    return mask


def _coerce_modality_mask(value: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.as_tensor(value, dtype=torch.bool, device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    if tuple(mask.shape) != (batch_size, len(MODALITY_ORDER)):
        raise ValueError(f"modality mask must have shape {(batch_size, len(MODALITY_ORDER))}, got {tuple(mask.shape)}.")
    return mask


def _coerce_modality_temporal_mask(value: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.as_tensor(value, dtype=torch.bool, device=device)
    expected = (batch_size, seq_len, len(MODALITY_ORDER))
    if tuple(mask.shape) == expected:
        return mask.permute(0, 2, 1).contiguous()
    if tuple(mask.shape) == (batch_size, len(MODALITY_ORDER), seq_len):
        return mask
    raise ValueError(f"modality_temporal_mask must have shape {expected}, got {tuple(mask.shape)}.")


def _missing_metadata(availability: torch.Tensor, input_metadata: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "availability_mask": availability,
        "missing_counts": {
            modality: int((~availability[:, index]).sum().detach().cpu().item())
            for index, modality in enumerate(MODALITY_ORDER)
        },
    }
    if isinstance(input_metadata, dict):
        payload["input_metadata"] = dict(input_metadata)
    return payload


__all__ = [
    "BeamClassificationHead",
    "GpsMLPEncoder",
    "IdentityProjector",
    "LidarCNNEncoder",
    "LinearProjector",
    "ModularSequenceModel",
    "RadarCNNEncoder",
]
