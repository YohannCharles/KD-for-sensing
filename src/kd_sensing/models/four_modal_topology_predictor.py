from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank, beam_topology_positions
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.models.beam_posterior import beam_posterior_statistics
from kd_sensing.models.temporal_transformer import SharedTemporalTransformer
from kd_sensing.registries import ENCODERS, MODELS


PROBABILITY_PARAMETERIZATION = "prototype_probability_mean_v1"
STATIC_RELIABILITY_PARAMETERIZATION = "prototype_probability_static_reliability_v1"
BOUNDED_STATIC_RELIABILITY_PARAMETERIZATION = "prototype_probability_bounded_static_reliability_v1"
MASKED_FEATURE_MLP_PARAMETERIZATION = "prototype_probability_masked_feature_mlp_v1"
FUSION_MODES = {
    "mean",
    "trainable_static_reliability",
    "bounded_static_reliability",
    "masked_feature_mlp",
}


@MODELS.register("four_modal_topology_predictor")
class FourModalTopologyPredictor(nn.Module):
    """Single-stage four-sensing beam posterior predictor."""

    supports_modality_kwargs = True
    supports_force_modality_mask = True
    probability_parameterization = PROBABILITY_PARAMETERIZATION

    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int = 64,
        num_classes: int = 64,
        num_pred: int = 1,
        seq_length: int = 5,
        dropout: float = 0.1,
        encoders: Mapping[str, Mapping[str, Any]] | None = None,
        beam_proto_temperature: float = 0.1,
        fusion_mode: str = "mean",
        temporal_transformer: Mapping[str, Any] | None = None,
        prototype_topology_id: str = "cyclic_index_v1",
        prototype_topology_permutation: list[int] | tuple[int, ...] | None = None,
        prototype_topology_descriptor_sha256: str = "",
        prototype_topology_audit_path: str = "",
        prototype_topology_audit_sha256: str = "",
        consume_missing_modality_metadata: bool = True,
        image_channels: int = 3,
        radar_channels: int = 2,
        lidar_channels: int = 3,
        gps_input_size: int = 3,
    ) -> None:
        super().__init__()
        self.modalities = normalize_modalities(
            modalities or MODALITY_ORDER,
            context="four_modal_topology_predictor.modalities",
        )
        if tuple(self.modalities) != tuple(MODALITY_ORDER):
            raise ValueError(f"Topology predictor requires canonical modalities {list(MODALITY_ORDER)}.")
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.seq_length = int(seq_length)
        self.consume_missing_modality_metadata = bool(consume_missing_modality_metadata)
        if (self.d_model, self.num_classes, self.num_pred, self.seq_length) != (64, 64, 1, 5):
            raise ValueError("Topology predictor requires d_model=64, num_classes=64, num_pred=1, seq_length=5.")
        if not 0.0 <= float(dropout) < 1.0 or float(beam_proto_temperature) <= 0.0:
            raise ValueError("dropout and beam_proto_temperature are invalid.")
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {sorted(FUSION_MODES)}.")
        if self.fusion_mode in {"trainable_static_reliability", "bounded_static_reliability"}:
            self.fusion_logits = nn.Parameter(torch.zeros(len(self.modalities), dtype=torch.float32))
            self.probability_parameterization = (
                BOUNDED_STATIC_RELIABILITY_PARAMETERIZATION
                if self.fusion_mode == "bounded_static_reliability"
                else STATIC_RELIABILITY_PARAMETERIZATION
            )
        else:
            self.register_parameter("fusion_logits", None)
            self.probability_parameterization = (
                MASKED_FEATURE_MLP_PARAMETERIZATION
                if self.fusion_mode == "masked_feature_mlp"
                else PROBABILITY_PARAMETERIZATION
            )

        temporal = _strict_mapping(
            temporal_transformer,
            allowed={"num_layers", "num_heads", "dim_feedforward", "dropout", "norm_first", "causal", "adapter_enabled"},
            context="model.primary.temporal_transformer",
        )
        raw_encoders = encoders if isinstance(encoders, Mapping) else {}
        encoder_configs = {name: dict(raw_encoders.get(name, {})) for name in self.modalities}
        missing = [name for name, config in encoder_configs.items() if not config]
        if missing:
            raise ValueError(f"Topology predictor requires encoders for {missing}.")
        defaults = {
            "image": {"image_channels": int(image_channels)},
            "radar": {"radar_channels": int(radar_channels)},
            "gps": {"gps_input_size": int(gps_input_size)},
            "lidar": {"lidar_channels": int(lidar_channels)},
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
            self.encoder_projections[name] = (
                nn.Identity() if output_dim == self.d_model else nn.Linear(output_dim, self.d_model)
            )
            self.encoder_configs[name] = config

        self.temporal_transformer = SharedTemporalTransformer(
            d_model=self.d_model,
            num_modalities=len(self.modalities),
            seq_length=self.seq_length,
            num_layers=int(temporal.get("num_layers", 2)),
            num_heads=int(temporal.get("num_heads", 4)),
            dim_feedforward=int(temporal.get("dim_feedforward", 128)),
            dropout=float(temporal.get("dropout", dropout)),
            norm_first=bool(temporal.get("norm_first", True)),
            causal=bool(temporal.get("causal", False)),
            adapter_enabled=bool(temporal.get("adapter_enabled", True)),
        )
        self.prototype_bank = BeamPrototypeBank(
            self.d_model,
            self.num_classes,
            temperature=float(beam_proto_temperature),
        )
        if self.fusion_mode == "masked_feature_mlp":
            fusion_input_dim = len(self.modalities) * self.d_model + len(self.modalities)
            self.feature_fusion = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, 2 * self.d_model),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(2 * self.d_model, self.d_model),
            )
        else:
            self.feature_fusion = None
        self.prototype_topology_id = str(prototype_topology_id).strip().lower()
        self.prototype_topology_permutation = (
            list(prototype_topology_permutation) if prototype_topology_permutation is not None else None
        )
        beam_topology_positions(
            self.num_classes,
            topology_id=self.prototype_topology_id,
            topology_permutation=self.prototype_topology_permutation,
        )
        self.prototype_topology_descriptor_sha256 = str(prototype_topology_descriptor_sha256).strip().lower()
        self.prototype_topology_audit_path = str(prototype_topology_audit_path).strip()
        self.prototype_topology_audit_sha256 = str(prototype_topology_audit_sha256).strip().lower()
        _validate_topology_provenance(self)

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
    ) -> dict[str, Any]:
        values = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
        }
        sequences = [self._encode_sequence(name, values[name]) for name in self.modalities]
        latent_sequence = torch.stack(sequences, dim=2)
        requested = missing_mask if missing_mask is not None else force_modality_mask
        available = self._resolve_modality_mask(requested, available_modalities, latent_sequence)
        cell_mask = self._resolve_temporal_mask(
            latent_sequence,
            available,
            temporal_mask,
            modality_temporal_mask,
        )
        temporal = self.temporal_transformer(latent_sequence, cell_mask)
        modality_features = temporal["temporal_cls_features"]
        available = temporal["available_modalities"]
        unimodal_logits = self.prototype_bank(modality_features.reshape(-1, self.d_model)).reshape(
            modality_features.shape[0], len(self.modalities), self.num_classes
        )
        unimodal_probability = torch.softmax(unimodal_logits.float(), dim=-1)
        unimodal_probability = unimodal_probability * available.unsqueeze(-1).to(torch.float32)
        if self.feature_fusion is not None:
            masked_features = modality_features * available.unsqueeze(-1).to(modality_features.dtype)
            fusion_input = torch.cat(
                (
                    masked_features.flatten(start_dim=1),
                    available.to(modality_features.dtype),
                ),
                dim=1,
            )
            fused_features = self.feature_fusion(fusion_input)
            fused_logits = self.prototype_bank(fused_features)
            fused_probability = torch.softmax(fused_logits.float(), dim=-1)
            logits = fused_logits.unsqueeze(1)
            weights = None
        else:
            weights = self._fusion_weights(available)
            fused_probability = (unimodal_probability * weights.unsqueeze(-1)).sum(dim=1)
            fused_probability = fused_probability / fused_probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            fused_features = (modality_features * weights.unsqueeze(-1).to(modality_features.dtype)).sum(dim=1)
            tiny = torch.finfo(torch.float32).tiny
            logits = fused_probability.clamp_min(tiny).log().unsqueeze(1)
        statistics = beam_posterior_statistics(fused_probability.detach())
        return {
            "logits": logits,
            "input_features": modality_features,
            "output_features": fused_features,
            "modality_features": modality_features,
            "missing_mask": available,
            "available_modalities": available,
            "modality_temporal_mask": cell_mask,
            "temporal_mask": cell_mask.any(dim=2),
            "temporal_token_features": temporal["temporal_token_features"],
            "temporal_cls_features": modality_features,
            "temporal_attention_valid_fraction": temporal["temporal_attention_valid_fraction"],
            "temporal_pooling_type": temporal["temporal_pooling_type"],
            "temporal_pooling_param_count": temporal["temporal_pooling_param_count"],
            "unimodal_logits": unimodal_logits,
            "unimodal_probabilities": unimodal_probability,
            "fusion_weights": weights,
            "fused_probability": fused_probability,
            "prototype_state": self.prototype_bank.describe(fused_features),
            "metadata": self.checkpoint_metadata(),
            **statistics,
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "type": "four_modal_topology_predictor",
            "method": "four-modal-topology-predictor",
            "architecture_category": "single_stage_temporal_shared_prototype",
            "modalities": list(self.modalities),
            "probability_parameterization": self.probability_parameterization,
            "fusion": {
                "mean": "availability_masked_probability_mean",
                "trainable_static_reliability": "availability_masked_trainable_static_reliability",
                "bounded_static_reliability": "availability_masked_bounded_static_reliability",
                "masked_feature_mlp": "availability_masked_feature_mlp",
            }[self.fusion_mode],
            "fusion_mode": self.fusion_mode,
            "fusion_logit_constraint": (
                "tanh_unit_interval" if self.fusion_mode == "bounded_static_reliability" else "none"
            ),
            "global_fusion_weights": self._global_fusion_weights(),
            "fusion_has_explicit_modality_weights": self.fusion_mode != "masked_feature_mlp",
            "prototype_bank_count": 1,
            "prototype_feature_sources": [*self.modalities, "fused"],
            "prototype_topology": self.prototype_topology_metadata(),
            "claim_ineligible": True,
            "outer_test_accessed": False,
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return self.checkpoint_metadata()

    def prototype_topology_metadata(self) -> dict[str, Any]:
        return {
            "id": self.prototype_topology_id,
            "descriptor_sha256": self.prototype_topology_descriptor_sha256,
            "audit_path": self.prototype_topology_audit_path,
            "audit_sha256": self.prototype_topology_audit_sha256,
        }

    def _encode_sequence(self, modality: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"Topology predictor requires {modality}_batch.")
        features = self.encoders[modality](value)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError(f"{modality} encoder must return [B,T,D], got {tuple(features.shape)}.")
        features = self.encoder_projections[modality](features)
        if tuple(features.shape[1:]) != (self.seq_length, self.d_model):
            raise ValueError(
                f"{modality} encoder must return [B,{self.seq_length},{self.d_model}], got {tuple(features.shape)}."
            )
        return features

    def _fusion_weights(self, available: torch.Tensor) -> torch.Tensor:
        if self.fusion_logits is None:
            weights = available.to(torch.float32)
            return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        effective_logits = (
            torch.tanh(self.fusion_logits)
            if self.fusion_mode == "bounded_static_reliability"
            else self.fusion_logits
        )
        logits = effective_logits.unsqueeze(0).expand(available.shape[0], -1)
        return torch.softmax(logits.masked_fill(~available, -torch.inf), dim=1)

    def _global_fusion_weights(self) -> list[float] | None:
        if self.fusion_mode == "masked_feature_mlp":
            return None
        if self.fusion_logits is None:
            return [1.0 / len(self.modalities)] * len(self.modalities)
        logits = self.fusion_logits.detach().float()
        if self.fusion_mode == "bounded_static_reliability":
            logits = torch.tanh(logits)
        return torch.softmax(logits, dim=0).cpu().tolist()

    def _resolve_modality_mask(
        self,
        requested: torch.Tensor | None,
        available_modalities: torch.Tensor | None,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(sequence.shape[0])
        count = len(self.modalities)
        raw = requested if requested is not None else available_modalities
        if raw is None:
            return torch.ones(batch_size, count, device=sequence.device, dtype=torch.bool)
        mask = torch.as_tensor(raw, device=sequence.device, dtype=torch.bool)
        if tuple(mask.shape) == (count,):
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, count):
            raise ValueError(f"Topology predictor modality mask must have shape {(batch_size, count)}.")
        if requested is not None and available_modalities is not None:
            intrinsic = torch.as_tensor(available_modalities, device=sequence.device, dtype=torch.bool)
            if tuple(intrinsic.shape) != (batch_size, count):
                raise ValueError("available_modalities must match [B,4].")
            mask = mask & intrinsic
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("Topology predictor requires at least one available modality per sample.")
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
            expected = (batch_size, steps, modalities)
            if tuple(mask.shape) != expected:
                raise ValueError(f"modality_temporal_mask must have shape {expected}.")
        elif temporal_mask is not None:
            time = torch.as_tensor(temporal_mask, device=sequence.device, dtype=torch.bool)
            expected = (batch_size, steps)
            if tuple(time.shape) != expected:
                raise ValueError(f"temporal_mask must have shape {expected}.")
            mask = time.unsqueeze(-1).expand(-1, -1, modalities)
        else:
            mask = torch.ones(batch_size, steps, modalities, device=sequence.device, dtype=torch.bool)
        mask = mask & available.unsqueeze(1)
        if not bool(mask.any(dim=(1, 2)).all().item()):
            raise ValueError("Topology predictor requires at least one available temporal cell per sample.")
        return mask


def validate_topology_predictor_model_config(primary: Mapping[str, Any], dataset: Mapping[str, Any]) -> None:
    if not isinstance(primary, Mapping):
        raise ValueError("model.primary must be a mapping.")
    allowed = set(inspect.signature(FourModalTopologyPredictor).parameters) | {"type"}
    unknown = sorted(set(primary) - allowed)
    if unknown:
        raise ValueError(f"four_modal_topology_predictor contains unsupported fields: {unknown}.")
    if tuple(primary.get("modalities", ())) != tuple(MODALITY_ORDER):
        raise ValueError(f"Topology predictor requires modalities {list(MODALITY_ORDER)}.")
    if int(dataset.get("seq_len", 0)) != 5 or int(dataset.get("num_pred", 0)) != 1:
        raise ValueError("Topology predictor requires five history frames and one future label.")


def _validate_topology_provenance(model: FourModalTopologyPredictor) -> None:
    values = (
        model.prototype_topology_descriptor_sha256,
        model.prototype_topology_audit_path,
        model.prototype_topology_audit_sha256,
    )
    if model.prototype_topology_id != "ula_dft_phase_cycle_v1":
        if any(values):
            raise ValueError(f"Topology {model.prototype_topology_id!r} does not accept physical audit fields.")
        return
    if not _is_sha256(values[0]) or not values[1] or not _is_sha256(values[2]):
        raise ValueError("ULA-DFT topology requires descriptor/audit SHA256 and audit path.")
    audit_path = Path(values[1])
    if not audit_path.is_file() or hashlib.sha256(audit_path.read_bytes()).hexdigest() != values[2]:
        raise ValueError("ULA-DFT topology audit path or SHA256 is invalid.")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ULA-DFT topology audit is unreadable.") from exc
    descriptor = audit.get("descriptor") if isinstance(audit, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError("ULA-DFT topology audit lacks descriptor.")
    digest = hashlib.sha256(
        json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        digest != values[0]
        or audit.get("descriptor_sha256") != digest
        or descriptor.get("topology_id") != "ula_dft_phase_cycle_v1"
        or descriptor.get("codebook_type") != "ula_dft"
        or int(descriptor.get("num_beams", -1)) != model.num_classes
    ):
        raise ValueError("ULA-DFT topology audit descriptor does not match the model.")


def _strict_mapping(value: Mapping[str, Any] | None, *, allowed: set[str], context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    result = dict(value)
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unsupported fields: {unknown}.")
    return result


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


__all__ = [
    "BOUNDED_STATIC_RELIABILITY_PARAMETERIZATION",
    "FUSION_MODES",
    "FourModalTopologyPredictor",
    "MASKED_FEATURE_MLP_PARAMETERIZATION",
    "PROBABILITY_PARAMETERIZATION",
    "STATIC_RELIABILITY_PARAMETERIZATION",
    "validate_topology_predictor_model_config",
]
