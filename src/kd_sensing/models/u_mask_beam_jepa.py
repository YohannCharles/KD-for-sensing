from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.models.pcer_temporal_fusion import (
    HierarchicalTemporalBlockRouter,
    MaskConditionedResidualRouter,
    PCER_MODES,
    TemporalBlockEvidenceRouter,
    static_block_weights,
)
from kd_sensing.models.pgcd import PGCD_VARIANTS, PrototypeGuidedDegradationRouter
from kd_sensing.models.prototype_health_router import PrototypeReliabilityRouter
from kd_sensing.registries import ENCODERS, MODELS


DEFAULT_MODALITIES = MODALITY_ORDER


def _masked_softmax(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    available = available.to(device=logits.device, dtype=torch.bool)
    if logits.shape != available.shape:
        raise ValueError(f"router logits {tuple(logits.shape)} must match mask {tuple(available.shape)}.")
    if not bool(available.any(dim=1).all().item()):
        raise ValueError("u_mask_beam_jepa requires one available modality per sample.")
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=1) * available.to(dtype=logits.dtype)


def _freeze_module(module: nn.Module | None) -> None:
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


@MODELS.register("u_mask_beam_jepa")
class UMaskBeamJEPA(nn.Module):
    """T2/S1 temporal masked-pooling model with a supervised modality router."""

    supports_modality_kwargs = True
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int = 64,
        num_classes: int = 64,
        num_pred: int = 1,
        seq_length: int = 5,
        dropout: float = 0.1,
        encoders: dict[str, dict[str, Any]] | None = None,
        temporal_pooling: dict[str, Any] | bool | None = None,
        fusion_type: str = "supervised_router",
        head_type: str = "prototype",
        beam_proto_temperature: float = 0.2,
        router_use_pattern_features: bool = True,
        router_use_reliability_features: bool = True,
        router_use_prototype_margin: bool = True,
        router_use_entropy: bool = True,
        router_use_confidence: bool = True,
        router_use_logit_norm: bool = True,
        router_hidden_dim: int = 64,
        router_variant: str = "current",
        router_variant_config: dict[str, Any] | None = None,
        router_calibration_only: bool = False,
        pcer: dict[str, Any] | None = None,
        pgcd: dict[str, Any] | None = None,
        image_channels: int = 3,
        radar_channels: int = 2,
        lidar_channels: int = 3,
        gps_input_size: int = 3,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modalities = _validate_modalities(modalities or DEFAULT_MODALITIES)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.seq_length = int(seq_length)
        self.fusion_type = str(fusion_type).strip().lower()
        self.head_type = str(head_type).strip().lower()
        self.router_use_pattern_features = bool(router_use_pattern_features)
        self.router_use_reliability_features = bool(router_use_reliability_features)
        self.router_use_prototype_margin = bool(router_use_prototype_margin)
        self.router_use_entropy = bool(router_use_entropy)
        self.router_use_confidence = bool(router_use_confidence)
        self.router_use_logit_norm = bool(router_use_logit_norm)
        self.router_variant = str(router_variant).strip().lower()
        self.router_calibration_only = bool(router_calibration_only)
        self.temporal_pooling = _temporal_pooling_config(temporal_pooling)
        self.pcer_config = _pcer_config(pcer, seq_length=self.seq_length)
        self.pgcd_config = _pgcd_config(pgcd, seq_length=self.seq_length)

        if self.d_model <= 0 or self.num_classes <= 0 or self.num_pred <= 0:
            raise ValueError("d_model, num_classes, and num_pred must be positive.")
        if self.fusion_type not in {"supervised_router", "reliability_mean", "uniform_mean"}:
            raise ValueError("T2 fusion_type must be supervised_router, reliability_mean, or uniform_mean.")
        if self.head_type not in {"prototype", "classifier"}:
            raise ValueError("T2 head_type must be 'prototype' or 'classifier'.")
        if self.router_variant not in {"current", "patr", "h2r", "core", "unified_hpr"}:
            raise ValueError("router_variant must be current, patr, h2r, core, or unified_hpr.")
        if self.router_variant != "current" and (
            self.fusion_type != "supervised_router" or self.head_type != "prototype"
        ):
            raise ValueError("Dynamic prototype Router variants require supervised_router fusion and prototype head.")
        if self.router_calibration_only and self.router_variant == "current":
            raise ValueError("router_calibration_only requires a dynamic Router variant.")
        if self.pcer_config["enabled"] and (self.router_variant != "current" or self.head_type != "prototype"):
            raise ValueError("PCER requires router_variant=current and head_type=prototype.")
        if self.pcer_config["enabled"] and self.pcer_config["mode"] != "evidence_only" and self.fusion_type != "uniform_mean":
            raise ValueError("PCER fusion modes require fusion_type=uniform_mean.")
        if self.pgcd_config["enabled"] and self.pcer_config["enabled"]:
            raise ValueError("PGCD and PCER fusion modes are mutually exclusive.")
        if self.pgcd_config["enabled"] and (
            self.router_variant != "current" or self.head_type != "prototype" or self.fusion_type != "uniform_mean"
        ):
            raise ValueError("PGCD requires router_variant=current, head_type=prototype, and fusion_type=uniform_mean.")

        encoder_configs = {name: dict((encoders or {}).get(name, {})) for name in self.modalities}
        missing = [name for name, config in encoder_configs.items() if not config]
        if missing:
            raise ValueError(f"u_mask_beam_jepa requires encoders for {missing}.")
        defaults = {
            "image": {"image_channels": image_channels},
            "radar": {"radar_channels": radar_channels},
            "lidar": {"lidar_channels": lidar_channels},
            "gps": {"gps_input_size": gps_input_size},
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
            self.encoder_projections[name] = nn.Identity() if output_dim == self.d_model else nn.Linear(output_dim, self.d_model)
            self.encoder_configs[name] = config

        self.reliability_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))
                for name in self.modalities
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.num_classes),
        )
        self.prototype_bank = BeamPrototypeBank(self.d_model, self.num_classes, temperature=float(beam_proto_temperature))
        pcer_kwargs = {
            "d_model": self.d_model,
            "num_modalities": len(self.modalities),
            "num_timesteps": self.seq_length,
            "hidden_dim": int(self.pcer_config["hidden_dim"]),
            "embedding_dim": int(self.pcer_config["embedding_dim"]),
            "dropout": float(self.pcer_config["dropout"]),
        }
        if self.pcer_config["mode"] in {"counterfactual_router", "block_router"}:
            self.pcer_router: nn.Module | None = TemporalBlockEvidenceRouter(**pcer_kwargs)
        elif self.pcer_config["mode"] == "hierarchical_router":
            self.pcer_router = HierarchicalTemporalBlockRouter(**pcer_kwargs)
        elif self.pcer_config["mode"] == "mask_residual_router":
            self.pcer_router = MaskConditionedResidualRouter(**pcer_kwargs)
        else:
            self.pcer_router = None
        self.pgcd_router = (
            PrototypeGuidedDegradationRouter(
                d_model=self.d_model,
                num_modalities=len(self.modalities),
                num_timesteps=self.seq_length,
                variant=str(self.pgcd_config["variant"]),
                hidden_dim=int(self.pgcd_config["hidden_dim"]),
                embedding_dim=int(self.pgcd_config["embedding_dim"]),
                dropout=float(self.pgcd_config["dropout"]),
                beta_init=float(self.pgcd_config["beta_reliability_init"]),
            )
            if self.pgcd_config["enabled"]
            else None
        )
        self.temporal_attention_query: nn.Parameter | None = None
        if self.temporal_pooling["type"] == "masked_attention":
            self.temporal_attention_query = nn.Parameter(torch.empty(self.d_model))
            nn.init.normal_(self.temporal_attention_query, std=self.d_model**-0.5)

        self.router_feature_names = _router_feature_names(
            reliability=self.router_use_reliability_features,
            prototype_margin=self.router_use_prototype_margin and self.head_type == "prototype",
            entropy=self.router_use_entropy,
            confidence=self.router_use_confidence,
            logit_norm=self.router_use_logit_norm,
        )
        router_input_dim = len(self.router_feature_names) + len(self.modalities)
        self.supervised_router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, int(router_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), 1),
        )
        self.router_pattern_bias = (
            nn.Linear(len(self.modalities), len(self.modalities), bias=False)
            if self.router_use_pattern_features
            else None
        )
        variant_config = dict(router_variant_config or {})
        self.prototype_reliability_router = (
            PrototypeReliabilityRouter(
                variant=self.router_variant,
                modality_count=len(self.modalities),
                num_classes=self.num_classes,
                base_feature_dim=len(self.router_feature_names),
                prior_weights=variant_config.pop("prior_weights", None),
                topology_id=str(variant_config.pop("topology_id", "cyclic_index_v1")),
                topology_permutation=variant_config.pop("topology_permutation", None),
                circular=bool(variant_config.pop("circular", True)),
                residual_hidden_dim=int(variant_config.pop("residual_hidden_dim", router_hidden_dim)),
                health_hidden_dim=int(variant_config.pop("health_hidden_dim", 16)),
                residual_scale=float(variant_config.pop("residual_scale", 1.0)),
                top_k=int(variant_config.pop("top_k", 3)),
                dropout=float(variant_config.pop("dropout", 0.0)),
                evidence_profile=str(variant_config.pop("evidence_profile", "full")),
            )
            if self.router_variant != "current"
            else None
        )
        if variant_config:
            raise ValueError(f"Unknown router_variant_config fields: {sorted(variant_config)}.")
        frozen_branches: list[str] = []
        if self.head_type == "prototype":
            _freeze_module(self.classifier)
            frozen_branches.append("classifier")
        else:
            _freeze_module(self.prototype_bank)
            frozen_branches.append("prototype_bank")
        if self.fusion_type in {"reliability_mean", "uniform_mean"}:
            _freeze_module(self.supervised_router)
            frozen_branches.append("supervised_router")
            if self.router_pattern_bias is not None:
                _freeze_module(self.router_pattern_bias)
                frozen_branches.append("router_pattern_bias")
        if self.fusion_type == "uniform_mean":
            _freeze_module(self.reliability_heads)
            frozen_branches.append("reliability_heads")
        if self.router_calibration_only:
            for name in (
                "encoders",
                "encoder_projections",
                "reliability_heads",
                "classifier",
                "prototype_bank",
                "supervised_router",
                "router_pattern_bias",
            ):
                _freeze_module(getattr(self, name, None))
                frozen_branches.append(name)
            if self.temporal_attention_query is not None:
                self.temporal_attention_query.requires_grad_(False)
                frozen_branches.append("temporal_attention_query")
        self.frozen_branches = tuple(frozen_branches)

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
        return_router_state: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
        }
        sequences = [self._encode_sequence(name, inputs[name]) for name in self.modalities]
        latent_sequence = torch.stack(sequences, dim=2)
        available = self._resolve_modality_mask(
            missing_mask if missing_mask is not None else force_modality_mask,
            available_modalities,
            latent_sequence,
        )
        cell_mask = self._resolve_temporal_mask(latent_sequence, available, temporal_mask, modality_temporal_mask)
        current_latent, temporal_pooling_weights = self._pool_temporal_sequence(latent_sequence, cell_mask)
        current_reliability = self._modality_reliability(current_latent, available)
        current_unimodal_logits = self._head_logits(current_latent.reshape(-1, self.d_model)).reshape(
            current_latent.shape[0], len(self.modalities), self.num_classes
        )
        with (
            nullcontext()
            if self.fusion_type == "supervised_router" and self.router_variant == "current"
            else torch.no_grad()
        ):
            current_router_features = self._router_features(current_unimodal_logits, current_reliability, available)
            current_router_logits, current_router_weights = self.route_from_features(current_router_features, available)

        dynamic: dict[str, Any] | None = None
        if self.prototype_reliability_router is not None:
            dynamic = self._dynamic_route(latent_sequence, cell_mask, available)
            available = dynamic["available"]
            latent = dynamic["latent"]
            reliability = dynamic["reliability"]
            unimodal_logits = dynamic["unimodal_logits"]
            router_features = dynamic["router_features"]
            router_logits = dynamic["router_gate_logits"]
            router_weights = dynamic["router_gate_weights"]
        else:
            latent = current_latent
            reliability = current_reliability
            unimodal_logits = current_unimodal_logits
            router_features = current_router_features
            router_logits = current_router_logits
            router_weights = current_router_weights
        pcer_output: dict[str, torch.Tensor] | None = None
        if self.pcer_config["enabled"]:
            pcer_output = self._pcer_fusion(latent_sequence, cell_mask)
        pgcd_output: dict[str, torch.Tensor] | None = None
        if self.pgcd_config["enabled"]:
            pgcd_output = self._pgcd_fusion(latent_sequence, cell_mask)
        if pgcd_output is not None:
            fusion_weights = pgcd_output["modality_weights"]
            router_logits = pgcd_output["modality_logits"]
            router_weights = fusion_weights
            fused_features = pgcd_output["fused_features"]
            fused_logits = pgcd_output["fused_logits"]
        elif pcer_output is not None and self.pcer_config["mode"] != "evidence_only":
            fusion_weights = pcer_output["modality_weights"]
            router_logits = pcer_output["modality_logits"]
            router_weights = fusion_weights
            fused_features = pcer_output["fused_features"]
            fused_logits = pcer_output["fused_logits"]
        else:
            fusion_weights = (
                router_weights
                if self.fusion_type == "supervised_router"
                else (
                    _reliability_mean_weights(reliability, cell_mask)
                    if self.fusion_type == "reliability_mean"
                    else _uniform_mean_weights(cell_mask)
                )
            )
            fused_features = (fusion_weights.unsqueeze(-1) * latent).sum(dim=1)
            fused_logits = (fusion_weights.unsqueeze(-1) * unimodal_logits).sum(dim=1)
        output = {
            "logits": fused_logits.unsqueeze(1).expand(-1, self.num_pred, -1),
            "input_features": latent,
            "output_features": fused_features,
            "modality_features": latent,
            "missing_mask": available,
            "available_modalities": available,
            "modality_temporal_mask": cell_mask,
            "temporal_mask": cell_mask.any(dim=2),
            "temporal_pooling_type": self.temporal_pooling["type"],
            "temporal_pooling_param_count": self._temporal_pooling_param_count(),
            "modality_reliability": reliability,
            "global_reliability": reliability.squeeze(-1).sum(dim=1) / available.sum(dim=1).clamp_min(1),
            "unimodal_logits": unimodal_logits,
            "router_gate_logits": router_logits,
            "router_gate_weights": router_weights,
            "supervised_router_gate_logits": router_logits,
            "supervised_router_gate_weights": router_weights,
            "supervised_router_reliability_features": router_features,
            "supervised_router_feature_names": self.router_feature_names,
            "router_variant": self.router_variant,
            "pcer_mode": self.pcer_config["mode"],
            "pgcd_variant": self.pgcd_config["variant"],
            "router_evidence_profile": (
                self.prototype_reliability_router.evidence_profile
                if self.prototype_reliability_router is not None
                else "current"
            ),
            "reliability_fusion_mode": self.fusion_type,
            "reliability_fusion_weights": fusion_weights,
            "metadata": self.training_strategy_metadata(),
        }
        if dynamic is not None:
            output.update(
                {
                    "reference_router_gate_logits": current_router_logits,
                    "reference_router_gate_weights": current_router_weights,
                    "reference_unimodal_logits": current_unimodal_logits,
                    "router_static_prior_weights": dynamic["router_static_prior_weights"],
                    "router_residual_logits": dynamic["router_residual_logits"],
                    "router_temporal_weights": dynamic["router_temporal_weights"],
                    "router_frame_health_logits": dynamic["frame_health_logits"],
                    "router_effective_cell_weights": dynamic["effective_cell_weights"],
                    "router_variant_features": dynamic["router_variant_features"],
                    "router_consensus_features": dynamic["consensus_features"],
                }
            )
            if return_router_state:
                output["candidate_router_state"] = {
                    "latent_sequence": latent_sequence.detach(),
                    "cell_mask": cell_mask.detach(),
                    "available": available.detach(),
                }
        if pcer_output is not None:
            output.update(
                {
                    key: value
                    for key, value in pcer_output.items()
                    if key not in {"modality_weights", "modality_logits", "fused_features", "fused_logits"}
                }
            )
        if pgcd_output is not None:
            output.update(
                {
                    key: value
                    for key, value in pgcd_output.items()
                    if key not in {"modality_weights", "modality_logits", "fused_features", "fused_logits"}
                }
            )
        if temporal_pooling_weights is not None:
            output["temporal_pooling_weights"] = temporal_pooling_weights
        return output

    def training_strategy_metadata(self) -> dict[str, Any]:
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "type": "u_mask_beam_jepa",
            "architecture_category": "t2_temporal_supervised_router",
            "modalities": list(self.modalities),
            "enabled_modalities": list(self.modalities),
            "consumes_missing_mask": True,
            "consumes_missing_modality_metadata": True,
            "fusion_type": self.fusion_type,
            "head_type": self.head_type,
            "active_head": self.head_type,
            "router_variant": self.router_variant,
            "pcer": dict(self.pcer_config),
            "pcer_mode": self.pcer_config["mode"],
            "pcer_router_trainable": self.pcer_router is not None,
            "pgcd": dict(self.pgcd_config),
            "pgcd_variant": self.pgcd_config["variant"],
            "pgcd_router_trainable": self.pgcd_router is not None,
            "router_evidence_profile": (
                self.prototype_reliability_router.evidence_profile
                if self.prototype_reliability_router is not None
                else "current"
            ),
            "router_calibration_only": self.router_calibration_only,
            "frozen_branches": list(self.frozen_branches),
            "router_trainable": self.fusion_type == "supervised_router" and not self.router_calibration_only,
            "prototype_reliability_router_trainable": self.prototype_reliability_router is not None,
            "router_feature_names": list(self.router_feature_names),
            "temporal_pooling": dict(self.temporal_pooling),
            "temporal_pooling_type": self.temporal_pooling["type"],
            "temporal_pooling_param_count": self._temporal_pooling_param_count(),
            "encoder_configs": self.encoder_configs,
            "gps_encoder": self._gps_encoder_metadata(),
            "total_params": total_params,
            "trainable_params": trainable_params,
        }

    def train(self, mode: bool = True) -> "UMaskBeamJEPA":
        super().train(mode)
        if self.router_calibration_only:
            for name in (
                "encoders",
                "encoder_projections",
                "reliability_heads",
                "classifier",
                "prototype_bank",
                "supervised_router",
                "router_pattern_bias",
            ):
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()
        return self

    def route_from_candidate_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rerun only the candidate Router on detached expert state."""

        if self.prototype_reliability_router is None:
            raise RuntimeError("route_from_candidate_state requires a dynamic Router variant.")
        required = {"latent_sequence", "cell_mask", "available"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError(f"Candidate Router state requires fields {sorted(required)}.")
        dynamic = self._dynamic_route(
            state["latent_sequence"].detach(),
            state["cell_mask"].to(dtype=torch.bool),
            state["available"].to(dtype=torch.bool),
        )
        return {
            "router_gate_logits": dynamic["router_gate_logits"],
            "router_gate_weights": dynamic["router_gate_weights"],
            "router_static_prior_weights": dynamic["router_static_prior_weights"],
            "router_residual_logits": dynamic["router_residual_logits"],
            "unimodal_logits": dynamic["unimodal_logits"],
            "fused_logits": dynamic["fused_logits"],
            "available": dynamic["available"],
            "frame_health_logits": dynamic["frame_health_logits"],
            "frame_unimodal_logits": dynamic["frame_unimodal_logits"],
            "cell_mask": dynamic["cell_mask"],
            "router_temporal_weights": dynamic["router_temporal_weights"],
            "effective_cell_weights": dynamic["effective_cell_weights"],
        }

    def _dynamic_route(
        self,
        latent_sequence: torch.Tensor,
        cell_mask: torch.Tensor,
        available: torch.Tensor,
    ) -> dict[str, Any]:
        router = self.prototype_reliability_router
        if router is None:
            raise RuntimeError("Dynamic routing requires prototype_reliability_router.")
        batch_size, steps, modality_count, _ = latent_sequence.shape
        detached_sequence = latent_sequence.detach()
        frame_logits = self._head_logits(detached_sequence.reshape(-1, self.d_model)).reshape(
            batch_size, steps, modality_count, self.num_classes
        )
        frame_reliability = torch.stack(
            [
                torch.sigmoid(self.reliability_heads[name](detached_sequence[:, :, index]))
                for index, name in enumerate(self.modalities)
            ],
            dim=2,
        )
        frame_reliability = frame_reliability * cell_mask.unsqueeze(-1).to(dtype=frame_reliability.dtype)
        evidence = router.prepare(detached_sequence, frame_logits, frame_reliability, cell_mask)
        temporal = router.temporal_pool(latent_sequence, evidence)
        effective_available = available & cell_mask.any(dim=1)
        latent = temporal.features
        reliability = self._modality_reliability(latent, effective_available)
        unimodal_logits = self._head_logits(latent.reshape(-1, self.d_model)).reshape(
            batch_size, modality_count, self.num_classes
        )
        base_features = self._router_features(unimodal_logits, reliability, effective_available)
        base_features = router.filter_base_features(base_features, self.router_feature_names)
        route = router.route(base_features, unimodal_logits, evidence, temporal, effective_available)
        fused_features = (route.weights.unsqueeze(-1) * latent).sum(dim=1)
        fused_logits = (route.weights.unsqueeze(-1) * unimodal_logits).sum(dim=1)
        return {
            "latent": latent,
            "reliability": reliability,
            "router_features": base_features,
            "unimodal_logits": unimodal_logits,
            "fused_features": fused_features,
            "fused_logits": fused_logits,
            "available": effective_available,
            "cell_mask": cell_mask,
            "frame_unimodal_logits": frame_logits,
            "frame_health_logits": temporal.logits,
            "router_temporal_weights": temporal.weights,
            "router_gate_logits": route.gate_logits,
            "router_gate_weights": route.weights,
            "router_static_prior_weights": route.prior_weights,
            "router_residual_logits": route.residual_logits,
            "router_variant_features": route.modality_features,
            "consensus_features": route.consensus_features,
            "effective_cell_weights": route.effective_cell_weights,
        }

    def _modality_reliability(self, latent: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        reliability = torch.stack(
            [torch.sigmoid(self.reliability_heads[name](latent[:, index])) for index, name in enumerate(self.modalities)],
            dim=1,
        )
        return reliability * available.unsqueeze(-1).to(dtype=reliability.dtype)

    def _pcer_fusion(
        self,
        latent_sequence: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, timesteps, modalities, width = latent_sequence.shape
        if (timesteps, modalities, width) != (self.seq_length, len(self.modalities), self.d_model):
            raise ValueError(
                f"PCER expected [B,{self.seq_length},{len(self.modalities)},{self.d_model}] block features, "
                f"got {tuple(latent_sequence.shape)}."
            )
        block_features = latent_sequence.reshape(batch, timesteps * modalities, width)
        block_evidence = self.prototype_bank(block_features.reshape(-1, width)).reshape(
            batch, timesteps, modalities, self.num_classes
        )
        extra: dict[str, torch.Tensor] = {}
        if self.pcer_router is None:
            block_logits, block_weights = static_block_weights(cell_mask)
            block_logits = block_logits.to(device=latent_sequence.device, dtype=latent_sequence.dtype)
            block_weights = block_weights.to(device=latent_sequence.device, dtype=latent_sequence.dtype)
        elif isinstance(self.pcer_router, TemporalBlockEvidenceRouter):
            block_logits, block_weights = self.pcer_router(latent_sequence, block_evidence, cell_mask)
        else:
            routed = self.pcer_router(latent_sequence, block_evidence, cell_mask)
            block_logits, block_weights = routed["logits"], routed["weights"]
            extra = {f"pcer_{key}": value for key, value in routed.items() if key not in {"logits", "weights"}}
        flat_evidence = block_evidence.reshape(batch, timesteps * modalities, self.num_classes)
        fused_features = (block_weights.unsqueeze(-1) * block_features).sum(dim=1)
        fused_logits = (block_weights.unsqueeze(-1) * flat_evidence).sum(dim=1)
        cell_weights = block_weights.reshape(batch, timesteps, modalities)
        modality_weights = cell_weights.sum(dim=1)
        modality_available = cell_mask.any(dim=1)
        modality_logits = modality_weights.clamp_min(torch.finfo(modality_weights.dtype).tiny).log()
        modality_logits = modality_logits.masked_fill(~modality_available, -torch.inf)
        return {
            "pcer_block_features": block_features,
            "pcer_block_evidence_logits": flat_evidence,
            "pcer_block_router_logits": block_logits,
            "pcer_block_router_weights": block_weights,
            "pcer_block_availability": cell_mask.reshape(batch, timesteps * modalities),
            "pcer_effective_cell_weights": cell_weights,
            "modality_weights": modality_weights,
            "modality_logits": modality_logits,
            "fused_features": fused_features,
            "fused_logits": fused_logits,
            **extra,
        }

    def route_pcer_cached(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Rerun only the opt-in PCER router on cached block tensors."""

        if self.pcer_router is None:
            raise RuntimeError("route_pcer_cached requires a trainable PCER router.")
        batch, blocks, width = block_features.shape
        if blocks != self.seq_length * len(self.modalities) or width != self.d_model:
            raise ValueError("Cached PCER features must have shape [B,T*M,D].")
        features = block_features.reshape(batch, self.seq_length, len(self.modalities), self.d_model)
        evidence = block_evidence_logits.reshape(batch, self.seq_length, len(self.modalities), self.num_classes)
        available = availability_mask.reshape(batch, self.seq_length, len(self.modalities)).bool()
        if isinstance(self.pcer_router, TemporalBlockEvidenceRouter):
            logits, weights = self.pcer_router(features, evidence, available)
            return {"logits": logits, "weights": weights}
        return self.pcer_router(features, evidence, available)

    def route_pgcd_cached(
        self,
        block_features: torch.Tensor,
        block_evidence_logits: torch.Tensor,
        availability_mask: torch.Tensor,
        *,
        degradation_override: torch.Tensor | None = None,
        use_dynamic: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Rerun PGCD weights on corrupted deployment-time block evidence only."""

        if self.pgcd_router is None:
            raise RuntimeError("route_pgcd_cached requires an enabled PGCD Router.")
        batch, blocks, width = block_features.shape
        expected = self.seq_length * len(self.modalities)
        if (blocks, width) != (expected, self.d_model):
            raise ValueError("Cached PGCD block features must have shape [B,T*M,D].")
        features = block_features.reshape(batch, self.seq_length, len(self.modalities), self.d_model)
        evidence = block_evidence_logits.reshape(batch, self.seq_length, len(self.modalities), self.num_classes)
        available = availability_mask.reshape(batch, self.seq_length, len(self.modalities)).bool()
        routed = self.pgcd_router(
            features,
            evidence,
            available,
            degradation_override=degradation_override,
            use_dynamic=use_dynamic,
        )
        weights = routed["weights"]
        return {
            **routed,
            "fused_features": (weights.unsqueeze(-1) * block_features).sum(dim=1),
            "fused_logits": (weights.unsqueeze(-1) * block_evidence_logits).sum(dim=1),
        }

    def _pgcd_fusion(
        self,
        latent_sequence: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.pgcd_router is None:
            raise RuntimeError("PGCD fusion requires an enabled PGCD Router.")
        batch, timesteps, modalities, width = latent_sequence.shape
        block_features = latent_sequence.reshape(batch, timesteps * modalities, width)
        block_evidence = self.prototype_bank(block_features.reshape(-1, width)).reshape(
            batch, timesteps, modalities, self.num_classes
        )
        routed = self.pgcd_router(latent_sequence, block_evidence, cell_mask)
        block_weights = routed["weights"]
        flat_evidence = block_evidence.reshape(batch, timesteps * modalities, self.num_classes)
        cell_weights = block_weights.reshape(batch, timesteps, modalities)
        modality_weights = cell_weights.sum(dim=1)
        modality_available = cell_mask.any(dim=1)
        modality_logits = modality_weights.clamp_min(torch.finfo(modality_weights.dtype).tiny).log()
        modality_logits = modality_logits.masked_fill(~modality_available, -torch.inf)
        return {
            "pgcd_block_features": block_features,
            "pgcd_block_evidence_logits": flat_evidence,
            "pgcd_block_availability": cell_mask.reshape(batch, -1),
            "pgcd_quality_logits": routed["quality_logits"],
            "pgcd_predicted_degradation": routed["predicted_degradation"],
            "pgcd_predicted_reliability": routed["predicted_reliability"],
            "pgcd_fusion_degradation": routed["fusion_degradation"],
            "pgcd_fusion_reliability": routed["fusion_reliability"],
            "pgcd_block_router_logits": routed["fusion_logits"],
            "pgcd_block_router_weights": block_weights,
            "pgcd_prior_logits": routed["prior_logits"],
            "pgcd_prior_weights": routed["prior_weights"],
            "pgcd_confidence": routed["confidence"],
            "pgcd_entropy": routed["entropy"],
            "pgcd_margin": routed["margin"],
            "pgcd_beta_reliability": routed["beta_reliability"],
            "pgcd_dynamic_enabled": routed["dynamic_enabled"],
            "modality_weights": modality_weights,
            "modality_logits": modality_logits,
            "fused_features": (block_weights.unsqueeze(-1) * block_features).sum(dim=1),
            "fused_logits": (block_weights.unsqueeze(-1) * flat_evidence).sum(dim=1),
        }

    def _pool_temporal_sequence(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.temporal_pooling["type"] == "masked_mean":
            return _masked_mean(sequence, mask), None
        if self.temporal_attention_query is None:
            raise RuntimeError("masked_attention pooling requires a temporal attention query.")
        return _masked_attention(sequence, mask, self.temporal_attention_query)

    def _temporal_pooling_param_count(self) -> int:
        return 0 if self.temporal_attention_query is None else int(self.temporal_attention_query.numel())

    def _gps_encoder_metadata(self) -> dict[str, Any]:
        if "gps" not in self.encoders:
            return {"enabled": False}
        config = self.encoder_configs["gps"]
        encoder = self.encoders["gps"]
        jitter_std = float(
            getattr(encoder, "normalized_feature_jitter_std", config.get("normalized_feature_jitter_std", 0.0))
        )
        return {
            "enabled": True,
            "type": str(config.get("type", "")),
            "output_dim": int(getattr(encoder, "output_dim", config.get("output_dim", self.d_model))),
            "hidden_size": config.get("hidden_size"),
            "dropout": config.get("dropout"),
            "normalized_feature_jitter_std": jitter_std,
            "jitter_mode": "training_only_normalized_features" if jitter_std else "disabled",
        }

    def _encode_sequence(self, modality: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"u_mask_beam_jepa requires {modality}_batch.")
        features = self.encoders[modality](value)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError(f"{modality} encoder must return [B,T,D] or [B,D], got {tuple(features.shape)}.")
        return self.encoder_projections[modality](features)

    def _resolve_modality_mask(
        self,
        missing_mask: torch.Tensor | None,
        available_modalities: torch.Tensor | None,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sequence.shape[0]
        count = len(self.modalities)
        raw = missing_mask if missing_mask is not None else available_modalities
        if raw is None:
            return torch.ones(batch_size, count, dtype=torch.bool, device=sequence.device)
        mask = torch.as_tensor(raw, device=sequence.device, dtype=torch.bool)
        if tuple(mask.shape) == (count,):
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, count):
            raise ValueError(f"missing_mask must have shape {(batch_size, count)}, got {tuple(mask.shape)}.")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("u_mask_beam_jepa requires at least one available modality per sample.")
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
            if tuple(mask.shape) != (batch_size, steps, modalities):
                raise ValueError(
                    "modality_temporal_mask must have shape "
                    f"{(batch_size, steps, modalities)}, got {tuple(mask.shape)}."
                )
        elif temporal_mask is not None:
            time = torch.as_tensor(temporal_mask, device=sequence.device, dtype=torch.bool)
            if tuple(time.shape) != (batch_size, steps):
                raise ValueError(f"temporal_mask must have shape {(batch_size, steps)}, got {tuple(time.shape)}.")
            mask = time.unsqueeze(-1).expand(-1, -1, modalities)
        else:
            mask = torch.ones(batch_size, steps, modalities, dtype=torch.bool, device=sequence.device)
        mask = mask & available.unsqueeze(1)
        if not bool(mask.any(dim=(1, 2)).all().item()):
            raise ValueError("u_mask_beam_jepa requires at least one available temporal cell per sample.")
        return mask

    def _head_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.prototype_bank(features) if self.head_type == "prototype" else self.classifier(features)

    def _router_features(
        self,
        unimodal_logits: torch.Tensor,
        reliability: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        probabilities = torch.softmax(unimodal_logits, dim=-1)
        if self.router_use_reliability_features:
            features.append(reliability)
        if self.router_use_prototype_margin and self.head_type == "prototype":
            top2 = unimodal_logits.topk(min(2, self.num_classes), dim=-1).values
            margin = top2[..., :1] if self.num_classes == 1 else (top2[..., :1] - top2[..., 1:2])
            features.append(margin)
        if self.router_use_entropy:
            entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=-1, keepdim=True)
            features.append(entropy)
        if self.router_use_confidence:
            features.append(probabilities.amax(dim=-1, keepdim=True))
        if self.router_use_logit_norm:
            features.append(unimodal_logits.norm(dim=-1, keepdim=True))
        if not features:
            features.append(available.unsqueeze(-1).to(dtype=unimodal_logits.dtype))
        return torch.cat(features, dim=-1)

    def route_from_features(
        self,
        router_features: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only the supervised Router on precomputed modality features."""
        if router_features.ndim != 3 or router_features.shape[:2] != available.shape:
            raise ValueError("router_features must be [B,M,F] and match available [B,M].")
        eye = torch.eye(
            len(self.modalities), device=router_features.device, dtype=router_features.dtype
        ).unsqueeze(0).expand(router_features.shape[0], -1, -1)
        logits = self.supervised_router(torch.cat([router_features, eye], dim=-1)).squeeze(-1)
        if self.router_pattern_bias is not None:
            logits = logits + self.router_pattern_bias(available.to(dtype=router_features.dtype))
        return logits, _masked_softmax(logits, available)


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=sequence.device, dtype=sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_attention(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("btmd,d->btm", sequence, query) * (sequence.shape[-1] ** -0.5)
    valid = mask.to(device=sequence.device, dtype=scores.dtype)
    weights = torch.softmax(scores.masked_fill(~mask, torch.finfo(scores.dtype).min), dim=1) * valid
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (sequence * weights.unsqueeze(-1)).sum(dim=1), weights


def _reliability_mean_weights(reliability: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
    available = cell_mask.any(dim=1)
    scores = reliability.squeeze(-1) * available.to(dtype=reliability.dtype)
    total = scores.sum(dim=1, keepdim=True)
    fallback = available.to(dtype=reliability.dtype)
    fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.where(total.gt(0), scores / total.clamp_min(torch.finfo(scores.dtype).tiny), fallback)


def _uniform_mean_weights(cell_mask: torch.Tensor) -> torch.Tensor:
    available = cell_mask.any(dim=1).to(dtype=torch.float32)
    return available / available.sum(dim=1, keepdim=True).clamp_min(1.0)


def _temporal_pooling_config(raw: dict[str, Any] | bool | None) -> dict[str, Any]:
    config = dict(raw) if isinstance(raw, dict) else {"enabled": bool(raw)}
    pooling_type = str(config.get("type", "masked_mean")).strip().lower()
    if not bool(config.get("enabled", False)) or pooling_type not in {"masked_mean", "masked_attention"}:
        raise ValueError(
            "T2 requires temporal_pooling.enabled=true and temporal_pooling.type='masked_mean' or 'masked_attention'."
        )
    return {"enabled": True, "type": pooling_type}


def _pcer_config(raw: dict[str, Any] | None, *, seq_length: int) -> dict[str, Any]:
    if raw is None:
        return {
            "enabled": False,
            "mode": "disabled",
            "hidden_dim": 64,
            "embedding_dim": 8,
            "dropout": 0.0,
        }
    if not isinstance(raw, dict):
        raise ValueError("model.primary.pcer must be a mapping when declared.")
    unknown = sorted(set(raw) - {"mode", "hidden_dim", "embedding_dim", "dropout"})
    if unknown:
        raise ValueError(f"Unknown model.primary.pcer fields: {unknown}.")
    mode = str(raw.get("mode", "")).strip().lower()
    if mode not in PCER_MODES:
        raise ValueError(f"model.primary.pcer.mode must be one of {sorted(PCER_MODES)}.")
    if int(seq_length) <= 0:
        raise ValueError("PCER seq_length must be positive.")
    hidden = int(raw.get("hidden_dim", 64))
    embedding = int(raw.get("embedding_dim", 8))
    dropout = float(raw.get("dropout", 0.0))
    if min(hidden, embedding) <= 0 or not 0.0 <= dropout < 1.0:
        raise ValueError("PCER hidden/embedding dimensions must be positive and dropout must be in [0,1).")
    return {
        "enabled": True,
        "mode": mode,
        "hidden_dim": hidden,
        "embedding_dim": embedding,
        "dropout": dropout,
    }


def _pgcd_config(raw: dict[str, Any] | None, *, seq_length: int) -> dict[str, Any]:
    if raw is None:
        return {
            "enabled": False,
            "variant": "disabled",
            "hidden_dim": 64,
            "embedding_dim": 8,
            "dropout": 0.0,
            "beta_reliability_init": 1.0,
        }
    if not isinstance(raw, dict):
        raise ValueError("model.primary.pgcd must be a mapping when declared.")
    unknown = sorted(set(raw) - {"variant", "hidden_dim", "embedding_dim", "dropout", "beta_reliability_init"})
    if unknown:
        raise ValueError(f"Unknown model.primary.pgcd fields: {unknown}.")
    variant = str(raw.get("variant", "")).strip().lower()
    if variant not in PGCD_VARIANTS:
        raise ValueError(f"model.primary.pgcd.variant must be one of {sorted(PGCD_VARIANTS)}.")
    if int(seq_length) <= 0:
        raise ValueError("PGCD seq_length must be positive.")
    hidden = int(raw.get("hidden_dim", 64))
    embedding = int(raw.get("embedding_dim", 8))
    dropout = float(raw.get("dropout", 0.0))
    beta = float(raw.get("beta_reliability_init", 1.0))
    if min(hidden, embedding) <= 0 or not 0.0 <= dropout < 1.0 or beta <= 0.0:
        raise ValueError("PGCD dimensions/beta must be positive and dropout must be in [0,1).")
    return {
        "enabled": True,
        "variant": variant,
        "hidden_dim": hidden,
        "embedding_dim": embedding,
        "dropout": dropout,
        "beta_reliability_init": beta,
    }


def _router_feature_names(
    *,
    reliability: bool,
    prototype_margin: bool,
    entropy: bool,
    confidence: bool,
    logit_norm: bool,
) -> tuple[str, ...]:
    names = []
    if reliability:
        names.append("reliability")
    if prototype_margin:
        names.append("prototype_margin")
    if entropy:
        names.append("entropy")
    if confidence:
        names.append("confidence")
    if logit_norm:
        names.append("logit_norm")
    return tuple(names or ["availability"])


def _validate_modalities(modalities: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(str(item) for item in modalities)
    unknown = sorted(set(values) - set(DEFAULT_MODALITIES))
    if not values or unknown or len(set(values)) != len(values):
        raise ValueError(
            f"u_mask_beam_jepa modalities must be a non-empty unique subset of {list(DEFAULT_MODALITIES)}; got {list(values)}."
        )
    return values
