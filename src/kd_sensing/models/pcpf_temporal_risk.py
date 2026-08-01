from __future__ import annotations

import math
import inspect
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank, beam_topology_positions
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder
from kd_sensing.models.temporal_transformer import SharedTemporalTransformer
from kd_sensing.registries import ENCODERS, MODELS


TRAINING_STAGES = (
    "stage1_expert",
    "stage2_risk",
    "stage3_fusion",
)
FUSION_MODES = (
    "uniform",
    "static_prior",
    "direct_router_control",
    "cuaf_local_adaptation",
    "pcpf_analytic",
)
RISK_COMPONENT_NAMES = ("var", "proto", "temp", "conflict")
PCPF_SPARSE_CSI_MODALITY = "csi"
PROTOCOL_LINEAGE_KEYS = (
    "mode",
    "protocol_id",
    "protocol_fingerprint",
    "audit_id",
    "audit_sha256",
    "split_seed",
    "train_role",
    "validation_role",
    "train_sample_count",
    "validation_sample_count",
    "train_sample_id_hash",
    "validation_sample_id_hash",
)
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "channel",
        "channels",
        "csi",
        "path",
        "paths",
        "beam_power",
        "beam_powers",
        "historical_beam",
        "history_beam",
        "weather",
        "scene",
        "domain",
        "corruption_type",
        "severity",
    }
)


class ProbabilityEmbeddingHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int = 64,
        min_logvar: float = -8.0,
        max_logvar: float = 4.0,
        initial_logvar: float = -4.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        if int(hidden_dim) <= 0 or not self.min_logvar < self.max_logvar:
            raise ValueError("Probability head hidden_dim and logvar bounds are invalid.")
        if not self.min_logvar <= float(initial_logvar) <= self.max_logvar:
            raise ValueError("initial_logvar must be inside the configured clamp interval.")
        self.input_norm = nn.LayerNorm(self.d_model)
        self.delta_mu = nn.Sequential(
            nn.Linear(self.d_model, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.d_model),
        )
        self.logvar_head = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.delta_mu[-1].weight)
        nn.init.zeros_(self.delta_mu[-1].bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, float(initial_logvar))

    def forward(self, features: torch.Tensor, *, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.input_norm(features)
        mu = features + self.delta_mu(normalized)
        logvar = self.logvar_head(normalized).clamp(self.min_logvar, self.max_logvar)
        if sample:
            sampled = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            sampled = mu
        return mu, logvar, sampled


def topology_risk_components(
    *,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    frame_features: torch.Tensor,
    temporal_mask: torch.Tensor,
    probabilities: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_temperature: float,
    topology_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the four PCPF-T risk observations in FP32."""
    if mu.ndim != 3 or logvar.shape != mu.shape:
        raise ValueError("mu and logvar must share shape [B,M,D].")
    if frame_features.ndim != 4 or frame_features.shape[0] != mu.shape[0] or frame_features.shape[2:] != mu.shape[1:]:
        raise ValueError("frame_features must have shape [B,T,M,D].")
    mask = torch.as_tensor(temporal_mask, device=mu.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(frame_features.shape[:3]):
        raise ValueError("temporal_mask must match frame_features [B,T,M].")
    available = mask.any(dim=1)
    if tuple(probabilities.shape[:2]) != tuple(mu.shape[:2]):
        raise ValueError("probabilities must have shape [B,M,C].")

    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu32 = mu.float()
        logvar32 = logvar.float()
        frames32 = frame_features.float()
        probabilities32 = probabilities.float()
        prototypes32 = prototypes.float()
        u_var = torch.exp(logvar32).mean(dim=-1)
        cosine = F.normalize(mu32, dim=-1) @ F.normalize(prototypes32, dim=-1).t()
        u_proto = 1.0 - cosine.amax(dim=-1)

        frame_logits = F.normalize(frames32, dim=-1) @ F.normalize(prototypes32, dim=-1).t() / float(prototype_temperature)
        frame_probability = torch.softmax(frame_logits, dim=-1)
        u_temp, temp_valid, circular_frame_mean = _temporal_circular_residual(
            frame_probability,
            mask,
            topology_positions.to(device=mu.device, dtype=torch.float32),
        )
        u_conflict = _cross_modal_js(probabilities32, available)
        available_float = available.to(dtype=torch.float32)
        components = torch.stack([u_var, u_proto, u_temp, u_conflict], dim=-1)
        components = components * available_float.unsqueeze(-1)
        valid = torch.stack([available, available, temp_valid, available], dim=-1)
    return {
        "components": components,
        "component_valid": valid,
        "temp_valid": temp_valid,
        "circular_frame_mean": circular_frame_mean,
    }


def analytic_fusion_weights(
    *,
    risk: torch.Tensor,
    available: torch.Tensor,
    static_capability: torch.Tensor,
    tau: torch.Tensor | float,
) -> torch.Tensor:
    """Return normalized precision-style weights using a stable FP32 log score."""
    mask = torch.as_tensor(available, device=risk.device, dtype=torch.bool)
    if risk.ndim != 2 or tuple(mask.shape) != tuple(risk.shape):
        raise ValueError("risk and available must have shape [B,M].")
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("Analytic fusion requires at least one available modality per sample.")
    with torch.autocast(device_type=risk.device.type, enabled=False):
        risk32 = risk.float()
        capability32 = torch.as_tensor(static_capability, device=risk.device, dtype=torch.float32)
        if capability32.ndim == 1:
            capability32 = capability32.unsqueeze(0).expand_as(risk32)
        if tuple(capability32.shape) != tuple(risk32.shape) or bool((capability32 <= 0).any().item()):
            raise ValueError("static_capability must be positive and broadcast to [B,M].")
        tau32 = torch.as_tensor(tau, device=risk.device, dtype=torch.float32)
        if bool((tau32 <= 0).any().item()):
            raise ValueError("tau must be positive.")
        log_score = capability32.log() - risk32 / tau32
        weights = _masked_softmax_fp32(log_score, mask)
    return weights


@MODELS.register("pcpf_temporal_risk_fusion")
class PCPFTemporalRiskFusion(nn.Module):
    """Prototype-calibrated probability fusion with constrained temporal risk."""

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
        beam_proto_temperature: float = 0.2,
        training_stage: str = "stage1_expert",
        fusion_mode: str = "uniform",
        temporal_transformer: Mapping[str, Any] | None = None,
        probability_head: Mapping[str, Any] | None = None,
        use_sparse_csi: bool = False,
        sparse_csi_encoder: Mapping[str, Any] | None = None,
        risk: Mapping[str, Any] | None = None,
        fusion: Mapping[str, Any] | None = None,
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
        self.sensing_modalities = normalize_modalities(
            modalities or MODALITY_ORDER,
            context="pcpf_temporal_risk_fusion.modalities",
        )
        if tuple(self.sensing_modalities) != tuple(MODALITY_ORDER):
            raise ValueError(f"PCPF-T requires canonical modalities {list(MODALITY_ORDER)}.")
        self.use_sparse_csi = bool(use_sparse_csi)
        self.modalities = (
            (*self.sensing_modalities, PCPF_SPARSE_CSI_MODALITY)
            if self.use_sparse_csi
            else self.sensing_modalities
        )
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.seq_length = int(seq_length)
        self.training_stage = str(training_stage).strip().lower()
        self.fusion_mode = str(fusion_mode).strip().lower()
        self.consume_missing_modality_metadata = bool(consume_missing_modality_metadata)
        if (self.d_model, self.num_classes, self.num_pred, self.seq_length) != (64, 64, 1, 5):
            raise ValueError("PCPF-T currently requires d_model=64, num_classes=64, num_pred=1, seq_length=5.")
        if self.training_stage not in TRAINING_STAGES:
            raise ValueError(f"training_stage must be one of {list(TRAINING_STAGES)}.")
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {list(FUSION_MODES)}.")
        if float(beam_proto_temperature) <= 0:
            raise ValueError("beam_proto_temperature must be positive.")

        temporal_cfg = _strict_mapping(
            temporal_transformer,
            allowed={
                "num_layers",
                "num_heads",
                "dim_feedforward",
                "dropout",
                "norm_first",
                "causal",
                "adapter_enabled",
            },
            context="model.primary.temporal_transformer",
        )
        probability_cfg = _strict_mapping(
            probability_head,
            allowed={"hidden_dim", "min_logvar", "max_logvar", "initial_logvar"},
            context="model.primary.probability_head",
        )
        sparse_csi_cfg = _strict_mapping(
            sparse_csi_encoder,
            allowed={
                "hidden_dim",
                "num_heads",
                "num_layers",
                "dropout",
                "quality_dim",
                "include_index_embeddings",
                "num_frequency_indices",
                "maximum_time_steps",
            },
            context="model.primary.sparse_csi_encoder",
        )
        if sparse_csi_cfg and not self.use_sparse_csi:
            raise ValueError("model.primary.sparse_csi_encoder requires use_sparse_csi=true.")
        risk_cfg = _strict_mapping(
            risk,
            allowed={"enabled_components", "coefficient_init", "bias_init", "normalization_epsilon"},
            context="model.primary.risk",
        )
        fusion_cfg = _strict_mapping(
            fusion,
            allowed={
                "temperature_min",
                "temperature_init",
                "tau_min",
                "tau_init",
                "eta_init",
                "train_eta",
                "finetune_risk_coefficients",
                "use_static_capability",
                "static_prior_source",
                "direct_router_hidden_dim",
            },
            context="model.primary.fusion",
        )

        encoder_configs = {name: dict((encoders or {}).get(name, {})) for name in self.sensing_modalities}
        missing = [name for name, config in encoder_configs.items() if not config]
        if missing:
            raise ValueError(f"PCPF-T requires encoders for {missing}.")
        defaults = {
            "image": {"image_channels": image_channels},
            "radar": {"radar_channels": radar_channels},
            "gps": {"gps_input_size": gps_input_size},
            "lidar": {"lidar_channels": lidar_channels},
        }
        self.encoder_configs: dict[str, dict[str, Any]] = {}
        self.encoders = nn.ModuleDict()
        self.encoder_projections = nn.ModuleDict()
        for name in self.sensing_modalities:
            config = {**defaults[name], **encoder_configs[name]}
            config.setdefault("output_dim", self.d_model)
            encoder = ENCODERS.build(config)
            output_dim = int(getattr(encoder, "output_dim", config["output_dim"]))
            self.encoders[name] = encoder
            self.encoder_projections[name] = nn.Identity() if output_dim == self.d_model else nn.Linear(output_dim, self.d_model)
            self.encoder_configs[name] = config

        self.csi_encoder: SparsePilotEncoder | None = None
        self.csi_projection: nn.Module | None = None
        if self.use_sparse_csi:
            csi_config = {
                "hidden_dim": 128,
                "num_heads": 4,
                "num_layers": 0,
                "dropout": float(dropout),
                "quality_dim": 16,
                "include_index_embeddings": True,
                "num_frequency_indices": 16,
                "maximum_time_steps": self.seq_length,
                **sparse_csi_cfg,
            }
            if csi_config["include_index_embeddings"] is not True:
                raise ValueError("PCPF sparse CSI requires include_index_embeddings=true.")
            if int(csi_config["num_frequency_indices"]) != 16 or int(csi_config["maximum_time_steps"]) != 5:
                raise ValueError("PCPF sparse CSI requires 16 mother frequency indices and five history steps.")
            self.csi_encoder = SparsePilotEncoder(num_candidate_patterns=32, **csi_config)
            self.csi_projection = nn.Sequential(
                nn.Linear(int(csi_config["hidden_dim"]), self.d_model),
                nn.LayerNorm(self.d_model),
            )
            self.encoder_configs[PCPF_SPARSE_CSI_MODALITY] = {
                "type": "sparse_pilot_encoder",
                **csi_config,
            }

        self.temporal_transformer = SharedTemporalTransformer(
            d_model=self.d_model,
            num_modalities=len(self.modalities),
            seq_length=self.seq_length,
            num_layers=int(temporal_cfg.get("num_layers", 2)),
            num_heads=int(temporal_cfg.get("num_heads", 4)),
            dim_feedforward=int(temporal_cfg.get("dim_feedforward", 128)),
            dropout=float(temporal_cfg.get("dropout", dropout)),
            norm_first=bool(temporal_cfg.get("norm_first", True)),
            causal=bool(temporal_cfg.get("causal", False)),
            adapter_enabled=bool(temporal_cfg.get("adapter_enabled", True)),
        )
        self.prototype_bank = BeamPrototypeBank(
            self.d_model,
            self.num_classes,
            temperature=float(beam_proto_temperature),
        )
        self.probability_head = ProbabilityEmbeddingHead(self.d_model, **probability_cfg)

        enabled_components = risk_cfg.get("enabled_components", list(RISK_COMPONENT_NAMES))
        if not isinstance(enabled_components, (list, tuple)):
            raise ValueError("risk.enabled_components must be a list.")
        enabled_components = tuple(str(value).strip().lower() for value in enabled_components)
        unknown_components = sorted(set(enabled_components) - set(RISK_COMPONENT_NAMES))
        if unknown_components or len(set(enabled_components)) != len(enabled_components):
            raise ValueError(f"risk.enabled_components is invalid: {unknown_components or enabled_components}.")
        self.register_buffer(
            "risk_component_enabled",
            torch.tensor([name in enabled_components for name in RISK_COMPONENT_NAMES], dtype=torch.bool),
        )
        coefficient_init = risk_cfg.get("coefficient_init", [1.0, 1.0, 1.0, 0.25])
        if not isinstance(coefficient_init, (list, tuple)) or len(coefficient_init) != 4:
            raise ValueError("risk.coefficient_init must contain four positive values.")
        coefficient_init = torch.tensor([float(value) for value in coefficient_init], dtype=torch.float32)
        if not bool(torch.isfinite(coefficient_init).all().item()) or bool((coefficient_init <= 0).any().item()):
            raise ValueError("risk.coefficient_init values must be finite and positive.")
        self.risk_coefficient_raw = nn.Parameter(_inverse_softplus(coefficient_init))
        self.risk_bias = nn.Parameter(torch.tensor(float(risk_cfg.get("bias_init", 0.0))))
        self.risk_normalization_epsilon = _positive(
            risk_cfg.get("normalization_epsilon", 0.01),
            "risk.normalization_epsilon",
        )
        self.register_buffer("risk_component_mean", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("risk_component_std", torch.ones(4, dtype=torch.float32))
        self.register_buffer("risk_component_count", torch.zeros(4, dtype=torch.long))
        self.register_buffer("risk_stats_fitted", torch.tensor(False))
        self.register_buffer("mean_train_risk", torch.ones(len(self.modalities), dtype=torch.float32))
        self.register_buffer("mean_train_risk_count", torch.zeros(len(self.modalities), dtype=torch.long))
        self.register_buffer("static_capability_fitted", torch.tensor(False))
        self.register_buffer("train_confidence_p90", torch.ones(len(self.modalities), dtype=torch.float32))
        self.register_buffer("train_confidence_count", torch.zeros(len(self.modalities), dtype=torch.long))

        topology = str(prototype_topology_id).strip().lower()
        if topology == "linear_index_v1":
            raise ValueError("PCPF-T temporal risk requires a circular beam topology.")
        positions = beam_topology_positions(
            self.num_classes,
            topology_id=topology,
            topology_permutation=prototype_topology_permutation,
        )
        self.prototype_topology_id = topology
        self.prototype_topology_permutation = list(prototype_topology_permutation) if prototype_topology_permutation is not None else None
        self.prototype_topology_descriptor_sha256 = str(prototype_topology_descriptor_sha256).strip().lower()
        self.prototype_topology_audit_path = str(prototype_topology_audit_path).strip()
        self.prototype_topology_audit_sha256 = str(prototype_topology_audit_sha256).strip().lower()
        physical_values = (
            self.prototype_topology_descriptor_sha256,
            self.prototype_topology_audit_path,
            self.prototype_topology_audit_sha256,
        )
        if topology == "ula_dft_phase_cycle_v1":
            if not _is_sha256(self.prototype_topology_descriptor_sha256) or not _is_sha256(self.prototype_topology_audit_sha256):
                raise ValueError("ULA-DFT topology requires descriptor and audit SHA256 provenance.")
            if not self.prototype_topology_audit_path:
                raise ValueError("ULA-DFT topology requires an audit path.")
            audit_path = Path(self.prototype_topology_audit_path)
            if not audit_path.is_file():
                raise ValueError(f"ULA-DFT topology audit does not exist: {audit_path}.")
            audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            if audit_sha256 != self.prototype_topology_audit_sha256:
                raise ValueError("ULA-DFT topology audit SHA256 does not match the configured provenance.")
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"ULA-DFT topology audit is unreadable: {audit_path}.") from exc
            descriptor = audit.get("descriptor") if isinstance(audit, Mapping) else None
            if not isinstance(descriptor, Mapping):
                raise ValueError("ULA-DFT topology audit is missing its descriptor.")
            descriptor_sha256 = hashlib.sha256(
                json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                descriptor_sha256 != self.prototype_topology_descriptor_sha256
                or audit.get("descriptor_sha256") != descriptor_sha256
                or descriptor.get("topology_id") != "ula_dft_phase_cycle_v1"
                or descriptor.get("codebook_type") != "ula_dft"
                or int(descriptor.get("num_beams", -1)) != self.num_classes
            ):
                raise ValueError("ULA-DFT topology audit descriptor does not match the configured model.")
        elif any(physical_values):
            raise ValueError(f"Prototype topology {topology!r} does not accept physical audit provenance.")
        self.register_buffer("topology_positions", positions.to(dtype=torch.float32))

        self.temperature_min = _positive(fusion_cfg.get("temperature_min", 0.05), "fusion.temperature_min")
        temperature_init = _positive(fusion_cfg.get("temperature_init", 1.0), "fusion.temperature_init")
        if temperature_init <= self.temperature_min:
            raise ValueError("fusion.temperature_init must exceed temperature_min.")
        self.temperature_raw = nn.Parameter(_inverse_softplus(torch.full((len(self.modalities),), temperature_init - self.temperature_min)))
        self.tau_min = _positive(fusion_cfg.get("tau_min", 0.05), "fusion.tau_min")
        tau_init = _positive(fusion_cfg.get("tau_init", 1.0), "fusion.tau_init")
        if tau_init <= self.tau_min:
            raise ValueError("fusion.tau_init must exceed tau_min.")
        self.tau_raw = nn.Parameter(_inverse_softplus(torch.tensor(tau_init - self.tau_min)))
        eta_init = _positive(fusion_cfg.get("eta_init", 1.0), "fusion.eta_init")
        self.eta_raw = nn.Parameter(_inverse_softplus(torch.tensor(eta_init)))
        self.train_eta = bool(fusion_cfg.get("train_eta", False))
        self.finetune_risk_coefficients = bool(fusion_cfg.get("finetune_risk_coefficients", False))
        self.use_static_capability = bool(fusion_cfg.get("use_static_capability", True))
        self.static_prior_source = str(fusion_cfg.get("static_prior_source", "train_risk")).strip().lower()
        if self.static_prior_source not in {"train_risk", "learned"}:
            raise ValueError("fusion.static_prior_source must be train_risk or learned.")
        self.static_prior_logits = (
            nn.Parameter(torch.zeros(len(self.modalities)))
            if self.fusion_mode == "static_prior" and self.static_prior_source == "learned"
            else None
        )
        self.direct_router = None
        if self.fusion_mode == "direct_router_control":
            hidden = int(fusion_cfg.get("direct_router_hidden_dim", 32))
            if hidden <= 0:
                raise ValueError("fusion.direct_router_hidden_dim must be positive.")
            self.direct_router = nn.Sequential(
                nn.LayerNorm(5),
                nn.Linear(5, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        self._configure_training_stage()

    @property
    def risk_coefficients(self) -> torch.Tensor:
        return F.softplus(self.risk_coefficient_raw.float())

    @property
    def temperatures(self) -> torch.Tensor:
        return self.temperature_min + F.softplus(self.temperature_raw.float())

    @property
    def tau(self) -> torch.Tensor:
        return self.tau_min + F.softplus(self.tau_raw.float())

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.eta_raw.float())

    def forward(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        csi_batch: torch.Tensor | None = None,
        csi_pattern_ids: torch.Tensor | None = None,
        csi_frequency_positions: torch.Tensor | None = None,
        csi_pilot_mask: torch.Tensor | None = None,
        csi_frequency_ids: torch.Tensor | None = None,
        csi_snr_db: torch.Tensor | float | None = None,
        csi_snr_available: torch.Tensor | bool | None = None,
        missing_mask: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        modality_temporal_mask: torch.Tensor | None = None,
        available_modalities: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
        }
        sequences = [self._encode_sequence(name, inputs[name]) for name in self.sensing_modalities]
        csi_diagnostics: dict[str, torch.Tensor] = {}
        intrinsic_masks: list[torch.Tensor] = [
            torch.ones(sequence.shape[:2], device=sequence.device, dtype=torch.bool) for sequence in sequences
        ]
        if self.use_sparse_csi:
            csi_sequence, csi_diagnostics = self._encode_csi_sequence(
                csi_batch,
                pattern_ids=csi_pattern_ids,
                frequency_positions=csi_frequency_positions,
                pilot_mask=csi_pilot_mask,
                frequency_ids=csi_frequency_ids,
                snr_db=csi_snr_db,
                snr_available=csi_snr_available,
            )
            sequences.append(csi_sequence)
            intrinsic_masks.append(csi_diagnostics["csi_frame_available"])
        latent_sequence = torch.stack(sequences, dim=2)
        intrinsic_temporal_mask = torch.stack(intrinsic_masks, dim=2)
        requested = missing_mask if missing_mask is not None else force_modality_mask
        available = self._resolve_modality_mask(requested, available_modalities, latent_sequence)
        cell_mask = self._resolve_temporal_mask(
            latent_sequence,
            available,
            temporal_mask,
            modality_temporal_mask,
        )
        cell_mask = cell_mask & intrinsic_temporal_mask
        if not bool(cell_mask.any(dim=(1, 2)).all().item()):
            raise ValueError("PCPF-T requires at least one intrinsically available temporal cell per sample.")
        temporal = self.temporal_transformer(latent_sequence, cell_mask)
        cls_features = temporal["temporal_cls_features"]
        available = temporal["available_modalities"]

        probability_stage = self.training_stage != "stage1_expert"
        if probability_stage:
            mu, logvar, sampled = self.probability_head(
                cls_features,
                sample=self.training and self.training_stage == "stage2_risk",
            )
        else:
            mu = cls_features
            logvar = torch.full_like(cls_features, -4.0)
            sampled = mu
        available_float = available.unsqueeze(-1).to(dtype=mu.dtype)
        mu = mu * available_float
        sampled = sampled * available_float
        deterministic_logits = self.prototype_bank(mu.reshape(-1, self.d_model)).reshape(
            mu.shape[0], len(self.modalities), self.num_classes
        )
        sampled_logits = self.prototype_bank(sampled.reshape(-1, self.d_model)).reshape_as(deterministic_logits)
        deterministic_probability = torch.softmax(deterministic_logits.float(), dim=-1)
        deterministic_probability = deterministic_probability * available.unsqueeze(-1).to(torch.float32)

        if probability_stage:
            risk_observations = topology_risk_components(
                mu=mu,
                logvar=logvar,
                frame_features=temporal["temporal_token_features"],
                temporal_mask=cell_mask,
                probabilities=deterministic_probability,
                prototypes=self.prototype_bank.prototypes,
                prototype_temperature=self.prototype_bank.temperature,
                topology_positions=self.topology_positions,
            )
            components = risk_observations["components"]
            normalized = (components - self.risk_component_mean.view(1, 1, 4)) / self.risk_component_std.clamp_min(
                self.risk_normalization_epsilon
            ).view(1, 1, 4)
            normalized = normalized * self.risk_component_enabled.view(1, 1, 4).to(torch.float32)
            normalized = normalized * available.unsqueeze(-1).to(torch.float32)
            with torch.autocast(device_type=mu.device.type, enabled=False):
                raw_risk = F.softplus((normalized * self.risk_coefficients.view(1, 1, 4)).sum(dim=-1) + self.risk_bias.float())
                raw_risk = raw_risk * available.to(torch.float32)
        else:
            components = torch.zeros(*mu.shape[:2], 4, device=mu.device, dtype=torch.float32)
            normalized = torch.zeros_like(components)
            raw_risk = torch.zeros(mu.shape[:2], device=mu.device, dtype=torch.float32)
            risk_observations = {
                "component_valid": torch.zeros_like(components, dtype=torch.bool),
                "temp_valid": torch.zeros_like(available),
                "circular_frame_mean": torch.zeros(
                    mu.shape[0], self.seq_length, len(self.modalities), device=mu.device, dtype=torch.float32
                ),
            }

        weights, calibrated_probability, static_capability, effective_fusion_mode = self._fuse(
            deterministic_logits,
            deterministic_probability,
            raw_risk,
            components,
            available,
        )
        fused_probability = (weights.unsqueeze(-1) * calibrated_probability).sum(dim=1)
        fused_probability = fused_probability / fused_probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        fused_features = (weights.to(dtype=mu.dtype).unsqueeze(-1) * mu).sum(dim=1)
        logits = fused_probability.clamp_min(torch.finfo(torch.float32).tiny).log().unsqueeze(1)
        return {
            "logits": logits,
            "input_features": cls_features,
            "output_features": fused_features,
            "modality_features": mu,
            "missing_mask": available,
            "available_modalities": available,
            "modality_temporal_mask": cell_mask,
            "temporal_mask": cell_mask.any(dim=2),
            **temporal,
            "probability_mu": mu,
            "probability_logvar": logvar,
            "unimodal_logits": deterministic_logits,
            "sampled_unimodal_logits": sampled_logits,
            "unimodal_probabilities": deterministic_probability,
            "calibrated_unimodal_probabilities": calibrated_probability,
            "fused_probability": fused_probability,
            "fusion_weights": weights,
            "reliability_fusion_weights": weights,
            "reliability_fusion_mode": effective_fusion_mode,
            "static_capability": static_capability,
            "raw_risk": raw_risk,
            "risk_components": components,
            "normalized_risk_components": normalized,
            "risk_component_valid": risk_observations["component_valid"],
            "risk_u_var": components[..., 0],
            "risk_u_proto": components[..., 1],
            "risk_u_temp": components[..., 2],
            "risk_u_conflict": components[..., 3],
            "risk_temp_valid": risk_observations["temp_valid"],
            "temporal_circular_beam_mean": risk_observations["circular_frame_mean"],
            "risk_coefficients": self.risk_coefficients,
            "modality_temperatures": self.temperatures,
            "fusion_tau": self.tau,
            "static_eta": self.eta,
            "prototype_state": self.prototype_bank.describe(fused_features),
            "metadata": self.training_strategy_metadata(),
            **csi_diagnostics,
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        return {
            "type": "pcpf_temporal_risk_fusion",
            "method": "PCPF-T",
            "architecture_category": "temporal_prototype_calibrated_analytic_fusion",
            "training_stage": self.training_stage,
            "fusion_mode": self.fusion_mode,
            "control_only": self.fusion_mode in {"direct_router_control", "cuaf_local_adaptation"},
            "claim_ineligible": True,
            "outer_test_accessed": False,
            "modalities": list(self.modalities),
            "use_sparse_csi": self.use_sparse_csi,
            "consumes_missing_mask": True,
            "consumes_missing_modality_metadata": self.consume_missing_modality_metadata,
            "cross_modal_attention": False,
            "feature_concatenation_before_risk": False,
            "prototype_bank_count": 1,
            "prototype_topology_id": self.prototype_topology_id,
            "prototype_topology": self.prototype_topology_metadata(),
            "temporal_pooling_type": "shared_temporal_transformer",
            "temporal_pooling_param_count": self.temporal_transformer.parameter_count,
            "encoder_configs": self.encoder_configs,
            "trainable_parameter_names": trainable,
            "trainable_params": int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)),
            "total_params": int(sum(parameter.numel() for parameter in self.parameters())),
            "risk_stats_fitted": bool(self.risk_stats_fitted.item()),
            "static_capability_fitted": bool(self.static_capability_fitted.item()),
            "train_confidence_fitted": bool((self.train_confidence_count > 0).all().item()),
        }

    def prototype_topology_metadata(self) -> dict[str, Any]:
        return {
            "id": self.prototype_topology_id,
            "descriptor_sha256": self.prototype_topology_descriptor_sha256,
            "audit_path": self.prototype_topology_audit_path,
            "audit_sha256": self.prototype_topology_audit_sha256,
            "formal_r0_r7_eligible": self.prototype_topology_id == "ula_dft_phase_cycle_v1",
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        metadata = self.training_strategy_metadata()
        metadata.update(
            {
                "expert_fingerprint": self._expert_fingerprint(),
                "risk_component_mean": self.risk_component_mean.detach().cpu().tolist(),
                "risk_component_std": self.risk_component_std.detach().cpu().tolist(),
                "risk_component_count": self.risk_component_count.detach().cpu().tolist(),
                "mean_train_risk": self.mean_train_risk.detach().cpu().tolist(),
                "mean_train_risk_count": self.mean_train_risk_count.detach().cpu().tolist(),
                "train_confidence_p90": self.train_confidence_p90.detach().cpu().tolist(),
                "train_confidence_count": self.train_confidence_count.detach().cpu().tolist(),
            }
        )
        return metadata

    def compute_validation_loss(self, model_output: Any, labels: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
        from kd_sensing.losses.pcpf_temporal_risk import pcpf_temporal_risk_loss
        from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config

        output = {
            "logits": model_output.logits,
            "input_features": model_output.input_features,
            "output_features": model_output.output_features,
            **model_output.diagnostics,
        }
        return pcpf_temporal_risk_loss(
            output,
            labels,
            prototype_bank=self.prototype_bank,
            config=pcpf_temporal_risk_config(cfg),
        )["loss"]

    def prepare_training_stage(
        self,
        *,
        cfg: dict[str, Any],
        train_loader: Any,
        device: torch.device,
        run_dir: Path,
        non_blocking: bool,
    ) -> dict[str, Any]:
        from torch.utils.data import DataLoader, IterableDataset

        from kd_sensing.data.temporal_missing import apply_training_temporal_missing
        from kd_sensing.engine.runtime import prepare_task_batch, prepare_task_labels, run_model_step
        from kd_sensing.losses.pcpf_temporal_risk import topology_risk_target
        from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config

        config = pcpf_temporal_risk_config(cfg)
        if self.training_stage == "stage1_expert":
            return {"status": "not_required", "training_stage": self.training_stage}
        if bool(cfg.get("training", {}).get("resume", False)):
            return {"status": "deferred_to_resume_checkpoint", "training_stage": self.training_stage}
        if train_loader is None or not hasattr(train_loader, "dataset"):
            raise ValueError("PCPF-T stage preparation requires the train dataloader only.")
        if isinstance(train_loader.dataset, IterableDataset):
            raise ValueError("PCPF-T train-only statistics require a finite map-style train dataset.")
        if not config["stage_preparation"]["enabled"]:
            raise ValueError(f"{self.training_stage} requires stage preparation.")
        if self.training_stage == "stage3_fusion":
            if not bool(self.risk_stats_fitted.item()):
                raise ValueError("Stage 3 requires fitted Stage 2 risk normalization in its checkpoint.")
            self._validate_stage2_gate_binding(cfg)

        batch_size = int(getattr(train_loader, "batch_size", 0) or cfg["data"]["dataloader"]["train_batch_size"])
        workers = int(getattr(train_loader, "num_workers", 0))
        loader_options: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": bool(getattr(train_loader, "pin_memory", False)),
            "collate_fn": getattr(train_loader, "collate_fn", None),
            "drop_last": False,
            "worker_init_fn": getattr(train_loader, "worker_init_fn", None),
        }
        if workers:
            loader_options["prefetch_factor"] = getattr(train_loader, "prefetch_factor", None) or 2
        sequential = DataLoader(train_loader.dataset, **loader_options)
        max_batches = config["stage_preparation"]["max_batches"]
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        module_states = [(module, module.training) for module in self.modules()]
        component_sum = torch.zeros(4, device=device, dtype=torch.float64)
        component_square_sum = torch.zeros(4, device=device, dtype=torch.float64)
        component_count = torch.zeros(4, device=device, dtype=torch.long)
        risk_sum = torch.zeros(len(self.modalities), device=device, dtype=torch.float64)
        risk_count = torch.zeros(len(self.modalities), device=device, dtype=torch.long)
        confidence_values: list[list[torch.Tensor]] = [[] for _ in self.modalities]
        sample_count = 0
        batch_count = 0
        try:
            self.eval()
            with torch.no_grad():
                for step_index, raw_batch in enumerate(sequential):
                    if max_batches is not None and step_index >= max_batches:
                        break
                    batch = apply_training_temporal_missing(prepare_task_batch(raw_batch), cfg, epoch=0, step=step_index)
                    labels = prepare_task_labels(
                        batch,
                        num_pred=self.num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    step = run_model_step(
                        self,
                        cfg["experiment"].get("task", "fusion"),
                        batch,
                        seq_length=self.seq_length,
                        num_pred=self.num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    diagnostics = step.model_output.diagnostics
                    available = diagnostics["available_modalities"].to(device=device, dtype=torch.bool)
                    valid_label = labels[:, 0].ne(-100)
                    confidence = diagnostics["unimodal_probabilities"].amax(dim=-1)
                    confidence_valid = available & valid_label.unsqueeze(1)
                    for index in range(len(self.modalities)):
                        selected = confidence[:, index][confidence_valid[:, index]]
                        if selected.numel():
                            confidence_values[index].append(selected.detach().float().cpu())
                    if self.training_stage == "stage2_risk":
                        components = diagnostics["risk_components"].to(dtype=torch.float64)
                        valid = diagnostics["risk_component_valid"].to(device=device, dtype=torch.bool)
                        valid = valid & valid_label.view(-1, 1, 1)
                        for index in range(4):
                            values = components[..., index][valid[..., index]]
                            component_sum[index] += values.sum()
                            component_square_sum[index] += values.square().sum()
                            component_count[index] += values.numel()
                    else:
                        probability = diagnostics["unimodal_probabilities"]
                        target = topology_risk_target(
                            probability,
                            labels,
                            available,
                            topology_id=self.prototype_topology_id,
                            topology_permutation=self.prototype_topology_permutation,
                        )
                        valid = available & valid_label.unsqueeze(1)
                        risk_sum += (target.to(torch.float64) * valid).sum(dim=0)
                        risk_count += valid.sum(dim=0)
                    sample_count += int(labels.shape[0])
                    batch_count += 1
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            for module, was_training in module_states:
                module.training = was_training

        if batch_count == 0:
            raise ValueError("PCPF-T stage preparation observed zero train batches.")
        confidence_count = torch.tensor(
            [sum(int(chunk.numel()) for chunk in chunks) for chunks in confidence_values],
            device=self.train_confidence_count.device,
            dtype=torch.long,
        )
        if bool((confidence_count == 0).any().item()):
            raise ValueError(f"PCPF-T train confidence fit has an empty modality: {confidence_count.tolist()}.")
        confidence_p90 = torch.stack([torch.quantile(torch.cat(chunks), 0.9) for chunks in confidence_values]).to(
            device=self.train_confidence_p90.device, dtype=torch.float32
        )
        self.train_confidence_p90.copy_(confidence_p90)
        self.train_confidence_count.copy_(confidence_count)
        if self.training_stage == "stage2_risk":
            if bool((component_count == 0).any().item()):
                raise ValueError(f"PCPF-T risk normalization has empty components: {component_count.tolist()}.")
            mean = component_sum / component_count.to(torch.float64)
            variance = component_square_sum / component_count.to(torch.float64) - mean.square()
            std = variance.clamp_min(self.risk_normalization_epsilon**2).sqrt()
            self.risk_component_mean.copy_(mean.to(torch.float32))
            self.risk_component_std.copy_(std.to(torch.float32))
            self.risk_component_count.copy_(component_count)
            self.risk_stats_fitted.fill_(True)
        else:
            if bool((risk_count == 0).any().item()):
                raise ValueError(f"PCPF-T static capability has an empty modality: {risk_count.tolist()}.")
            self.mean_train_risk.copy_((risk_sum / risk_count.to(torch.float64)).to(torch.float32))
            self.mean_train_risk_count.copy_(risk_count)
            self.static_capability_fitted.fill_(True)

        protocol = cfg.get("data_protocol")
        source_split = str(protocol.get("train_role", "train")) if isinstance(protocol, Mapping) else "train"
        payload = {
            "status": "fitted",
            "training_stage": self.training_stage,
            "source_split": source_split,
            "sample_count": sample_count,
            "batch_count": batch_count,
            "bounded_smoke_pass": max_batches is not None,
            "claim_ineligible": True,
            "outer_test_accessed": False,
            "protocol": dict(protocol) if isinstance(protocol, Mapping) else {},
            "risk_component_mean": self.risk_component_mean.detach().cpu().tolist(),
            "risk_component_std": self.risk_component_std.detach().cpu().tolist(),
            "risk_component_count": self.risk_component_count.detach().cpu().tolist(),
            "mean_train_risk": self.mean_train_risk.detach().cpu().tolist(),
            "mean_train_risk_count": self.mean_train_risk_count.detach().cpu().tolist(),
            "train_confidence_p90": self.train_confidence_p90.detach().cpu().tolist(),
            "train_confidence_count": self.train_confidence_count.detach().cpu().tolist(),
        }
        statistics_path = Path(run_dir) / "pcpf_stage_statistics.json"
        statistics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**payload, "path": str(statistics_path)}

    def _validate_stage2_gate_binding(self, cfg: dict[str, Any]) -> None:
        from kd_sensing.utils.checkpoint import checkpoint_file_digest
        from kd_sensing.utils.paths import resolve_path

        gate = cfg.get("training", {}).get("pcpf_stage2_gate")
        if not isinstance(gate, Mapping) or gate.get("stage2_gate_passed") is not True:
            raise ValueError("Stage 3 requires an explicitly passed Stage 2 gate binding.")
        report_path = resolve_path(gate.get("report_path"))
        if report_path is None or not report_path.is_file():
            raise ValueError(f"Stage 2 gate report does not exist: {report_path}.")
        actual_sha256, _ = checkpoint_file_digest(report_path)
        if actual_sha256 != str(gate.get("sha256", "")).strip().lower():
            raise ValueError("Stage 2 gate report SHA256 does not match the resolved config.")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Stage 2 gate report is unreadable.") from exc
        if not isinstance(report, Mapping) or report.get("stage2_gate_passed") is not True:
            raise ValueError("Stage 2 gate report did not pass.")
        if report.get("claim_ineligible") is not True or report.get("outer_test_accessed") is not False:
            raise ValueError("Stage 2 gate report has invalid claim or outer-test provenance.")
        if report.get("source_training_stage") != "stage2_risk":
            raise ValueError("Stage 2 gate report has an invalid source training stage.")
        if report.get("bounded_evaluation") is not False:
            raise ValueError("Stage 3 refuses a bounded Stage 2 gate report.")
        if report.get("expert_fingerprint") != self._expert_fingerprint():
            raise ValueError("Stage 2 gate report expert fingerprint does not match the initialized model.")
        if _topology_identity(report.get("prototype_topology")) != _topology_identity(self.prototype_topology_metadata()):
            raise ValueError("Stage 2 gate report topology provenance does not match the initialized model.")
        protocol = cfg.get("data_protocol")
        report_protocol = report.get("data_protocol")
        if not isinstance(protocol, Mapping) or not isinstance(report_protocol, Mapping):
            raise ValueError("Stage 2 gate report is missing data protocol provenance.")
        if any(report_protocol.get(key) != protocol.get(key) for key in PROTOCOL_LINEAGE_KEYS):
            raise ValueError("Stage 2 gate report data protocol does not match Stage 3.")
        if (
            report.get("source_split") != protocol.get("validation_role")
            or report.get("train_confidence_source_split") != protocol.get("train_role")
            or int(report.get("experiment_seed", -1)) != int(cfg.get("experiment", {}).get("seed", -2))
        ):
            raise ValueError("Stage 2 gate report split or seed lineage does not match Stage 3.")
        identity = report.get("validation_identity")
        if not isinstance(identity, Mapping) or (
            int(identity.get("sample_count", -1)) != int(protocol.get("validation_sample_count", -2))
            or identity.get("protocol_sample_id_sha256") != protocol.get("validation_sample_id_hash")
            or identity.get("bound_sample_id_sha256") != protocol.get("validation_sample_id_hash")
        ):
            raise ValueError("Stage 2 gate report validation identity does not match Stage 3.")
        initialization = cfg.get("training", {}).get("initialization_checkpoint")
        if not isinstance(initialization, Mapping) or report.get("stage2_checkpoint_sha256") != initialization.get("sha256"):
            raise ValueError("Stage 2 gate report checkpoint does not match Stage 3 initialization.")

    def _expert_fingerprint(self) -> str:
        digest = hashlib.sha256()
        prefixes = (
            "encoders.",
            "encoder_projections.",
            "csi_encoder.",
            "csi_projection.",
            "temporal_transformer.",
            "prototype_bank.",
        )
        for name, tensor in sorted(self.state_dict().items()):
            if not name.startswith(prefixes):
                continue
            value = tensor.detach().cpu().contiguous().reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def validation_best_alias(self) -> str:
        return {
            "stage1_expert": "stage1_best.pth",
            "stage2_risk": "stage2_best.pth",
            "stage3_fusion": "stage3_best.pth",
        }[self.training_stage]

    def assert_trainable_parameters(self) -> None:
        actual = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if actual != self._expected_trainable_names:
            missing = sorted(self._expected_trainable_names - actual)
            extra = sorted(actual - self._expected_trainable_names)
            raise RuntimeError(f"PCPF-T stage freeze mismatch: missing={missing}, extra={extra}.")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.modules():
                parameters = tuple(module.parameters(recurse=True))
                if parameters and not any(parameter.requires_grad for parameter in parameters):
                    module.eval()
        return self
    def _configure_training_stage(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.training_stage == "stage1_expert":
            _set_module_trainable(self.encoders)
            _set_module_trainable(self.encoder_projections)
            if self.csi_encoder is not None and self.csi_projection is not None:
                _set_module_trainable(self.csi_encoder)
                for parameter in self.csi_encoder.quality_projection.parameters():
                    parameter.requires_grad_(False)
                _set_module_trainable(self.csi_projection)
            _set_module_trainable(self.temporal_transformer)
            _set_module_trainable(self.prototype_bank)
            if self.static_prior_logits is not None:
                self.static_prior_logits.requires_grad_(True)
        elif self.training_stage == "stage2_risk":
            _set_module_trainable(self.probability_head)
            self.risk_coefficient_raw.requires_grad_(True)
            self.risk_bias.requires_grad_(True)
        else:
            self.temperature_raw.requires_grad_(True)
            if self.fusion_mode in {"pcpf_analytic", "cuaf_local_adaptation"}:
                self.tau_raw.requires_grad_(True)
            if self.train_eta and self.use_static_capability:
                self.eta_raw.requires_grad_(True)
            if self.fusion_mode == "direct_router_control":
                assert self.direct_router is not None
                _set_module_trainable(self.direct_router)
            if self.static_prior_logits is not None:
                self.static_prior_logits.requires_grad_(True)
            if self.finetune_risk_coefficients:
                self.risk_coefficient_raw.requires_grad_(True)
                self.risk_bias.requires_grad_(True)
        self._expected_trainable_names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        self.assert_trainable_parameters()

    def _fuse(
        self,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        risk: torch.Tensor,
        components: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        with torch.autocast(device_type=logits.device.type, enabled=False):
            temperatures = self.temperatures.view(1, -1, 1)
            calibrated = torch.softmax(logits.float() / temperatures, dim=-1)
            calibrated = calibrated * available.unsqueeze(-1).to(torch.float32)
            capability = self._static_capability().to(device=logits.device)
            effective_mode = self.fusion_mode
            if self.training_stage == "stage2_risk":
                effective_mode = "uniform"
            elif self.training_stage == "stage1_expert" and self.fusion_mode != "static_prior":
                effective_mode = "uniform"
            if effective_mode == "uniform":
                weights = _masked_softmax_fp32(torch.zeros_like(risk, dtype=torch.float32), available)
            elif effective_mode == "static_prior":
                if self.static_prior_logits is not None:
                    score = self.static_prior_logits.float().unsqueeze(0).expand_as(risk)
                else:
                    score = capability.clamp_min(1e-12).log().unsqueeze(0).expand_as(risk)
                weights = _masked_softmax_fp32(score, available)
            elif effective_mode == "pcpf_analytic":
                weights = analytic_fusion_weights(
                    risk=risk,
                    available=available,
                    static_capability=capability,
                    tau=self.tau,
                )
            elif effective_mode == "cuaf_local_adaptation":
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1) / math.log(self.num_classes)
                top2 = probabilities.topk(2, dim=-1).values
                margin_risk = 1.0 - (top2[..., 0] - top2[..., 1])
                local_risk = entropy + margin_risk + 0.25 * components[..., 3]
                weights = analytic_fusion_weights(
                    risk=local_risk,
                    available=available,
                    static_capability=capability,
                    tau=self.tau,
                )
            elif effective_mode == "direct_router_control":
                if self.direct_router is None:
                    raise RuntimeError("direct_router_control was selected without its control module.")
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
                top2 = probabilities.topk(2, dim=-1).values
                margin = top2[..., :1] - top2[..., 1:2]
                confidence = top2[..., :1]
                norm = logits.float().norm(dim=-1, keepdim=True)
                features = torch.cat([components[..., 1:2], entropy, margin, confidence, norm], dim=-1)
                router_logits = self.direct_router(features).squeeze(-1)
                weights = _masked_softmax_fp32(router_logits.float(), available)
            else:  # pragma: no cover - constructor validates the mode.
                raise RuntimeError(f"Unsupported fusion mode {effective_mode!r}.")
        return weights, calibrated, capability, effective_mode

    def _static_capability(self) -> torch.Tensor:
        if not self.use_static_capability:
            return torch.ones_like(self.mean_train_risk, dtype=torch.float32)
        return torch.exp(-self.eta * self.mean_train_risk.float())

    def _encode_sequence(self, modality: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"PCPF-T requires {modality}_batch.")
        features = self.encoders[modality](value)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError(f"{modality} encoder must return [B,T,D], got {tuple(features.shape)}.")
        features = self.encoder_projections[modality](features)
        if tuple(features.shape[1:]) != (self.seq_length, self.d_model):
            raise ValueError(f"{modality} encoder must return [B,{self.seq_length},{self.d_model}], got {tuple(features.shape)}.")
        return features

    def _encode_csi_sequence(
        self,
        value: torch.Tensor | None,
        *,
        pattern_ids: torch.Tensor | None,
        frequency_positions: torch.Tensor | None,
        pilot_mask: torch.Tensor | None,
        frequency_ids: torch.Tensor | None,
        snr_db: torch.Tensor | float | None,
        snr_available: torch.Tensor | bool | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.csi_encoder is None or self.csi_projection is None:
            raise RuntimeError("PCPF sparse CSI encoder is not initialized.")
        if value is None or pattern_ids is None or frequency_positions is None or pilot_mask is None or frequency_ids is None:
            raise ValueError("PCPF sparse CSI requires observations, pattern/frequency ids, positions, and pilot mask.")
        observations = torch.as_tensor(value)
        if not torch.is_complex(observations) or observations.ndim != 4:
            raise ValueError("csi_batch must be complex [B,5,2,2].")
        batch_size, steps, patterns, frequencies = observations.shape
        if (steps, patterns, frequencies) != (self.seq_length, 2, 2):
            raise ValueError(f"csi_batch must have shape [B,{self.seq_length},2,2].")
        ids = torch.as_tensor(pattern_ids, device=observations.device, dtype=torch.long)
        valid = torch.as_tensor(pilot_mask, device=observations.device, dtype=torch.bool)
        if tuple(ids.shape) != (batch_size, steps, patterns) or tuple(valid.shape) != tuple(observations.shape):
            raise ValueError("csi_pattern_ids and csi_pilot_mask must have shapes [B,5,2] and [B,5,2,2].")

        positions = _repeat_csi_index_values(
            frequency_positions,
            batch_size=batch_size,
            steps=steps,
            width=frequencies,
            device=observations.device,
            dtype=observations.real.dtype,
            field="csi_frequency_positions",
        )
        frequency_index = _repeat_csi_index_values(
            frequency_ids,
            batch_size=batch_size,
            steps=steps,
            width=frequencies,
            device=observations.device,
            dtype=torch.long,
            field="csi_frequency_ids",
        )
        flattened_snr: torch.Tensor | float | None = snr_db
        if snr_available is not None:
            availability = torch.as_tensor(snr_available, device=observations.device, dtype=torch.bool)
            if availability.numel() == 1:
                availability = availability.reshape(1, 1).expand(batch_size, steps)
            elif tuple(availability.shape) == (batch_size,):
                availability = availability[:, None].expand(-1, steps)
            elif tuple(availability.shape) != (batch_size, steps):
                raise ValueError("csi_snr_available must be scalar, [B], or [B,5].")
            if snr_db is None and bool(availability.any().item()):
                raise ValueError("csi_snr_available=true requires real csi_snr_db observations.")
            if snr_db is not None and not bool(availability.all().item()):
                raise ValueError("Partial CSI SNR availability is unsupported without per-frame nullable values.")
        if torch.is_tensor(snr_db):
            snr = torch.as_tensor(snr_db, device=observations.device, dtype=observations.real.dtype)
            if tuple(snr.shape) == (batch_size,):
                snr = snr[:, None].expand(-1, steps)
            if tuple(snr.shape) != (batch_size, steps):
                raise ValueError("csi_snr_db must be scalar, [B], or [B,5].")
            flattened_snr = snr.reshape(-1)
        encoded = self.csi_encoder(
            observations.reshape(batch_size * steps, patterns, frequencies),
            ids.reshape(batch_size * steps, patterns),
            positions,
            valid.reshape(batch_size * steps, patterns, frequencies),
            flattened_snr,
            frequency_ids=frequency_index,
            time_ids=torch.arange(steps, device=observations.device).view(1, steps).expand(batch_size, -1).reshape(-1),
        )
        frame_available = encoded["csi_available"].reshape(batch_size, steps)
        features = self.csi_projection(encoded["csi_feature"]).reshape(batch_size, steps, self.d_model)
        features = features * frame_available.unsqueeze(-1).to(dtype=features.dtype)
        return features, {
            "csi_frame_available": frame_available,
            "csi_quality": encoded["csi_quality"].reshape(batch_size, steps, -1),
            "csi_quality_confidence": encoded["quality_confidence"].reshape(batch_size, steps),
            "csi_valid_ratio": encoded["valid_ratio"].reshape(batch_size, steps),
            "csi_log_rms": encoded["log_rms"].reshape(batch_size, steps),
            "csi_snr_available": encoded["snr_available"].reshape(batch_size, steps),
        }

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
            raise ValueError(f"PCPF-T modality mask must have shape {(batch_size, count)}.")
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError("PCPF-T requires at least one available modality per sample.")
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
            raise ValueError("PCPF-T requires at least one available temporal cell per sample.")
        return mask


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _topology_identity(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        return "", "", ""
    return tuple(str(value.get(key, "")) for key in ("id", "descriptor_sha256", "audit_sha256"))


def _repeat_csi_index_values(
    value: torch.Tensor,
    *,
    batch_size: int,
    steps: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    field: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(tensor.shape) == (width,):
        return tensor.view(1, width).expand(batch_size * steps, -1)
    if tuple(tensor.shape) == (batch_size, width):
        return tensor[:, None, :].expand(-1, steps, -1).reshape(batch_size * steps, width)
    if tuple(tensor.shape) == (batch_size, steps, width):
        return tensor.reshape(batch_size * steps, width)
    raise ValueError(f"{field} must have shape [2], [B,2], or [B,5,2].")


def _temporal_circular_residual(
    probability: torch.Tensor,
    temporal_mask: torch.Tensor,
    topology_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beams = int(probability.shape[-1])
    angles = 2.0 * math.pi * topology_positions / float(beams)
    sin_mean = (probability * angles.sin().view(1, 1, 1, beams)).sum(dim=-1)
    cos_mean = (probability * angles.cos().view(1, 1, 1, beams)).sum(dim=-1)
    circular_mean = torch.atan2(sin_mean, cos_mean)
    batch, steps, modalities = temporal_mask.shape
    residual = torch.zeros(batch, modalities, device=probability.device, dtype=torch.float32)
    valid = temporal_mask.sum(dim=1) >= 3
    times = torch.arange(steps, device=probability.device, dtype=torch.float32)
    for batch_index in range(batch):
        for modality_index in range(modalities):
            selected = temporal_mask[batch_index, :, modality_index]
            if int(selected.sum().item()) < 3:
                continue
            theta = circular_mean[batch_index, selected, modality_index]
            delta = torch.atan2(torch.sin(theta[1:] - theta[:-1]), torch.cos(theta[1:] - theta[:-1]))
            unwrapped = torch.cat([theta[:1], theta[:1] + delta.cumsum(dim=0)])
            selected_times = times[selected]
            centered_time = selected_times - selected_times.mean()
            centered_angle = unwrapped - unwrapped.mean()
            slope = (centered_time * centered_angle).sum() / centered_time.square().sum().clamp_min(1e-12)
            prediction = unwrapped.mean() + slope * centered_time
            error = torch.atan2(torch.sin(unwrapped - prediction), torch.cos(unwrapped - prediction))
            residual[batch_index, modality_index] = error.square().mean() / (math.pi**2)
    return residual, valid, circular_mean * temporal_mask.to(torch.float32)


def _cross_modal_js(probability: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    available_float = available.to(dtype=torch.float32)
    counts = available_float.sum(dim=1, keepdim=True)
    others = ((probability * available_float.unsqueeze(-1)).sum(dim=1, keepdim=True) - probability) / (
        counts.unsqueeze(-1) - 1.0
    ).clamp_min(1.0)
    midpoint = 0.5 * (probability + others)
    p = probability.clamp_min(1e-12)
    q = others.clamp_min(1e-12)
    middle = midpoint.clamp_min(1e-12)
    divergence = 0.5 * ((p * (p.log() - middle.log())).sum(dim=-1) + (q * (q.log() - middle.log())).sum(dim=-1))
    valid = available & counts.gt(1)
    return divergence * valid.to(torch.float32)


def _masked_softmax_fp32(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(available, device=logits.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(logits.shape):
        raise ValueError("Fusion logits and availability mask must share shape [B,M].")
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError("Fusion requires at least one available modality per sample.")
    masked = logits.float().masked_fill(~mask, -torch.inf)
    return torch.softmax(masked, dim=1) * mask.to(torch.float32)


def _set_module_trainable(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


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


def _positive(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive.")
    return result


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def validate_pcpf_model_config(primary: Mapping[str, Any], dataset: Mapping[str, Any]) -> None:
    """Validate the strict model surface before any dataset or model is built."""
    signature = inspect.signature(PCPFTemporalRiskFusion)
    allowed = {name for name in signature.parameters if name != "self"} | {"type"}
    unknown = sorted(set(primary) - allowed)
    if unknown:
        raise ValueError(f"model.primary contains unsupported PCPF-T fields: {unknown}.")
    forbidden = sorted(_find_forbidden_keys(primary))
    if forbidden:
        raise ValueError(f"PCPF-T model config contains forbidden sensing inputs: {forbidden}.")
    dimensions = (
        int(primary.get("d_model", 64)),
        int(primary.get("num_classes", 64)),
        int(primary.get("num_pred", 1)),
        int(primary.get("seq_length", 5)),
    )
    if dimensions != (64, 64, 1, 5):
        raise ValueError("PCPF-T requires d_model=64, num_classes=64, num_pred=1, seq_length=5.")
    if int(dataset.get("seq_len", 0)) != dimensions[3] or int(dataset.get("num_pred", 0)) != dimensions[2]:
        raise ValueError("PCPF-T model sequence/prediction lengths must match data.dataset.")
    temporal = _strict_mapping(
        primary.get("temporal_transformer"),
        allowed={
            "num_layers",
            "num_heads",
            "dim_feedforward",
            "dropout",
            "norm_first",
            "causal",
            "adapter_enabled",
        },
        context="model.primary.temporal_transformer",
    )
    heads = int(temporal.get("num_heads", 4))
    if heads <= 0 or dimensions[0] % heads:
        raise ValueError("PCPF-T d_model must be divisible by temporal_transformer.num_heads.")
    if bool(temporal.get("causal", False)):
        raise ValueError("PCPF-T temporal_transformer.causal must be false.")
    _strict_mapping(
        primary.get("probability_head"),
        allowed={"hidden_dim", "min_logvar", "max_logvar", "initial_logvar"},
        context="model.primary.probability_head",
    )
    sparse_csi = _strict_mapping(
        primary.get("sparse_csi_encoder"),
        allowed={
            "hidden_dim",
            "num_heads",
            "num_layers",
            "dropout",
            "quality_dim",
            "include_index_embeddings",
            "num_frequency_indices",
            "maximum_time_steps",
        },
        context="model.primary.sparse_csi_encoder",
    )
    use_sparse_csi = bool(primary.get("use_sparse_csi", False))
    if sparse_csi and not use_sparse_csi:
        raise ValueError("model.primary.sparse_csi_encoder requires use_sparse_csi=true.")
    dataset_sparse_csi = dataset.get("sparse_csi")
    if use_sparse_csi:
        if not isinstance(dataset_sparse_csi, Mapping):
            raise ValueError("PCPF use_sparse_csi=true requires data.dataset.sparse_csi.")
        from kd_sensing.data.pcpf_sparse_csi import PCPFSparseCSISidecar

        PCPFSparseCSISidecar(dataset_sparse_csi)
    elif dataset_sparse_csi is not None:
        raise ValueError("data.dataset.sparse_csi requires model.primary.use_sparse_csi=true.")
    _strict_mapping(
        primary.get("risk"),
        allowed={"enabled_components", "coefficient_init", "bias_init", "normalization_epsilon"},
        context="model.primary.risk",
    )
    fusion = _strict_mapping(
        primary.get("fusion"),
        allowed={
            "temperature_min",
            "temperature_init",
            "tau_min",
            "tau_init",
            "eta_init",
            "train_eta",
            "finetune_risk_coefficients",
            "use_static_capability",
            "static_prior_source",
            "direct_router_hidden_dim",
        },
        context="model.primary.fusion",
    )
    for field in ("temperature_min", "temperature_init", "tau_min", "tau_init", "eta_init"):
        if field in fusion:
            _positive(fusion[field], f"fusion.{field}")
    stage = str(primary.get("training_stage", "")).strip().lower()
    mode = str(primary.get("fusion_mode", "")).strip().lower()
    if stage not in TRAINING_STAGES or mode not in FUSION_MODES:
        raise ValueError("PCPF-T training_stage or fusion_mode is invalid.")


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_INPUT_KEYS:
                found.add(normalized)
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found


__all__ = [
    "FUSION_MODES",
    "PCPFTemporalRiskFusion",
    "ProbabilityEmbeddingHead",
    "RISK_COMPONENT_NAMES",
    "TRAINING_STAGES",
    "analytic_fusion_weights",
    "topology_risk_components",
    "validate_pcpf_model_config",
]
