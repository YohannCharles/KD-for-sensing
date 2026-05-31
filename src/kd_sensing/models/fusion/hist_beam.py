from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.evaluation.hist_beam_residuals import residual_logits_to_absolute_logits
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.models.image import ImageFeatureExtractor
from kd_sensing.models.image_encoders import ResNet18ImageEncoder
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MMWAVE_INPUT_SIZE, MmWaveFeatureExtractor
from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.registries import MODELS


HIST_BEAM_VARIANTS = {
    "v0_flat",
    "flat",
    "v1_hierarchical",
    "hierarchical",
    "v2_shared_private",
    "shared_private",
    "v3_decoupled",
    "decoupled",
    "v4_adapter",
    "adapter",
    "v5_adapter_proto",
    "adapter_proto",
    "v6_radio_proto",
    "adapter_radio_proto",
    "v8_path_proto",
    "adapter_path_proto",
    "v8_target_prior_head",
    "v9_input_conditioned_target_adaptation",
    "v7_shared_physical_private_residual",
    "shared_physical_private_residual",
    "v6_full_finetune",
    "full_finetune",
}
DEFAULT_HIST_MODALITIES = ("image", "radar", "gps")


@dataclass(frozen=True)
class HistBeamConfig:
    num_classes: int = 64
    group_size: int = 8
    variant: str = "v3_decoupled"
    modalities: tuple[str, ...] = DEFAULT_HIST_MODALITIES
    lambda_hier: float = 1.0
    lambda_flat: float = 0.2
    lambda_orth: float = 0.01
    lambda_scene_c: float = 0.05
    lambda_scene_s: float = 0.05
    adapter_enabled: bool = False
    prototype_enabled: bool = False
    geometry_aware: bool = False
    geometry_fields: tuple[str, ...] = ()
    coarse_conditioned_adapter: bool = False
    radio_semantic_enabled: bool = False
    num_radio_classes: int = 24
    use_radio_head: bool = False
    use_radio_condition_in_beam_head: bool = False
    radio_embed_dim: int = 32
    radio_tau: float = 1.0
    proto_type: str = "none"
    radio_label_mode: str = "peak_spread"
    path_semantic_enabled: bool = False
    num_path_classes: int = 24
    use_path_head: bool = False
    use_path_condition_in_beam_head: bool = False
    path_embed_dim: int = 32
    path_tau: float = 1.0
    use_path_regression: bool = False
    path_descriptor_dim: int | None = None
    path_label_mode: str = "kmeans_path_descriptor"
    history_anchor_enabled: bool = False
    history_anchor_mode: str = "residual_delta"
    num_delta_classes: int = 64
    history_anchor_embedding_dim: int = 32
    lambda_absolute_aux: float = 0.0
    v7_residual_scale: float = 1.0
    v8_mode: str = "target_prior_head"
    v8_adapter_dim: int | None = None
    v8_adapter_dropout: float = 0.0
    v8_use_adapter: bool = True
    v8_use_target_prior: bool = True
    v8_use_source_logits_in_final: bool = False
    v8_lambda_src: float = 0.0
    v8_lambda_tgt: float = 1.0
    v8_beta_prior: float = 1.0
    v8_learnable_beta_prior: bool = False
    v8_use_coarse_to_fine: bool = False
    v8_sector_size: int = 8
    v8_unfreeze_last_fusion_block: bool = False
    v8_use_soft_beam_label: bool = True
    v8_soft_label_sigma: float = 1.0
    v8_loss_prior_smooth_weight: float = 0.001
    v8_run_prototype_probe: bool = False
    v9_use_target_prior: bool = True
    v9_beta_prior_max: float = 1.0
    v9_learnable_beta_prior: bool = True
    v9_prior_dropout: float = 0.0
    v9_use_prototype_logits: bool = True
    v9_prototype_type: str = "beam"
    v9_prototype_tau: float = 0.1
    v9_eta_prototype: float = 1.0
    v9_sector_size: int = 2
    v9_prototype_feature_source: str = "target_adapter"
    v9_use_widened_prior_marginal_kl: bool = False
    v9_widened_prior_sigma: float = 3.0
    v9_widened_prior_temperature: float = 1.5
    v9_loss_widened_prior_marginal_kl_weight: float = 0.0

    @property
    def num_groups(self) -> int:
        return self.num_classes // self.group_size

    @property
    def hierarchical_enabled(self) -> bool:
        return self.variant not in {"v0_flat", "flat"}

    @property
    def shared_private_enabled(self) -> bool:
        return self.variant in {
            "v2_shared_private",
            "shared_private",
            "v3_decoupled",
            "decoupled",
            "v4_adapter",
            "adapter",
            "v5_adapter_proto",
            "adapter_proto",
            "v6_radio_proto",
            "adapter_radio_proto",
            "v8_path_proto",
            "adapter_path_proto",
            "v7_shared_physical_private_residual",
            "shared_physical_private_residual",
            "v8_target_prior_head",
            "v9_input_conditioned_target_adaptation",
            "v6_full_finetune",
            "full_finetune",
        }

    @property
    def decoupled_enabled(self) -> bool:
        return self.variant in {
            "v3_decoupled",
            "decoupled",
            "v4_adapter",
            "adapter",
            "v5_adapter_proto",
            "adapter_proto",
            "v6_radio_proto",
            "adapter_radio_proto",
            "v8_path_proto",
            "adapter_path_proto",
            "v7_shared_physical_private_residual",
            "shared_physical_private_residual",
            "v8_target_prior_head",
            "v9_input_conditioned_target_adaptation",
            "v6_full_finetune",
            "full_finetune",
        }

    @property
    def v7_enabled(self) -> bool:
        return self.variant in {"v7_shared_physical_private_residual", "shared_physical_private_residual"}

    @property
    def v8_target_prior_enabled(self) -> bool:
        return self.variant == "v8_target_prior_head"

    @property
    def v9_enabled(self) -> bool:
        return self.variant == "v9_input_conditioned_target_adaptation"

    @property
    def target_prior_branch_enabled(self) -> bool:
        return self.v8_target_prior_enabled or self.v9_enabled


def resolve_hist_beam_config(
    *,
    num_classes: int = 64,
    group_size: int = 8,
    variant: str = "v3_decoupled",
    modalities: list[str] | tuple[str, ...] | None = None,
    loss_weights: dict[str, Any] | None = None,
    adapter: bool | dict[str, Any] | None = None,
    prototype: bool | dict[str, Any] | None = None,
    radio_semantic: bool | dict[str, Any] | None = None,
    num_radio_classes: int | None = None,
    use_radio_head: bool | None = None,
    use_radio_condition_in_beam_head: bool | None = None,
    radio_embed_dim: int | None = None,
    radio_tau: float | None = None,
    path_semantic: bool | dict[str, Any] | None = None,
    num_path_classes: int | None = None,
    use_path_head: bool | None = None,
    use_path_condition_in_beam_head: bool | None = None,
    path_embed_dim: int | None = None,
    path_tau: float | None = None,
    use_path_regression: bool | None = None,
    path_descriptor_dim: int | None = None,
    history_anchor: bool | dict[str, Any] | None = None,
    v7: bool | dict[str, Any] | None = None,
    v8: bool | dict[str, Any] | None = None,
    v9: bool | dict[str, Any] | None = None,
    residual_scale: float | None = None,
    proto_type: str | None = None,
    geometry_aware: bool | dict[str, Any] | None = None,
    geometry_fields: list[str] | tuple[str, ...] | None = None,
    **_: Any,
) -> HistBeamConfig:
    classes = int(num_classes)
    group = int(group_size)
    if classes <= 0:
        raise ValueError(f"HiST-Beam num_classes must be positive, got {classes}.")
    if group <= 0:
        raise ValueError(f"HiST-Beam group_size must be positive, got {group}.")
    if classes % group != 0:
        raise ValueError(
            f"HiST-Beam num_classes ({classes}) must be divisible by group_size ({group}). "
            "Use a group_size that evenly partitions the beam classes, for example 8 for 64 classes."
        )
    normalized_variant = str(variant).strip().lower()
    if normalized_variant not in HIST_BEAM_VARIANTS:
        raise ValueError(
            f"Unknown HiST-Beam variant '{variant}'. Available variants: {sorted(HIST_BEAM_VARIANTS)}."
        )
    selected_modalities = normalize_modalities(
        modalities or DEFAULT_HIST_MODALITIES,
        context="HiST-Beam modalities",
    )
    weights = loss_weights or {}
    geometry_enabled = _mapping_enabled(geometry_aware)
    adapter_cfg = adapter if isinstance(adapter, dict) else {}
    radio_cfg = radio_semantic if isinstance(radio_semantic, dict) else {}
    path_cfg = path_semantic if isinstance(path_semantic, dict) else {}
    history_cfg = history_anchor if isinstance(history_anchor, dict) else {}
    v7_cfg = v7 if isinstance(v7, dict) else {}
    v8_cfg = v8 if isinstance(v8, dict) else {}
    v9_cfg = v9 if isinstance(v9, dict) else {}
    is_v7 = normalized_variant in {"v7_shared_physical_private_residual", "shared_physical_private_residual"}
    is_v8 = normalized_variant == "v8_target_prior_head"
    is_v9 = normalized_variant == "v9_input_conditioned_target_adaptation"
    history_enabled = False if is_v7 else _mapping_enabled(history_anchor)
    history_mode = str(history_cfg.get("mode", "residual_delta")).strip().lower()
    if history_mode not in {"residual_delta", "absolute_with_history"}:
        raise ValueError(
            f"Unsupported HiST-Beam history_anchor.mode '{history_mode}'. "
            "Supported modes: ['absolute_with_history', 'residual_delta']."
        )
    radio_enabled = _mapping_enabled(radio_semantic) or normalized_variant in {"v6_radio_proto", "adapter_radio_proto"}
    path_enabled = _mapping_enabled(path_semantic) or normalized_variant in {"v8_path_proto", "adapter_path_proto"}
    resolved_num_radio = int(
        num_radio_classes
        or radio_cfg.get("num_radio_classes")
        or radio_cfg.get("num_classes")
        or (classes // group) * int(radio_cfg.get("num_spread_bins", 3))
    )
    resolved_proto_type = str(
        proto_type
        or path_cfg.get("proto_type")
        or radio_cfg.get("proto_type")
        or ("path" if normalized_variant in {"v8_path_proto", "adapter_path_proto"} else None)
        or ("radio_semantic" if normalized_variant in {"v6_radio_proto", "adapter_radio_proto"} else "coarse" if normalized_variant in {"v5_adapter_proto", "adapter_proto"} else "none")
    ).strip().lower()
    resolved_num_path = int(num_path_classes or path_cfg.get("num_path_classes") or path_cfg.get("num_classes") or 24)
    v8_mode = str(v8_cfg.get("mode", "target_prior_head")).strip().lower()
    if v8_mode not in {"target_linear_probe", "target_prior_head", "source_prior_only", "target_prior_coarse_to_fine"}:
        raise ValueError(
            f"Unsupported HiST-Beam v8.mode '{v8_mode}'. "
            "Supported modes: ['source_prior_only', 'target_linear_probe', 'target_prior_coarse_to_fine', 'target_prior_head']."
        )
    v8_use_adapter = bool(v8_cfg.get("use_adapter", v8_mode != "target_linear_probe"))
    v8_use_target_prior = bool(v8_cfg.get("use_target_prior", v8_mode != "target_linear_probe"))
    v8_use_source = bool(v8_cfg.get("use_source_logits_in_final", v8_mode == "source_prior_only"))
    v8_lambda_src = float(v8_cfg.get("lambda_src", 1.0 if v8_mode == "source_prior_only" else 0.0))
    v8_lambda_tgt = float(v8_cfg.get("lambda_tgt", 0.0 if v8_mode == "source_prior_only" else 1.0))
    v8_use_coarse_to_fine = bool(v8_cfg.get("use_coarse_to_fine", v8_mode == "target_prior_coarse_to_fine"))
    v8_sector_size = int(v8_cfg.get("sector_size", group))
    if (is_v8 or is_v9) and v8_sector_size <= 0:
        raise ValueError(f"hist_beam.v8.sector_size must be positive, got {v8_sector_size}.")
    v9_prototype_type = str(v9_cfg.get("prototype_type", "beam")).strip().lower()
    if v9_prototype_type not in {"beam", "sector", "none"}:
        raise ValueError("hist_beam.v9.prototype_type must be one of ['beam', 'sector', 'none'].")
    v9_sector_size = int(v9_cfg.get("sector_size", 2))
    if is_v9 and v9_sector_size not in {2, 3}:
        raise ValueError(f"hist_beam.v9.sector_size must be 2 or 3 for v9 quick validation, got {v9_sector_size}.")
    v9_beta_max = float(v9_cfg.get("beta_prior_max", 1.0))
    if is_v9 and v9_beta_max <= 0:
        raise ValueError(f"hist_beam.v9.beta_prior_max must be positive, got {v9_beta_max}.")
    v9_tau = float(v9_cfg.get("prototype_tau", v9_cfg.get("tau", 0.1)))
    if is_v9 and v9_tau <= 0:
        raise ValueError(f"hist_beam.v9.prototype_tau must be positive, got {v9_tau}.")
    return HistBeamConfig(
        num_classes=classes,
        group_size=group,
        variant=normalized_variant,
        modalities=selected_modalities,
        lambda_hier=float(weights.get("hierarchical", weights.get("lambda_hier", 1.0))),
        lambda_flat=float(weights.get("flat", weights.get("lambda_flat", 0.2))),
        lambda_orth=float(weights.get("orthogonality", weights.get("lambda_orth", 0.01))),
        lambda_scene_c=float(weights.get("scene_confusion", weights.get("lambda_scene_c", 0.05))),
        lambda_scene_s=float(weights.get("scene_private", weights.get("lambda_scene_s", 0.05))),
        adapter_enabled=_mapping_enabled(adapter)
        or normalized_variant in {"v4_adapter", "adapter", "v5_adapter_proto", "adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto", "v7_shared_physical_private_residual", "shared_physical_private_residual"},
        prototype_enabled=_mapping_enabled(prototype)
        or normalized_variant in {"v5_adapter_proto", "adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"},
        geometry_aware=geometry_enabled,
        geometry_fields=tuple(str(item) for item in (geometry_fields or ())),
        coarse_conditioned_adapter=bool(adapter_cfg.get("coarse_conditioned", adapter_cfg.get("coarse_conditioned_adapter", False))),
        radio_semantic_enabled=radio_enabled,
        num_radio_classes=resolved_num_radio,
        use_radio_head=bool(use_radio_head if use_radio_head is not None else radio_cfg.get("use_radio_head", radio_enabled)),
        use_radio_condition_in_beam_head=bool(
            use_radio_condition_in_beam_head
            if use_radio_condition_in_beam_head is not None
            else radio_cfg.get("use_radio_condition_in_beam_head", radio_cfg.get("condition_beam_head", False))
        ),
        radio_embed_dim=int(radio_embed_dim or radio_cfg.get("radio_embed_dim", min(32, max(classes, group)))),
        radio_tau=float(radio_tau or radio_cfg.get("radio_tau", radio_cfg.get("tau", 1.0))),
        proto_type=resolved_proto_type,
        radio_label_mode=str(radio_cfg.get("mode", radio_cfg.get("label_mode", "peak_spread"))),
        path_semantic_enabled=path_enabled,
        num_path_classes=resolved_num_path,
        use_path_head=bool(use_path_head if use_path_head is not None else path_cfg.get("use_path_head", path_enabled)),
        use_path_condition_in_beam_head=bool(
            use_path_condition_in_beam_head
            if use_path_condition_in_beam_head is not None
            else path_cfg.get("use_path_condition_in_beam_head", path_cfg.get("condition_beam_head", False))
        ),
        path_embed_dim=int(path_embed_dim or path_cfg.get("path_embed_dim", min(32, max(classes, group)))),
        path_tau=float(path_tau or path_cfg.get("path_tau", path_cfg.get("tau", 1.0))),
        use_path_regression=bool(
            use_path_regression if use_path_regression is not None else path_cfg.get("use_path_regression", False)
        ),
        path_descriptor_dim=(
            int(path_descriptor_dim or path_cfg.get("descriptor_dim"))
            if str(path_descriptor_dim or path_cfg.get("descriptor_dim", "")).lower() not in {"", "none", "auto"}
            else None
        ),
        path_label_mode=str(path_cfg.get("mode", path_cfg.get("label_mode", "kmeans_path_descriptor"))),
        history_anchor_enabled=history_enabled,
        history_anchor_mode=history_mode,
        num_delta_classes=int(history_cfg.get("num_delta_classes", classes)),
        history_anchor_embedding_dim=int(history_cfg.get("embedding_dim", history_cfg.get("history_anchor_embedding_dim", 32))),
        lambda_absolute_aux=float(history_cfg.get("lambda_absolute_aux", 0.0)),
        v7_residual_scale=float(
            residual_scale
            if residual_scale is not None
            else v7_cfg.get("residual_scale", v7_cfg.get("private_residual_scale", 1.0))
        ),
        v8_mode=v8_mode,
        v8_adapter_dim=(
            int(v8_cfg.get("adapter_dim"))
            if v8_cfg.get("adapter_dim") not in {None, "", "none", "auto"}
            else None
        ),
        v8_adapter_dropout=float(v8_cfg.get("adapter_dropout", 0.0)),
        v8_use_adapter=v8_use_adapter,
        v8_use_target_prior=v8_use_target_prior,
        v8_use_source_logits_in_final=v8_use_source,
        v8_lambda_src=v8_lambda_src,
        v8_lambda_tgt=v8_lambda_tgt,
        v8_beta_prior=float(v8_cfg.get("beta_prior", 1.0)),
        v8_learnable_beta_prior=bool(v8_cfg.get("learnable_beta_prior", False)),
        v8_use_coarse_to_fine=v8_use_coarse_to_fine,
        v8_sector_size=v8_sector_size,
        v8_unfreeze_last_fusion_block=bool(v8_cfg.get("unfreeze_last_fusion_block", False)),
        v8_use_soft_beam_label=bool(v8_cfg.get("use_soft_beam_label", True)),
        v8_soft_label_sigma=float(v8_cfg.get("soft_label_sigma", 1.0)),
        v8_loss_prior_smooth_weight=float(v8_cfg.get("loss_prior_smooth_weight", 0.001)),
        v8_run_prototype_probe=bool(v8_cfg.get("run_prototype_probe", False)),
        v9_use_target_prior=bool(v9_cfg.get("use_target_prior", True)),
        v9_beta_prior_max=v9_beta_max,
        v9_learnable_beta_prior=bool(v9_cfg.get("learnable_beta_prior", True)),
        v9_prior_dropout=float(v9_cfg.get("prior_dropout", 0.0)),
        v9_use_prototype_logits=bool(v9_cfg.get("use_prototype_logits", v9_prototype_type != "none")),
        v9_prototype_type=v9_prototype_type,
        v9_prototype_tau=v9_tau,
        v9_eta_prototype=float(v9_cfg.get("eta_prototype", 1.0)),
        v9_sector_size=v9_sector_size,
        v9_prototype_feature_source=str(v9_cfg.get("prototype_feature_source", "target_adapter")).strip().lower(),
        v9_use_widened_prior_marginal_kl=bool(v9_cfg.get("use_widened_prior_marginal_kl", False)),
        v9_widened_prior_sigma=float(v9_cfg.get("widened_prior_sigma", 3.0)),
        v9_widened_prior_temperature=float(v9_cfg.get("widened_prior_temperature", 1.5)),
        v9_loss_widened_prior_marginal_kl_weight=float(
            v9_cfg.get(
                "loss_widened_prior_marginal_kl_weight",
                v9_cfg.get("marginal_kl_weight", v9_cfg.get("loss_weight", 0.0)),
            )
        ),
    )


@MODELS.register("hist_beam_fusion")
class HistBeamFusionNet(nn.Module):
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        feature_size: int = 64,
        d_model: int = 256,
        num_classes: int = 64,
        num_pred: int = 1,
        group_size: int = 8,
        variant: str = "v3_decoupled",
        modalities: list[str] | tuple[str, ...] | None = None,
        loss_weights: dict[str, Any] | None = None,
        adapter: bool | dict[str, Any] | None = None,
        prototype: bool | dict[str, Any] | None = None,
        radio_semantic: bool | dict[str, Any] | None = None,
        num_radio_classes: int | None = None,
        use_radio_head: bool | None = None,
        use_radio_condition_in_beam_head: bool | None = None,
        radio_embed_dim: int | None = None,
        radio_tau: float | None = None,
        path_semantic: bool | dict[str, Any] | None = None,
        num_path_classes: int | None = None,
        use_path_head: bool | None = None,
        use_path_condition_in_beam_head: bool | None = None,
        path_embed_dim: int | None = None,
        path_tau: float | None = None,
        use_path_regression: bool | None = None,
        path_descriptor_dim: int | None = None,
        history_anchor: bool | dict[str, Any] | None = None,
        v7: bool | dict[str, Any] | None = None,
        v8: bool | dict[str, Any] | None = None,
        v9: bool | dict[str, Any] | None = None,
        residual_scale: float | None = None,
        proto_type: str | None = None,
        geometry_aware: bool | dict[str, Any] | None = None,
        geometry_fields: list[str] | tuple[str, ...] | None = None,
        geometry_input_size: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        image_channels: int = 3,
        image_encoder: str | dict[str, Any] | None = None,
        radar_channels: int = 2,
        gps_input_size: int = 3,
        lidar_channels: int = 3,
        mmwave_input_size: int = MMWAVE_INPUT_SIZE,
        num_scenes: int = 4,
        grl_lambda: float = 1.0,
        image_profile: str | None = "rgb_imagenet",
        **_: Any,
    ):
        super().__init__()
        self.name = "HistBeamFusionNet"
        self.hist_config = resolve_hist_beam_config(
            num_classes=num_classes,
            group_size=group_size,
            variant=variant,
            modalities=modalities,
            loss_weights=loss_weights,
            adapter=adapter,
            prototype=prototype,
            radio_semantic=radio_semantic,
            num_radio_classes=num_radio_classes,
            use_radio_head=use_radio_head,
            use_radio_condition_in_beam_head=use_radio_condition_in_beam_head,
            radio_embed_dim=radio_embed_dim,
            radio_tau=radio_tau,
            path_semantic=path_semantic,
            num_path_classes=num_path_classes,
            use_path_head=use_path_head,
            use_path_condition_in_beam_head=use_path_condition_in_beam_head,
            path_embed_dim=path_embed_dim,
            path_tau=path_tau,
            use_path_regression=use_path_regression,
            path_descriptor_dim=path_descriptor_dim,
            history_anchor=history_anchor,
            v7=v7,
            v8=v8,
            v9=v9,
            residual_scale=residual_scale,
            proto_type=proto_type,
            geometry_aware=geometry_aware,
            geometry_fields=geometry_fields,
        )
        self.modalities = self.hist_config.modalities
        self.feature_size = int(feature_size)
        self.d_model = int(d_model)
        self.num_classes = self.hist_config.num_classes
        self.group_size = self.hist_config.group_size
        self.num_groups = self.hist_config.num_groups
        self.num_radio_classes = self.hist_config.num_radio_classes
        self.num_path_classes = self.hist_config.num_path_classes
        self.num_delta_classes = self.hist_config.num_delta_classes
        self.radio_tau = self.hist_config.radio_tau
        self.path_tau = self.hist_config.path_tau
        self.num_pred = int(num_pred)
        self.horizon = self.num_pred
        self.max_seq_len = int(max_seq_len)
        self.cls_type_id = len(MODALITY_ORDER)
        self.geometry_type_id = len(MODALITY_ORDER) + 1
        self.num_scenes = int(num_scenes)
        self.grl_lambda = float(grl_lambda)
        if self.num_pred <= 0:
            raise ValueError(f"num_pred must be positive, got {num_pred}.")
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({num_heads}).")

        self.encoders = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        for modality in self.modalities:
            self.encoders[modality] = _build_hist_modality_encoder(
                modality,
                self.feature_size,
                image_channels=image_channels,
                image_encoder=image_encoder,
                image_profile=image_profile,
                radar_channels=radar_channels,
                gps_input_size=gps_input_size,
                lidar_channels=lidar_channels,
                mmwave_input_size=mmwave_input_size,
            )
            self.feature_projections[modality] = (
                nn.Identity() if self.feature_size == self.d_model else nn.Linear(self.feature_size, self.d_model)
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.token_type_embedding = nn.Embedding(len(MODALITY_ORDER) + 2, self.d_model)
        self.time_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=max(self.d_model * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(self.d_model)
        self.history_anchor_embedding = (
            nn.Embedding(self.num_classes, self.hist_config.history_anchor_embedding_dim)
            if self.hist_config.history_anchor_enabled
            else None
        )
        self.history_anchor_projection = (
            nn.Sequential(
                nn.LayerNorm(self.hist_config.history_anchor_embedding_dim),
                nn.Linear(self.hist_config.history_anchor_embedding_dim, self.d_model),
                nn.GELU(),
            )
            if self.hist_config.history_anchor_enabled
            else None
        )

        self.flat_head = nn.Linear(self.d_model, self.num_pred * self.num_classes)
        self.shared_branch = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model), nn.GELU())
        self.private_branch = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model), nn.GELU())
        adapter_hidden = _adapter_bottleneck_dim(adapter, self.d_model)
        if self.hist_config.coarse_conditioned_adapter:
            self.private_adapter = CoarseConditionedPrivateAdapter(
                self.d_model,
                self.num_groups,
                bottleneck_dim=adapter_hidden,
            )
        else:
            self.private_adapter = BottleneckPrivateAdapter(
                self.d_model,
                bottleneck_dim=adapter_hidden,
            )
        self.geometry_projection = (
            nn.Sequential(
                nn.LayerNorm(int(geometry_input_size)),
                nn.Linear(int(geometry_input_size), self.d_model),
                nn.GELU(),
            )
            if self.hist_config.geometry_aware
            else None
        )
        self.geometry_input_size = int(geometry_input_size)
        self.coarse_head = nn.Linear(self.d_model, self.num_pred * self.num_groups)
        self.radio_head = (
            nn.Linear(self.d_model, self.num_pred * self.num_radio_classes)
            if self.hist_config.use_radio_head
            else None
        )
        self.path_head = (
            nn.Linear(self.d_model, self.num_pred * self.num_path_classes)
            if self.hist_config.use_path_head
            else None
        )
        self.path_attr_head = (
            nn.Linear(self.d_model, self.num_pred * int(self.hist_config.path_descriptor_dim))
            if self.hist_config.use_path_regression and self.hist_config.path_descriptor_dim
            else None
        )
        self.radio_embedding = (
            nn.Embedding(self.num_radio_classes, self.hist_config.radio_embed_dim)
            if self.hist_config.use_radio_condition_in_beam_head
            else None
        )
        self.path_embedding = (
            nn.Embedding(self.num_path_classes, self.hist_config.path_embed_dim)
            if self.hist_config.use_path_condition_in_beam_head
            else None
        )
        fine_input_dim = self.d_model * 2 + (
            self.hist_config.radio_embed_dim if self.hist_config.use_radio_condition_in_beam_head else 0
        ) + (
            self.hist_config.path_embed_dim if self.hist_config.use_path_condition_in_beam_head else 0
        )
        self.fine_head = nn.Linear(fine_input_dim, self.num_pred * self.num_groups * self.group_size)
        self.residual_head = (
            nn.Linear(self.d_model, self.num_pred * self.num_delta_classes)
            if self.hist_config.history_anchor_enabled
            else None
        )
        self.shared_beam_head = (
            nn.Linear(self.d_model, self.num_pred * self.num_classes)
            if self.hist_config.v7_enabled
            else None
        )
        self.physical_beamspace_head = (
            nn.Linear(self.d_model, self.num_pred * self.num_classes)
            if self.hist_config.v7_enabled
            else None
        )
        self.private_residual_head = (
            nn.Linear(self.d_model, self.num_pred * self.num_classes)
            if self.hist_config.v7_enabled
            else None
        )
        self.residual_gate = (
            nn.Linear(self.d_model, self.num_pred)
            if self.hist_config.v7_enabled
            else None
        )
        self.target_adapter = (
            BottleneckAdapter(
                self.d_model,
                bottleneck_dim=self.hist_config.v8_adapter_dim,
                dropout=self.hist_config.v8_adapter_dropout,
            )
            if self.hist_config.target_prior_branch_enabled and self.hist_config.v8_use_adapter
            else None
        )
        self.target_head = (
            nn.Linear(self.d_model, self.num_pred * self.num_classes)
            if self.hist_config.target_prior_branch_enabled
            else None
        )
        self.target_prior_bias = (
            nn.Parameter(torch.zeros(self.num_classes))
            if self.hist_config.target_prior_branch_enabled
            else None
        )
        if self.hist_config.v9_enabled:
            initial = _inverse_sigmoid_clamped(
                float(self.hist_config.v8_beta_prior) / max(float(self.hist_config.v9_beta_prior_max), 1e-12)
            )
            if self.hist_config.v9_learnable_beta_prior:
                self.beta_prior_raw = nn.Parameter(torch.tensor(initial))
            else:
                self.register_buffer("beta_prior_raw", torch.tensor(initial), persistent=True)
            self.beta_prior = None
        elif self.hist_config.v8_target_prior_enabled and self.hist_config.v8_learnable_beta_prior:
            self.beta_prior = nn.Parameter(torch.tensor(float(self.hist_config.v8_beta_prior)))
            self.beta_prior_raw = None
        elif self.hist_config.v8_target_prior_enabled:
            self.register_buffer("beta_prior", torch.tensor(float(self.hist_config.v8_beta_prior)), persistent=True)
            self.beta_prior_raw = None
        else:
            self.beta_prior = None
            self.beta_prior_raw = None
        self.sector_head = (
            nn.Linear(
                self.d_model,
                self.num_pred * _num_v8_sectors(self.num_classes, self.hist_config.v8_sector_size),
            )
            if self.hist_config.v8_target_prior_enabled and self.hist_config.v8_use_coarse_to_fine
            else None
        )
        self.offset_head = (
            nn.Linear(self.d_model, self.num_pred * self.hist_config.v8_sector_size)
            if self.hist_config.v8_target_prior_enabled and self.hist_config.v8_use_coarse_to_fine
            else None
        )
        self.absolute_calibration_bias = (
            nn.Parameter(torch.zeros(self.num_classes))
            if self.hist_config.history_anchor_enabled
            else None
        )
        self.absolute_temperature_log = (
            nn.Parameter(torch.zeros(()))
            if self.hist_config.history_anchor_enabled
            else None
        )
        self.shared_scene_classifier = nn.Linear(self.d_model, self.num_scenes) if self.num_scenes > 0 else None
        self.private_scene_classifier = nn.Linear(self.d_model, self.num_scenes) if self.num_scenes > 0 else None

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        geometry_batch: torch.Tensor | None = None,
        geometry_mask: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        radio_assignment: torch.Tensor | None = None,
        radio_prototypes: torch.Tensor | None = None,
        radio_prototype_counts: torch.Tensor | None = None,
        path_assignment: torch.Tensor | None = None,
        path_prototypes: torch.Tensor | None = None,
        path_prototype_counts: torch.Tensor | None = None,
        target_prototypes: torch.Tensor | None = None,
        target_prototype_counts: torch.Tensor | None = None,
        mu_path_c: torch.Tensor | None = None,
        input_beam_batch: torch.Tensor | None = None,
        last_beam_batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...] | dict[str, Any]]:
        raw_inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
            "mmwave": mmwave_batch,
        }
        modality_features = []
        batch_size = None
        seq_len = None
        for modality in self.modalities:
            tensor = raw_inputs[modality]
            if tensor is None:
                raise ValueError(f"HiST-Beam requires '{modality}' input because it is enabled.")
            features = self.feature_projections[modality](self.encoders[modality](tensor))
            batch_size, seq_len = _check_temporal_features(features, modality, batch_size, seq_len)
            modality_features.append(features)
        assert batch_size is not None and seq_len is not None
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")

        stacked = torch.stack(modality_features, dim=1)
        effective_mask = _effective_modality_mask(
            batch_size,
            len(self.modalities),
            device=stacked.device,
            force_modality_mask=force_modality_mask,
        )
        if torch.any(~effective_mask.any(dim=1)):
            raise ValueError("force_modality_mask leaves no available modalities for at least one sample.")
        tokens = self._embed_modality_tokens(stacked)
        token_padding_mask = ~effective_mask.unsqueeze(-1).expand(batch_size, len(self.modalities), seq_len)
        diagnostic_tokens = tokens.masked_fill(token_padding_mask.unsqueeze(-1), 0.0)
        geometry_tokens, geometry_token_mask, geometry_summary, geometry_diag = self._geometry_tokens(
            geometry_batch,
            geometry_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=stacked.device,
            dtype=stacked.dtype,
        )
        if geometry_tokens is not None and geometry_token_mask is not None:
            flat_tokens = torch.cat([_serialize_time_first(tokens), geometry_tokens], dim=1)
            flat_padding_mask = torch.cat([_serialize_mask_time_first(token_padding_mask), geometry_token_mask], dim=1)
        else:
            flat_tokens = _serialize_time_first(tokens)
            flat_padding_mask = _serialize_mask_time_first(token_padding_mask)
        cls_ids = torch.full((batch_size, 1), self.cls_type_id, dtype=torch.long, device=stacked.device)
        cls = self.input_dropout(
            self.input_norm(self.cls_token.expand(batch_size, -1, -1) + self.token_type_embedding(cls_ids))
        )
        memory = self.transformer(
            torch.cat([cls, flat_tokens], dim=1),
            src_key_padding_mask=torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=stacked.device), flat_padding_mask],
                dim=1,
            ),
        )
        fused = self.output_norm(memory[:, 0, :])
        history_anchor = None
        history_context = None
        if self.hist_config.history_anchor_enabled:
            history_anchor = self._history_anchor(
                input_beam_batch=input_beam_batch,
                last_beam_batch=last_beam_batch,
                batch_size=batch_size,
                device=fused.device,
            )
            assert self.history_anchor_embedding is not None and self.history_anchor_projection is not None
            history_context = self.history_anchor_projection(self.history_anchor_embedding(history_anchor))
            fused = fused + history_context
        flat_logits = self.flat_head(fused).view(batch_size, self.num_pred, self.num_classes)

        shared = self.shared_branch(fused)
        private = self.private_branch(fused)
        coarse_logits = self.coarse_head(shared).view(batch_size, self.num_pred, self.num_groups)
        adapter_rep = self._adapt_private(private, coarse_logits) if self.hist_config.adapter_enabled else private
        radio_logits = (
            self.radio_head(shared).view(batch_size, self.num_pred, self.num_radio_classes)
            if self.radio_head is not None
            else None
        )
        path_logits = (
            self.path_head(shared).view(batch_size, self.num_pred, self.num_path_classes)
            if self.path_head is not None
            else None
        )
        path_attr_pred = (
            self.path_attr_head(shared).view(batch_size, self.num_pred, int(self.hist_config.path_descriptor_dim))
            if self.path_attr_head is not None and self.hist_config.path_descriptor_dim
            else None
        )
        radio_alpha, radio_condition_source, radio_condition_available = self._radio_assignment_for_beam_head(
            shared,
            radio_logits=radio_logits,
            radio_assignment=radio_assignment,
            radio_prototypes=radio_prototypes,
            radio_prototype_counts=radio_prototype_counts,
        )
        path_alpha, path_condition_source, path_condition_available = self._path_assignment_for_beam_head(
            shared,
            path_logits=path_logits,
            path_assignment=path_assignment,
            path_prototypes=path_prototypes if path_prototypes is not None else mu_path_c,
            path_prototype_counts=path_prototype_counts,
        )
        fine_parts = [shared, adapter_rep]
        if self.radio_embedding is not None:
            radio_context = radio_alpha @ self.radio_embedding.weight.to(device=radio_alpha.device, dtype=radio_alpha.dtype)
            fine_parts.append(radio_context)
        else:
            radio_context = None
        if self.path_embedding is not None:
            path_context = path_alpha @ self.path_embedding.weight.to(device=path_alpha.device, dtype=path_alpha.dtype)
            fine_parts.append(path_context)
        else:
            path_context = None
        fine_input = torch.cat(fine_parts, dim=-1)
        fine_logits = self.fine_head(fine_input).view(batch_size, self.num_pred, self.num_groups, self.group_size)
        beam_log_probs = hierarchical_beam_log_probs(coarse_logits, fine_logits)
        logits = flat_logits if not self.hist_config.hierarchical_enabled else beam_log_probs
        residual_logits = None
        reconstructed_beam_logits = None
        logits_shared = None
        logits_final = None
        delta_logits_private = None
        alpha = None
        pred_beamspace_power = None
        source_logits = None
        target_logits = None
        target_prior_bias = None
        prototype_logits = None
        sector_logits = None
        offset_logits = None
        if self.hist_config.history_anchor_enabled:
            if self.hist_config.history_anchor_mode == "residual_delta":
                if self.residual_head is None:
                    raise RuntimeError("history_anchor residual head is missing.")
                residual_logits = self.residual_head(shared).view(batch_size, self.num_pred, self.num_delta_classes)
                if history_anchor is None:
                    raise RuntimeError("history_anchor residual mode requires last_beam_batch.")
                reconstructed_beam_logits = residual_logits_to_absolute_logits(
                    residual_logits,
                    history_anchor,
                    num_classes=self.num_classes,
                )
                if self.absolute_calibration_bias is not None:
                    reconstructed_beam_logits = reconstructed_beam_logits + self.absolute_calibration_bias.view(1, 1, -1)
                if self.absolute_temperature_log is not None:
                    reconstructed_beam_logits = reconstructed_beam_logits / self.absolute_temperature_log.exp().clamp_min(1e-6)
                logits = reconstructed_beam_logits
            else:
                reconstructed_beam_logits = flat_logits
                logits = flat_logits
        if self.hist_config.v7_enabled:
            if (
                self.shared_beam_head is None
                or self.physical_beamspace_head is None
                or self.private_residual_head is None
                or self.residual_gate is None
            ):
                raise RuntimeError("V7 heads are not initialized.")
            logits_shared = self.shared_beam_head(shared).view(batch_size, self.num_pred, self.num_classes)
            delta_logits_private = self.private_residual_head(adapter_rep).view(batch_size, self.num_pred, self.num_classes)
            delta_logits_private = delta_logits_private * float(self.hist_config.v7_residual_scale)
            alpha = torch.sigmoid(self.residual_gate(adapter_rep).view(batch_size, self.num_pred, 1))
            logits_final = logits_shared + alpha * delta_logits_private
            pred_beamspace_power = torch.softmax(
                self.physical_beamspace_head(shared).view(batch_size, self.num_pred, self.num_classes),
                dim=-1,
            )
            logits = logits_final
            source_logits = logits_shared
        beta_effective = None
        prior_dropout_active = False
        prototype_status: dict[str, Any] = {
            "available": False,
            "unavailable_reason": "not_requested",
            "support_count": 0,
        }
        if self.hist_config.target_prior_branch_enabled:
            if self.target_head is None or self.target_prior_bias is None:
                raise RuntimeError("Target-prior heads are not initialized.")
            source_logits = beam_log_probs if self.hist_config.hierarchical_enabled else flat_logits
            target_rep = self.target_adapter(fused) if self.target_adapter is not None else fused
            target_logits = self.target_head(target_rep).view(batch_size, self.num_pred, self.num_classes)
            bias = self.target_prior_bias.to(device=target_logits.device, dtype=target_logits.dtype).view(1, 1, -1)
            target_prior_bias = bias.expand(batch_size, self.num_pred, -1)
            beta = self._effective_beta_prior(device=target_logits.device, dtype=target_logits.dtype)
            beta_effective = beta
            use_target_prior = (
                self.hist_config.v9_use_target_prior
                if self.hist_config.v9_enabled
                else self.hist_config.v8_use_target_prior
            )
            prior_term = beta * target_prior_bias if use_target_prior else target_logits.sum() * 0.0
            if self.hist_config.v9_enabled and self.training and float(self.hist_config.v9_prior_dropout) > 0.0:
                drop = torch.rand((), device=target_logits.device) < float(self.hist_config.v9_prior_dropout)
                prior_dropout_active = bool(drop.detach().cpu().item())
                if prior_dropout_active:
                    prior_term = target_logits.sum() * 0.0
            if self.hist_config.v9_enabled and self.hist_config.v9_use_prototype_logits:
                if target_prototypes is None:
                    target_prototypes = getattr(self, "_target_prototypes", None)
                if target_prototype_counts is None:
                    target_prototype_counts = getattr(self, "_target_prototype_counts", None)
                proto_feature = target_rep if self.hist_config.v9_prototype_feature_source == "target_adapter" else fused
                prototype_logits, prototype_status = _target_prototype_logits(
                    proto_feature,
                    target_prototypes=target_prototypes,
                    target_prototype_counts=target_prototype_counts,
                    num_classes=self.num_classes,
                    num_pred=self.num_pred,
                    prototype_type=self.hist_config.v9_prototype_type,
                    sector_size=self.hist_config.v9_sector_size,
                    tau=self.hist_config.v9_prototype_tau,
                )
                prototype_logits = prototype_logits.to(device=target_logits.device, dtype=target_logits.dtype)
            else:
                prototype_logits = target_logits.sum(dim=-1, keepdim=True).expand(-1, self.num_pred, self.num_classes) * 0.0
            if self.hist_config.v9_enabled:
                logits_final = target_logits + prior_term + float(self.hist_config.v9_eta_prototype) * prototype_logits
            elif self.hist_config.v8_use_source_logits_in_final:
                logits_final = (
                    float(self.hist_config.v8_lambda_src) * source_logits
                    + float(self.hist_config.v8_lambda_tgt) * target_logits
                    + prior_term
                )
            else:
                logits_final = target_logits + prior_term
            logits = logits_final
            if self.sector_head is not None:
                sector_logits = self.sector_head(target_rep).view(
                    batch_size,
                    self.num_pred,
                    _num_v8_sectors(self.num_classes, self.hist_config.v8_sector_size),
                )
            if self.offset_head is not None:
                offset_logits = self.offset_head(target_rep).view(batch_size, self.num_pred, self.hist_config.v8_sector_size)

        output_features = fused.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous()
        result: dict[str, torch.Tensor | tuple[str, ...] | dict[str, Any]] = {
            "logits": logits,
            "beam_logits": logits,
            "features": output_features,
            "logits_shared": logits_shared,
            "source_logits": source_logits,
            "logits_final": logits_final,
            "target_logits": target_logits,
            "target_prior_bias": target_prior_bias,
            "prototype_logits": prototype_logits,
            "sector_logits": sector_logits,
            "offset_logits": offset_logits,
            "delta_logits_private": delta_logits_private,
            "alpha": alpha,
            "pred_beamspace_power": pred_beamspace_power,
            "absolute_beam_logits": reconstructed_beam_logits,
            "residual_logits": residual_logits,
            "last_beam": history_anchor,
            "history_anchor_embedding": history_context,
            "beam_log_probs": beam_log_probs,
            "flat_logits": flat_logits,
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "radio_logits": radio_logits,
            "radio_assignment": radio_alpha,
            "radio_condition_embedding": radio_context,
            "path_logits": path_logits,
            "path_assignment": path_alpha,
            "path_condition_embedding": path_context,
            "path_attr_pred": path_attr_pred,
            "shared_representation": shared.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous(),
            "private_representation": private.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous(),
            "adapter_representation": adapter_rep.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous(),
            "shared_geometry_representation": shared.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous(),
            "geometry_representation": (
                geometry_summary.unsqueeze(1).expand(-1, self.num_pred, -1).contiguous()
                if geometry_summary is not None
                else None
            ),
            "geometry_diagnostics": geometry_diag,
            "input_features": _available_timewise_mean(diagnostic_tokens, effective_mask),
            "output_features": output_features,
            "token_features": diagnostic_tokens,
            "modalities": self.modalities,
            "effective_modality_mask": effective_mask,
            "fusion_memory": memory,
            "scene_diagnostics": self.scene_diagnostics(shared, private),
            "hist_beam": {
                "variant": self.hist_config.variant,
                "num_classes": self.num_classes,
                "group_size": self.group_size,
                "num_groups": self.num_groups,
                "adapter_enabled": self.hist_config.adapter_enabled,
                "prototype_enabled": self.hist_config.prototype_enabled,
                "proto_type": self.hist_config.proto_type,
                "radio_semantic_enabled": self.hist_config.radio_semantic_enabled,
                "num_radio_classes": self.num_radio_classes,
                "radio_label_mode": self.hist_config.radio_label_mode,
                "use_radio_head": self.hist_config.use_radio_head,
                "use_radio_condition_in_beam_head": self.hist_config.use_radio_condition_in_beam_head,
                "radio_condition_source": radio_condition_source,
                "radio_condition_available": radio_condition_available,
                "radio_tau": self.radio_tau,
                "path_semantic_enabled": self.hist_config.path_semantic_enabled,
                "num_path_classes": self.num_path_classes,
                "path_label_mode": self.hist_config.path_label_mode,
                "use_path_head": self.hist_config.use_path_head,
                "use_path_condition_in_beam_head": self.hist_config.use_path_condition_in_beam_head,
                "path_condition_source": path_condition_source,
                "path_condition_available": path_condition_available,
                "path_tau": self.path_tau,
                "use_path_regression": self.hist_config.use_path_regression,
                "path_descriptor_dim": self.hist_config.path_descriptor_dim,
                "geometry_aware": self.hist_config.geometry_aware,
                "geometry_fields": self.hist_config.geometry_fields,
                "coarse_conditioned_adapter": self.hist_config.coarse_conditioned_adapter,
                "history_anchor_enabled": self.hist_config.history_anchor_enabled,
                "history_anchor_mode": self.hist_config.history_anchor_mode,
                "num_delta_classes": self.num_delta_classes,
                "uses_input_beam_as_model_input": self.hist_config.history_anchor_enabled,
                "v7_shared_physical_private_residual": self.hist_config.v7_enabled,
                "residual_scale": float(self.hist_config.v7_residual_scale),
                "v8_target_prior_head": self.hist_config.v8_target_prior_enabled,
                "v9_input_conditioned_target_adaptation": self.hist_config.v9_enabled,
                "v8_mode": self.hist_config.v8_mode,
                "v8_use_adapter": self.hist_config.v8_use_adapter,
                "v8_use_target_prior": self.hist_config.v8_use_target_prior,
                "v8_use_source_logits_in_final": self.hist_config.v8_use_source_logits_in_final,
                "v8_lambda_src": float(self.hist_config.v8_lambda_src),
                "v8_lambda_tgt": float(self.hist_config.v8_lambda_tgt),
                "v8_beta_prior": float(beta_effective.detach().cpu().item())
                if torch.is_tensor(beta_effective)
                else float(self.hist_config.v8_beta_prior),
                "v8_learnable_beta_prior": self.hist_config.v8_learnable_beta_prior,
                "v8_use_coarse_to_fine": self.hist_config.v8_use_coarse_to_fine,
                "v8_sector_size": self.hist_config.v8_sector_size,
                "v8_unfreeze_last_fusion_block": self.hist_config.v8_unfreeze_last_fusion_block,
                "v8_use_soft_beam_label": self.hist_config.v8_use_soft_beam_label,
                "v8_soft_label_sigma": float(self.hist_config.v8_soft_label_sigma),
                "v8_loss_prior_smooth_weight": float(self.hist_config.v8_loss_prior_smooth_weight),
                "v8_run_prototype_probe": self.hist_config.v8_run_prototype_probe,
                "v9_use_target_prior": self.hist_config.v9_use_target_prior,
                "v9_beta_prior_max": float(self.hist_config.v9_beta_prior_max),
                "v9_beta_prior_parameterization": "cap_sigmoid" if self.hist_config.v9_enabled else "direct",
                "v9_learnable_beta_prior": self.hist_config.v9_learnable_beta_prior,
                "v9_prior_dropout": float(self.hist_config.v9_prior_dropout),
                "v9_prior_dropout_active": bool(prior_dropout_active),
                "v9_use_prototype_logits": self.hist_config.v9_use_prototype_logits,
                "v9_prototype_type": self.hist_config.v9_prototype_type,
                "v9_prototype_tau": float(self.hist_config.v9_prototype_tau),
                "v9_eta_prototype": float(self.hist_config.v9_eta_prototype),
                "v9_sector_size": int(self.hist_config.v9_sector_size),
                "v9_sector_mapping": "floor_division_shared_score_to_member_beams"
                if self.hist_config.v9_prototype_type == "sector"
                else None,
                "v9_prototype_feature_source": self.hist_config.v9_prototype_feature_source,
                "prototype_logits_available": bool(prototype_status.get("available", False)),
                "prototype_logits_unavailable_reason": prototype_status.get("unavailable_reason"),
                "prototype_support_count": int(prototype_status.get("support_count", 0) or 0),
                "prototype_support_counts": prototype_status.get("support_counts"),
                "source_logits_in_final": self.hist_config.v8_use_source_logits_in_final
                if self.hist_config.v8_target_prior_enabled
                else False,
                "prototype_probe_available": False if self.hist_config.v8_run_prototype_probe else None,
                "prototype_probe_unavailable_reason": "v8_prototype_probe_not_implemented"
                if self.hist_config.v8_run_prototype_probe
                else None,
                "residual_target_enabled": self.hist_config.history_anchor_enabled
                and self.hist_config.history_anchor_mode == "residual_delta",
                "private_calibration_type": "absolute_bias_temperature"
                if self.hist_config.history_anchor_enabled
                else "none",
            },
        }
        if self.shared_scene_classifier is not None:
            result["shared_scene_logits"] = self.shared_scene_classifier(
                gradient_reverse(shared, lambda_=self.grl_lambda)
            )
        if self.private_scene_classifier is not None:
            result["private_scene_logits"] = self.private_scene_classifier(private)
        return result

    def set_target_prior_from_labels(
        self,
        labels: torch.Tensor | list[int] | tuple[int, ...] | None,
        *,
        sigma: float = 1.5,
        eps: float = 1e-4,
    ) -> dict[str, Any]:
        if self.target_prior_bias is None:
            raise RuntimeError("set_target_prior_from_labels is only available for v8_target_prior_head.")
        from kd_sensing.engine.hist_beam_losses import gaussian_smooth_beam_prior

        device = self.target_prior_bias.device
        prior = gaussian_smooth_beam_prior(
            labels,
            self.num_classes,
            sigma=sigma,
            eps=eps,
            device=device,
        ).to(dtype=self.target_prior_bias.dtype)
        with torch.no_grad():
            self.target_prior_bias.copy_(torch.log(prior.clamp_min(float(eps))))
        flat = _labels_to_1d_tensor(labels, device=device)
        valid = flat[flat.ge(0) & flat.lt(self.num_classes)]
        hist = torch.bincount(valid.to(torch.long), minlength=self.num_classes) if valid.numel() else torch.zeros(
            self.num_classes,
            dtype=torch.long,
            device=device,
        )
        fallback_reason = "empty_support_labels" if valid.numel() == 0 else None
        metadata = {
            "target_prior_initialized": True,
            "target_prior_fallback_reason": fallback_reason,
            "target_support_label_count": int(valid.numel()),
            "target_support_label_hist": [int(item) for item in hist.detach().cpu().tolist()],
            "smoothed_target_prior_top_beams": _top_beam_records(prior.detach(), top_k=5),
            "target_prior_bias_top_beams": _top_beam_records(self.target_prior_bias.detach(), top_k=5),
        }
        self._target_prior_metadata = metadata
        return metadata

    def _effective_beta_prior(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.hist_config.v9_enabled:
            raw = self.beta_prior_raw
            if not torch.is_tensor(raw):
                raw = torch.tensor(0.0, device=device, dtype=dtype)
            return float(self.hist_config.v9_beta_prior_max) * torch.sigmoid(raw.to(device=device, dtype=dtype))
        if torch.is_tensor(self.beta_prior):
            return self.beta_prior.to(device=device, dtype=dtype)
        return torch.tensor(float(self.hist_config.v8_beta_prior), device=device, dtype=dtype)

    def target_prior_metadata(self) -> dict[str, Any]:
        return dict(getattr(self, "_target_prior_metadata", {}))

    def set_target_prototypes_from_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | list[int] | tuple[int, ...],
        *,
        prototype_type: str | None = None,
        sector_size: int | None = None,
    ) -> dict[str, Any]:
        if not self.hist_config.v9_enabled:
            raise RuntimeError("target support prototypes are only available for v9 input-conditioned adaptation.")
        feature_tensor = features.detach()
        if feature_tensor.ndim == 3:
            feature_tensor = feature_tensor[:, 0, :]
        if feature_tensor.ndim != 2:
            raise ValueError(f"features must have shape [N, D] or [N, H, D], got {tuple(feature_tensor.shape)}.")
        labels_t = _labels_to_1d_tensor(labels, device=feature_tensor.device)
        if labels_t.numel() != feature_tensor.shape[0]:
            raise ValueError(f"labels count {labels_t.numel()} does not match feature count {feature_tensor.shape[0]}.")
        ptype = str(prototype_type or self.hist_config.v9_prototype_type).strip().lower()
        ssize = int(sector_size or self.hist_config.v9_sector_size)
        class_count = self.num_classes if ptype == "beam" else _num_v8_sectors(self.num_classes, ssize)
        assignment = labels_t if ptype == "beam" else torch.div(labels_t, ssize, rounding_mode="floor")
        valid = labels_t.ge(0) & labels_t.lt(self.num_classes) & assignment.ge(0) & assignment.lt(class_count)
        prototypes = torch.zeros(class_count, feature_tensor.shape[-1], device=feature_tensor.device, dtype=feature_tensor.dtype)
        counts = torch.zeros(class_count, device=feature_tensor.device, dtype=torch.long)
        normalized = F.normalize(feature_tensor, dim=-1)
        for class_index in range(class_count):
            mask = valid & assignment.eq(class_index)
            if torch.any(mask):
                prototypes[class_index] = F.normalize(normalized[mask].mean(dim=0), dim=0)
                counts[class_index] = int(mask.sum().item())
        self._target_prototypes = prototypes
        self._target_prototype_counts = counts
        metadata = {
            "target_prototypes_initialized": bool(counts.gt(0).any().item()),
            "target_prototype_type": ptype,
            "target_prototype_sector_size": ssize if ptype == "sector" else None,
            "target_prototype_support_count": int(counts.sum().detach().cpu().item()),
            "target_prototype_available_count": int(counts.gt(0).sum().detach().cpu().item()),
            "target_prototype_counts": [int(item) for item in counts.detach().cpu().tolist()],
            "target_prototype_unavailable_reason": None if counts.gt(0).any().item() else "empty_support_labels",
        }
        self._target_prototype_metadata = metadata
        return metadata

    def target_prototype_metadata(self) -> dict[str, Any]:
        return dict(getattr(self, "_target_prototype_metadata", {}))

    def _history_anchor(
        self,
        *,
        input_beam_batch: torch.Tensor | None,
        last_beam_batch: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if last_beam_batch is not None:
            anchor = last_beam_batch.to(device=device, dtype=torch.long)
            if anchor.ndim == 2:
                if anchor.shape[1] == 1:
                    anchor = anchor[:, 0]
                else:
                    anchor = anchor[:, -1]
        elif input_beam_batch is not None:
            history = input_beam_batch.to(device=device, dtype=torch.long)
            if history.ndim == 1:
                anchor = history
            elif history.ndim == 2:
                anchor = history[:, -1]
            else:
                raise ValueError(f"input_beam_batch must have shape [B] or [B, T], got {tuple(history.shape)}.")
        else:
            raise ValueError(
                "HiST-Beam history_anchor is enabled, but neither input_beam_batch nor last_beam_batch was provided."
            )
        if anchor.shape[0] != batch_size:
            raise ValueError(f"last_beam_batch batch size {anchor.shape[0]} does not match input batch size {batch_size}.")
        if torch.any(anchor.lt(0) | anchor.ge(self.num_classes)):
            bad = int(anchor[(anchor.lt(0) | anchor.ge(self.num_classes))][0].detach().cpu().item())
            raise ValueError(f"last_beam_batch contains invalid beam label {bad}; expected [0, {self.num_classes}).")
        return anchor

    def _radio_assignment_for_beam_head(
        self,
        shared: torch.Tensor,
        *,
        radio_logits: torch.Tensor | None,
        radio_assignment: torch.Tensor | None,
        radio_prototypes: torch.Tensor | None,
        radio_prototype_counts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, str, bool]:
        if self.radio_embedding is None:
            empty = torch.zeros(shared.shape[0], self.num_radio_classes, device=shared.device, dtype=shared.dtype)
            return empty, "disabled", False
        if radio_assignment is not None:
            alpha = radio_assignment.to(device=shared.device, dtype=shared.dtype)
            if alpha.ndim == 3:
                alpha = alpha.mean(dim=1)
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return alpha, "provided_assignment", True
        if radio_prototypes is not None:
            proto = radio_prototypes.to(device=shared.device, dtype=shared.dtype)
            if proto.ndim == 2 and proto.shape[0] == self.num_radio_classes and proto.shape[1] == shared.shape[-1]:
                scores = F.normalize(shared, dim=-1) @ F.normalize(proto, dim=-1).t()
                if radio_prototype_counts is not None:
                    available = radio_prototype_counts.to(device=shared.device).reshape(-1).gt(0)
                    scores = scores.masked_fill(~available.view(1, -1), -1e9)
                alpha = torch.softmax(scores / max(float(self.radio_tau), 1e-6), dim=-1)
                return alpha, "source_radio_prototype", True
        if radio_logits is not None:
            alpha = torch.softmax(radio_logits.mean(dim=1) / max(float(self.radio_tau), 1e-6), dim=-1)
            return alpha, "radio_logits", True
        alpha = torch.full(
            (shared.shape[0], self.num_radio_classes),
            1.0 / max(int(self.num_radio_classes), 1),
            device=shared.device,
            dtype=shared.dtype,
        )
        return alpha, "uniform_fallback", False

    def _path_assignment_for_beam_head(
        self,
        shared: torch.Tensor,
        *,
        path_logits: torch.Tensor | None,
        path_assignment: torch.Tensor | None,
        path_prototypes: torch.Tensor | None,
        path_prototype_counts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, str, bool]:
        if self.path_embedding is None:
            empty = torch.zeros(shared.shape[0], self.num_path_classes, device=shared.device, dtype=shared.dtype)
            return empty, "disabled", False
        if path_assignment is not None:
            alpha = path_assignment.to(device=shared.device, dtype=shared.dtype)
            if alpha.ndim == 3:
                alpha = alpha.mean(dim=1)
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return alpha, "provided_assignment", True
        if path_prototypes is not None:
            proto = path_prototypes.to(device=shared.device, dtype=shared.dtype)
            if proto.ndim == 2 and proto.shape[0] == self.num_path_classes and proto.shape[1] == shared.shape[-1]:
                scores = F.normalize(shared, dim=-1) @ F.normalize(proto, dim=-1).t()
                if path_prototype_counts is not None:
                    available = path_prototype_counts.to(device=shared.device).reshape(-1).gt(0)
                    if available.numel() == proto.shape[0]:
                        scores = scores.masked_fill(~available.view(1, -1), -1e9)
                alpha = torch.softmax(scores / max(float(self.path_tau), 1e-6), dim=-1)
                return alpha, "source_path_prototype", True
        if path_logits is not None:
            alpha = torch.softmax(path_logits.mean(dim=1) / max(float(self.path_tau), 1e-6), dim=-1)
            return alpha, "path_logits", True
        alpha = torch.full(
            (shared.shape[0], self.num_path_classes),
            1.0 / max(int(self.num_path_classes), 1),
            device=shared.device,
            dtype=shared.dtype,
        )
        return alpha, "uniform_fallback", False

    def scene_diagnostics(self, shared: torch.Tensor, private: torch.Tensor) -> dict[str, Any]:
        return {
            "shared_norm": float(shared.detach().norm(dim=-1).mean().cpu().item()),
            "private_norm": float(private.detach().norm(dim=-1).mean().cpu().item()),
            "shared_private_cosine": float(
                F.cosine_similarity(shared.detach(), private.detach(), dim=-1).mean().cpu().item()
            ),
        }

    def _embed_modality_tokens(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, modality_count, seq_len, _ = features.shape
        time_ids = torch.arange(seq_len, device=features.device)
        time = self.time_embedding(time_ids).view(1, 1, seq_len, self.d_model)
        type_ids = torch.tensor(
            [MODALITY_ORDER.index(name) for name in self.modalities],
            dtype=torch.long,
            device=features.device,
        )
        token_type = self.token_type_embedding(type_ids).view(1, modality_count, 1, self.d_model)
        return self.input_dropout(self.input_norm(features + time + token_type))

    def _adapt_private(self, private: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        if isinstance(self.private_adapter, CoarseConditionedPrivateAdapter):
            return self.private_adapter(private, coarse_logits=coarse_logits)
        return self.private_adapter(private)

    def _geometry_tokens(
        self,
        geometry_batch: torch.Tensor | None,
        geometry_mask: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
        if not self.hist_config.geometry_aware:
            return None, None, None, {"enabled": False, "coverage": 0.0}
        fields = self.hist_config.geometry_fields or tuple(f"geometry_{idx}" for idx in range(self.geometry_input_size))
        if geometry_batch is None or self.geometry_projection is None:
            return None, None, None, {
                "enabled": True,
                "available": False,
                "coverage": 0.0,
                "unavailable_reason": "geometry_batch_missing",
                "fields": list(fields),
                "direct_fields": list(fields),
                "proxy_fields": [],
            }
        geometry = geometry_batch.to(device=device, dtype=dtype)
        if geometry.ndim != 3:
            raise ValueError(f"geometry_batch must have shape [B, T, F], got {tuple(geometry.shape)}.")
        geometry = geometry[:, -seq_len:, :]
        if geometry.shape[-1] != self.geometry_input_size:
            raise ValueError(
                f"geometry_batch feature dimension {geometry.shape[-1]} does not match geometry_input_size "
                f"{self.geometry_input_size}."
            )
        if geometry_mask is None:
            mask = torch.isfinite(geometry)
        else:
            mask = geometry_mask.to(device=device, dtype=torch.bool)[:, -seq_len:, :]
        valid_steps = mask.any(dim=-1)
        projected = self.geometry_projection(torch.nan_to_num(geometry, nan=0.0))
        time_ids = torch.arange(projected.shape[1], device=device)
        time = self.time_embedding(time_ids).view(1, projected.shape[1], self.d_model)
        type_ids = torch.full((batch_size, projected.shape[1]), self.geometry_type_id, dtype=torch.long, device=device)
        tokens = self.input_dropout(self.input_norm(projected + time + self.token_type_embedding(type_ids)))
        padding_mask = ~valid_steps
        denom = valid_steps.to(dtype=dtype).sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        summary = (projected * valid_steps.unsqueeze(-1).to(dtype=dtype)).sum(dim=1) / denom
        coverage = float(mask.float().mean().detach().cpu().item())
        return tokens, padding_mask, summary, {
            "enabled": True,
            "available": bool(valid_steps.any().detach().cpu().item()),
            "coverage": coverage,
            "fields": list(fields),
            "direct_fields": list(fields),
            "proxy_fields": [],
        }


class BottleneckPrivateAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int | None = None):
        super().__init__()
        hidden = int(bottleneck_dim or max(dim // 4, 1))
        self.down = nn.Linear(dim, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.activation(self.down(x)))


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = int(bottleneck_dim or max(dim // 4, 1))
        self.down = nn.Linear(dim, hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.dropout(self.activation(self.down(x))))


class CoarseConditionedPrivateAdapter(nn.Module):
    def __init__(self, dim: int, num_groups: int, bottleneck_dim: int | None = None):
        super().__init__()
        hidden = int(bottleneck_dim or max(dim // 4, 1))
        self.group_embedding = nn.Embedding(int(num_groups), dim)
        self.down = nn.Linear(dim * 2, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, *, coarse_logits: torch.Tensor | None = None) -> torch.Tensor:
        if coarse_logits is None:
            context = torch.zeros_like(x)
        else:
            probs = torch.softmax(coarse_logits.detach().mean(dim=1), dim=-1)
            context = probs @ self.group_embedding.weight
        return x + self.up(self.activation(self.down(torch.cat([x, context], dim=-1))))


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: torch.Tensor, *, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, float(lambda_))


def hierarchical_beam_log_probs(coarse_logits: torch.Tensor, fine_logits: torch.Tensor) -> torch.Tensor:
    if coarse_logits.ndim != 3:
        raise ValueError(f"coarse_logits must have shape [B, H, G], got {tuple(coarse_logits.shape)}.")
    if fine_logits.ndim != 4:
        raise ValueError(f"fine_logits must have shape [B, H, G, S], got {tuple(fine_logits.shape)}.")
    if coarse_logits.shape[:3] != fine_logits.shape[:3]:
        raise ValueError("coarse_logits and fine_logits must share [B, H, G] dimensions.")
    coarse_lp = F.log_softmax(coarse_logits, dim=-1).unsqueeze(-1)
    fine_lp = F.log_softmax(fine_logits, dim=-1)
    return (coarse_lp + fine_lp).reshape(*coarse_logits.shape[:2], -1)


def _build_hist_modality_encoder(
    modality: str,
    feature_size: int,
    *,
    image_channels: int,
    image_encoder: str | dict[str, Any] | None,
    image_profile: str | None,
    radar_channels: int,
    gps_input_size: int,
    lidar_channels: int,
    mmwave_input_size: int,
) -> nn.Module:
    if modality == "image":
        encoder_cfg = image_encoder
        if encoder_cfg is None:
            encoder_cfg = {
                "type": "resnet18_imagenet_rgb",
                "output_dim": feature_size,
                "pretrained": False,
                "weights": None,
            }
        if isinstance(encoder_cfg, str):
            encoder_cfg = {"type": encoder_cfg}
        if isinstance(encoder_cfg, dict) and encoder_cfg.get("type") == "resnet18_imagenet_rgb":
            cfg = dict(encoder_cfg)
            cfg.pop("type", None)
            cfg.setdefault("output_dim", feature_size)
            cfg.setdefault("image_profile", image_profile)
            cfg.setdefault("image_channels", image_channels)
            return ResNet18ImageEncoder(**cfg)
        return ImageFeatureExtractor(feature_size, image_channels)
    if modality == "radar":
        return RadarFeatureExtractor(feature_size, radar_channels)
    if modality == "gps":
        return GpsFeatureExtractor(feature_size, gps_input_size=gps_input_size)
    if modality == "lidar":
        return LidarFeatureExtractor(feature_size, in_channels=lidar_channels)
    if modality == "mmwave":
        return MmWaveFeatureExtractor(feature_size=feature_size, mmwave_input_size=mmwave_input_size)
    available = ", ".join(MODALITY_ORDER)
    raise ValueError(f"Unknown HiST-Beam modality '{modality}'. Available modalities: {available}.")


def _check_temporal_features(
    features: torch.Tensor,
    modality: str,
    batch_size: int | None,
    seq_len: int | None,
) -> tuple[int, int]:
    if features.ndim != 3:
        raise ValueError(f"{modality} features must have shape [B, T, D], got {tuple(features.shape)}.")
    current_batch = int(features.shape[0])
    current_seq = int(features.shape[1])
    if batch_size is not None and (batch_size != current_batch or seq_len != current_seq):
        raise ValueError("Enabled HiST-Beam modalities must share batch and sequence dimensions.")
    return current_batch, current_seq


def _effective_modality_mask(
    batch_size: int,
    modality_count: int,
    *,
    device: torch.device,
    force_modality_mask: torch.Tensor | None,
) -> torch.Tensor:
    mask = torch.ones(batch_size, modality_count, dtype=torch.bool, device=device)
    if force_modality_mask is None:
        return mask
    forced = force_modality_mask.to(device=device, dtype=torch.bool)
    if forced.ndim == 1:
        if forced.shape[0] != modality_count:
            raise ValueError(f"force_modality_mask shape must be [K] or [B, K], got {tuple(forced.shape)}.")
        forced = forced.unsqueeze(0).expand(batch_size, -1)
    if forced.shape != mask.shape:
        raise ValueError(f"force_modality_mask shape must be {tuple(mask.shape)}, got {tuple(forced.shape)}.")
    return mask & forced


def _serialize_time_first(tokens: torch.Tensor) -> torch.Tensor:
    batch_size, modality_count, seq_len, d_model = tokens.shape
    return tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len * modality_count, d_model)


def _serialize_mask_time_first(mask: torch.Tensor) -> torch.Tensor:
    batch_size, modality_count, seq_len = mask.shape
    return mask.permute(0, 2, 1).contiguous().view(batch_size, seq_len * modality_count)


def _available_timewise_mean(tokens: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
    valid = effective_mask.to(device=tokens.device, dtype=tokens.dtype).view(tokens.shape[0], tokens.shape[1], 1, 1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return (tokens * valid).sum(dim=1) / counts


def _mapping_enabled(value: bool | dict[str, Any] | None) -> bool:
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return bool(value)


def _adapter_bottleneck_dim(adapter: bool | dict[str, Any] | None, dim: int) -> int:
    if isinstance(adapter, dict):
        return int(adapter.get("bottleneck_dim", adapter.get("hidden_dim", max(dim // 4, 1))))
    return max(dim // 4, 1)


def _num_v8_sectors(num_classes: int, sector_size: int) -> int:
    return (int(num_classes) + int(sector_size) - 1) // int(sector_size)


def _target_prototype_logits(
    features: torch.Tensor,
    *,
    target_prototypes: torch.Tensor | None,
    target_prototype_counts: torch.Tensor | None,
    num_classes: int,
    num_pred: int,
    prototype_type: str,
    sector_size: int,
    tau: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    zeros = features.new_zeros(features.shape[0], int(num_pred), int(num_classes))
    if prototype_type == "none":
        return zeros, {"available": False, "unavailable_reason": "prototype_type_none", "support_count": 0}
    if target_prototypes is None:
        return zeros, {"available": False, "unavailable_reason": "target_prototypes_missing", "support_count": 0}
    proto = target_prototypes.to(device=features.device, dtype=features.dtype)
    if proto.ndim != 2 or proto.shape[-1] != features.shape[-1]:
        return zeros, {
            "available": False,
            "unavailable_reason": f"prototype_shape_mismatch:{tuple(proto.shape)}",
            "support_count": 0,
        }
    counts = (
        target_prototype_counts.to(device=features.device).reshape(-1)
        if target_prototype_counts is not None
        else torch.ones(proto.shape[0], dtype=torch.long, device=features.device)
    )
    if counts.numel() != proto.shape[0]:
        return zeros, {
            "available": False,
            "unavailable_reason": f"prototype_count_shape_mismatch:{tuple(counts.shape)}",
            "support_count": 0,
        }
    available = counts.gt(0)
    if not bool(available.any().detach().cpu().item()):
        return zeros, {"available": False, "unavailable_reason": "no_available_target_prototypes", "support_count": 0}
    scores = F.normalize(features, dim=-1) @ F.normalize(proto, dim=-1).t()
    scores = scores / max(float(tau), 1e-6)
    scores = scores.masked_fill(~available.view(1, -1), -1e9)
    if prototype_type == "beam":
        if proto.shape[0] != int(num_classes):
            return zeros, {
                "available": False,
                "unavailable_reason": f"beam_prototype_count_mismatch:{proto.shape[0]}",
                "support_count": int(counts[available].sum().detach().cpu().item()),
            }
        logits = scores.unsqueeze(1).expand(-1, int(num_pred), -1).contiguous()
    elif prototype_type == "sector":
        num_sectors = _num_v8_sectors(int(num_classes), int(sector_size))
        if proto.shape[0] != num_sectors:
            return zeros, {
                "available": False,
                "unavailable_reason": f"sector_prototype_count_mismatch:{proto.shape[0]}!={num_sectors}",
                "support_count": int(counts[available].sum().detach().cpu().item()),
            }
        beam_to_sector = torch.div(
            torch.arange(int(num_classes), device=features.device),
            int(sector_size),
            rounding_mode="floor",
        ).clamp(max=num_sectors - 1)
        logits = scores[:, beam_to_sector].unsqueeze(1).expand(-1, int(num_pred), -1).contiguous()
    else:
        return zeros, {"available": False, "unavailable_reason": f"unsupported_prototype_type:{prototype_type}", "support_count": 0}
    return logits, {
        "available": True,
        "unavailable_reason": None,
        "support_count": int(counts[available].sum().detach().cpu().item()),
        "support_counts": [int(item) for item in counts.detach().cpu().tolist()],
    }


def _inverse_sigmoid_clamped(value: float) -> float:
    clipped = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return float(torch.logit(torch.tensor(clipped)).item())


def _labels_to_1d_tensor(labels: torch.Tensor | list[int] | tuple[int, ...] | None, *, device: torch.device) -> torch.Tensor:
    if labels is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if torch.is_tensor(labels):
        return labels.detach().to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(list(labels), dtype=torch.long, device=device).reshape(-1)


def _top_beam_records(values: torch.Tensor, *, top_k: int) -> list[dict[str, float | int]]:
    tensor = values.detach().cpu().reshape(-1).to(torch.float32)
    if tensor.numel() == 0:
        return []
    count = min(int(top_k), int(tensor.numel()))
    scores, indices = torch.topk(tensor, k=count)
    return [{"beam": int(idx.item()), "value": float(score.item())} for score, idx in zip(scores, indices)]


__all__ = [
    "DEFAULT_HIST_MODALITIES",
    "HIST_BEAM_VARIANTS",
    "BottleneckAdapter",
    "BottleneckPrivateAdapter",
    "HistBeamConfig",
    "HistBeamFusionNet",
    "gradient_reverse",
    "hierarchical_beam_log_probs",
    "resolve_hist_beam_config",
]
