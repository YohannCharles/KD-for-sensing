from __future__ import annotations

import copy
from typing import Any

from kd_sensing.config.canonical import SNAPSHOT_TRAIN_CSV, SNAPSHOT_VAL_CSV, SNAPSHOT_VARIANT
from kd_sensing.config.lidar_normalization import canonicalize_lidar_normalization_config
from kd_sensing.data.scenes import normalize_deepsense_config
from kd_sensing.engine.objectives.metadata import (
    configure_objective_defaults,
    objective_requires_occlusion,
    objective_requires_position,
    resolve_prediction_objective,
)
from kd_sensing.modalities import (
    dataset_defaults_for_modalities,
    dataset_flags_for_modalities,
    image_profile_spec,
    model_defaults_for_modalities,
    normalize_modalities,
    resolve_image_profile,
)

IMAGE_MODEL_TYPES = {
    "cls_token_transformer_fusion",
    "token_transformer_fusion",
    "gps_conditioned_jepa",
    "vision_position_late_fusion",
    "vision_position_transformer_fusion",
    "bev_fusion_2604",
    "modular_sequence",
}
MODULAR_MODEL_TYPES = {"modular_sequence"}
ENCODER_CONFIG_MODEL_TYPES = set(MODULAR_MODEL_TYPES)
MODULAR_ROLE_ONLY_KEYS = {
    "encoders",
    "projectors",
    "representation_core",
    "heads",
    "image_profile",
}
FUSION_MODEL_TYPES = {
    "cls_token_transformer_fusion",
    "token_transformer_fusion",
    "gps_conditioned_jepa",
    "gps_sequence_baseline",
    "vision_position_late_fusion",
    "vision_position_transformer_fusion",
    "bev_fusion_2604",
}
AUXILIARY_HEAD_MODEL_TYPES = {
    "cls_token_transformer_fusion",
    "modular_sequence",
}
D_MODEL_ROLE_TYPES = {
    "cls_token_transformer_fusion",
    "token_transformer_fusion",
    *MODULAR_MODEL_TYPES,
    "gps_sequence_baseline",
    "vision_position_late_fusion",
    "vision_position_transformer_fusion",
    "bev_fusion_2604",
}


def normalize_loaded_config(
    cfg: dict[str, Any],
    *,
    file_cfg: dict[str, Any],
    override_cfg: dict[str, Any],
    explicit_early_stopping_metric: bool,
    explicit_early_stopping_mode: bool,
) -> None:
    configure_objective_defaults(
        cfg,
        explicit_early_stopping_metric=explicit_early_stopping_metric,
        explicit_early_stopping_mode=explicit_early_stopping_mode,
    )
    apply_objective_runtime_requirements(cfg)
    apply_fusion_modality_selection(cfg, override_cfg=override_cfg)
    normalize_dataloader_batch_size_alias(cfg, file_cfg=file_cfg, override_cfg=override_cfg)
    normalize_csi_hardening_alias(cfg)
    canonicalize_lidar_normalization_config(cfg, file_cfg=file_cfg, override_cfg=override_cfg)
    normalize_model_role_defaults(cfg)
    normalize_deepsense_config(cfg)
    normalize_image_profile_config(cfg)
    apply_snapshot_runtime_requirements(cfg)


def normalize_dataloader_batch_size_alias(
    cfg: dict[str, Any],
    *,
    file_cfg: dict[str, Any],
    override_cfg: dict[str, Any],
) -> None:
    """Map explicit dataloader.batch_size aliases onto split-aware sizes.

    The default config carries train/test batch-size defaults. Without this
    normalization, a task config that only sets data.dataloader.batch_size keeps
    the default train_batch_size/test_batch_size values and silently trains with
    the wrong batch size.
    """

    loader_cfg = cfg.setdefault("data", {}).setdefault("dataloader", {})
    if not isinstance(loader_cfg, dict):
        return
    file_loader = _dataloader_cfg(file_cfg)
    override_loader = _dataloader_cfg(override_cfg)

    batch_size = None
    batch_source = None
    if isinstance(file_loader, dict) and "batch_size" in file_loader:
        batch_size = copy.deepcopy(file_loader["batch_size"])
        batch_source = "file"
    if isinstance(override_loader, dict) and "batch_size" in override_loader:
        batch_size = copy.deepcopy(override_loader["batch_size"])
        batch_source = "override"
    if batch_source is None:
        return

    for split in ("train", "test"):
        if _has_explicit_split_batch_size(override_loader, split):
            continue
        if batch_source == "file" and _has_explicit_split_batch_size(file_loader, split):
            continue
        loader_cfg[f"{split}_batch_size"] = copy.deepcopy(batch_size)


def _dataloader_cfg(source: dict[str, Any] | None) -> dict[str, Any] | None:
    data_cfg = source.get("data") if isinstance(source, dict) else None
    loader_cfg = data_cfg.get("dataloader") if isinstance(data_cfg, dict) else None
    return loader_cfg if isinstance(loader_cfg, dict) else None


def _has_explicit_split_batch_size(loader_cfg: dict[str, Any] | None, split: str) -> bool:
    if not isinstance(loader_cfg, dict):
        return False
    if f"{split}_batch_size" in loader_cfg:
        return True
    split_cfg = loader_cfg.get(split)
    return isinstance(split_cfg, dict) and "batch_size" in split_cfg


def apply_fusion_modality_selection(cfg: dict[str, Any], *, override_cfg: dict[str, Any] | None = None) -> None:
    """Let fusion configs select modalities once via model.modalities."""

    if cfg.get("experiment", {}).get("task", "image") != "fusion":
        return
    model_cfg = cfg.setdefault("model", {})
    selected_raw = modalities_from_role_overrides(override_cfg) or model_cfg.get("modalities")
    if selected_raw is None:
        return
    selected = list(normalize_modalities(selected_raw, context="model.modalities"))
    model_cfg["modalities"] = selected
    model_defaults = model_defaults_for_modalities(selected)
    primary_cfg = model_cfg.get("primary")
    if isinstance(primary_cfg, dict):
        primary_cfg["modalities"] = list(selected)
        for key, value in model_defaults.items():
            primary_cfg.setdefault(key, value)
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    dataset_cfg.update(dataset_flags_for_modalities(selected))
    for key, value in dataset_defaults_for_modalities(selected).items():
        dataset_cfg.setdefault(key, value)


def normalize_csi_hardening_alias(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    alias = dataset_cfg.get("csi_hardening")
    if alias is None:
        return
    model_cfg = cfg.setdefault("model", {})
    primary_cfg = model_cfg.get("primary")
    if not isinstance(primary_cfg, dict) or "csi" not in primary_cfg.get("modalities", []):
        return
    encoders = primary_cfg.setdefault("encoders", {})
    if isinstance(encoders, dict):
        csi_cfg = encoders.setdefault("csi", {"type": "pilot_dual_view_csi"})
        if isinstance(csi_cfg, dict) and "csi_hardening" not in csi_cfg:
            csi_cfg["csi_hardening"] = copy.deepcopy(alias)
            return
    primary_cfg.setdefault("csi_hardening", copy.deepcopy(alias))


def apply_objective_runtime_requirements(cfg: dict[str, Any]) -> None:
    objective = resolve_prediction_objective(cfg)
    if objective == "beam":
        return
    if objective in {
        "current_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
    }:
        ensure_objective_loss_defaults(cfg, objective)
        return
    if objective == "gps_conditioned_jepa":
        ensure_jepa_runtime_requirements(cfg)
        ensure_objective_loss_defaults(cfg, objective)
        return
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    if objective_requires_occlusion(cfg):
        ensure_occlusion_target(dataset_cfg)
    if objective_requires_position(cfg):
        ensure_position_target(dataset_cfg)
    ensure_primary_auxiliary_heads(cfg, objective)
    ensure_objective_loss_defaults(cfg, objective)


def ensure_occlusion_target(dataset_cfg: dict[str, Any]) -> None:
    target = dataset_cfg.get("occlusion_target")
    if target is None:
        dataset_cfg["occlusion_target"] = {"enabled": True, "threshold_percentile": 20.0}
    elif isinstance(target, dict):
        target.setdefault("enabled", True)


def ensure_position_target(dataset_cfg: dict[str, Any]) -> None:
    target = dataset_cfg.get("position_target")
    if target is None:
        dataset_cfg["position_target"] = {
            "enabled": True,
            "source": "future_gps_local_xy",
            "normalize": True,
        }
    elif isinstance(target, dict):
        target.setdefault("enabled", True)
        target.setdefault("source", "future_gps_local_xy")
        target.setdefault("normalize", True)
    switch_default_position_csv(dataset_cfg)


def switch_default_position_csv(dataset_cfg: dict[str, Any]) -> None:
    replacements = {
        "train_csv_name": ("train_seqs_RA_GPS_LIDAR.csv", "train_seqs_RA_GPS_LIDAR_POS.csv"),
        "test_csv_name": ("test_seqs_RA_GPS_LIDAR.csv", "test_seqs_RA_GPS_LIDAR_POS.csv"),
    }
    for key, (default_name, position_name) in replacements.items():
        if dataset_cfg.get(key) in (None, default_name):
            dataset_cfg[key] = position_name


def ensure_primary_auxiliary_heads(cfg: dict[str, Any], objective: str) -> None:
    model_cfg = cfg.setdefault("model", {})
    primary_cfg = model_cfg.setdefault("primary", {})
    raw = primary_cfg.get("auxiliary_heads")
    if isinstance(raw, dict):
        heads = raw
    elif raw is None:
        heads = {}
    else:
        heads = {"enabled": bool(raw)}
    if objective in {"occlusion", "multitask"}:
        heads["occlusion"] = True
    if objective in {"position", "multitask"}:
        heads["position"] = True
    heads["enabled"] = bool(heads.get("occlusion", False) or heads.get("position", False) or heads.get("enabled", False))
    primary_cfg["auxiliary_heads"] = heads
    primary_cfg.setdefault(
        "num_pred",
        int(model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 3))),
    )


def ensure_objective_loss_defaults(cfg: dict[str, Any], objective: str) -> None:
    loss_cfg = cfg.setdefault("loss", {})
    objective_cfg = loss_cfg.setdefault("objective", {})
    weights_cfg = objective_cfg.setdefault("weights", {})
    if objective == "gps_conditioned_jepa":
        loss_cfg.setdefault("jepa", {"type": "mse", "latent_normalize": False, "weight": 1.0})
        weights_cfg.setdefault("jepa", 1.0)
        return
    if objective == "selection_multitask":
        weights_cfg.setdefault("beam_selection", 1.0)
        weights_cfg.setdefault("los", 0.5)
        weights_cfg.setdefault("link_quality", 0.2)
        objective_cfg.setdefault("los", {}).setdefault("pos_weight", None)
        objective_cfg.setdefault("link_quality", {}).setdefault("type", "smooth_l1")
        return
    if objective == "current_beam_selection":
        weights_cfg.setdefault("beam_selection", 1.0)
        return
    if objective == "current_los_classification":
        objective_cfg.setdefault("los", {}).setdefault("pos_weight", None)
        return
    if objective == "current_link_quality":
        objective_cfg.setdefault("link_quality", {}).setdefault("type", "smooth_l1")
        return
    weights_cfg.setdefault("beam", 1.0)
    weights_cfg.setdefault("occlusion", 1.0)
    weights_cfg.setdefault("position", 1.0 if objective in {"position", "multitask"} else 0.01)
    if objective in {"occlusion", "multitask"}:
        objective_cfg.setdefault("occlusion", {}).setdefault("pos_weight", "auto")
    if objective in {"position", "multitask"}:
        objective_cfg.setdefault("position", {}).setdefault("type", "mse")


def ensure_jepa_runtime_requirements(cfg: dict[str, Any]) -> None:
    model_cfg = cfg.setdefault("model", {})
    model_cfg["modalities"] = ["image", "gps"]
    primary_cfg = model_cfg.setdefault("primary", {})
    primary_cfg.setdefault("type", "gps_conditioned_jepa")
    primary_cfg["modalities"] = ["image", "gps"]
    primary_cfg.setdefault("image_profile", "rgb_imagenet")
    primary_cfg.setdefault("image_channels", 3)
    primary_cfg.setdefault("gps_input_size", 3)
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    dataset_cfg["use_gps"] = True
    dataset_cfg.setdefault("gps_feature_mode", "relative_polar")
    dataset_cfg.setdefault("gps_normalize", True)
    dataset_cfg.setdefault("image_profile", "rgb_imagenet")


def modalities_from_role_overrides(override_cfg: dict[str, Any] | None) -> list[str] | None:
    if not isinstance(override_cfg, dict):
        return None
    override_model = override_cfg.get("model")
    if not isinstance(override_model, dict) or "modalities" in override_model:
        return None
    primary_cfg = override_model.get("primary")
    if isinstance(primary_cfg, dict) and primary_cfg.get("modalities") is not None:
        return list(normalize_modalities(primary_cfg["modalities"], context="model.primary.modalities"))
    return None


def normalize_model_role_defaults(cfg: dict[str, Any]) -> None:
    """Remove modular default-only fields after a config selects a different model type."""

    model_cfg = cfg.setdefault("model", {})
    for _, role_cfg in iter_model_configs(cfg):
        model_type = str(role_cfg.get("type", ""))
        if model_type in MODULAR_MODEL_TYPES:
            continue
        for key in MODULAR_ROLE_ONLY_KEYS:
            if key == "encoders" and model_type in ENCODER_CONFIG_MODEL_TYPES:
                continue
            role_cfg.pop(key, None)
        if model_type not in FUSION_MODEL_TYPES:
            role_cfg.pop("modalities", None)
        if model_type not in D_MODEL_ROLE_TYPES:
            role_cfg.pop("d_model", None)
            if not keeps_auxiliary_num_pred(model_type, role_cfg):
                role_cfg.pop("num_pred", None)


def normalize_image_profile_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    raw_profile = dataset_cfg.get("image_profile")
    profile = resolve_image_profile(raw_profile)
    dataset_cfg["image_profile"] = profile
    if uses_image(cfg):
        spec = image_profile_spec(profile)
        dataset_cfg.setdefault("image_size", list(spec.default_size))
        model_cfg = cfg.setdefault("model", {})
        model_cfg["image_profile"] = profile
        for _, role_cfg in iter_model_configs(cfg):
            role_cfg.setdefault("image_profile", profile)
            role_cfg.setdefault("image_channels", spec.channels)


def apply_snapshot_runtime_requirements(cfg: dict[str, Any]) -> None:
    experiment = cfg.setdefault("experiment", {})
    explicit_variant = experiment.get("variant")
    if explicit_variant is None and uses_snapshot_frame_core(cfg):
        experiment["variant"] = SNAPSHOT_VARIANT
    if experiment.get("variant") != SNAPSHOT_VARIANT:
        return
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    model_cfg = cfg.setdefault("model", {})
    require_snapshot_int(dataset_cfg, "seq_len", 1, "data.dataset.seq_len")
    require_snapshot_int(dataset_cfg, "num_pred", 1, "data.dataset.num_pred")
    require_snapshot_int(model_cfg, "seq_length", 1, "model.seq_length")
    require_snapshot_int(model_cfg, "num_pred", 1, "model.num_pred")
    if dataset_cfg.get("train_csv_name") in (None, ""):
        dataset_cfg["train_csv_name"] = SNAPSHOT_TRAIN_CSV
    if dataset_cfg.get("val_csv_name") in (None, ""):
        dataset_cfg["val_csv_name"] = SNAPSHOT_VAL_CSV
    if dataset_cfg.get("test_csv_name") in (None, ""):
        dataset_cfg["test_csv_name"] = SNAPSHOT_VAL_CSV
    if dataset_cfg.get("train_csv_name") != SNAPSHOT_TRAIN_CSV:
        raise ValueError(
            f"snapshot_next_frame requires data.dataset.train_csv_name={SNAPSHOT_TRAIN_CSV!r}; "
            "run the snapshot preprocessing or explicitly change experiment.variant to leave snapshot mode."
        )
    val_csv = dataset_cfg.get("val_csv_name") or dataset_cfg.get("test_csv_name")
    if val_csv != SNAPSHOT_VAL_CSV:
        raise ValueError(
            f"snapshot_next_frame requires data.dataset.val_csv_name={SNAPSHOT_VAL_CSV!r}; "
            "run the snapshot preprocessing or explicitly change experiment.variant to leave snapshot mode."
        )
    primary_cfg = model_cfg.get("primary")
    if not isinstance(primary_cfg, dict):
        raise ValueError("snapshot_next_frame requires model.primary.")
    if str(primary_cfg.get("type")) not in MODULAR_MODEL_TYPES:
        raise ValueError("snapshot_next_frame requires model.primary.type='modular_sequence' with snapshot_frame core.")
    core_type = str(primary_cfg.get("representation_core", {}).get("type", ""))
    if core_type != "snapshot_frame":
        raise ValueError(
            f"snapshot_next_frame requires model.primary.representation_core.type='snapshot_frame', got {core_type!r}."
        )
    primary_cfg["num_pred"] = 1
    primary_cfg["uses_temporal_core"] = False
    experiment["uses_history_window"] = False
    experiment["uses_temporal_core"] = False


def uses_snapshot_frame_core(cfg: dict[str, Any]) -> bool:
    for _, role_cfg in iter_model_configs(cfg):
        core = role_cfg.get("representation_core")
        if isinstance(core, dict) and core.get("type") == "snapshot_frame":
            return True
    return False


def require_snapshot_int(mapping: dict[str, Any], key: str, expected: int, dotted_key: str) -> None:
    actual = mapping.get(key)
    if int(actual) != int(expected):
        raise ValueError(
            f"snapshot_next_frame requires {dotted_key}={expected}; got {actual!r}. "
            "Use seq_len=1 and num_pred=1, or change experiment.variant to leave snapshot mode."
        )


def mapping_or_bool_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return False


def auxiliary_head_enabled(model_cfg: dict[str, Any], name: str) -> bool:
    heads_raw = model_cfg.get("auxiliary_heads")
    if isinstance(heads_raw, bool):
        return heads_raw
    heads = heads_raw if isinstance(heads_raw, dict) else {}
    aliases = {
        "occlusion": ("occlusion", "occlusion_head"),
        "position": ("position", "position_head"),
    }[name]
    return bool(heads.get(aliases[0], heads.get(aliases[1], heads.get("enabled", False))))


def keeps_auxiliary_num_pred(model_type: str, model_cfg: dict[str, Any]) -> bool:
    return model_supports_auxiliary_heads(model_type) and (
        auxiliary_head_enabled(model_cfg, "occlusion") or auxiliary_head_enabled(model_cfg, "position")
    )


def model_supports_auxiliary_heads(model_type: str) -> bool:
    return str(model_type) in AUXILIARY_HEAD_MODEL_TYPES


def image_encoder_type(model_cfg: dict[str, Any]) -> str | None:
    image_encoder = model_cfg.get("image_encoder")
    if isinstance(image_encoder, str):
        return image_encoder
    if isinstance(image_encoder, dict) and "type" in image_encoder:
        return str(image_encoder["type"])
    encoders = model_cfg.get("encoders")
    if not isinstance(encoders, dict):
        return None
    image_cfg = encoders.get("image")
    if isinstance(image_cfg, str):
        return image_cfg
    if isinstance(image_cfg, dict) and "type" in image_cfg:
        return str(image_cfg["type"])
    return None


def iter_model_configs(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {})
    if isinstance(primary_cfg, dict):
        yield "model.primary", primary_cfg


def uses_image(cfg: dict[str, Any]) -> bool:
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "image":
        return True
    return task == "fusion" and "image" in fusion_modalities(cfg)


def uses_radar(cfg: dict[str, Any]) -> bool:
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "radar":
        return True
    return task == "fusion" and "radar" in fusion_modalities(cfg)


def fusion_modalities(cfg: dict[str, Any]) -> set[str]:
    top_level_modalities = cfg.get("model", {}).get("modalities")
    if top_level_modalities:
        return set(normalize_modalities(top_level_modalities, context="model.modalities"))
    primary_modalities = cfg.get("model", {}).get("primary", {}).get("modalities")
    return set(str(name) for name in primary_modalities or [])
