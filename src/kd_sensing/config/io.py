"""YAML config loading and command-line override parsing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    yaml = None

from kd_sensing.config.canonical import SNAPSHOT_TRAIN_CSV, SNAPSHOT_VAL_CSV, SNAPSHOT_VARIANT, build_virtual_config
from kd_sensing.config.defaults import DEFAULT_CONFIG
from kd_sensing.data.scenes import normalize_deepsense_config
from kd_sensing.engine.prediction_objectives import (
    configure_objective_defaults,
    objective_requires_occlusion,
    objective_requires_position,
    resolve_prediction_objective,
)
from kd_sensing.modalities import (
    REMOVED_IMAGE_ENCODERS,
    dataset_defaults_for_modalities,
    dataset_flags_for_modalities,
    image_profile_spec,
    resolve_image_profile,
    model_defaults_for_modalities,
    normalize_modalities,
    validate_image_encoder_profile,
    validate_image_profile_size,
)
from kd_sensing.utils.paths import resolve_path

IMAGE_MODEL_TYPES = {
    "image_teacher",
    "image_student",
    "fusion_teacher",
    "fusion_student",
    "craf_fusion",
    "cls_token_transformer_fusion",
    "marf_fusion",
    "token_transformer_fusion",
}
MODULAR_MODEL_TYPES = {"modular_sequence", "modular_sequence_model"}
MODULAR_ROLE_ONLY_KEYS = {
    "encoders",
    "projectors",
    "representation_core",
    "heads",
    "image_profile",
}
FUSION_MODEL_TYPES = {
    "fusion_teacher",
    "fusion_student",
    "craf_fusion",
    "cls_token_transformer_fusion",
    "marf_fusion",
    "token_transformer_fusion",
}
AUXILIARY_HEAD_MODEL_TYPES = {
    "cls_token_transformer_fusion",
    "modular_sequence",
    "modular_sequence_model",
    "gps_teacher",
    "gps_student",
    "radar_teacher",
    "radar_student",
    "mmwave_teacher",
    "mmwave_student",
}
D_MODEL_ROLE_TYPES = {
    "craf_fusion",
    "cls_token_transformer_fusion",
    "marf_fusion",
    "token_transformer_fusion",
    *MODULAR_MODEL_TYPES,
}
REMOVED_IMAGE_OPTION_PREFIX = "image_" + "motion_"


def load_config(config_path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = resolve_path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                file_cfg = safe_load_yaml(f.read()) or {}
        else:
            file_cfg = build_virtual_config(path)
            if file_cfg is None:
                raise FileNotFoundError(f"Config file not found: {path}")
        cfg = deep_merge(cfg, file_cfg)
    override_cfg = parse_overrides(overrides) if overrides else {}
    if override_cfg:
        cfg = deep_merge(cfg, override_cfg)
    file_cfg_for_keys = file_cfg if config_path else {}
    override_changes_objective = _has_dotted_key(override_cfg, "experiment.objective")
    explicit_early_metric = _has_dotted_key(override_cfg, "training.early_stopping_metric") or (
        _has_dotted_key(file_cfg_for_keys, "training.early_stopping_metric") and not override_changes_objective
    )
    explicit_early_mode = _has_dotted_key(override_cfg, "training.early_stopping_mode") or (
        _has_dotted_key(file_cfg_for_keys, "training.early_stopping_mode") and not override_changes_objective
    )
    configure_objective_defaults(
        cfg,
        explicit_early_stopping_metric=explicit_early_metric,
        explicit_early_stopping_mode=explicit_early_mode,
    )
    apply_objective_runtime_requirements(cfg)
    reject_removed_image_path_config(cfg)
    apply_fusion_modality_selection(cfg, override_cfg=override_cfg)
    normalize_model_role_defaults(cfg)
    normalize_deepsense_config(cfg)
    normalize_image_profile_config(cfg)
    apply_snapshot_runtime_requirements(cfg)
    validate_config(cfg)
    return cfg


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            json.dump(cfg, f, indent=2)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in overrides:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override must use key=value format, got: {item}")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(result, key.strip(), parse_scalar(raw_value.strip()))
    return result


def set_by_dotted_key(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid dotted override key: {key}")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set nested override through non-dict key: {key}")
    cursor[parts[-1]] = value


def _has_dotted_key(target: dict[str, Any], key: str) -> bool:
    cursor: Any = target
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        body = raw[1:-1].strip()
        if not body:
            return []
        return [parse_scalar(item.strip()) for item in body.split(",")]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def safe_load_yaml(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text)
    return parse_simple_yaml(text)


def apply_fusion_modality_selection(cfg: dict[str, Any], *, override_cfg: dict[str, Any] | None = None) -> None:
    """Let fusion configs select modalities once via model.modalities."""

    if cfg.get("experiment", {}).get("task", "image") != "fusion":
        return
    model_cfg = cfg.setdefault("model", {})
    selected_raw = _modalities_from_role_overrides(override_cfg) or model_cfg.get("modalities")
    if selected_raw is None:
        return
    selected = list(normalize_modalities(selected_raw, context="model.modalities"))
    model_cfg["modalities"] = selected
    model_defaults = model_defaults_for_modalities(selected)
    for role in ("teacher", "student"):
        role_cfg = model_cfg.get(role)
        if not isinstance(role_cfg, dict):
            continue
        role_cfg["modalities"] = list(selected)
        for key, value in model_defaults.items():
            role_cfg.setdefault(key, value)
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    dataset_cfg.update(dataset_flags_for_modalities(selected))
    for key, value in dataset_defaults_for_modalities(selected).items():
        dataset_cfg.setdefault(key, value)


def apply_objective_runtime_requirements(cfg: dict[str, Any]) -> None:
    objective = resolve_prediction_objective(cfg)
    if objective == "beam":
        return
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    if objective_requires_occlusion(cfg):
        _ensure_occlusion_target(dataset_cfg)
    if objective_requires_position(cfg):
        _ensure_position_target(dataset_cfg)
    _ensure_student_auxiliary_heads(cfg, objective)
    _ensure_objective_loss_defaults(cfg, objective)


def _ensure_occlusion_target(dataset_cfg: dict[str, Any]) -> None:
    target = dataset_cfg.get("occlusion_target")
    if target is None:
        dataset_cfg["occlusion_target"] = {"enabled": True, "threshold_percentile": 20.0}
    elif isinstance(target, dict):
        target.setdefault("enabled", True)


def _ensure_position_target(dataset_cfg: dict[str, Any]) -> None:
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
    _switch_default_position_csv(dataset_cfg)


def _switch_default_position_csv(dataset_cfg: dict[str, Any]) -> None:
    replacements = {
        "train_csv_name": ("train_seqs_RA_GPS_LIDAR.csv", "train_seqs_RA_GPS_LIDAR_POS.csv"),
        "test_csv_name": ("test_seqs_RA_GPS_LIDAR.csv", "test_seqs_RA_GPS_LIDAR_POS.csv"),
    }
    for key, (default_name, position_name) in replacements.items():
        if dataset_cfg.get(key) in (None, default_name):
            dataset_cfg[key] = position_name


def _ensure_student_auxiliary_heads(cfg: dict[str, Any], objective: str) -> None:
    model_cfg = cfg.setdefault("model", {})
    student_cfg = model_cfg.setdefault("student", {})
    raw = student_cfg.get("auxiliary_heads")
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
    student_cfg["auxiliary_heads"] = heads
    student_cfg.setdefault(
        "num_pred",
        int(model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 3))),
    )


def _ensure_objective_loss_defaults(cfg: dict[str, Any], objective: str) -> None:
    loss_cfg = cfg.setdefault("loss", {})
    objective_cfg = loss_cfg.setdefault("objective", {})
    weights_cfg = objective_cfg.setdefault("weights", {})
    weights_cfg.setdefault("beam", 1.0)
    weights_cfg.setdefault("occlusion", 1.0)
    weights_cfg.setdefault("position", 1.0 if objective in {"position", "multitask"} else 0.01)
    if objective in {"occlusion", "multitask"}:
        objective_cfg.setdefault("occlusion", {}).setdefault("pos_weight", "auto")
    if objective in {"position", "multitask"}:
        objective_cfg.setdefault("position", {}).setdefault("type", "mse")


def _modalities_from_role_overrides(override_cfg: dict[str, Any] | None) -> list[str] | None:
    if not isinstance(override_cfg, dict):
        return None
    override_model = override_cfg.get("model")
    if not isinstance(override_model, dict) or "modalities" in override_model:
        return None
    role_modalities = []
    for role in ("teacher", "student"):
        role_cfg = override_model.get(role)
        if isinstance(role_cfg, dict) and role_cfg.get("modalities") is not None:
            role_modalities.append(list(normalize_modalities(role_cfg["modalities"], context=f"model.{role}.modalities")))
    if len(role_modalities) != 2 or role_modalities[0] != role_modalities[1]:
        return None
    return role_modalities[0]


def normalize_model_role_defaults(cfg: dict[str, Any]) -> None:
    """Remove modular default-only fields after a config selects a different model type."""

    model_cfg = cfg.setdefault("model", {})
    for role in ("teacher", "student"):
        role_cfg = model_cfg.get(role)
        if not isinstance(role_cfg, dict):
            continue
        model_type = str(role_cfg.get("type", ""))
        if model_type in MODULAR_MODEL_TYPES:
            continue
        for key in MODULAR_ROLE_ONLY_KEYS:
            role_cfg.pop(key, None)
        if model_type not in FUSION_MODEL_TYPES:
            role_cfg.pop("modalities", None)
        if model_type not in D_MODEL_ROLE_TYPES:
            role_cfg.pop("d_model", None)
            if not _keeps_auxiliary_num_pred(model_type, role_cfg):
                role_cfg.pop("num_pred", None)


def normalize_image_profile_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    raw_profile = dataset_cfg.get("image_profile")
    profile = resolve_image_profile(raw_profile)
    dataset_cfg["image_profile"] = profile
    if _uses_image(cfg):
        spec = image_profile_spec(profile)
        dataset_cfg.setdefault("image_size", list(spec.default_size))
        model_cfg = cfg.setdefault("model", {})
        model_cfg["image_profile"] = profile
        for _, role_cfg in _iter_model_configs(cfg):
            role_cfg.setdefault("image_profile", profile)
            role_cfg.setdefault("image_channels", spec.channels)


def apply_snapshot_runtime_requirements(cfg: dict[str, Any]) -> None:
    experiment = cfg.setdefault("experiment", {})
    explicit_variant = experiment.get("variant")
    if explicit_variant is None and _uses_snapshot_frame_core(cfg):
        experiment["variant"] = SNAPSHOT_VARIANT
    if experiment.get("variant") != SNAPSHOT_VARIANT:
        return
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    model_cfg = cfg.setdefault("model", {})
    _require_snapshot_int(dataset_cfg, "seq_len", 1, "data.dataset.seq_len")
    _require_snapshot_int(dataset_cfg, "num_pred", 1, "data.dataset.num_pred")
    _require_snapshot_int(model_cfg, "seq_length_teacher", 1, "model.seq_length_teacher")
    _require_snapshot_int(model_cfg, "seq_length_student", 1, "model.seq_length_student")
    _require_snapshot_int(model_cfg, "num_pred", 1, "model.num_pred")
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
    for role in ("teacher", "student"):
        role_cfg = model_cfg.get(role)
        if not isinstance(role_cfg, dict):
            continue
        if str(role_cfg.get("type")) not in MODULAR_MODEL_TYPES:
            raise ValueError(
                f"snapshot_next_frame requires model.{role}.type='modular_sequence' with snapshot_frame core."
            )
        core_type = str(role_cfg.get("representation_core", {}).get("type", ""))
        if core_type != "snapshot_frame":
            raise ValueError(
                f"snapshot_next_frame requires model.{role}.representation_core.type='snapshot_frame', got {core_type!r}."
            )
        role_cfg["num_pred"] = 1
        role_cfg["uses_temporal_core"] = False
    distill = cfg.setdefault("distillation", {})
    if distill.get("type", "no_kd") != "no_kd":
        raise ValueError("snapshot_next_frame baselines require distillation.type='no_kd'.")
    distill["teacher_model_name"] = None
    experiment["uses_history_window"] = False
    experiment["uses_temporal_core"] = False


def _uses_snapshot_frame_core(cfg: dict[str, Any]) -> bool:
    for _, role_cfg in _iter_model_configs(cfg):
        core = role_cfg.get("representation_core")
        if isinstance(core, dict) and core.get("type") == "snapshot_frame":
            return True
    return False


def _require_snapshot_int(mapping: dict[str, Any], key: str, expected: int, dotted_key: str) -> None:
    actual = mapping.get(key)
    if int(actual) != int(expected):
        raise ValueError(
            f"snapshot_next_frame requires {dotted_key}={expected}; got {actual!r}. "
            "Use seq_len=1 and num_pred=1, or change experiment.variant to leave snapshot mode."
        )


def reject_removed_image_path_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    removed_dataset_keys = sorted(
        str(key) for key in dataset_cfg if str(key).startswith(REMOVED_IMAGE_OPTION_PREFIX)
    )
    if removed_dataset_keys:
        keys = ", ".join(removed_dataset_keys)
        raise ValueError(
            f"Removed image motion dataset option(s): {keys}. "
            "Image motion cache support has been removed; use RGB/ImageNet image input."
        )
    cache_cfg = cfg.get("data", {}).get("cache", {})
    if isinstance(cache_cfg, dict) and "image" in cache_cfg:
        raise ValueError(
            "Image cache policy has been removed with the image motion path. "
            "Only supported cache modalities, such as LiDAR, may define cache policy overrides."
        )
    image_profile = dataset_cfg.get("image_profile")
    if image_profile is not None:
        resolve_image_profile(image_profile)
    for location, model_cfg in _iter_model_configs(cfg):
        encoder_type = _image_encoder_type(model_cfg)
        if encoder_type in REMOVED_IMAGE_ENCODERS:
            raise ValueError(
                f"Removed image encoder '{encoder_type}' in {location}. "
                "Use 'resnet18_imagenet_rgb' with RGB/ImageNet image input."
            )


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate structural constraints that current model implementations rely on."""

    _validate_prediction_objective_config(cfg)
    cache_policy = str(cfg.get("data", {}).get("cache", {}).get("policy", "auto"))
    _validate_cache_policy(cache_policy, "data.cache.policy")
    if cfg.get("data", {}).get("cache", {}).get("image") is not None:
        raise ValueError(
            "data.cache.image has been removed with the image motion path. "
            "Configure supported cache modalities such as data.cache.lidar instead."
        )
    for modality in ("lidar",):
        modality_policy = cfg.get("data", {}).get("cache", {}).get(modality, {}).get("policy")
        if modality_policy is not None:
            _validate_cache_policy(str(modality_policy), f"data.cache.{modality}.policy")

    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if _uses_image(cfg):
        image_profile = resolve_image_profile(dataset_cfg.get("image_profile"))
        dataset_cfg["image_profile"] = image_profile
        image_size = tuple(dataset_cfg.get("image_size", [224, 224]))
        validate_image_profile_size(image_profile, image_size)
        _validate_image_model_profiles(cfg, image_profile)
    if _uses_radar(cfg):
        radar_size = dataset_cfg.get("radar_size")
        if radar_size is not None and tuple(radar_size) != (128, 64):
            raise ValueError(
                "Current radar branch requires RA/DA input size 128x64, "
                f"got radar_size {list(radar_size)}."
            )
        fft_tuple = tuple(dataset_cfg.get("fft_tuple", [64, 256, 128]))
        clipped_range = int(dataset_cfg.get("clipped_range", 128))
        if len(fft_tuple) < 3 or clipped_range != 128 or int(fft_tuple[0]) != 64 or int(fft_tuple[2]) != 128:
            raise ValueError(
                "Current radar branch requires RA/DA input size 128x64. "
                f"Use clipped_range=128 and fft_tuple first/third values 64/128; "
                f"got clipped_range={clipped_range}, fft_tuple={list(fft_tuple)}."
            )
    _validate_multitask_config(cfg)


def _validate_prediction_objective_config(cfg: dict[str, Any]) -> None:
    objective = resolve_prediction_objective(cfg)
    cfg.setdefault("experiment", {})["objective"] = objective
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {}).get("student", {})
    model_type = str(model_cfg.get("type", ""))
    head_occlusion = _auxiliary_head_enabled(model_cfg, "occlusion")
    head_position = _auxiliary_head_enabled(model_cfg, "position")

    if objective_requires_occlusion(cfg):
        if not _mapping_or_bool_enabled(dataset_cfg.get("occlusion_target")):
            raise ValueError(
                "experiment.objective='occlusion' or 'multitask' requires "
                "data.dataset.occlusion_target.enabled=true. "
                "Enable that target or set experiment.objective='beam'."
            )
        if not _model_supports_auxiliary_heads(model_type) or not head_occlusion:
            raise ValueError(
                "experiment.objective='occlusion' or 'multitask' requires "
                "a student model type with auxiliary head support and "
                "model.student.auxiliary_heads.occlusion=true."
            )

    if objective_requires_position(cfg):
        if not _mapping_or_bool_enabled(dataset_cfg.get("position_target")):
            raise ValueError(
                "experiment.objective='position' or 'multitask' requires "
                "data.dataset.position_target.enabled=true. "
                "Enable that target or set experiment.objective='beam'."
            )
        if not _model_supports_auxiliary_heads(model_type) or not head_position:
            raise ValueError(
                "experiment.objective='position' or 'multitask' requires "
                "a student model type with auxiliary head support and "
                "model.student.auxiliary_heads.position=true."
            )


def _validate_cache_policy(policy: str, key: str) -> None:
    if policy not in {"off", "read_only", "auto", "rebuild"}:
        raise ValueError(f"{key} must be one of off, read_only, auto, or rebuild; got '{policy}'.")


def _validate_image_model_profiles(cfg: dict[str, Any], image_profile: str) -> None:
    for _, role_cfg in _iter_model_configs(cfg):
        model_type = str(role_cfg.get("type", ""))
        if model_type in IMAGE_MODEL_TYPES and "image_channels" in role_cfg:
            validate_image_encoder_profile(
                encoder_name=model_type,
                image_profile=image_profile,
                expected_channels=image_profile_spec(image_profile).channels,
                actual_channels=role_cfg.get("image_channels"),
            )
            continue
        encoder_type = _image_encoder_type(role_cfg)
        if encoder_type is None:
            continue
        expected_channels = None
        if encoder_type == "resnet18_imagenet_rgb":
            expected_channels = 3
        elif encoder_type in REMOVED_IMAGE_ENCODERS:
            raise ValueError(
                f"Removed image encoder '{encoder_type}' is no longer supported. "
                "Use 'resnet18_imagenet_rgb' with RGB/ImageNet image input."
            )
        if expected_channels is not None:
            validate_image_encoder_profile(
                encoder_name=encoder_type,
                image_profile=image_profile,
                expected_channels=expected_channels,
            )


def _validate_multitask_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {}).get("student", {})
    loss_cfg = cfg.get("loss", {})
    occlusion_target = _mapping_or_bool_enabled(dataset_cfg.get("occlusion_target"))
    position_target = _mapping_or_bool_enabled(dataset_cfg.get("position_target"))
    position_cfg = dataset_cfg.get("position_target") if isinstance(dataset_cfg.get("position_target"), dict) else {}
    if position_target:
        source = position_cfg.get("source", position_cfg.get("position_target_source"))
        if source not in {"future_gps_local_xy", "last_input_gps_local_xy"}:
            raise ValueError(
                "data.dataset.position_target.source must be one of: "
                "future_gps_local_xy, last_input_gps_local_xy."
            )

    auxiliary_loss_raw = loss_cfg.get("auxiliary") or loss_cfg.get("multitask") or loss_cfg.get("multi_task") or {}
    auxiliary_loss = auxiliary_loss_raw if isinstance(auxiliary_loss_raw, dict) else {}
    aux_occlusion = _mapping_or_bool_enabled(auxiliary_loss.get("occlusion")) or (
        float(auxiliary_loss.get("occlusion_weight", auxiliary_loss.get("lambda_occlusion", 0.0))) > 0.0
    )
    aux_position = _mapping_or_bool_enabled(auxiliary_loss.get("position")) or (
        float(auxiliary_loss.get("position_weight", auxiliary_loss.get("lambda_position", 0.0))) > 0.0
    )
    if not (occlusion_target or position_target or aux_occlusion or aux_position):
        return

    model_type = str(model_cfg.get("type", ""))
    model_occlusion = _auxiliary_head_enabled(model_cfg, "occlusion")
    model_position = _auxiliary_head_enabled(model_cfg, "position")
    if aux_occlusion or occlusion_target:
        if not _model_supports_auxiliary_heads(model_type) or not model_occlusion:
            raise ValueError(
                "Occlusion auxiliary supervision requires a student model type with auxiliary head support "
                "and model.student.auxiliary_heads.occlusion=true."
            )
    if aux_position or position_target:
        if not _model_supports_auxiliary_heads(model_type) or not model_position:
            raise ValueError(
                "Position auxiliary supervision requires a student model type with auxiliary head support "
                "and model.student.auxiliary_heads.position=true."
            )


def _mapping_or_bool_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return False


def _auxiliary_head_enabled(model_cfg: dict[str, Any], name: str) -> bool:
    heads_raw = model_cfg.get("auxiliary_heads")
    if isinstance(heads_raw, bool):
        return heads_raw
    heads = heads_raw if isinstance(heads_raw, dict) else {}
    aliases = {
        "occlusion": ("occlusion", "occlusion_head"),
        "position": ("position", "position_head"),
    }[name]
    return bool(heads.get(aliases[0], heads.get(aliases[1], heads.get("enabled", False))))


def _keeps_auxiliary_num_pred(model_type: str, model_cfg: dict[str, Any]) -> bool:
    return _model_supports_auxiliary_heads(model_type) and (
        _auxiliary_head_enabled(model_cfg, "occlusion") or _auxiliary_head_enabled(model_cfg, "position")
    )


def _model_supports_auxiliary_heads(model_type: str) -> bool:
    return str(model_type) in AUXILIARY_HEAD_MODEL_TYPES


def _image_encoder_type(model_cfg: dict[str, Any]) -> str | None:
    encoders = model_cfg.get("encoders")
    if not isinstance(encoders, dict):
        return None
    image_cfg = encoders.get("image")
    if isinstance(image_cfg, str):
        return image_cfg
    if isinstance(image_cfg, dict) and "type" in image_cfg:
        return str(image_cfg["type"])
    return None


def _iter_model_configs(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {})
    for role in ("teacher", "student"):
        role_cfg = model_cfg.get(role, {})
        if isinstance(role_cfg, dict):
            yield f"model.{role}", role_cfg


def _uses_image(cfg: dict[str, Any]) -> bool:
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "image":
        return True
    return task == "fusion" and "image" in _fusion_modalities(cfg)


def _uses_radar(cfg: dict[str, Any]) -> bool:
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "radar":
        return True
    return task == "fusion" and "radar" in _fusion_modalities(cfg)


def _fusion_modalities(cfg: dict[str, Any]) -> set[str]:
    top_level_modalities = cfg.get("model", {}).get("modalities")
    if top_level_modalities:
        return set(normalize_modalities(top_level_modalities, context="model.modalities"))
    modalities: set[str] = set()
    for role in ("teacher", "student"):
        role_modalities = cfg.get("model", {}).get(role, {}).get("modalities")
        if role_modalities:
            modalities.update(str(name) for name in role_modalities)
    return modalities


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple nested mapping subset used by this repo's configs."""

    lines: list[tuple[int, str, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip(), raw_line))

    def parse_node(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        current_indent, stripped, _ = lines[index]
        if current_indent < indent:
            return {}, index
        if stripped.startswith("- "):
            return parse_list(index, current_indent)
        return parse_mapping(index, current_indent)

    def parse_mapping(index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(lines):
            current_indent, stripped, raw_line = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected nested YAML line: {raw_line}")
            if stripped.startswith("- "):
                break
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML line without ':': {raw_line}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value == "":
                if (
                    index < len(lines)
                    and lines[index][0] == current_indent
                    and lines[index][1].startswith("- ")
                ):
                    result[key], index = parse_node(index, lines[index][0])
                elif index >= len(lines) or lines[index][0] <= current_indent:
                    result[key] = {}
                else:
                    result[key], index = parse_node(index, lines[index][0])
            else:
                result[key] = parse_scalar(value)
        return result, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(lines):
            current_indent, stripped, raw_line = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected nested YAML list line: {raw_line}")
            if not stripped.startswith("- "):
                break
            value = stripped[2:].strip()
            index += 1
            if value == "":
                if index >= len(lines) or lines[index][0] <= current_indent:
                    result.append(None)
                else:
                    child, index = parse_node(index, lines[index][0])
                    result.append(child)
            elif ":" in value and not value.startswith(("http://", "https://")):
                key, item_value = value.split(":", 1)
                item: dict[str, Any] = {}
                item[key.strip()] = parse_scalar(item_value.strip()) if item_value.strip() else {}
                result.append(item)
            else:
                result.append(parse_scalar(value))
        return result, index

    parsed, final_index = parse_node(0, lines[0][0] if lines else 0)
    if final_index != len(lines):
        _, _, raw_line = lines[final_index]
        raise ValueError(f"Unsupported YAML structure near: {raw_line}")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML document must be a mapping.")
    return parsed
