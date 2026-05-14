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

from kd_sensing.config.canonical import build_virtual_config
from kd_sensing.config.defaults import DEFAULT_CONFIG
from kd_sensing.data.scenes import normalize_deepsense_config
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
    reject_removed_image_path_config(cfg)
    apply_fusion_modality_selection(cfg, override_cfg=override_cfg)
    normalize_model_role_defaults(cfg)
    normalize_deepsense_config(cfg)
    normalize_image_profile_config(cfg)
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


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
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

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line without ':': {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root
