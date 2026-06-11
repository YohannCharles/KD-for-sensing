from __future__ import annotations

from pathlib import Path
from typing import Any

from kd_sensing.config.normalization import image_encoder_type, iter_model_configs
from kd_sensing.modalities import REMOVED_IMAGE_ENCODERS, resolve_image_profile

REMOVED_IMAGE_OPTION_PREFIX = "image_" + "motion_"
REMOVED_KD_CONFIG_TOKENS = ("logits_kd", "rkd", "teacher_no_kd", "student_no_kd", "no_kd")
REMOVED_KD_OVERRIDE_KEYS = {
    "kd_mode",
    "kd.type",
    "kd.temperature",
    "kd.alpha",
    "teacher_model_name",
}
RETIRED_HIST_MODEL_NAMES = {
    "hist_beam_fusion",
}
RETIRED_RAYMOBTIME_DATASET_TYPES = {"raymobtime_s008"}
RETIRED_RAYMOBTIME_MODEL_TYPES = {
    "simple_concat_multitask_selection",
    "task_aware_gated_multitask_selection",
}
RETIRED_RAYMOBTIME_ENCODER_TYPES = {
    "coord_mlp",
    "ray_mlp",
    "raymobtime_lidar_3d_cnn",
}
RETIRED_RAYMOBTIME_PREPROCESSORS = {
    "raymobtime_s008_audit",
    "raymobtime_s008_index",
    "raymobtime_s008_ray_features",
    "raymobtime_s008_cache",
}


def reject_removed_config_path(config_path: str | Path | None) -> None:
    if config_path is None:
        return
    path = Path(config_path)
    if path.suffix not in {".yaml", ".yml"}:
        return
    if _is_retired_hist_config_path(path):
        raise ValueError(
            f"HiST-Beam/Hist research line has been retired; legacy config path "
            f"'{path.as_posix()}' is no longer supported. Use current supervised, adapter, "
            "GPS candidate, residual fusion, MMW GPS v2, CSI, or viewer workflows."
        )
    if _is_retired_raymobtime_config_path(path):
        raise ValueError(
            f"Raymobtime s008 has been retired; legacy config path '{path.as_posix()}' "
            "is no longer supported and no compatibility migration is provided."
        )
    suggestion = _replacement_config_path(path)
    if suggestion is None:
        return
    raise ValueError(
        f"KD support has been removed; legacy config path '{path.as_posix()}' is no longer supported. "
        f"Use '{suggestion}' instead."
    )


def reject_removed_override_key(key: str) -> None:
    normalized = str(key).strip()
    lowered = normalized.lower()
    if lowered == "distillation" or lowered.startswith("distillation."):
        raise ValueError(
            f"KD support has been removed; override '{normalized}' is no longer supported. "
            "Use supervised, strong, lightweight, or adaptation configuration entries."
        )
    if lowered == "hist_beam" or lowered.startswith("hist_beam."):
        raise ValueError(
            f"HiST-Beam/Hist research line has been retired; override '{normalized}' is no longer supported. "
            "Use current supervised, adapter, GPS candidate, residual fusion, MMW GPS v2, CSI, or viewer workflows."
        )
    if lowered in REMOVED_KD_OVERRIDE_KEYS:
        raise ValueError(
            f"KD support has been removed; override '{normalized}' is no longer supported. "
            "Use supervised/adaptation configuration entries."
        )


def reject_removed_kd_config(cfg: dict[str, Any]) -> None:
    if "distillation" in cfg:
        raise ValueError(
            "KD support has been removed; config key 'distillation' is no longer supported. "
            "Use supervised/adaptation loss settings under 'loss'."
        )
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        removed = [key for key in ("teacher", "student") if key in model_cfg]
        if removed:
            keys = ", ".join(f"model.{key}" for key in removed)
            raise ValueError(
                f"KD support has been removed; {keys} are no longer supported. "
                "Use a single 'model.primary' config."
            )
    _reject_removed_kd_values(cfg)


def reject_retired_hist_config(cfg: dict[str, Any]) -> None:
    if "hist_beam" in cfg:
        raise ValueError(
            "HiST-Beam/Hist research line has been retired; config key 'hist_beam' is no longer supported. "
            "Use current supervised, adapter, GPS candidate, residual fusion, MMW GPS v2, CSI, or viewer workflows."
        )
    for location, model_cfg in iter_model_configs(cfg):
        model_type = str(model_cfg.get("type", "")).strip()
        if model_type in RETIRED_HIST_MODEL_NAMES:
            raise ValueError(
                f"HiST-Beam/Hist research line has been retired; {location}.type='{model_type}' is no longer supported. "
                "Use current supervised, adapter, GPS candidate, residual fusion, MMW GPS v2, CSI, or viewer workflows."
            )
    _reject_retired_hist_values(cfg)


def reject_retired_raymobtime_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if isinstance(dataset_cfg, dict):
        dataset_type = str(dataset_cfg.get("type", "")).strip()
        if dataset_type in RETIRED_RAYMOBTIME_DATASET_TYPES:
            raise ValueError(
                "Raymobtime s008 has been retired; data.dataset.type='raymobtime_s008' "
                "is no longer supported and no compatibility migration is provided."
            )

    preprocessing_cfg = cfg.get("preprocessing", {})
    if isinstance(preprocessing_cfg, dict):
        preprocessor_type = str(preprocessing_cfg.get("type", "")).strip()
        if preprocessor_type in RETIRED_RAYMOBTIME_PREPROCESSORS:
            raise ValueError(
                f"Raymobtime s008 has been retired; preprocessing.type='{preprocessor_type}' "
                "is no longer supported and no compatibility migration is provided."
            )

    for location, model_cfg in iter_model_configs(cfg):
        model_type = str(model_cfg.get("type", "")).strip()
        if model_type in RETIRED_RAYMOBTIME_MODEL_TYPES:
            raise ValueError(
                f"Raymobtime s008 has been retired; {location}.type='{model_type}' "
                "is no longer supported and no compatibility migration is provided."
            )
        modalities = model_cfg.get("modalities", ())
        if isinstance(modalities, str):
            modalities = (modalities,)
        if isinstance(modalities, (list, tuple, set)):
            retired_modalities = sorted({str(item) for item in modalities if str(item) in {"coord", "ray"}})
            if retired_modalities:
                names = ", ".join(retired_modalities)
                raise ValueError(
                    f"Raymobtime s008 has been retired; {location}.modalities contains retired "
                    f"modality/modalities: {names}."
                )
        encoders = model_cfg.get("encoders", {})
        if isinstance(encoders, dict):
            for modality, encoder_cfg in encoders.items():
                encoder_type = encoder_cfg if isinstance(encoder_cfg, str) else None
                if isinstance(encoder_cfg, dict):
                    encoder_type = encoder_cfg.get("type")
                encoder_name = str(encoder_type or "").strip()
                if encoder_name in RETIRED_RAYMOBTIME_ENCODER_TYPES:
                    raise ValueError(
                        f"Raymobtime s008 has been retired; {location}.encoders.{modality}.type="
                        f"'{encoder_name}' is no longer supported."
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
    removed_cache_keys: list[str] = []
    if isinstance(cache_cfg, dict):
        removed_cache_keys.extend(
            str(key)
            for key in cache_cfg
            if str(key).startswith(REMOVED_IMAGE_OPTION_PREFIX)
            or str(key) in {"image_motion", "motion_image", "motion_mask"}
        )
        image_cache_cfg = cache_cfg.get("image")
        if isinstance(image_cache_cfg, dict):
            removed_cache_keys.extend(
                f"image.{key}"
                for key in image_cache_cfg
                if str(key).startswith("motion") or str(key).startswith(REMOVED_IMAGE_OPTION_PREFIX)
            )
    if removed_cache_keys:
        keys = ", ".join(sorted(removed_cache_keys))
        raise ValueError(
            f"Removed image motion cache option(s): {keys}. "
            "Use data.cache.image.policy for RGB/ImageNet image-derived cache or disable image cache."
        )
    image_profile = dataset_cfg.get("image_profile")
    if image_profile is not None:
        resolve_image_profile(image_profile)
    for location, model_cfg in iter_model_configs(cfg):
        encoder_type = image_encoder_type(model_cfg)
        if encoder_type in REMOVED_IMAGE_ENCODERS:
            raise ValueError(
                f"Removed image encoder '{encoder_type}' in {location}. "
                "Use 'resnet18_imagenet_rgb' with RGB/ImageNet image input."
            )


def _replacement_config_path(path: Path) -> str | None:
    parts = path.parts
    try:
        configs_index = parts.index("configs")
        rel_parts = parts[configs_index:]
    except ValueError:
        rel_parts = parts[-3:] if len(parts) >= 3 else parts
    if len(rel_parts) < 3 or rel_parts[0] != "configs":
        return None
    section = rel_parts[1]
    stem = path.stem
    if not any(token in stem for token in REMOVED_KD_CONFIG_TOKENS):
        return None
    if stem.endswith("_logits_kd") or stem.endswith("_rkd"):
        if section == "fusion":
            suffix = "_logits_kd" if stem.endswith("_logits_kd") else "_rkd"
            slug = stem[: -len(suffix)]
            return f"configs/fusion/{slug}_lightweight.yaml"
        return f"configs/{section}/lightweight.yaml"
    if stem in {"logits_kd", "rkd"}:
        return f"configs/{section}/lightweight.yaml"
    if section == "fusion" and stem.endswith("_no_kd"):
        supervised = path.with_name(stem[: -len("_no_kd")] + "_supervised" + path.suffix)
        if supervised.exists():
            return f"configs/fusion/{supervised.name}"
        return f"configs/fusion/{stem[: -len('_no_kd')]}_lightweight.yaml"
    new_stem = stem
    replacements = (
        ("teacher_no_kd", "strong"),
        ("student_no_kd", "lightweight"),
        ("snapshot_next_frame_no_kd", "snapshot_next_frame_supervised"),
        ("no_kd", "supervised"),
    )
    for old, new in replacements:
        new_stem = new_stem.replace(old, new)
    return f"configs/{section}/{new_stem}{path.suffix}"


def _is_retired_hist_config_path(path: Path) -> bool:
    parts = path.parts
    try:
        configs_index = parts.index("configs")
        rel_parts = parts[configs_index:]
    except ValueError:
        rel_parts = parts[-3:] if len(parts) >= 3 else parts
    return len(rel_parts) >= 2 and rel_parts[0] == "configs" and rel_parts[1] == "hist_beam"


def _is_retired_raymobtime_config_path(path: Path) -> bool:
    parts = path.parts
    try:
        configs_index = parts.index("configs")
        rel_parts = parts[configs_index:]
    except ValueError:
        rel_parts = parts[-3:] if len(parts) >= 3 else parts
    if len(rel_parts) >= 2 and rel_parts[0] == "configs" and rel_parts[1] == "raymobtime":
        return True
    if len(rel_parts) >= 3 and rel_parts[0] == "configs" and rel_parts[1] == "preprocess":
        return path.stem.startswith("raymobtime_s008_")
    return False


def _reject_removed_kd_values(value: Any, *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            lowered_key = str(key).lower()
            if lowered_key in {"teacher_model_name", "teacher_checkpoint"}:
                raise ValueError(
                    f"KD support has been removed; config key '{child_path}' is no longer supported."
                )
            _reject_removed_kd_values(child, path=child_path)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_removed_kd_values(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in ("logits_kd", "rkd", "legacy_kd")):
            raise ValueError(
                f"KD support has been removed; config value at '{path}' references removed KD entry '{value}'."
            )


def _reject_retired_hist_values(value: Any, *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _reject_retired_hist_values(child, path=child_path)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_retired_hist_values(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.lower()
        if "configs/hist_beam" in lowered or lowered in RETIRED_HIST_MODEL_NAMES:
            raise ValueError(
                f"HiST-Beam/Hist research line has been retired; config value at '{path}' references retired entry '{value}'."
            )
