"""Generated canonical experiment configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kd_sensing.utils.paths import project_root

CANONICAL_FUSION_MODALITIES = ("image", "radar", "gps", "lidar", "mmwave")
CANONICAL_FUSION_MODES = ("teacher_no_kd", "student_no_kd", "logits_kd", "rkd")

_FUSION_MODE_SUFFIXES = tuple((f"_{mode}", mode) for mode in CANONICAL_FUSION_MODES)
_MODALITY_INDEX = {name: index for index, name in enumerate(CANONICAL_FUSION_MODALITIES)}
_CANONICAL_ORDER_TEXT = " > ".join(CANONICAL_FUSION_MODALITIES)


def build_virtual_config(config_path: Path) -> dict[str, Any] | None:
    """Build a virtual config override for missing canonical config paths."""

    if _is_fusion_config_path(config_path):
        return build_virtual_fusion_config(config_path.stem)
    return None


def build_virtual_fusion_config(stem: str) -> dict[str, Any]:
    slug, modalities, mode = parse_fusion_config_stem(stem)
    name = f"{slug}_{mode}"
    image_radar = modalities == ["image", "radar"]

    teacher_cfg: dict[str, Any] = {
        "type": "fusion_teacher",
        "modalities": modalities,
        "gru_params": [64, 64, 2],
    }
    student_gru = [64, 64, 1] if image_radar and mode != "teacher_no_kd" else [64, 64, 2]
    student_cfg: dict[str, Any] = {
        "type": "fusion_teacher" if mode == "teacher_no_kd" else "fusion_student",
        "modalities": modalities,
        "gru_params": [64, 64, 2] if mode == "teacher_no_kd" else student_gru,
    }
    _add_modality_model_fields(teacher_cfg, modalities)
    _add_modality_model_fields(student_cfg, modalities)

    cfg: dict[str, Any] = {
        "experiment": {
            "name": name,
            "task": "fusion",
            "seed": 42 if image_radar else 0,
        },
        "data": {
            "dataset": _dataset_overrides(modalities),
        },
        "model": {
            "teacher": teacher_cfg,
            "student": student_cfg,
        },
        "distillation": _distillation_overrides(slug, mode, image_radar),
        "training": _training_overrides(mode, image_radar),
        "output": {
            "run_name": name,
        },
    }
    if mode in {"logits_kd", "rkd"}:
        cfg["paths"] = {
            "weights_dir": "All_models" if image_radar else f"outputs/scene32/{slug}_teacher_no_kd/checkpoints"
        }
    return cfg


def parse_fusion_config_stem(stem: str) -> tuple[str, list[str], str]:
    for suffix, mode in _FUSION_MODE_SUFFIXES:
        if stem.endswith(suffix):
            slug = stem[: -len(suffix)]
            break
    else:
        modes = ", ".join(CANONICAL_FUSION_MODES)
        raise ValueError(f"Canonical fusion config '{stem}.yaml' must end with one of: {modes}.")

    if not slug:
        raise ValueError("Canonical fusion config slug cannot be empty.")

    modalities = slug.split("_")
    invalid = [name for name in modalities if name not in _MODALITY_INDEX]
    if invalid:
        raise ValueError(
            f"Unknown fusion modalities {invalid} in canonical fusion config '{stem}.yaml'. "
            f"Available modalities: {list(CANONICAL_FUSION_MODALITIES)}."
        )

    duplicates = sorted({name for name in modalities if modalities.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Canonical fusion config slug '{slug}' cannot contain duplicate modalities: {duplicates}."
        )

    if len(modalities) < 2:
        modality = modalities[0]
        raise ValueError(
            f"Canonical fusion configs require at least two modalities; use "
            f"configs/{modality}/{mode}.yaml for single-modality {modality} experiments."
        )

    canonical_modalities = sorted(modalities, key=_MODALITY_INDEX.__getitem__)
    if canonical_modalities != modalities:
        canonical_slug = "_".join(canonical_modalities)
        raise ValueError(
            f"Canonical fusion config slug '{slug}' must follow modality order "
            f"{_CANONICAL_ORDER_TEXT}; use '{canonical_slug}_{mode}.yaml'."
        )

    return slug, modalities, mode


def _is_fusion_config_path(path: Path) -> bool:
    if path.suffix not in {".yaml", ".yml"}:
        return False
    try:
        relative = path.resolve().relative_to(project_root())
    except ValueError:
        parts = path.parts
        return len(parts) >= 3 and parts[-3:-1] == ("configs", "fusion")
    return len(relative.parts) == 3 and relative.parts[:2] == ("configs", "fusion")


def _dataset_overrides(modalities: list[str]) -> dict[str, Any]:
    dataset: dict[str, Any] = {}
    if "gps" in modalities:
        dataset.update(
            {
                "use_gps": True,
                "gps_feature_mode": "relative_polar",
                "gps_normalize": True,
            }
        )
    if "lidar" in modalities:
        dataset.update(
            {
                "use_lidar": True,
                "lidar_bev_size": [224, 224],
                "lidar_roi": [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0],
                "lidar_normalize": False,
            }
        )
    if "mmwave" in modalities:
        dataset.update(
            {
                "use_mmwave": True,
                "mmwave_normalize": True,
            }
        )
    return dataset


def _add_modality_model_fields(model_cfg: dict[str, Any], modalities: list[str]) -> None:
    if "gps" in modalities:
        model_cfg["gps_input_size"] = 3
    if "lidar" in modalities:
        model_cfg["lidar_channels"] = 3
    if "mmwave" in modalities:
        model_cfg["mmwave_input_size"] = 64


def _distillation_overrides(slug: str, mode: str, image_radar: bool) -> dict[str, Any]:
    if image_radar:
        params = {
            "teacher_no_kd": {
                "type": "no_kd",
                "teacher_model_name": None,
                "temperature": 3.0,
                "alpha": 0.4,
                "alpha_warmup_epochs": 10,
                "rkd_pairs_per_anchor": 4,
                "rkd_distance_weight": 2.0,
                "rkd_angle_weight": 2.0,
            },
            "student_no_kd": {
                "type": "no_kd",
                "teacher_model_name": None,
                "temperature": 3.0,
                "alpha": 0.4,
                "alpha_warmup_epochs": 0,
                "rkd_pairs_per_anchor": 4,
                "rkd_distance_weight": 2.0,
                "rkd_angle_weight": 2.0,
            },
            "logits_kd": {
                "type": "logits_kd",
                "teacher_model_name": "BothTeacher_best.pth",
                "temperature": 2.0,
                "alpha": 0.4,
                "alpha_warmup_epochs": 0,
                "rkd_pairs_per_anchor": 4,
                "rkd_distance_weight": 5.0,
                "rkd_angle_weight": 5.0,
            },
            "rkd": {
                "type": "rkd",
                "teacher_model_name": "BothTeacher_best.pth",
                "temperature": 2.0,
                "alpha": 0.3,
                "alpha_warmup_epochs": 0,
                "rkd_pairs_per_anchor": 4,
                "rkd_distance_weight": 10.0,
                "rkd_angle_weight": 10.0,
            },
        }
        return params[mode]

    cfg: dict[str, Any] = {"type": "no_kd", "teacher_model_name": None}
    if mode in {"logits_kd", "rkd"}:
        cfg.update(
            {
                "type": mode,
                "temperature": 3.0,
                "alpha": 0.4,
                "alpha_warmup_epochs": 0,
                "teacher_model_name": "best.pth",
            }
        )
    if mode == "rkd":
        cfg.update(
            {
                "rkd_pairs_per_anchor": 4,
                "rkd_distance_weight": 10.0,
                "rkd_angle_weight": 10.0,
            }
        )
    return cfg


def _training_overrides(mode: str, image_radar: bool) -> dict[str, Any]:
    if not image_radar:
        return {"lr": 0.00075, "weight_decay": 0.0001}
    params = {
        "teacher_no_kd": {"lr": 0.00075, "weight_decay": 0.0001},
        "student_no_kd": {"lr": 0.0004, "weight_decay": 0.0},
        "logits_kd": {"lr": 0.00095, "weight_decay": 0.0},
        "rkd": {"lr": 0.00095, "weight_decay": 0.0},
    }
    return params[mode]
