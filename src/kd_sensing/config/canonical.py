"""Generated canonical experiment configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from kd_sensing.config.canonical_recipes import (
    advanced_overlay_recipe,
    available_advanced_overlay_names,
    deep_merge as _deep_merge,
    distillation_overrides,
    objective_overlay_recipe,
    training_overrides,
)
from kd_sensing.modalities import (
    MODALITY_ORDER,
    dataset_defaults_for_modalities,
    dataset_flags_for_modalities,
    model_defaults_for_modalities,
)
from kd_sensing.utils.paths import project_root

CANONICAL_FUSION_MODALITIES = MODALITY_ORDER
CANONICAL_FUSION_MODES = ("teacher_no_kd", "student_no_kd", "logits_kd", "rkd")
CANONICAL_FUSION_OBJECTIVES = ("beam", "occlusion", "position", "multitask")
CANONICAL_OBJECTIVE_FUSION_MODE = "no_kd"
CANONICAL_OBJECTIVE_SUBSET_ALIASES = {
    "all_modalities": list(CANONICAL_FUSION_MODALITIES),
    "strong_only": ["gps", "mmwave"],
    "weak_only": ["image", "radar", "lidar"],
}
REMOVED_FUSION_CONFIG_STEMS = {
    "no_kd": "image_radar_student_no_kd.yaml",
    "logits_kd": "image_radar_logits_kd.yaml",
    "rkd": "image_radar_rkd.yaml",
}

_FUSION_MODE_SUFFIXES = tuple((f"_{mode}", mode) for mode in CANONICAL_FUSION_MODES)
_MODALITY_INDEX = {name: index for index, name in enumerate(CANONICAL_FUSION_MODALITIES)}
_CANONICAL_ORDER_TEXT = " > ".join(CANONICAL_FUSION_MODALITIES)


def build_virtual_config(config_path: Path) -> dict[str, Any] | None:
    """Build a virtual config override for missing canonical config paths."""

    if _is_fusion_config_path(config_path):
        if config_path.stem in REMOVED_FUSION_CONFIG_STEMS:
            replacement = REMOVED_FUSION_CONFIG_STEMS[config_path.stem]
            raise ValueError(
                f"Removed fusion config alias '{config_path.name}'. "
                f"Use 'configs/fusion/{replacement}' instead."
            )
        return build_virtual_fusion_config(config_path.stem)
    return None


def build_virtual_fusion_config(stem: str) -> dict[str, Any]:
    advanced = build_advanced_fusion_overlay_config(stem)
    if advanced is not None:
        return advanced

    objective_config = parse_objective_fusion_config_stem(stem)
    if objective_config is not None:
        slug, modalities, objective = objective_config
        return build_objective_fusion_config(slug, modalities, objective)

    slug, modalities, mode = parse_fusion_config_stem(stem)
    name = f"{slug}_{mode}"
    image_radar = modalities == ["image", "radar"]
    teacher_cfg = _fusion_teacher_baseline_model(modalities)
    student_cfg = deepcopy(teacher_cfg) if mode == "teacher_no_kd" else _cls_token_transformer_fusion_model(modalities)

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
            "modalities": list(modalities),
            "teacher": teacher_cfg,
            "student": student_cfg,
        },
        "distillation": distillation_overrides(slug, mode, image_radar),
        "training": training_overrides(mode, image_radar),
        "output": {
            "run_name": name,
        },
    }
    if mode in {"logits_kd", "rkd"}:
        cfg["paths"] = {
            "weights_dir": f"outputs/scene31/{slug}_teacher_no_kd/checkpoints"
        }
    return cfg


def build_objective_fusion_config(slug: str, modalities: list[str], objective: str) -> dict[str, Any]:
    recipe = objective_overlay_recipe(objective)
    name = f"{slug}_{objective}_{CANONICAL_OBJECTIVE_FUSION_MODE}"
    teacher_cfg = _fusion_teacher_baseline_model(modalities)
    student_cfg = _cls_token_transformer_fusion_model(modalities)
    for head_name, enabled in recipe.auxiliary_heads.items():
        student_cfg.setdefault("auxiliary_heads", {})[head_name] = bool(enabled)
    if "auxiliary_heads" in student_cfg:
        student_cfg["auxiliary_heads"]["enabled"] = True

    dataset = _dataset_overrides(modalities)
    dataset.update(deepcopy(recipe.dataset))

    cfg: dict[str, Any] = {
        "experiment": {
            "name": name,
            "task": "fusion",
            "objective": objective,
            "seed": 42 if modalities == ["image", "radar"] else 0,
        },
        "data": {"dataset": dataset},
        "model": {
            "teacher": teacher_cfg,
            "student": student_cfg,
        },
        "distillation": {"type": "no_kd", "teacher_model_name": None},
        "loss": deepcopy(recipe.loss),
        "training": {
            "early_stopping_metric": recipe.early_stopping_metric,
            "early_stopping_mode": recipe.early_stopping_mode,
            "lr": 0.00075,
            "weight_decay": 0.0001,
        },
        "output": {"run_name": name},
    }
    return cfg


def build_advanced_fusion_overlay_config(stem: str) -> dict[str, Any] | None:
    if not stem.startswith("overlay_"):
        return None
    recipe = stem[len("overlay_") :]
    overlay_recipe = advanced_overlay_recipe(recipe)
    if overlay_recipe is None:
        raise ValueError(
            f"Unknown advanced fusion overlay '{stem}.yaml'. "
            f"Available overlays: {available_advanced_overlay_names()}."
        )
    if overlay_recipe.builder == "g2d":
        return _g2d_overlay(str(overlay_recipe.options["mode"]), stem)
    if overlay_recipe.builder == "craf":
        return _craf_overlay(stem, deepcopy(overlay_recipe.options.get("ablation") or {}))
    if overlay_recipe.builder == "marf":
        return _marf_overlay(stem, deepcopy(overlay_recipe.options.get("ablation") or {}))
    if overlay_recipe.builder == "multitask_occlusion_position":
        return _multitask_occlusion_position_overlay(stem)
    raise ValueError(f"Advanced overlay recipe '{recipe}' references unknown builder '{overlay_recipe.builder}'.")


def _advanced_fusion_base(name: str) -> dict[str, Any]:
    modalities = list(CANONICAL_FUSION_MODALITIES)
    teacher_cfg: dict[str, Any] = {
        "type": "fusion_teacher",
        "modalities": modalities,
        "image_channels": 3,
        "radar_channels": 2,
        "gps_input_size": 3,
        "lidar_channels": 3,
        "mmwave_input_size": 64,
        "feature_size": 64,
        "num_classes": 64,
        "gru_params": [64, 64, 2],
    }
    dataset = {
        "type": "deepsense6g",
        "scene": 31,
        "train_csv_name": "train_seqs_RA_GPS_LIDAR.csv",
        "test_csv_name": "test_seqs_RA_GPS_LIDAR.csv",
        "seq_len": 8,
        "num_pred": 3,
        "portion": 1.0,
    }
    dataset.update(dataset_flags_for_modalities(modalities))
    dataset.update(dataset_defaults_for_modalities(modalities))
    return {
        "experiment": {"name": name, "task": "fusion", "seed": 0, "device": "auto"},
        "data": {
            "dataset": dataset,
            "dataloader": {
                "train_batch_size": 32,
                "test_batch_size": 32,
                "num_workers": 4,
                "prefetch_factor": 1,
            },
        },
        "model": {
            "modalities": modalities,
            "feature_size": 64,
            "num_classes": 64,
            "seq_length_teacher": 8,
            "seq_length_student": 8,
            "num_pred": 3,
            "downsample_ratio": 1,
            "teacher": teacher_cfg,
            "student": {
                "type": "fusion_student",
                "modalities": modalities,
                "image_channels": 3,
                "radar_channels": 2,
                "gps_input_size": 3,
                "lidar_channels": 3,
                "mmwave_input_size": 64,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64, 2],
            },
        },
        "distillation": {"type": "no_kd", "teacher_model_name": None},
        "training": {
            "epochs": 100,
            "lr": 0.00075,
            "weight_decay": 0.0001,
            "grad_clip": 10.0,
            "patience": 20,
            "use_early_stopping": True,
            "early_stopping_metric": "val_adba",
            "early_stopping_mode": "max",
            "min_delta": 0.0001,
        },
        "output": {"dir": "outputs", "run_name": name},
        "scheduler": {"type": "cosine_warm_restarts", "T_0": 10, "T_mult": 2, "eta_min": 1.0e-06},
    }


def _g2d_overlay(mode: str, name: str) -> dict[str, Any]:
    modalities = list(CANONICAL_FUSION_MODALITIES)
    cfg = _advanced_fusion_base(name)
    modular_model = _modular_resnet_fusion_model(modalities, num_layers=2)
    cfg["model"]["teacher"] = modular_model
    cfg["model"]["student"] = _modular_resnet_fusion_model(modalities, num_layers=2)
    cfg["loss"] = {"type": "cross_entropy"}
    cfg["distillation"] = {
        "type": "g2d",
        "teacher_model_name": None,
        "g2d": {
            "mode": mode,
            "modalities": modalities,
            "teachers": {
                modality: {
                    "model": _g2d_teacher_model_cfg(modality),
                    "checkpoint": None,
                    "strict_load": True,
                }
                for modality in modalities
            },
            "loss": {
                "supervised_weight": 1.0,
                "feature_weight": 0.1,
                "logit_weight": 0.5,
                "temperature": 4.0,
                "horizons": "all",
                "feature_align": {
                    "enabled": True,
                    "mode": "mse",
                    "pool": "last",
                    "normalize": True,
                    "projection": "auto",
                    "projection_dim": 64,
                },
                "logit_align": {"enabled": True},
            },
            "smp": {
                "enabled": mode == "global",
                "mode": "confidence" if mode == "global" else "none",
                "tau": {"per_modality": 5, "joint": 30},
                "prioritize_low_confidence_first": True,
            },
            "diagnostics": {"enabled": True},
        },
    }
    return cfg


def _g2d_teacher_model_cfg(modality: str) -> dict[str, Any]:
    if modality == "image":
        return _modular_resnet_fusion_model(["image"], num_layers=1)
    if modality == "lidar":
        return _modular_lidar_model(num_layers=1)
    return {"type": f"{modality}_teacher"}


def _craf_overlay(name: str, ablation: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _advanced_fusion_base(name)
    cfg["model"]["student"] = {
        **cfg["model"]["student"],
        "type": "craf_fusion",
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "reliability": {"gate_type": "sigmoid", "min_gate": 0.05, "use_dataset_prior": False},
    }
    cfg["loss"] = {
        "type": "focal_loss",
        "alpha": 1,
        "gamma": 2,
        "beam_soft": {"enabled": True, "weight": 0.03, "sigma": 2.0, "circular": True},
        "unimodal_aux": {"weight": 0.1},
    }
    cfg["training"] = _deep_merge(
        cfg["training"],
        {
            "warmup_epochs": 5,
            "modality_dropout": {"enabled": True, "drop_prob": 0.1, "min_keep": 1},
            "counterfactual": {
                "enabled": True,
                "mode": "sample_one",
                "start_epoch": 5,
                "weight": 0.1,
                "target_temperature": 1.0,
                "ignore_delta_eps": 0.0,
                "use_ce_only": True,
                "no_grad_drop_forward": True,
            },
        },
    )
    return _deep_merge(cfg, ablation or {})


def _marf_overlay(name: str, ablation: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _advanced_fusion_base(name)
    cfg["model"]["student"] = {
        **cfg["model"]["student"],
        "type": "marf_fusion",
        "d_model": 64,
        "num_heads": 4,
        "dropout": 0.1,
        "router": {
            "hidden_size": 64,
            "temperature": 1.0,
            "use_prior_bias": True,
            "prior_anchor_scale": 0.5,
            "prior_residual_scale": 0.25,
            "use_confidence_features": True,
            "zero_init": True,
        },
        "residual_adapter": {"enabled": True, "residual_scale": 0.2},
    }
    cfg["teacher"] = {
        "registry_path": "outputs/scene31/teacher_registry.json",
        "load_encoders": True,
        "freeze_encoders": True,
        "strict": True,
    }
    cfg["loss"] = {
        "type": "cross_entropy",
        "label_smoothing": 0.03,
        "beam_soft": {"enabled": True, "weight": 0.01, "sigma": 2.0, "circular": True},
        "unimodal_aux": {"weight": 0.0},
        "prior_regularization": {"enabled": False, "weight": 0.0},
        "marf": {
            "residual_norm": {"enabled": True, "weight": 0.01},
            "prior_regularization": {"enabled": True, "weight": 0.01, "loss_type": "mse"},
            "anchor_entropy": {"enabled": False, "weight": 0.0, "maximize": True},
            "subset_ce": {"weight": 0.3},
            "subset_kd": {"weight": 0.2, "temperature": 3.0},
        },
    }
    cfg["training"] = _deep_merge(
        cfg["training"],
        {
            "warmup_epochs": 0,
            "modality_dropout": {"enabled": False, "drop_prob": 0.0, "min_keep": 1},
            "counterfactual": {"enabled": False, "weight": 0.0},
            "subset_training": {
                "enabled": False,
                "modes": [],
                "top_prior_k": 2,
                "min_keep": 1,
                "ce_weight": 0.3,
                "kd_weight": 0.2,
                "temperature": 3.0,
            },
        },
    )
    cfg["evaluation"] = {
        "modality_subsets": {
            "enabled": True,
            "subsets": ["all", "top_prior", "single_best_prior", "strong_only", "weak_only"],
            "top_prior_k": 2,
        }
    }
    return _deep_merge(cfg, ablation or {})


def _multitask_occlusion_position_overlay(name: str) -> dict[str, Any]:
    modalities = list(CANONICAL_FUSION_MODALITIES)
    cfg = _advanced_fusion_base(name)
    cfg["experiment"]["name"] = name
    cfg["data"]["dataset"].update(
        {
            "train_csv_name": "train_seqs_RA_GPS_LIDAR_POS.csv",
            "test_csv_name": "test_seqs_RA_GPS_LIDAR_POS.csv",
            "occlusion_target": {
                "enabled": True,
                "threshold_percentile": 20.0,
            },
            "position_target": {
                "enabled": True,
                "source": "future_gps_local_xy",
                "normalize": True,
            },
        }
    )
    cfg["model"]["teacher"] = _fusion_teacher_baseline_model(modalities)
    cfg["model"]["student"] = _cls_token_transformer_fusion_model(modalities)
    cfg["model"]["student"]["auxiliary_heads"] = {
        "enabled": True,
        "occlusion": True,
        "position": True,
    }
    cfg["loss"] = {
        "type": "focal_loss",
        "alpha": 1,
        "gamma": 2,
        "beam_soft": {"enabled": False, "weight": 0.0},
        "unimodal_aux": {"weight": 0.0},
        "auxiliary": {
            "enabled": True,
            "occlusion": {"enabled": True, "weight": 1.0, "pos_weight": "auto"},
            "position": {"enabled": True, "weight": 0.01},
        },
    }
    cfg["output"]["run_name"] = name
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


def parse_objective_fusion_config_stem(stem: str) -> tuple[str, list[str], str] | None:
    suffix = f"_{CANONICAL_OBJECTIVE_FUSION_MODE}"
    if not stem.endswith(suffix):
        return None
    prefix = stem[: -len(suffix)]
    if "_" not in prefix:
        return None
    slug, objective = prefix.rsplit("_", 1)
    if objective not in CANONICAL_FUSION_OBJECTIVES:
        return None
    if not slug:
        raise ValueError("Objective fusion config slug cannot be empty.")
    modalities = CANONICAL_OBJECTIVE_SUBSET_ALIASES.get(slug)
    if modalities is None:
        modalities = slug.split("_")
        invalid = [name for name in modalities if name not in _MODALITY_INDEX]
        if invalid:
            raise ValueError(
                f"Unknown fusion modalities {invalid} in objective fusion config '{stem}.yaml'. "
                f"Available modalities: {list(CANONICAL_FUSION_MODALITIES)}."
            )
        duplicates = sorted({name for name in modalities if modalities.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Objective fusion config slug '{slug}' cannot contain duplicate modalities: {duplicates}."
            )
        if len(modalities) < 2:
            modality = modalities[0]
            raise ValueError(
                f"Objective fusion configs require at least two modalities; use "
                f"configs/{modality}/... for single-modality {modality} experiments."
            )
        canonical_modalities = sorted(modalities, key=_MODALITY_INDEX.__getitem__)
        if canonical_modalities != modalities:
            canonical_slug = "_".join(canonical_modalities)
            raise ValueError(
                f"Objective fusion config slug '{slug}' must follow modality order "
                f"{_CANONICAL_ORDER_TEXT}; use '{canonical_slug}_{objective}_{CANONICAL_OBJECTIVE_FUSION_MODE}.yaml'."
            )
    else:
        modalities = list(modalities)
    return slug, list(modalities), objective


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
    dataset = dataset_flags_for_modalities(modalities)
    dataset.update(dataset_defaults_for_modalities(modalities))
    return {key: value for key, value in dataset.items() if value not in (False, None)}


def _add_modality_model_fields(model_cfg: dict[str, Any], modalities: list[str]) -> None:
    model_cfg.update(model_defaults_for_modalities(modalities))


def _fusion_teacher_baseline_model(modalities: list[str]) -> dict[str, Any]:
    if "image" in modalities or "lidar" in modalities:
        return _modular_resnet_fusion_model(modalities, num_layers=2)
    cfg: dict[str, Any] = {
        "type": "fusion_teacher",
        "modalities": modalities,
        "feature_size": 64,
        "num_classes": 64,
        "gru_params": [64, 64, 2],
    }
    _add_modality_model_fields(cfg, modalities)
    return cfg


def _cls_token_transformer_fusion_model(modalities: list[str]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": "cls_token_transformer_fusion",
        "modalities": modalities,
        "feature_size": 64,
        "d_model": 64,
        "num_classes": 64,
        "num_pred": 3,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "max_seq_len": 16,
    }
    _add_modality_model_fields(cfg, modalities)
    return cfg


def _modular_resnet_fusion_model(modalities: list[str], *, num_layers: int) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": "modular_sequence",
        "modalities": list(modalities),
        "image_profile": "rgb_imagenet",
        "feature_size": 64,
        "d_model": 64,
        "num_classes": 64,
        "num_pred": 3,
        "encoders": {},
        "representation_core": {
            "type": "early_concat_gru" if len(modalities) > 1 else "single_gru",
            "d_model": 64,
            "hidden_size": 64,
            "num_layers": int(num_layers),
        },
        "heads": {
            "beam": {
                "type": "beam_head",
                "dropout": 0.1,
            },
        },
    }
    if "image" in modalities:
        cfg["encoders"]["image"] = {
            "type": "resnet18_imagenet_rgb",
            "output_dim": 64,
            "pretrained": True,
            "weights": "DEFAULT",
            "freeze_backbone": True,
            "unfreeze_stages": ["layer4"],
            "dropout": 0.1,
        }
    if "lidar" in modalities:
        cfg["encoders"]["lidar"] = {
            "type": "lidar_cnn",
            "output_dim": 64,
            "lidar_channels": 3,
        }
    _add_modality_model_fields(cfg, modalities)
    return cfg


def _modular_lidar_model(*, num_layers: int) -> dict[str, Any]:
    return {
        "type": "modular_sequence",
        "modalities": ["lidar"],
        "feature_size": 64,
        "d_model": 64,
        "num_classes": 64,
        "num_pred": 3,
        "lidar_channels": 3,
        "encoders": {
            "lidar": {
                "type": "lidar_cnn",
                "output_dim": 64,
                "lidar_channels": 3,
            }
        },
        "representation_core": {
            "type": "single_gru",
            "d_model": 64,
            "hidden_size": 64,
            "num_layers": int(num_layers),
        },
        "heads": {
            "beam": {
                "type": "beam_head",
                "dropout": 0.1,
            },
        },
    }
