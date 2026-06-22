"""Generated canonical experiment configuration helpers."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from kd_sensing.modalities import (
    MODALITY_ORDER,
    dataset_defaults_for_modalities,
    dataset_flags_for_modalities,
    model_defaults_for_modalities,
)
from kd_sensing.utils.paths import project_root

CANONICAL_DEEPSENSE_MODALITIES = tuple(MODALITY_ORDER)
CANONICAL_FUSION_MODALITIES = tuple(modality for modality in CANONICAL_DEEPSENSE_MODALITIES if modality != "csi")
CANONICAL_SINGLE_MODALITIES = CANONICAL_DEEPSENSE_MODALITIES
CANONICAL_FUSION_MODES = ("strong", "lightweight")
CANONICAL_FUSION_OBJECTIVES = ("beam", "occlusion", "position", "multitask")
CANONICAL_OBJECTIVE_FUSION_MODE = "supervised"
VISION_POSITION_BASELINE_PRESETS = (
    "camera_ae_gps",
    "resnet_gps",
    "transformer_image_gps",
    "gps_only_neural",
)
SNAPSHOT_VARIANT = "snapshot_next_frame"
SNAPSHOT_MODE = "snapshot_next_frame_supervised"
SNAPSHOT_TRAIN_CSV = "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
SNAPSHOT_VAL_CSV = "val_seqs_SNAPSHOT_NEXT_FRAME.csv"
CANONICAL_OBJECTIVE_SUBSET_ALIASES = {
    "all_modalities": list(CANONICAL_FUSION_MODALITIES),
    "strong_only": ["gps", "mmwave"],
    "weak_only": ["image", "radar", "lidar"],
}
REMOVED_FUSION_CONFIG_STEMS = {
    "no_kd": "image_radar_lightweight.yaml",
}
RETIRED_FUSION_KD_MODES = ("logits_kd", "rkd", "teacher_no_kd", "student_no_kd")

_FUSION_MODE_SUFFIXES = tuple((f"_{mode}", mode) for mode in CANONICAL_FUSION_MODES)
_RETIRED_FUSION_KD_SUFFIXES = tuple((f"_{mode}", mode) for mode in RETIRED_FUSION_KD_MODES)
_MODALITY_INDEX = {name: index for index, name in enumerate(CANONICAL_FUSION_MODALITIES)}
_CANONICAL_ORDER_TEXT = " > ".join(CANONICAL_FUSION_MODALITIES)
_IMAGE_RADAR_TRAINING = {
    "strong": {"lr": 0.00075, "weight_decay": 0.0001},
    "lightweight": {"lr": 0.0004, "weight_decay": 0.0},
}
_ADVANCED_OVERLAY_BUILDERS = {
    "multitask_occlusion_position": "multitask_occlusion_position",
}
_ADVANCED_OVERLAY_ALIASES: dict[str, str] = {}
_BASE_OBJECTIVE_LOSS = {
    "type": "focal_loss",
    "alpha": 1,
    "gamma": 2,
    "soft_targets": {"enabled": False, "ignore_index": -100},
    "beam_soft": {"enabled": False, "weight": 0.0},
    "unimodal_aux": {"weight": 0.0},
    "objective": {
        "weights": {"beam": 1.0, "occlusion": 1.0, "position": 0.01},
        "occlusion": {"pos_weight": "auto"},
        "position": {"type": "mse"},
    },
}


def training_overrides(mode: str, image_radar: bool) -> dict[str, float | str]:
    try:
        training = _IMAGE_RADAR_TRAINING[mode] if image_radar else {"lr": 0.00075, "weight_decay": 0.0001}
    except KeyError as exc:
        supported = ", ".join(sorted(_IMAGE_RADAR_TRAINING))
        raise ValueError(f"Unknown canonical fusion mode '{mode}'. Available modes: {supported}.") from exc
    return {"early_stopping_metric": "val_adba", "early_stopping_mode": "max", **training}


def _objective_loss(*, position_weight: float) -> dict[str, Any]:
    cfg = deepcopy(_BASE_OBJECTIVE_LOSS)
    cfg["objective"]["weights"]["position"] = position_weight
    return cfg


_OBJECTIVE_OVERLAY_RECIPES: dict[str, dict[str, Any]] = {
    "beam": {
        "dataset": {},
        "auxiliary_heads": {},
        "loss": _objective_loss(position_weight=0.01),
        "early_stopping_metric": "val_adba",
        "early_stopping_mode": "max",
    },
    "occlusion": {
        "dataset": {"occlusion_target": {"enabled": True, "threshold_percentile": 20.0}},
        "auxiliary_heads": {"occlusion": True},
        "loss": _objective_loss(position_weight=0.01),
        "early_stopping_metric": "val_occlusion_blocked_f1",
        "early_stopping_mode": "max",
    },
    "position": {
        "dataset": {
            "train_csv_name": "train_seqs_RA_GPS_LIDAR_POS.csv",
            "test_csv_name": "test_seqs_RA_GPS_LIDAR_POS.csv",
            "position_target": {"enabled": True, "source": "future_gps_local_xy", "normalize": True},
        },
        "auxiliary_heads": {"position": True},
        "loss": _objective_loss(position_weight=1.0),
        "early_stopping_metric": "val_position_rmse",
        "early_stopping_mode": "min",
    },
    "multitask": {
        "dataset": {
            "train_csv_name": "train_seqs_RA_GPS_LIDAR_POS.csv",
            "test_csv_name": "test_seqs_RA_GPS_LIDAR_POS.csv",
            "occlusion_target": {"enabled": True, "threshold_percentile": 20.0},
            "position_target": {"enabled": True, "source": "future_gps_local_xy", "normalize": True},
        },
        "auxiliary_heads": {"occlusion": True, "position": True},
        "loss": _objective_loss(position_weight=1.0),
        "early_stopping_metric": "val_multitask_loss",
        "early_stopping_mode": "min",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    from kd_sensing.config.io import deep_merge

    return deep_merge(base, override)


def build_virtual_config(config_path: Path) -> dict[str, Any] | None:
    """Build a virtual config override for missing canonical config paths."""

    if _is_vision_position_baseline_config_path(config_path):
        return build_vision_position_baseline_config(config_path.stem)
    if _is_fusion_config_path(config_path):
        if config_path.stem in REMOVED_FUSION_CONFIG_STEMS:
            replacement = REMOVED_FUSION_CONFIG_STEMS[config_path.stem]
            raise ValueError(
                f"Removed fusion config alias '{config_path.name}'. "
                f"Use 'configs/fusion/{replacement}' instead."
            )
        return build_virtual_fusion_config(config_path.stem)
    single = parse_single_snapshot_config_path(config_path)
    if single is not None:
        return build_snapshot_single_config(single)
    return None


def build_vision_position_baseline_config(preset: str) -> dict[str, Any]:
    if preset not in VISION_POSITION_BASELINE_PRESETS:
        available = ", ".join(VISION_POSITION_BASELINE_PRESETS)
        raise ValueError(f"Unknown vision-position baseline preset '{preset}'. Available presets: {available}.")

    seq_len = 1
    num_pred = 1
    num_classes = 64
    gps_feature_mode = "paper_distance_angle"
    gps_input_size = 2
    image_enabled = preset != "gps_only_neural"
    modalities = ["gps"] if preset == "gps_only_neural" else ["image", "gps"]
    dataset = {
        "type": "deepsense6g",
        "scene": 31,
        "train_csv_name": "train_seqs_RA_GPS_LIDAR.csv",
        "test_csv_name": "test_seqs_RA_GPS_LIDAR.csv",
        "seq_len": seq_len,
        "num_pred": num_pred,
        "portion": 1.0,
        "gps_feature_mode": gps_feature_mode,
        "gps_angle_offset_source": "paper_scene_default",
        "gps_normalize": True,
        "gps_normalization_artifact": "train_split:gps_scaler",
        "beam_target_source": "current",
        "mock_data": False,
        "return_metadata": True,
    }
    dataset.update(dataset_flags_for_modalities(modalities))
    for key, value in dataset_defaults_for_modalities(modalities).items():
        dataset.setdefault(key, value)
    if image_enabled:
        dataset.update(
            {
                "image_profile": "rgb_imagenet",
                "image_size": [224, 224],
                "image_normalization": "imagenet",
                "image_augment": False,
            }
        )

    cfg: dict[str, Any] = {
        "experiment": {
            "name": preset,
            "task": "fusion",
            "baseline_preset": preset,
            "seed": 42,
            "device": "auto",
        },
        "data": {
            "dataset": dataset,
            "dataloader": {
                "train_batch_size": 16,
                "test_batch_size": 16,
                "num_workers": 4,
                "prefetch_factor": 1,
            },
        },
        "model": {
            "baseline_preset": preset,
            "modalities": modalities,
            "feature_size": 64,
            "d_model": 64,
            "num_classes": num_classes,
            "num_pred": num_pred,
            "downsample_ratio": 1,
            "seq_length": seq_len,
            "primary": _vision_position_primary_model(
                preset,
                seq_len=seq_len,
                num_pred=num_pred,
                gps_feature_mode=gps_feature_mode,
                gps_input_size=gps_input_size,
            ),
        },
        "evaluation": {
            "k_values": [1, 3, 5],
            "dba_delta": 5,
            "dba_distance_mode": "linear",
            "metric_profile": "beambench_linear_topk",
            "label_space": "64_beam",
            "beam_shift": 0,
            "circular_beam_distance": False,
        },
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
        "output": {
            "dir": "outputs",
            "run_name": preset,
        },
    }
    if preset == "camera_ae_gps":
        cfg["beambench_paper"] = {
            "official_pretrained_weights": False,
            "official_test_set": False,
            "official_search_procedure": False,
            "require_checkpoint": True,
            "table_iii_equivalent": False,
            "protocol_aligned": True,
            "protocol_alignment": "beambench_input_split_metric_only",
            "recommended_table_iii_config": "configs/fusion/beambench_image_ae_gps_direct.yaml",
            "recommended_table_iii_cli": "kd-sensing-run-beambench-image-ae-gps-tableiii",
            "non_equivalent_reason": (
                "This virtual preset uses the shared supervised trainer and vision_position_late_fusion model; "
                "BeamBench Table III Camera=AE GPS=Direct uses the dedicated BeamBenchImageAEGPSDirectModel."
            ),
        }
    if preset == "gps_only_neural":
        cfg["experiment"]["uses_neural_network"] = True
        cfg["beambench_paper"] = {
            "official_pretrained_weights": False,
            "official_test_set": False,
            "official_search_procedure": False,
            "table_iii_equivalent": False,
            "protocol_aligned": False,
            "paper_rows_not_equivalent": ["Classical*", "Dense\u2020"],
            "non_equivalent_reason": (
                "Arnold22 BeamBench GPS Classical* is a calibrated least-square rule and GPS Dense\u2020 "
                "uses the official dense_model/gps_dense.cfg pipeline; this preset is a project-local "
                "supervised neural GPS baseline."
            ),
            "recommended_table_iii_source": "official BeamBench classical.py and challenge.py --type_list gps_dense",
        }
    return cfg


def _vision_position_primary_model(
    preset: str,
    *,
    seq_len: int,
    num_pred: int,
    gps_feature_mode: str,
    gps_input_size: int,
) -> dict[str, Any]:
    common = {
        "baseline_preset": preset,
        "num_classes": 64,
        "num_pred": num_pred,
        "history_length": seq_len,
        "seq_length": seq_len,
        "gps_input_size": gps_input_size,
        "gps_feature_mode": gps_feature_mode,
    }
    if preset == "camera_ae_gps":
        return {
            **common,
            "type": "vision_position_late_fusion",
            "modalities": ["image", "gps"],
            "feature_size": 64,
            "fusion_hidden_size": 128,
            "temporal_aggregation": "mean",
            "image_profile": "rgb_imagenet",
            "image_encoder_type": "camera_ae_frozen",
            "image_encoder": {
                "type": "camera_ae_frozen",
                "latent_dim": 512,
                "output_dim": 64,
                "image_size": 64,
                "freeze_encoder": True,
                "require_checkpoint": True,
                "checkpoint_path": "",
            },
            "gps_encoder": {
                "type": "gps_mlp",
                "output_dim": 64,
                "gps_input_size": gps_input_size,
                "hidden_size": 128,
                "dropout": 0.1,
            },
            "paper_style": {
                "target": "Arnold22_BeamBench_Table_III",
                "official_pretrained_weights": False,
                "official_test_set": False,
                "official_search_procedure": False,
            },
        }
    if preset == "resnet_gps":
        return {
            **common,
            "type": "vision_position_late_fusion",
            "modalities": ["image", "gps"],
            "feature_size": 64,
            "fusion_hidden_size": 64,
            "temporal_hidden_size": 64,
            "temporal_aggregation": "gru",
            "image_profile": "rgb_imagenet",
            "image_encoder_type": "resnet18_imagenet_rgb",
            "image_encoder": {
                "type": "resnet18_imagenet_rgb",
                "output_dim": 64,
                "pretrained": True,
                "weights": "DEFAULT",
                "freeze_backbone": True,
                "unfreeze_stages": ["layer4"],
                "dropout": 0.1,
            },
            "gps_encoder": {
                "type": "gps_mlp",
                "output_dim": 64,
                "hidden_size": 64,
                "dropout": 0.1,
            },
        }
    if preset == "transformer_image_gps":
        return {
            **common,
            "type": "vision_position_transformer_fusion",
            "modalities": ["image", "gps"],
            "feature_size": 64,
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "max_seq_len": seq_len,
            "token_organization": "cls_time_major_image_gps_tokens",
            "image_channels": 3,
        }
    if preset == "gps_only_neural":
        return {
            **common,
            "type": "gps_sequence_baseline",
            "modalities": ["gps"],
            "feature_size": 64,
            "hidden_size": 64,
            "temporal_model": "lstm",
            "num_layers": 1,
            "uses_neural_network": True,
        }
    raise ValueError(f"Unknown vision-position baseline preset '{preset}'.")


def build_virtual_fusion_config(stem: str) -> dict[str, Any]:
    retired_kd_mode = retired_fusion_kd_mode(stem)
    if retired_kd_mode is not None:
        raise ValueError(
            f"KD support has been removed for legacy fusion config '{stem}.yaml' "
            f"({retired_kd_mode}). Use '<slug>_strong.yaml' or '<slug>_lightweight.yaml'."
        )

    advanced = build_advanced_fusion_overlay_config(stem)
    if advanced is not None:
        return advanced

    snapshot_config = parse_snapshot_fusion_config_stem(stem)
    if snapshot_config is not None:
        name_slug, modalities = snapshot_config
        return build_snapshot_fusion_config(name_slug, modalities)

    objective_config = parse_objective_fusion_config_stem(stem)
    if objective_config is not None:
        slug, modalities, objective = objective_config
        return build_objective_fusion_config(slug, modalities, objective)

    slug, modalities, mode = parse_fusion_config_stem(stem)
    name = f"{slug}_{mode}"
    image_radar = modalities == ["image", "radar"]
    primary_cfg = _fusion_strong_baseline_model(modalities) if mode == "strong" else _cls_token_transformer_fusion_model(modalities)

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
            "primary": primary_cfg,
        },
        "training": training_overrides(mode, image_radar),
        "output": {
            "run_name": name,
        },
    }
    return cfg


def retired_fusion_kd_mode(stem: str) -> str | None:
    if stem in RETIRED_FUSION_KD_MODES:
        return stem
    for suffix, mode in _RETIRED_FUSION_KD_SUFFIXES:
        if stem.endswith(suffix):
            return mode
    return None


def build_snapshot_single_config(modality: str) -> dict[str, Any]:
    name = f"{modality}_{SNAPSHOT_MODE}"
    model_cfg = _snapshot_modular_model([modality])
    dataset = _snapshot_dataset_overrides([modality])
    return {
        "experiment": {
            "name": name,
            "task": modality,
            "variant": SNAPSHOT_VARIANT,
            "uses_history_window": False,
            "uses_temporal_core": False,
            "seed": 42,
        },
        "data": {"dataset": dataset},
        "model": {
            "feature_size": 64,
            "d_model": 64,
            "num_classes": 64,
            "seq_length": 1,
            "num_pred": 1,
            "downsample_ratio": 1,
            "primary": model_cfg,
        },
        "training": {
            "early_stopping_metric": "val_adba",
            "early_stopping_mode": "max",
            "lr": 0.00075,
            "weight_decay": 0.0001,
        },
        "output": {"run_name": name},
    }


def build_snapshot_fusion_config(name_slug: str, modalities: list[str]) -> dict[str, Any]:
    name = f"{name_slug}_{SNAPSHOT_MODE}"
    model_cfg = _snapshot_modular_model(modalities)
    dataset = _snapshot_dataset_overrides(modalities)
    return {
        "experiment": {
            "name": name,
            "task": "fusion",
            "variant": SNAPSHOT_VARIANT,
            "uses_history_window": False,
            "uses_temporal_core": False,
            "seed": 0,
        },
        "data": {"dataset": dataset},
        "model": {
            "modalities": list(modalities),
            "feature_size": 64,
            "d_model": 64,
            "num_classes": 64,
            "seq_length": 1,
            "num_pred": 1,
            "downsample_ratio": 1,
            "primary": model_cfg,
        },
        "training": {
            "early_stopping_metric": "val_adba",
            "early_stopping_mode": "max",
            "lr": 0.00075,
            "weight_decay": 0.0001,
        },
        "output": {"run_name": name},
    }


def build_objective_fusion_config(slug: str, modalities: list[str], objective: str) -> dict[str, Any]:
    try:
        recipe = _OBJECTIVE_OVERLAY_RECIPES[objective]
    except KeyError as exc:
        supported = ", ".join(sorted(_OBJECTIVE_OVERLAY_RECIPES))
        raise ValueError(f"Unknown objective overlay recipe '{objective}'. Available objectives: {supported}.") from exc
    name = f"{slug}_{objective}_{CANONICAL_OBJECTIVE_FUSION_MODE}"
    primary_cfg = _cls_token_transformer_fusion_model(modalities)
    for head_name, enabled in recipe["auxiliary_heads"].items():
        primary_cfg.setdefault("auxiliary_heads", {})[head_name] = bool(enabled)
    if "auxiliary_heads" in primary_cfg:
        primary_cfg["auxiliary_heads"]["enabled"] = True

    dataset = _dataset_overrides(modalities)
    dataset.update(deepcopy(recipe["dataset"]))

    cfg: dict[str, Any] = {
        "experiment": {
            "name": name,
            "task": "fusion",
            "objective": objective,
            "seed": 42 if modalities == ["image", "radar"] else 0,
        },
        "data": {"dataset": dataset},
        "model": {
            "primary": primary_cfg,
        },
        "loss": deepcopy(recipe["loss"]),
        "training": {
            "early_stopping_metric": recipe["early_stopping_metric"],
            "early_stopping_mode": recipe["early_stopping_mode"],
            "lr": 0.00075,
            "weight_decay": 0.0001,
        },
        "output": {"run_name": name},
    }
    return cfg


def build_advanced_fusion_overlay_config(stem: str) -> dict[str, Any] | None:
    recipe = _resolve_advanced_overlay_recipe_name(stem)
    if recipe is None:
        return None
    builder = _ADVANCED_OVERLAY_BUILDERS.get(recipe)
    if builder is None:
        raise ValueError(
            f"Unknown advanced fusion overlay '{stem}.yaml'. "
            f"Available overlays: {_available_advanced_overlay_names()}."
        )
    if builder == "multitask_occlusion_position":
        return _multitask_occlusion_position_overlay(stem)
    raise ValueError(f"Advanced overlay recipe '{recipe}' references unknown builder '{builder}'.")


def _resolve_advanced_overlay_recipe_name(stem: str) -> str | None:
    if stem.startswith("overlay_"):
        return stem[len("overlay_") :]
    return _ADVANCED_OVERLAY_ALIASES.get(stem)


def _available_advanced_overlay_names() -> list[str]:
    return sorted([f"overlay_{name}" for name in _ADVANCED_OVERLAY_BUILDERS] + list(_ADVANCED_OVERLAY_ALIASES))


def _advanced_fusion_base(name: str) -> dict[str, Any]:
    modalities = list(CANONICAL_FUSION_MODALITIES)
    primary_cfg = _cls_token_transformer_fusion_model(modalities)
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
            "seq_length": 8,
            "num_pred": 3,
            "downsample_ratio": 1,
            "primary": primary_cfg,
        },
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
    cfg["model"]["primary"] = _cls_token_transformer_fusion_model(modalities)
    cfg["model"]["primary"]["auxiliary_heads"] = {
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


def parse_snapshot_fusion_config_stem(stem: str) -> tuple[str, list[str]] | None:
    suffix = f"_{SNAPSHOT_MODE}"
    if not stem.endswith(suffix):
        return None
    slug = stem[: -len(suffix)]
    if slug == "all_modalities":
        return slug, list(CANONICAL_FUSION_MODALITIES)
    if not slug:
        raise ValueError("Snapshot fusion config slug cannot be empty.")
    modalities = slug.split("_")
    invalid = [name for name in modalities if name not in _MODALITY_INDEX]
    if invalid:
        raise ValueError(
            f"Unknown fusion modalities {invalid} in snapshot fusion config '{stem}.yaml'. "
            f"Available modalities: {list(CANONICAL_FUSION_MODALITIES)}."
        )
    duplicates = sorted({name for name in modalities if modalities.count(name) > 1})
    if duplicates:
        raise ValueError(f"Snapshot fusion config slug '{slug}' cannot contain duplicate modalities: {duplicates}.")
    if len(modalities) < 2:
        modality = modalities[0]
        raise ValueError(
            f"Snapshot fusion configs require at least two modalities; use "
            f"configs/{modality}/{SNAPSHOT_MODE}.yaml for single-modality {modality} experiments."
        )
    canonical_modalities = sorted(modalities, key=_MODALITY_INDEX.__getitem__)
    if canonical_modalities != modalities:
        canonical_slug = "_".join(canonical_modalities)
        raise ValueError(
            f"Snapshot fusion config slug '{slug}' must follow modality order "
            f"{_CANONICAL_ORDER_TEXT}; use '{canonical_slug}_{SNAPSHOT_MODE}.yaml'."
        )
    return slug, modalities


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


def parse_single_snapshot_config_path(path: Path) -> str | None:
    if path.suffix not in {".yaml", ".yml"} or path.stem != SNAPSHOT_MODE:
        return None
    parts = _config_path_parts(path)
    if len(parts) != 3 or parts[0] != "configs":
        return None
    modality = parts[1]
    if modality == "fusion":
        return None
    if modality not in CANONICAL_SINGLE_MODALITIES:
        return None
    return modality


def _is_fusion_config_path(path: Path) -> bool:
    if path.suffix not in {".yaml", ".yml"}:
        return False
    parts = _config_path_parts(path)
    return len(parts) == 3 and parts[:2] == ("configs", "fusion")


def _is_vision_position_baseline_config_path(path: Path) -> bool:
    if path.suffix not in {".yaml", ".yml"} or path.stem not in VISION_POSITION_BASELINE_PRESETS:
        return False
    parts = _config_path_parts(path)
    return len(parts) == 3 and parts[0] == "configs" and parts[1] in {"fusion", "gps"}


def _config_path_parts(path: Path) -> tuple[str, ...]:
    try:
        relative = path.resolve().relative_to(project_root())
        return tuple(relative.parts)
    except ValueError:
        parts = path.parts
        if len(parts) >= 3 and parts[-3] == "configs":
            return tuple(parts[-3:])
        return tuple(parts)


def _dataset_overrides(modalities: list[str]) -> dict[str, Any]:
    dataset = dataset_flags_for_modalities(modalities)
    dataset.update(dataset_defaults_for_modalities(modalities))
    return {key: value for key, value in dataset.items() if value not in (False, None)}


def _snapshot_dataset_overrides(modalities: list[str]) -> dict[str, Any]:
    dataset = _dataset_overrides(modalities)
    dataset.update(
        {
            "type": "deepsense6g",
            "scene": 31,
            "train_csv_name": SNAPSHOT_TRAIN_CSV,
            "val_csv_name": SNAPSHOT_VAL_CSV,
            "test_csv_name": SNAPSHOT_VAL_CSV,
            "seq_len": 1,
            "num_pred": 1,
            "portion": 1.0,
            "split_metadata_path": "split_metadata_SNAPSHOT_NEXT_FRAME.json",
        }
    )
    return dataset


def _add_modality_model_fields(model_cfg: dict[str, Any], modalities: list[str]) -> None:
    model_cfg.update(model_defaults_for_modalities(modalities))


def _fusion_strong_baseline_model(modalities: list[str]) -> dict[str, Any]:
    return _modular_resnet_fusion_model(modalities, num_layers=2)


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


def _snapshot_modular_model(modalities: list[str]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": "modular_sequence",
        "modalities": list(modalities),
        "image_profile": "rgb_imagenet",
        "feature_size": 64,
        "d_model": 64,
        "num_classes": 64,
        "num_pred": 1,
        "encoders": {},
        "representation_core": {
            "type": "snapshot_frame",
            "d_model": 64,
            "hidden_size": 64,
            "dropout": 0.1,
        },
        "heads": {
            "beam": {
                "type": "beam_head",
                "dropout": 0.1,
            },
        },
        "uses_temporal_core": False,
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
