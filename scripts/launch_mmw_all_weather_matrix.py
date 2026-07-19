#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from kd_sensing.config import load_config
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.utils.artifact_registry import router_architecture_profile_sha256


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("S1", "T2", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
T2_ABLATION_METHODS = ("T2-NoBPA", "T2-BPA2CMA", "T2-Linear", "T2-CLS", "T2-CLS-CMA")
ALL_METHODS = (*METHODS, *T2_ABLATION_METHODS)
T2_BASE_CONFIG = "configs/mmw/t2.yaml"
UMASK_MAINLINE_METHODS = ("S1", "T2")
H4_MATCHED_BASELINES = ("masktrain_cls", "amr_net_4m")
METHOD_BASES = {
    "S1": "configs/mmw/s1.yaml",
    "T2": T2_BASE_CONFIG,
    "masktrain_cls": "configs/mmw/masktrain_cls.yaml",
    "amber_full": "configs/mmw/amber_full.yaml",
    "rmbp_mm": "configs/mmw/rmbp_mm.yaml",
    "amr_net_4m": "configs/mmw/amr_net_4m.yaml",
    **{method: T2_BASE_CONFIG for method in T2_ABLATION_METHODS},
}
UMASK_TRAINING_PROFILES = {
    "legacy_h0_v1": {
        "training": {
            "optimizer": {"type": "adam"},
            "lr": 5.0e-4,
            "weight_decay": 1.0e-4,
        },
        "scheduler": {"type": "none"},
    },
    "umask_h4_v1": {
        "training": {
            "optimizer": {"type": "adamw"},
            "lr": 5.0e-4,
            "weight_decay": 3.0e-4,
        },
        "scheduler": {
            "type": "cosine_warm_restarts",
            "T_0": 40,
            "T_mult": 1,
            "eta_min": 1.0e-6,
        },
    },
}
UMASK_ROUTER_ARCHITECTURE_PROFILES = {
    "umask_router_pattern_v1": {
        "model": {"primary": {"router_use_pattern_features": True}},
        "selection_scope": "legacy_control",
        "source_design_candidate": "base_t2_yaml",
    },
    "umask_router_nopattern_v1": {
        "model": {"primary": {"router_use_pattern_features": False}},
        "selection_scope": "development_mainline_pending_inner_mask_multiseed",
        "source_design_candidate": "RouterNoPattern",
    },
}
WEATHERS = ("sunny", "rainy", "foggy")
SCENES = (
    "Town03_5wayroad_seed28",
    "Town03_Tjunction_wiz_slope_seed42",
    "Town03_crossroad_wiz_slope_seed42",
    "Town03_gastation_seed40",
    "Town03_roundabout_seed42",
)
MODALITIES = ("image", "radar", "gps", "lidar")
SPLIT_TAG = "h5p1_strict_v2"
DEFAULT_OUTPUT_ROOT = "outputs/mmw_all_weather_h5p1_seed1_v2"
BASELINE_FIDELITY = {
    "masktrain_cls": {
        "reproduction_scope": "plain_mask_training_control",
        "paper_equivalent": False,
        "architecture_scope": "shared_encoders_availability_normalized_mean_classifier",
        "temporal_result_scope": "fair_mask_training_control",
    },
    "amber_full": {
        "reproduction_scope": "amber_full_local_adaptation",
        "paper_equivalent": False,
        "omitted_paper_inputs": ["historical_beam_index"],
        "architecture_scope": "reduced_four_sensor_transformer_with_approximate_auxiliary_losses",
        "temporal_result_scope": "local_adaptation_diagnostic",
    },
    "rmbp_mm": {
        "reproduction_scope": "rmbp_mm_channel_attention_local",
        "paper_equivalent": False,
        "omitted_paper_inputs": ["partial_beam_measurement"],
        "omitted_paper_training_stages": ["unimodal_pretraining", "label_guided_similarity_imputation"],
        "architecture_scope": "channel_attention_fusion_only",
        "temporal_result_scope": "out_of_paper_scope_diagnostic",
    },
    "amr_net_4m": {
        "reproduction_scope": "four_modality_window5_local_adaptation",
        "paper_equivalent": False,
        "architecture_scope": "gaussian_embedding_uncertainty_aware_feature_fusion",
        "adaptations": ["add_radar", "window5", "shared_four_modality_encoders"],
        "temporal_result_scope": "local_adaptation_diagnostic",
    },
}

T2_ABLATION_PROTOCOL = {
    "T2-NoBPA": {
        "intervention": "disable_bpa_auxiliary_only",
        "matched_control": "T2",
        "head_type": "prototype",
        "bpa_auxiliary": False,
        "amber_cma_analogue": False,
        "router_prototype_margin": True,
        "prototype_target_geometry": "not_applicable",
        "router_oracle_geometry": "circular",
        "evaluation_geometry": "circular",
    },
    "T2-BPA2CMA": {
        "intervention": "replace_bpa_auxiliary_with_amber_style_cma_analogue",
        "matched_control": "T2-NoBPA",
        "head_type": "prototype",
        "bpa_auxiliary": False,
        "amber_cma_analogue": True,
        "router_prototype_margin": True,
        "prototype_target_geometry": "not_applicable",
        "router_oracle_geometry": "circular",
        "evaluation_geometry": "circular",
    },
    "T2-Linear": {
        "intervention": "remove_prototype_target_wrap_prior_only",
        "matched_control": "T2",
        "head_type": "prototype",
        "bpa_auxiliary": True,
        "amber_cma_analogue": False,
        "router_prototype_margin": True,
        "prototype_target_geometry": "linear",
        "router_oracle_geometry": "circular",
        "evaluation_geometry": "circular",
    },
    "T2-CLS": {
        "intervention": "remove_prototype_package",
        "matched_control": "T2",
        "head_type": "classifier",
        "bpa_auxiliary": False,
        "amber_cma_analogue": False,
        "router_prototype_margin": False,
        "prototype_target_geometry": "not_applicable",
        "router_oracle_geometry": "circular",
        "evaluation_geometry": "circular",
    },
    "T2-CLS-CMA": {
        "intervention": "add_amber_style_cma_to_classifier_control",
        "matched_control": "T2-CLS",
        "head_type": "classifier",
        "bpa_auxiliary": False,
        "amber_cma_analogue": True,
        "router_prototype_margin": False,
        "prototype_target_geometry": "not_applicable",
        "router_oracle_geometry": "circular",
        "evaluation_geometry": "circular",
    },
}


def _apply_t2_ablation(payload: dict[str, Any], method: str) -> None:
    if method not in T2_ABLATION_METHODS:
        return

    primary = payload.setdefault("model", {}).setdefault("primary", {})
    u_mask = payload.setdefault("loss", {}).setdefault("u_mask_beam_jepa", {})

    cma_enabled = method in {"T2-BPA2CMA", "T2-CLS-CMA"}
    bpa_enabled = method == "T2-Linear"
    classifier = method in {"T2-CLS", "T2-CLS-CMA"}

    primary["head_type"] = "classifier" if classifier else "prototype"
    primary["router_use_prototype_margin"] = not classifier
    payload.setdefault("experiment", {})["ablation_id"] = method
    u_mask.update(
        {
            "use_beam_prototype_alignment": bpa_enabled,
            "lambda_proto": 0.2 if bpa_enabled else 0.0,
            "lambda_modality_proto": 0.1 if bpa_enabled else 0.0,
            "use_amber_cma_analogue": cma_enabled,
            "lambda_amber_cma": 0.2 if cma_enabled else 0.0,
            "amber_cma_temperature": 0.2,
        }
    )

    if method == "T2-Linear":
        # This counterfactual removes only the 0/63 wrap prior from BPA targets.
        u_mask.update({"prototype_target_circular": False, "circular_beam_distance": True})
        payload.setdefault("evaluation", {}).update(
            {
                "beam_distance_circular": True,
                "circular_beam_distance": True,
                "dba_distance_mode": "circular",
                "metric_profile": "64_beam_circular_topk",
            }
        )


def default_umask_training_profile(method: str) -> str | None:
    if method in {*UMASK_MAINLINE_METHODS, *H4_MATCHED_BASELINES}:
        return "umask_h4_v1"
    if method in T2_ABLATION_METHODS:
        return "legacy_h0_v1"
    return None


def default_umask_router_architecture_profile(method: str) -> str | None:
    if method in UMASK_MAINLINE_METHODS:
        return "umask_router_nopattern_v1"
    if method in T2_ABLATION_METHODS:
        return "umask_router_pattern_v1"
    return None


def _profile_sha256(profile_id: str, canonical_values: dict[str, Any]) -> str:
    payload = json.dumps(
        {"id": profile_id, "canonical_values": canonical_values},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _apply_umask_training_profile(
    payload: dict[str, Any],
    *,
    method: str,
    profile_id: str | None,
) -> dict[str, Any] | None:
    accepts_profile = method in {*UMASK_MAINLINE_METHODS, *T2_ABLATION_METHODS, *H4_MATCHED_BASELINES}
    if not accepts_profile:
        if profile_id is not None:
            raise ValueError(f"{method} does not accept a U-Mask training profile.")
        return None
    if profile_id is None:
        raise ValueError(f"{method} requires an explicit U-Mask training profile.")
    if method in T2_ABLATION_METHODS and profile_id != "legacy_h0_v1":
        raise ValueError(f"{method} must use the legacy_h0_v1 U-Mask training profile.")
    try:
        canonical_values = deepcopy(UMASK_TRAINING_PROFILES[profile_id])
    except KeyError as exc:
        supported = ", ".join(sorted(UMASK_TRAINING_PROFILES))
        raise ValueError(f"Unknown U-Mask training profile {profile_id!r}; supported: {supported}.") from exc

    training_values = canonical_values["training"]
    payload.setdefault("training", {}).update(deepcopy(training_values))
    payload["scheduler"] = deepcopy(canonical_values["scheduler"])
    return {
        "id": profile_id,
        "canonical_values": canonical_values,
        "sha256": _profile_sha256(profile_id, canonical_values),
    }


def _apply_umask_router_architecture_profile(
    payload: dict[str, Any],
    *,
    method: str,
    profile_id: str | None,
) -> dict[str, Any] | None:
    is_umask_method = method in {*UMASK_MAINLINE_METHODS, *T2_ABLATION_METHODS}
    if not is_umask_method:
        if profile_id is not None:
            raise ValueError(f"{method} does not accept a U-Mask router architecture profile.")
        return None
    if profile_id is None:
        raise ValueError(f"{method} requires an explicit U-Mask router architecture profile.")
    if method in T2_ABLATION_METHODS and profile_id != "umask_router_pattern_v1":
        raise ValueError(f"{method} must use the umask_router_pattern_v1 router architecture profile.")
    try:
        canonical_values = deepcopy(UMASK_ROUTER_ARCHITECTURE_PROFILES[profile_id])
    except KeyError as exc:
        supported = ", ".join(sorted(UMASK_ROUTER_ARCHITECTURE_PROFILES))
        raise ValueError(
            f"Unknown U-Mask router architecture profile {profile_id!r}; supported: {supported}."
        ) from exc
    primary_values = canonical_values["model"]["primary"]
    payload.setdefault("model", {}).setdefault("primary", {}).update(deepcopy(primary_values))
    return {
        "id": profile_id,
        "canonical_values": canonical_values,
        "sha256": router_architecture_profile_sha256(profile_id, canonical_values),
    }


def domains() -> list[dict[str, str]]:
    result = []
    for weather in WEATHERS:
        for scene in SCENES:
            prefix = f"Prepared/{scene}/splits/{SPLIT_TAG}"
            result.append(
                {
                    "id": f"{weather}/{scene}",
                    "condition": weather,
                    "scene": scene,
                    "data_root": f"dataset/MMW/{weather}",
                    "train_csv_name": f"{prefix}/train_with_radar_with_bs_gps.csv",
                    "val_csv_name": f"{prefix}/test_with_radar_with_bs_gps.csv",
                    "test_csv_name": f"{prefix}/test_with_radar_with_bs_gps.csv",
                }
            )
    return result


def preflight(
    domain_inventory: list[dict[str, str]],
    *,
    enabled_modalities: tuple[str, ...] = MODALITIES,
) -> dict[str, Any]:
    selected_modalities = tuple(enabled_modalities)
    unknown_modalities = sorted(set(selected_modalities) - set(MODALITIES))
    if not selected_modalities or unknown_modalities:
        raise ValueError(f"Invalid MMW preflight modalities: {list(selected_modalities)}.")
    reports = []
    failures = []
    prefix_by_modality = {"image": ("camera",), "radar": ("radar",), "gps": ("gps", "bs_gps"), "lidar": ("lidar",)}
    required = {"future_beam_label1"}
    for modality in selected_modalities:
        required.update(
            f"{prefix}{index}"
            for prefix in prefix_by_modality[modality]
            for index in range(1, 6)
        )
    for domain in domain_inventory:
        root = ROOT / domain["data_root"]
        split_root = root / "Prepared" / domain["scene"] / "splits" / SPLIT_TAG
        metadata_path = split_root / "split_metadata.json"
        metadata = _read_json(metadata_path)
        domain_report = {
            "id": domain["id"],
            "condition": domain["condition"],
            "scene": domain["scene"],
            "split_metadata_path": str(metadata_path.relative_to(ROOT)),
            "strict_validation_eligible": bool(metadata.get("strict_validation_eligible", False)),
            "splits": {},
        }
        if not domain_report["strict_validation_eligible"]:
            failures.append(f"{domain['id']}: strict split metadata failed")
        for role, key in (("train", "train_csv_name"), ("validation", "val_csv_name")):
            csv_path = root / domain[key]
            split_report = _inspect_csv(csv_path, root, required)
            domain_report["splits"][role] = split_report
            failures.extend(f"{domain['id']}/{role}: {item}" for item in split_report["failures"])
        reports.append(domain_report)
    return {
        "status": "ready" if not failures and len(reports) == 15 else "blocked",
        "domain_count": len(reports),
        "enabled_modalities": list(selected_modalities),
        "split_tag": SPLIT_TAG,
        "gps_feature_mode": "relative_polar",
        "gps_angle_frame": "world",
        "domains": reports,
        "failures": failures,
    }


def _inspect_csv(
    path: Path,
    root: Path,
    required: set[str],
) -> dict[str, Any]:
    failures = []
    if not path.exists():
        return {"path": str(path), "sample_count": 0, "failures": ["CSV missing"]}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_columns = sorted(required - fields)
        if missing_columns:
            failures.append(f"missing columns: {','.join(missing_columns)}")
        sample_count = 0
        invalid_entries = []
        path_columns = sorted(column for column in required if not column.startswith("future_beam_label"))
        label_columns = sorted(column for column in required if column.startswith("future_beam_label"))
        for row_index, row in enumerate(reader):
            sample_count += 1
            for column in path_columns:
                if column not in fields:
                    continue
                value = str(row.get(column, "")).strip()
                issue = _resource_issue(root, column, value)
                if issue is not None and len(invalid_entries) < 20:
                    invalid_entries.append(f"row {row_index} {issue}")
                if column.startswith("radar") and issue is None:
                    if "_RA" not in value:
                        if len(invalid_entries) < 20:
                            invalid_entries.append(f"row {row_index} {column} must reference an _RA map")
                    else:
                        da_issue = _resource_issue(root, f"{column} (_DA)", value.replace("_RA", "_DA"))
                        if da_issue is not None and len(invalid_entries) < 20:
                            invalid_entries.append(f"row {row_index} {da_issue}")
            for column in label_columns:
                if column not in fields:
                    continue
                issue = _label_issue(column, row.get(column))
                if issue is not None and len(invalid_entries) < 20:
                    invalid_entries.append(f"row {row_index} {issue}")
        if sample_count == 0:
            failures.append("empty CSV")
        if invalid_entries:
            failures.append(f"invalid inputs: {invalid_entries[:4]}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sample_count": sample_count,
        "columns": sorted(fields),
        "failures": failures,
    }


def _resource_issue(root: Path, column: str, value: str) -> str | None:
    if not value or value.lower() in {"-99", "nan", "none"}:
        return f"is missing {column}"
    try:
        path = joined_resource(root, value)
    except ValueError as exc:
        return f"has invalid {column} path {value!r}: {exc}"
    if not path.is_file():
        return f"is missing {column} artifact {value!r}"
    return None


def _label_issue(column: str, value: object) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"has invalid {column}={value!r}"
    if not math.isfinite(numeric) or not numeric.is_integer() or not 0 <= numeric < 64:
        return f"has invalid {column}={value!r}; expected integer in [0, 63]"
    return None


def build_config(
    method: str,
    output_root: Path,
    *,
    seed: int = 1,
    smoke: bool,
    epochs: int,
    batch_size: int,
    umask_training_profile: str | None = None,
    umask_router_architecture_profile: str | None = None,
) -> dict[str, Any]:
    selected_modalities = MODALITIES
    base_path = ROOT / METHOD_BASES[method]
    payload = load_config(base_path)
    dataset = payload.setdefault("data", {}).setdefault("dataset", {})
    for key in (
        "scene", "scene_id", "scene_slug", "scenes", "train_scenes", "validation_scenes", "test_scenes",
        "split_protocol", "split_strategy", "split_source_splits", "split_fractions",
    ):
        dataset.pop(key, None)
    dataset.update(
        {
            "type": "mmw",
            "data_root": "dataset/MMW/sunny",
            "domains": domains(),
            "seq_len": 5,
            "num_pred": 1,
            "portion": 0.002 if smoke else 1.0,
            "use_gps": "gps" in selected_modalities,
            "use_lidar": "lidar" in selected_modalities,
            "gps_feature_mode": "relative_polar",
            "gps_normalize": True,
        }
    )
    payload["data"]["domain_balanced_sampling"] = {
        "enabled": True,
        "seed": int(seed),
        "replacement": True,
    }
    payload["data"]["dataloader"] = {
        "train_batch_size": batch_size,
        "test_batch_size": batch_size,
        "validation_batch_size": batch_size,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "train_drop_last": False,
        "test_drop_last": False,
    }
    payload.setdefault("experiment", {}).update({"name": method, "seed": int(seed), "device": "auto"})
    payload.setdefault("paths", {})["data_root"] = "dataset/MMW"
    model = payload.setdefault("model", {})
    model.update({"seq_length": 5, "num_pred": 1})
    if payload["experiment"].get("task") == "fusion":
        model["modalities"] = list(selected_modalities)
    else:
        model.pop("modalities", None)
    primary = model.setdefault("primary", {})
    primary.update({"modalities": list(selected_modalities), "seq_length": 5, "num_pred": 1})
    if method in BASELINE_FIDELITY:
        primary.setdefault("paper_metadata", {}).update(deepcopy(BASELINE_FIDELITY[method]))
    primary.pop("encoder_checkpoint_paths", None)
    for encoder in primary.get("encoders", {}).values():
        if isinstance(encoder, dict):
            encoder["pretrained"] = False
            encoder["weights"] = None
            encoder["freeze_backbone"] = False
    training_profile = _apply_umask_training_profile(
        payload,
        method=method,
        profile_id=umask_training_profile,
    )
    if training_profile and training_profile["id"] == "umask_h4_v1" and not smoke and int(epochs) != 40:
        raise ValueError("umask_h4_v1 is fixed to a 40-epoch budget outside smoke runs.")
    training = payload.setdefault("training", {})
    training.update(
        {
            "epochs": epochs,
            "max_epochs": epochs,
            "resume": False,
            "amp": {"enabled": True, "dtype": "float16", "grad_scaler": True},
            "validation": {"interval_epochs": 1 if smoke else epochs},
            "allow_tf32": True,
            "cudnn_benchmark": True,
        }
    )
    temporal = payload.get("temporal_missing")
    if not isinstance(temporal, dict):
        raise ValueError(f"Base config for {method} must define temporal_missing.")
    payload["temporal_missing"] = deepcopy(temporal)
    payload["temporal_missing"]["seed"] = int(seed)
    _apply_t2_ablation(payload, method)
    if umask_router_architecture_profile is None:
        umask_router_architecture_profile = default_umask_router_architecture_profile(method)
    router_architecture_profile = _apply_umask_router_architecture_profile(
        payload,
        method=method,
        profile_id=umask_router_architecture_profile,
    )
    patterns = [
        "full", "missing_image", "missing_radar", "missing_gps", "missing_lidar",
        "image_only", "radar_only", "gps_only", "lidar_only",
        "missing_image_radar", "missing_image_gps", "missing_image_lidar",
        "missing_radar_gps", "missing_radar_lidar", "missing_gps_lidar",
    ]
    payload.setdefault("evaluation", {})["missing_patterns"] = {
        "enabled": True,
        "patterns": patterns,
        "prediction_index": "last",
    }
    payload["output"] = {
        "dir": str(output_root / method),
        "run_name": f"seed{seed}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    protocol = payload.setdefault("mmw_all_weather_protocol", {})
    protocol.update(
        {
            "split_tag": SPLIT_TAG,
            "screening_role": "local_validation",
            "checkpoint_policy": "fixed_epoch_last_pth",
            "domain_macro_primary": True,
            "weather_label_used_as_input": False,
            "enabled_modalities": list(selected_modalities),
            "gps_feature_mode": "relative_polar",
            "gps_angle_frame": "world",
            "baseline_fidelity": deepcopy(BASELINE_FIDELITY.get(method, {"reproduction_scope": "project_mainline"})),
            "seed": int(seed),
        }
    )
    if training_profile is not None:
        protocol["training_profile"] = training_profile
    else:
        protocol.pop("training_profile", None)
    if router_architecture_profile is not None:
        protocol["router_architecture_profile"] = router_architecture_profile
    else:
        protocol.pop("router_architecture_profile", None)
    if method in T2_ABLATION_PROTOCOL:
        protocol["t2_ablation"] = {
            "protocol": "mmw_t2_bpa_cma_ablation_v1",
            "paper_equivalent": False,
            "cma_scope": "pooled_feature_objective_analogue_not_full_amber_class_former",
            "cma_temperature": 0.2,
            "cma_weight": 0.2 if method in {"T2-BPA2CMA", "T2-CLS-CMA"} else 0.0,
            **deepcopy(T2_ABLATION_PROTOCOL[method]),
        }
    return payload


def build_job_matrix(
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    gpus: tuple[int, ...] | None,
    output_root: Path,
) -> list[dict[str, Any]]:
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")
    if not seeds or any(seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique positive integers")
    pairs = [(method, seed) for method in methods for seed in seeds]
    if gpus is None:
        if seeds == (1,):
            gpu_by_method = {method: index for index, method in enumerate(ALL_METHODS)}
            selected_gpus = tuple(gpu_by_method[method] for method in methods)
        else:
            selected_gpus = tuple(range(len(pairs)))
    else:
        selected_gpus = gpus
    if len(selected_gpus) != len(pairs):
        raise ValueError(f"Expected {len(pairs)} GPUs for {len(pairs)} jobs, got {len(selected_gpus)}")
    if any(gpu < 0 for gpu in selected_gpus) or len(set(selected_gpus)) != len(selected_gpus):
        raise ValueError("GPUs must be unique non-negative integers")
    generated = output_root / "generated_configs"
    logs = output_root / "logs"
    return [
        {
            "method": method,
            "seed": seed,
            "gpu": gpu,
            "config_path": generated / f"{method}_seed{seed}.yaml",
            "log_path": logs / f"{method}_seed{seed}.log",
            "run_dir": output_root / method / f"seed{seed}",
            "status": "planned",
            "return_code": None,
        }
        for (method, seed), gpu in zip(pairs, selected_gpus)
    ]


def validate_job_targets(jobs: list[dict[str, Any]], manifest_path: Path) -> None:
    targets = [manifest_path]
    for job in jobs:
        targets.extend((job["config_path"], job["log_path"], job["run_dir"]))
    conflicts = [path for path in targets if path.exists()]
    if conflicts:
        joined = "\n".join(f"- {path}" for path in conflicts)
        raise FileExistsError(f"Refusing to overwrite existing MMW multiseed artifacts:\n{joined}")


def _manifest_name(seeds: tuple[int, ...]) -> str:
    if seeds == (1,):
        return "job_manifest.json"
    return "job_manifest_seeds_" + "_".join(str(seed) for seed in seeds) + ".json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch MMW 15-domain all-weather training matrix.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--gpus", default=None)
    parser.add_argument(
        "--umask-router-architecture-profile",
        choices=("auto", *sorted(UMASK_ROUTER_ARCHITECTURE_PROFILES)),
        default="auto",
        help="Router architecture profile for T2/S1; auto selects the current mainline or legacy control.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selected_methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(selected_methods) - set(ALL_METHODS))
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    try:
        selected_seeds = tuple(int(item) for item in _csv(args.seeds))
        selected_gpus = None if args.gpus is None else tuple(int(item) for item in _csv(args.gpus))
    except ValueError as exc:
        parser.error(str(exc))
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    report = preflight(domains())
    (output_root / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "ready":
        print(json.dumps(report, indent=2))
        return 2
    if args.preflight_only:
        print(json.dumps({"status": "ready", "domain_count": report["domain_count"]}, indent=2))
        return 0
    epochs = 1 if args.smoke else int(args.epochs)
    generated = output_root / "generated_configs"
    logs = output_root / "logs"
    try:
        jobs = build_job_matrix(selected_methods, selected_seeds, selected_gpus, output_root)
    except ValueError as exc:
        parser.error(str(exc))
    manifest_path = output_root / _manifest_name(selected_seeds)
    try:
        validate_job_targets(jobs, manifest_path)
    except FileExistsError as exc:
        parser.error(str(exc))
    generated.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    for job in jobs:
        method = job["method"]
        seed = int(job["seed"])
        router_profile = (
            default_umask_router_architecture_profile(method)
            if args.umask_router_architecture_profile == "auto"
            else args.umask_router_architecture_profile
        )
        if method not in {*UMASK_MAINLINE_METHODS, *T2_ABLATION_METHODS}:
            router_profile = None
        config_payload = build_config(
            method,
            Path(args.output_root),
            seed=seed,
            smoke=args.smoke,
            epochs=epochs,
            batch_size=args.batch_size,
            umask_training_profile=default_umask_training_profile(method),
            umask_router_architecture_profile=router_profile,
        )
        job["config_path"].write_text(
            yaml.safe_dump(
                config_payload,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        job["config_path"] = str(job["config_path"].relative_to(ROOT))
        job["log_path"] = str(job["log_path"].relative_to(ROOT))
        job["run_dir"] = str(job["run_dir"].relative_to(ROOT))
    manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    running = []
    for job in jobs:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "4",
            }
        )
        handle = (ROOT / job["log_path"]).open("w", encoding="utf-8")
        command = ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", job["config_path"]]
        job.update({"status": "running", "start_time": _now(), "command": command})
        running.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), job, handle))
    manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    failed = False
    for process, job, handle in running:
        code = process.wait()
        handle.close()
        job.update({"status": "done" if code == 0 else "failed", "return_code": code, "end_time": _now()})
        failed = failed or code != 0
        manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
