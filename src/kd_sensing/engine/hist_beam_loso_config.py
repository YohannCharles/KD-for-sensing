from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.data.scenes import normalize_deepsense_dataset_config, retarget_deepsense_dataset_config
from kd_sensing.engine.hist_beam_history_anchor import apply_history_anchor_model_config
from kd_sensing.engine.hist_beam_image_only import (
    IMAGE_ONLY_ADAPTATION_VARIANTS,
    IMAGE_ONLY_SOURCE_VARIANTS,
    IMAGE_ONLY_VARIANTS,
    canonical_image_only_variant,
    image_only_protocol_enabled,
)
from kd_sensing.engine.hist_beam_v7 import apply_v7_stage_defaults, is_v7_variant
from kd_sensing.engine.modality_resolution import (
    SENSOR_ASSISTED_DISALLOWED_MODALITIES,
    SENSOR_ASSISTED_PROFILE,
    sensor_assisted_profile_enabled,
    resolve_enabled_modalities,
)
from kd_sensing.engine.run_lineage import ensure_distillation_defaults
from kd_sensing.modalities import normalize_modalities

EXECUTION_STATUSES = ("completed", "failed", "partial_failed")
SOURCE_ONLY_VARIANTS = {"v0_flat", "v1_hierarchical"} | IMAGE_ONLY_SOURCE_VARIANTS
ADAPTATION_VARIANTS = {"v4_adapter", "v5_adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto", "v8_target_prior_head", "v9_input_conditioned_target_adaptation", "v7_shared_physical_private_residual", "v6_full_finetune"} | IMAGE_ONLY_ADAPTATION_VARIANTS
SUPPORTED_VARIANTS = SOURCE_ONLY_VARIANTS | ADAPTATION_VARIANTS
DEFAULT_SOURCE_BASELINE_VARIANT = "v1_hierarchical"
RETIRED_HIST_BEAM_VARIANTS = {"v2_shared_private", "shared_private", "v3_decoupled", "decoupled"}
DEFAULT_QUICK_VARIANTS = ["v0_flat", "v1_hierarchical", "v4_adapter", "v5_adapter_proto", "v6_radio_proto", "v8_path_proto", "v6_full_finetune"]
SENSOR_ASSISTED_QUICK_VARIANTS = ["v1_hierarchical", "v4_adapter", "v6_radio_proto", "v8_path_proto", "adapter_path_proto", "v7_shared_physical_private_residual", "v8_target_prior_head", "v9_input_conditioned_target_adaptation", "v6_full_finetune"]
SENSOR_ASSISTED_QUICK_BUDGETS = [10]
SENSOR_ASSISTED_QUICK_SEEDS = [0, 1]
DEFAULT_QUICK_BUDGETS = [0, 10]
DEFAULT_QUICK_SEEDS = [0]
DEFAULT_QUICK_TARGET_SCENES = [34]
EXECUTION_PROGRESS_FILENAME = "execution_progress.jsonl"


def retired_hist_beam_variant_message(variant: Any) -> str:
    return (
        f"HiST-Beam variant '{variant}' is retired: the legacy simple shared/private "
        "knowledge-decoupling route is no longer supported. Use a current baseline such as "
        "'v0_flat', 'v1_hierarchical', 'v4_adapter', 'v5_adapter_proto', 'v6_radio_proto', "
        "'v8_path_proto', 'v7_shared_physical_private_residual', 'v8_target_prior_head', "
        "'v9_input_conditioned_target_adaptation', or 'v6_full_finetune'."
    )


def validate_loso_variant(variant: Any) -> str:
    normalized = str(variant).strip().lower()
    if normalized in RETIRED_HIST_BEAM_VARIANTS:
        raise ValueError(retired_hist_beam_variant_message(variant))
    if normalized not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported HiST-Beam LOSO variant '{variant}'. Supported variants: {sorted(SUPPORTED_VARIANTS)}.")
    return normalized

def _cpu_thread_config(cfg: dict[str, Any]) -> dict[str, Any]:
    thread_cfg = cfg.get("training", {}).get("cpu_threads", {}) if isinstance(cfg.get("training"), dict) else {}
    return dict(thread_cfg) if isinstance(thread_cfg, dict) else {}


def _excluded_sensitive_fields(cfg: dict[str, Any]) -> tuple[str, ...]:
    if sensor_assisted_profile_enabled(cfg):
        return SENSOR_ASSISTED_DISALLOWED_MODALITIES
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    configured = hist_cfg.get("excluded_sensitive_fields")
    if configured:
        return tuple(str(item) for item in configured)
    return ()


def _modality_profile_metadata(plan: Mapping[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = list(_enabled_modalities(dict(plan), cfg))
    profile = _matrix_profile(cfg)
    return {
        "profile": profile,
        "enabled_modalities": enabled,
        "excluded_sensitive_fields": list(_excluded_sensitive_fields(cfg)),
        "sensor_assisted": bool(sensor_assisted_profile_enabled(cfg)),
    }


def _cfg_for_scene(cfg: dict[str, Any], scene: Any) -> dict[str, Any]:
    scene_cfg = deepcopy(cfg)
    dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
    if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw":
        dataset_cfg["scene"] = str(scene)
        loso_cfg = scene_cfg.get("loso", {}) if isinstance(scene_cfg.get("loso"), dict) else {}
        roots = loso_cfg.get("scene_data_roots") if isinstance(loso_cfg.get("scene_data_roots"), dict) else {}
        root = roots.get(str(scene), roots.get(scene)) if isinstance(roots, dict) else None
        if root:
            dataset_cfg["data_root"] = str(root)
        csv_names = loso_cfg.get("scene_csv_names") if isinstance(loso_cfg.get("scene_csv_names"), dict) else {}
        scene_csv = csv_names.get(str(scene), csv_names.get(scene)) if isinstance(csv_names, dict) else None
        if isinstance(scene_csv, dict):
            for key in ("train_csv_name", "test_csv_name", "val_csv_name"):
                if scene_csv.get(key):
                    dataset_cfg[key] = scene_csv[key]
        return scene_cfg
    normalize_deepsense_dataset_config(dataset_cfg)
    retarget_deepsense_dataset_config(dataset_cfg, scene)
    loso_cfg = scene_cfg.get("loso", {}) if isinstance(scene_cfg.get("loso"), dict) else {}
    roots = loso_cfg.get("scene_data_roots") if isinstance(loso_cfg.get("scene_data_roots"), dict) else {}
    root = roots.get(str(scene), roots.get(scene)) if isinstance(roots, dict) else None
    if root:
        dataset_cfg["data_root"] = str(root)
    csv_names = loso_cfg.get("scene_csv_names") if isinstance(loso_cfg.get("scene_csv_names"), dict) else {}
    scene_csv = csv_names.get(str(scene), csv_names.get(scene)) if isinstance(csv_names, dict) else None
    if isinstance(scene_csv, dict):
        for key in ("train_csv_name", "test_csv_name", "val_csv_name"):
            if scene_csv.get(key):
                dataset_cfg[key] = scene_csv[key]
    return scene_cfg


def _stage_cfg(
    cfg: dict[str, Any],
    run: Mapping[str, Any],
    *,
    variant: str,
    stage_name: str,
    stage_dir: Path,
) -> dict[str, Any]:
    stage_cfg = deepcopy(cfg)
    apply_history_anchor_model_config(stage_cfg)
    ensure_distillation_defaults(stage_cfg)
    stage_cfg.setdefault("experiment", {})["seed"] = int(run.get("seed", 0))
    stage_cfg["experiment"]["name"] = f"{cfg.get('experiment', {}).get('name', 'hist_beam_loso')}_{stage_name}"
    model_variant = canonical_image_only_variant(variant)
    model_cfg = stage_cfg.setdefault("model", {})
    model_cfg["modalities"] = list(_enabled_modalities({"enabled_modalities": model_cfg.get("modalities")}, stage_cfg))
    for key in ("student", "teacher"):
        role = model_cfg.get(key)
        if isinstance(role, dict):
            role["variant"] = model_variant
            role["modalities"] = list(model_cfg["modalities"])
    stage_cfg.setdefault("hist_beam", {})["variant"] = model_variant
    hist_cfg = stage_cfg.setdefault("hist_beam", {})
    student_cfg = model_cfg.get("student") if isinstance(model_cfg.get("student"), dict) else {}
    if image_only_protocol_enabled(stage_cfg):
        _apply_image_only_stage_defaults(stage_cfg, hist_cfg, student_cfg, run_variant=variant, model_variant=model_variant)
    if model_variant in {"v6_radio_proto", "adapter_radio_proto"}:
        radio_cfg = hist_cfg.setdefault("radio_semantic", {})
        radio_cfg.setdefault("enabled", True)
        radio_cfg.setdefault("mode", "peak_spread")
        radio_cfg.setdefault("num_spread_bins", 3)
        radio_cfg.setdefault("entropy_thresholds", [0.35, 0.65])
        hist_cfg["proto_type"] = "radio_semantic"
        hist_cfg.setdefault("prototype", {})["proto_type"] = "radio_semantic"
        weights = hist_cfg.setdefault("loss_weights", {})
        weights.setdefault("radio_semantic", 1.0)
        dataset_cfg = stage_cfg.setdefault("data", {}).setdefault("dataset", {})
        dataset_cfg.setdefault("radio_semantic", dict(radio_cfg))
        if isinstance(student_cfg, dict):
            student_cfg.setdefault("radio_semantic", dict(radio_cfg))
            student_cfg.setdefault("use_radio_head", True)
            student_cfg.setdefault("num_radio_classes", int(radio_cfg.get("num_radio_classes", 24)))
            student_cfg.setdefault("proto_type", "radio_semantic")
            student_cfg.setdefault("radio_tau", float(hist_cfg.get("radio_tau", 1.0)))
            if model_variant == "adapter_radio_proto":
                student_cfg.setdefault("use_radio_condition_in_beam_head", False)
            else:
                student_cfg.setdefault(
                    "use_radio_condition_in_beam_head",
                    bool(radio_cfg.get("use_radio_condition_in_beam_head", True)),
                )
    elif is_v7_variant(model_variant):
        apply_v7_stage_defaults(stage_cfg, hist_cfg, student_cfg)
    elif model_variant in {"v8_target_prior_head", "v9_input_conditioned_target_adaptation"}:
        v8_cfg = hist_cfg.setdefault("v8", {})
        mode = str(v8_cfg.get("mode", "target_prior_head")).strip().lower()
        v8_cfg.setdefault("mode", mode)
        v8_cfg.setdefault("adapter_dim", 64)
        v8_cfg.setdefault("adapter_dropout", 0.1)
        v8_cfg.setdefault("use_adapter", mode != "target_linear_probe")
        v8_cfg.setdefault("use_target_prior", mode != "target_linear_probe")
        v8_cfg.setdefault("use_source_logits_in_final", mode == "source_prior_only")
        v8_cfg.setdefault("lambda_src", 1.0 if mode == "source_prior_only" else 0.0)
        v8_cfg.setdefault("lambda_tgt", 0.0 if mode == "source_prior_only" else 1.0)
        v8_cfg.setdefault("beta_prior", 1.0)
        v8_cfg.setdefault("learnable_beta_prior", False)
        v8_cfg.setdefault("use_coarse_to_fine", mode == "target_prior_coarse_to_fine")
        v8_cfg.setdefault("sector_size", int(hist_cfg.get("group_size", student_cfg.get("group_size", 8) if isinstance(student_cfg, dict) else 8)))
        v8_cfg.setdefault("unfreeze_last_fusion_block", False)
        v8_cfg.setdefault("use_soft_beam_label", True)
        v8_cfg.setdefault("soft_label_sigma", 1.0)
        v8_cfg.setdefault("prior_sigma", 1.5)
        v8_cfg.setdefault("prior_eps", 1.0e-4)
        v8_cfg.setdefault("loss_prior_smooth_weight", 0.001)
        v8_cfg.setdefault("run_prototype_probe", False)
        hist_cfg.setdefault("adaptation", {}).setdefault("strategy", "v8_target_head_only")
        weights = hist_cfg.setdefault("loss_weights", {})
        weights.setdefault("v8_final_ce", 1.0)
        weights.setdefault("v8_prior_smooth", float(v8_cfg.get("loss_prior_smooth_weight", 0.001)))
        weights.setdefault("v8_sector_ce", 0.2)
        weights.setdefault("v8_offset_ce", 0.2)
        source_train = hist_cfg.setdefault("source_train", {})
        source_train.setdefault("loss_type", "cross_entropy")
        source_train.setdefault("class_prior_from_source_train", False)
        source_train.setdefault("logit_adjust_tau", 1.0)
        source_train.setdefault("debiased_loss_available", False)
        source_train.setdefault("unsupported_reason", "source_long_tail_debias_not_implemented")
        if model_variant == "v9_input_conditioned_target_adaptation":
            v9_cfg = hist_cfg.setdefault("v9", {})
            v9_cfg.setdefault("use_target_prior", True)
            v9_cfg.setdefault("beta_prior_max", 1.0)
            v9_cfg.setdefault("learnable_beta_prior", True)
            v9_cfg.setdefault("prior_dropout", 0.0)
            v9_cfg.setdefault("use_prototype_logits", True)
            v9_cfg.setdefault("prototype_type", "beam")
            v9_cfg.setdefault("prototype_tau", 0.1)
            v9_cfg.setdefault("eta_prototype", 1.0)
            v9_cfg.setdefault("sector_size", 2)
            v9_cfg.setdefault("prototype_feature_source", "target_adapter")
            v9_cfg.setdefault("use_widened_prior_marginal_kl", False)
            v9_cfg.setdefault("widened_prior_sigma", 3.0)
            v9_cfg.setdefault("widened_prior_temperature", 1.5)
            v9_cfg.setdefault("loss_widened_prior_marginal_kl_weight", 0.0)
            hist_cfg.setdefault("adaptation", {}).setdefault("strategy", "v9_target_head_only")
            weights.setdefault(
                "v9_widened_prior_marginal_kl",
                float(v9_cfg.get("loss_widened_prior_marginal_kl_weight", 0.0)),
            )
        if isinstance(student_cfg, dict):
            student_cfg.setdefault("v8", dict(v8_cfg))
            if model_variant == "v9_input_conditioned_target_adaptation":
                student_cfg.setdefault("v9", dict(hist_cfg.get("v9", {})))
            student_cfg.setdefault("adapter", {"enabled": True})
    elif model_variant in {"v8_path_proto", "adapter_path_proto"}:
        path_cfg = hist_cfg.setdefault("path_semantic", {})
        path_cfg.setdefault("enabled", True)
        path_cfg.setdefault("mode", "kmeans_path_descriptor")
        path_cfg.setdefault("num_path_classes", 24)
        path_cfg.setdefault("fit_on_source_only", True)
        path_cfg.setdefault("fallback_if_missing", "radio_power")
        path_cfg.setdefault("use_path_regression", True)
        hist_cfg["proto_type"] = "path"
        hist_cfg.setdefault("prototype", {})["proto_type"] = "path"
        adapt_cfg = hist_cfg.setdefault("adaptation", {})
        adapt_cfg.setdefault("proto_type", "path")
        adapt_cfg.setdefault("proto_tau", 0.1)
        adapt_cfg.setdefault("confidence_threshold", 0.75)
        adapt_cfg.setdefault("proto_warmup_epochs", 5)
        adapt_cfg.setdefault("target_proto_momentum", 0.9)
        adapt_cfg.setdefault("allow_labeled_target_path_supervision", False)
        weights = hist_cfg.setdefault("loss_weights", {})
        weights.setdefault("lambda_path", 0.3)
        weights.setdefault("lambda_path_reg", 0.05)
        dataset_cfg = stage_cfg.setdefault("data", {}).setdefault("dataset", {})
        dataset_cfg.setdefault("path_semantic", dict(path_cfg))
        if isinstance(student_cfg, dict):
            student_cfg.setdefault("path_semantic", dict(path_cfg))
            student_cfg.setdefault("use_path_head", True)
            student_cfg.setdefault("use_path_condition_in_beam_head", model_variant != "adapter_path_proto")
            student_cfg.setdefault("path_embed_dim", 32)
            student_cfg.setdefault("num_path_classes", int(path_cfg.get("num_path_classes", 24)))
            student_cfg.setdefault("proto_type", "path")
    elif model_variant in {"v5_adapter_proto", "adapter_proto"}:
        hist_cfg["proto_type"] = "coarse"
        hist_cfg.setdefault("prototype", {})["proto_type"] = "coarse"
    stage_cfg.setdefault("output", {})["dir"] = str(stage_dir)
    stage_cfg["output"]["run_name"] = stage_name
    stage_cfg["output"]["group_by_scene"] = False
    stage_cfg["output"].setdefault("progress", {})["enabled"] = False
    if model_variant == "v0_flat":
        weights = stage_cfg.setdefault("hist_beam", {}).setdefault("loss_weights", {})
        weights.update({"hierarchical": 0.0, "flat": 1.0})
    return stage_cfg


def _enabled_modalities(plan: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, ...]:
    if plan.get("enabled_modalities"):
        return tuple(str(item) for item in plan["enabled_modalities"])
    if not sensor_assisted_profile_enabled(cfg):
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        for context, raw in (
            ("model.modalities", model_cfg.get("modalities")),
            ("model.student.modalities", model_cfg.get("student", {}).get("modalities") if isinstance(model_cfg.get("student"), dict) else None),
            ("model.teacher.modalities", model_cfg.get("teacher", {}).get("modalities") if isinstance(model_cfg.get("teacher"), dict) else None),
        ):
            if raw:
                return tuple(normalize_modalities(raw, context=context))
    return resolve_enabled_modalities(cfg)


def _reuse_source_checkpoint(cfg: dict[str, Any]) -> bool:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    return bool(loso_cfg.get("reuse_source_checkpoint", True))


def _prototype_decision(run: Mapping[str, Any], cfg: dict[str, Any], *, source_variant: str) -> dict[str, Any]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    proto_cfg = hist_cfg.get("prototype", {}) if isinstance(hist_cfg.get("prototype"), dict) else {}
    strategy = str(proto_cfg.get("strategy", proto_cfg.get("generation", "auto"))).strip().lower()
    run_variant = str(run.get("variant"))
    requires = run_variant in {"v5_adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}
    explicit_save = bool(proto_cfg.get("save_source_prototypes", False))
    if strategy in {"off", "skip", "none"}:
        return {"generate": False, "status": "skipped", "reason": "prototype_strategy_off", "strategy": strategy}
    if strategy in {"always", "force"} or explicit_save:
        return {"generate": True, "status": "pending", "reason": "prototype_strategy_forced", "strategy": strategy}
    if requires:
        return {"generate": True, "status": "pending", "reason": f"variant_requires_prototype:{run_variant}", "strategy": strategy}
    return {
        "generate": False,
        "status": "skipped",
        "reason": f"source_only_variant:{source_variant}",
        "strategy": strategy,
    }


def _throughput_config_summary(cfg: dict[str, Any], *, prototype_strategy: str | None) -> dict[str, Any]:
    loader_cfg = cfg.get("data", {}).get("dataloader", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    cache_cfg = cfg.get("data", {}).get("cache", {}) if isinstance(cfg.get("data"), dict) else {}
    image_cache_cfg = cache_cfg.get("image", {}) if isinstance(cache_cfg.get("image", {}), dict) else {}
    lidar_cache_cfg = cache_cfg.get("lidar", {}) if isinstance(cache_cfg.get("lidar", {}), dict) else {}
    return {
        "batch_size": loader_cfg.get("batch_size", loader_cfg.get("train_batch_size")),
        "num_workers": loader_cfg.get("train_num_workers", loader_cfg.get("num_workers")),
        "persistent_workers": loader_cfg.get("train_persistent_workers", loader_cfg.get("persistent_workers")),
        "prefetch_factor": loader_cfg.get("train_prefetch_factor", loader_cfg.get("prefetch_factor")),
        "enabled_modalities": list(resolve_enabled_modalities(cfg)),
        "seq_len": dataset_cfg.get("seq_len"),
        "modality_profile": _matrix_profile(cfg),
        "image_cache_policy": image_cache_cfg.get("policy", cache_cfg.get("policy", "auto")),
        "lidar_cache_policy": lidar_cache_cfg.get("policy", cache_cfg.get("policy", "auto")),
        "lidar_cache_dir": dataset_cfg.get("lidar_cache_dir"),
        "cpu_threads": _cpu_thread_config(cfg),
        "prototype_strategy": prototype_strategy,
    }


def _source_variant_for(run: Mapping[str, Any]) -> str:
    variant = str(run.get("variant"))
    if variant in IMAGE_ONLY_VARIANTS:
        return "v0_flat"
    if is_v7_variant(variant):
        return DEFAULT_SOURCE_BASELINE_VARIANT
    if variant in {"v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}:
        return DEFAULT_SOURCE_BASELINE_VARIANT
    if variant in ADAPTATION_VARIANTS:
        return DEFAULT_SOURCE_BASELINE_VARIANT
    return variant


def _apply_image_only_stage_defaults(
    stage_cfg: dict[str, Any],
    hist_cfg: dict[str, Any],
    student_cfg: Mapping[str, Any] | dict[str, Any],
    *,
    run_variant: str,
    model_variant: str,
) -> None:
    dataset_cfg = stage_cfg.setdefault("data", {}).setdefault("dataset", {})
    model_cfg = stage_cfg.setdefault("model", {})
    dataset_cfg["enabled_modalities"] = ["image"]
    dataset_cfg["use_gps"] = False
    dataset_cfg["use_lidar"] = False
    dataset_cfg["use_mmwave"] = False
    dataset_cfg["use_csi"] = False
    dataset_cfg["return_beam_power"] = False
    dataset_cfg["return_geometry"] = False
    dataset_cfg["return_modality_availability"] = True
    dataset_cfg["radio_semantic"] = {"enabled": False, "return_beam_power": False}
    dataset_cfg["path_semantic"] = {"enabled": False}
    dataset_cfg["physical_label"] = {"enabled": False}
    model_cfg["modalities"] = ["image"]
    hist_cfg["modalities"] = ["image"]
    hist_cfg.setdefault("disabled_modalities", ["gps", "lidar", "radar", "mmwave", "csi"])
    hist_cfg.setdefault("excluded_sensitive_fields", ["gps", "lidar", "radar", "mmwave", "csi", "channel", "path", "beam_power"])
    hist_cfg.setdefault("protocol", {})["image_only"] = True
    hist_cfg.setdefault("image_only", {})["fusion_mode"] = "identity"
    hist_cfg.setdefault("collapse_diagnostics", {})["enabled"] = True
    if isinstance(student_cfg, dict):
        student_cfg["variant"] = model_variant
        student_cfg["modalities"] = ["image"]
        student_cfg.setdefault("image_only", {})["fusion_mode"] = "identity"
        student_cfg.setdefault("radio_semantic", {"enabled": False})
        student_cfg.setdefault("path_semantic", {"enabled": False})
        student_cfg["use_radio_head"] = False
        student_cfg["use_path_head"] = False
    if run_variant == "image_target_linear_probe":
        v8_cfg = hist_cfg.setdefault("v8", {})
        v8_cfg.update(
            {
                "mode": "target_linear_probe",
                "use_adapter": False,
                "use_target_prior": False,
                "use_source_logits_in_final": False,
                "learnable_beta_prior": False,
                "beta_prior": 0.0,
                "loss_prior_smooth_weight": 0.0,
            }
        )
        hist_cfg.setdefault("adaptation", {})["strategy"] = "image_target_linear_probe"
        if isinstance(student_cfg, dict):
            student_cfg["v8"] = dict(v8_cfg)
    elif run_variant == "image_v8_target_prior_head":
        v8_cfg = hist_cfg.setdefault("v8", {})
        v8_cfg.setdefault("mode", "target_prior_head")
        v8_cfg.setdefault("use_adapter", True)
        v8_cfg.setdefault("use_target_prior", True)
        v8_cfg.setdefault("use_source_logits_in_final", False)
        v8_cfg.setdefault("beta_prior", 0.5)
        v8_cfg.setdefault("learnable_beta_prior", True)
        hist_cfg.setdefault("adaptation", {})["strategy"] = "v8_target_head_only"
        if isinstance(student_cfg, dict):
            student_cfg["v8"] = dict(v8_cfg)
    elif run_variant == "image_v9_sector_proto":
        v8_cfg = hist_cfg.setdefault("v8", {})
        v8_cfg.setdefault("mode", "target_prior_head")
        v8_cfg.setdefault("use_source_logits_in_final", False)
        v9_cfg = hist_cfg.setdefault("v9", {})
        v9_cfg.update(
            {
                "use_target_prior": True,
                "use_prototype_logits": True,
                "prototype_type": "sector",
                "use_beam_proto": False,
                "sector_size": int(v9_cfg.get("sector_size", 2)),
                "prototype_tau": float(v9_cfg.get("prototype_tau", 0.1)),
                "eta_prototype": float(v9_cfg.get("eta_prototype", 1.0)),
            }
        )
        hist_cfg.setdefault("adaptation", {})["strategy"] = "v9_target_head_only"
        if isinstance(student_cfg, dict):
            student_cfg["v8"] = dict(v8_cfg)
            student_cfg["v9"] = dict(v9_cfg)


def _source_cache_key(run: Mapping[str, Any], variant: str) -> str:
    sources = "-".join(str(item) for item in run.get("source_scenes", []))
    return f"{run.get('fold')}|target={run.get('target_scene')}|sources={sources}|variant={variant}|seed={run.get('seed')}"


def _adaptation_cache_key(run: Mapping[str, Any]) -> str:
    return f"{run.get('fold')}|{run.get('variant')}|budget={run.get('budget')}|seed={run.get('seed')}"


def _matrix_profile(cfg: dict[str, Any]) -> str | None:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    for value in (
        loso_cfg.get("profile"),
        loso_cfg.get("matrix_profile"),
        hist_cfg.get("profile"),
        dataset_cfg.get("modality_profile"),
    ):
        if value not in (None, ""):
            return str(value)
    if sensor_assisted_profile_enabled(cfg):
        return SENSOR_ASSISTED_PROFILE
    return None
