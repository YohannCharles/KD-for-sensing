from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.data.scenes import normalize_deepsense_dataset_config, retarget_deepsense_dataset_config
from kd_sensing.engine.hist_beam_history_anchor import apply_history_anchor_model_config
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
SOURCE_ONLY_VARIANTS = {"v0_flat", "v1_hierarchical", "v2_shared_private", "v3_decoupled"}
ADAPTATION_VARIANTS = {"v4_adapter", "v5_adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto", "v7_shared_physical_private_residual", "v6_full_finetune"}
SUPPORTED_VARIANTS = SOURCE_ONLY_VARIANTS | ADAPTATION_VARIANTS
DEFAULT_QUICK_VARIANTS = ["v0_flat", "v3_decoupled", "v4_adapter", "v5_adapter_proto", "v6_radio_proto", "v8_path_proto", "v6_full_finetune"]
SENSOR_ASSISTED_QUICK_VARIANTS = ["v3_decoupled", "v4_adapter", "v6_radio_proto", "v8_path_proto", "adapter_path_proto", "v7_shared_physical_private_residual", "v6_full_finetune"]
SENSOR_ASSISTED_QUICK_BUDGETS = [10]
SENSOR_ASSISTED_QUICK_SEEDS = [0, 1]
DEFAULT_QUICK_BUDGETS = [0, 10]
DEFAULT_QUICK_SEEDS = [0]
DEFAULT_QUICK_TARGET_SCENES = [34]
EXECUTION_PROGRESS_FILENAME = "execution_progress.jsonl"

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
    model_cfg = stage_cfg.setdefault("model", {})
    model_cfg["modalities"] = list(_enabled_modalities({"enabled_modalities": model_cfg.get("modalities")}, stage_cfg))
    for key in ("student", "teacher"):
        role = model_cfg.get(key)
        if isinstance(role, dict):
            role["variant"] = variant
            role["modalities"] = list(model_cfg["modalities"])
    stage_cfg.setdefault("hist_beam", {})["variant"] = variant
    hist_cfg = stage_cfg.setdefault("hist_beam", {})
    student_cfg = model_cfg.get("student") if isinstance(model_cfg.get("student"), dict) else {}
    if variant in {"v6_radio_proto", "adapter_radio_proto"}:
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
            if variant == "adapter_radio_proto":
                student_cfg.setdefault("use_radio_condition_in_beam_head", False)
            else:
                student_cfg.setdefault(
                    "use_radio_condition_in_beam_head",
                    bool(radio_cfg.get("use_radio_condition_in_beam_head", True)),
                )
    elif is_v7_variant(variant):
        apply_v7_stage_defaults(stage_cfg, hist_cfg, student_cfg)
    elif variant in {"v8_path_proto", "adapter_path_proto"}:
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
            student_cfg.setdefault("use_path_condition_in_beam_head", variant != "adapter_path_proto")
            student_cfg.setdefault("path_embed_dim", 32)
            student_cfg.setdefault("num_path_classes", int(path_cfg.get("num_path_classes", 24)))
            student_cfg.setdefault("proto_type", "path")
    elif variant in {"v5_adapter_proto", "adapter_proto"}:
        hist_cfg["proto_type"] = "coarse"
        hist_cfg.setdefault("prototype", {})["proto_type"] = "coarse"
    stage_cfg.setdefault("output", {})["dir"] = str(stage_dir)
    stage_cfg["output"]["run_name"] = stage_name
    stage_cfg["output"]["group_by_scene"] = False
    stage_cfg["output"].setdefault("progress", {})["enabled"] = False
    if variant == "v0_flat":
        weights = stage_cfg.setdefault("hist_beam", {}).setdefault("loss_weights", {})
        weights.update({"hierarchical": 0.0, "flat": 1.0, "orthogonality": 0.0, "scene_confusion": 0.0, "scene_private": 0.0})
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
    if is_v7_variant(variant):
        return variant
    if variant in {"v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}:
        return variant
    if variant in ADAPTATION_VARIANTS:
        return "v3_decoupled"
    return variant


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