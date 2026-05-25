from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

from kd_sensing.data.samples import create_samples
from kd_sensing.data.scenes import normalize_deepsense_dataset_config
from kd_sensing.data.transform_ops.lidar import lidar_cache_path, parameterized_lidar_cache_dir
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.utils.paths import resolve_path


def recommend_parallel_training(
    cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    parallel_runs: int = 4,
    cpu_count: int | None = None,
    cache_min_coverage: float = 0.95,
    check_cache: bool = True,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parallel_runs = max(1, int(parallel_runs))
    cpu_count = int(cpu_count or os.cpu_count() or 1)
    modalities = list(resolve_enabled_modalities(cfg))
    worker_budget = _recommended_worker_budget(
        parallel_runs=parallel_runs,
        cpu_count=cpu_count,
        modality_count=len(modalities),
    )
    prefetch_factor = 1 if parallel_runs >= 3 or len(modalities) >= 4 else 2
    is_multimodal_nf = str(cfg.get("data", {}).get("dataset", {}).get("type", "")).lower() == "multimodal_nf"
    lidar_cache = lidar_cache_coverage(cfg) if check_cache and "lidar" in modalities else _skipped_cache_status(modalities)
    cache_policy = _recommended_cache_policy(lidar_cache, cache_min_coverage)
    overrides = [
        f"data.dataloader.train_num_workers={worker_budget['train_num_workers']}",
        f"data.dataloader.test_num_workers={worker_budget['test_num_workers']}",
        f"data.dataloader.train_persistent_workers={str(worker_budget['train_persistent_workers']).lower()}",
        "data.dataloader.test_persistent_workers=false",
        f"data.dataloader.train_prefetch_factor={prefetch_factor}",
        f"data.dataloader.test_prefetch_factor={prefetch_factor}",
        "output.progress.enabled=false",
    ]
    if cache_policy is not None and not is_multimodal_nf:
        overrides.append(f"data.cache.policy={cache_policy}")
    multimodal_nf_cache = (
        multimodal_nf_derived_cache_recommendation(cfg, modalities=modalities, min_coverage=cache_min_coverage)
        if is_multimodal_nf
        else {}
    )
    if is_multimodal_nf:
        for modality in ("image", "lidar"):
            if modality in modalities:
                overrides.append(f"data.cache.multimodal_nf.{modality}.policy={multimodal_nf_cache[modality]['recommended_policy']}")
                overrides.append(f"data.cache.multimodal_nf.{modality}.validation_mode=lightweight")
        if {"image", "lidar"} & set(modalities):
            overrides.extend(
                [
                    "training.epoch_subsampling.shuffle=false",
                    "training.epoch_subsampling.order=locality",
                ]
            )

    optional_overrides = []
    if not is_multimodal_nf or {"image", "lidar"} & set(modalities):
        optional_overrides.extend(
            [
                "training.amp.enabled=true",
                "training.amp.dtype=float16",
            ]
        )
    multimodal_nf_io = (
        _multimodal_nf_io_recommendations(
            modalities=modalities,
            parallel_runs=parallel_runs,
            cpu_count=cpu_count,
            worker_budget=worker_budget,
            profile=profile,
        )
        if is_multimodal_nf
        else {}
    )
    train_command = _train_command(config_path, overrides)
    return {
        "parallel_runs": parallel_runs,
        "cpu_count": cpu_count,
        "modalities": modalities,
        "scenario": "background_parallel_training",
        "overrides": overrides,
        "optional_overrides": optional_overrides,
        "recommendations": {
            **worker_budget,
            "prefetch_factor": prefetch_factor,
            "progress_enabled": False,
            "cache_policy": cache_policy,
            "amp": "optional_after_cache_and_loader_wait_are_under_control",
            "multimodal_nf_io": multimodal_nf_io,
        },
        "cache": {
            "lidar": lidar_cache,
            "multimodal_nf": multimodal_nf_cache,
            "min_coverage_for_read_only": float(cache_min_coverage),
            "prewarm_command": (
                _multimodal_nf_prewarm_command() if is_multimodal_nf and multimodal_nf_cache else
                _lidar_prewarm_command() if _needs_lidar_prewarm(lidar_cache, cache_min_coverage) else None
            ),
        },
        "commands": {
            "train": train_command,
            "tensorboard": "tensorboard --logdir outputs",
        },
        "notes": [
            "These overrides are for background parallel runs, not a replacement for single-experiment defaults.",
            "Keep AMP optional until profile output shows DataLoader wait is no longer dominating GPU step time.",
            "With output.progress.enabled=false, use epoch logs, train_log.json, and TensorBoard instead of batch tqdm.",
            *(_profile_driven_notes(profile) if is_multimodal_nf else []),
            *(_multimodal_nf_notes(modalities) if is_multimodal_nf else []),
        ],
    }


def multimodal_nf_derived_cache_recommendation(
    cfg: dict[str, Any],
    *,
    modalities: list[str] | None = None,
    min_coverage: float = 0.95,
) -> dict[str, Any]:
    selected = modalities or list(resolve_enabled_modalities(cfg))
    cache_cfg = cfg.get("data", {}).get("cache", {}).get("multimodal_nf", {})
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    cache_dir = dataset_cfg.get("cache_dir") or "dataset/MultimodalNF/cache"
    result = {}
    for modality in ("image", "lidar"):
        if modality not in selected:
            continue
        configured = cache_cfg.get(modality, {}) if isinstance(cache_cfg, dict) else {}
        configured_path = configured.get("path") or configured.get("cache_path") if isinstance(configured, dict) else None
        exists = bool(configured_path and Path(str(configured_path)).exists())
        coverage = 1.0 if exists else 0.0
        result[modality] = {
            "status": "warm" if coverage >= min_coverage else "unknown_or_cold",
            "coverage": coverage,
            "cache_dir": str(cache_dir),
            "configured_path": str(configured_path) if configured_path else None,
            "recommended_policy": "read_only" if coverage >= min_coverage else "auto",
            "recommended_validation_mode": "lightweight",
            "prewarm_command": _multimodal_nf_prewarm_command(),
        }
    return result


def lidar_cache_coverage(cfg: dict[str, Any], *, splits: tuple[str, ...] = ("train", "test")) -> dict[str, Any]:
    try:
        modalities = list(resolve_enabled_modalities(cfg))
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "coverage": 0.0}
    if "lidar" not in modalities:
        return {"status": "not_applicable", "enabled": False, "coverage": 1.0}

    dataset_cfg = _resolved_dataset_config(cfg, modalities)
    cache_dir = _resolved_lidar_cache_dir(dataset_cfg)
    if cache_dir is None:
        return {"status": "missing_cache_dir", "enabled": True, "coverage": 0.0}

    all_paths: set[str] = set()
    split_errors: dict[str, str] = {}
    for split in splits:
        try:
            all_paths.update(_lidar_paths_for_split(dataset_cfg, modalities, split=split))
        except Exception as exc:
            split_errors[split] = str(exc)
    if split_errors and not all_paths:
        return {
            "status": "error",
            "enabled": True,
            "cache_dir": str(cache_dir),
            "coverage": 0.0,
            "split_errors": split_errors,
        }

    existing = sum(1 for rel_path in all_paths if lidar_cache_path(cache_dir, rel_path).exists())
    total = len(all_paths)
    coverage = float(existing / total) if total else 1.0
    status = "warm" if total and coverage >= 1.0 else "partial" if total and coverage > 0 else "cold"
    return {
        "status": status,
        "enabled": True,
        "cache_dir": str(cache_dir),
        "coverage": coverage,
        "existing": int(existing),
        "missing": int(max(total - existing, 0)),
        "total": int(total),
        "split_errors": split_errors,
    }


def _recommended_worker_budget(*, parallel_runs: int, cpu_count: int, modality_count: int) -> dict[str, Any]:
    reserve = max(1, parallel_runs)
    usable_cpus = max(1, cpu_count - reserve)
    modality_factor = 2 if modality_count >= 4 else 1
    train_num_workers = usable_cpus // max(parallel_runs * modality_factor, 1)
    train_num_workers = max(1, min(4, train_num_workers))
    test_num_workers = max(0, min(1, train_num_workers // 2))
    return {
        "train_num_workers": int(train_num_workers),
        "test_num_workers": int(test_num_workers),
        "train_persistent_workers": bool(train_num_workers > 0),
        "test_persistent_workers": False,
    }


def _recommended_cache_policy(lidar_cache: dict[str, Any], min_coverage: float) -> str | None:
    if lidar_cache.get("status") == "not_applicable":
        return None
    if float(lidar_cache.get("coverage", 0.0)) >= float(min_coverage):
        return "read_only"
    return "auto"


def _needs_lidar_prewarm(lidar_cache: dict[str, Any], min_coverage: float) -> bool:
    return lidar_cache.get("enabled", False) and float(lidar_cache.get("coverage", 0.0)) < float(min_coverage)


def _resolved_dataset_config(cfg: dict[str, Any], modalities: list[str]) -> dict[str, Any]:
    dataset_cfg = deepcopy(cfg.get("data", {}).get("dataset", {}))
    normalize_deepsense_dataset_config(dataset_cfg)
    dataset_cfg.update(dataset_flags_for_modalities(modalities))
    return dataset_cfg


def _resolved_lidar_cache_dir(dataset_cfg: dict[str, Any]) -> Path | None:
    cache_dir = dataset_cfg.get("lidar_cache_dir")
    if not cache_dir:
        return None
    data_root = resolve_path(dataset_cfg.get("data_root", "."))
    base = Path(cache_dir).expanduser()
    if not base.is_absolute():
        base = data_root / base
    return parameterized_lidar_cache_dir(
        base,
        bev_size=dataset_cfg.get("lidar_bev_size", [224, 224]),
        roi=dataset_cfg.get("lidar_roi", [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0]),
        fov_degrees=dataset_cfg.get("lidar_fov_degrees"),
        remove_ground=bool(dataset_cfg.get("lidar_remove_ground", False)),
        ground_z_threshold=float(dataset_cfg.get("lidar_ground_z_threshold", 0.1)),
        background_path=dataset_cfg.get("lidar_background_path"),
        background_distance_threshold=float(dataset_cfg.get("lidar_background_distance_threshold", 0.2)),
    )


def _lidar_paths_for_split(dataset_cfg: dict[str, Any], modalities: list[str], *, split: str) -> set[str]:
    data_root = resolve_path(dataset_cfg.get("data_root", "."))
    csv_name = dataset_cfg.get("train_csv_name") if split == "train" else dataset_cfg.get("test_csv_name")
    if not csv_name:
        return set()
    root_csv = Path(str(csv_name))
    if not root_csv.is_absolute():
        root_csv = data_root / root_csv
    samples = create_samples(
        root_csv,
        portion=float(dataset_cfg.get("portion", 1.0)),
        enabled_modalities=modalities,
        seq_len=int(dataset_cfg.get("seq_len", 8)),
        num_pred=int(dataset_cfg.get("num_pred", 3)),
        portion_strategy=str(dataset_cfg.get("portion_strategy", "even")),
        portion_seed=int(dataset_cfg.get("portion_seed", 42)),
    )
    paths = getattr(samples, "lidar_paths", None) or []
    seq_len = int(dataset_cfg.get("seq_len", 8))
    return {
        str(rel_path)
        for sequence in paths
        for rel_path in sequence[-seq_len:]
        if str(rel_path).strip() and str(rel_path).strip() != "-99"
    }


def _train_command(config_path: str | Path | None, overrides: list[str]) -> str | None:
    if config_path is None:
        return None
    override_args = " ".join(f"-o {item}" for item in overrides)
    return f"conda run -n kd_mm_beam python scripts/train.py --config {config_path} {override_args}".strip()


def _lidar_prewarm_command() -> str:
    return "conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/lidar_bev_cache.yaml"


def _multimodal_nf_prewarm_command() -> str:
    return "conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/multimodal_nf_derived_cache.yaml"


def _multimodal_nf_notes(modalities: list[str]) -> list[str]:
    if not ({"image", "lidar"} & set(modalities)):
        return ["Multimodal-NF GPS-only runs do not need image/LiDAR derived cache; tune ordinary DataLoader and model step settings first."]
    return [
        "For Multimodal-NF image/LiDAR/fusion runs, prewarm derived caches before long training runs or use auto policy for the first run.",
        "Warm cache training should use lightweight runtime validation; reserve strong source fingerprint scans for audit, rebuild, or suspected data drift.",
        "Use training.epoch_subsampling.shuffle=false or training.epoch_subsampling.order=locality when random cache reads dominate loader wait.",
        "Do not treat read_only derived cache as guaranteed fast if the epoch order still performs random windows over large mmap files.",
        "Spread image/LiDAR/fusion jobs across available GPUs before stacking multiple heavy IO runs on one GPU.",
    ]


def _multimodal_nf_io_recommendations(
    *,
    modalities: list[str],
    parallel_runs: int,
    cpu_count: int,
    worker_budget: dict[str, Any],
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    heavy_modalities = sorted({"image", "lidar"} & set(modalities))
    if not heavy_modalities:
        return {
            "heavy_io_modalities": [],
            "cache_validation_mode": None,
            "epoch_subsampling_order": None,
            "gpu_assignment": "not_applicable_for_gps_csi_only",
            "profile_command": "Run scripts/profile_training_io.py only if DataLoader wait appears in logs.",
        }
    io_risk = profile.get("io_risk", {}) if isinstance(profile, dict) else {}
    loader_wait_risk = bool(io_risk.get("loader_wait_dominates_step", False))
    cache_tail_risk = bool(io_risk.get("cache_read_tail_risk", False))
    max_workers = int(worker_budget.get("train_num_workers", 1))
    if loader_wait_risk or cache_tail_risk or parallel_runs >= 4:
        max_workers = max(1, min(max_workers, 2))
    return {
        "heavy_io_modalities": heavy_modalities,
        "cache_validation_mode": "lightweight",
        "avoid_repeated_strong_validation": True,
        "epoch_subsampling_shuffle": False,
        "epoch_subsampling_order": "locality",
        "progress_enabled": False,
        "amp": "recommended_for_cuda_image_or_fusion_when_model_step_is_not_loader_bound",
        "train_num_workers_upper_bound": int(max_workers),
        "prefetch_factor": 1,
        "gpu_assignment": "spread_heavy_image_lidar_fusion_runs_evenly_across_gpus",
        "parallelism_advice": (
            "reduce_parallel_runs_or_profile_first"
            if parallel_runs > max(1, cpu_count // max(max_workers, 1)) or loader_wait_risk or cache_tail_risk
            else "start_with_even_gpu_distribution"
        ),
        "profile_command": "conda run -n kd_mm_beam python scripts/profile_training_io.py --config <config> --samples 32",
        "profile_used": isinstance(profile, dict),
        "profile_io_risk": dict(io_risk) if isinstance(io_risk, dict) else {},
    }


def _profile_driven_notes(profile: dict[str, Any] | None) -> list[str]:
    if isinstance(profile, dict):
        return ["Profile IO-risk fields were supplied; worker and locality advice reflects observed loader/cache timing."]
    return ["No Multimodal-NF profile was supplied; run scripts/profile_training_io.py for cache read P95 and loader-wait driven tuning."]


def _skipped_cache_status(modalities: list[str]) -> dict[str, Any]:
    if "lidar" not in modalities:
        return {"status": "not_applicable", "enabled": False, "coverage": 1.0}
    return {"status": "skipped", "enabled": True, "coverage": 0.0}


__all__ = [
    "lidar_cache_coverage",
    "multimodal_nf_derived_cache_recommendation",
    "recommend_parallel_training",
]
