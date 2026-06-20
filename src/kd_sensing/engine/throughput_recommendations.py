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
    mmw_image_heavy = _is_mmw_image_heavy(cfg, modalities)
    if mmw_image_heavy:
        worker_budget = _mmw_worker_budget(
            cfg,
            parallel_runs=parallel_runs,
            cpu_count=cpu_count,
            worker_budget=worker_budget,
            profile=profile,
        )
        prefetch_factor = 1
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
    if cache_policy is not None:
        overrides.append(f"data.cache.policy={cache_policy}")
    if mmw_image_heavy:
        recommended_batch = _mmw_recommended_batch_size(cfg, parallel_runs=parallel_runs, profile=profile)
        current_batch = int(cfg.get("data", {}).get("dataloader", {}).get("batch_size", recommended_batch) or recommended_batch)
        if recommended_batch < current_batch:
            overrides.append(f"data.dataloader.batch_size={recommended_batch}")
        overrides.extend(
            [
                "data.cache.image.policy=auto",
                "data.dataloader.train_persistent_workers=false",
            ]
        )
    optional_overrides = [
        "training.amp.enabled=true",
        "training.amp.dtype=float16",
    ]
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
            "mmw_image_heavy": _mmw_image_heavy_recommendations(
                cfg,
                modalities=modalities,
                parallel_runs=parallel_runs,
                worker_budget=worker_budget,
                profile=profile,
            ),
        },
        "cache": {
            "lidar": lidar_cache,
            "image": (
                {
                    "status": "image_heavy_policy_recommended",
                    "recommended_policy": "auto",
                    "prewarm_command": _image_derived_cache_prewarm_command(),
                }
                if mmw_image_heavy
                else {"status": "not_applicable", "enabled": "image" in modalities}
            ),
            "min_coverage_for_read_only": float(cache_min_coverage),
            "prewarm_command": (
                _image_derived_cache_prewarm_command() if mmw_image_heavy else
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
            *(_mmw_image_heavy_notes(profile) if mmw_image_heavy else []),
        ],
    }


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


def _is_mmw_image_heavy(cfg: dict[str, Any], modalities: list[str]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset_type = str(dataset_cfg.get("type", "")).strip().lower()
    seq_len = int(dataset_cfg.get("seq_len", cfg.get("model", {}).get("seq_length", 0)) or 0)
    return dataset_type == "mmw" and "image" in modalities and seq_len >= 8


def _mmw_worker_budget(
    cfg: dict[str, Any],
    *,
    parallel_runs: int,
    cpu_count: int,
    worker_budget: dict[str, Any],
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    risk = _profile_memory_or_loader_risk(profile)
    cap = 1 if parallel_runs >= 4 or risk else 2
    train_workers = max(0, min(int(worker_budget.get("train_num_workers", 1)), cap))
    if cpu_count <= parallel_runs * 2:
        train_workers = min(train_workers, 1)
    return {
        **worker_budget,
        "train_num_workers": int(train_workers),
        "test_num_workers": 0,
        "train_persistent_workers": False,
        "test_persistent_workers": False,
    }


def _mmw_recommended_batch_size(
    cfg: dict[str, Any],
    *,
    parallel_runs: int,
    profile: dict[str, Any] | None,
) -> int:
    loader_cfg = cfg.get("data", {}).get("dataloader", {}) if isinstance(cfg.get("data"), dict) else {}
    current = int(loader_cfg.get("batch_size", loader_cfg.get("train_batch_size", 4)) or 4)
    if parallel_runs >= 4 or _profile_memory_or_loader_risk(profile):
        return max(1, min(current, 2))
    return max(1, min(current, 4))


def _mmw_image_heavy_recommendations(
    cfg: dict[str, Any],
    *,
    modalities: list[str],
    parallel_runs: int,
    worker_budget: dict[str, Any],
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _is_mmw_image_heavy(cfg, modalities):
        return {"enabled": False}
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    current_batch = int(loader_cfg.get("batch_size", loader_cfg.get("train_batch_size", 4)) or 4)
    recommended_batch = _mmw_recommended_batch_size(cfg, parallel_runs=parallel_runs, profile=profile)
    recommended_parallel_runs = max(1, min(int(parallel_runs), 2 if _profile_memory_or_loader_risk(profile) else 3))
    return {
        "enabled": True,
        "risk": "mmw_image_heavy_worker_rss",
        "seq_len": int(dataset_cfg.get("seq_len", 0) or 0),
        "batch_size": current_batch,
        "recommended_batch_size": recommended_batch,
        "parallel_runs": int(parallel_runs),
        "recommended_parallel_runs": recommended_parallel_runs,
        "train_num_workers_upper_bound": int(worker_budget.get("train_num_workers", 0)),
        "prefetch_factor": 1,
        "persistent_workers": False,
        "image_cache_policy": "auto",
        "amp_limit": "AMP does not reduce PNG decode, resize, or DataLoader worker RSS.",
        "actions": [
            "limit_parallel_runs",
            "reduce_batch_size",
            "cap_train_num_workers",
            "disable_persistent_workers",
            "enable_image_derived_cache",
        ],
        "profile_used": isinstance(profile, dict),
        "profile_io_risk": profile.get("io_risk", {}) if isinstance(profile, dict) else {},
    }


def _profile_memory_or_loader_risk(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return False
    io_risk = profile.get("io_risk", {}) if isinstance(profile.get("io_risk"), dict) else {}
    text = json_dumps_lower(profile)
    return bool(
        io_risk.get("loader_wait_dominates_step")
        or io_risk.get("worker_memory_risk")
        or "exit code 137" in text
        or "killed" in text
        or "oom" in text
    )


def json_dumps_lower(payload: dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(payload, sort_keys=True).lower()
    except Exception:
        return str(payload).lower()


def _mmw_image_heavy_notes(profile: dict[str, Any] | None) -> list[str]:
    notes = [
        "MMW image-heavy runs are usually limited by PNG decode/resize and DataLoader worker RSS before AMP helps.",
        "Prefer reducing parallel runs, batch size, train workers, and persistent workers before increasing workers.",
        "Use data.cache.image.policy=auto or prewarm image-derived cache before long LOSO runs.",
    ]
    if _profile_memory_or_loader_risk(profile):
        notes.append("Supplied profile/log risk indicates loader wait, OOM, or killed process; keep recommendations conservative.")
    return notes


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
    return f"conda run -n kd_mm_beam kd-sensing-train --config {config_path} {override_args}".strip()


def _lidar_prewarm_command() -> str:
    return "conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml"


def _image_derived_cache_prewarm_command() -> str:
    return "conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/mmw_image_derived_cache.yaml"


def _skipped_cache_status(modalities: list[str]) -> dict[str, Any]:
    if "lidar" not in modalities:
        return {"status": "not_applicable", "enabled": False, "coverage": 1.0}
    return {"status": "skipped", "enabled": True, "coverage": 0.0}


__all__ = [
    "lidar_cache_coverage",
    "recommend_parallel_training",
]
