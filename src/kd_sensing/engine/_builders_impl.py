from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from kd_sensing.data.split_metadata import split_metadata_summary_for_csv
from kd_sensing.data.transform_ops.normalization import LidarBEVNormalizer, MmWaveStandardScaler, load_gps_scaler
from kd_sensing.data.scenes import normalize_deepsense_dataset_config
from kd_sensing.engine.runtime import amp_runtime_metadata, transfer_non_blocking
from kd_sensing.modalities import MODALITY_ORDER, dataset_flags_for_modalities, normalize_modalities
from kd_sensing.registries import DATASETS, DISTILLERS, LOSSES, METRICS, MODELS, import_default_components
from kd_sensing.utils.paths import resolve_path


VALID_MODALITIES = MODALITY_ORDER
CACHE_POLICIES = ("off", "read_only", "auto", "rebuild")


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any):
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    normalize_deepsense_dataset_config(dataset_cfg)
    dataset_type = dataset_cfg.get("type")
    dataset_cfg["split"] = split
    enabled_modalities = resolve_enabled_modalities(cfg)
    dataset_cfg["enabled_modalities"] = list(enabled_modalities)
    dataset_cfg.update(dataset_flags_for_modalities(enabled_modalities))
    apply_cache_policy(dataset_cfg, cfg, enabled_modalities)
    if dataset_type not in {"synthetic", "synthetic_sequence"}:
        csv_key = "train_csv_name" if split == "train" else "test_csv_name"
        dataset_cfg["csv_name"] = dataset_cfg.get(csv_key)
    dataset_cfg.update(extra_dataset_kwargs)
    return DATASETS.build(dataset_cfg)


def apply_cache_policy(
    dataset_cfg: dict[str, Any],
    cfg: dict[str, Any],
    enabled_modalities: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Resolve high-level cache policy into concrete DeepSense6G dataset knobs."""

    selected = set(enabled_modalities)
    cache_cfg = cfg.get("data", {}).get("cache", {})
    global_policy = _normalize_cache_policy(cache_cfg.get("policy", "auto"), "data.cache.policy")
    dataset_cfg["_cache_policy"] = global_policy
    dataset_cfg["_cache_enabled_modalities"] = list(enabled_modalities)

    if "image" in selected:
        image_policy = _normalize_cache_policy(
            cache_cfg.get("image", {}).get("policy", global_policy) or global_policy,
            "data.cache.image.policy",
        )
        _apply_modality_cache_policy(
            dataset_cfg,
            policy=image_policy,
            use_key="image_motion_use_cache",
            write_key="image_motion_write_cache",
            policy_key="image_motion_cache_policy",
        )
    else:
        dataset_cfg["image_motion_use_cache"] = False
        dataset_cfg["image_motion_write_cache"] = False
        dataset_cfg["image_motion_cache_policy"] = "off"

    if "lidar" in selected:
        lidar_policy = _normalize_cache_policy(
            cache_cfg.get("lidar", {}).get("policy", global_policy) or global_policy,
            "data.cache.lidar.policy",
        )
        _apply_modality_cache_policy(
            dataset_cfg,
            policy=lidar_policy,
            use_key="lidar_use_cache",
            write_key="lidar_write_cache",
            policy_key="lidar_cache_policy",
        )
    else:
        dataset_cfg["lidar_use_cache"] = False
        dataset_cfg["lidar_write_cache"] = False
        dataset_cfg["lidar_cache_policy"] = "off"
    return dataset_cfg


def _apply_modality_cache_policy(
    dataset_cfg: dict[str, Any],
    *,
    policy: str,
    use_key: str,
    write_key: str,
    policy_key: str,
) -> None:
    use_cache, write_cache = _cache_policy_flags(policy)
    if dataset_cfg.get(use_key) is None:
        dataset_cfg[use_key] = use_cache
    if dataset_cfg.get(write_key) is None:
        dataset_cfg[write_key] = write_cache
    dataset_cfg[policy_key] = policy


def _cache_policy_flags(policy: str) -> tuple[bool, bool]:
    if policy == "off":
        return False, False
    if policy == "read_only":
        return True, False
    if policy == "auto":
        return True, True
    if policy == "rebuild":
        return False, True
    raise ValueError(f"Unsupported cache policy '{policy}'.")


def _normalize_cache_policy(raw_policy: Any, key: str) -> str:
    policy = str(raw_policy).lower()
    if policy not in CACHE_POLICIES:
        raise ValueError(f"{key} must be one of {', '.join(CACHE_POLICIES)}; got '{raw_policy}'.")
    return policy


def build_dataloaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    loader_cfg = cfg["data"]["dataloader"]
    train_dataset = build_dataset(cfg, "train")
    prepare_lidar_normalizer(cfg, train_dataset)
    dataset_kwargs = {}
    if getattr(train_dataset, "use_gps", False):
        dataset_kwargs["gps_scaler"] = getattr(train_dataset, "gps_scaler", None)
    if getattr(train_dataset, "use_lidar", False):
        dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
    if getattr(train_dataset, "use_mmwave", False):
        dataset_kwargs["mmwave_scaler"] = getattr(train_dataset, "mmwave_scaler", None)
    test_dataset = build_dataset(cfg, "test", **dataset_kwargs)
    return {
        "train": build_dataloader(train_dataset, loader_cfg, split="train"),
        "test": build_dataloader(test_dataset, loader_cfg, split="test"),
    }


def resolve_enabled_modalities(cfg: dict[str, Any]) -> tuple[str, ...]:
    task = cfg.get("experiment", {}).get("task", "image")
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if task == "fusion":
        selected = _resolve_fusion_modalities(cfg)
        _validate_dataset_modality_flags(dataset_cfg, selected)
        return selected
    if task not in VALID_MODALITIES:
        raise ValueError(f"Unsupported experiment.task '{task}'.")
    selected = (task,)
    _validate_dataset_modality_flags(dataset_cfg, selected)
    return selected


def _resolve_fusion_modalities(cfg: dict[str, Any]) -> tuple[str, ...]:
    model_cfg = cfg.get("model", {})
    top_level_modalities = model_cfg.get("modalities")
    if top_level_modalities:
        return normalize_modalities(top_level_modalities, context="model.modalities")
    role_modalities = []
    for role in ("teacher", "student"):
        modalities = model_cfg.get(role, {}).get("modalities")
        if modalities:
            role_modalities.append((role, normalize_modalities(modalities, context=f"model.{role}.modalities")))
    if not role_modalities:
        return ("image", "radar")
    first_role, selected = role_modalities[0]
    for role, modalities in role_modalities[1:]:
        if modalities != selected:
            raise ValueError(
                "Fusion teacher/student modalities must match unless an explicit cross-modal "
                f"distillation mode is implemented; {first_role}={list(selected)}, {role}={list(modalities)}."
            )
    return selected


def _validate_dataset_modality_flags(dataset_cfg: dict[str, Any], selected: tuple[str, ...]) -> None:
    for modality, key in (("gps", "use_gps"), ("lidar", "use_lidar"), ("mmwave", "use_mmwave")):
        if dataset_cfg.get(key, False) and modality not in selected:
            raise ValueError(
                f"data.dataset.{key}=true conflicts with enabled modalities {list(selected)}. "
                f"Add '{modality}' to the task/modalities or disable {key}."
            )


def build_dataloader(dataset: Any, loader_cfg: dict[str, Any], *, split: str) -> DataLoader:
    return DataLoader(dataset, **build_dataloader_kwargs(loader_cfg, split=split))


def build_dataloader_kwargs(loader_cfg: dict[str, Any], *, split: str) -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported DataLoader split '{split}'.")
    num_workers = int(loader_cfg.get("num_workers", 0))
    batch_size_key = "train_batch_size" if split == "train" else "test_batch_size"
    drop_last_key = "train_drop_last" if split == "train" else "test_drop_last"
    kwargs: dict[str, Any] = {
        "batch_size": loader_cfg.get(batch_size_key, 3),
        "shuffle": split == "train",
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", False)),
        "drop_last": bool(loader_cfg.get(drop_last_key, loader_cfg.get("drop_last", False if split == "train" else False))),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", False))
        prefetch_factor = loader_cfg.get("prefetch_factor")
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def dataset_run_metadata(dataset: Any) -> dict[str, Any]:
    csv_path = getattr(dataset, "root_csv", None)
    csv_name = Path(csv_path).name if csv_path is not None else None
    split_family = _split_family(csv_name)
    metadata = {
        "split": getattr(dataset, "split", None),
        "scene_id": getattr(dataset, "scene_id", None),
        "scene_slug": getattr(dataset, "scene_slug", None),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "csv_name": csv_name,
        "num_samples": len(dataset),
        "enabled_modalities": list(getattr(dataset, "enabled_modalities", [])),
        "split_family": split_family,
    }
    if csv_path is not None:
        split_metadata = split_metadata_summary_for_csv(
            csv_path,
            split=getattr(dataset, "split", None),
            require_balanced=split_family == "unified_gps_lidar",
            warn=split_family == "unified_gps_lidar",
        )
        metadata["split_metadata"] = split_metadata
        if split_metadata.get("available"):
            metadata["split_protocol"] = split_metadata.get("split_protocol")
            metadata["split_seed"] = split_metadata.get("split_seed")
            metadata["split_metadata_path"] = split_metadata.get("path")
            metadata["split_sequence_count"] = split_metadata.get("split_sequence_count")
            metadata["split_num_samples"] = split_metadata.get("split_num_samples")
    lidar_cache_dir = getattr(dataset, "lidar_cache_dir", None)
    if lidar_cache_dir is not None:
        metadata["lidar_cache_dir"] = str(lidar_cache_dir)
        metadata["lidar_use_cache"] = bool(getattr(dataset, "lidar_use_cache", False))
        metadata["lidar_write_cache"] = bool(getattr(dataset, "lidar_write_cache", False))
        metadata["lidar_cache_policy"] = getattr(dataset, "lidar_cache_policy", None)
    if getattr(dataset, "use_mmwave", False):
        metadata["mmwave_normalize"] = bool(getattr(dataset, "mmwave_normalize", False))
    image_motion_cache_dir = getattr(dataset, "image_motion_cache_dir", None)
    if image_motion_cache_dir is not None:
        metadata["image_motion_cache_dir"] = str(image_motion_cache_dir)
        metadata["image_motion_use_cache"] = bool(getattr(dataset, "image_motion_use_cache", False))
        metadata["image_motion_write_cache"] = bool(getattr(dataset, "image_motion_write_cache", False))
        metadata["image_motion_cache_policy"] = getattr(dataset, "image_motion_cache_policy", None)
    if hasattr(dataset, "beam_label_cache_mode"):
        metadata["beam_label_cache"] = {
            "mode": getattr(dataset, "beam_label_cache_mode", None),
            "items": len(getattr(dataset, "_beam_label_cache", {})),
        }
    sample_metadata = getattr(getattr(dataset, "samples", None), "metadata", None)
    if sample_metadata is not None:
        metadata["sampling"] = sample_metadata
    return metadata


def dataloaders_run_metadata(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    return {split: dataset_run_metadata(loader.dataset) for split, loader in dataloaders.items()}


def throughput_run_metadata(
    cfg: dict[str, Any],
    dataloaders: dict[str, DataLoader] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    train_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="train")
    test_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="test")
    metadata: dict[str, Any] = {
        "dataloader": {
            "train": _serializable_loader_kwargs(train_loader_kwargs),
            "test": _serializable_loader_kwargs(test_loader_kwargs),
        },
        "transfer": {
            "non_blocking": transfer_non_blocking(cfg),
        },
        "cache": cache_run_metadata(cfg, dataloaders),
    }
    if device is not None:
        metadata["amp"] = amp_runtime_metadata(cfg, device)
    else:
        amp_cfg = cfg.get("training", {}).get("amp", {})
        metadata["amp"] = {
            "enabled": bool(amp_cfg.get("enabled", False)),
            "dtype": amp_cfg.get("dtype", "float16"),
            "grad_scaler": bool(amp_cfg.get("grad_scaler", True)),
        }
    if dataloaders is not None:
        metadata["splits"] = dataloaders_run_metadata(dataloaders)
    return metadata


def cache_run_metadata(cfg: dict[str, Any], dataloaders: dict[str, DataLoader] | None = None) -> dict[str, Any]:
    cache_cfg = cfg.get("data", {}).get("cache", {})
    global_policy = str(cache_cfg.get("policy", "auto"))
    try:
        enabled_modalities = list(resolve_enabled_modalities(cfg))
    except Exception:
        enabled_modalities = []
    metadata: dict[str, Any] = {
        "policy": global_policy,
        "enabled_modalities": enabled_modalities,
        "image": {
            "policy": str(cache_cfg.get("image", {}).get("policy") or global_policy),
        },
        "lidar": {
            "policy": str(cache_cfg.get("lidar", {}).get("policy") or global_policy),
        },
    }
    if dataloaders is not None:
        splits = dataloaders_run_metadata(dataloaders)
        metadata["splits"] = {
            split: {
                key: value
                for key, value in split_metadata.items()
                if key
                in {
                    "enabled_modalities",
                    "image_motion_cache_dir",
                    "image_motion_use_cache",
                    "image_motion_write_cache",
                    "image_motion_cache_policy",
                    "lidar_cache_dir",
                    "lidar_use_cache",
                    "lidar_write_cache",
                    "lidar_cache_policy",
                }
            }
            for split, split_metadata in splits.items()
        }
    return metadata


def _serializable_loader_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key
        in {
            "batch_size",
            "shuffle",
            "num_workers",
            "pin_memory",
            "drop_last",
            "persistent_workers",
            "prefetch_factor",
        }
    }


def save_normalization_artifacts(dataloaders: dict[str, DataLoader], run_dir: str | Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    train_dataset = dataloaders.get("train").dataset if dataloaders.get("train") is not None else None
    if train_dataset is None:
        return artifacts
    artifact_dir = Path(run_dir) / "artifacts"

    gps_scaler = getattr(train_dataset, "gps_scaler", None)
    if gps_scaler is not None:
        gps_path = artifact_dir / "gps_scaler.npz"
        gps_scaler.save(gps_path)
        artifacts["gps_scaler"] = str(gps_path)

    lidar_normalizer = getattr(train_dataset, "lidar_normalizer", None)
    if lidar_normalizer is not None and getattr(train_dataset, "lidar_normalize", False):
        lidar_path = artifact_dir / "lidar_normalizer.npz"
        lidar_normalizer.save(lidar_path)
        artifacts["lidar_normalizer"] = str(lidar_path)

    mmwave_scaler = getattr(train_dataset, "mmwave_scaler", None)
    if mmwave_scaler is not None and getattr(train_dataset, "mmwave_normalize", False):
        mmwave_path = artifact_dir / "mmwave_scaler.npz"
        mmwave_scaler.save(mmwave_path)
        artifacts["mmwave_scaler"] = str(mmwave_path)
    return artifacts


def load_normalization_artifacts(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    artifacts = metadata.get("normalization_artifacts") or {}
    dataset_kwargs: dict[str, Any] = {}
    gps_path = artifacts.get("gps_scaler")
    if gps_path:
        path = Path(gps_path)
        if not path.exists():
            raise FileNotFoundError(f"GPS scaler artifact not found: {path}")
        dataset_kwargs["gps_scaler"] = load_gps_scaler(path)
    lidar_path = artifacts.get("lidar_normalizer")
    if lidar_path:
        path = Path(lidar_path)
        if not path.exists():
            raise FileNotFoundError(f"LiDAR normalizer artifact not found: {path}")
        dataset_kwargs["lidar_normalizer"] = LidarBEVNormalizer.load(path)
    mmwave_path = artifacts.get("mmwave_scaler")
    if mmwave_path:
        path = Path(mmwave_path)
        if not path.exists():
            raise FileNotFoundError(f"mmWave scaler artifact not found: {path}")
        dataset_kwargs["mmwave_scaler"] = MmWaveStandardScaler.load(path)
    return dataset_kwargs


def _split_family(csv_name: str | None) -> str | None:
    if csv_name in {"train_seqs_RA_GPS_LIDAR.csv", "test_seqs_RA_GPS_LIDAR.csv"}:
        return "unified_gps_lidar"
    if csv_name is None:
        return None
    return "configured"


def prepare_lidar_normalizer(cfg: dict[str, Any], dataset: Any) -> None:
    if not getattr(dataset, "needs_lidar_streaming_stats", False):
        return
    progress_enabled = cfg.get("output", {}).get("progress", {}).get("enabled", True)
    dataset.fit_lidar_normalizer_streaming(progress_enabled=progress_enabled)


def _config_uses_gps(cfg: dict[str, Any]) -> bool:
    return "gps" in resolve_enabled_modalities(cfg)


def _config_uses_lidar(cfg: dict[str, Any]) -> bool:
    return "lidar" in resolve_enabled_modalities(cfg)


def _config_uses_mmwave(cfg: dict[str, Any]) -> bool:
    return "mmwave" in resolve_enabled_modalities(cfg)


def build_model(model_cfg: dict[str, Any]):
    import_default_components()
    return MODELS.build(model_cfg)


def build_task_criterion(cfg: dict[str, Any]):
    import_default_components()
    loss_cfg = deepcopy(cfg["loss"])
    for auxiliary_key in (
        "beam_soft",
        "unimodal_aux",
        "gate",
        "gate_weight",
        "gate_ramp_epochs",
        "uni_weight_warmup",
        "uni_weight_after_warmup",
        "prior_regularization",
        "marf",
    ):
        loss_cfg.pop(auxiliary_key, None)
    if loss_cfg.get("type") == "cross_entropy":
        loss_cfg.pop("alpha", None)
        loss_cfg.pop("gamma", None)
    return LOSSES.build(loss_cfg)


def build_distiller(cfg: dict[str, Any], task_criterion):
    import_default_components()
    model_cfg = cfg.get("model", {})
    modalities = (
        model_cfg.get("modalities")
        or model_cfg.get("student", {}).get("modalities")
        or model_cfg.get("teacher", {}).get("modalities")
    )
    return DISTILLERS.build(
        cfg["distillation"],
        task_criterion=task_criterion,
        num_pred=model_cfg.get("num_pred", 3),
        num_classes=model_cfg.get("num_classes", 64),
        feature_size=model_cfg.get("feature_size", 64),
        modalities=modalities,
    )


def build_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    import_default_components()
    eval_cfg = cfg.get("evaluation", {})
    return {
        "topk": METRICS.build(
            {
                "type": "topk_accuracy",
                "k_values": eval_cfg.get("k_values", [1, 2, 3, 5, 10]),
            }
        ),
        "dba": METRICS.build(
            {
                "type": "dba",
                "delta": eval_cfg.get("dba_delta", 5),
            }
        ),
    }


def build_optimizer(cfg: dict[str, Any], model) -> torch.optim.Optimizer:
    training_cfg = cfg["training"]
    param_group_cfg = cfg.get("finetune", {}).get("param_groups", {})
    if param_group_cfg.get("enabled", False):
        groups = _teacher_prior_param_groups(cfg, model, param_group_cfg)
        if not groups:
            raise ValueError("No trainable parameters found for Stage 3 optimizer parameter groups.")
        return torch.optim.Adam(groups, weight_decay=training_cfg.get("weight_decay", 0.0))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    return torch.optim.Adam(
        [{"params": trainable_params, "name": "main", "param_count": _param_count(trainable_params)}],
        lr=training_cfg.get("lr", 7.5e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )


def optimizer_param_group_summary(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    summary = []
    for index, group in enumerate(optimizer.param_groups):
        params = list(group.get("params", []))
        summary.append(
            {
                "index": index,
                "name": str(group.get("name", f"group_{index}")),
                "lr": float(group.get("lr", 0.0)),
                "param_count": int(group.get("param_count", _param_count(params))),
            }
        )
    return summary


def _teacher_prior_param_groups(cfg: dict[str, Any], model, param_group_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    training_lr = float(cfg.get("training", {}).get("lr", 7.5e-4))
    strong_modalities = set(str(name) for name in param_group_cfg.get("strong_modalities", ["gps", "mmwave"]))
    groups: dict[str, list[torch.nn.Parameter]] = {
        "fusion": [],
        "head": [],
        "gate": [],
        "strong_encoder": [],
        "weak_encoder": [],
    }
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        role = _parameter_role(name, strong_modalities)
        groups[role].append(param)
        seen.add(id(param))
    param_groups = []
    lr_by_role = {
        "fusion": float(param_group_cfg.get("fusion_lr", training_lr)),
        "head": float(param_group_cfg.get("head_lr", training_lr)),
        "gate": float(param_group_cfg.get("gate_lr", training_lr)),
        "strong_encoder": float(param_group_cfg.get("strong_encoder_lr", training_lr * 0.2)),
        "weak_encoder": float(param_group_cfg.get("weak_encoder_lr", training_lr * 0.05)),
    }
    for role, params in groups.items():
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "name": role,
                "lr": lr_by_role[role],
                "param_count": _param_count(params),
            }
        )
    return param_groups


def _parameter_role(name: str, strong_modalities: set[str]) -> str:
    if name.startswith("encoders."):
        parts = name.split(".")
        modality = parts[1] if len(parts) > 1 else ""
        return "strong_encoder" if modality in strong_modalities else "weak_encoder"
    if name.startswith(("reliability_estimator", "router")):
        return "gate"
    if name.startswith(("prediction_head", "unimodal_head")):
        return "head"
    return "fusion"


def _param_count(params: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> int:
    return int(sum(param.numel() for param in params))


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_cfg = cfg.get("scheduler", {})
    if scheduler_cfg.get("type", "cosine_warm_restarts") == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_cfg.get("T_0", 10),
        T_mult=scheduler_cfg.get("T_mult", 2),
        eta_min=scheduler_cfg.get("eta_min", 1e-6),
    )


def build_device(cfg: dict[str, Any]) -> torch.device:
    requested = cfg.get("experiment", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_weight_path(cfg: dict[str, Any], weight_name: str | None) -> Path | None:
    if not weight_name:
        return None
    candidate = Path(weight_name).expanduser()
    if candidate.is_absolute():
        return candidate
    weights_dir = cfg.get("paths", {}).get("weights_dir", "All_models")
    return resolve_path(Path(weights_dir) / candidate)
