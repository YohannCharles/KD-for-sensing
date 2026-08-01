import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from kd_sensing.data.mmw.protocol import validate_mmw_config_protocol
from kd_sensing.data.mmw.trajectory_protocol import TRAJECTORY_PROTOCOL_MODE
from kd_sensing.engine.data_factory_scalers import (
    fit_gps_scaler,
    gps_scaler_kwargs,
)
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.registries import DATASETS, import_default_components


RETAINED_MODALITIES = frozenset(("image", "radar", "gps", "lidar"))
RETAINED_DATASETS = frozenset(("mmw", "deepsense6g"))


def build_dataloader_kwargs(loader_cfg: dict[str, Any], *, split: str) -> dict[str, Any]:
    settings = resolve_dataloader_split_config(loader_cfg, split=split)
    kwargs: dict[str, Any] = {
        "batch_size": settings["batch_size"],
        "shuffle": settings["shuffle"],
        "num_workers": settings["num_workers"],
        "pin_memory": settings["pin_memory"],
        "drop_last": settings["drop_last"],
    }
    if settings["num_workers"] > 0:
        kwargs["persistent_workers"] = settings["persistent_workers"]
        if settings["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = settings["prefetch_factor"]
    return kwargs


def resolve_dataloader_split_config(loader_cfg: dict[str, Any], *, split: str) -> dict[str, Any]:
    if split not in {"train", "test", "validation"}:
        raise ValueError(f"Unsupported DataLoader split '{split}'.")
    num_workers = int(loader_cfg.get("num_workers", 0))
    prefetch_factor = loader_cfg.get("prefetch_factor")
    persistent_workers = bool(loader_cfg.get("persistent_workers", False))
    return {
        "batch_size": int(loader_cfg[f"{split}_batch_size"]),
        "shuffle": split == "train",
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", False)),
        "drop_last": bool(loader_cfg.get(f"{split}_drop_last", False)),
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
    }


def shutdown_dataloader_workers(dataloader: DataLoader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None and callable(getattr(iterator, "_shutdown_workers", None)):
        iterator._shutdown_workers()
    dataloader._iterator = None  # type: ignore[attr-defined]


def has_validation_csv(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    domains = dataset_cfg.get("domains") if isinstance(dataset_cfg, dict) else None
    if isinstance(domains, list) and domains:
        if cfg.get("data_protocol", {}).get("mode") == TRAJECTORY_PROTOCOL_MODE:
            return any(isinstance(domain, dict) and bool(domain.get("val_csv_name")) for domain in domains)
        return all(isinstance(domain, dict) and bool(domain.get("val_csv_name")) for domain in domains)
    return bool(dataset_cfg.get("val_csv_name"))


def final_test_enabled(cfg: dict[str, Any]) -> bool:
    final_test = cfg.get("training", {}).get("final_test")
    dataset_type = str(cfg.get("data", {}).get("dataset", {}).get("type", "")).strip().lower()
    if final_test is None:
        enabled = dataset_type != "mmw"
    elif isinstance(final_test, bool):
        enabled = final_test
    elif not isinstance(final_test, dict):
        raise ValueError("training.final_test must be a mapping or boolean.")
    else:
        enabled = bool(final_test.get("enabled", True))
    if dataset_type == "mmw":
        requested = bool(cfg.get("runtime", {}).get("evaluate_test_requested", False))
        if enabled != requested:
            raise ValueError("MMW test loading requires the explicit --evaluate-test runtime authorization.")
        return requested
    return enabled


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any) -> Any:
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    dataset_type = str(dataset_cfg.get("type", "")).strip().lower()
    if dataset_type not in RETAINED_DATASETS:
        raise ValueError(f"Supported data.dataset.type values are {sorted(RETAINED_DATASETS)}, got {dataset_type!r}.")
    enabled_modalities = resolve_enabled_modalities(cfg)
    if set(enabled_modalities) != RETAINED_MODALITIES:
        raise ValueError(f"The retained data surface requires image/radar/gps/lidar, got {list(enabled_modalities)}.")
    dataset_cfg["type"] = dataset_type
    if dataset_type == "mmw":
        dataset_cfg.setdefault("data_root", "dataset/MMW/sunny")
    dataset_cfg["split"] = split
    dataset_cfg["enabled_modalities"] = list(enabled_modalities)
    dataset_cfg.update(dataset_flags_for_modalities(enabled_modalities))
    dataset_cfg.update(extra_dataset_kwargs)
    return DATASETS.build(dataset_cfg)


def build_dataloaders(
    cfg: dict[str, Any],
    *,
    normalization_overrides: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    protocol_audit = validate_mmw_config_protocol(cfg)
    loader_cfg = cfg["data"]["dataloader"]
    provided = dict(normalization_overrides or {})
    train_dataset = build_split_dataset(
        cfg,
        "train",
        normalization_overrides=provided or None,
        _mmw_protocol_validated=protocol_audit is not None,
    )
    fit_gps_scaler(
        train_dataset,
        source="provided_train_artifact" if provided else "train_split_streaming_fit",
        provided=provided.get("gps_scaler"),
    )
    artifacts = gps_scaler_kwargs(train_dataset)
    validation_dataset = (
        build_split_dataset(cfg, "validation", _mmw_protocol_validated=protocol_audit is not None, **artifacts)
        if has_validation_csv(cfg)
        else None
    )
    dataset_type = str(cfg.get("data", {}).get("dataset", {}).get("type", "")).strip().lower()
    if dataset_type == "mmw" and validation_dataset is None:
        raise ValueError("MMW trajectory training requires a validation DataLoader.")
    dataloaders = {
        "train": build_dataloader(
            train_dataset,
            loader_cfg,
            split="train",
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
            domain_balanced_sampling_cfg=cfg.get("data", {}).get("domain_balanced_sampling"),
        ),
    }
    if validation_dataset is not None:
        dataloaders["validation"] = build_dataloader(
            validation_dataset,
            loader_cfg,
            split="validation",
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
        )
    if final_test_enabled(cfg):
        test_dataset = build_split_dataset(cfg, "test", _mmw_protocol_validated=protocol_audit is not None, **artifacts)
        dataloaders["test"] = build_dataloader(
            test_dataset,
            loader_cfg,
            split="test",
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
        )
    if protocol_audit is not None:
        identity = _mmw_protocol_identity(protocol_audit, cfg)
        for loader in dataloaders.values():
            loader.data_protocol_identity = dict(identity)
            loader.dataset.data_protocol_identity = dict(identity)
    return dataloaders


def build_split_dataset(
    cfg: dict[str, Any],
    split: str,
    *,
    normalization_overrides: dict[str, Any] | None = None,
    _mmw_protocol_validated: bool = False,
    **extra_dataset_kwargs: Any,
) -> Any:
    dataset_type = str(cfg.get("data", {}).get("dataset", {}).get("type", "")).strip().lower()
    if dataset_type == "mmw" and not _mmw_protocol_validated:
        validate_mmw_config_protocol(cfg)
    kwargs = dict(extra_dataset_kwargs)
    kwargs.update(normalization_overrides or {})
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    domains = dataset_cfg.get("domains")
    return _build_pooled_domain_dataset(cfg, split, **kwargs) if domains is not None else build_dataset(cfg, split, **kwargs)


def build_dataloader(
    dataset: Any,
    loader_cfg: dict[str, Any],
    *,
    split: str,
    experiment_seed: int | None = None,
    domain_balanced_sampling_cfg: dict[str, Any] | None = None,
) -> DataLoader:
    kwargs = build_dataloader_kwargs(loader_cfg, split=split)
    if experiment_seed is not None:
        metadata = dataloader_generator_metadata(dataset, split=split, base_seed=experiment_seed)
        generator = torch.Generator().manual_seed(int(metadata["derived_seed"]))
        kwargs.update(generator=generator, worker_init_fn=_seed_dataloader_worker)
    else:
        metadata = None
    if split == "train":
        domain_sampler = build_domain_balanced_sampler(dataset, domain_balanced_sampling_cfg, experiment_seed=experiment_seed)
        if domain_sampler is not None:
            kwargs.update(shuffle=False, sampler=domain_sampler)
        if _dataset_uses_lidar_augmentation(dataset) and int(kwargs.get("num_workers", 0)) > 0:
            # Worker dataset copies need the current epoch to derive reproducible augmentation RNGs.
            kwargs["persistent_workers"] = False
    dataloader = DataLoader(dataset, **kwargs)
    if metadata is not None:
        dataloader.generator_metadata = metadata
    return dataloader


def _dataset_uses_lidar_augmentation(dataset: Any) -> bool:
    if bool(getattr(dataset, "lidar_augment", False)):
        return True
    nested = getattr(dataset, "datasets", None)
    if isinstance(nested, (list, tuple)) and any(_dataset_uses_lidar_augmentation(item) for item in nested):
        return True
    parent = getattr(dataset, "dataset", None)
    return _dataset_uses_lidar_augmentation(parent) if parent is not None else False


def dataloader_generator_metadata(dataset: Any, *, split: str, base_seed: int) -> dict[str, Any]:
    identity = {
        "algorithm": "sha256-v1",
        "base_seed": int(base_seed),
        "split": str(split),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    identity["derived_seed"] = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return identity


def capture_dataloaders_random_state(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    return {split: _capture_dataloader_random_state(loader) for split, loader in dataloaders.items()}


def restore_dataloaders_random_state(dataloaders: dict[str, DataLoader], state: dict[str, Any]) -> None:
    if set(dataloaders) != set(state):
        raise ValueError(f"DataLoader random-state splits do not match: current={sorted(dataloaders)}, checkpoint={sorted(state)}.")
    for split, loader in dataloaders.items():
        _restore_dataloader_random_state(loader, state[split], split=split)


def _capture_dataloader_random_state(dataloader: DataLoader) -> dict[str, Any]:
    generator = getattr(dataloader, "generator", None)
    sampler = getattr(dataloader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    return {
        "schema_version": 1,
        "identity": dict(getattr(dataloader, "generator_metadata", {})),
        "generator_state": generator.get_state().clone() if isinstance(generator, torch.Generator) else None,
        "sampler": {
            "type": type(sampler).__name__ if sampler is not None else None,
            "epoch": int(getattr(sampler, "epoch")) if hasattr(sampler, "epoch") else None,
            "generator_state": sampler_generator.get_state().clone()
            if isinstance(sampler_generator, torch.Generator) and sampler_generator is not generator
            else None,
        },
    }


def _restore_dataloader_random_state(dataloader: DataLoader, state: dict[str, Any], *, split: str) -> None:
    if state.get("identity") != dict(getattr(dataloader, "generator_metadata", {})):
        raise ValueError(f"DataLoader generator identity mismatch for split '{split}'.")
    generator = getattr(dataloader, "generator", None)
    if isinstance(generator, torch.Generator) and state.get("generator_state") is not None:
        generator.set_state(state["generator_state"])
    sampler = getattr(dataloader, "sampler", None)
    sampler_state = state.get("sampler", {})
    if sampler_state.get("type") != (type(sampler).__name__ if sampler is not None else None):
        raise ValueError(f"DataLoader sampler type mismatch for split '{split}'.")
    if sampler is not None and sampler_state.get("epoch") is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(sampler_state["epoch"]))
    sampler_generator = getattr(sampler, "generator", None)
    if isinstance(sampler_generator, torch.Generator) and sampler_generator is not generator and sampler_state.get("generator_state") is not None:
        sampler_generator.set_state(sampler_state["generator_state"])


def _dataset_fingerprint(dataset: Any) -> str:
    if isinstance(dataset, Subset):
        payload: Any = {
            "kind": "subset",
            "parent": _dataset_fingerprint(dataset.dataset),
            "indices": hashlib.sha256(np.asarray(dataset.indices, dtype=np.int64).tobytes()).hexdigest(),
            "length": len(dataset),
        }
    elif isinstance(dataset, ConcatDataset):
        payload = {"kind": "concat", "components": [_dataset_fingerprint(item) for item in dataset.datasets], "length": len(dataset)}
    else:
        payload = {
            "kind": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "length": len(dataset),
            "split": getattr(dataset, "split", None),
            "root_csv": str(getattr(dataset, "root_csv", "")),
            "scene_id": getattr(dataset, "scene_id", None),
            "domain_id": getattr(dataset, "domain_id", None),
            "schema_identity": getattr(dataset, "schema_identity", None),
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _build_pooled_domain_dataset(cfg: dict[str, Any], split: str, **dataset_kwargs: Any) -> ConcatDataset:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    domains = dataset_cfg.get("domains")
    dataset_type = str(dataset_cfg.get("type", "")).strip().lower()
    if dataset_type not in RETAINED_DATASETS or not isinstance(domains, list) or not domains:
        raise ValueError("data.dataset.domains must be a non-empty retained-dataset domain list.")
    seen: set[str] = set()
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise ValueError(f"Pooled domain at index {index} must be a mapping.")
        required = ("id", "scene", "data_root") if dataset_type == "deepsense6g" else ("id", "condition", "scene", "data_root")
        missing = [key for key in required if domain.get(key) in (None, "")]
        if missing:
            raise ValueError(f"Pooled domain at index {index} is missing required fields: {', '.join(missing)}.")
        domain_id = str(domain["id"])
        if domain_id in seen:
            raise ValueError(f"Duplicate pooled domain id: {domain_id}.")
        seen.add(domain_id)
    if cfg.get("data_protocol", {}).get("mode") == TRAJECTORY_PROTOCOL_MODE:
        csv_key = {"train": "train_csv_name", "validation": "val_csv_name", "test": "test_csv_name"}[split]
        domains = [domain for domain in domains if bool(domain.get(csv_key))]
        if not domains:
            raise ValueError(f"Trajectory protocol has no authorized {split} domains.")
    import_default_components()

    def build_domain(domain: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        assert isinstance(domain, dict)
        domain_id = str(domain["id"])
        csv_key, csv_name = _domain_csv_for_split(domain, split)
        csv_path = Path(str(csv_name))
        if not csv_path.is_absolute():
            csv_path = Path(str(domain["data_root"])) / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Pooled domain {domain_id} {split} CSV is missing: {csv_path}.")
        leaf_cfg = deepcopy(cfg)
        leaf_dataset_cfg = leaf_cfg["data"]["dataset"]
        leaf_dataset_cfg.pop("domains", None)
        leaf_dataset_cfg.update(scene=domain["scene"], data_root=str(domain["data_root"]), **{csv_key: str(csv_name)})
        if dataset_type == "mmw":
            leaf_dataset_cfg["condition"] = str(domain["condition"])
        dataset = build_dataset(leaf_cfg, split, **dataset_kwargs)
        dataset.domain_id = domain_id
        dataset.domain_condition = str(domain.get("condition", ""))
        dataset.domain_scene = str(domain["scene"])
        dataset.domain_split_path = str(csv_path)
        inventory = {
            "id": domain_id,
            "condition": str(domain.get("condition", "")),
            "scene": str(domain["scene"]),
            "data_root": str(domain["data_root"]),
            "split": split,
            "split_path": str(csv_path),
            "sample_count": int(len(dataset)),
        }
        return dataset, inventory

    workers = min(len(domains), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{dataset_type}-{split}") as executor:
        built = list(executor.map(build_domain, domains))
    datasets = [dataset for dataset, _ in built]
    inventory = [item for _, item in built]
    pooled = ConcatDataset(datasets)
    pooled.domain_inventory = inventory
    pooled.initialization_workers = workers
    return pooled


def _domain_csv_for_split(domain: dict[str, Any], split: str) -> tuple[str, str]:
    key = {"train": "train_csv_name", "validation": "val_csv_name", "test": "test_csv_name"}.get(split)
    if key is None:
        raise ValueError(f"Unsupported pooled-dataset split: {split}.")
    if domain.get(key):
        return key, str(domain[key])
    raise ValueError(f"Pooled domain {domain.get('id', '<unknown>')} is missing {split} CSV field ({key}).")


def _mmw_protocol_identity(audit: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg["data_protocol"]
    protocol_path = Path(str(section["path"])).resolve()
    report_path = Path(str(section["audit_report"])).resolve()
    return {
        "split_protocol": str(section["protocol_id"]),
        "protocol_version": int(section["protocol_version"]),
        "split_protocol_version": int(section["protocol_version"]),
        "split_seed": int(section["split_seed"]),
        "block_size": int(section["block_size"]),
        "split_manifest": str(section.get("split_manifest", section["path"])),
        "split_manifest_hash": str(section["split_manifest_hash"]),
        "data_source_hash": str(section["data_source_hash"]),
        "window_config_hash": str(section["window_config_hash"]),
        "weather_binding": bool(section["weather_binding"]),
        "protocol_id": str(section["protocol_id"]),
        "protocol_fingerprint": audit["protocol_fingerprint"],
        "protocol_file_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "audit_id": audit["audit_id"],
        "audit_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "train_sample_id_hash": audit["train_sample_id_hash"],
        "validation_sample_id_hash": audit["validation_sample_id_hash"],
        "test_sample_id_hash": audit["test_sample_id_hash"],
        "train_sample_count": int(audit["train_sample_count"]),
        "validation_sample_count": int(audit["validation_sample_count"]),
        "test_sample_count": int(audit["test_sample_count"]),
        "train_block_count": int(audit["block_counts"]["train"]),
        "validation_block_count": int(audit["block_counts"]["validation"]),
        "test_block_count": int(audit["block_counts"]["test"]),
        "train_trajectory_count": int(audit["trajectory_counts"]["train"]),
        "validation_trajectory_count": int(audit["trajectory_counts"]["validation"]),
        "test_trajectory_count": int(audit["trajectory_counts"]["test"]),
        "train_seed": int(cfg.get("experiment", {}).get("train_seed", cfg.get("experiment", {}).get("seed", 0))),
        "test_evaluated": bool(section.get("test_evaluated", False)),
        "outer_test_accessed": bool(section.get("test_evaluated", False)),
    }


def build_domain_balanced_sampler(
    dataset: Any,
    config: dict[str, Any] | None,
    *,
    experiment_seed: int | None = None,
) -> WeightedRandomSampler | None:
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return None
    inventory = getattr(dataset, "domain_inventory", None)
    components = getattr(dataset, "datasets", None)
    if not isinstance(dataset, ConcatDataset) or not isinstance(inventory, list) or not components:
        raise ValueError("domain-balanced sampling requires a pooled dataset built from data.dataset.domains.")
    counts = [len(component) for component in components]
    if any(count <= 0 for count in counts):
        raise ValueError("domain-balanced sampling requires every domain to contain at least one sample.")
    weights = torch.cat([torch.full((count,), 1.0 / count, dtype=torch.double) for count in counts])
    seed = int(config.get("seed", experiment_seed or 0))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=int(config.get("num_samples", len(dataset))),
        replacement=bool(config.get("replacement", True)),
        generator=torch.Generator().manual_seed(seed),
    )
    sampler.domain_balanced_metadata = {
        "sampler": "WeightedRandomSampler",
        "seed": seed,
        "replacement": bool(config.get("replacement", True)),
        "num_samples": int(config.get("num_samples", len(dataset))),
        "domains": [
            {"id": str(item["id"]), "sample_count": count, "sample_weight": 1.0 / count, "total_weight": 1.0}
            for item, count in zip(inventory, counts)
        ],
    }
    return sampler


def _seed_dataloader_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


__all__ = [
    "build_dataloader",
    "build_dataloader_kwargs",
    "build_dataloaders",
    "build_dataset",
    "build_domain_balanced_sampler",
    "build_split_dataset",
    "capture_dataloaders_random_state",
    "dataloader_generator_metadata",
    "final_test_enabled",
    "has_validation_csv",
    "resolve_dataloader_split_config",
    "restore_dataloaders_random_state",
    "shutdown_dataloader_workers",
]
