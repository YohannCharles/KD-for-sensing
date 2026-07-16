import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from kd_sensing.engine.data_factory_scalers import (
    fit_gps_scaler,
    gps_scaler_kwargs,
)
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.registries import DATASETS, import_default_components


RETAINED_MODALITIES = frozenset(("image", "radar", "gps", "lidar"))


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
        return all(isinstance(domain, dict) and bool(domain.get("val_csv_name")) for domain in domains)
    return bool(dataset_cfg.get("val_csv_name"))


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any) -> Any:
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    if str(dataset_cfg.get("type", "mmw")).strip().lower() != "mmw":
        raise ValueError("Only data.dataset.type='mmw' is retained.")
    enabled_modalities = resolve_enabled_modalities(cfg)
    retired = sorted(set(enabled_modalities) - RETAINED_MODALITIES)
    if retired:
        raise ValueError(f"The retained MMW surface supports only image/radar/gps/lidar, got {retired}.")
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
    loader_cfg = cfg["data"]["dataloader"]
    provided = dict(normalization_overrides or {})
    train_dataset = build_split_dataset(cfg, "train", normalization_overrides=provided or None)
    fit_gps_scaler(
        train_dataset,
        source="provided_train_artifact" if provided else "train_split_streaming_fit",
        provided=provided.get("gps_scaler"),
    )
    artifacts = gps_scaler_kwargs(train_dataset)
    validation_dataset = build_split_dataset(cfg, "validation", **artifacts) if has_validation_csv(cfg) else None
    test_dataset = build_split_dataset(cfg, "test", **artifacts)
    dataloaders = {
        "train": build_dataloader(
            train_dataset,
            loader_cfg,
            split="train",
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
            domain_balanced_sampling_cfg=cfg.get("data", {}).get("domain_balanced_sampling"),
        ),
        "test": build_dataloader(test_dataset, loader_cfg, split="test", experiment_seed=cfg.get("experiment", {}).get("seed", 0)),
    }
    if validation_dataset is not None:
        dataloaders["validation"] = build_dataloader(
            validation_dataset,
            loader_cfg,
            split="validation",
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
        )
    return dataloaders


def build_split_dataset(
    cfg: dict[str, Any],
    split: str,
    *,
    normalization_overrides: dict[str, Any] | None = None,
    **extra_dataset_kwargs: Any,
) -> Any:
    kwargs = dict(extra_dataset_kwargs)
    kwargs.update(normalization_overrides or {})
    domains = cfg.get("data", {}).get("dataset", {}).get("domains")
    return _build_mmw_domain_dataset(cfg, split, **kwargs) if domains is not None else build_dataset(cfg, split, **kwargs)


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
    dataloader = DataLoader(dataset, **kwargs)
    if metadata is not None:
        dataloader.generator_metadata = metadata
    return dataloader


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


def _build_mmw_domain_dataset(cfg: dict[str, Any], split: str, **dataset_kwargs: Any) -> ConcatDataset:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    domains = dataset_cfg.get("domains")
    if str(dataset_cfg.get("type", "")).strip().lower() != "mmw" or not isinstance(domains, list) or not domains:
        raise ValueError("data.dataset.domains must be a non-empty MMW domain list.")
    seen: set[str] = set()
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise ValueError(f"MMW domain at index {index} must be a mapping.")
        missing = [key for key in ("id", "condition", "scene", "data_root") if not domain.get(key)]
        if missing:
            raise ValueError(f"MMW domain at index {index} is missing required fields: {', '.join(missing)}.")
        domain_id = str(domain["id"])
        if domain_id in seen:
            raise ValueError(f"Duplicate MMW domain id: {domain_id}.")
        seen.add(domain_id)
    datasets = []
    inventory = []
    for domain in domains:
        assert isinstance(domain, dict)
        domain_id = str(domain["id"])
        csv_key, csv_name = _domain_csv_for_split(domain, split)
        csv_path = Path(str(csv_name))
        if not csv_path.is_absolute():
            csv_path = Path(str(domain["data_root"])) / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"MMW domain {domain_id} {split} CSV is missing: {csv_path}.")
        leaf_cfg = deepcopy(cfg)
        leaf_dataset_cfg = leaf_cfg["data"]["dataset"]
        leaf_dataset_cfg.pop("domains", None)
        leaf_dataset_cfg.update(
            condition=str(domain["condition"]),
            scene=str(domain["scene"]),
            data_root=str(domain["data_root"]),
            **{csv_key: str(csv_name)},
        )
        dataset = build_dataset(leaf_cfg, split, **dataset_kwargs)
        dataset.domain_id = domain_id
        dataset.domain_condition = str(domain["condition"])
        dataset.domain_scene = str(domain["scene"])
        dataset.domain_split_path = str(csv_path)
        datasets.append(dataset)
        inventory.append(
            {
                "id": domain_id,
                "condition": str(domain["condition"]),
                "scene": str(domain["scene"]),
                "data_root": str(domain["data_root"]),
                "split": split,
                "split_path": str(csv_path),
                "sample_count": int(len(dataset)),
            }
        )
    pooled = ConcatDataset(datasets)
    pooled.domain_inventory = inventory
    return pooled


def _domain_csv_for_split(domain: dict[str, Any], split: str) -> tuple[str, str]:
    key = {"train": "train_csv_name", "validation": "val_csv_name", "test": "test_csv_name"}.get(split)
    if key is None:
        raise ValueError(f"Unsupported MMW split: {split}.")
    if domain.get(key):
        return key, str(domain[key])
    raise ValueError(f"MMW domain {domain.get('id', '<unknown>')} is missing {split} CSV field ({key}).")


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
    "has_validation_csv",
    "resolve_dataloader_split_config",
    "restore_dataloaders_random_state",
    "shutdown_dataloader_workers",
]
