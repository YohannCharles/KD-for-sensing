import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from kd_sensing.config.lidar_normalization import canonicalize_lidar_dataset_config
from kd_sensing.data.dataset_descriptors import dataset_descriptor, resolve_dataset_profiles
from kd_sensing.engine.cache_policy import apply_cache_policy
from kd_sensing.engine.data_factory_protocols import (
    build_protocol_split_datasets as _build_protocol_split_datasets,
    dataset_scenes_for_split,
    retarget_cfg_for_scene,
)
from kd_sensing.engine.data_factory_scalers import (
    fit_internal_validation_normalizers,
    fit_train_normalizers,
    harmonize_multi_scene_train_normalizers,
    normalization_fit_placeholders,
    normalization_kwargs,
    prepare_lidar_normalizer,
)
from kd_sensing.engine.epoch_subsampling import build_epoch_subsample_sampler
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.modalities import dataset_flags_for_modalities
from kd_sensing.registries import DATASETS, import_default_components


SNAPSHOT_TRAIN_CSV = "train_seqs_SNAPSHOT_NEXT_FRAME.csv"
SNAPSHOT_VAL_CSV = "val_seqs_SNAPSHOT_NEXT_FRAME.csv"


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
    if split not in {"train", "test", "validation", "val"}:
        raise ValueError(f"Unsupported DataLoader split '{split}'.")
    normalized_split = "validation" if split == "val" else split
    fallback_split = "test" if normalized_split == "validation" else normalized_split
    num_workers = int(_split_loader_value(loader_cfg, normalized_split, "num_workers", 0, fallback_split=fallback_split) or 0)
    prefetch_factor = _split_loader_value(loader_cfg, normalized_split, "prefetch_factor", None, fallback_split=fallback_split)
    persistent_workers = bool(
        _split_loader_value(loader_cfg, normalized_split, "persistent_workers", False, fallback_split=fallback_split)
    )
    if num_workers <= 0:
        persistent_workers = False
    return {
        "batch_size": int(_split_loader_value(loader_cfg, normalized_split, "batch_size", 3, fallback_split=fallback_split) or 3),
        "shuffle": normalized_split == "train",
        "num_workers": num_workers,
        "pin_memory": bool(_split_loader_value(loader_cfg, normalized_split, "pin_memory", False, fallback_split=fallback_split)),
        "drop_last": bool(_split_loader_value(loader_cfg, normalized_split, "drop_last", False, fallback_split=fallback_split)),
        "persistent_workers": persistent_workers,
        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
    }


def _split_loader_value(
    loader_cfg: dict[str, Any],
    split: str,
    key: str,
    default: Any,
    *,
    fallback_split: str | None = None,
) -> Any:
    split_cfg = loader_cfg.get(split)
    if isinstance(split_cfg, dict) and key in split_cfg:
        return split_cfg[key]
    prefixed_key = f"{split}_{key}"
    if prefixed_key in loader_cfg:
        return loader_cfg[prefixed_key]
    if fallback_split and fallback_split != split:
        fallback_cfg = loader_cfg.get(fallback_split)
        if isinstance(fallback_cfg, dict) and key in fallback_cfg:
            return fallback_cfg[key]
        fallback_prefixed_key = f"{fallback_split}_{key}"
        if fallback_prefixed_key in loader_cfg:
            return loader_cfg[fallback_prefixed_key]
    return loader_cfg.get(key, default)


def shutdown_dataloader_workers(dataloader: DataLoader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    dataloader._iterator = None  # type: ignore[attr-defined]


def validation_from_train_enabled(cfg: dict[str, Any]) -> bool:
    validation_cfg = cfg.get("data", {}).get("validation_from_train")
    if isinstance(validation_cfg, dict):
        return bool(validation_cfg.get("enabled", False))
    return bool(validation_cfg)


def validation_from_train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = cfg.get("data", {}).get("validation_from_train")
    if isinstance(validation_cfg, dict):
        return validation_cfg
    return {"enabled": bool(validation_cfg)}


def has_validation_csv(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    domains = dataset_cfg.get("domains") if isinstance(dataset_cfg, dict) else None
    if isinstance(domains, list) and domains:
        return all(
            isinstance(domain, dict)
            and bool(domain.get("val_csv_name") or domain.get("validation_csv_name"))
            for domain in domains
        )
    return bool(dataset_cfg.get("val_csv_name"))


def build_train_and_internal_validation_datasets(
    cfg: dict[str, Any],
    *,
    split_dataset_builder: Callable[..., Any],
    provided: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    full_train = split_dataset_builder(
        cfg,
        "train",
        normalization_overrides=provided,
    )
    train_dataset, validation_dataset = split_dataset_for_internal_validation(full_train, cfg)
    if provided:
        fit_train_normalizers(
            train_dataset,
            validation_dataset,
            source="internal_train_subset_streaming_fit",
            provided=provided,
        )
    else:
        fit_internal_validation_normalizers(train_dataset, validation_dataset)
    return train_dataset, validation_dataset


def split_dataset_for_internal_validation(dataset: Any, cfg: dict[str, Any]) -> tuple[Any, Any]:
    if isinstance(dataset, ConcatDataset):
        train_parts = []
        validation_parts = []
        for offset, component in enumerate(dataset.datasets):
            train_subset, validation_subset = split_single_dataset_for_internal_validation(
                component,
                cfg,
                seed_offset=offset,
            )
            train_parts.append(train_subset)
            validation_parts.append(validation_subset)
        return ConcatDataset(train_parts), ConcatDataset(validation_parts)
    return split_single_dataset_for_internal_validation(dataset, cfg, seed_offset=0)


def split_single_dataset_for_internal_validation(dataset: Any, cfg: dict[str, Any], *, seed_offset: int) -> tuple[Subset, Subset]:
    validation_cfg = validation_from_train_cfg(cfg)
    fraction = float(validation_cfg.get("fraction", validation_cfg.get("val_fraction", 0.1)))
    if not 0.0 < fraction < 1.0:
        raise ValueError("data.validation_from_train.fraction must be between 0 and 1.")
    seed = int(validation_cfg.get("seed", cfg.get("experiment", {}).get("seed", 0))) + int(seed_offset)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset), dtype=np.int64)
    rng.shuffle(indices)
    validation_count = max(1, int(round(float(len(indices)) * fraction)))
    if validation_count >= len(indices):
        validation_count = max(1, len(indices) - 1)
    validation_indices = np.sort(indices[:validation_count]).astype(int).tolist()
    train_indices = np.sort(indices[validation_count:]).astype(int).tolist()
    if not train_indices:
        raise ValueError("Internal validation split left no training samples.")
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    annotate_internal_subset(train_subset, role="train", source_dataset=dataset, fraction=fraction, seed=seed)
    annotate_internal_subset(validation_subset, role="validation", source_dataset=dataset, fraction=fraction, seed=seed)
    return train_subset, validation_subset


def annotate_internal_subset(subset: Subset, *, role: str, source_dataset: Any, fraction: float, seed: int) -> None:
    subset.split = role  # type: ignore[attr-defined]
    subset.internal_split = {
        "enabled": True,
        "source_split": "train",
        "role": role,
        "fraction": float(fraction),
        "seed": int(seed),
        "parent_num_samples": int(len(source_dataset)),
    }  # type: ignore[attr-defined]


def build_dataset(cfg: dict[str, Any], split: str, **extra_dataset_kwargs: Any):
    import_default_components()
    dataset_cfg = deepcopy(cfg["data"]["dataset"])
    dataset_type = dataset_cfg.get("type")
    descriptor = _optional_dataset_descriptor(dataset_type)
    if descriptor is not None and not dataset_cfg.get("data_root"):
        dataset_cfg["data_root"] = descriptor.default_root
    dataset_cfg["split"] = split
    enabled_modalities = resolve_enabled_modalities(cfg)
    dataset_cfg["enabled_modalities"] = list(enabled_modalities)
    if descriptor is not None:
        dataset_cfg["input_profiles"] = resolve_dataset_profiles(dataset_type, enabled_modalities, dataset_cfg)
    dataset_cfg.update(dataset_flags_for_modalities(enabled_modalities))
    apply_cache_policy(dataset_cfg, cfg, enabled_modalities)
    canonicalize_lidar_dataset_config(dataset_cfg)
    if _uses_csv_split(dataset_type, descriptor):
        csv_name, dataset_split = _dataset_csv_for_split(dataset_cfg, split)
        dataset_cfg["csv_name"] = csv_name
        dataset_cfg["split"] = dataset_split
        _validate_snapshot_csv_exists(cfg, dataset_cfg, csv_name)
    dataset_cfg.update(extra_dataset_kwargs)
    return DATASETS.build(dataset_cfg)


def build_dataloaders(
    cfg: dict[str, Any],
    *,
    normalization_overrides: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    loader_cfg = cfg["data"]["dataloader"]
    training_cfg = cfg.get("training", {})
    provided = dict(normalization_overrides or {})
    protocol_splits = build_protocol_split_datasets(cfg, **provided)
    if protocol_splits is not None:
        train_dataset = protocol_splits["train"]
        validation_dataset = protocol_splits.get("validation")
        test_dataset = protocol_splits["test"]
    elif validation_from_train_enabled(cfg):
        train_dataset, validation_dataset = build_train_and_internal_validation_datasets(
            cfg,
            split_dataset_builder=build_split_dataset,
            provided=provided or None,
        )
        dataset_kwargs = provided or normalization_kwargs(train_dataset)
        test_dataset = build_split_dataset(
            cfg,
            "test",
            normalization_overrides=provided or None,
            **({} if provided else dataset_kwargs),
        )
    else:
        train_dataset = build_split_dataset(cfg, "train", normalization_overrides=provided or None)
        if provided:
            fit_train_normalizers(
                train_dataset,
                source="provided_train_artifact",
                provided=provided,
            )
            train_normalization = provided
        else:
            prepare_lidar_normalizer(cfg, train_dataset)
            train_normalization = normalization_kwargs(train_dataset)
        validation_dataset = (
            build_split_dataset(
                cfg,
                "validation",
                normalization_overrides=provided or None,
                **({} if provided else train_normalization),
            )
            if has_validation_csv(cfg)
            else None
        )
        dataset_kwargs = provided or normalization_kwargs(train_dataset)
        test_dataset = build_split_dataset(
            cfg,
            "test",
            normalization_overrides=provided or None,
            **({} if provided else dataset_kwargs),
        )
    if not provided:
        prepare_lidar_normalizer(cfg, train_dataset)
    dataloaders = {
        "train": build_dataloader(
            train_dataset,
            loader_cfg,
            split="train",
            epoch_subsampling_cfg=training_cfg.get("epoch_subsampling"),
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
            domain_balanced_sampling_cfg=cfg.get("data", {}).get("domain_balanced_sampling"),
        ),
        "test": build_dataloader(
            test_dataset,
            loader_cfg,
            split="test",
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
        ),
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
):
    provided = dict(normalization_overrides or {})
    build_kwargs = dict(extra_dataset_kwargs)
    if provided:
        build_kwargs.update(provided)
    protocol_splits = build_protocol_split_datasets(cfg, **build_kwargs)
    if protocol_splits is not None:
        split_key = "validation" if split == "val" else split
        if split_key not in protocol_splits:
            raise ValueError(f"Split protocol did not produce split '{split}'.")
        return protocol_splits[split_key]
    domain_dataset = _build_mmw_domain_dataset(
        cfg,
        split,
        normalization_overrides=provided or None,
        **extra_dataset_kwargs,
    )
    if domain_dataset is not None:
        return domain_dataset
    scenes = dataset_scenes_for_split(cfg, split)
    if not scenes:
        return build_dataset(cfg, split, **build_kwargs)
    if len(scenes) == 1:
        scene_cfg = retarget_cfg_for_scene(cfg, scenes[0])
        return build_dataset(scene_cfg, split, **build_kwargs)
    if split == "train":
        train_kwargs = normalization_fit_placeholders(cfg)
        train_kwargs.update(build_kwargs)
        build_kwargs = train_kwargs
    datasets = [build_dataset(retarget_cfg_for_scene(cfg, scene), split, **build_kwargs) for scene in scenes]
    if split == "train":
        if provided:
            pooled = ConcatDataset(datasets)
            fit_train_normalizers(
                pooled,
                source="provided_train_artifact",
                provided=provided,
            )
            return pooled
        harmonize_multi_scene_train_normalizers(datasets)
    return ConcatDataset(datasets)


def build_protocol_split_datasets(cfg: dict[str, Any], **extra_dataset_kwargs: Any) -> dict[str, Any] | None:
    return _build_protocol_split_datasets(
        cfg,
        dataset_builder=build_dataset,
        **extra_dataset_kwargs,
    )


def build_dataloader(
    dataset: Any,
    loader_cfg: dict[str, Any],
    *,
    split: str,
    epoch_subsampling_cfg: dict[str, Any] | None = None,
    experiment_seed: int | None = None,
    domain_balanced_sampling_cfg: dict[str, Any] | None = None,
) -> DataLoader:
    kwargs = build_dataloader_kwargs(loader_cfg, split=split)
    generator_metadata = None
    if experiment_seed is not None:
        generator_metadata = dataloader_generator_metadata(dataset, split=split, base_seed=experiment_seed)
        generator = torch.Generator()
        generator.manual_seed(int(generator_metadata["derived_seed"]))
        kwargs["generator"] = generator
        kwargs["worker_init_fn"] = _seed_dataloader_worker
    if split == "train":
        domain_sampler = build_domain_balanced_sampler(
            dataset,
            domain_balanced_sampling_cfg,
            experiment_seed=experiment_seed,
        )
        sampler = build_epoch_subsample_sampler(
            dataset,
            epoch_subsampling_cfg,
            experiment_seed=experiment_seed,
        )
        if domain_sampler is not None and sampler is not None:
            raise ValueError("domain-balanced sampling cannot be combined with training.epoch_subsampling.")
        if domain_sampler is not None:
            kwargs["shuffle"] = False
            kwargs["sampler"] = domain_sampler
        if sampler is not None:
            kwargs["shuffle"] = False
            kwargs["sampler"] = sampler
    dataloader = DataLoader(dataset, **kwargs)
    if generator_metadata is not None:
        dataloader.generator_metadata = generator_metadata
    return dataloader


def dataloader_generator_metadata(dataset: Any, *, split: str, base_seed: int) -> dict[str, Any]:
    normalized_split = "validation" if split == "val" else str(split)
    dataset_fingerprint = _dataset_fingerprint(dataset)
    identity = {
        "algorithm": "sha256-v1",
        "base_seed": int(base_seed),
        "split": normalized_split,
        "dataset_fingerprint": dataset_fingerprint,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    identity["derived_seed"] = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return identity


def capture_dataloaders_random_state(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    return {split: _capture_dataloader_random_state(loader) for split, loader in dataloaders.items()}


def restore_dataloaders_random_state(dataloaders: dict[str, DataLoader], state: dict[str, Any]) -> None:
    if set(dataloaders) != set(state):
        raise ValueError(
            "DataLoader random-state splits do not match: "
            f"current={sorted(dataloaders)}, checkpoint={sorted(state)}."
        )
    for split, loader in dataloaders.items():
        _restore_dataloader_random_state(loader, state[split], split=split)


def _capture_dataloader_random_state(dataloader: DataLoader) -> dict[str, Any]:
    generator = getattr(dataloader, "generator", None)
    sampler = getattr(dataloader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "identity": dict(getattr(dataloader, "generator_metadata", {})),
        "generator_state": generator.get_state().clone() if isinstance(generator, torch.Generator) else None,
        "sampler": {
            "type": type(sampler).__name__ if sampler is not None else None,
            "epoch": int(getattr(sampler, "epoch")) if hasattr(sampler, "epoch") else None,
            "generator_state": (
                sampler_generator.get_state().clone()
                if isinstance(sampler_generator, torch.Generator) and sampler_generator is not generator
                else None
            ),
        },
    }
    return payload


def _restore_dataloader_random_state(dataloader: DataLoader, state: dict[str, Any], *, split: str) -> None:
    current_identity = dict(getattr(dataloader, "generator_metadata", {}))
    if state.get("identity") != current_identity:
        raise ValueError(
            f"DataLoader generator identity mismatch for split '{split}': "
            f"checkpoint={state.get('identity')}, current={current_identity}."
        )
    generator = getattr(dataloader, "generator", None)
    generator_state = state.get("generator_state")
    if generator_state is not None:
        if not isinstance(generator, torch.Generator):
            raise ValueError(f"DataLoader split '{split}' has no generator to restore.")
        generator.set_state(generator_state)
    sampler = getattr(dataloader, "sampler", None)
    sampler_state = state.get("sampler") or {}
    expected_type = sampler_state.get("type")
    if expected_type != (type(sampler).__name__ if sampler is not None else None):
        raise ValueError(
            f"DataLoader sampler type mismatch for split '{split}': "
            f"checkpoint={expected_type}, current={type(sampler).__name__ if sampler is not None else None}."
        )
    if sampler_state.get("epoch") is not None:
        setter = getattr(sampler, "set_epoch", None)
        if not callable(setter):
            raise ValueError(f"DataLoader sampler for split '{split}' cannot restore epoch state.")
        setter(int(sampler_state["epoch"]))
    sampler_generator_state = sampler_state.get("generator_state")
    if sampler_generator_state is not None:
        sampler_generator = getattr(sampler, "generator", None)
        if not isinstance(sampler_generator, torch.Generator):
            raise ValueError(f"DataLoader sampler for split '{split}' has no generator to restore.")
        sampler_generator.set_state(sampler_generator_state)


def _dataset_fingerprint(dataset: Any) -> str:
    if isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices, dtype=np.int64)
        payload: Any = {
            "kind": "subset",
            "parent": _dataset_fingerprint(dataset.dataset),
            "indices": hashlib.sha256(indices.tobytes()).hexdigest(),
            "length": len(dataset),
        }
    elif isinstance(dataset, ConcatDataset):
        payload = {
            "kind": "concat",
            "components": [_dataset_fingerprint(component) for component in dataset.datasets],
            "length": len(dataset),
        }
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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _build_mmw_domain_dataset(
    cfg: dict[str, Any],
    split: str,
    *,
    normalization_overrides: dict[str, Any] | None = None,
    **extra_dataset_kwargs: Any,
):
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    domains = dataset_cfg.get("domains") if isinstance(dataset_cfg, dict) else None
    if domains is None:
        return None
    if str(dataset_cfg.get("type", "")).strip().lower() != "mmw":
        raise ValueError("data.dataset.domains is currently supported only for dataset type 'mmw'.")
    if not isinstance(domains, list) or not domains:
        raise ValueError("data.dataset.domains must be a non-empty list.")
    seen_ids: set[str] = set()
    for index, raw_domain in enumerate(domains):
        if not isinstance(raw_domain, dict):
            raise ValueError(f"MMW domain at index {index} must be a mapping.")
        missing = [key for key in ("id", "condition", "scene", "data_root") if not raw_domain.get(key)]
        if missing:
            raise ValueError(f"MMW domain at index {index} is missing required fields: {', '.join(missing)}.")
        domain_id = str(raw_domain["id"])
        if domain_id in seen_ids:
            raise ValueError(f"Duplicate MMW domain id: {domain_id}.")
        seen_ids.add(domain_id)
    datasets = []
    inventory = []
    provided = dict(normalization_overrides or {})
    build_kwargs = dict(extra_dataset_kwargs)
    build_kwargs.update(provided)
    if split == "train":
        build_kwargs = normalization_fit_placeholders(cfg)
        build_kwargs.update(extra_dataset_kwargs)
        build_kwargs.update(provided)
    for index, raw_domain in enumerate(domains):
        domain_id = str(raw_domain["id"])
        csv_key, csv_name = _domain_csv_for_split(raw_domain, split)
        csv_path = Path(str(csv_name))
        if not csv_path.is_absolute():
            csv_path = Path(str(raw_domain["data_root"])) / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"MMW domain {domain_id} {split} CSV is missing: {csv_path}.")
        leaf_cfg = deepcopy(cfg)
        leaf_dataset_cfg = leaf_cfg["data"]["dataset"]
        leaf_dataset_cfg.pop("domains", None)
        leaf_dataset_cfg.update(
            {
                "condition": str(raw_domain["condition"]),
                "scene": str(raw_domain["scene"]),
                "data_root": str(raw_domain["data_root"]),
                csv_key: str(csv_name),
            }
        )
        if split in {"validation", "val"}:
            leaf_dataset_cfg["val_csv_name"] = str(csv_name)
        dataset = build_dataset(leaf_cfg, split, **build_kwargs)
        dataset.domain_id = domain_id
        dataset.domain_condition = str(raw_domain["condition"])
        dataset.domain_scene = str(raw_domain["scene"])
        dataset.domain_split_path = str(csv_path)
        datasets.append(dataset)
        inventory.append(
            {
                "id": domain_id,
                "condition": str(raw_domain["condition"]),
                "scene": str(raw_domain["scene"]),
                "data_root": str(raw_domain["data_root"]),
                "split": "validation" if split == "val" else split,
                "split_path": str(csv_path),
                "sample_count": int(len(dataset)),
            }
        )
    pooled = ConcatDataset(datasets)
    pooled.domain_inventory = inventory
    if split == "train":
        if provided:
            fit_train_normalizers(
                pooled,
                source="provided_train_artifact",
                provided=provided,
            )
        else:
            harmonize_multi_scene_train_normalizers(datasets)
    return pooled


def _domain_csv_for_split(domain: dict[str, Any], split: str) -> tuple[str, str]:
    if split == "train":
        keys = ("train_csv_name",)
    elif split in {"validation", "val"}:
        keys = ("val_csv_name", "validation_csv_name")
    else:
        keys = ("test_csv_name",)
    for key in keys:
        if domain.get(key):
            return ("val_csv_name" if split in {"validation", "val"} else key), str(domain[key])
    raise ValueError(f"MMW domain {domain.get('id', '<unknown>')} is missing {split} CSV field ({', '.join(keys)}).")


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
    sample_counts = [len(component) for component in components]
    if any(count <= 0 for count in sample_counts):
        raise ValueError("domain-balanced sampling requires every domain to contain at least one sample.")
    weights = torch.cat(
        [torch.full((count,), 1.0 / float(count), dtype=torch.double) for count in sample_counts]
    )
    seed = int(config.get("seed", experiment_seed or 0))
    generator = torch.Generator().manual_seed(seed)
    replacement = bool(config.get("replacement", True))
    num_samples = int(config.get("num_samples", len(dataset)))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )
    sampler.domain_balanced_metadata = {
        "sampler": "WeightedRandomSampler",
        "seed": seed,
        "replacement": replacement,
        "num_samples": num_samples,
        "domains": [
            {
                "id": str(item["id"]),
                "sample_count": int(count),
                "sample_weight": 1.0 / float(count),
                "total_weight": 1.0,
            }
            for item, count in zip(inventory, sample_counts)
        ],
    }
    return sampler


def _seed_dataloader_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset_csv_for_split(dataset_cfg: dict[str, Any], split: str) -> tuple[str | None, str]:
    if split == "train":
        return dataset_cfg.get("train_csv_name"), "train"
    if split in {"validation", "val"}:
        val_csv = dataset_cfg.get("val_csv_name")
        if not val_csv:
            raise ValueError("data.dataset.val_csv_name is required when building a validation split.")
        return val_csv, "validation"
    return dataset_cfg.get("test_csv_name"), split


def _optional_dataset_descriptor(dataset_type: Any):
    dataset_key = str(dataset_type or "deepsense6g").strip().lower()
    if dataset_key in {"synthetic", "synthetic_sequence"}:
        return None
    return dataset_descriptor(dataset_key)


def _uses_csv_split(dataset_type: Any, descriptor: Any) -> bool:
    dataset_key = str(dataset_type or "deepsense6g").strip().lower()
    if dataset_key in {"synthetic", "synthetic_sequence"}:
        return False
    return descriptor is None or descriptor.storage_kind == "csv_sequence"


def _validate_snapshot_csv_exists(cfg: dict[str, Any], dataset_cfg: dict[str, Any], csv_name: str | None) -> None:
    if cfg.get("experiment", {}).get("variant") != "snapshot_next_frame":
        return
    if not csv_name:
        raise FileNotFoundError(
            "Snapshot next-frame baseline requires snapshot CSVs. "
            "Run: kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml"
        )
    data_root = dataset_cfg.get("data_root")
    csv_path = Path(str(csv_name))
    if not csv_path.is_absolute():
        csv_path = Path(str(data_root or ".")) / csv_path
    if not csv_path.exists():
        expected = {SNAPSHOT_TRAIN_CSV, SNAPSHOT_VAL_CSV}
        raise FileNotFoundError(
            f"Snapshot next-frame baseline expected {csv_path}. "
            f"Generate {sorted(expected)} first with: "
            "kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml"
        )


__all__ = [
    "build_dataloader",
    "build_dataloader_kwargs",
    "build_dataloaders",
    "build_dataset",
    "build_protocol_split_datasets",
    "build_split_dataset",
    "capture_dataloaders_random_state",
    "dataloader_generator_metadata",
    "prepare_lidar_normalizer",
    "resolve_dataloader_split_config",
    "restore_dataloaders_random_state",
    "shutdown_dataloader_workers",
]
