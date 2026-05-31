from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from kd_sensing.data.loso import LOSOFold, TargetSplit, resolve_loso_fold, split_target_records
from kd_sensing.data.mmw.protocol import MMWFold
from kd_sensing.data.samples import _select_portion
from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.engine.data_factory import build_dataloader, build_dataloader_kwargs, build_dataset, prepare_lidar_normalizer


class NamedSplitSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int] | tuple[int, ...],
        *,
        split_name: str,
        csv_indices: list[int] | tuple[int, ...] | None = None,
    ):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.csv_indices = tuple(int(index) for index in csv_indices) if csv_indices is not None else self.indices
        self.split = split_name
        self.split_name = split_name
        self.scene_id = getattr(dataset, "scene_id", None)
        self.scene_slug = getattr(dataset, "scene_slug", None)
        self.enabled_modalities = getattr(dataset, "enabled_modalities", ())
        self.root_csv = getattr(dataset, "root_csv", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample = self.dataset[self.indices[index]]
        if isinstance(sample, dict):
            metadata = sample.get("metadata")
            if isinstance(metadata, dict):
                metadata = dict(metadata)
                metadata["split"] = self.split_name
                metadata["base_dataset_index"] = self.indices[index]
                if index < len(self.csv_indices):
                    metadata["base_csv_index"] = self.csv_indices[index]
                sample = dict(sample)
                sample["metadata"] = metadata
        return sample


def build_source_multi_scene_dataset(
    cfg: dict[str, Any],
    fold: LOSOFold | MMWFold | dict[str, Any] | None = None,
    *,
    target_scene: Any | None = None,
    source_scenes: list[Any] | tuple[Any, ...] | None = None,
    split: str = "train",
    return_metadata: bool = True,
) -> ConcatDataset:
    resolved = _resolve_fold(fold, target_scene=target_scene, source_scenes=source_scenes)
    datasets = []
    dataset_kwargs: dict[str, Any] = {}
    for index, scene in enumerate(resolved.source_scenes):
        scene_cfg = deepcopy(cfg)
        dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
        _set_soft_beam_label_domain(dataset_cfg, "source")
        _retarget_dataset_config_for_fold(dataset_cfg, scene)
        _apply_loso_scene_overrides(scene_cfg, scene)
        dataset = build_dataset(scene_cfg, split, return_metadata=return_metadata, **dataset_kwargs)
        if index == 0:
            prepare_lidar_normalizer(scene_cfg, dataset)
            dataset_kwargs = _normalization_kwargs(dataset)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def build_target_adapt_test_datasets(
    cfg: dict[str, Any],
    fold: LOSOFold | MMWFold | dict[str, Any] | None = None,
    *,
    target_scene: Any | None = None,
    source_scenes: list[Any] | tuple[Any, ...] | None = None,
    target_split: str = "test",
    adapt_fraction: float = 0.2,
    split_seed: int = 0,
    return_metadata: bool = True,
    **dataset_kwargs: Any,
) -> tuple[NamedSplitSubset, NamedSplitSubset, TargetSplit]:
    resolved = _resolve_fold(fold, target_scene=target_scene, source_scenes=source_scenes)
    target_cfg = deepcopy(cfg)
    dataset_cfg = target_cfg.setdefault("data", {}).setdefault("dataset", {})
    _set_soft_beam_label_domain(dataset_cfg, "target")
    _retarget_dataset_config_for_fold(dataset_cfg, resolved.target_scene)
    _apply_loso_scene_overrides(target_cfg, resolved.target_scene)
    target_dataset = build_dataset(target_cfg, target_split, return_metadata=return_metadata, **dataset_kwargs)
    split_result, selected_csv_indices = _split_target_dataset_records(
        target_dataset,
        dataset_cfg,
        adapt_fraction=adapt_fraction,
        seed=split_seed,
    )
    adapt_indices = tuple(int(index) for index in split_result.adapt_indices)
    test_indices = tuple(int(index) for index in split_result.test_indices)
    adapt_csv_indices = tuple(selected_csv_indices[index] for index in adapt_indices)
    test_csv_indices = tuple(selected_csv_indices[index] for index in test_indices)
    metadata = dict(split_result.metadata)
    metadata.update(
        {
            "target_scene": resolved.target_scene,
            "source_scenes": list(resolved.source_scenes),
            "target_split": target_split,
            "target_dataset_count": len(target_dataset),
            "target_selected_csv_indices": [int(index) for index in selected_csv_indices],
            "target_adapt_csv_indices": [int(index) for index in adapt_csv_indices],
            "target_test_csv_indices": [int(index) for index in test_csv_indices],
        }
    )
    split_result = TargetSplit(adapt_indices=adapt_indices, test_indices=test_indices, metadata=metadata)
    return (
        NamedSplitSubset(target_dataset, adapt_indices, split_name="target_adapt", csv_indices=adapt_csv_indices),
        NamedSplitSubset(target_dataset, test_indices, split_name="target_test", csv_indices=test_csv_indices),
        split_result,
    )


def build_loso_dataloaders(
    cfg: dict[str, Any],
    fold: LOSOFold | MMWFold | dict[str, Any] | None = None,
    *,
    target_scene: Any | None = None,
    source_scenes: list[Any] | tuple[Any, ...] | None = None,
    adapt_fraction: float = 0.2,
    split_seed: int = 0,
) -> dict[str, DataLoader | TargetSplit | LOSOFold]:
    resolved = _resolve_fold(fold, target_scene=target_scene, source_scenes=source_scenes)
    loader_cfg = cfg["data"]["dataloader"]
    source_dataset = build_source_multi_scene_dataset(cfg, resolved)
    first_source = source_dataset.datasets[0] if getattr(source_dataset, "datasets", None) else None
    dataset_kwargs = _normalization_kwargs(first_source)
    target_adapt, target_test, split_result = build_target_adapt_test_datasets(
        cfg,
        resolved,
        adapt_fraction=adapt_fraction,
        split_seed=split_seed,
        **dataset_kwargs,
    )
    return {
        "fold": resolved,
        "target_split": split_result,
        "source_train": _build_source_train_dataloader(source_dataset, loader_cfg, cfg),
        "target_adapt": build_dataloader(target_adapt, loader_cfg, split="train"),
        "target_test": build_dataloader(target_test, loader_cfg, split="test"),
    }


def build_loso_source_train_loader(
    cfg: dict[str, Any],
    fold: LOSOFold | MMWFold | dict[str, Any] | None = None,
    *,
    target_scene: Any | None = None,
    source_scenes: list[Any] | tuple[Any, ...] | None = None,
) -> dict[str, DataLoader | LOSOFold | dict[str, Any]]:
    resolved = _resolve_fold(fold, target_scene=target_scene, source_scenes=source_scenes)
    loader_cfg = cfg["data"]["dataloader"]
    source_dataset = build_source_multi_scene_dataset(cfg, resolved)
    first_source = source_dataset.datasets[0] if getattr(source_dataset, "datasets", None) else None
    source_sampling = _source_sampling_metadata(source_dataset, cfg)
    return {
        "fold": resolved,
        "source_train": _build_source_train_dataloader(source_dataset, loader_cfg, cfg),
        "normalization_kwargs": _normalization_kwargs(first_source),
        "source_sampling": source_sampling,
    }


def build_loso_target_stage_loader(
    cfg: dict[str, Any],
    fold: LOSOFold | MMWFold | dict[str, Any] | None = None,
    *,
    stage: str,
    target_scene: Any | None = None,
    source_scenes: list[Any] | tuple[Any, ...] | None = None,
    adapt_fraction: float = 0.2,
    split_seed: int = 0,
    dataset_kwargs: dict[str, Any] | None = None,
) -> dict[str, DataLoader | TargetSplit | LOSOFold]:
    if stage not in {"target_adapt", "target_test"}:
        raise ValueError("stage must be 'target_adapt' or 'target_test'.")
    resolved = _resolve_fold(fold, target_scene=target_scene, source_scenes=source_scenes)
    loader_cfg = cfg["data"]["dataloader"]
    target_adapt, target_test, split_result = build_target_adapt_test_datasets(
        cfg,
        resolved,
        adapt_fraction=adapt_fraction,
        split_seed=split_seed,
        **(dataset_kwargs or {}),
    )
    dataset = target_adapt if stage == "target_adapt" else target_test
    loader_split = "train" if stage == "target_adapt" else "test"
    return {
        "fold": resolved,
        "target_split": split_result,
        stage: build_dataloader(dataset, loader_cfg, split=loader_split),
    }


def _resolve_fold(
    fold: LOSOFold | MMWFold | dict[str, Any] | None,
    *,
    target_scene: Any | None,
    source_scenes: list[Any] | tuple[Any, ...] | None,
) -> LOSOFold | MMWFold:
    if isinstance(fold, (LOSOFold, MMWFold)):
        return fold
    if isinstance(fold, dict):
        if str(fold.get("dataset_family", fold.get("scene_family", ""))).upper() == "MMW":
            target = str(fold.get("target_scene", target_scene))
            sources = tuple(str(item) for item in fold.get("source_scenes", source_scenes or ()))
            return MMWFold(
                fold_id=str(fold.get("fold", fold.get("fold_id", f"target_{target}"))),
                target_scene=target,
                source_scenes=sources,
                condition=str(fold.get("condition", "sunny")),
                town=str(fold.get("town", "Town10")),
                protocol=str(fold.get("protocol", "mmw_scenario_loso")),
                claim_scope=str(fold.get("claim_scope", "scenario_loso" if sources else "single_scene_smoke")),
                cross_scene_claim_allowed=bool(fold.get("cross_scene_claim_allowed", bool(sources))),
            )
        target_scene = fold.get("target_scene", target_scene)
        source_scenes = fold.get("source_scenes", source_scenes)
    if target_scene is None:
        raise ValueError("target_scene is required to build LOSO data.")
    return resolve_loso_fold(target_scene=target_scene, source_scenes=source_scenes)


def _set_soft_beam_label_domain(dataset_cfg: dict[str, Any], domain: str) -> None:
    soft_cfg = dataset_cfg.get("soft_beam_labels")
    if soft_cfg is None:
        return
    if isinstance(soft_cfg, bool):
        dataset_cfg["soft_beam_labels"] = {"enabled": soft_cfg, "domain": domain}
        return
    if isinstance(soft_cfg, dict):
        resolved = dict(soft_cfg)
        resolved["domain"] = domain
        dataset_cfg["soft_beam_labels"] = resolved


def _normalization_kwargs(dataset: Any) -> dict[str, Any]:
    if dataset is None:
        return {}
    kwargs: dict[str, Any] = {}
    for attr, key in (
        ("gps_scaler", "gps_scaler"),
        ("lidar_normalizer", "lidar_normalizer"),
        ("mmwave_scaler", "mmwave_scaler"),
        ("csi_rms_normalizer", "csi_rms_normalizer"),
        ("occlusion_target_stats", "occlusion_target_stats"),
        ("position_target_scaler", "position_target_scaler"),
        ("codebook_metadata", "codebook_metadata"),
    ):
        if hasattr(dataset, attr):
            kwargs[key] = getattr(dataset, attr)
    return kwargs


def _build_source_train_dataloader(dataset: Any, loader_cfg: dict[str, Any], cfg: dict[str, Any]) -> DataLoader:
    sampler = _source_scene_balance_sampler(dataset, cfg)
    if sampler is None:
        return build_dataloader(dataset, loader_cfg, split="train")
    kwargs = build_dataloader_kwargs(loader_cfg, split="train")
    kwargs["shuffle"] = False
    kwargs["sampler"] = sampler
    return DataLoader(dataset, **kwargs)


def _source_scene_balance_sampler(dataset: Any, cfg: dict[str, Any]) -> WeightedRandomSampler | None:
    if not isinstance(dataset, ConcatDataset):
        return None
    datasets = list(getattr(dataset, "datasets", []))
    if len(datasets) <= 1:
        return None
    sampling_cfg = _source_scene_balance_cfg(cfg)
    if not sampling_cfg.get("enabled", False):
        return None
    lengths = [len(item) for item in datasets]
    if not lengths or any(length <= 0 for length in lengths):
        return None
    import torch

    weights: list[float] = []
    for length in lengths:
        weights.extend([1.0 / float(length)] * int(length))
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("experiment", {}).get("seed", 0)))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(sum(lengths)),
        replacement=True,
        generator=generator,
    )


def _source_sampling_metadata(dataset: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    datasets = list(getattr(dataset, "datasets", [])) if isinstance(dataset, ConcatDataset) else []
    lengths = [len(item) for item in datasets]
    scene_ids = [getattr(item, "scene_slug", getattr(item, "scene_id", None)) for item in datasets]
    balance_cfg = _source_scene_balance_cfg(cfg)
    enabled = bool(balance_cfg.get("enabled", False) and len(datasets) > 1 and all(length > 0 for length in lengths))
    return {
        "scene_balance_enabled": enabled,
        "source_scene_count": len(datasets),
        "source_scene_lengths": [int(length) for length in lengths],
        "source_scene_ids": [None if scene is None else str(scene) for scene in scene_ids],
        "strategy": "scene_balanced_weighted_sampler" if enabled else "concat_default",
    }


def _source_scene_balance_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    source_sampling = hist_cfg.get("source_sampling") if isinstance(hist_cfg.get("source_sampling"), dict) else {}
    scene_balance = source_sampling.get("scene_balance") if isinstance(source_sampling.get("scene_balance"), dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    training_balance = training_cfg.get("source_scene_balance") if isinstance(training_cfg.get("source_scene_balance"), dict) else {}
    merged = dict(scene_balance)
    merged.update(training_balance)
    return merged


def _split_target_dataset_records(
    target_dataset: Dataset,
    dataset_cfg: dict[str, Any],
    *,
    adapt_fraction: float,
    seed: int,
) -> tuple[TargetSplit, tuple[int, ...]]:
    import pandas as pd

    root_csv = getattr(target_dataset, "root_csv", None)
    if root_csv is None:
        raise ValueError("Target dataset does not expose root_csv for LOSO target split.")
    frame = pd.read_csv(root_csv, na_values="").fillna(-99)
    samples = getattr(target_dataset, "samples", None)
    sample_metadata = getattr(samples, "metadata", None) if samples is not None else None
    sample_metadata = sample_metadata if isinstance(sample_metadata, dict) else {}
    portion = float(sample_metadata.get("portion", dataset_cfg.get("portion", 1.0)))
    portion_strategy = str(sample_metadata.get("portion_strategy", dataset_cfg.get("portion_strategy", "even")))
    portion_seed = int(sample_metadata.get("portion_seed", dataset_cfg.get("portion_seed", 42)))
    selected_frame, selection_metadata = _select_portion(
        frame,
        portion=portion,
        strategy=portion_strategy,
        seed=portion_seed,
    )
    if len(selected_frame) != len(target_dataset):
        raise ValueError(
            "LOSO target split selection does not match the built target dataset: "
            f"selected {len(selected_frame)} rows from {root_csv}, but dataset length is {len(target_dataset)}."
        )
    selected_csv_indices = tuple(int(index) for index in selected_frame.index)
    split_result = split_target_records(
        selected_frame.to_dict(orient="records"),
        adapt_fraction=adapt_fraction,
        seed=seed,
    )
    metadata = dict(split_result.metadata)
    metadata.update(
        {
            "target_selection": {
                "source_csv": str(root_csv),
                "total_rows": int(selection_metadata.get("total_rows", len(frame))),
                "selected_rows": int(selection_metadata.get("selected_rows", len(selected_frame))),
                "portion": float(selection_metadata.get("portion", portion)),
                "portion_strategy": str(selection_metadata.get("portion_strategy", portion_strategy)),
                "portion_seed": int(selection_metadata.get("portion_seed", portion_seed)),
            }
        }
    )
    return TargetSplit(split_result.adapt_indices, split_result.test_indices, metadata), selected_csv_indices


def _apply_loso_scene_overrides(cfg: dict[str, Any], scene: Any) -> None:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    roots = loso_cfg.get("scene_data_roots")
    if isinstance(roots, dict):
        root = roots.get(str(scene), roots.get(scene))
        if root:
            dataset_cfg["data_root"] = str(root)
    csv_names = loso_cfg.get("scene_csv_names")
    if isinstance(csv_names, dict):
        scene_csv = csv_names.get(str(scene), csv_names.get(scene))
        if isinstance(scene_csv, dict):
            for key in ("train_csv_name", "test_csv_name", "val_csv_name"):
                if scene_csv.get(key):
                    dataset_cfg[key] = scene_csv[key]


def _retarget_dataset_config_for_fold(dataset_cfg: dict[str, Any], scene: Any) -> None:
    if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw":
        dataset_cfg["scene"] = str(scene)
        return
    retarget_deepsense_dataset_config(dataset_cfg, scene)


__all__ = [
    "NamedSplitSubset",
    "build_loso_dataloaders",
    "build_loso_source_train_loader",
    "build_loso_target_stage_loader",
    "build_source_multi_scene_dataset",
    "build_target_adapt_test_datasets",
]
