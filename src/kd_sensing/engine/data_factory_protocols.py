from copy import deepcopy
from typing import Any, Callable

from torch.utils.data import ConcatDataset, Subset

from kd_sensing.data.scenes import (
    is_deepsense_dataset_type,
    retarget_deepsense_dataset_config,
)
from kd_sensing.engine.data_factory_groups import (
    audit_temporal_split_datasets,
    sequence_group_keys_for_dataset,
    stratified_indices_by_label,
    stratified_indices_by_label_and_sequence_group,
    target_labels_for_dataset,
)
from kd_sensing.engine.data_factory_scalers import (
    fit_or_apply_protocol_normalizers,
    normalization_fit_placeholders,
)


STRATIFIED_2604_PROTOCOLS = {
    "stratified_80_10_10",
    "deepsense6g_2604_stratified_80_10_10",
    "2604_stratified_80_10_10",
}
STRATIFIED_SAMPLE_STRATEGIES = {
    "stratified_by_target_beam_per_scene",
    "sample_stratified_by_target_beam_per_scene",
}
STRATIFIED_SEQUENCE_GROUP_STRATEGIES = {
    "stratified_by_target_beam_per_scene_sequence_group",
    "stratified_by_target_beam_per_scene_group_safe",
    "sequence_group_stratified_by_target_beam_per_scene",
    "group_safe_stratified_by_target_beam_per_scene",
}
DEFAULT_SAMPLE_STRATEGY = "stratified_by_target_beam_per_scene"
DEFAULT_SEQUENCE_GROUP_STRATEGY = "stratified_by_target_beam_per_scene_sequence_group"
SEQUENCE_GROUP_IDENTITY_POLICY = "scene_id:seq_index"


def build_protocol_split_datasets(
    cfg: dict[str, Any],
    *,
    dataset_builder: Callable[..., Any],
    **extra_dataset_kwargs: Any,
) -> dict[str, Any] | None:
    if not stratified_2604_split_enabled(cfg):
        return None
    split_cfg = stratified_2604_split_cfg(cfg)
    role_scenes = {
        "train": dataset_scenes_for_protocol_role(cfg, "train"),
        "validation": dataset_scenes_for_protocol_role(cfg, "validation"),
        "test": dataset_scenes_for_protocol_role(cfg, "test"),
    }
    all_scenes = ordered_unique(
        scene
        for scenes in role_scenes.values()
        for scene in scenes
    )
    if not all_scenes:
        raise ValueError("stratified_80_10_10 split requires at least one DeepSense6G scene.")
    source_splits = tuple(str(item) for item in split_cfg.get("source_splits", ("train", "test")))
    build_kwargs = normalization_fit_placeholders(cfg)
    build_kwargs.update(extra_dataset_kwargs)
    scene_subsets: dict[str, dict[Any, Any]] = {"train": {}, "validation": {}, "test": {}}
    scene_sources: dict[Any, Any] = {}
    scene_labels: dict[Any, list[int]] = {}
    scene_indices: dict[Any, dict[str, list[int]]] = {}
    for scene_offset, scene in enumerate(all_scenes):
        full_scene = build_protocol_union_dataset(cfg, scene, source_splits, build_kwargs, dataset_builder)
        labels = target_labels_for_dataset(full_scene)
        index_splits = protocol_indices_by_strategy(
            full_scene,
            labels,
            split_cfg=split_cfg,
            seed=int(split_cfg["seed"]) + scene_offset,
            validation_fraction=float(split_cfg["validation_fraction"]),
            test_fraction=float(split_cfg["test_fraction"]),
        )
        scene_sources[scene] = full_scene
        scene_labels[scene] = labels
        scene_indices[scene] = index_splits
        for role, indices in index_splits.items():
            scene_subsets[role][scene] = Subset(full_scene, indices)
    result: dict[str, Any] = {}
    for role, scenes in role_scenes.items():
        parts = [scene_subsets[role][scene] for scene in scenes]
        result[role] = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    identity_audit = audit_temporal_split_datasets(result)
    for role, scenes in role_scenes.items():
        for scene in scenes:
            subset = scene_subsets[role][scene]
            annotate_protocol_subset(
                subset,
                role=role,
                source_dataset=scene_sources[scene],
                scene=scene,
                split_cfg=split_cfg,
                source_splits=source_splits,
                labels=[scene_labels[scene][int(index)] for index in scene_indices[scene][role]],
                identity_audit=identity_audit,
            )
    fit_or_apply_protocol_normalizers(
        result["train"],
        result.get("validation"),
        result["test"],
        provided=extra_dataset_kwargs,
    )
    return result


def dataset_scenes_for_split(cfg: dict[str, Any], split: str) -> tuple[Any, ...]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return ()
    if split == "train":
        raw = dataset_cfg.get("train_scenes")
    elif split in {"validation", "val"}:
        raw = dataset_cfg.get("validation_scenes", dataset_cfg.get("val_scenes", dataset_cfg.get("train_scenes")))
    else:
        raw = dataset_cfg.get("test_scenes", dataset_cfg.get("eval_scenes", dataset_cfg.get("validation_scenes")))
    if raw is None:
        return ()
    if isinstance(raw, (str, int, float)):
        return (raw,)
    return tuple(raw)


def stratified_2604_split_enabled(cfg: dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return False
    protocol = str(dataset_cfg.get("split_protocol") or "").strip().lower()
    return protocol in STRATIFIED_2604_PROTOCOLS


def stratified_2604_split_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    fractions = dataset_cfg.get("split_fractions") if isinstance(dataset_cfg.get("split_fractions"), dict) else {}
    validation_fraction = float(fractions.get("validation", fractions.get("val", 0.1)))
    test_fraction = float(fractions.get("test", 0.1))
    if validation_fraction <= 0.0 or test_fraction <= 0.0 or validation_fraction + test_fraction >= 1.0:
        raise ValueError("data.dataset.split_fractions must define positive validation/test fractions with train > 0.")
    history_window = int(dataset_cfg.get("seq_len", cfg.get("model", {}).get("seq_length", 1)) or 1)
    default_strategy = DEFAULT_SEQUENCE_GROUP_STRATEGY if history_window > 1 else DEFAULT_SAMPLE_STRATEGY
    strategy = str(dataset_cfg.get("split_strategy") or default_strategy).strip().lower()
    if history_window > 1 and strategy in STRATIFIED_SAMPLE_STRATEGIES:
        raise ValueError(
            "Overlapping temporal windows cannot use a sample-level label-stratified split. "
            f"Use the group-safe sequence strategy '{DEFAULT_SEQUENCE_GROUP_STRATEGY}'."
        )
    group_identity_policy = str(
        dataset_cfg.get("split_group_identity_policy")
        or (SEQUENCE_GROUP_IDENTITY_POLICY if strategy in STRATIFIED_SEQUENCE_GROUP_STRATEGIES else "not_applicable")
    ).strip()
    if strategy in STRATIFIED_SEQUENCE_GROUP_STRATEGIES and group_identity_policy != SEQUENCE_GROUP_IDENTITY_POLICY:
        raise ValueError(
            "Sequence-group split requires split_group_identity_policy "
            f"'{SEQUENCE_GROUP_IDENTITY_POLICY}'."
        )
    return {
        "protocol": str(dataset_cfg.get("split_protocol")),
        "strategy": strategy,
        "group_identity_policy": group_identity_policy,
        "history_window": history_window,
        "seed": int(dataset_cfg.get("split_seed", cfg.get("experiment", {}).get("seed", 0))),
        "train_fraction": float(1.0 - validation_fraction - test_fraction),
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "source_splits": tuple(dataset_cfg.get("split_source_splits") or ("train", "test")),
        "label_source": str(dataset_cfg.get("split_label_source") or "future_beam1"),
    }


def protocol_indices_by_strategy(
    dataset: Any,
    labels: list[int],
    *,
    split_cfg: dict[str, Any],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[int]]:
    strategy = str(split_cfg.get("strategy") or "stratified_by_target_beam_per_scene").strip().lower()
    if strategy in STRATIFIED_SAMPLE_STRATEGIES:
        if int(split_cfg.get("history_window", 1) or 1) > 1:
            raise ValueError(
                "Overlapping temporal windows cannot use a sample-level label-stratified split. "
                f"Use the group-safe sequence strategy '{DEFAULT_SEQUENCE_GROUP_STRATEGY}'."
            )
        return stratified_indices_by_label(
            labels,
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    if strategy in STRATIFIED_SEQUENCE_GROUP_STRATEGIES:
        return stratified_indices_by_label_and_sequence_group(
            labels,
            sequence_group_keys_for_dataset(dataset),
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    supported = sorted(STRATIFIED_SAMPLE_STRATEGIES | STRATIFIED_SEQUENCE_GROUP_STRATEGIES)
    raise ValueError(f"Unsupported stratified_80_10_10 split_strategy '{strategy}'. Expected one of {supported}.")


def dataset_scenes_for_protocol_role(cfg: dict[str, Any], role: str) -> tuple[Any, ...]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return ()
    if role == "train":
        raw = dataset_cfg.get("train_scenes", dataset_cfg.get("scenes", dataset_cfg.get("scene")))
    elif role == "validation":
        raw = dataset_cfg.get(
            "validation_scenes",
            dataset_cfg.get("val_scenes", dataset_cfg.get("train_scenes", dataset_cfg.get("scenes"))),
        )
    else:
        raw = dataset_cfg.get(
            "test_scenes",
            dataset_cfg.get("eval_scenes", dataset_cfg.get("scenes", dataset_cfg.get("train_scenes"))),
        )
    if raw is None:
        raw = dataset_cfg.get("scene")
    if isinstance(raw, (str, int, float)):
        return (raw,)
    return tuple(raw or ())


def ordered_unique(values) -> tuple[Any, ...]:
    result = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def build_protocol_union_dataset(
    cfg: dict[str, Any],
    scene: Any,
    source_splits: tuple[str, ...],
    extra_dataset_kwargs: dict[str, Any],
    dataset_builder: Callable[..., Any],
) -> Any:
    parts = []
    for source_split in source_splits:
        scene_cfg = retarget_cfg_for_scene(cfg, scene)
        parts.append(dataset_builder(scene_cfg, source_split, **extra_dataset_kwargs))
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def annotate_protocol_subset(
    subset: Subset,
    *,
    role: str,
    source_dataset: Any,
    scene: Any,
    split_cfg: dict[str, Any],
    source_splits: tuple[str, ...],
    labels: list[int],
    identity_audit: dict[str, Any],
) -> None:
    subset.split = role  # type: ignore[attr-defined]
    label_counts = {str(label): int(labels.count(label)) for label in sorted(set(labels))}
    subset.stratified_split = {  # type: ignore[attr-defined]
        "enabled": True,
        "protocol": split_cfg["protocol"],
        "strategy": split_cfg["strategy"],
        "group_identity_policy": split_cfg["group_identity_policy"],
        "source_split": "train+test",
        "source_splits": list(source_splits),
        "role": role,
        "scene": scene,
        "seed": int(split_cfg["seed"]),
        "train_fraction": float(split_cfg["train_fraction"]),
        "validation_fraction": float(split_cfg["validation_fraction"]),
        "test_fraction": float(split_cfg["test_fraction"]),
        "label_source": split_cfg["label_source"],
        "parent_num_samples": int(len(source_dataset)),
        "label_count": len(label_counts),
        "label_distribution": label_counts,
        "identity_audit": identity_audit,
    }


def retarget_cfg_for_scene(cfg: dict[str, Any], scene: Any) -> dict[str, Any]:
    scene_cfg = deepcopy(cfg)
    dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
    if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw":
        dataset_cfg["scene"] = str(scene)
        return scene_cfg
    if not is_deepsense_dataset_type(dataset_cfg.get("type", "deepsense6g")):
        raise ValueError("data.dataset.train_scenes/test_scenes are currently supported only for DeepSense6G.")
    retarget_deepsense_dataset_config(dataset_cfg, scene)
    return scene_cfg


__all__ = [
    "annotate_protocol_subset",
    "build_protocol_split_datasets",
    "build_protocol_union_dataset",
    "dataset_scenes_for_protocol_role",
    "dataset_scenes_for_split",
    "ordered_unique",
    "protocol_indices_by_strategy",
    "retarget_cfg_for_scene",
    "stratified_2604_split_cfg",
    "stratified_2604_split_enabled",
]
