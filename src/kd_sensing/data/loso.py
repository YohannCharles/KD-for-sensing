from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.data.transform_ops.io import joined_resource


DEFAULT_LOSO_SCENES = (31, 32, 33, 34)
DEFAULT_TARGET_ORDER = (34, 33, 32, 31)
SUPPORTED_LABEL_BUDGETS = (0, 5, 10, 20, 50)


@dataclass(frozen=True)
class LOSOFold:
    fold_id: str
    target_scene: int
    source_scenes: tuple[int, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "target_scene": self.target_scene,
            "source_scenes": list(self.source_scenes),
            "scene_family": "DeepSense6G",
            "protocol": "loso_31_34",
        }


@dataclass(frozen=True)
class TargetSplit:
    adapt_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FewShotSampling:
    labeled_indices: tuple[int, ...]
    unlabeled_indices: tuple[int, ...]
    manifest: dict[str, Any]


def default_loso_folds(
    *,
    scenes: Sequence[Any] = DEFAULT_LOSO_SCENES,
    target_order: Sequence[Any] = DEFAULT_TARGET_ORDER,
) -> list[LOSOFold]:
    scene_ids = _normalize_scene_ids(scenes)
    target_ids = _normalize_scene_ids(target_order)
    missing = [scene for scene in target_ids if scene not in scene_ids]
    if missing:
        raise ValueError(f"LOSO target scenes {missing} are not in source scene universe {scene_ids}.")
    return [resolve_loso_fold(target_scene=target, scenes=scene_ids) for target in target_ids]


def resolve_loso_fold(
    *,
    target_scene: Any,
    source_scenes: Sequence[Any] | None = None,
    scenes: Sequence[Any] = DEFAULT_LOSO_SCENES,
) -> LOSOFold:
    target = resolve_deepsense_scene(target_scene).scene_id
    scene_ids = _normalize_scene_ids(scenes)
    if target not in scene_ids:
        raise ValueError(f"target_scene {target} must be one of {scene_ids}.")
    if source_scenes is None:
        sources = tuple(scene for scene in scene_ids if scene != target)
    else:
        sources = _normalize_scene_ids(source_scenes)
    if target in sources:
        raise ValueError("LOSO source/target scene must not overlap.")
    unknown_sources = [scene for scene in sources if scene not in scene_ids]
    if unknown_sources:
        raise ValueError(f"source_scenes {unknown_sources} are not in supported LOSO scenes {scene_ids}.")
    if not sources:
        raise ValueError("LOSO fold requires at least one source scene.")
    return LOSOFold(
        fold_id=f"target_scene{target}",
        target_scene=target,
        source_scenes=tuple(sources),
    )


def split_target_records(
    records: Sequence[Mapping[str, Any]],
    *,
    adapt_fraction: float = 0.2,
    seed: int = 0,
    seq_key: str = "seq_index",
    sample_id_key: str = "sample_id",
) -> TargetSplit:
    if not 0.0 < float(adapt_fraction) < 1.0:
        raise ValueError(f"adapt_fraction must be between 0 and 1, got {adapt_fraction}.")
    total = len(records)
    if total == 0:
        metadata = _target_split_metadata(
            seed=seed,
            adapt_fraction=adapt_fraction,
            total=0,
            adapt_indices=(),
            test_indices=(),
            unit="empty",
            adapt_units=(),
            test_units=(),
            sample_ids=(),
        )
        return TargetSplit(adapt_indices=(), test_indices=(), metadata=metadata)

    rng = np.random.default_rng(int(seed))
    sample_ids = tuple(_sample_id(record, index, sample_id_key=sample_id_key) for index, record in enumerate(records))
    seq_values = [_clean_group_value(record.get(seq_key)) for record in records]
    use_seq = all(value is not None for value in seq_values) and len(set(seq_values)) > 1
    if use_seq:
        groups: dict[Any, list[int]] = {}
        for index, value in enumerate(seq_values):
            groups.setdefault(value, []).append(index)
        ordered_units = list(groups)
        rng.shuffle(ordered_units)
        adapt_unit_count = _adapt_count(len(ordered_units), adapt_fraction)
        adapt_units = tuple(sorted(ordered_units[:adapt_unit_count]))
        test_units = tuple(sorted(ordered_units[adapt_unit_count:]))
        adapt_set = set(adapt_units)
        adapt_indices = tuple(index for unit in adapt_units for index in groups[unit])
        test_indices = tuple(index for unit in test_units for index in groups[unit])
        unit = seq_key
    else:
        indices = np.arange(total)
        rng.shuffle(indices)
        adapt_count = _adapt_count(total, adapt_fraction)
        adapt_indices = tuple(sorted(int(index) for index in indices[:adapt_count]))
        test_indices = tuple(sorted(int(index) for index in indices[adapt_count:]))
        adapt_set = set(adapt_indices)
        adapt_units = adapt_indices
        test_units = test_indices
        unit = "sample"

    if not adapt_indices and total > 0:
        adapt_indices = (0,)
        test_indices = tuple(index for index in range(1, total))
    if not test_indices and total > 1:
        test_indices = tuple(index for index in adapt_indices[1:])
        adapt_indices = adapt_indices[:1]
    metadata = _target_split_metadata(
        seed=seed,
        adapt_fraction=adapt_fraction,
        total=total,
        adapt_indices=adapt_indices,
        test_indices=test_indices,
        unit=unit,
        adapt_units=adapt_units,
        test_units=test_units,
        sample_ids=sample_ids,
    )
    if use_seq:
        metadata["adapt_seq_index"] = list(adapt_units)
        metadata["test_seq_index"] = list(test_units)
        metadata["seq_index_overlap"] = sorted(set(adapt_units) & set(test_units))
    metadata["adapt_test_sample_id_overlap"] = sorted(
        set(sample_ids[index] for index in adapt_indices) & set(sample_ids[index] for index in test_indices)
    )
    metadata["selection_unit"] = unit
    metadata["adapt_selection_units"] = list(adapt_units)
    metadata["test_selection_units"] = list(test_units)
    metadata["selection_set_debug"] = sorted(str(item) for item in adapt_set)[:10]
    return TargetSplit(
        adapt_indices=tuple(sorted(int(index) for index in adapt_indices)),
        test_indices=tuple(sorted(int(index) for index in test_indices)),
        metadata=metadata,
    )


def split_target_csv(
    csv_path: str | Path,
    *,
    adapt_fraction: float = 0.2,
    seed: int = 0,
    seq_key: str = "seq_index",
) -> TargetSplit:
    import pandas as pd

    frame = pd.read_csv(csv_path, na_values="").fillna(-99)
    return split_target_records(
        frame.to_dict(orient="records"),
        adapt_fraction=adapt_fraction,
        seed=seed,
        seq_key=seq_key,
    )


def sample_few_shot_records(
    records: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    seed: int,
    group_size: int = 8,
    label_key: str = "beam",
    sample_id_key: str = "sample_id",
    data_root: str | Path | None = None,
    num_classes: int = 64,
) -> FewShotSampling:
    requested = int(budget)
    if requested not in SUPPORTED_LABEL_BUDGETS:
        raise ValueError(f"label budget must be one of {list(SUPPORTED_LABEL_BUDGETS)}, got {budget}.")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    total = len(records)
    if requested == 0 or total == 0:
        return _few_shot_result(
            (),
            tuple(range(total)),
            records,
            requested,
            seed,
            group_size,
            None,
            sample_id_key,
            label_key=label_key,
            data_root=data_root,
            num_classes=num_classes,
        )
    if requested >= total:
        reason = "requested_budget_exceeds_available_target_adapt"
        return _few_shot_result(
            tuple(range(total)),
            (),
            records,
            requested,
            seed,
            group_size,
            reason,
            sample_id_key,
            label_key=label_key,
            data_root=data_root,
            num_classes=num_classes,
        )

    rng = np.random.default_rng(int(seed))
    group_to_indices: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        label = _resolve_beam_label(
            record,
            label_key=label_key,
            data_root=data_root,
            num_classes=num_classes,
        )
        group = _coarse_group(label, group_size)
        group_to_indices.setdefault(group, []).append(index)
    selected: list[int] = []
    for group in sorted(group_to_indices):
        if len(selected) >= requested:
            break
        choices = list(group_to_indices[group])
        rng.shuffle(choices)
        selected.append(int(choices[0]))
    if len(selected) < requested:
        remaining = [index for index in range(total) if index not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(int(index) for index in remaining[: requested - len(selected)])
    selected = sorted(selected[:requested])
    unlabeled = tuple(index for index in range(total) if index not in set(selected))
    return _few_shot_result(
        tuple(selected),
        unlabeled,
        records,
        requested,
        seed,
        group_size,
        None,
        sample_id_key,
        label_key=label_key,
        data_root=data_root,
        num_classes=num_classes,
    )


def _few_shot_result(
    labeled: tuple[int, ...],
    unlabeled: tuple[int, ...],
    records: Sequence[Mapping[str, Any]],
    requested_budget: int,
    seed: int,
    group_size: int,
    degrade_reason: str | None,
    sample_id_key: str,
    label_key: str = "beam",
    data_root: str | Path | None = None,
    num_classes: int = 64,
) -> FewShotSampling:
    labeled_samples = []
    for index in labeled:
        record = records[index]
        beam, source = _resolve_beam_label_with_source(
            record,
            label_key=label_key,
            data_root=data_root,
            num_classes=num_classes,
        )
        labeled_samples.append(
            {
                "index": int(index),
                "sample_id": _sample_id(record, index, sample_id_key=sample_id_key),
                "beam": beam,
                "coarse_group": None if beam is None else int(beam // group_size),
                "label_key": label_key,
                "label_source": source,
                "labeled": True,
            }
        )
    manifest = {
        "protocol": "coarse_group_stratified_few_shot",
        "requested_budget": int(requested_budget),
        "actual_labeled_count": len(labeled),
        "unlabeled_count": len(unlabeled),
        "seed": int(seed),
        "group_size": int(group_size),
        "degrade_reason": degrade_reason,
        "labeled_samples": labeled_samples,
    }
    return FewShotSampling(labeled_indices=labeled, unlabeled_indices=unlabeled, manifest=manifest)


def _target_split_metadata(
    *,
    seed: int,
    adapt_fraction: float,
    total: int,
    adapt_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    unit: str,
    adapt_units: tuple[Any, ...],
    test_units: tuple[Any, ...],
    sample_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "protocol": "target_adapt_test",
        "split_seed": int(seed),
        "adapt_fraction": float(adapt_fraction),
        "test_fraction": float(1.0 - adapt_fraction),
        "selection_unit": unit,
        "num_samples": int(total),
        "target_adapt_count": len(adapt_indices),
        "target_test_count": len(test_indices),
        "target_adapt_indices": [int(index) for index in adapt_indices],
        "target_test_indices": [int(index) for index in test_indices],
        "target_adapt_sample_ids": [sample_ids[index] for index in adapt_indices],
        "target_test_sample_ids": [sample_ids[index] for index in test_indices],
        "adapt_selection_units": list(adapt_units),
        "test_selection_units": list(test_units),
    }


def _normalize_scene_ids(scenes: Sequence[Any]) -> tuple[int, ...]:
    values = tuple(resolve_deepsense_scene(scene).scene_id for scene in scenes)
    if len(set(values)) != len(values):
        raise ValueError(f"LOSO scenes must be unique, got {values}.")
    return values


def _adapt_count(total: int, fraction: float) -> int:
    if total <= 1:
        return total
    count = int(round(total * float(fraction)))
    return max(1, min(total - 1, count))


def _sample_id(record: Mapping[str, Any], index: int, *, sample_id_key: str) -> str:
    value = record.get(sample_id_key)
    if value not in (None, "", -99, "-99"):
        return str(value)
    scene = record.get("scene_slug") or record.get("scene") or record.get("scene_id") or "target"
    return f"{scene}:{index}"


def _clean_group_value(value: Any) -> Any | None:
    if value in (None, "", -99, "-99"):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _beam_value(record: Mapping[str, Any]) -> int | None:
    for key in ("beam", "target_beam", "label", "true_beam"):
        value = record.get(key)
        if value in (None, "", -99, "-99"):
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        return int(value)
    return None


def _resolve_beam_label(
    record: Mapping[str, Any],
    *,
    label_key: str,
    data_root: str | Path | None,
    num_classes: int,
) -> int:
    value, _ = _resolve_beam_label_with_source(
        record,
        label_key=label_key,
        data_root=data_root,
        num_classes=num_classes,
    )
    if value is None:
        raise ValueError(f"Could not resolve beam label from key '{label_key}' in record keys {sorted(record)}.")
    return int(value)


def _resolve_beam_label_with_source(
    record: Mapping[str, Any],
    *,
    label_key: str,
    data_root: str | Path | None,
    num_classes: int,
) -> tuple[int | None, str | None]:
    label_candidates = _label_candidate_keys(str(label_key))
    for key in label_candidates:
        if key not in record:
            continue
        value = record.get(key)
        parsed = _parse_int_label(value)
        if parsed is not None:
            return parsed, key

    if label_key in record:
        value = record.get(label_key)
        parsed = _parse_int_label(value)
        if parsed is not None:
            return parsed, label_key
        if data_root is not None and _valid_path_value(value):
            return _beam_label_from_power_path(data_root, str(value), num_classes=num_classes), f"{label_key}:power_argmax"

    for key in ("beam", "target_beam", "label", "true_beam"):
        if key not in record:
            continue
        parsed = _parse_int_label(record.get(key))
        if parsed is not None:
            return parsed, key
    return None, None


def _label_candidate_keys(label_key: str) -> list[str]:
    candidates: list[str] = []
    if label_key.startswith("future_beam"):
        suffix = label_key[len("future_beam") :]
        candidates.append(f"future_beam_label{suffix}")
    if label_key.startswith("beam"):
        suffix = label_key[len("beam") :]
        candidates.append(f"beam_label{suffix}")
    candidates.append(f"{label_key}_label")
    return [key for index, key in enumerate(candidates) if key and key not in candidates[:index]]


def _parse_int_label(value: Any) -> int | None:
    if value in (None, "", -99, "-99"):
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if not text or text == "-99":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return None


def _valid_path_value(value: Any) -> bool:
    text = str(value).strip()
    return bool(text) and text != "-99"


def _beam_label_from_power_path(data_root: str | Path, rel_path: str, *, num_classes: int) -> int:
    path = joined_resource(data_root, rel_path)
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed to read beam power label from {path}: {exc}") from exc
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != int(num_classes):
        raise ValueError(
            f"Beam power label file {path} contains {vector.size} values; expected {int(num_classes)}."
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"Beam power label file {path} contains NaN or Inf values.")
    return int(np.argmax(vector))


def _coarse_group(value: Any, group_size: int) -> int:
    if value in (None, "", -99, "-99"):
        return -1
    if isinstance(value, (list, tuple)):
        value = value[0]
    return int(value) // int(group_size)


__all__ = [
    "DEFAULT_LOSO_SCENES",
    "DEFAULT_TARGET_ORDER",
    "SUPPORTED_LABEL_BUDGETS",
    "FewShotSampling",
    "LOSOFold",
    "TargetSplit",
    "default_loso_folds",
    "resolve_loso_fold",
    "sample_few_shot_records",
    "split_target_csv",
    "split_target_records",
]
