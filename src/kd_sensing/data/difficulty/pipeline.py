from copy import deepcopy
from typing import Any, Mapping

import torch

from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
    DifficultyOperatorOutcome,
    DifficultyProfile,
    DifficultyResult,
    DifficultyWarning,
    difficulty_runtime_metadata,
    profiles_from_resolved_config,
    select_profiles_for_context,
)
from kd_sensing.registries import DIFFICULTY_OPERATORS, import_default_difficulty_operators


PROTECTED_BATCH_KEYS = (
    "target_beam",
    "beam_power",
    "power",
    "target_beam_distribution",
    "target_beam_distribution_mask",
    "soft_target",
    "soft_targets",
    "input_beam",
    "occlusion_label",
    "occlusion_valid",
    "position_target",
    "position_valid",
    "los_label",
    "link_quality",
    "sample_id",
    "sample_ids",
    "history_indices",
    "history_timestamps",
    "target_index",
    "target_timestamp",
    "split",
    "split_metadata",
)
PROTECTED_METADATA_KEYS = (
    "sample_id",
    "sample_ids",
    "split",
    "split_name",
    "dataset_split",
    "split_metadata",
    "scene",
    "scene_id",
    "dataset_family",
)


def apply_configured_difficulty(
    batch: Mapping[str, Any],
    cfg: Mapping[str, Any],
    context: DifficultyContext,
) -> DifficultyResult:
    profiles = profiles_from_resolved_config(cfg)
    selected = select_profiles_for_context(profiles, context)
    if not selected:
        clean = dict(batch)
        return DifficultyResult(batch=clean, metadata={"enabled": False, "state": "clean"}, warnings=())
    current = dict(batch)
    warnings: list[DifficultyWarning] = []
    metadata: list[Mapping[str, Any]] = []
    for profile in selected:
        result = apply_difficulty_pipeline(current, profile, context)
        current = result.batch
        warnings.extend(result.warnings)
        metadata.append(result.metadata)
    return DifficultyResult(
        batch=current,
        metadata={"enabled": True, "profiles": metadata},
        warnings=tuple(warnings),
    )


def apply_difficulty_pipeline(
    batch: Mapping[str, Any],
    profile: DifficultyProfile,
    context: DifficultyContext,
) -> DifficultyResult:
    import_default_difficulty_operators()
    result = _clone_value(batch)
    protected_before = _protected_snapshot(result)
    operator_metadata: list[Mapping[str, Any]] = []
    warnings: list[DifficultyWarning] = []
    for operator_cfg in profile.operators:
        operator = DIFFICULTY_OPERATORS.build({"type": operator_cfg.type, **dict(operator_cfg.params)})
        outcome = operator(result, config=operator_cfg, profile=profile, context=context)
        if outcome is None:
            outcome = DifficultyOperatorOutcome()
        if not isinstance(outcome, DifficultyOperatorOutcome):
            raise TypeError(
                f"Difficulty operator '{operator_cfg.type}' must return DifficultyOperatorOutcome or None, "
                f"got {type(outcome).__name__}."
            )
        _assert_target_preserved(protected_before, _protected_snapshot(result), operator=operator_cfg.type)
        operator_metadata.append(
            {
                "type": operator_cfg.type,
                "registry_name": operator_cfg.type,
                "modality": operator_cfg.modality,
                "affected_modalities": list(operator_cfg.affected_modalities),
                "digest": operator_cfg.digest,
                "metadata": dict(outcome.metadata),
            }
        )
        warnings.extend(outcome.warnings)
    metadata = {
        "profile": profile.to_dict(),
        "profile_id": profile.id,
        "profile_digest": profile.digest,
        "condition": profile.condition,
        "severity": float(profile.severity),
        "stage": context.normalized_stage(),
        "split": context.normalized_split(),
        "seed": int(context.seed if context.seed is not None else profile.seed),
        "operators": operator_metadata,
        "warnings": [warning.to_dict() for warning in warnings],
        "replay": context.to_dict(),
    }
    _attach_difficulty_metadata(result, metadata)
    return DifficultyResult(batch=result, metadata=metadata, warnings=tuple(warnings))


def assert_target_preserved(before: Mapping[str, Any], after: Mapping[str, Any], *, operator: str = "difficulty") -> None:
    _assert_target_preserved(_protected_snapshot(before), _protected_snapshot(after), operator=operator)


def runtime_difficulty_metadata(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    return difficulty_runtime_metadata(cfg)


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return deepcopy(value)


def _protected_snapshot(batch: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in PROTECTED_BATCH_KEYS:
        if key in batch:
            snapshot[key] = _clone_value(batch[key])
    metadata = batch.get("metadata")
    if isinstance(metadata, Mapping):
        protected_metadata = {
            key: _clone_value(metadata[key])
            for key in PROTECTED_METADATA_KEYS
            if key in metadata
        }
        if protected_metadata:
            snapshot["metadata"] = protected_metadata
    return snapshot


def _assert_target_preserved(before: Mapping[str, Any], after: Mapping[str, Any], *, operator: str) -> None:
    if set(before) != set(after):
        raise RuntimeError(
            f"Difficulty operator '{operator}' changed protected target/sample metadata keys. "
            "Difficulty may only alter input modalities and input reliability metadata."
        )
    for key, value in before.items():
        if not _values_equal(value, after[key]):
            raise RuntimeError(
                f"Difficulty operator '{operator}' changed protected field '{key}'. "
                "Difficulty may only alter input modalities and input reliability metadata."
            )


def _values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right))
    return left == right


def _attach_difficulty_metadata(batch: dict[str, Any], metadata: Mapping[str, Any]) -> None:
    batch["difficulty"] = dict(metadata)
    current = batch.get("metadata")
    if isinstance(current, Mapping):
        enriched = dict(current)
        profiles = list(enriched.get("difficulty_profiles", []))
        profiles.append(dict(metadata))
        enriched["difficulty_profiles"] = profiles
        batch["metadata"] = enriched
