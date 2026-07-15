import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from kd_sensing.modalities import normalize_modalities
from kd_sensing.registries import DIFFICULTY_OPERATORS, import_default_difficulty_operators


ALLOWED_STAGES = ("train", "validation", "test", "evaluation", "benchmark")
ALLOWED_SPLITS = ("train", "validation", "test")
PSEUDO_MODALITY_HINTS = {
    "delayed_gps": "gps",
    "gps_noisy": "gps",
    "noisy_gps": "gps",
    "stale_gps": "gps",
    "image_hard": "image",
    "degraded_image": "image",
    "missing_image_modality": "image",
}
TARGET_SHIFT_KEYS = (
    "target_shift",
    "shift_target",
    "shift_targets",
    "move_target",
    "move_targets",
    "target_fields",
    "target_keys",
    "label_shift",
    "move_label",
    "split_shift",
)


@dataclass(frozen=True)
class DifficultyWarning:
    code: str
    message: str
    profile_id: str | None = None
    operator: str | None = None
    condition: str | None = None
    severity: float | None = None
    sample_count: int | None = None
    affected_count: int | None = None
    fallback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "profile_id": self.profile_id,
            "operator": self.operator,
            "condition": self.condition,
            "severity": self.severity,
            "sample_count": self.sample_count,
            "affected_count": self.affected_count,
            "fallback": self.fallback,
        }
        return {key: value for key, value in payload.items() if value not in (None, "")}


@dataclass(frozen=True)
class DifficultyOperatorConfig:
    type: str
    modality: str
    affected_modalities: tuple[str, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "modality": self.modality,
            "affected_modalities": list(self.affected_modalities),
            "params": _json_safe(self.params),
        }
        payload["digest"] = self.digest or stable_digest(payload)
        return payload


@dataclass(frozen=True)
class DifficultyProfile:
    id: str
    operators: tuple[DifficultyOperatorConfig, ...]
    stages: tuple[str, ...]
    splits: tuple[str, ...]
    condition: str
    severity: float
    seed: int
    fallback: str
    affected_modalities: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "operators": [operator.to_dict() for operator in self.operators],
            "stages": list(self.stages),
            "splits": list(self.splits),
            "condition": self.condition,
            "severity": float(self.severity),
            "seed": int(self.seed),
            "fallback": self.fallback,
            "affected_modalities": list(self.affected_modalities),
            "metadata": _json_safe(self.metadata),
        }
        payload["digest"] = self.digest or stable_digest(payload)
        return payload


@dataclass(frozen=True)
class DifficultyContext:
    stage: str
    split: str | None = None
    seed: int | None = None
    epoch: int | None = None
    step: int | None = None
    sample_ids: tuple[str, ...] = ()

    def normalized_stage(self) -> str:
        return _normalize_stage(self.stage)

    def normalized_split(self) -> str:
        return _normalize_split(self.split or self.stage)

    def derived_seed(self, profile: DifficultyProfile, operator: DifficultyOperatorConfig) -> int:
        base_seed = int(self.seed if self.seed is not None else profile.seed)
        return stable_int_seed(
            base_seed,
            profile.digest,
            operator.digest,
            self.normalized_stage(),
            self.normalized_split(),
            self.epoch,
            self.step,
            self.sample_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.normalized_stage(),
            "split": self.normalized_split(),
            "seed": self.seed,
            "epoch": self.epoch,
            "step": self.step,
            "sample_ids": list(self.sample_ids),
        }


@dataclass(frozen=True)
class DifficultyOperatorOutcome:
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[DifficultyWarning, ...] = ()


@dataclass(frozen=True)
class DifficultyResult:
    batch: dict[str, Any]
    metadata: Mapping[str, Any]
    warnings: tuple[DifficultyWarning, ...] = ()


def normalize_config_difficulty(cfg: dict[str, Any]) -> list[DifficultyProfile]:
    raw_profiles = _raw_profiles_from_config(cfg)
    if not raw_profiles:
        return []
    profiles = normalize_difficulty_profiles(raw_profiles, default_seed=cfg.get("experiment", {}).get("seed", 0))
    cfg["difficulty"] = {
        "enabled": True,
        "profiles": [profile.to_dict() for profile in profiles],
    }
    return profiles


def difficulty_runtime_metadata(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    profiles = profiles_from_resolved_config(cfg)
    if not profiles:
        return None
    return {
        "enabled": True,
        "profiles": [profile.to_dict() for profile in profiles],
        "warnings_summary": [],
    }


def profiles_from_resolved_config(cfg: Mapping[str, Any]) -> list[DifficultyProfile]:
    if not isinstance(cfg, Mapping):
        return []
    raw = _raw_profiles_from_config(cfg)
    if not raw:
        return []
    return normalize_difficulty_profiles(raw, default_seed=cfg.get("experiment", {}).get("seed", 0))


def normalize_difficulty_profiles(
    profiles: Iterable[Any],
    *,
    default_seed: Any = 0,
    default_stage: str | None = None,
    default_split: str | None = None,
) -> list[DifficultyProfile]:
    import_default_difficulty_operators()
    resolved: list[DifficultyProfile] = []
    for index, raw in enumerate(profiles):
        profile = _normalize_profile(
            raw,
            index=index,
            default_seed=default_seed,
            default_stage=default_stage,
            default_split=default_split,
        )
        resolved.append(profile)
    return resolved


def select_profiles_for_context(
    profiles: Iterable[DifficultyProfile],
    context: DifficultyContext,
) -> list[DifficultyProfile]:
    stage = context.normalized_stage()
    split = context.normalized_split()
    selected = []
    for profile in profiles:
        if not _stage_matches(profile.stages, stage):
            continue
        if profile.splits and split not in profile.splits:
            continue
        selected.append(profile)
    return selected


def stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def stable_int_seed(*parts: Any) -> int:
    encoded = json.dumps(_json_safe(parts), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16) % (2**31)


def _raw_profiles_from_config(cfg: Mapping[str, Any]) -> list[Any]:
    raw: list[Any] = []
    difficulty = cfg.get("difficulty")
    if difficulty not in (None, False):
        raw.extend(_profiles_from_container(difficulty, default_stage=None))
    random_dropout = _random_modality_dropout_profile(cfg)
    if random_dropout is not None:
        raw.append(random_dropout)
    temporal_missing = _temporal_missing_profile(cfg)
    if temporal_missing is not None:
        raw.append(temporal_missing)
    data_difficulty = cfg.get("data", {}).get("difficulty") if isinstance(cfg.get("data"), Mapping) else None
    if data_difficulty not in (None, False):
        raw.extend(_profiles_from_container(data_difficulty, default_stage="train"))
    evaluation_difficulty = cfg.get("evaluation", {}).get("difficulty") if isinstance(cfg.get("evaluation"), Mapping) else None
    if evaluation_difficulty not in (None, False):
        raw.extend(_profiles_from_container(evaluation_difficulty, default_stage="evaluation"))
    return raw


def _temporal_missing_profile(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    temporal = cfg.get("temporal_missing") if isinstance(cfg, Mapping) else None
    if not isinstance(temporal, Mapping):
        return None
    mode = str(temporal.get("mode", temporal.get("temporal_missing_mode", "none"))).strip().lower()
    if str(temporal.get("mask_sampler", "")).strip().lower() == "stratified_modality_temporal":
        mode = "stratified_modality_temporal"
    prob = float(temporal.get("prob", temporal.get("temporal_missing_prob", 0.0)) or 0.0)
    if not (bool(temporal.get("enabled", False)) or mode != "none" or prob > 0.0) or mode == "none":
        return None
    apply = str(temporal.get("apply", temporal.get("temporal_missing_apply", "train"))).strip().lower()
    if apply == "train":
        stages = ["train"]
    elif apply == "eval":
        stages = ["evaluation", "benchmark"]
    elif apply == "both":
        stages = ["train", "evaluation", "benchmark"]
    else:
        raise ValueError("temporal_missing.apply must be one of train, eval, both.")
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    primary = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), Mapping) else {}
    modalities = temporal.get(
        "modalities",
        primary.get("modalities", model_cfg.get("modalities", ["image", "radar", "gps", "lidar"])),
    )
    return {
        "id": str(temporal.get("id", "temporal_missing")),
        "stage": stages,
        "split": temporal.get("split", ["train", "validation", "test"]),
        "condition": str(temporal.get("condition", f"temporal_missing_{mode}")),
        "severity": prob,
        "seed": temporal.get("seed", temporal.get("temporal_missing_seed", cfg.get("experiment", {}).get("seed", 0))),
        "fallback": temporal.get("fallback", "zero_fill"),
        "affected_modalities": modalities,
        "metadata": {
            "source": "temporal_missing",
            "history_window": temporal.get("history_window"),
            "prediction_window": temporal.get("prediction_window"),
        },
        "operators": [
            {
                "type": "temporal_missing",
                "modality": str(_as_list(modalities)[0]),
                "affected_modalities": modalities,
                "mode": mode,
                "prob": prob,
                "block_len": temporal.get("block_len", temporal.get("temporal_missing_block_len", 1)),
                "ensure_at_least_one_frame": temporal.get("ensure_at_least_one_frame", True),
                "ensure_at_least_one_cell": temporal.get("ensure_at_least_one_cell", True),
                "ensure_at_least_one_modality": temporal.get("ensure_at_least_one_modality", True),
                "preserve_unmasked_for_superset": temporal.get("preserve_unmasked_for_superset", False),
                "train_missing_drop_counts": temporal.get("train_missing_drop_counts", "0,1,2,3"),
                "train_temporal_missing_rates": temporal.get("train_temporal_missing_rates", "0.0,0.2,0.4,0.6,0.8"),
                "train_temporal_missing_types": temporal.get(
                    "train_temporal_missing_types",
                    "modality_level,frame_level,modality_frame,block",
                ),
                "ensure_at_least_one_modality_per_frame": temporal.get(
                    "ensure_at_least_one_modality_per_frame",
                    False,
                ),
            }
        ],
    }


def _random_modality_dropout_profile(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    training = cfg.get("training") if isinstance(cfg.get("training"), Mapping) else {}
    raw = training.get("random_modality_dropout") if isinstance(training, Mapping) else None
    if raw in (None, False):
        raw = cfg.get("random_modality_dropout")
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return None
    modalities = raw.get("modalities", raw.get("affected_modalities", ["image", "radar", "gps", "lidar"]))
    return {
        "id": str(raw.get("id", "random_modality_dropout_train")),
        "stage": "train",
        "split": "train",
        "condition": str(raw.get("condition", "random_modality_dropout")),
        "severity": 1.0,
        "seed": raw.get("seed", cfg.get("experiment", {}).get("seed", 0)),
        "fallback": raw.get("fallback", "zero_fill"),
        "affected_modalities": modalities,
        "metadata": {
            "source": "training.random_modality_dropout",
            "mode": raw.get("mode", "random_nonempty_subset"),
        },
        "operators": [
            {
                "type": "random_modality_dropout",
                "modality": str(_as_list(modalities)[0]),
                "affected_modalities": modalities,
                "mode": raw.get("mode", "random_nonempty_subset"),
                "keep_prob": raw.get("keep_prob", 0.75),
                "pattern_probs": raw.get("pattern_probs", raw.get("patterns")),
                "ensure_at_least_one_modality": raw.get("ensure_at_least_one_modality", True),
            }
        ],
    }


def _profiles_from_container(container: Any, *, default_stage: str | None) -> list[Any]:
    if isinstance(container, list):
        profiles = container
    elif isinstance(container, Mapping):
        if container.get("enabled") is False:
            return []
        profiles = container.get("profiles", container.get("profile", container.get("sweep")))
        if profiles is None and "operators" in container:
            profiles = [container]
        elif isinstance(profiles, Mapping):
            profiles = [{**dict(value), "id": key} if isinstance(value, Mapping) else value for key, value in profiles.items()]
    else:
        raise ValueError("difficulty must be a mapping or list of profiles.")
    if profiles is None:
        return []
    if not isinstance(profiles, list):
        profiles = [profiles]
    if default_stage is None:
        return list(profiles)
    patched = []
    for profile in profiles:
        if isinstance(profile, Mapping) and "stage" not in profile and "stages" not in profile:
            patched.append({**dict(profile), "stage": default_stage})
        else:
            patched.append(profile)
    return patched


def _normalize_profile(
    raw: Any,
    *,
    index: int,
    default_seed: Any,
    default_stage: str | None,
    default_split: str | None,
) -> DifficultyProfile:
    if not isinstance(raw, Mapping):
        raise ValueError(f"difficulty profile {index} must be a mapping.")
    from kd_sensing.data.difficulty.presets import (
        expand_missing_modality_stress_profile,
        is_missing_modality_stress_profile,
    )

    if is_missing_modality_stress_profile(raw):
        raw = expand_missing_modality_stress_profile(
            raw,
            profile_id=str(raw.get("id", raw.get("name", f"profile_{index}"))).strip() or f"profile_{index}",
        )
    _reject_target_shift(raw, path=f"difficulty.profiles[{index}]")
    profile_id = str(raw.get("id", raw.get("name", f"profile_{index}"))).strip()
    if not profile_id:
        raise ValueError(f"difficulty profile {index} must have a non-empty id.")
    stages = tuple(_normalize_stage(item) for item in _as_list(raw.get("stages", raw.get("stage", default_stage or "train"))))
    if not stages:
        raise ValueError(f"difficulty profile '{profile_id}' must select at least one stage.")
    splits = tuple(_normalize_split(item) for item in _as_list(raw.get("splits", raw.get("split", default_split or ())))
                   if str(item).strip() not in {"", "*", "all"})
    condition = str(raw.get("condition", profile_id)).strip() or profile_id
    severity = _severity(raw.get("severity", raw.get("severities", 0.0)), profile_id=profile_id)
    seed = int(raw.get("seed", default_seed or 0))
    fallback = str(raw.get("fallback", "identity")).strip() or "identity"
    operators_raw = raw.get("operators", raw.get("operator"))
    if operators_raw is None:
        raise ValueError(f"difficulty profile '{profile_id}' must define a non-empty operators list.")
    operators_raw = _as_list(operators_raw)
    if not operators_raw:
        raise ValueError(f"difficulty profile '{profile_id}' must define a non-empty operators list.")
    operators = tuple(_normalize_operator(item, profile_id=profile_id, index=i) for i, item in enumerate(operators_raw))
    profile_modalities = raw.get("affected_modalities", raw.get("modalities"))
    affected = (
        _normalize_affected_modalities(profile_modalities, path=f"difficulty profile '{profile_id}'")
        if profile_modalities is not None
        else tuple(dict.fromkeys(modality for operator in operators for modality in operator.affected_modalities))
    )
    metadata = raw.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"difficulty profile '{profile_id}'.metadata must be a mapping.")
    payload = {
        "id": profile_id,
        "operators": [operator.to_dict() for operator in operators],
        "stages": list(stages),
        "splits": list(splits),
        "condition": condition,
        "severity": float(severity),
        "seed": seed,
        "fallback": fallback,
        "affected_modalities": list(affected),
        "metadata": _json_safe(metadata),
    }
    return DifficultyProfile(
        id=profile_id,
        operators=operators,
        stages=stages,
        splits=splits,
        condition=condition,
        severity=float(severity),
        seed=seed,
        fallback=fallback,
        affected_modalities=affected,
        metadata=dict(metadata),
        digest=stable_digest(payload),
    )


def _normalize_operator(raw: Any, *, profile_id: str, index: int) -> DifficultyOperatorConfig:
    if isinstance(raw, str):
        item: dict[str, Any] = {"type": raw}
    elif isinstance(raw, Mapping):
        item = dict(raw)
    else:
        raise ValueError(f"difficulty profile '{profile_id}' operator {index} must be a mapping or string.")
    _reject_target_shift(item, path=f"difficulty profile '{profile_id}' operator {index}")
    operator_type = str(item.get("type", item.get("name", ""))).strip()
    if not operator_type:
        raise ValueError(f"difficulty profile '{profile_id}' operator {index} must define type.")
    DIFFICULTY_OPERATORS.get(operator_type)
    modality = _infer_operator_modality(operator_type, item)
    affected = _normalize_affected_modalities(
        item.get("affected_modalities", item.get("affected_modality", modality)),
        path=f"difficulty profile '{profile_id}' operator '{operator_type}'",
    )
    nested_params = item.get("params")
    params = dict(nested_params) if isinstance(nested_params, Mapping) else {}
    params.update({
        key: value
        for key, value in item.items()
        if key not in {"type", "name", "affected_modalities", "affected_modality", "params", "digest"}
    })
    params.setdefault("modality", modality)
    payload = {
        "type": operator_type,
        "modality": modality,
        "affected_modalities": list(affected),
        "params": _json_safe(params),
    }
    return DifficultyOperatorConfig(
        type=operator_type,
        modality=modality,
        affected_modalities=affected,
        params=params,
        digest=stable_digest(payload),
    )


def _infer_operator_modality(operator_type: str, item: Mapping[str, Any]) -> str:
    raw = item.get("modality")
    if raw is None:
        if operator_type.startswith("image_"):
            raw = "image"
        elif operator_type.startswith("gps_") or operator_type in {
            "temporal_delay",
            "sampling_rate_mismatch",
            "scenario_c",
            "scenario_c_async_position_feedback",
        }:
            raw = "gps"
        else:
            raw = item.get("affected_modality", "gps")
    modalities = _normalize_affected_modalities(raw, path=f"difficulty operator '{operator_type}'.modality")
    if len(modalities) != 1:
        raise ValueError(f"difficulty operator '{operator_type}' must select one primary modality.")
    return modalities[0]


def _normalize_affected_modalities(raw: Any, *, path: str) -> tuple[str, ...]:
    values = _as_list(raw)
    names = []
    for value in values:
        name = str(value).strip()
        if name in PSEUDO_MODALITY_HINTS:
            canonical = PSEUDO_MODALITY_HINTS[name]
            raise ValueError(f"{path} uses pseudo modality '{name}'. Use canonical modality '{canonical}' in difficulty profiles.")
        names.append(name)
    try:
        return tuple(normalize_modalities(tuple(names), context=path))
    except ValueError as exc:
        raise ValueError(f"{path} must use canonical modality names; {exc}") from exc


def _normalize_stage(value: Any) -> str:
    stage = str(value).strip().lower()
    if stage == "val":
        stage = "validation"
    if stage not in ALLOWED_STAGES:
        allowed = ", ".join(ALLOWED_STAGES)
        raise ValueError(f"Illegal difficulty stage '{value}'. Allowed stages: {allowed}.")
    return stage


def _normalize_split(value: Any) -> str:
    split = str(value).strip().lower()
    if split in {"", "*", "all", "evaluation", "benchmark"}:
        return "test"
    if split == "val":
        split = "validation"
    if split not in ALLOWED_SPLITS:
        allowed = ", ".join(ALLOWED_SPLITS)
        raise ValueError(f"Illegal difficulty split '{value}'. Allowed splits: {allowed}.")
    return split


def _stage_matches(profile_stages: tuple[str, ...], stage: str) -> bool:
    if stage in profile_stages:
        return True
    return stage in {"validation", "test", "evaluation"} and "evaluation" in profile_stages


def _severity(raw: Any, *, profile_id: str) -> float:
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError(f"difficulty profile '{profile_id}' severity list must not be empty.")
        raw = raw[0]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"difficulty profile '{profile_id}' severity must be numeric.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"difficulty profile '{profile_id}' severity must be a finite non-negative number.")
    return value


def _reject_target_shift(mapping: Mapping[str, Any], *, path: str) -> None:
    for key in TARGET_SHIFT_KEYS:
        if key not in mapping:
            continue
        value = mapping[key]
        if value in (None, False, "", [], {}, ()):
            continue
        raise ValueError(
            f"{path}.{key} attempts to move target fields. "
            "Difficulty pipeline may only perturb input modalities and input reliability metadata."
        )


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
