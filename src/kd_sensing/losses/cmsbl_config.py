import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_FIELDS = {"enabled", "aux_schedule", "capacity_reference", "capacity_gap", "hard_mask", "diagnostics"}
_CAPACITY_FIELDS = {
    "enabled",
    "ema_momentum",
    "warmup_epochs",
    "update_interval",
    "alpha",
    "gamma",
    "min_weight",
    "max_weight",
    "apply_to",
    "eps",
}
_HARD_MASK_FIELDS = {
    "enabled",
    "ema_momentum",
    "warmup_epochs",
    "update_interval",
    "min_count",
    "gamma",
    "min_weight",
    "max_weight",
    "normalize_mean_to_one",
    "full_mask_min_weight",
    "eps",
}


def resolve_cmsbl_config(cfg: Mapping[str, Any], bcacl: Mapping[str, Any]) -> dict[str, Any]:
    raw = cfg.get("cmsbl", {})
    if raw in (None, False):
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("cmsbl must be a mapping.")
    if not bool(raw.get("enabled", False)):
        return {"enabled": False}
    _reject_unknown(raw, _FIELDS, "cmsbl")
    if not bcacl.get("enabled") or bcacl.get("stage") != "aux_joint":
        raise ValueError("Enabled CMSBL requires BCACL U2 aux_joint training.")
    modalities = tuple(str(value) for value in bcacl.get("modalities", ()))
    if modalities != ("image", "radar", "gps", "lidar"):
        raise ValueError("CMSBL requires canonical modalities [image, radar, gps, lidar].")

    auxiliary = _mapping(raw.get("aux_schedule"), "cmsbl.aux_schedule")
    _reject_unknown(auxiliary, {"enabled", "private", "shared"}, "cmsbl.aux_schedule")
    reference = _mapping(raw.get("capacity_reference"), "cmsbl.capacity_reference")
    _reject_unknown(reference, {"stats_path", "source_split"}, "cmsbl.capacity_reference")
    capacity = _mapping(raw.get("capacity_gap"), "cmsbl.capacity_gap")
    _reject_unknown(capacity, _CAPACITY_FIELDS, "cmsbl.capacity_gap")
    hard = _mapping(raw.get("hard_mask"), "cmsbl.hard_mask")
    _reject_unknown(hard, _HARD_MASK_FIELDS, "cmsbl.hard_mask")
    diagnostics = _mapping(raw.get("diagnostics"), "cmsbl.diagnostics")
    _reject_unknown(diagnostics, {"enabled"}, "cmsbl.diagnostics")

    capacity_enabled = bool(capacity.get("enabled", False))
    stats_path = reference.get("stats_path")
    if capacity_enabled and (not isinstance(stats_path, str) or not stats_path.strip()):
        raise ValueError("Enabled capacity_gap requires cmsbl.capacity_reference.stats_path.")
    apply_to = tuple(str(value).strip().lower() for value in capacity.get("apply_to", ("private", "shared")))
    if not apply_to or len(set(apply_to)) != len(apply_to) or set(apply_to) - {"private", "shared"}:
        raise ValueError("cmsbl.capacity_gap.apply_to must be a unique non-empty subset of private/shared.")
    capacity_min = _finite(capacity.get("min_weight", 1.0), "cmsbl.capacity_gap.min_weight")
    capacity_max = _finite(capacity.get("max_weight", 2.0), "cmsbl.capacity_gap.max_weight")
    if not 0.0 < capacity_min <= 1.0 <= capacity_max:
        raise ValueError("cmsbl.capacity_gap bounds must satisfy 0 < min_weight <= 1 <= max_weight.")
    mask_min = _finite(hard.get("min_weight", 0.75), "cmsbl.hard_mask.min_weight")
    mask_max = _finite(hard.get("max_weight", 1.75), "cmsbl.hard_mask.max_weight")
    full_min = _finite(hard.get("full_mask_min_weight", 1.0), "cmsbl.hard_mask.full_mask_min_weight")
    if not 0.0 < mask_min <= 1.0 <= mask_max or not mask_min <= full_min <= mask_max:
        raise ValueError("cmsbl.hard_mask bounds must contain 1 and full_mask_min_weight.")

    return {
        "enabled": True,
        "modalities": modalities,
        "aux_schedule": {
            "enabled": bool(auxiliary.get("enabled", False)),
            "private": _schedule(auxiliary.get("private"), "cmsbl.aux_schedule.private"),
            "shared": _schedule(auxiliary.get("shared"), "cmsbl.aux_schedule.shared"),
        },
        "capacity_reference": {
            "mode": "standalone_checkpoint",
            "metric": "top1",
            "stats_path": str(Path(stats_path).expanduser()) if isinstance(stats_path, str) and stats_path.strip() else None,
            "source_split": str(reference.get("source_split", "inner_train")).strip().lower(),
        },
        "capacity_gap": {
            "enabled": capacity_enabled,
            "ema_momentum": _momentum(capacity.get("ema_momentum", 0.9), "cmsbl.capacity_gap.ema_momentum"),
            "warmup_epochs": _non_negative_int(capacity.get("warmup_epochs", 3), "cmsbl.capacity_gap.warmup_epochs"),
            "update_interval": _positive_int(capacity.get("update_interval", 1), "cmsbl.capacity_gap.update_interval"),
            "alpha": _non_negative(capacity.get("alpha", 1.0), "cmsbl.capacity_gap.alpha"),
            "gamma": _positive(capacity.get("gamma", 1.0), "cmsbl.capacity_gap.gamma"),
            "min_weight": capacity_min,
            "max_weight": capacity_max,
            "apply_to": apply_to,
            "eps": _positive(capacity.get("eps", 1.0e-6), "cmsbl.capacity_gap.eps"),
        },
        "hard_mask": {
            "enabled": bool(hard.get("enabled", False)),
            "ema_momentum": _momentum(hard.get("ema_momentum", 0.9), "cmsbl.hard_mask.ema_momentum"),
            "warmup_epochs": _non_negative_int(hard.get("warmup_epochs", 3), "cmsbl.hard_mask.warmup_epochs"),
            "update_interval": _positive_int(hard.get("update_interval", 1), "cmsbl.hard_mask.update_interval"),
            "min_count": _positive_int(hard.get("min_count", 8), "cmsbl.hard_mask.min_count"),
            "gamma": _positive(hard.get("gamma", 0.5), "cmsbl.hard_mask.gamma"),
            "min_weight": mask_min,
            "max_weight": mask_max,
            "normalize_mean_to_one": bool(hard.get("normalize_mean_to_one", True)),
            "full_mask_min_weight": full_min,
            "eps": _positive(hard.get("eps", 1.0e-6), "cmsbl.hard_mask.eps"),
        },
        "diagnostics": {"enabled": bool(diagnostics.get("enabled", True))},
    }


def _schedule(value: Any, field: str) -> dict[str, Any]:
    raw = _mapping(value, field)
    _reject_unknown(raw, {"start_weight", "end_weight", "start_epoch", "end_epoch"}, field)
    start_epoch = _positive_int(raw.get("start_epoch", 1), f"{field}.start_epoch")
    end_epoch = _positive_int(raw.get("end_epoch", start_epoch), f"{field}.end_epoch")
    if end_epoch < start_epoch:
        raise ValueError(f"{field}.end_epoch must be >= start_epoch.")
    return {
        "start_weight": _non_negative(raw.get("start_weight", 1.0), f"{field}.start_weight"),
        "end_weight": _non_negative(raw.get("end_weight", 1.0), f"{field}.end_weight"),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
    }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field} does not support fields: {unknown}.")


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _positive(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive.")
    return result


def _non_negative(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative.")
    return result


def _positive_int(value: Any, field: str) -> int:
    result = int(value)
    if isinstance(value, bool) or result <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return result


def _non_negative_int(value: Any, field: str) -> int:
    result = int(value)
    if isinstance(value, bool) or result < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return result


def _momentum(value: Any, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{field} must be in [0, 1).")
    return result


__all__ = ["resolve_cmsbl_config"]
