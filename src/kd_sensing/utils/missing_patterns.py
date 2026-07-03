from itertools import combinations
from typing import Any

import torch


DEFAULT_MODALITIES = ["gps", "image", "radar", "lidar"]
MODALITY_ALIASES = {
    "gps": "gps",
    "gnss": "gps",
    "image": "image",
    "img": "image",
    "rgb": "image",
    "camera": "image",
    "vision": "image",
    "radar": "radar",
    "rad": "radar",
    "ra": "radar",
    "radar_ra": "radar",
    "radar_da": "radar",
    "lidar": "lidar",
    "laser": "lidar",
}
SINGLE_MODALITY_PATTERNS = ["gps_only", "image_only", "radar_only", "lidar_only"]
WEAK_SINGLE_MODALITY_PATTERNS = ["radar_only", "lidar_only"]
SENSING_ONLY_PATTERNS = ["image_only", "radar_only", "lidar_only", "missing_gps", "non_gps_only"]


def get_default_missing_patterns(modalities: list[str]) -> dict[str, list[int]]:
    return list_standard_missing_patterns(modalities, include_avg=False)


def list_standard_missing_patterns(
    modality_names: list[str] | tuple[str, ...] | None = None,
    *,
    include_avg: bool = False,
) -> dict[str, list[int] | None]:
    names = _normalize_modalities(modality_names or DEFAULT_MODALITIES)
    patterns: dict[str, list[int]] = {"full": [1] * len(names)}
    for name in names:
        _add_pattern(patterns, f"missing_{name}", _mask_for(names, missing={name}))
    if "gps" in names and any(name != "gps" for name in names):
        _add_pattern(patterns, "non_gps_only", _mask_for(names, missing={"gps"}))
    for name in names:
        _add_pattern(patterns, f"{name}_only", _mask_for(names, available={name}))
    for left, right in combinations(names, 2):
        _add_pattern(patterns, f"missing_{left}_{right}", _mask_for(names, missing={left, right}))
    if include_avg:
        return {**patterns, "avg_missing": None}
    return patterns


def resolve_missing_patterns(patterns: str | list[str] | tuple[str, ...] | None, modalities: list[str]) -> dict[str, list[int]]:
    if patterns is None or patterns == "default" or (not isinstance(patterns, str) and list(patterns) == ["default"]):
        return get_default_missing_patterns(modalities)
    if isinstance(patterns, str):
        names = [item for item in patterns.replace(",", " ").split() if item]
    else:
        names = [str(item) for item in patterns]
    resolved: dict[str, list[int]] = {}
    for name in names:
        canonical = canonical_missing_pattern_name(name)
        if canonical.startswith("random_") or canonical == "avg_missing":
            continue
        resolved[canonical] = get_missing_pattern_mask(canonical, modalities)
    return resolved


def get_missing_pattern_mask(pattern_name: str, modality_names: list[str] | tuple[str, ...] | None = None) -> list[int]:
    names = _normalize_modalities(modality_names or DEFAULT_MODALITIES)
    name = canonical_missing_pattern_name(pattern_name)
    if name == "avg_missing":
        raise ValueError("avg_missing is an aggregate over missing patterns, not a direct modality mask.")
    if name == "full":
        return [1] * len(names)
    if name == "non_gps_only":
        if "gps" not in names:
            raise ValueError(f"missing pattern '{name}' requires gps in modalities {list(names)}.")
        return _mask_for(names, missing={"gps"})
    if name.endswith("_only"):
        modality = _normalize_modality(name.removesuffix("_only"))
        if modality not in names:
            raise ValueError(f"missing pattern '{name}' references unavailable modality '{modality}' in {list(names)}.")
        return _mask_for(names, available={modality})
    if name.startswith("missing_"):
        missing = {_normalize_modality(item) for item in name.removeprefix("missing_").split("_") if item}
        unknown = sorted(missing - set(names))
        if unknown:
            raise ValueError(f"missing pattern '{name}' references unavailable modalities {unknown} in {list(names)}.")
        return _mask_for(names, missing=missing)
    raise ValueError(f"Unknown missing pattern '{pattern_name}' for modalities {list(names)}.")


def get_missing_pattern_name(
    available_mask: torch.Tensor | list[Any] | tuple[Any, ...],
    modality_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    names = _normalize_modalities(modality_names or DEFAULT_MODALITIES)
    mask = torch.as_tensor(available_mask, dtype=torch.bool).flatten()
    if int(mask.numel()) != len(names):
        raise ValueError(f"available_mask must have {len(names)} values, got {int(mask.numel())}.")
    available = [name for name, keep in zip(names, mask.tolist()) if bool(keep)]
    if len(available) == len(names):
        return "full"
    if len(available) == 1:
        return f"{available[0]}_only"
    missing = [name for name, keep in zip(names, mask.tolist()) if not bool(keep)]
    if missing == ["gps"]:
        return "missing_gps"
    if len(missing) == 1:
        return f"missing_{missing[0]}"
    return "custom_" + "".join("1" if bool(item) else "0" for item in mask.tolist())


def canonical_missing_pattern_name(pattern_name: str) -> str:
    name = str(pattern_name).strip().lower().replace("-", "_")
    if name in {"full", "avg_missing", "non_gps_only", "missing_one_random", "only_one_random"} or name.startswith("random_"):
        return name
    if name.startswith("only_"):
        name = f"{name.removeprefix('only_')}_only"
    if name.endswith("_only"):
        return f"{_normalize_modality(name.removesuffix('_only'))}_only"
    if name.startswith("missing_"):
        suffix = name.removeprefix("missing_")
        if suffix:
            return "missing_" + "_".join(_normalize_modality(item) for item in suffix.split("_") if item)
    return name


def is_single_modality_pattern(pattern_name: str) -> bool:
    return canonical_missing_pattern_name(pattern_name) in SINGLE_MODALITY_PATTERNS


def is_weak_single_modality_pattern(pattern_name: str) -> bool:
    return canonical_missing_pattern_name(pattern_name) in WEAK_SINGLE_MODALITY_PATTERNS


def is_sensing_only_pattern(pattern_name: str) -> bool:
    return canonical_missing_pattern_name(pattern_name) in SENSING_ONLY_PATTERNS


def make_fixed_missing_mask(
    batch_size: int,
    pattern: list[int] | torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    mask = torch.as_tensor(pattern, device=device, dtype=dtype)
    if mask.ndim != 1 or mask.numel() <= 0:
        raise ValueError("pattern must be a non-empty 1D mask.")
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("pattern values must be 0 or 1.")
    if not bool(mask.bool().any().item()):
        raise ValueError("all-missing patterns are not allowed.")
    return mask.unsqueeze(0).expand(batch_size, -1).clone()


def sample_eval_random_missing_mask(
    batch_size: int,
    num_modalities: int,
    p_missing: float | list[float],
    ensure_at_least_one: bool = True,
    always_available_indices: list[int] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    batch_size = int(batch_size)
    num_modalities = int(num_modalities)
    probs = torch.as_tensor(p_missing, dtype=torch.float32, device=device)
    if probs.ndim == 0:
        probs = probs.expand(num_modalities)
    mask = torch.rand(batch_size, num_modalities, device=device).ge(probs)
    for index in always_available_indices or ():
        mask[:, int(index)] = True
    if ensure_at_least_one:
        empty = ~mask.any(dim=1)
        if torch.any(empty):
            mask[empty, 0] = True
    return mask.to(dtype=torch.float32)


def _add_pattern(patterns: dict[str, list[int]], name: str, mask: list[int]) -> None:
    if any(mask):
        patterns[name] = mask


def _mask_for(
    names: tuple[str, ...],
    *,
    available: set[str] | None = None,
    missing: set[str] | None = None,
) -> list[int]:
    if available is not None:
        return [1 if name in available else 0 for name in names]
    return [0 if name in (missing or set()) else 1 for name in names]


def _normalize_modalities(modality_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(_normalize_modality(item) for item in modality_names)
    if not names:
        raise ValueError("modalities must include at least one modality.")
    duplicates = sorted({item for item in names if names.count(item) > 1})
    if duplicates:
        raise ValueError(f"modalities must not contain duplicates after alias normalization: {duplicates}.")
    return names


def _normalize_modality(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    canonical = MODALITY_ALIASES.get(key)
    if canonical is None:
        raise ValueError(f"Unknown modality '{value}'. Available aliases map to canonical modalities {DEFAULT_MODALITIES}.")
    return canonical


__all__ = [
    "DEFAULT_MODALITIES",
    "MODALITY_ALIASES",
    "SENSING_ONLY_PATTERNS",
    "SINGLE_MODALITY_PATTERNS",
    "WEAK_SINGLE_MODALITY_PATTERNS",
    "canonical_missing_pattern_name",
    "get_default_missing_patterns",
    "get_missing_pattern_mask",
    "get_missing_pattern_name",
    "is_sensing_only_pattern",
    "is_single_modality_pattern",
    "is_weak_single_modality_pattern",
    "list_standard_missing_patterns",
    "make_fixed_missing_mask",
    "resolve_missing_patterns",
    "sample_eval_random_missing_mask",
]
