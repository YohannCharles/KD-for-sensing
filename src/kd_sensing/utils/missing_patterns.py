from itertools import combinations
import torch

from kd_sensing.modalities import MODALITY_ORDER


DEFAULT_MODALITIES = MODALITY_ORDER


def get_default_missing_patterns(modalities: list[str] | tuple[str, ...]) -> dict[str, list[int]]:
    return _standard_missing_patterns(modalities)


def _standard_missing_patterns(modality_names: list[str] | tuple[str, ...]) -> dict[str, list[int]]:
    names = _normalize_modalities(modality_names)
    patterns: dict[str, list[int]] = {"full": [1] * len(names)}
    for name in names:
        patterns[f"missing_{name}"] = _mask_for(names, missing={name})
        patterns[f"{name}_only"] = _mask_for(names, available={name})
    if "gps" in names:
        patterns["non_gps_only"] = _mask_for(names, missing={"gps"})
    for left, right in combinations(names, 2):
        patterns[f"missing_{left}_{right}"] = _mask_for(names, missing={left, right})
    return patterns


def resolve_missing_patterns(
    patterns: str | list[str] | tuple[str, ...] | None,
    modalities: list[str] | tuple[str, ...],
) -> dict[str, list[int]]:
    if patterns is None or patterns == "default" or (not isinstance(patterns, str) and list(patterns) == ["default"]):
        return get_default_missing_patterns(modalities)
    names = patterns.replace(",", " ").split() if isinstance(patterns, str) else [str(item) for item in patterns]
    return {
        name: get_missing_pattern_mask(name, modalities)
        for raw in names
        if (name := canonical_missing_pattern_name(raw)) != "avg_missing"
    }


def get_missing_pattern_mask(
    pattern_name: str,
    modality_names: list[str] | tuple[str, ...] | None = None,
) -> list[int]:
    names = _normalize_modalities(modality_names or DEFAULT_MODALITIES)
    name = canonical_missing_pattern_name(pattern_name)
    if name == "full":
        return [1] * len(names)
    if name == "avg_missing":
        raise ValueError("avg_missing is an aggregate, not a modality mask.")
    if name == "non_gps_only":
        missing = {"gps"}
    elif name.endswith("_only"):
        return _mask_for(names, available={_normalize_modality(name.removesuffix("_only"))})
    elif name.startswith("missing_"):
        missing = {_normalize_modality(item) for item in name.removeprefix("missing_").split("_") if item}
    else:
        raise ValueError(f"Unknown missing pattern '{pattern_name}'.")
    unknown = sorted(missing - set(names))
    if unknown:
        raise ValueError(f"Missing pattern '{name}' references unavailable modalities {unknown}.")
    return _mask_for(names, missing=missing)


def canonical_missing_pattern_name(pattern_name: str) -> str:
    name = str(pattern_name).strip().lower().replace("-", "_")
    if name in {"full", "avg_missing", "non_gps_only"}:
        return name
    if name.endswith("_only"):
        return f"{_normalize_modality(name.removesuffix('_only'))}_only"
    if name.startswith("missing_"):
        values = [_normalize_modality(item) for item in name.removeprefix("missing_").split("_") if item]
        return "missing_" + "_".join(values)
    return name


def make_fixed_missing_mask(
    batch_size: int,
    pattern: list[int] | torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    mask = torch.as_tensor(pattern, device=device, dtype=dtype)
    if mask.ndim != 1 or mask.numel() <= 0 or torch.any((mask != 0) & (mask != 1)) or not bool(mask.bool().any()):
        raise ValueError("pattern must be a non-empty binary mask with one available modality.")
    return mask.unsqueeze(0).expand(int(batch_size), -1).clone()


def _mask_for(names: tuple[str, ...], *, available: set[str] | None = None, missing: set[str] | None = None) -> list[int]:
    return [int(name in available) for name in names] if available is not None else [int(name not in (missing or set())) for name in names]


def _normalize_modalities(modality_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(_normalize_modality(name) for name in modality_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("modalities must be a non-empty unique subset of the MMW modality order.")
    return names


def _normalize_modality(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if name not in DEFAULT_MODALITIES:
        raise ValueError(f"Unknown modality '{value}'. Expected one of {list(DEFAULT_MODALITIES)}.")
    return name


__all__ = [
    "DEFAULT_MODALITIES",
    "canonical_missing_pattern_name",
    "get_default_missing_patterns",
    "get_missing_pattern_mask",
    "make_fixed_missing_mask",
    "resolve_missing_patterns",
]
