from itertools import combinations

import torch

from kd_sensing.data.missing_mask import pattern_mask, sample_missing_mask

DEFAULT_MODALITIES = ["image", "radar", "lidar", "gps"]


def get_default_missing_patterns(modalities: list[str]) -> dict[str, list[int]]:
    names = [str(item) for item in (modalities or DEFAULT_MODALITIES)]
    if not names:
        raise ValueError("modalities must include at least one modality.")
    if len(set(names)) != len(names):
        raise ValueError("modalities must not contain duplicates.")

    count = len(names)
    patterns: dict[str, list[int]] = {"full": [1] * count}
    for index, name in enumerate(names):
        mask = [1] * count
        mask[index] = 0
        _add_pattern(patterns, f"missing_{name}", mask)
    for index, name in enumerate(names):
        mask = [0] * count
        mask[index] = 1
        _add_pattern(patterns, f"only_{name}", mask)
    for left, right in combinations(range(count), 2):
        mask = [1] * count
        mask[left] = 0
        mask[right] = 0
        _add_pattern(patterns, f"missing_{names[left]}_{names[right]}", mask)
    if names == DEFAULT_MODALITIES:
        patterns["non_gps_only"] = [1, 1, 1, 0]
    return patterns


def resolve_missing_patterns(patterns: str | list[str] | tuple[str, ...] | None, modalities: list[str]) -> dict[str, list[int]]:
    if patterns is None or patterns == "default" or list(patterns) == ["default"]:
        return get_default_missing_patterns(modalities)
    if isinstance(patterns, str):
        names = [item for item in patterns.replace(",", " ").split() if item]
    else:
        names = [str(item) for item in patterns]
    return {
        name: [int(value) for value in pattern_mask(name, modalities, dtype=torch.bool).to(dtype=torch.int64).tolist()]
        for name in names
        if not name.startswith("random_")
    }


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
    return sample_missing_mask(
        batch_size,
        num_modalities,
        p_missing,
        ensure_at_least_one=ensure_at_least_one,
        always_available_indices=always_available_indices,
        device=device,
        dtype=torch.float32,
    )


def _add_pattern(patterns: dict[str, list[int]], name: str, mask: list[int]) -> None:
    if any(mask):
        patterns[name] = mask
