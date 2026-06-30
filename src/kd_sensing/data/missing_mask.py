from typing import Any

import torch


CANONICAL_MODALITIES = ("image", "radar", "lidar", "gps")


def sample_missing_mask(
    batch_size: int,
    num_modalities: int,
    p_missing: float | list[float] | tuple[float, ...] = 0.25,
    *,
    always_available_indices: list[int] | tuple[int, ...] | None = None,
    ensure_at_least_one: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size = int(batch_size)
    num_modalities = int(num_modalities)
    if batch_size <= 0 or num_modalities <= 0:
        raise ValueError("batch_size and num_modalities must be positive.")
    probs = _p_missing_tensor(p_missing, num_modalities, device=device)
    mask = torch.rand(batch_size, num_modalities, device=device, generator=generator).ge(probs)
    for index in always_available_indices or ():
        mask[:, _valid_index(index, num_modalities)] = True
    if ensure_at_least_one:
        empty = ~mask.any(dim=1)
        if torch.any(empty):
            fallback = next(iter(always_available_indices or ()), 0)
            mask[empty, _valid_index(fallback, num_modalities)] = True
    return mask.to(dtype=dtype)


def make_pattern_mask(
    batch_size: int,
    modalities: list[str] | tuple[str, ...],
    *,
    available_modalities: list[str] | tuple[str, ...] | None = None,
    pattern_mask: torch.Tensor | list[Any] | tuple[Any, ...] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    batch_size = int(batch_size)
    names = tuple(str(item) for item in modalities)
    if batch_size <= 0 or not names:
        raise ValueError("batch_size and modalities must be non-empty.")
    if pattern_mask is not None:
        mask = torch.as_tensor(pattern_mask, dtype=torch.bool, device=device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, len(names)):
            raise ValueError(f"pattern_mask must have shape [{batch_size}, {len(names)}], got {tuple(mask.shape)}.")
        if not mask.any(dim=1).all():
            raise ValueError("pattern_mask must keep at least one modality available per sample.")
        return mask.to(dtype=dtype)
    if available_modalities is None:
        return torch.ones(batch_size, len(names), dtype=dtype, device=device)
    available = {str(item) for item in available_modalities}
    unknown = sorted(available - set(names))
    if unknown:
        raise ValueError(f"Unknown available modalities {unknown}; configured modalities: {list(names)}.")
    if not available:
        raise ValueError("available_modalities must keep at least one modality.")
    row = torch.tensor([name in available for name in names], dtype=torch.bool, device=device)
    return row.unsqueeze(0).expand(batch_size, -1).to(dtype=dtype)


def pattern_mask(
    pattern: str | dict[str, Any] | list[int] | tuple[int, ...],
    modalities: list[str] | tuple[str, ...] = CANONICAL_MODALITIES,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    names = _validate_modalities(modalities)
    if isinstance(pattern, dict):
        return make_pattern_mask(
            1,
            names,
            available_modalities=pattern.get("available_modalities"),
            pattern_mask=pattern.get("pattern_mask"),
            device=device,
            dtype=dtype,
        )[0]
    if isinstance(pattern, (list, tuple)):
        return make_pattern_mask(1, names, pattern_mask=pattern, device=device, dtype=dtype)[0]
    name = str(pattern)
    count = len(names)
    if name == "full":
        mask = torch.ones(count, dtype=torch.bool, device=device)
    elif name == "missing_gps" or name == "non_gps_only":
        mask = torch.tensor([item != "gps" for item in names], dtype=torch.bool, device=device)
    elif name == "only_gps":
        mask = torch.tensor([item == "gps" for item in names], dtype=torch.bool, device=device)
    elif name.startswith("missing_") and name.removeprefix("missing_") in names:
        missing = name.removeprefix("missing_")
        mask = torch.tensor([item != missing for item in names], dtype=torch.bool, device=device)
    elif name.startswith("only_") and name.removeprefix("only_") in names:
        only = name.removeprefix("only_")
        mask = torch.tensor([item == only for item in names], dtype=torch.bool, device=device)
    elif name == "missing_one_random":
        index = int(torch.randint(count, (1,), device=device, generator=generator).item())
        mask = torch.ones(count, dtype=torch.bool, device=device)
        mask[index] = False
    elif name == "only_one_random":
        index = int(torch.randint(count, (1,), device=device, generator=generator).item())
        mask = torch.zeros(count, dtype=torch.bool, device=device)
        mask[index] = True
    elif name.startswith("random_"):
        p_missing = float(name.split("_", 1)[1])
        mask = sample_missing_mask(
            1,
            count,
            p_missing,
            ensure_at_least_one=True,
            device=device,
            generator=generator,
        )[0]
    else:
        raise ValueError(f"Unknown missing pattern '{name}' for modalities {list(names)}.")
    if not bool(mask.any().item()):
        raise ValueError(f"missing pattern '{name}' keeps no modalities available.")
    return mask.to(dtype=dtype)


def sample_pattern_balanced_mask(
    batch_size: int,
    modalities: list[str] | tuple[str, ...] = CANONICAL_MODALITIES,
    pattern_probs: dict[str, float] | list[str] | tuple[str, ...] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
    ensure_at_least_one: bool = True,
    return_pattern_ids: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    names = _validate_modalities(modalities)
    probs = _normalize_pattern_probs(pattern_probs)
    pattern_names = list(probs)
    weights = torch.tensor([probs[name] for name in pattern_names], dtype=torch.float32, device=device)
    choices = torch.multinomial(weights, batch_size, replacement=True, generator=generator)
    rows: list[torch.Tensor] = []
    sampled_names: list[str] = []
    for index in choices.detach().cpu().tolist():
        name = pattern_names[int(index)]
        row = pattern_mask(name, names, device=device, dtype=torch.bool, generator=generator)
        if ensure_at_least_one and not bool(row.any().item()):
            row[0] = True
        rows.append(row)
        sampled_names.append(name)
    mask = torch.stack(rows, dim=0).to(dtype=dtype)
    return mask, sampled_names, choices if return_pattern_ids else None


def apply_modality_corruption(batch: dict[str, Any], corruption_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = corruption_config or {}
    output = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    for key in ("image", "vision"):
        if key in output:
            output[key] = _corrupt_tensor(output[key], cfg.get("image", cfg.get("vision", {})))
    if "gps" in output:
        output["gps"] = _corrupt_tensor(output["gps"], cfg.get("gps", {}))
    for key in ("lidar", "radar", "radar_ra", "radar_da"):
        if key in output:
            modality = "radar" if key.startswith("radar") else key
            output[key] = _dropout_tensor(output[key], cfg.get(modality, {}))
    return output


def _p_missing_tensor(value: float | list[float] | tuple[float, ...], num_modalities: int, *, device) -> torch.Tensor:
    probs = torch.as_tensor(value, dtype=torch.float32, device=device)
    if probs.ndim == 0:
        probs = probs.expand(num_modalities)
    if tuple(probs.shape) != (num_modalities,):
        raise ValueError(f"p_missing must be a float or length-{num_modalities} list, got shape {tuple(probs.shape)}.")
    if torch.any((probs < 0) | (probs > 1)):
        raise ValueError("p_missing values must be in [0, 1].")
    return probs


def _validate_modalities(modalities: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(str(item) for item in modalities)
    if not names:
        raise ValueError("modalities must be non-empty.")
    duplicates = sorted({item for item in names if names.count(item) > 1})
    invalid = sorted({item for item in names if item not in CANONICAL_MODALITIES})
    if duplicates or invalid:
        raise ValueError(
            f"modalities must use canonical names {list(CANONICAL_MODALITIES)}; duplicates={duplicates}, invalid={invalid}."
        )
    return names


def _normalize_pattern_probs(pattern_probs: dict[str, float] | list[str] | tuple[str, ...] | None) -> dict[str, float]:
    if pattern_probs is None:
        pattern_probs = {
            "full": 0.2,
            "missing_gps": 0.15,
            "non_gps_only": 0.15,
            "only_gps": 0.1,
            "missing_one_random": 0.1,
            "only_one_random": 0.1,
            "random_0.25": 0.1,
            "random_0.5": 0.05,
            "random_0.75": 0.05,
        }
    if isinstance(pattern_probs, (list, tuple)):
        pattern_probs = {str(item): 1.0 for item in pattern_probs}
    if not isinstance(pattern_probs, dict) or not pattern_probs:
        raise ValueError("pattern_probs must be a non-empty mapping or list of pattern names.")
    cleaned = {str(key): float(value) for key, value in pattern_probs.items()}
    if any(value < 0 for value in cleaned.values()) or sum(cleaned.values()) <= 0:
        raise ValueError("pattern_probs values must be non-negative with a positive sum.")
    total = sum(cleaned.values())
    return {key: value / total for key, value in cleaned.items() if value > 0}


def _valid_index(index: int, num_modalities: int) -> int:
    index = int(index)
    if index < 0 or index >= num_modalities:
        raise ValueError(f"modality index {index} out of range for {num_modalities} modalities.")
    return index


def _corrupt_tensor(value: Any, cfg: Any) -> Any:
    if not torch.is_tensor(value) or not isinstance(cfg, dict):
        return value
    result = value
    noise_std = float(cfg.get("gaussian_noise_std", cfg.get("noise_std", 0.0)) or 0.0)
    if noise_std > 0.0:
        result = result + torch.randn_like(result) * noise_std
    if bool(cfg.get("zero_out", False)):
        result = torch.zeros_like(result)
    return _dropout_tensor(result, cfg)


def _dropout_tensor(value: Any, cfg: Any) -> Any:
    if not torch.is_tensor(value) or not isinstance(cfg, dict):
        return value
    prob = float(cfg.get("dropout_prob", cfg.get("drop_prob", 0.0)) or 0.0)
    if prob <= 0.0:
        return value
    keep = torch.rand(value.shape[:1], device=value.device).ge(prob).view(-1, *([1] * (value.ndim - 1)))
    return torch.where(keep, value, torch.zeros_like(value))
