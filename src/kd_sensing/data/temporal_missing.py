import random
from typing import Any, Mapping

import torch

from kd_sensing.data.temporal_missing_contract import (
    TEMPORAL_SUPERSET_PAYLOAD_KEY,
    normalize_temporal_missing_mode,
)


DEFAULT_TEMPORAL_MODALITIES = ("image", "radar", "gps", "lidar")
STRATIFIED_TEMPORAL_MISSING_TYPES = ("modality_level", "frame_level", "modality_frame", "block")
_MODALITY_KEYS = {
    "image": ("image",),
    "radar": ("radar_ra", "radar_da"),
    "gps": ("gps",),
    "lidar": ("lidar",),
}


def masked_temporal_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError(f"values must include batch and time dimensions, got {tuple(values.shape)}.")
    if mask is None:
        return values.mean(dim=1)
    valid = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
    if valid.ndim == 1:
        valid = valid.unsqueeze(0)
    if valid.shape[:2] != values.shape[:2]:
        raise ValueError(f"mask shape {tuple(valid.shape)} does not match values batch/time {tuple(values.shape[:2])}.")
    weights = valid.to(dtype=values.dtype).view(*valid.shape, *([1] * (values.ndim - 2)))
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def sample_stratified_modality_temporal_mask(
    *,
    history_window: int = 5,
    modalities: tuple[str, ...] | list[str] = DEFAULT_TEMPORAL_MODALITIES,
    drop_counts: tuple[int, ...] | list[int] = (0, 1, 2, 3),
    temporal_missing_rates: tuple[float, ...] | list[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    temporal_missing_types: tuple[str, ...] | list[str] = STRATIFIED_TEMPORAL_MISSING_TYPES,
    rng: random.Random | None = None,
    fixed_drop_modalities: tuple[str, ...] | list[str] | None = None,
    fixed_rate: float | None = None,
    fixed_mask_type: str | None = None,
) -> dict[str, Any]:
    names = tuple(str(item) for item in modalities)
    if names != DEFAULT_TEMPORAL_MODALITIES:
        raise ValueError(f"Four-modality temporal masks require modalities {list(DEFAULT_TEMPORAL_MODALITIES)}.")
    steps = int(history_window)
    if steps <= 0:
        raise ValueError("history_window must be positive.")
    rng = rng or random.Random()
    if fixed_drop_modalities is None:
        choices = [min(max(int(value), 0), len(names) - 1) for value in drop_counts]
        drop_count = rng.choice(choices)
        dropped_indices = tuple(sorted(rng.sample(range(len(names)), drop_count))) if drop_count else ()
    else:
        dropped = {str(item) for item in fixed_drop_modalities}
        unknown = dropped - set(names)
        if unknown:
            raise ValueError(f"Unknown MMW modalities in fixed mask: {sorted(unknown)}.")
        dropped_indices = tuple(index for index, name in enumerate(names) if name in dropped)
        drop_count = len(dropped_indices)
    rates = tuple(float(value) for value in temporal_missing_rates)
    types = tuple(str(value) for value in temporal_missing_types)
    if not rates or not types:
        raise ValueError("Temporal mask rates and types must be non-empty.")
    rate = max(0.0, min(float(fixed_rate if fixed_rate is not None else rng.choice(rates)), 1.0))
    mask_type = str(fixed_mask_type or rng.choice(types)).strip()
    if mask_type not in STRATIFIED_TEMPORAL_MISSING_TYPES:
        raise ValueError(f"Unsupported four-modality temporal mask type {mask_type!r}.")
    mask = [[True] * len(names) for _ in range(steps)]
    for index in dropped_indices:
        for row in mask:
            row[index] = False
    active = [index for index in range(len(names)) if index not in dropped_indices]
    _apply_temporal_type(mask, active, rate=rate, mask_type=mask_type, rng=rng)
    fixes = _ensure_available(mask, active)
    tensor = torch.tensor(mask, dtype=torch.bool)
    return {
        "modality_temporal_mask": tensor,
        "temporal_mask": tensor.any(dim=1),
        "modality_mask": tensor.any(dim=0),
        "dropped_modalities": [names[index] for index in dropped_indices],
        "mask_type": mask_type,
        "rate": rate,
        "drop_count": drop_count,
        "num_fallback_fixes": fixes,
    }


def apply_training_temporal_missing(
    batch: dict[str, Any],
    cfg: Mapping[str, Any],
    *,
    epoch: int,
    step: int,
) -> dict[str, Any]:
    temporal = cfg.get("temporal_missing", {})
    if not isinstance(temporal, Mapping) or not temporal.get("enabled", False):
        return batch
    mode = normalize_temporal_missing_mode(temporal.get("mode"))
    if mode == "none":
        return batch
    modalities = _configured_modalities(cfg)
    batch_size, steps = _batch_time_shape(batch, modalities)
    base = _base_mask(batch, modalities, batch_size, steps)
    if not bool(base.any(dim=(1, 2)).all().item()):
        raise ValueError("Four-modality temporal missing requires one source cell per sample.")
    preserve_superset = bool(temporal.get("preserve_unmasked_for_superset", False))
    original_inputs = _input_tensors(batch, modalities) if preserve_superset else None
    seed = _training_seed(cfg, temporal, epoch=epoch, step=step)
    rng = random.Random(seed)
    sampled = torch.stack(
        [
            sample_stratified_modality_temporal_mask(
                history_window=steps,
                modalities=modalities,
                drop_counts=_csv_values(temporal.get("train_missing_drop_counts"), int, (0, 1, 2, 3)),
                temporal_missing_rates=_csv_values(
                    temporal.get("train_temporal_missing_rates"), float, (0.0, 0.2, 0.4, 0.6, 0.8)
                ),
                temporal_missing_types=_csv_values(
                    temporal.get("train_temporal_missing_types"), str, STRATIFIED_TEMPORAL_MISSING_TYPES
                ),
                rng=rng,
            )["modality_temporal_mask"]
            for _ in range(batch_size)
        ]
    )
    mask, fixes = _restore_missing_samples(base & sampled, base)
    apply_modality_temporal_mask_to_batch(batch, mask, modalities=modalities)
    if original_inputs is not None:
        batch[TEMPORAL_SUPERSET_PAYLOAD_KEY] = {
            "inputs": original_inputs,
            "base_mask": base,
            "modalities": modalities,
        }
    batch["temporal_missing_metadata"] = {
        "mode": mode,
        "seed": seed,
        "available_rate": float(mask.to(dtype=torch.float32).mean().item()),
        "num_fallback_fixes": fixes,
    }
    return batch


def apply_modality_temporal_mask_to_batch(
    batch: dict[str, Any],
    modality_temporal_mask: torch.Tensor,
    *,
    modalities: tuple[str, ...] | list[str] = DEFAULT_TEMPORAL_MODALITIES,
) -> dict[str, Any]:
    names = tuple(str(item) for item in modalities)
    if names != DEFAULT_TEMPORAL_MODALITIES:
        raise ValueError(f"Four-modality temporal masks require modalities {list(DEFAULT_TEMPORAL_MODALITIES)}.")
    mask = torch.as_tensor(modality_temporal_mask, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[-1] != len(names):
        raise ValueError(f"modality_temporal_mask must have shape [B,T,{len(names)}] or [T,{len(names)}].")
    batch_size = _batch_size(batch, names)
    if mask.shape[0] == 1 and batch_size != 1:
        mask = mask.expand(batch_size, -1, -1)
    if mask.shape[0] != batch_size:
        raise ValueError(f"temporal mask batch size {mask.shape[0]} does not match batch size {batch_size}.")
    for index, modality in enumerate(names):
        keys = [key for key in _MODALITY_KEYS[modality] if torch.is_tensor(batch.get(key))]
        if not keys:
            raise ValueError(f"Four-modality batch is missing {modality} inputs.")
        keep = mask[:, :, index]
        for key in keys:
            tensor = batch[key]
            local_keep = keep.to(device=tensor.device)
            batch[key] = tensor.masked_fill(_expand_mask(~local_keep, tensor.ndim), 0)
        batch[f"{modality}_valid_mask"] = keep.to(device=batch[keys[0]].device)
        batch[f"{modality}_dropout_mask"] = (~keep).to(device=batch[keys[0]].device)
    batch["modality_temporal_mask"] = mask
    batch["temporal_mask"] = mask.any(dim=2)
    batch["modality_mask"] = mask.any(dim=1)
    batch["available_modalities"] = mask.any(dim=1)
    return batch


def _configured_modalities(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    model = cfg.get("model", {})
    primary = model.get("primary", {}) if isinstance(model, Mapping) else {}
    names = tuple(str(item) for item in primary.get("modalities", DEFAULT_TEMPORAL_MODALITIES))
    if names != DEFAULT_TEMPORAL_MODALITIES:
        raise ValueError(f"Four-modality temporal missing requires modalities {list(DEFAULT_TEMPORAL_MODALITIES)}.")
    return names


def _batch_time_shape(batch: Mapping[str, Any], modalities: tuple[str, ...]) -> tuple[int, int]:
    for modality in modalities:
        for key in _MODALITY_KEYS[modality]:
            tensor = batch.get(key)
            if torch.is_tensor(tensor) and tensor.ndim >= 2:
                return int(tensor.shape[0]), int(tensor.shape[1])
    raise ValueError("Four-modality temporal missing requires a batched sequence input.")


def _batch_size(batch: Mapping[str, Any], modalities: tuple[str, ...]) -> int:
    return _batch_time_shape(batch, modalities)[0]


def _base_mask(batch: Mapping[str, Any], modalities: tuple[str, ...], batch_size: int, steps: int) -> torch.Tensor:
    base = torch.ones(batch_size, steps, len(modalities), dtype=torch.bool)
    for index, modality in enumerate(modalities):
        valid = batch.get(f"{modality}_valid_mask")
        dropped = batch.get(f"{modality}_dropout_mask")
        if torch.is_tensor(valid):
            base[:, :, index] &= _coerce_temporal_mask(valid, batch_size, steps)
        if torch.is_tensor(dropped):
            base[:, :, index] &= ~_coerce_temporal_mask(dropped, batch_size, steps)
    return base


def _input_tensors(batch: Mapping[str, Any], modalities: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for modality in modalities
        for key in _MODALITY_KEYS[modality]
        if torch.is_tensor(batch.get(key))
    }


def _training_seed(cfg: Mapping[str, Any], temporal: Mapping[str, Any], *, epoch: int, step: int) -> int:
    experiment = cfg.get("experiment", {})
    base = int(temporal.get("seed") or (experiment.get("seed", 0) if isinstance(experiment, Mapping) else 0))
    return base + int(epoch) * 1_000_003 + int(step)


def _csv_values(value: Any, convert, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if value in (None, ""):
        return default
    values = value if isinstance(value, (list, tuple)) else str(value).split(",")
    result = tuple(convert(item.strip() if isinstance(item, str) else item) for item in values if str(item).strip())
    if not result:
        raise ValueError("Temporal mask configuration must not be empty.")
    return result


def _apply_temporal_type(mask: list[list[bool]], active: list[int], *, rate: float, mask_type: str, rng: random.Random) -> None:
    if rate <= 0.0 or not active:
        return
    steps = len(mask)
    if mask_type == "modality_level":
        count = min(max(len(active) - 1, 0), int(round(rate * len(active))))
        for index in rng.sample(active, count):
            for row in mask:
                row[index] = False
    elif mask_type == "frame_level":
        for row in rng.sample(range(steps), min(steps, int(round(rate * steps)))):
            for index in active:
                mask[row][index] = False
    elif mask_type == "block":
        length = min(steps, max(1, int(round(rate * steps))))
        start = rng.randint(0, max(steps - length, 0))
        for row in range(start, start + length):
            for index in active:
                mask[row][index] = False
    else:
        cells = [(row, index) for row in range(steps) for index in active]
        for row, index in rng.sample(cells, min(len(cells), int(round(rate * len(cells))))):
            mask[row][index] = False


def _ensure_available(mask: list[list[bool]], active: list[int]) -> int:
    if any(any(row) for row in mask) or not active:
        return 0
    mask[-1][active[-1]] = True
    return 1


def _restore_missing_samples(mask: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, int]:
    result = mask.clone()
    fixes = 0
    for row in (~result.any(dim=(1, 2))).nonzero(as_tuple=False).flatten().tolist():
        choices = base[row].nonzero(as_tuple=False)
        if not len(choices):
            raise ValueError("Four-modality temporal missing requires one source cell per sample.")
        time_index, modality_index = choices[-1].tolist()
        result[row, time_index, modality_index] = True
        fixes += 1
    return result, fixes


def _coerce_temporal_mask(value: torch.Tensor, batch_size: int, steps: int) -> torch.Tensor:
    mask = value.detach().to(device="cpu", dtype=torch.bool)
    if mask.ndim == 1:
        if mask.shape[0] != batch_size:
            raise ValueError(f"Temporal mask must have {batch_size} rows.")
        mask = mask.unsqueeze(1).expand(-1, steps)
    if tuple(mask.shape) != (batch_size, steps):
        raise ValueError(f"Temporal mask must have shape [{batch_size}, {steps}], got {tuple(mask.shape)}.")
    return mask


def _expand_mask(mask: torch.Tensor, dimensions: int) -> torch.Tensor:
    return mask.view(*mask.shape, *([1] * (dimensions - mask.ndim)))


__all__ = [
    "DEFAULT_TEMPORAL_MODALITIES",
    "STRATIFIED_TEMPORAL_MISSING_TYPES",
    "TEMPORAL_SUPERSET_PAYLOAD_KEY",
    "apply_modality_temporal_mask_to_batch",
    "apply_training_temporal_missing",
    "masked_temporal_mean",
    "sample_stratified_modality_temporal_mask",
]
