from dataclasses import dataclass
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

import torch

from kd_sensing.data.temporal_missing_contract import (
    TEMPORAL_AGGREGATION_MODES,
    TEMPORAL_MISSING_MODES,
    normalize_temporal_aggregation,
    normalize_temporal_missing_mode,
)

DEFAULT_TEMPORAL_MODALITIES = ("image", "radar", "lidar", "gps")
STRATIFIED_TEMPORAL_MISSING_TYPES = ("modality_level", "frame_level", "modality_frame", "block")


@dataclass(frozen=True)
class TemporalMissingConfig:
    mode: str = "none"
    prob: float = 0.0
    block_len: int = 1
    ensure_at_least_one_frame: bool = True
    ensure_at_least_one_modality_per_frame: bool = False


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
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def aggregate_temporal(values: torch.Tensor, mask: torch.Tensor | None, mode: str = "masked_mean") -> torch.Tensor:
    mode = normalize_temporal_aggregation(mode)
    if mode == "mean":
        return values.mean(dim=1)
    if mode == "masked_mean":
        return masked_temporal_mean(values, mask)
    if mode == "last":
        if mask is None:
            return values[:, -1, ...]
        valid = torch.as_tensor(mask, device=values.device, dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape[:2] != values.shape[:2]:
            raise ValueError(f"mask shape {tuple(valid.shape)} does not match values batch/time {tuple(values.shape[:2])}.")
        fallback = torch.full((valid.shape[0],), valid.shape[1] - 1, dtype=torch.long, device=values.device)
        positions = torch.arange(valid.shape[1], device=values.device).view(1, -1).expand_as(valid)
        last = torch.where(valid, positions, torch.full_like(positions, -1)).max(dim=1).values
        last = torch.where(last.ge(0), last, fallback)
        return values[torch.arange(values.shape[0], device=values.device), last]
    raise NotImplementedError("temporal_aggregation='flatten' requires model-specific input dimension changes.")


def parse_csv_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def parse_csv_floats(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def parse_csv_strings(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value in (None, ""):
        return default
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


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
    ensure_at_least_one_cell: bool = True,
    ensure_at_least_one_frame: bool = True,
    ensure_at_least_one_modality: bool = True,
) -> dict[str, Any]:
    rng = rng or random.Random()
    names = tuple(str(item) for item in modalities)
    if not names:
        raise ValueError("modalities must not be empty.")
    steps = int(history_window)
    if steps <= 0:
        raise ValueError("history_window must be positive.")
    max_drop = len(names) - 1
    if fixed_drop_modalities is None:
        allowed_drop_counts = [min(max(int(item), 0), max_drop) for item in drop_counts]
        drop_count = rng.choice(allowed_drop_counts)
        dropped_indices = tuple(sorted(rng.sample(range(len(names)), drop_count))) if drop_count else ()
    else:
        dropped = {str(item) for item in fixed_drop_modalities}
        dropped_indices = tuple(index for index, name in enumerate(names) if name in dropped)
        drop_count = len(dropped_indices)
    rate = float(fixed_rate if fixed_rate is not None else rng.choice(tuple(float(item) for item in temporal_missing_rates)))
    rate = max(0.0, min(rate, 1.0))
    mask_type = str(fixed_mask_type or rng.choice(tuple(temporal_missing_types))).strip()
    if mask_type not in STRATIFIED_TEMPORAL_MISSING_TYPES:
        raise ValueError(f"temporal missing type must be one of {STRATIFIED_TEMPORAL_MISSING_TYPES}, got {mask_type!r}.")
    mask = [[True for _ in names] for _ in range(steps)]
    for modality_index in dropped_indices:
        for step in range(steps):
            mask[step][modality_index] = False
    active = [index for index in range(len(names)) if index not in dropped_indices]
    _apply_temporal_type(mask, active, rate=rate, mask_type=mask_type, rng=rng)
    fixed = _repair_stratified_mask(
        mask,
        active,
        ensure_at_least_one_cell=ensure_at_least_one_cell,
        ensure_at_least_one_frame=ensure_at_least_one_frame,
        ensure_at_least_one_modality=ensure_at_least_one_modality,
    )
    tensor = torch.tensor(mask, dtype=torch.bool)
    return {
        "modality_temporal_mask": tensor,
        "temporal_mask": tensor.any(dim=1),
        "modality_mask": tensor.any(dim=0),
        "dropped_modalities": [names[index] for index in dropped_indices],
        "mask_type": mask_type,
        "rate": rate,
        "drop_count": int(drop_count),
        "num_fallback_fixes": int(fixed),
        "history_window": steps,
        "num_modalities": len(names),
        "modalities": list(names),
    }


def generate_fixed_eval_mask_cache(
    cache_dir: str | Path,
    *,
    rates: tuple[float, ...] | list[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    drop_counts: tuple[int, ...] | list[int] = (0, 1, 2, 3),
    mask_types: tuple[str, ...] | list[str] = ("modality_frame", "frame_level", "block"),
    num_masks_per_cell: int = 16,
    seed: int = 20260708,
    history_window: int = 5,
    modalities: tuple[str, ...] | list[str] = DEFAULT_TEMPORAL_MODALITIES,
) -> dict[tuple[float, int], dict[str, Any]]:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for rate in rates:
        for drop_count in drop_counts:
            path = eval_mask_cache_path(root, float(rate), int(drop_count))
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                _validate_mask_cache(payload)
            else:
                payload = _build_eval_mask_payload(
                    rate=float(rate),
                    drop_count=int(drop_count),
                    mask_types=tuple(mask_types),
                    num_masks=int(num_masks_per_cell),
                    seed=int(seed),
                    history_window=int(history_window),
                    modalities=tuple(modalities),
                )
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result[(float(rate), int(drop_count))] = payload
    return result


def eval_mask_cache_path(cache_dir: str | Path, rate: float, drop_count: int) -> Path:
    rate_token = f"{float(rate):.1f}"
    return Path(cache_dir) / f"rate_{rate_token}_drop{int(drop_count)}.json"


def apply_modality_temporal_mask_to_batch(
    batch: dict[str, Any],
    modality_temporal_mask: torch.Tensor,
    *,
    modalities: tuple[str, ...] | list[str] = DEFAULT_TEMPORAL_MODALITIES,
) -> dict[str, Any]:
    from kd_sensing.data.difficulty.operators.modality import MODALITY_BATCH_KEYS, _merge_valid_mask, _zero_fill

    names = tuple(str(item) for item in modalities)
    mask = torch.as_tensor(modality_temporal_mask, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[-1] != len(names):
        raise ValueError(f"modality_temporal_mask must have shape [B,T,{len(names)}] or [T,{len(names)}], got {tuple(mask.shape)}.")
    for index, modality in enumerate(names):
        missing = ~mask[:, :, index]
        keys = [key for key in MODALITY_BATCH_KEYS.get(modality, (modality,)) if torch.is_tensor(batch.get(key))]
        for key in keys:
            tensor = batch[key]
            local_missing = missing.to(device=tensor.device)
            if local_missing.shape[0] == 1 and int(tensor.shape[0]) != 1:
                local_missing = local_missing.expand(int(tensor.shape[0]), -1)
            batch[key] = _zero_fill(tensor, local_missing)
            batch[f"{modality}_valid_mask"] = _merge_valid_mask(
                batch.get(f"{modality}_valid_mask"),
                ~local_missing,
                device=tensor.device,
            )
            batch[f"{modality}_dropout_mask"] = local_missing
            batch[f"{modality}_missing_mask"] = local_missing
    expanded = mask
    first = next((value for value in batch.values() if torch.is_tensor(value) and value.ndim >= 2), None)
    if first is not None and expanded.shape[0] == 1 and int(first.shape[0]) != 1:
        expanded = expanded.expand(int(first.shape[0]), -1, -1)
    batch["modality_temporal_mask"] = expanded
    batch["temporal_mask"] = expanded.any(dim=2)
    batch["modality_mask"] = expanded.any(dim=1)
    batch["available_modalities"] = expanded.any(dim=1)
    return batch


def _apply_temporal_type(mask: list[list[bool]], active: list[int], *, rate: float, mask_type: str, rng: random.Random) -> None:
    if rate <= 0.0 or not active:
        return
    steps = len(mask)
    if mask_type == "modality_level":
        count = min(len(active) - 1 if len(active) > 1 else 0, int(round(rate * len(active))))
        for modality in rng.sample(active, max(count, 0)):
            for step in range(steps):
                mask[step][modality] = False
        return
    if mask_type == "frame_level":
        count = min(steps, int(round(rate * steps)))
        for step in rng.sample(range(steps), max(count, 0)):
            for modality in active:
                mask[step][modality] = False
        return
    if mask_type == "block":
        length = min(steps, max(1, int(round(rate * steps))))
        start = rng.randint(0, max(steps - length, 0))
        for step in range(start, start + length):
            for modality in active:
                mask[step][modality] = False
        return
    cells = [(step, modality) for step in range(steps) for modality in active]
    count = min(len(cells), int(round(rate * len(cells))))
    for step, modality in rng.sample(cells, max(count, 0)):
        mask[step][modality] = False


def _repair_stratified_mask(
    mask: list[list[bool]],
    active: list[int],
    *,
    ensure_at_least_one_cell: bool,
    ensure_at_least_one_frame: bool,
    ensure_at_least_one_modality: bool,
) -> int:
    if not active:
        return 0
    fixed = 0
    if ensure_at_least_one_cell and not any(any(row) for row in mask):
        mask[-1][active[-1]] = True
        fixed += 1
    if ensure_at_least_one_frame and not any(any(row) for row in mask):
        mask[-1][active[-1]] = True
        fixed += 1
    if ensure_at_least_one_modality and not any(mask[step][modality] for step in range(len(mask)) for modality in active):
        mask[-1][active[-1]] = True
        fixed += 1
    return fixed


def _build_eval_mask_payload(
    *,
    rate: float,
    drop_count: int,
    mask_types: tuple[str, ...],
    num_masks: int,
    seed: int,
    history_window: int,
    modalities: tuple[str, ...],
) -> dict[str, Any]:
    combos = list(itertools.combinations(modalities, int(drop_count)))
    if not combos:
        combos = [()]
    rng = random.Random((int(seed) * 1009) + int(round(rate * 1000)) * 17 + int(drop_count))
    masks = []
    for index in range(int(num_masks)):
        dropped = combos[index % len(combos)]
        item = sample_stratified_modality_temporal_mask(
            history_window=history_window,
            modalities=modalities,
            fixed_drop_modalities=dropped,
            fixed_rate=rate,
            fixed_mask_type=mask_types[index % len(mask_types)],
            rng=rng,
        )
        masks.append(
            {
                "modality_temporal_mask": item["modality_temporal_mask"].int().tolist(),
                "dropped_modalities": list(dropped),
                "mask_type": item["mask_type"],
                "num_fallback_fixes": item["num_fallback_fixes"],
            }
        )
    payload = {
        "version": "temporal_eval_masks_v1",
        "rate": float(rate),
        "drop_count": int(drop_count),
        "num_masks": int(num_masks),
        "history_window": int(history_window),
        "num_modalities": len(modalities),
        "modalities": list(modalities),
        "seed": int(seed),
        "masks": masks,
    }
    payload["checksum"] = _mask_payload_checksum(payload)
    return payload


def _validate_mask_cache(payload: dict[str, Any]) -> None:
    expected = payload.get("checksum")
    if not expected or expected != _mask_payload_checksum(payload):
        raise ValueError("Temporal eval mask cache checksum mismatch.")


def _mask_payload_checksum(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("checksum", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


__all__ = [
    "DEFAULT_TEMPORAL_MODALITIES",
    "STRATIFIED_TEMPORAL_MISSING_TYPES",
    "TEMPORAL_AGGREGATION_MODES",
    "TEMPORAL_MISSING_MODES",
    "TemporalMissingConfig",
    "aggregate_temporal",
    "apply_modality_temporal_mask_to_batch",
    "eval_mask_cache_path",
    "generate_fixed_eval_mask_cache",
    "masked_temporal_mean",
    "normalize_temporal_aggregation",
    "normalize_temporal_missing_mode",
    "parse_csv_floats",
    "parse_csv_ints",
    "parse_csv_strings",
    "sample_stratified_modality_temporal_mask",
]
