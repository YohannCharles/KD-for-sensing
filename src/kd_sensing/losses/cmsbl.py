import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


CANONICAL_MODALITIES = ("image", "radar", "gps", "lidar")
NUM_NON_EMPTY_MASKS = (1 << len(CANONICAL_MODALITIES)) - 1


def auxiliary_schedule_weight(config: dict[str, Any], epoch_number: int) -> float:
    epoch = int(epoch_number)
    if epoch <= 0:
        raise ValueError("CMSBL epoch_number must be one-based and positive.")
    start = float(config["start_weight"])
    end = float(config["end_weight"])
    start_epoch = int(config["start_epoch"])
    end_epoch = int(config["end_epoch"])
    if epoch < start_epoch:
        return start
    if epoch >= end_epoch or end_epoch <= start_epoch:
        return end
    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return start + (end - start) * progress


def effective_auxiliary_weights(config: dict[str, Any], epoch_number: int) -> tuple[float, float]:
    aux = config["aux_schedule"]
    if not aux["enabled"]:
        return 1.0, 1.0
    return (
        auxiliary_schedule_weight(aux["private"], epoch_number),
        auxiliary_schedule_weight(aux["shared"], epoch_number),
    )


def load_capacity_reference(
    config: dict[str, Any], *, dataset: str, modalities: tuple[str, ...]
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(str(config["stats_path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CMSBL capacity stats not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CMSBL capacity stats must be a JSON object.")
    source_split = str(payload.get("source_split", config.get("source_split", ""))).strip().lower()
    if "test" in source_split or "outer" in source_split:
        raise ValueError("CMSBL capacity stats must not use outer/test data.")
    if source_split not in {"train", "inner_train", "development", "inner_validation"}:
        raise ValueError("CMSBL capacity stats source_split must be train or fixed inner/development.")
    if source_split != str(config.get("source_split", "inner_train")).strip().lower():
        raise ValueError("CMSBL capacity stats source_split does not match the resolved config.")
    if payload.get("protocol") != "cmsbl_capacity_reference_v1":
        raise ValueError("CMSBL capacity stats protocol must be cmsbl_capacity_reference_v1.")
    payload_dataset = str(payload.get("dataset", "")).strip().lower()
    if payload_dataset != str(dataset).strip().lower():
        raise ValueError(f"CMSBL capacity dataset mismatch: {payload_dataset!r} != {dataset!r}.")
    metric = str(payload.get("metric", "")).strip().lower()
    if metric != "top1":
        raise ValueError("CMSBL capacity stats metric must be top1.")
    source_sha256 = str(payload.get("source_sha256", "")).strip().lower()
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("CMSBL capacity stats require a 64-character source_sha256.")
    raw_modalities = payload.get("modalities")
    if not isinstance(raw_modalities, dict) or set(raw_modalities) != set(modalities):
        raise ValueError("CMSBL capacity stats must contain exactly the configured modalities.")
    values = []
    for name in modalities:
        entry = raw_modalities[name]
        value = entry.get(metric) if isinstance(entry, dict) else entry
        number = float(value)
        if not math.isfinite(number) or number <= 0.0 or (metric == "top1" and number > 1.0):
            raise ValueError(f"Invalid CMSBL capacity value for {name}: {value!r}.")
        values.append(number)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "schema_version": int(payload.get("schema_version", 1)),
        "protocol": str(payload.get("protocol", "cmsbl_capacity_reference_v1")),
        "path": str(path),
        "sha256": digest,
        "dataset": payload_dataset,
        "source_split": source_split,
        "claim_eligible": bool(payload.get("claim_eligible", False)),
        "metric": metric,
        "source_sha256": source_sha256,
        "modalities": list(modalities),
        "source": payload.get("source", {}),
    }
    return torch.tensor(values, dtype=torch.float32), identity


def update_metric_ema(
    previous: torch.Tensor,
    initialized: torch.Tensor,
    current: torch.Tensor,
    valid: torch.Tensor,
    *,
    momentum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous.shape != current.shape or initialized.shape != current.shape or valid.shape != current.shape:
        raise ValueError("CMSBL metric EMA tensors must share shape [M].")
    estimate = current.to(device=previous.device, dtype=previous.dtype)
    active = valid.to(device=previous.device, dtype=torch.bool)
    old = initialized.to(device=previous.device, dtype=torch.bool)
    updated = torch.where(old, float(momentum) * previous + (1.0 - float(momentum)) * estimate, estimate)
    return torch.where(active, updated, previous), old | active


def capacity_gap_weights(
    reference: torch.Tensor,
    metric_ema: torch.Tensor,
    initialized: torch.Tensor,
    *,
    epoch_number: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.shape != metric_ema.shape or initialized.shape != reference.shape:
        raise ValueError("CMSBL capacity tensors must share shape [M].")
    eps = float(config["eps"])
    gap = (reference - metric_ema).clamp_min(0.0) / reference.clamp_min(eps)
    gap = torch.where(initialized.bool(), gap, torch.zeros_like(gap))
    weights = 1.0 + float(config["alpha"]) * gap.pow(float(config["gamma"]))
    weights = weights.clamp(float(config["min_weight"]), float(config["max_weight"]))
    if int(epoch_number) <= int(config["warmup_epochs"]):
        weights = torch.ones_like(weights)
    return weights, gap


def canonical_mask_id(modalities: Iterable[str]) -> int:
    names = {str(value).strip().lower() for value in modalities}
    if not names or names - set(CANONICAL_MODALITIES):
        raise ValueError("CMSBL mask must contain a non-empty subset of canonical modalities.")
    return sum(1 << CANONICAL_MODALITIES.index(name) for name in names)


def fusion_mask_ids(mask: torch.Tensor, modalities: tuple[str, ...]) -> torch.Tensor:
    available = mask.to(dtype=torch.bool)
    if available.ndim != 2 or available.shape[1] != len(modalities):
        raise ValueError("CMSBL fusion mask must have shape [B,M].")
    if set(modalities) != set(CANONICAL_MODALITIES) or len(modalities) != len(CANONICAL_MODALITIES):
        raise ValueError("CMSBL fusion mask requires the four canonical modalities.")
    if not bool(available.any(dim=1).all().item()):
        raise ValueError("CMSBL fusion mask must be non-empty for every sample.")
    ids = torch.zeros(available.shape[0], device=available.device, dtype=torch.long)
    for column, name in enumerate(modalities):
        ids += available[:, column].long() * (1 << CANONICAL_MODALITIES.index(name))
    return ids


def accumulate_mask_losses(
    loss_sums: torch.Tensor,
    counts: torch.Tensor,
    per_sample_loss: torch.Tensor,
    mask_ids: torch.Tensor,
) -> None:
    if loss_sums.shape != (NUM_NON_EMPTY_MASKS,) or counts.shape != loss_sums.shape:
        raise ValueError("CMSBL mask accumulators must have shape [15].")
    losses = per_sample_loss.detach().to(device=loss_sums.device, dtype=loss_sums.dtype).reshape(-1)
    ids = mask_ids.detach().to(device=loss_sums.device, dtype=torch.long).reshape(-1)
    if losses.shape != ids.shape or bool(((ids < 1) | (ids > NUM_NON_EMPTY_MASKS)).any().item()):
        raise ValueError("CMSBL per-sample losses and mask IDs are invalid.")
    indices = ids - 1
    loss_sums.index_add_(0, indices, losses)
    counts.index_add_(0, indices, torch.ones_like(indices, dtype=counts.dtype))


def all_reduce_mask_statistics(loss_sums: torch.Tensor, counts: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)


def update_mask_loss_ema(
    previous: torch.Tensor,
    initialized: torch.Tensor,
    cumulative_counts: torch.Tensor,
    epoch_loss_sums: torch.Tensor,
    epoch_counts: torch.Tensor,
    *,
    momentum: float,
    min_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(tensor.shape != (NUM_NON_EMPTY_MASKS,) for tensor in (
        previous, initialized, cumulative_counts, epoch_loss_sums, epoch_counts
    )):
        raise ValueError("CMSBL mask EMA tensors must have shape [15].")
    new_counts = cumulative_counts + epoch_counts.to(cumulative_counts)
    current = epoch_loss_sums / epoch_counts.clamp_min(1).to(epoch_loss_sums)
    valid = epoch_counts.gt(0) & new_counts.ge(int(min_count))
    updated = torch.where(
        initialized.bool(),
        float(momentum) * previous + (1.0 - float(momentum)) * current,
        current,
    )
    return torch.where(valid, updated, previous), initialized.bool() | valid, new_counts


def hard_mask_weights(
    loss_ema: torch.Tensor,
    counts: torch.Tensor,
    initialized: torch.Tensor,
    *,
    epoch_number: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_ema.shape != (NUM_NON_EMPTY_MASKS,) or counts.shape != loss_ema.shape or initialized.shape != loss_ema.shape:
        raise ValueError("CMSBL hard-mask tensors must have shape [15].")
    weights = torch.ones_like(loss_ema)
    difficulty = torch.ones_like(loss_ema)
    eligible = initialized.bool() & counts.ge(int(config["min_count"]))
    if int(epoch_number) <= int(config["warmup_epochs"]) or not bool(eligible.any().item()):
        return weights, difficulty
    mean = loss_ema[eligible].mean().clamp_min(float(config["eps"]))
    difficulty[eligible] = (loss_ema[eligible] / mean).clamp_min(0.0).pow(float(config["gamma"]))
    raw = difficulty[eligible]
    lower = torch.full_like(raw, float(config["min_weight"]))
    upper = torch.full_like(raw, float(config["max_weight"]))
    eligible_indices = eligible.nonzero(as_tuple=False).flatten()
    full = NUM_NON_EMPTY_MASKS - 1
    full_position = (eligible_indices == full).nonzero(as_tuple=False).flatten()
    if full_position.numel():
        lower[full_position[0]] = max(float(config["min_weight"]), float(config["full_mask_min_weight"]))
    if bool(config["normalize_mean_to_one"]):
        projected = _bounded_mean_projection(raw, lower, upper)
    else:
        projected = raw.clamp(lower, upper)
    weights[eligible] = projected
    return weights, difficulty


def _bounded_mean_projection(scores: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    if float(lower.mean()) > 1.0 + 1.0e-7 or float(upper.mean()) < 1.0 - 1.0e-7:
        raise ValueError("CMSBL mask bounds cannot produce mean-one weights.")
    if not bool(scores.gt(0).any().item()):
        return torch.ones_like(scores).clamp(lower, upper)
    low, high = 0.0, 1.0
    while float(torch.clamp(scores * high, lower, upper).mean()) < 1.0:
        high *= 2.0
    for _ in range(64):
        middle = (low + high) / 2.0
        if float(torch.clamp(scores * middle, lower, upper).mean()) < 1.0:
            low = middle
        else:
            high = middle
    return torch.clamp(scores * ((low + high) / 2.0), lower, upper)


def mask_name(mask_id: int) -> str:
    value = int(mask_id)
    if not 1 <= value <= NUM_NON_EMPTY_MASKS:
        raise ValueError("CMSBL mask ID must be in [1, 15].")
    return "+".join(name for index, name in enumerate(CANONICAL_MODALITIES) if value & (1 << index))




__all__ = [
    "CANONICAL_MODALITIES",
    "NUM_NON_EMPTY_MASKS",
    "accumulate_mask_losses",
    "all_reduce_mask_statistics",
    "auxiliary_schedule_weight",
    "canonical_mask_id",
    "capacity_gap_weights",
    "effective_auxiliary_weights",
    "fusion_mask_ids",
    "hard_mask_weights",
    "load_capacity_reference",
    "mask_name",
    "update_mask_loss_ema",
    "update_metric_ema",
]
