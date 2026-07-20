from __future__ import annotations

import hashlib
import random
from itertools import combinations
from typing import Any, Sequence

import torch


PCER_MASK_TYPES = (
    "full",
    "sparse_easy",
    "single_modality_burst2",
    "single_modality_missing",
    "latest_sync_missing",
    "two_modality_recent_async",
)
PCER_STABLE_PROBABILITIES = {
    "full": 0.20,
    "sparse_easy": 0.20,
    "single_modality_burst2": 0.20,
    "single_modality_missing": 0.15,
    "latest_sync_missing": 0.15,
    "two_modality_recent_async": 0.10,
}
PCER_WARMUP_PROBABILITIES = {"full": 0.50, "sparse_easy": 0.50}
PCER_TRANSITION_PROBABILITIES = {
    "full": 0.30,
    "sparse_easy": 0.30,
    "single_modality_burst2": 0.25,
    "single_modality_missing": 0.15,
}


class TemporalBlockMaskGenerator:
    """Deterministic modality-time availability masks with optional frame grouping."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def __call__(
        self,
        *,
        batch_size: int,
        num_modalities: int,
        num_timesteps: int,
        sample_ids: Sequence[Any],
        mask_type: str | Sequence[str],
        severity: Any = None,
        seed: int | None = None,
        training: bool,
        source_frame_ids: Any = None,
        variant_ids: int | Sequence[int] = 0,
    ) -> dict[str, Any]:
        batch = int(batch_size)
        modalities = int(num_modalities)
        timesteps = int(num_timesteps)
        if min(batch, modalities, timesteps) <= 0:
            raise ValueError("batch_size, num_modalities, and num_timesteps must be positive.")
        identities = tuple(str(value) for value in sample_ids)
        if len(identities) != batch or any(not value for value in identities):
            raise ValueError(f"sample_ids must contain {batch} non-empty values.")
        mask_types = _expand(mask_type, batch, name="mask_type", convert=str)
        unknown = sorted(set(mask_types) - set(PCER_MASK_TYPES))
        if unknown:
            raise ValueError(f"Unsupported PCER mask types: {unknown}.")
        variants = _expand(variant_ids, batch, name="variant_ids", convert=int)
        groups = _source_frame_groups(source_frame_ids, batch, modalities, timesteps)
        effective_seed = self.seed if seed is None else int(seed)

        availability = torch.ones(batch, modalities, timesteps, dtype=torch.bool)
        metadata = []
        for index, (identity, kind, variant) in enumerate(zip(identities, mask_types, variants)):
            rng = random.Random(
                _derived_seed(effective_seed, identity, kind, int(variant), "train" if training else "eval")
            )
            item = availability[index]
            details = _apply_mask(item, kind, rng=rng, variant=int(variant), training=bool(training))
            grouped = _apply_source_frame_groups(item, groups[index]) if groups is not None else 0
            if not bool(item.any().item()):
                raise ValueError(f"PCER mask {kind!r} removed every block for sample {identity!r}.")
            metadata.append(
                {
                    "sample_id": identity,
                    "mask_type": kind,
                    "variant_id": int(variant),
                    "severity": severity,
                    "training": bool(training),
                    "grouped_replica_count": grouped,
                    **details,
                }
            )
        return {
            "availability_mask": availability,
            "mask_type": mask_types[0] if len(set(mask_types)) == 1 else list(mask_types),
            "mask_metadata": metadata,
        }


def pcer_curriculum_probabilities(epoch: int, total_epochs: int) -> dict[str, float]:
    total = int(total_epochs)
    current = int(epoch)
    if total <= 0 or current < 0:
        raise ValueError("epoch must be non-negative and total_epochs must be positive.")
    progress = current / float(total)
    if progress < 0.10:
        return dict(PCER_WARMUP_PROBABILITIES)
    if progress < 0.30:
        return dict(PCER_TRANSITION_PROBABILITIES)
    return dict(PCER_STABLE_PROBABILITIES)


def sample_pcer_curriculum_mask_types(
    sample_ids: Sequence[Any],
    *,
    seed: int,
    epoch: int,
    total_epochs: int,
) -> list[str]:
    probabilities = pcer_curriculum_probabilities(epoch, total_epochs)
    names = tuple(probabilities)
    cumulative = []
    total = 0.0
    for name in names:
        value = float(probabilities[name])
        if value < 0.0:
            raise ValueError("PCER curriculum probabilities must be non-negative.")
        total += value
        cumulative.append(total)
    if abs(total - 1.0) > 1e-9:
        raise ValueError("PCER curriculum probabilities must sum to one.")
    result = []
    for identity in sample_ids:
        draw = random.Random(_derived_seed(seed, str(identity), "pcer-curriculum", int(epoch))).random()
        result.append(next(name for name, threshold in zip(names, cumulative) if draw <= threshold))
    return result


def _apply_mask(
    mask: torch.Tensor,
    kind: str,
    *,
    rng: random.Random,
    variant: int,
    training: bool,
) -> dict[str, Any]:
    modalities, timesteps = (int(value) for value in mask.shape)
    if kind == "full":
        return {"dropped_blocks": []}
    if kind == "sparse_easy":
        candidates = [(m, t) for m in range(modalities) for t in range(max(timesteps - 1, 1))]
        pairs = [
            pair
            for pair in combinations(candidates, 2)
            if pair[0][0] != pair[1][0]
            and (pair[0][1] != pair[1][1] or timesteps == 1)
            and all(sum(cell[0] == modality for cell in pair) < timesteps for modality in range(modalities))
        ]
        if not pairs:
            pairs = [pair for pair in combinations(candidates, 2)]
        if not pairs:
            raise ValueError("sparse_easy requires at least two modality-time blocks.")
        selected = pairs[rng.randrange(len(pairs))] if training else pairs[variant % len(pairs)]
        for modality, time in selected:
            mask[modality, time] = False
        return {"dropped_blocks": [list(item) for item in selected]}
    if kind == "single_modality_burst2":
        length = min(2, timesteps)
        starts = tuple(range(timesteps - length + 1))
        choice = rng.randrange(modalities * len(starts)) if training else variant % (modalities * len(starts))
        modality, start_index = divmod(choice, len(starts))
        start = starts[start_index]
        mask[modality, start : start + length] = False
        return {"modalities": [modality], "burst_starts": [start], "burst_length": length}
    if kind == "single_modality_missing":
        modality = rng.randrange(modalities) if training else variant % modalities
        mask[modality] = False
        return {"modalities": [modality]}
    if kind == "latest_sync_missing":
        if timesteps < 2:
            raise ValueError("latest_sync_missing requires at least two timesteps.")
        mask[:, -1] = False
        return {"dropped_time": timesteps - 1}
    if kind == "two_modality_recent_async":
        if modalities < 2 or timesteps < 3:
            raise ValueError("two_modality_recent_async requires at least two modalities and three timesteps.")
        pairs = tuple(combinations(range(modalities), 2))
        choice = rng.randrange(len(pairs) * 2) if training else variant % (len(pairs) * 2)
        pair = pairs[choice // 2]
        latest, previous = (timesteps - 2, timesteps - 3)
        starts = (latest, previous) if choice % 2 == 0 else (previous, latest)
        mask[pair[0], starts[0] : starts[0] + 2] = False
        mask[pair[1], starts[1] : starts[1] + 2] = False
        return {"modalities": list(pair), "burst_starts": list(starts), "burst_length": 2}
    raise AssertionError(f"Unhandled PCER mask type {kind!r}.")


def _source_frame_groups(value: Any, batch: int, modalities: int, timesteps: int) -> list[list[list[Any]]] | None:
    if value is None:
        return None
    raw = value.detach().cpu().tolist() if torch.is_tensor(value) else value
    if len(raw) != batch:
        raise ValueError(f"source_frame_ids must contain {batch} samples.")
    for sample in raw:
        if len(sample) != modalities or any(len(row) != timesteps for row in sample):
            raise ValueError(f"source_frame_ids must have shape [{batch},{modalities},{timesteps}].")
    return raw


def _apply_source_frame_groups(mask: torch.Tensor, groups: list[list[Any]]) -> int:
    grouped = 0
    for modality, identities in enumerate(groups):
        for identity in {str(value) for value in identities if value not in (None, "", -1)}:
            indices = [index for index, value in enumerate(identities) if str(value) == identity]
            if len(indices) > 1 and not bool(mask[modality, indices].all().item()):
                grouped += int(mask[modality, indices].sum().item())
                mask[modality, indices] = False
    return grouped


def _expand(value: Any, count: int, *, name: str, convert) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(convert(item) for item in value)
    else:
        result = (convert(value),) * count
    if len(result) != count:
        raise ValueError(f"{name} must contain {count} values.")
    return result


def _derived_seed(seed: int, *parts: Any) -> int:
    payload = ":".join((str(int(seed)), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "PCER_MASK_TYPES",
    "PCER_STABLE_PROBABILITIES",
    "PCER_TRANSITION_PROBABILITIES",
    "PCER_WARMUP_PROBABILITIES",
    "TemporalBlockMaskGenerator",
    "pcer_curriculum_probabilities",
    "sample_pcer_curriculum_mask_types",
]
