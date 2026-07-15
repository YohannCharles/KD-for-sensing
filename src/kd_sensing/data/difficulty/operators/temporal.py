from typing import Any

import torch

from kd_sensing.data.difficulty.operators.modality import MODALITY_BATCH_KEYS, _merge_valid_mask, _zero_fill
from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
    DifficultyOperatorConfig,
    DifficultyOperatorOutcome,
    DifficultyProfile,
    DifficultyWarning,
)
from kd_sensing.data.temporal_missing import (
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    normalize_temporal_missing_mode,
    sample_stratified_modality_temporal_mask,
)
from kd_sensing.modalities import normalize_modalities


TEMPORAL_SUPERSET_PAYLOAD_KEY = "temporal_superset_payload"


class TemporalMissingOperator:
    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        mode = normalize_temporal_missing_mode(self.params.get("mode", self.params.get("temporal_missing_mode", "none")))
        modalities = tuple(_available_modalities(batch, config.affected_modalities))
        if not modalities:
            return DifficultyOperatorOutcome(metadata={"mode": mode, "fallback_count": len(config.affected_modalities)})
        batch_size, steps = _batch_time_shape(batch, modalities)
        if batch_size <= 0 or steps <= 0:
            return DifficultyOperatorOutcome(metadata={"mode": mode, "fallback_count": len(modalities)})
        base = torch.ones(batch_size, steps, len(modalities), dtype=torch.bool)
        for index, modality in enumerate(modalities):
            valid = _existing_valid_mask(batch, modality, batch_size=batch_size, steps=steps)
            if valid is not None:
                base[:, :, index] &= valid.cpu()
        preserve_superset = bool(self.params.get("preserve_unmasked_for_superset", False))
        if preserve_superset and not bool(base.any(dim=(1, 2)).all().item()):
            raise ValueError("Temporal superset preservation requires at least one valid history cell per sample.")
        original_inputs = (
            {
                key: batch[key]
                for modality in modalities
                for key in _tensor_keys(batch, modality)
            }
            if preserve_superset
            else None
        )

        generator = torch.Generator(device="cpu")
        seed = int(context.derived_seed(profile, config))
        generator.manual_seed(seed)
        stratified_stats: dict[str, Any] = {}
        if mode == "stratified_modality_temporal":
            temporal_keep, stratified_stats = _sample_stratified_keep(
                batch_size=batch_size,
                steps=steps,
                modalities=modalities,
                params=self.params,
                generator_seed=seed,
            )
        else:
            temporal_keep = _sample_temporal_keep(
                mode,
                batch_size=batch_size,
                steps=steps,
                modalities=len(modalities),
                prob=float(self.params.get("prob", self.params.get("temporal_missing_prob", profile.severity))),
                block_len=int(self.params.get("block_len", self.params.get("temporal_missing_block_len", 1))),
                generator=generator,
            )
        combined = base & temporal_keep
        fixed = _apply_fallbacks(
            combined,
            base,
            ensure_at_least_one_frame=bool(self.params.get("ensure_at_least_one_frame", True)),
            ensure_at_least_one_modality_per_frame=bool(
                self.params.get("ensure_at_least_one_modality_per_frame", False)
            ),
        )
        combined = fixed.mask

        affected: dict[str, int] = {}
        for index, modality in enumerate(modalities):
            keep = combined[:, :, index]
            missing = ~keep
            for key in _tensor_keys(batch, modality):
                batch[key] = _zero_fill(batch[key], missing.to(device=batch[key].device))
            reference = batch[_tensor_keys(batch, modality)[0]]
            device_keep = keep.to(device=reference.device)
            batch[f"{modality}_valid_mask"] = _merge_valid_mask(
                batch.get(f"{modality}_valid_mask"),
                device_keep,
                device=reference.device,
            )
            batch[f"{modality}_dropout_mask"] = (~device_keep).to(device=reference.device)
            batch[f"{modality}_missing_mask"] = (~device_keep).to(device=reference.device)
            affected[modality] = int(missing.sum().item())

        temporal_mask = combined.any(dim=2)
        available_modalities = combined.any(dim=1)
        batch["temporal_mask"] = temporal_mask
        batch["modality_temporal_mask"] = combined
        batch["available_modalities"] = available_modalities
        batch["temporal_missing_modalities"] = list(modalities)
        if original_inputs is not None:
            batch[TEMPORAL_SUPERSET_PAYLOAD_KEY] = {
                "inputs": original_inputs,
                "base_mask": base,
                "modalities": modalities,
            }
        metadata = {
            "operator": config.type,
            "mode": mode,
            "prob": float(self.params.get("prob", self.params.get("temporal_missing_prob", profile.severity))),
            "block_len": int(self.params.get("block_len", self.params.get("temporal_missing_block_len", 1))),
            "seed": seed,
            "modalities": list(modalities),
            "temporal_available_rate": float(temporal_mask.to(torch.float32).mean().item()),
            "modality_temporal_available_rate": float(combined.to(torch.float32).mean().item()),
            "num_all_missing_fixed": fixed.num_all_missing_fixed,
            "num_empty_frame_fixed": fixed.num_empty_frame_fixed,
            "affected_count": affected,
            "preserve_unmasked_for_superset": preserve_superset,
        }
        metadata.update(stratified_stats)
        batch["temporal_missing_metadata"] = metadata
        warnings = ()
        if fixed.num_all_missing_fixed:
            warnings = (
                DifficultyWarning(
                    code="temporal_missing_all_missing_fixed",
                    message="Temporal missing fallback restored at least one modality-time observation.",
                    profile_id=profile.id,
                    operator=config.type,
                    condition=profile.condition,
                    severity=float(profile.severity),
                    affected_count=fixed.num_all_missing_fixed,
                    fallback="restore_last_available",
                ),
            )
        return DifficultyOperatorOutcome(metadata=metadata, warnings=warnings)


class _FallbackResult(tuple):
    __slots__ = ()

    @property
    def mask(self) -> torch.Tensor:
        return self[0]

    @property
    def num_all_missing_fixed(self) -> int:
        return self[1]

    @property
    def num_empty_frame_fixed(self) -> int:
        return self[2]


def _sample_temporal_keep(
    mode: str,
    *,
    batch_size: int,
    steps: int,
    modalities: int,
    prob: float,
    block_len: int,
    generator: torch.Generator,
) -> torch.Tensor:
    prob = max(0.0, min(float(prob), 1.0))
    if mode == "none" or prob <= 0.0:
        return torch.ones(batch_size, steps, modalities, dtype=torch.bool)
    if mode == "frame_bernoulli":
        return torch.rand(batch_size, steps, 1, generator=generator).ge(prob).expand(-1, -1, modalities).clone()
    if mode == "modality_frame_bernoulli":
        return torch.rand(batch_size, steps, modalities, generator=generator).ge(prob)
    if mode == "block":
        keep = torch.ones(batch_size, steps, modalities, dtype=torch.bool)
        length = max(1, min(int(block_len), int(steps)))
        max_start = max(int(steps) - length, 0)
        apply = torch.rand(batch_size, generator=generator).lt(prob)
        starts = torch.randint(max_start + 1, (batch_size,), generator=generator)
        for row in range(batch_size):
            if bool(apply[row].item()):
                start = int(starts[row].item())
                keep[row, start : start + length, :] = False
        return keep
    raise ValueError(f"Unsupported temporal missing mode {mode!r}.")


def _sample_stratified_keep(
    *,
    batch_size: int,
    steps: int,
    modalities: tuple[str, ...],
    params: dict[str, Any],
    generator_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    import random

    rng = random.Random(int(generator_seed))
    drop_counts = parse_csv_ints(params.get("train_missing_drop_counts"), (0, 1, 2, 3))
    rates = parse_csv_floats(params.get("train_temporal_missing_rates"), (0.0, 0.2, 0.4, 0.6, 0.8))
    types = parse_csv_strings(
        params.get("train_temporal_missing_types"),
        ("modality_level", "frame_level", "modality_frame", "block"),
    )
    masks = []
    dropped_counter: dict[str, int] = {}
    type_counter: dict[str, int] = {}
    rate_counter: dict[str, int] = {}
    fallback_total = 0
    for _ in range(batch_size):
        item = sample_stratified_modality_temporal_mask(
            history_window=steps,
            modalities=modalities,
            drop_counts=drop_counts,
            temporal_missing_rates=rates,
            temporal_missing_types=types,
            rng=rng,
            ensure_at_least_one_cell=bool(params.get("ensure_at_least_one_cell", True)),
            ensure_at_least_one_frame=bool(params.get("ensure_at_least_one_frame", True)),
            ensure_at_least_one_modality=bool(params.get("ensure_at_least_one_modality", True)),
        )
        masks.append(item["modality_temporal_mask"])
        dropped_counter[str(item["drop_count"])] = dropped_counter.get(str(item["drop_count"]), 0) + 1
        type_counter[str(item["mask_type"])] = type_counter.get(str(item["mask_type"]), 0) + 1
        rate_counter[f"{float(item['rate']):g}"] = rate_counter.get(f"{float(item['rate']):g}", 0) + 1
        fallback_total += int(item.get("num_fallback_fixes", 0))
    return torch.stack(masks, dim=0), {
        "mask_sampler": "stratified_modality_temporal",
        "stratified_drop_count_hist": dropped_counter,
        "stratified_missing_type_hist": type_counter,
        "stratified_rate_hist": rate_counter,
        "stratified_sampler_fallback_fixes": fallback_total,
    }


def _apply_fallbacks(
    mask: torch.Tensor,
    base: torch.Tensor,
    *,
    ensure_at_least_one_frame: bool,
    ensure_at_least_one_modality_per_frame: bool,
) -> _FallbackResult:
    result = mask.clone()
    empty_frame_fixed = 0
    if ensure_at_least_one_modality_per_frame:
        empty_frames = ~result.any(dim=2)
        for row, step in empty_frames.nonzero(as_tuple=False).tolist():
            candidates = base[row, step].nonzero(as_tuple=False).flatten()
            if int(candidates.numel()) <= 0:
                continue
            result[row, step, int(candidates[0].item())] = True
            empty_frame_fixed += 1
    all_missing_fixed = 0
    if ensure_at_least_one_frame:
        empty_samples = ~result.any(dim=(1, 2))
        for row in empty_samples.nonzero(as_tuple=False).flatten().tolist():
            candidates = base[int(row)].nonzero(as_tuple=False)
            if int(candidates.numel()) > 0:
                step, modality = candidates[-1].tolist()
            else:
                step, modality = int(result.shape[1]) - 1, 0
            result[int(row), int(step), int(modality)] = True
            all_missing_fixed += 1
    return _FallbackResult((result, all_missing_fixed, empty_frame_fixed))


def _available_modalities(batch: dict[str, Any], configured: tuple[str, ...]) -> list[str]:
    try:
        candidates = normalize_modalities(tuple(configured), context="temporal missing affected_modalities")
    except ValueError:
        candidates = tuple(str(item) for item in configured)
    return [modality for modality in candidates if _tensor_keys(batch, modality)]


def _tensor_keys(batch: dict[str, Any], modality: str) -> list[str]:
    return [key for key in MODALITY_BATCH_KEYS.get(modality, (modality,)) if torch.is_tensor(batch.get(key))]


def _batch_time_shape(batch: dict[str, Any], modalities: tuple[str, ...]) -> tuple[int, int]:
    for modality in modalities:
        keys = _tensor_keys(batch, modality)
        if not keys:
            continue
        tensor = batch[keys[0]]
        if tensor.ndim >= 2:
            return int(tensor.shape[0]), int(tensor.shape[1])
    return 0, 0


def _existing_valid_mask(batch: dict[str, Any], modality: str, *, batch_size: int, steps: int) -> torch.Tensor | None:
    valid = batch.get(f"{modality}_valid_mask")
    dropout = batch.get(f"{modality}_dropout_mask", batch.get(f"{modality}_missing_mask"))
    result = _coerce_mask(valid, batch_size=batch_size, steps=steps) if torch.is_tensor(valid) else None
    if torch.is_tensor(dropout):
        keep = ~_coerce_mask(dropout, batch_size=batch_size, steps=steps)
        result = keep if result is None else result & keep
    return result


def _coerce_mask(value: torch.Tensor, *, batch_size: int, steps: int) -> torch.Tensor:
    mask = value.detach().to(device="cpu", dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(1).expand(batch_size, steps)
    if mask.ndim != 2:
        raise ValueError(f"Temporal reliability mask must have shape [B] or [B,T], got {tuple(value.shape)}.")
    if tuple(mask.shape) != (batch_size, steps):
        raise ValueError(f"Temporal reliability mask must have shape [{batch_size}, {steps}], got {tuple(value.shape)}.")
    return mask


__all__ = ["TEMPORAL_SUPERSET_PAYLOAD_KEY", "TemporalMissingOperator"]
