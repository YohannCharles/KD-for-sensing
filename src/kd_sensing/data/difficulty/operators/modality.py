from collections import Counter
from typing import Any

import torch

from kd_sensing.data.missing_mask import sample_pattern_balanced_mask
from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
    DifficultyOperatorConfig,
    DifficultyOperatorOutcome,
    DifficultyProfile,
    DifficultyWarning,
)


MODALITY_BATCH_KEYS = {
    "image": ("image", "images", "image_batch"),
    "radar": ("radar", "radar_ra", "radar_da", "radar_batch"),
    "gps": ("gps", "gps_batch"),
    "lidar": ("lidar", "lidar_batch"),
    "mmwave": ("mmwave", "mmwave_batch"),
    "csi": ("csi", "csi_batch"),
}


class ModalityMissingOperator:
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
        generator = torch.Generator(device="cpu")
        seed = int(context.derived_seed(profile, config))
        generator.manual_seed(seed)
        affected: dict[str, int] = {}
        rates: dict[str, float] = {}
        warnings: list[DifficultyWarning] = []
        fallback_count = 0
        for modality in config.affected_modalities:
            keys = [key for key in MODALITY_BATCH_KEYS.get(modality, (modality,)) if torch.is_tensor(batch.get(key))]
            if not keys:
                fallback_count += 1
                warnings.append(
                    DifficultyWarning(
                        code=f"{modality}_unavailable_for_missing_operator",
                        message=f"{modality} tensor is unavailable; modality_missing operator was skipped.",
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        fallback="skip",
                    )
                )
                continue
            tensor = batch[keys[0]]
            rate = _dropout_rate(self.params, modality, profile.severity)
            rates[modality] = rate
            mask = _dropout_mask(tensor, rate=rate, generator=generator)
            for key in keys:
                batch[key] = _zero_fill(batch[key], mask)
            valid = _merge_valid_mask(batch.get(f"{modality}_valid_mask"), ~mask, device=tensor.device)
            batch[f"{modality}_dropout_mask"] = mask.to(device=tensor.device)
            batch[f"{modality}_missing_mask"] = mask.to(device=tensor.device)
            batch[f"{modality}_valid_mask"] = valid
            affected_count = int(mask.sum().item())
            affected[modality] = affected_count
            if affected_count:
                warnings.append(
                    DifficultyWarning(
                        code=f"{modality}_missing_zero_fill",
                        message=f"Missing {modality} input was represented as zero-filled tensor with valid_mask=false.",
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        sample_count=int(tensor.shape[0]) if tensor.ndim else None,
                        affected_count=affected_count,
                        fallback=str(self.params.get("fallback", profile.fallback or "zero_fill")),
                    )
                )
        batch["missing_modality_metadata"] = {
            "operator": config.type,
            "condition": profile.condition,
            "affected_modalities": list(config.affected_modalities),
            "rates": rates,
            "seed": seed,
            "profile_digest": profile.digest,
            "operator_digest": config.digest,
            "fallback": str(self.params.get("fallback", profile.fallback or "zero_fill")),
            "fallback_count": fallback_count,
            "affected_count": affected,
            "mask_fields": {modality: f"{modality}_valid_mask" for modality in config.affected_modalities},
        }
        return DifficultyOperatorOutcome(
            metadata={
                "rates": rates,
                "seed": seed,
                "fallback_count": fallback_count,
                "affected_count": affected,
                "mask_metadata": "missing_modality_metadata",
            },
            warnings=tuple(warnings),
        )


class RandomModalityDropoutOperator:
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
        modalities = tuple(config.affected_modalities)
        available = [modality for modality in modalities if _tensor_keys(batch, modality)]
        if not available:
            return DifficultyOperatorOutcome(
                metadata={"mode": self.params.get("mode", "random_nonempty_subset"), "fallback_count": len(modalities)}
            )
        reference = batch[_tensor_keys(batch, available[0])[0]]
        batch_size = int(reference.shape[0])
        seq_len = int(reference.shape[1]) if reference.ndim >= 3 else None
        seed = int(context.derived_seed(profile, config))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        mode = str(self.params.get("mode", "random_nonempty_subset")).strip().lower()
        if mode == "pattern_balanced":
            keep, sampled_names, _ = sample_pattern_balanced_mask(
                batch_size,
                available,
                self.params.get("pattern_probs", self.params.get("patterns")),
                ensure_at_least_one=bool(self.params.get("ensure_at_least_one_modality", True)),
                generator=generator,
            )
        else:
            keep = _random_keep_matrix(
                batch_size,
                len(available),
                mode=mode,
                keep_prob=float(self.params.get("keep_prob", 0.75)),
                ensure_at_least_one=bool(self.params.get("ensure_at_least_one_modality", True)),
                generator=generator,
            )
            sampled_names = None
        stats = _pattern_stats(available, keep, epoch=context.epoch, sampled_names=sampled_names)
        affected: dict[str, int] = {}
        for index, modality in enumerate(available):
            sample_missing = ~keep[:, index]
            mask = sample_missing
            if seq_len is not None:
                mask = sample_missing.unsqueeze(1).expand(batch_size, seq_len)
            keys = _tensor_keys(batch, modality)
            tensor = batch[keys[0]]
            for key in keys:
                batch[key] = _zero_fill(batch[key], mask.to(device=batch[key].device))
            valid = _merge_valid_mask(batch.get(f"{modality}_valid_mask"), ~mask, device=tensor.device)
            batch[f"{modality}_dropout_mask"] = mask.to(device=tensor.device)
            batch[f"{modality}_missing_mask"] = mask.to(device=tensor.device)
            batch[f"{modality}_valid_mask"] = valid
            affected[modality] = int(mask.sum().item())
        batch["random_dropout_pattern_stats"] = stats
        batch["missing_modality_metadata"] = {
            "operator": config.type,
            "mode": mode,
            "condition": profile.condition,
            "affected_modalities": list(available),
            "seed": seed,
            "profile_digest": profile.digest,
            "operator_digest": config.digest,
            "fallback": str(self.params.get("fallback", profile.fallback or "zero_fill")),
            "fallback_count": len(modalities) - len(available),
            "affected_count": affected,
            "mask_fields": {modality: f"{modality}_valid_mask" for modality in available},
            "pattern_stats": stats,
        }
        return DifficultyOperatorOutcome(
            metadata={
                "mode": mode,
                "keep_prob": float(self.params.get("keep_prob", 0.75)),
                "ensure_at_least_one_modality": bool(self.params.get("ensure_at_least_one_modality", True)),
                "seed": seed,
                "fallback_count": len(modalities) - len(available),
                "affected_count": affected,
                "pattern_stats": stats,
            }
        )


def _dropout_rate(params: dict[str, Any], modality: str, severity: float) -> float:
    rates = params.get("rates")
    if isinstance(rates, dict) and modality in rates:
        raw = rates[modality]
    else:
        raw = params.get(
            f"{modality}_dropout_rate",
            params.get(f"{modality}_dropout_prob", params.get("dropout_rate", params.get("dropout_prob", severity))),
        )
    return max(0.0, min(float(raw), 1.0))


def _tensor_keys(batch: dict[str, Any], modality: str) -> list[str]:
    return [key for key in MODALITY_BATCH_KEYS.get(modality, (modality,)) if torch.is_tensor(batch.get(key))]


def _random_keep_matrix(
    batch_size: int,
    modality_count: int,
    *,
    mode: str,
    keep_prob: float,
    ensure_at_least_one: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    if mode == "bernoulli":
        keep = torch.rand((batch_size, modality_count), generator=generator) < max(0.0, min(float(keep_prob), 1.0))
    elif mode in {"drop_one_available", "mask_one_available", "random_single_missing"}:
        keep = torch.ones((batch_size, modality_count), dtype=torch.bool)
        if modality_count > 1:
            missing = torch.randint(0, modality_count, (batch_size,), generator=generator)
            keep[torch.arange(batch_size), missing] = False
    elif mode == "random_nonempty_subset":
        choices = torch.randint(1, 2**modality_count, (batch_size,), generator=generator)
        bits = 2 ** torch.arange(modality_count)
        keep = (choices.unsqueeze(1) & bits.unsqueeze(0)).bool()
    else:
        raise ValueError(
            "random_modality_dropout mode must be 'bernoulli', 'drop_one_available', "
            "'random_nonempty_subset', or 'pattern_balanced'."
        )
    if ensure_at_least_one:
        empty = ~keep.any(dim=1)
        if empty.any():
            replacement = torch.randint(0, modality_count, (int(empty.sum().item()),), generator=generator)
            keep[empty] = False
            keep[empty, replacement] = True
    return keep


def _pattern_stats(
    modalities: list[str],
    keep: torch.Tensor,
    *,
    epoch: int | None,
    sampled_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, int]] = Counter()
    for index, row in enumerate(keep):
        available = [modality for modality, visible in zip(modalities, row.tolist()) if bool(visible)]
        name = sampled_names[index] if sampled_names is not None else "available:" + "+".join(available)
        counts[(name, len(modalities) - len(available))] += 1
    total = max(int(keep.shape[0]), 1)
    return [
        {
            "epoch": "" if epoch is None else int(epoch),
            "pattern_or_available_set": name,
            "num_samples": count,
            "fraction": float(count / total),
            "missing_count": missing_count,
        }
        for (name, missing_count), count in sorted(counts.items())
    ]


def _dropout_mask(tensor: torch.Tensor, *, rate: float, generator: torch.Generator) -> torch.Tensor:
    if tensor.ndim >= 3:
        shape = tuple(int(value) for value in tensor.shape[:2])
    else:
        shape = (int(tensor.shape[0]),)
    return torch.rand(shape, generator=generator, dtype=torch.float32).to(tensor.device) < float(rate)


def _zero_fill(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    view = mask.reshape(*mask.shape, *([1] * (tensor.ndim - mask.ndim)))
    return torch.where(view.to(device=tensor.device), torch.zeros_like(tensor), tensor)


def _merge_valid_mask(existing: Any, valid: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    valid = valid.to(device=device, dtype=torch.bool)
    if existing is None:
        return valid
    current = torch.as_tensor(existing, dtype=torch.bool, device=device)
    if current.ndim == 1 and valid.ndim == 2:
        current = current.unsqueeze(1).expand_as(valid)
    if current.shape != valid.shape:
        return valid
    return current & valid
