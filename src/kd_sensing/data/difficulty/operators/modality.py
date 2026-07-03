from typing import Any

import torch

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
