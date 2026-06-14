from __future__ import annotations

import math
from typing import Any

import torch

from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
    DifficultyOperatorConfig,
    DifficultyOperatorOutcome,
    DifficultyProfile,
    DifficultyWarning,
)


class _BaseImageOperator:
    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def _generator(
        self,
        profile: DifficultyProfile,
        config: DifficultyOperatorConfig,
        context: DifficultyContext,
    ) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(context.derived_seed(profile, config)))
        return generator

    def _image_key(self, batch: dict[str, Any]) -> str | None:
        if "image" in batch:
            return "image"
        if "images" in batch:
            return "images"
        return None

    def _missing_image(
        self,
        profile: DifficultyProfile,
        config: DifficultyOperatorConfig,
    ) -> DifficultyOperatorOutcome:
        return DifficultyOperatorOutcome(
            warnings=(
                DifficultyWarning(
                    code="missing_image",
                    message="Image tensor is unavailable; image difficulty operator was skipped.",
                    profile_id=profile.id,
                    operator=config.type,
                    condition=profile.condition,
                    severity=float(profile.severity),
                    fallback="skip",
                ),
            )
        )


class ImageCleanOperator(_BaseImageOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        return DifficultyOperatorOutcome(metadata={"operator": config.type, "modality": "image", "state": "clean"})


class ImageFogRainOperator(_BaseImageOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        key = self._image_key(batch)
        image = batch.get(key) if key else None
        if not torch.is_tensor(image):
            return self._missing_image(profile, config)
        generator = self._generator(profile, config, context)
        original_dtype = image.dtype
        value = image.to(dtype=torch.float32)
        severity = max(0.0, float(profile.severity))
        alpha = min(0.85, float(self.params.get("alpha_scale", 0.65)) * severity)
        center = torch.full_like(value, float(self.params.get("fog_value", 0.75)))
        value = value * (1.0 - alpha) + center * alpha
        if severity > 0:
            value = _add_rain_streaks(value, generator=generator, strength=severity)
        batch[str(key)] = value.to(dtype=original_dtype)
        _attach_image_metadata(batch, config, profile, frame_range=_frame_range(image), parameters={"alpha": alpha})
        return DifficultyOperatorOutcome(metadata={"input_space": "normalized_image_tensor", "alpha": alpha, "frame_range": _frame_range(image)})


class ImageNightOperator(_BaseImageOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        key = self._image_key(batch)
        image = batch.get(key) if key else None
        if not torch.is_tensor(image):
            return self._missing_image(profile, config)
        original_dtype = image.dtype
        severity = max(0.0, float(profile.severity))
        brightness = max(0.05, 1.0 - 0.75 * severity)
        batch[str(key)] = (image.to(dtype=torch.float32) * brightness).to(dtype=original_dtype)
        _attach_image_metadata(batch, config, profile, frame_range=_frame_range(image), parameters={"brightness_scale": brightness})
        return DifficultyOperatorOutcome(
            metadata={"input_space": "normalized_image_tensor", "brightness_scale": brightness, "frame_range": _frame_range(image)}
        )


class ImageOcclusionOperator(_BaseImageOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        key = self._image_key(batch)
        image = batch.get(key) if key else None
        if not torch.is_tensor(image):
            return self._missing_image(profile, config)
        generator = self._generator(profile, config, context)
        original_dtype = image.dtype
        value, mask = _apply_rectangular_occlusion(image.to(dtype=torch.float32), severity=float(profile.severity), generator=generator)
        batch[str(key)] = value.to(dtype=original_dtype)
        batch["image_occlusion_mask"] = mask.to(device=image.device)
        _attach_image_metadata(batch, config, profile, frame_range=_frame_range(image), parameters={"mask": "image_occlusion_mask"})
        return DifficultyOperatorOutcome(
            metadata={"input_space": "normalized_image_tensor", "frame_range": _frame_range(image), "mask": "image_occlusion_mask"}
        )


class ImageMotionBlurOperator(_BaseImageOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        key = self._image_key(batch)
        image = batch.get(key) if key else None
        if not torch.is_tensor(image):
            return self._missing_image(profile, config)
        original_dtype = image.dtype
        radius = max(1, int(round(float(profile.severity) * float(self.params.get("radius_scale", 4.0)))))
        value = _apply_motion_blur(image.to(dtype=torch.float32), radius=radius)
        batch[str(key)] = value.to(dtype=original_dtype)
        _attach_image_metadata(batch, config, profile, frame_range=_frame_range(image), parameters={"radius": radius})
        return DifficultyOperatorOutcome(
            metadata={"input_space": "normalized_image_tensor", "frame_range": _frame_range(image), "radius": radius}
        )


def _attach_image_metadata(
    batch: dict[str, Any],
    config: DifficultyOperatorConfig,
    profile: DifficultyProfile,
    *,
    frame_range: list[int],
    parameters: dict[str, Any],
) -> None:
    batch["image_degradation_metadata"] = {
        "operator": config.type,
        "condition": profile.condition,
        "severity": float(profile.severity),
        "input_space": "normalized_image_tensor",
        "frame_range": frame_range,
        "parameters": dict(parameters),
    }


def _frame_range(image: torch.Tensor) -> list[int]:
    if image.ndim >= 5:
        return [0, max(int(image.shape[1]) - 1, 0)]
    return [0, 0]


def _add_rain_streaks(value: torch.Tensor, *, generator: torch.Generator, strength: float) -> torch.Tensor:
    if value.ndim < 4:
        return value
    out = value.clone()
    width = int(value.shape[-1])
    if width <= 0:
        return out
    count = max(1, int(round(width * min(0.25, 0.06 * float(strength)))))
    columns = torch.randint(0, width, (count,), generator=generator)
    out[..., columns] = torch.maximum(out[..., columns], torch.full_like(out[..., columns], 0.85))
    return out


def _apply_rectangular_occlusion(
    value: torch.Tensor,
    *,
    severity: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim < 4:
        return value, torch.zeros(value.shape[:1], dtype=torch.bool, device=value.device)
    out = value.clone()
    mask = torch.zeros_like(value, dtype=torch.bool)
    height = int(value.shape[-2])
    width = int(value.shape[-1])
    if height <= 0 or width <= 0:
        return out, mask
    area_ratio = max(0.0, min(float(severity), 1.0))
    side = math.sqrt(area_ratio)
    block_h = max(1, min(height, int(round(height * side))))
    block_w = max(1, min(width, int(round(width * side))))
    y = int(torch.randint(0, max(height - block_h + 1, 1), (1,), generator=generator).item())
    x = int(torch.randint(0, max(width - block_w + 1, 1), (1,), generator=generator).item())
    out[..., y : y + block_h, x : x + block_w] = 0.0
    mask[..., y : y + block_h, x : x + block_w] = True
    return out, mask


def _apply_motion_blur(value: torch.Tensor, *, radius: int) -> torch.Tensor:
    if value.ndim < 4 or radius <= 0:
        return value
    chunks = [value]
    for offset in range(1, radius + 1):
        chunks.append(torch.roll(value, shifts=offset, dims=-1))
        chunks.append(torch.roll(value, shifts=-offset, dims=-1))
    return torch.stack(chunks, dim=0).mean(dim=0)
