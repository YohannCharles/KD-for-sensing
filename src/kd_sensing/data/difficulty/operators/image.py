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


class ImageObservabilityTransform(_BaseImageOperator):
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
        if image.ndim < 4:
            return DifficultyOperatorOutcome(
                warnings=(
                    DifficultyWarning(
                        code="image_observability_invalid_shape",
                        message=(
                            "Image observability transform expects image tensors with batch/time/spatial dimensions; "
                            f"got shape {tuple(image.shape)}."
                        ),
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        fallback="skip",
                    ),
                )
            )
        generator = self._generator(profile, config, context)
        seed = int(context.derived_seed(profile, config))
        original_dtype = image.dtype
        value = image.to(dtype=torch.float32)
        batch_size, steps = _image_time_shape(value)
        device = image.device
        params = dict(self.params)
        condition = str(params.get("condition", profile.condition))

        valid_mask = torch.ones((batch_size, steps), dtype=torch.bool, device=device)
        dropout_mask = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
        burst_mask = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
        score = torch.ones((batch_size, steps), dtype=torch.float32, device=device)
        corruption_types: list[str] = []
        counts: dict[str, int] = {}

        weather = _prob(params.get("image_weather_severity", 0.0), name="image_weather_severity")
        if weather > 0.0:
            alpha = min(0.85, float(params.get("weather_alpha_scale", 0.65)) * weather)
            center = torch.full_like(value, float(params.get("fog_value", 0.75)))
            value = value * (1.0 - alpha) + center * alpha
            value = _add_rain_streaks(value, generator=generator, strength=weather)
            score = (score - 0.15 * weather).clamp(0.0, 1.0)
            corruption_types.append("weather")
            counts["weather_frames"] = int(batch_size * steps)

        lowlight_mask = _random_frame_mask(
            batch_size,
            steps,
            probability=_prob(params.get("image_lowlight_prob", 0.0), name="image_lowlight_prob"),
            generator=generator,
            device=device,
        )
        if bool(lowlight_mask.any()):
            severity = _prob(params.get("image_lowlight_severity", 0.5), name="image_lowlight_severity")
            scale = max(0.05, 1.0 - 0.75 * severity)
            value = torch.where(_expand_temporal_mask(lowlight_mask, value), value * scale, value)
            score = torch.where(lowlight_mask, (score - 0.20 * severity).clamp(0.0, 1.0), score)
            corruption_types.append("low_light")
            counts["lowlight_frames"] = int(lowlight_mask.sum().item())

        blur_mask = _random_frame_mask(
            batch_size,
            steps,
            probability=_prob(params.get("image_blur_prob", 0.0), name="image_blur_prob"),
            generator=generator,
            device=device,
        )
        if bool(blur_mask.any()):
            radius = max(1, int(round(float(params.get("image_blur_radius", 2)))))
            blurred = _apply_motion_blur(value, radius=radius)
            value = torch.where(_expand_temporal_mask(blur_mask, value), blurred, value)
            score = torch.where(blur_mask, (score - 0.18 * min(radius / 4.0, 1.0)).clamp(0.0, 1.0), score)
            corruption_types.append("motion_blur")
            counts["blur_frames"] = int(blur_mask.sum().item())

        occlusion_mask = _random_frame_mask(
            batch_size,
            steps,
            probability=_prob(params.get("image_occlusion_prob", 0.0), name="image_occlusion_prob"),
            generator=generator,
            device=device,
        )
        occlusion_ratio = _prob(params.get("image_occlusion_ratio", 0.0), name="image_occlusion_ratio")
        pixel_occlusion_mask: torch.Tensor | None = None
        if bool(occlusion_mask.any()) and occlusion_ratio > 0.0:
            value, pixel_occlusion_mask = _apply_temporal_occlusion(
                value,
                frame_mask=occlusion_mask,
                ratio=occlusion_ratio,
                generator=generator,
            )
            score = torch.where(occlusion_mask, (score - 0.45 * occlusion_ratio).clamp(0.0, 1.0), score)
            corruption_types.append("partial_occlusion")
            counts["occlusion_frames"] = int(occlusion_mask.sum().item())
            batch["image_occlusion_mask"] = pixel_occlusion_mask.to(device=device)

        dropout_mask = _random_frame_mask(
            batch_size,
            steps,
            probability=_prob(params.get("image_dropout_prob", 0.0), name="image_dropout_prob"),
            generator=generator,
            device=device,
        )
        burst_mask = _burst_frame_mask(
            batch_size,
            steps,
            probability=_prob(params.get("image_burst_dropout_prob", 0.0), name="image_burst_dropout_prob"),
            max_burst_len=int(params.get("max_burst_len", 1) or 1),
            generator=generator,
            device=device,
        )
        missing_mask = dropout_mask | burst_mask
        if bool(missing_mask.any()):
            value = torch.where(_expand_temporal_mask(missing_mask, value), torch.zeros_like(value), value)
            valid_mask = valid_mask & ~missing_mask
            score = torch.where(missing_mask, torch.zeros_like(score), score)
            if bool(dropout_mask.any()):
                corruption_types.append("frame_dropout")
                counts["dropout_frames"] = int(dropout_mask.sum().item())
            if bool(burst_mask.any()):
                corruption_types.append("burst_missing")
                counts["burst_missing_frames"] = int(burst_mask.sum().item())

        batch[str(key)] = value.to(dtype=original_dtype)
        batch["image_valid_mask"] = valid_mask
        batch["image_dropout_mask"] = dropout_mask
        batch["image_burst_dropout_mask"] = burst_mask
        batch["image_observability_score"] = score
        metadata = {
            "operator": config.type,
            "condition": condition,
            "severity": float(profile.severity),
            "seed": seed,
            "input_space": "normalized_image_tensor",
            "frame_range": _frame_range(image),
            "corruption_types": corruption_types or ["clean"],
            "corruption_counts": counts,
            "physical_corruption_keeps_valid": True,
            "missing_invalidates_frame": True,
            "missing_expression": str(params.get("missing_expression", "zero_fill")),
            "parameters": _metadata_parameters(params),
            "score_factors": {
                "weather": weather,
                "lowlight": float(params.get("image_lowlight_prob", 0.0) or 0.0),
                "blur": float(params.get("image_blur_prob", 0.0) or 0.0),
                "occlusion": occlusion_ratio,
                "dropout": float(params.get("image_dropout_prob", 0.0) or 0.0),
                "burst": float(params.get("image_burst_dropout_prob", 0.0) or 0.0),
            },
            "masks": {
                "valid_mask": "image_valid_mask",
                "dropout_mask": "image_dropout_mask",
                "burst_dropout_mask": "image_burst_dropout_mask",
                "observability_score": "image_observability_score",
            },
        }
        batch["image_degradation_metadata"] = metadata
        batch["image_observability_replay"] = {
            "operator": config.type,
            "condition": condition,
            "seed": seed,
            "profile_digest": profile.digest,
            "operator_digest": config.digest,
            "sample_ids": list(context.sample_ids),
            "frame_range": _frame_range(image),
        }

        warnings: tuple[DifficultyWarning, ...] = ()
        affected = int(missing_mask.sum().item())
        if affected:
            warnings = (
                DifficultyWarning(
                    code="image_missing_zero_fill",
                    message="Missing image frames were represented as zero-filled image input with image_valid_mask=false.",
                    profile_id=profile.id,
                    operator=config.type,
                    condition=condition,
                    severity=float(profile.severity),
                    sample_count=batch_size,
                    affected_count=affected,
                    fallback=str(params.get("missing_expression", "zero_fill")),
                ),
            )
        return DifficultyOperatorOutcome(
            metadata={
                "condition": condition,
                "input_space": "normalized_image_tensor",
                "frame_range": _frame_range(image),
                "valid_mask": "image_valid_mask",
                "dropout_mask": "image_dropout_mask",
                "burst_dropout_mask": "image_burst_dropout_mask",
                "observability_score": "image_observability_score",
                "degradation_metadata": "image_degradation_metadata",
                "replay": "image_observability_replay",
                "corruption_counts": counts,
            },
            warnings=warnings,
        )


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
    if image.ndim >= 4:
        return [0, max(int(image.shape[1]) - 1, 0)]
    return [0, 0]


def _image_time_shape(image: torch.Tensor) -> tuple[int, int]:
    if image.ndim >= 4:
        return int(image.shape[0]), int(image.shape[1])
    return int(image.shape[0]), 1


def _expand_temporal_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if value.ndim >= 4:
        return mask.reshape(mask.shape[0], mask.shape[1], *([1] * (value.ndim - 2))).to(device=value.device)
    return mask.reshape(mask.shape[0], *([1] * (value.ndim - 1))).to(device=value.device)


def _random_frame_mask(
    batch_size: int,
    steps: int,
    *,
    probability: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    probability = max(0.0, min(float(probability), 1.0))
    if probability <= 0.0:
        return torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
    return (torch.rand((batch_size, steps), generator=generator, dtype=torch.float32) < probability).to(device=device)


def _burst_frame_mask(
    batch_size: int,
    steps: int,
    *,
    probability: float,
    max_burst_len: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, steps), dtype=torch.bool)
    probability = max(0.0, min(float(probability), 1.0))
    if probability <= 0.0 or steps <= 0:
        return mask.to(device=device)
    max_len = max(1, min(int(max_burst_len), steps))
    for batch_index in range(batch_size):
        if float(torch.rand((), generator=generator).item()) >= probability:
            continue
        length = int(torch.randint(1, max_len + 1, (1,), generator=generator).item())
        start = int(torch.randint(0, max(steps - length + 1, 1), (1,), generator=generator).item())
        mask[batch_index, start : start + length] = True
    return mask.to(device=device)


def _apply_temporal_occlusion(
    value: torch.Tensor,
    *,
    frame_mask: torch.Tensor,
    ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = value.clone()
    pixel_mask = torch.zeros_like(value, dtype=torch.bool)
    batch_size, steps = _image_time_shape(value)
    flat_out = out.reshape(batch_size * steps, *out.shape[2:])
    flat_pixels = pixel_mask.reshape(batch_size * steps, *pixel_mask.shape[2:])
    flat_frame_mask = frame_mask.reshape(batch_size * steps)
    for flat_index in torch.nonzero(flat_frame_mask.detach().cpu(), as_tuple=False).flatten().tolist():
        _occlude_single_frame(flat_out[int(flat_index)], flat_pixels[int(flat_index)], ratio=ratio, generator=generator)
    return out, pixel_mask


def _occlude_single_frame(
    frame: torch.Tensor,
    mask: torch.Tensor,
    *,
    ratio: float,
    generator: torch.Generator,
) -> None:
    if frame.ndim < 2:
        return
    height = int(frame.shape[-2])
    width = int(frame.shape[-1])
    if height <= 0 or width <= 0:
        return
    area_ratio = max(0.0, min(float(ratio), 1.0))
    side = math.sqrt(area_ratio)
    block_h = max(1, min(height, int(round(height * side))))
    block_w = max(1, min(width, int(round(width * side))))
    y = int(torch.randint(0, max(height - block_h + 1, 1), (1,), generator=generator).item())
    x = int(torch.randint(0, max(width - block_w + 1, 1), (1,), generator=generator).item())
    frame[..., y : y + block_h, x : x + block_w] = 0.0
    mask[..., y : y + block_h, x : x + block_w] = True


def _prob(value: Any, *, name: str) -> float:
    try:
        resolved = float(value or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {resolved}.")
    return resolved


def _metadata_parameters(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if key
        in {
            "condition",
            "image_dropout_prob",
            "image_burst_dropout_prob",
            "max_burst_len",
            "image_weather_severity",
            "image_blur_prob",
            "image_blur_radius",
            "image_occlusion_prob",
            "image_occlusion_ratio",
            "image_lowlight_prob",
            "image_lowlight_severity",
            "missing_expression",
        }
    }


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
