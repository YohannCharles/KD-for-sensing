import math
from typing import Any

import torch

from kd_sensing.data.difficulty.presets import PREDICTIVE_JEPA_CONDITION_IDS, SCENARIO_D_CONDITION_IDS
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
        condition = str(params.get("scenario_d_condition", params.get("condition", profile.condition)))

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
            "available_conditions": list(SCENARIO_D_CONDITION_IDS),
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


class PredictiveJepaRobustnessOperator(_BaseImageOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        params = dict(self.params)
        condition = str(params.get("predictive_condition", params.get("condition", profile.condition)))
        seed = int(context.derived_seed(profile, config))
        generator = self._generator(profile, config, context)
        history_window = int(params.get("history_window", 4))
        warnings: list[DifficultyWarning] = []
        image_metadata = self._apply_image(batch, params=params, condition=condition, seed=seed, generator=generator)
        gps_metadata, gps_warnings = self._apply_gps(batch, params=params, condition=condition, seed=seed, generator=generator)
        warnings.extend(gps_warnings)
        replay = {
            "operator": config.type,
            "condition": condition,
            "available_conditions": list(PREDICTIVE_JEPA_CONDITION_IDS),
            "seed": seed,
            "profile_digest": profile.digest,
            "operator_digest": config.digest,
            "sample_ids": list(context.sample_ids),
            "history_window": history_window,
            "image": image_metadata,
            "gps": gps_metadata,
            "counterfactual_input_intervention": bool(gps_metadata.get("counterfactual_status")),
        }
        batch["predictive_jepa_replay_metadata"] = replay
        return DifficultyOperatorOutcome(
            metadata={
                "condition": condition,
                "available_conditions": list(PREDICTIVE_JEPA_CONDITION_IDS),
                "history_window": history_window,
                "image": image_metadata,
                "gps": gps_metadata,
                "replay": "predictive_jepa_replay_metadata",
            },
            warnings=tuple(warnings),
        )

    def _apply_image(
        self,
        batch: dict[str, Any],
        *,
        params: dict[str, Any],
        condition: str,
        seed: int,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        key = self._image_key(batch)
        image = batch.get(key) if key else None
        if not torch.is_tensor(image):
            return {"state": "missing", "fallback": "skip"}
        original_dtype = image.dtype
        value = image.to(dtype=torch.float32).clone()
        batch_size, steps = _image_time_shape(value)
        current_step = _target_time_index(params, steps)
        device = image.device
        valid_mask = torch.ones((batch_size, steps), dtype=torch.bool, device=device)
        score = torch.ones((batch_size, steps), dtype=torch.float32, device=device)
        source_index = torch.arange(steps, dtype=torch.long, device=device).reshape(1, steps).expand(batch_size, steps).clone()
        history_available = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
        for step in range(steps):
            if step > 0:
                history_available[:, step] = True
        source_ranges = _history_source_ranges(steps, history_window=int(params.get("history_window", 4)))
        current_missing_mask = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
        semantic_frame_mask = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
        corruption_types: list[str] = []
        counts: dict[str, int] = {}
        visual_ambiguous_metadata: dict[str, Any] = {}

        if bool(params.get("visual_ambiguous_peer", False)):
            visual_ambiguous_metadata = _select_visual_ambiguous_peers(
                batch,
                image=image,
                params=params,
                current_step=current_step,
                generator=generator,
                seed=seed,
            )
            batch["visual_ambiguous_hard_negative_metadata"] = visual_ambiguous_metadata

        if bool(params.get("semantic_occlusion", False)):
            ratio = _prob(params.get("occlusion_ratio", 0.35), name="occlusion_ratio")
            pixel_mask = torch.zeros_like(value, dtype=torch.bool)
            for batch_index in range(batch_size):
                _occlude_single_frame(
                    value[batch_index, current_step],
                    pixel_mask[batch_index, current_step],
                    ratio=ratio,
                    generator=generator,
                )
            semantic_frame_mask[:, current_step] = True
            score[:, current_step] = (score[:, current_step] - 0.45 * ratio).clamp(0.0, 1.0)
            batch["image_semantic_occlusion_mask"] = pixel_mask.to(device=device)
            batch["image_occlusion_mask"] = pixel_mask.to(device=device)
            corruption_types.append("semantic_occlusion_proxy")
            counts["semantic_occlusion_frames"] = int(batch_size)

        if bool(params.get("novel_weather", False)):
            severity = _prob(params.get("weather_severity", 0.65), name="weather_severity")
            alpha = min(0.85, 0.65 * severity)
            current = value[:, current_step]
            current = current * (1.0 - alpha) + torch.full_like(current, 0.75) * alpha
            current = _add_rain_streaks(current, generator=generator, strength=severity)
            value[:, current_step] = current
            score[:, current_step] = (score[:, current_step] - 0.25 * severity).clamp(0.0, 1.0)
            corruption_types.append("novel_weather")
            counts["novel_weather_frames"] = int(batch_size)

        if bool(params.get("current_frame_missing", False)):
            current_missing_mask[:, current_step] = True
            value = torch.where(_expand_temporal_mask(current_missing_mask, value), torch.zeros_like(value), value)
            valid_mask[:, current_step] = False
            score[:, current_step] = 0.0
            corruption_types.append("current_frame_missing")
            counts["current_missing_frames"] = int(batch_size)

        batch[str(key)] = value.to(dtype=original_dtype)
        batch["image_valid_mask"] = valid_mask
        batch["image_observability_score"] = score
        batch["image_source_index"] = source_index
        batch["image_history_available_mask"] = history_available
        batch["image_current_missing_mask"] = current_missing_mask
        batch["image_semantic_frame_mask"] = semantic_frame_mask
        metadata = {
            "operator": "predictive_jepa_robustness",
            "condition": condition,
            "available_conditions": list(PREDICTIVE_JEPA_CONDITION_IDS),
            "seed": seed,
            "input_space": "normalized_image_tensor",
            "frame_range": _frame_range(image),
            "target_time_index": current_step,
            "history_window": int(params.get("history_window", 4)),
            "history_source_range": source_ranges,
            "history_available_mask": history_available.detach().cpu().tolist(),
            "valid_mask": "image_valid_mask",
            "observability_score": "image_observability_score",
            "source_index": "image_source_index",
            "current_frame_missing_mask": "image_current_missing_mask",
            "semantic_frame_mask": "image_semantic_frame_mask",
            "corruption_types": corruption_types or ["clean"],
            "corruption_counts": counts,
            "missing_expression": str(params.get("missing_expression", "zero_fill")),
            "semantic_occlusion_proxy": bool(params.get("semantic_occlusion", False)),
            "visual_ambiguous_peer": visual_ambiguous_metadata,
            "parameters": {
                key: params.get(key)
                for key in (
                    "current_frame_missing",
                    "semantic_occlusion",
                    "occlusion_ratio",
                    "visual_ambiguous_peer",
                    "visual_similarity_source",
                    "min_beam_offset",
                    "beam_offset_min",
                    "scene_constraint",
                    "novel_weather",
                    "weather_severity",
                    "history_window",
                    "target_time_index",
                    "missing_expression",
                )
                if key in params
            },
        }
        batch["image_degradation_metadata"] = metadata
        batch["image_predictive_replay"] = {
            "operator": "predictive_jepa_robustness",
            "condition": condition,
            "seed": seed,
            "profile_digest": batch.get("difficulty", {}).get("profile_digest"),
            "target_time_index": current_step,
            "history_source_range": source_ranges,
        }
        return metadata

    def _apply_gps(
        self,
        batch: dict[str, Any],
        *,
        params: dict[str, Any],
        condition: str,
        seed: int,
        generator: torch.Generator,
    ) -> tuple[dict[str, Any], list[DifficultyWarning]]:
        gps = batch.get("gps")
        warnings: list[DifficultyWarning] = []
        if not torch.is_tensor(gps):
            return {"state": "missing", "fallback": "skip"}, warnings
        value = gps.clone()
        batch_size = int(value.shape[0])
        steps = int(value.shape[1]) if value.ndim >= 3 else 1
        current_step = _target_time_index(params, steps)
        device = gps.device
        valid_mask = torch.ones((batch_size, steps), dtype=torch.bool, device=device)
        source_index = torch.arange(steps, dtype=torch.long, device=device).reshape(1, steps).expand(batch_size, steps).clone()
        source_sample_index = torch.arange(batch_size, dtype=torch.long, device=device).reshape(batch_size, 1).expand(batch_size, steps).clone()
        counterfactual_mask = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
        status = ""
        fallback = "none"
        fallback_reason = ""
        distance = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        beam_offset: list[float] = []
        peer_indices: list[int] = list(range(batch_size))
        peer_sample_ids: list[str | None] = [None for _ in range(batch_size)]
        selection_pool_sizes: list[int] = [0 for _ in range(batch_size)]
        fallback_reasons: list[str] = ["" for _ in range(batch_size)]

        if bool(params.get("plausible_wrong_gps", False)):
            min_offset = _beam_offset_threshold(params)
            beam_constrained = bool(params.get("beam_offset_constrained_wrong_gps", False)) or min_offset > 0
            scene_constraint = str(params.get("scene_constraint", "same_split_or_batch"))
            fallback = str(params.get("gps_counterfactual_fallback", "deterministic_jitter"))
            sample_ids = _sample_ids(batch, batch_size)
            peer_indices, selection_pool_sizes, fallback_reasons = _select_peer_indices(
                batch,
                batch_size=batch_size,
                min_beam_offset=min_offset,
                scene_constraint=scene_constraint,
                fallback=fallback,
                generator=generator,
            )
            peer_tensor = torch.tensor([max(index, 0) for index in peer_indices], dtype=torch.long, device=device)
            has_peer = torch.tensor([index >= 0 for index in peer_indices], dtype=torch.bool, device=device)
            if bool(has_peer.any()):
                if value.ndim >= 3:
                    original = value[:, current_step, :].clone()
                    replacement = value[peer_tensor, current_step, :].clone()
                    value[has_peer, current_step, :] = replacement[has_peer]
                else:
                    original = value.clone()
                    replacement = value[peer_tensor].clone()
                    value[has_peer] = replacement[has_peer]
                source_sample_index[has_peer, current_step] = peer_tensor[has_peer]
                counterfactual_mask[has_peer, current_step] = True
                distance = (replacement.to(torch.float32) - original.to(torch.float32)).reshape(batch_size, -1).pow(2).sum(dim=-1).sqrt()
                distance = torch.where(has_peer, distance, torch.zeros_like(distance))
            no_peer = ~has_peer
            if bool(no_peer.any()):
                if fallback == "fail":
                    raise ValueError(
                        "Beam-offset-constrained wrong GPS could not find enough peer samples; "
                        "adjust min_beam_offset, scene_constraint, or fallback."
                    )
                if fallback not in {"skip", "identity"}:
                    jitter_std = float(params.get("gps_jitter_std", 0.5))
                    if value.ndim >= 3:
                        noise = torch.randn(value[:, current_step, :].shape, generator=generator, dtype=torch.float32).to(device)
                        value[no_peer, current_step, :] = value[no_peer, current_step, :] + noise[no_peer].to(dtype=value.dtype) * jitter_std
                        distance = torch.where(no_peer, noise.reshape(batch_size, -1).pow(2).sum(dim=-1).sqrt() * jitter_std, distance)
                    else:
                        noise = torch.randn(value.shape, generator=generator, dtype=torch.float32).to(device)
                        value[no_peer] = value[no_peer] + noise[no_peer].to(dtype=value.dtype) * jitter_std
                        distance = torch.where(no_peer, noise.reshape(batch_size, -1).pow(2).sum(dim=-1).sqrt() * jitter_std, distance)
                    source_sample_index[no_peer, current_step] = -1
                    counterfactual_mask[no_peer, current_step] = True
            beam_offset = _beam_offsets_for_indices(batch.get("target_beam"), peer_indices)
            peer_sample_ids = [sample_ids[index] if index >= 0 and index < len(sample_ids) else None for index in peer_indices]
            fallback_count = sum(1 for reason in fallback_reasons if reason)
            if bool(has_peer.all()):
                status = "counterfactual_peer_replacement"
            elif not bool(has_peer.any()) and bool(counterfactual_mask[:, current_step].any()):
                status = "counterfactual_fallback_jitter"
            elif bool(counterfactual_mask[:, current_step].any()):
                status = "counterfactual_partial_peer_replacement"
            else:
                status = "counterfactual_skipped"
            fallback_reason = (
                "none"
                if fallback_count == 0
                else fallback_reasons[0]
                if fallback_count == batch_size and fallback_reasons[0]
                else "partial_peer_selection_fallback"
            )
            if fallback_count:
                warnings.append(
                    DifficultyWarning(
                        code=(
                            "predictive_jepa_beam_offset_wrong_gps_fallback"
                            if beam_constrained
                            else "predictive_jepa_plausible_wrong_gps_fallback"
                        ),
                        message="Beam-offset-constrained wrong GPS peer selection used fallback for some samples.",
                        operator="predictive_jepa_robustness",
                        condition=condition,
                        sample_count=batch_size,
                        affected_count=fallback_count,
                        fallback=fallback,
                    )
                )

        batch["gps"] = value
        batch["gps_valid_mask"] = valid_mask
        batch["gps_source_index"] = source_index
        batch["gps_source_sample_index"] = source_sample_index
        batch["gps_counterfactual_mask"] = counterfactual_mask
        metadata = {
            "operator": "predictive_jepa_robustness",
            "condition": condition,
            "seed": seed,
            "input_space": "gps_tensor",
            "target_time_index": current_step,
            "valid_mask": "gps_valid_mask",
            "source_index": "gps_source_index",
            "source_sample_index": "gps_source_sample_index",
            "counterfactual_mask": "gps_counterfactual_mask",
            "counterfactual_status": status,
            "scene_constraint": str(params.get("scene_constraint", "same_split_or_batch")),
            "min_beam_offset": _beam_offset_threshold(params),
            "peer_sample_id": peer_sample_ids,
            "peer_sample_index": peer_indices,
            "selection_pool_size": selection_pool_sizes,
            "fallback_count": sum(1 for reason in fallback_reasons if reason),
            "fallback_reasons": fallback_reasons,
            "distance_criteria": {
                "min_l2": float(distance.min().item()) if distance.numel() else 0.0,
                "mean_l2": float(distance.mean().item()) if distance.numel() else 0.0,
                "per_sample_l2": distance.detach().cpu().tolist(),
            },
            "beam_offset_criteria": {
                "offsets": beam_offset,
                "min_abs_offset": min(beam_offset) if beam_offset else None,
            },
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "counterfactual_input_intervention": bool(status),
        }
        batch["gps_counterfactual_metadata"] = metadata
        return metadata, warnings


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
    if image.ndim == 4:
        return [0, max(int(image.shape[1]) - 1, 0)]
    return [0, 0]


def _target_time_index(params: dict[str, Any], steps: int) -> int:
    if steps <= 0:
        return 0
    raw = int(params.get("target_time_index", -1) or -1)
    if raw < 0:
        raw = steps + raw
    return max(0, min(raw, steps - 1))


def _history_source_ranges(steps: int, *, history_window: int) -> list[list[int] | None]:
    ranges: list[list[int] | None] = []
    window = max(1, int(history_window))
    for step in range(steps):
        start = max(0, step - window)
        end = step
        ranges.append([start, end - 1] if end > start else None)
    return ranges


def _beam_offsets(target_beam: Any, source_sample_index: torch.Tensor) -> list[float]:
    if not torch.is_tensor(target_beam):
        return []
    target = target_beam.detach().cpu()
    if target.ndim == 0:
        return []
    flat = target.reshape(int(target.shape[0]), -1)[:, 0].to(dtype=torch.float32)
    if int(flat.shape[0]) != int(source_sample_index.numel()):
        return []
    source = source_sample_index.to(dtype=torch.long).clamp(0, max(int(flat.shape[0]) - 1, 0))
    return (flat[source] - flat).abs().tolist()


def _select_visual_ambiguous_peers(
    batch: dict[str, Any],
    *,
    image: torch.Tensor,
    params: dict[str, Any],
    current_step: int,
    generator: torch.Generator,
    seed: int,
) -> dict[str, Any]:
    batch_size = int(image.shape[0])
    min_offset = _beam_offset_threshold(params)
    scene_constraint = str(params.get("scene_constraint", "same_split_or_batch"))
    features = _visual_feature_matrix(batch, image=image, current_step=current_step, source=str(params.get("visual_similarity_source", "image_tensor_current")))
    sample_ids = _sample_ids(batch, batch_size)
    peer_indices: list[int] = []
    peer_ids: list[str | None] = []
    beam_offsets: list[float | None] = []
    similarity_scores: list[float | None] = []
    selection_pool_sizes: list[int] = []
    fallback_reasons: list[str] = []
    top_k = max(1, int(params.get("visual_ambiguous_top_k", 1) or 1))
    for sample_index in range(batch_size):
        candidates = _candidate_peer_indices(
            batch,
            sample_index=sample_index,
            batch_size=batch_size,
            min_beam_offset=min_offset,
            scene_constraint=scene_constraint,
        )
        selection_pool_sizes.append(len(candidates))
        if not candidates:
            peer_indices.append(-1)
            peer_ids.append(None)
            beam_offsets.append(None)
            similarity_scores.append(None)
            fallback_reasons.append("insufficient_visual_ambiguous_peer_pool")
            continue
        distances = []
        for candidate in candidates:
            distance = float((features[sample_index] - features[candidate]).pow(2).sum().sqrt().item())
            distances.append((distance, candidate))
        distances.sort(key=lambda item: (item[0], item[1]))
        top = distances[: min(top_k, len(distances))]
        chosen_pos = int(torch.randint(0, len(top), (1,), generator=generator).item()) if len(top) > 1 else 0
        distance, peer_index = top[chosen_pos]
        peer_indices.append(peer_index)
        peer_ids.append(sample_ids[peer_index] if peer_index < len(sample_ids) else None)
        offset = _beam_offsets_for_indices(batch.get("target_beam"), [peer_index], sample_index=sample_index)
        beam_offsets.append(offset[0] if offset else None)
        similarity_scores.append(1.0 / (1.0 + distance))
        fallback_reasons.append("")
    metadata = {
        "operator": "visual_ambiguous_hard_negative",
        "seed": seed,
        "visual_similarity_source": str(params.get("visual_similarity_source", "image_tensor_current")),
        "scene_constraint": scene_constraint,
        "min_beam_offset": min_offset,
        "peer_sample_index": peer_indices,
        "peer_sample_id": peer_ids,
        "similarity_score": similarity_scores,
        "beam_offset": beam_offsets,
        "selection_pool_size": selection_pool_sizes,
        "fallback_count": sum(1 for reason in fallback_reasons if reason),
        "fallback_reasons": fallback_reasons,
        "counterfactual_input_intervention": False,
    }
    batch["visual_ambiguous_peer_index"] = torch.tensor(peer_indices, dtype=torch.long, device=image.device)
    batch["visual_ambiguous_peer_sample_id"] = peer_ids
    return metadata


def _visual_feature_matrix(
    batch: dict[str, Any],
    *,
    image: torch.Tensor,
    current_step: int,
    source: str,
) -> torch.Tensor:
    for key in (source, "image_embedding", "image_embeddings", "image_features"):
        value = batch.get(key)
        if not torch.is_tensor(value):
            continue
        feature = value.detach().to(dtype=torch.float32)
        if feature.ndim >= 3:
            step = max(0, min(current_step, int(feature.shape[1]) - 1))
            feature = feature[:, step]
        return feature.reshape(int(feature.shape[0]), -1).cpu()
    if image.ndim >= 5:
        step = max(0, min(current_step, int(image.shape[1]) - 1))
        feature = image[:, step]
    elif image.ndim >= 4:
        feature = image
    else:
        feature = image.reshape(int(image.shape[0]), -1)
    return feature.detach().to(dtype=torch.float32).reshape(int(feature.shape[0]), -1).cpu()


def _select_peer_indices(
    batch: dict[str, Any],
    *,
    batch_size: int,
    min_beam_offset: float,
    scene_constraint: str,
    fallback: str,
    generator: torch.Generator,
) -> tuple[list[int], list[int], list[str]]:
    peers: list[int] = []
    pool_sizes: list[int] = []
    fallback_reasons: list[str] = []
    for sample_index in range(batch_size):
        candidates = _candidate_peer_indices(
            batch,
            sample_index=sample_index,
            batch_size=batch_size,
            min_beam_offset=min_beam_offset,
            scene_constraint=scene_constraint,
        )
        pool_sizes.append(len(candidates))
        reason = ""
        if not candidates and fallback == "relax_beam_offset":
            candidates = _candidate_peer_indices(
                batch,
                sample_index=sample_index,
                batch_size=batch_size,
                min_beam_offset=0.0,
                scene_constraint=scene_constraint,
            )
            reason = "relaxed_beam_offset" if candidates else ""
        if not candidates:
            peers.append(-1)
            fallback_reasons.append(reason or "insufficient_batch_peer_pool")
            continue
        choice = int(torch.randint(0, len(candidates), (1,), generator=generator).item()) if len(candidates) > 1 else 0
        peers.append(int(candidates[choice]))
        fallback_reasons.append(reason)
    return peers, pool_sizes, fallback_reasons


def _candidate_peer_indices(
    batch: dict[str, Any],
    *,
    sample_index: int,
    batch_size: int,
    min_beam_offset: float,
    scene_constraint: str,
) -> list[int]:
    target_values = _target_beam_values(batch.get("target_beam"), batch_size=batch_size)
    scenes = _metadata_values(batch, ("scene", "scene_id"), batch_size=batch_size)
    splits = _metadata_values(batch, ("split", "split_name", "dataset_split"), batch_size=batch_size)
    candidates: list[int] = []
    for candidate in range(batch_size):
        if candidate == sample_index:
            continue
        if not _constraint_allows(
            sample_index,
            candidate,
            scene_constraint=scene_constraint,
            scenes=scenes,
            splits=splits,
        ):
            continue
        if target_values is not None and abs(float(target_values[candidate]) - float(target_values[sample_index])) < min_beam_offset:
            continue
        candidates.append(candidate)
    return candidates


def _constraint_allows(
    sample_index: int,
    candidate: int,
    *,
    scene_constraint: str,
    scenes: list[str | None],
    splits: list[str | None],
) -> bool:
    mode = scene_constraint.strip().lower()
    if mode in {"batch", "same_batch", "none", "any"}:
        return True
    if mode == "same_scene":
        return bool(scenes[sample_index]) and scenes[sample_index] == scenes[candidate]
    if mode == "same_split":
        return bool(splits[sample_index]) and splits[sample_index] == splits[candidate]
    if mode in {"same_scene_or_split", "same_scene_then_split"}:
        if scenes[sample_index] is not None and scenes[candidate] is not None:
            return scenes[sample_index] == scenes[candidate]
        return bool(splits[sample_index]) and splits[sample_index] == splits[candidate]
    if mode == "same_split_or_batch":
        if splits[sample_index] is not None and splits[candidate] is not None:
            return splits[sample_index] == splits[candidate]
        return True
    return True


def _beam_offset_threshold(params: dict[str, Any]) -> float:
    raw = params.get("min_beam_offset", params.get("beam_offset_min", params.get("wrong_gps_min_beam_offset", 0)))
    try:
        return max(0.0, float(raw or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _target_beam_values(target_beam: Any, *, batch_size: int) -> list[float] | None:
    if not torch.is_tensor(target_beam):
        return None
    target = target_beam.detach().cpu()
    if target.ndim == 0 or int(target.shape[0]) < batch_size:
        return None
    return target.reshape(int(target.shape[0]), -1)[:batch_size, 0].to(dtype=torch.float32).tolist()


def _beam_offsets_for_indices(target_beam: Any, peer_indices: list[int], *, sample_index: int | None = None) -> list[float]:
    max_peer = max([index for index in peer_indices if index >= 0], default=0)
    batch_size = max(len(peer_indices), max_peer + 1)
    if sample_index is not None:
        batch_size = max(batch_size, sample_index + 1)
    target_values = _target_beam_values(target_beam, batch_size=max(batch_size, len(peer_indices)))
    if target_values is None:
        return []
    offsets: list[float] = []
    if sample_index is not None:
        for peer_index in peer_indices:
            offsets.append(abs(float(target_values[peer_index]) - float(target_values[sample_index])) if peer_index >= 0 else 0.0)
        return offsets
    for index, peer_index in enumerate(peer_indices):
        offsets.append(abs(float(target_values[peer_index]) - float(target_values[index])) if peer_index >= 0 else 0.0)
    return offsets


def _sample_ids(batch: dict[str, Any], batch_size: int) -> list[str]:
    values = _metadata_values(batch, ("sample_id", "sample_ids"), batch_size=batch_size)
    return [str(value) if value not in (None, "") else f"sample_{index}" for index, value in enumerate(values)]


def _metadata_values(batch: dict[str, Any], names: tuple[str, ...], *, batch_size: int) -> list[str | None]:
    metadata = batch.get("metadata")
    for name in names:
        value = None
        if isinstance(metadata, dict) and name in metadata:
            value = metadata[name]
        elif name in batch:
            value = batch[name]
        if value is None:
            continue
        if torch.is_tensor(value):
            flattened = value.detach().cpu().reshape(-1).tolist()
            return [str(flattened[index]) if index < len(flattened) else None for index in range(batch_size)]
        if isinstance(value, (list, tuple)):
            return [str(value[index]) if index < len(value) else None for index in range(batch_size)]
        return [str(value) for _ in range(batch_size)]
    return [None for _ in range(batch_size)]


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
            "scenario_d_condition",
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
